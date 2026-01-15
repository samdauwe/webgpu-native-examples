#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * Float16 Conversion Helpers
 * -------------------------------------------------------------------------- */

/* Convert float32 to float16 (IEEE 754 binary16) */
static uint16_t float32_to_float16(float value)
{
  union {
    float f;
    uint32_t i;
  } v;
  v.f        = value;
  uint32_t i = v.i;

  uint32_t sign     = (i >> 16) & 0x8000;
  int32_t exponent  = ((i >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = i & 0x007FFFFF;

  /* Handle special cases */
  if (exponent <= 0) {
    /* Underflow or zero */
    if (exponent < -10)
      return sign; /* Too small, flush to zero */
    mantissa = (mantissa | 0x00800000) >> (1 - exponent);
    return sign | (mantissa >> 13);
  }
  else if (exponent >= 0x1F) {
    /* Overflow or infinity */
    return sign | 0x7C00 | (mantissa ? 0x0200 : 0);
  }

  /* Normalized value */
  return sign | (exponent << 10) | (mantissa >> 13);
}

/* -------------------------------------------------------------------------- *
 * WebGPU Example - PBR with IBL (Physically Based Rendering with Image Based
 * Lighting)
 *
 * This example demonstrates Physically Based Rendering with Image Based
 * Lighting using WebGPU.
 *
 * Ref:
 * https://github.com/tchayen/pbr-webgpu
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader Code - PBR Shader Functions
 * -------------------------------------------------------------------------- */

/* PBR Distribution GGX function */
static const char* distribution_ggx_wgsl;

/* PBR Geometry Schlick GGX function */
static const char* geometry_schlick_ggx_wgsl;

/* PBR Geometry Smith function */
static const char* geometry_smith_wgsl;

/* PBR Fresnel Schlick function */
static const char* fresnel_schlick_wgsl;

/* PBR Fresnel Schlick Roughness function */
static const char* fresnel_schlick_roughness_wgsl;

/* Radical Inverse Van Der Corpus function */
static const char* radical_inverse_vdc_wgsl;

/* Hammersley function */
static const char* hammersley_wgsl;

/* Importance Sample GGX function */
static const char* importance_sample_ggx_wgsl;

/* Cubemap vertex shader */
static const char* cubemap_vertex_shader_wgsl;

/* Tone mapping functions */
static const char* tone_mapping_aces_wgsl;
/* static const char* tone_mapping_reinhard_wgsl; */   /* Currently unused */
/* static const char* tone_mapping_uncharted2_wgsl; */ /* Currently unused */
/* static const char* tone_mapping_lottes_wgsl; */     /* Currently unused */

/* Full PBR shader */
/* static const char* pbr_shader_wgsl; */ /* Currently unused - shader built
                                             dynamically */

/* -------------------------------------------------------------------------- *
 * String Utility - String Replacement Function
 * -------------------------------------------------------------------------- */

/**
 * @brief Replaces all occurrences of a substring within a string.
 * @param str The source string
 * @param from The substring to replace
 * @param to The replacement substring
 * @param result Buffer to store the result
 * @param result_size Size of the result buffer
 * @return Pointer to the result buffer, or NULL on error
 */
static char* str_replace_all(const char* str, const char* from, const char* to,
                             char* result, size_t result_size)
{
  if (!str || !from || !to || !result || result_size == 0) {
    return NULL;
  }

  const size_t from_len = strlen(from);
  const size_t to_len   = strlen(to);
  const char* pos       = str;
  char* dest            = result;
  size_t remaining = result_size - 1; /* Reserve space for null terminator */

  while ((pos = strstr(pos, from)) != NULL) {
    /* Copy the part before the match */
    const size_t prefix_len = pos - str;
    if (prefix_len > remaining) {
      return NULL; /* Buffer overflow */
    }
    memcpy(dest, str, prefix_len);
    dest += prefix_len;
    remaining -= prefix_len;

    /* Copy the replacement string */
    if (to_len > remaining) {
      return NULL; /* Buffer overflow */
    }
    memcpy(dest, to, to_len);
    dest += to_len;
    remaining -= to_len;

    /* Move past the matched substring */
    str = pos + from_len;
    pos = str;
  }

  /* Copy the remaining part of the string */
  const size_t tail_len = strlen(str);
  if (tail_len > remaining) {
    return NULL; /* Buffer overflow */
  }
  memcpy(dest, str, tail_len);
  dest[tail_len] = '\0';

  return result;
}

/* -------------------------------------------------------------------------- *
 * Camera Structure and Functions
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 target;          /* Camera target position */
  float distance;       /* Distance from target */
  float pitch;          /* Camera pitch angle (radians) */
  float yaw;            /* Camera yaw angle (radians) */
  int scroll_direction; /* Mouse wheel scroll direction */

  /* Mouse state */
  float last_x;
  float last_y;
  bool is_dragging;
} camera_t;

/**
 * @brief Initialize camera with default values
 */
static void camera_init(camera_t* this, float pitch, float yaw, float distance)
{
  memset(this, 0, sizeof(camera_t));

  glm_vec3_zero(this->target);
  this->pitch            = pitch;
  this->yaw              = yaw;
  this->distance         = distance > 0.0f ? distance : 10.0f;
  this->scroll_direction = 0;
  this->last_x           = 0.0f;
  this->last_y           = 0.0f;
  this->is_dragging      = false;
}

/**
 * @brief Handle mouse wheel event for zooming
 */
static void camera_handle_mouse_wheel(camera_t* this, float delta)
{
  this->scroll_direction = (delta > 0.0f) ? 1 : ((delta < 0.0f) ? -1 : 0);

  const float zoom_speed = 0.5f;
  this->distance -= this->scroll_direction * zoom_speed;

  const float min_distance = 1.0f;
  this->distance           = fmaxf(this->distance, min_distance);
}

/**
 * @brief Handle mouse button down event
 */
static void camera_handle_mouse_down(camera_t* this, float x, float y)
{
  this->is_dragging = true;
  this->last_x      = x;
  this->last_y      = y;
}

/**
 * @brief Handle mouse move event
 */
static void camera_handle_mouse_move(camera_t* this, float x, float y)
{
  if (!this->is_dragging) {
    return;
  }

  const float dx = x - this->last_x;
  const float dy = y - this->last_y;

  this->last_x = x;
  this->last_y = y;

  this->pitch -= dy * 0.003f;
  this->yaw -= dx * 0.003f;
}

/**
 * @brief Handle mouse button up event
 */
static void camera_handle_mouse_up(camera_t* this)
{
  this->is_dragging = false;
}

/**
 * @brief Get camera position in world space
 */
static void camera_get_position(camera_t* this, vec3 position)
{
  position[0] = cosf(this->pitch) * cosf(this->yaw);
  position[1] = sinf(this->pitch);
  position[2] = cosf(this->pitch) * sinf(this->yaw);

  glm_vec3_scale(position, this->distance, position);
  glm_vec3_add(position, this->target, position);
}

/**
 * @brief Get camera view matrix
 */
static void camera_get_view(camera_t* this, mat4 view)
{
  vec3 position, up = {0.0f, 1.0f, 0.0f};
  camera_get_position(this, position);
  glm_lookat(position, this->target, up, view);
}

/* -------------------------------------------------------------------------- *
 * Buffer Creation Utility
 * -------------------------------------------------------------------------- */

/**
 * @brief Creates and initializes a GPU buffer with the provided data
 */
static WGPUBuffer create_buffer_with_data(wgpu_context_t* wgpu_context,
                                          const void* data, uint64_t size,
                                          WGPUBufferUsage usage)
{
  /* Align to 4 bytes */
  const uint64_t aligned_size = (size + 3) & ~3;

  WGPUBufferDescriptor buffer_desc = {
    .usage            = usage | WGPUBufferUsage_CopyDst,
    .size             = aligned_size,
    .mappedAtCreation = false,
  };

  WGPUBuffer buffer
    = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);

  if (data && size > 0) {
    wgpuQueueWriteBuffer(wgpu_context->queue, buffer, 0, data, size);
  }

  return buffer;
}

/* -------------------------------------------------------------------------- *
 * Cubemap Shared Data
 * -------------------------------------------------------------------------- */

/* Cubemap view matrices for rendering to each face */
static mat4 cubemap_view_matrices[6] = {
  /* +X */ GLM_MAT4_IDENTITY_INIT,
  /* -X */ GLM_MAT4_IDENTITY_INIT,
  /* +Y */ GLM_MAT4_IDENTITY_INIT,
  /* -Y */ GLM_MAT4_IDENTITY_INIT,
  /* +Z */ GLM_MAT4_IDENTITY_INIT,
  /* -Z */ GLM_MAT4_IDENTITY_INIT,
};

/* Inverted cubemap view matrices */
static mat4 cubemap_view_matrices_inverted[6] = {
  /* +X */ GLM_MAT4_IDENTITY_INIT,
  /* -X */ GLM_MAT4_IDENTITY_INIT,
  /* +Y */ GLM_MAT4_IDENTITY_INIT,
  /* -Y */ GLM_MAT4_IDENTITY_INIT,
  /* +Z */ GLM_MAT4_IDENTITY_INIT,
  /* -Z */ GLM_MAT4_IDENTITY_INIT,
};

/* Cube vertex array for cubemap rendering */
static const float cube_vertex_array[] = {
  /* clang-format off */
   1.0f, -1.0f,  1.0f, 1.0f,
  -1.0f, -1.0f,  1.0f, 1.0f,
  -1.0f, -1.0f, -1.0f, 1.0f,
   1.0f, -1.0f, -1.0f, 1.0f,
   1.0f, -1.0f,  1.0f, 1.0f,
  -1.0f, -1.0f, -1.0f, 1.0f,

   1.0f,  1.0f,  1.0f, 1.0f,
   1.0f, -1.0f,  1.0f, 1.0f,
   1.0f, -1.0f, -1.0f, 1.0f,
   1.0f,  1.0f, -1.0f, 1.0f,
   1.0f,  1.0f,  1.0f, 1.0f,
   1.0f, -1.0f, -1.0f, 1.0f,

  -1.0f,  1.0f,  1.0f, 1.0f,
   1.0f,  1.0f,  1.0f, 1.0f,
   1.0f,  1.0f, -1.0f, 1.0f,
  -1.0f,  1.0f, -1.0f, 1.0f,
  -1.0f,  1.0f,  1.0f, 1.0f,
   1.0f,  1.0f, -1.0f, 1.0f,

  -1.0f, -1.0f,  1.0f, 1.0f,
  -1.0f,  1.0f,  1.0f, 1.0f,
  -1.0f,  1.0f, -1.0f, 1.0f,
  -1.0f, -1.0f, -1.0f, 1.0f,
  -1.0f, -1.0f,  1.0f, 1.0f,
  -1.0f,  1.0f, -1.0f, 1.0f,

   1.0f,  1.0f,  1.0f, 1.0f,
  -1.0f,  1.0f,  1.0f, 1.0f,
  -1.0f, -1.0f,  1.0f, 1.0f,
  -1.0f, -1.0f,  1.0f, 1.0f,
   1.0f, -1.0f,  1.0f, 1.0f,
   1.0f,  1.0f,  1.0f, 1.0f,

   1.0f, -1.0f, -1.0f, 1.0f,
  -1.0f, -1.0f, -1.0f, 1.0f,
  -1.0f,  1.0f, -1.0f, 1.0f,
  -1.0f,  1.0f, -1.0f, 1.0f,
   1.0f,  1.0f, -1.0f, 1.0f,
   1.0f, -1.0f, -1.0f, 1.0f,
  /* clang-format on */
};

/**
 * @brief Initialize cubemap view matrices
 */
static void init_cubemap_view_matrices(void)
{
  vec3 center = {0.0f, 0.0f, 0.0f};
  vec3 target, up;
  mat4 temp;

  /* +X face */
  target[0] = 1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, cubemap_view_matrices[0]);

  /* -X face */
  target[0] = -1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, cubemap_view_matrices[1]);

  /* +Y face - looking DOWN (swap with -Y to fix inversion) */
  target[0] = 0.0f;
  target[1] = -1.0f; /* Look DOWN for +Y */
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = -1.0f; /* -Z is up when looking down */
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, cubemap_view_matrices[2]);

  /* -Y face - looking UP (swap with +Y to fix inversion) */
  target[0] = 0.0f;
  target[1] = 1.0f; /* Look UP for -Y */
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = 1.0f; /* +Z is up when looking up */
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, cubemap_view_matrices[3]);

  /* +Z face */
  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = 1.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, cubemap_view_matrices[4]);

  /* -Z face */
  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = -1.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, cubemap_view_matrices[5]);

  /* Initialize inverted matrices */
  target[0] = 1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[0]);

  target[0] = -1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[1]);

  target[0] = 0.0f;
  target[1] = 1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = -1.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[2]);

  target[0] = 0.0f;
  target[1] = -1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = 1.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[3]);

  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = 1.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[4]);

  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = -1.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[5]);
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader Implementations
 * -------------------------------------------------------------------------- */

/* PBR Distribution GGX shader function */
static const char* distribution_ggx_wgsl
  = CODE(fn distributionGGX(n : vec3f, h : vec3f, roughness : f32)->f32 {
      let a      = roughness * roughness;
      let a2     = a * a;
      let nDotH  = max(dot(n, h), 0.0);
      let nDotH2 = nDotH * nDotH;
      var denom  = (nDotH2 * (a2 - 1.0) + 1.0);
      denom      = PI * denom * denom;
      return a2 / denom;
    });

/* PBR Geometry Schlick GGX shader function */
static const char* geometry_schlick_ggx_wgsl
  = CODE(fn geometrySchlickGGX(nDotV : f32, roughness : f32)->f32 {
      let r = (roughness + 1.0);
      let k = (r * r) / 8.0;
      return nDotV / (nDotV * (1.0 - k) + k);
    });

/* PBR Geometry Smith shader function */
static const char* geometry_smith_wgsl = CODE(
  fn geometrySmith(n : vec3f, v : vec3f, l : vec3f, roughness : f32)->f32 {
    let nDotV = max(dot(n, v), 0.0);
    let nDotL = max(dot(n, l), 0.0);
    let ggx2  = geometrySchlickGGX(nDotV, roughness);
    let ggx1  = geometrySchlickGGX(nDotL, roughness);
    return ggx1 * ggx2;
  });

/* PBR Fresnel Schlick shader function */
static const char* fresnel_schlick_wgsl
  = CODE(fn fresnelSchlick(cosTheta : f32, f0 : vec3f)->vec3f {
      return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    });

/* PBR Fresnel Schlick Roughness shader function */
static const char* fresnel_schlick_roughness_wgsl
  = CODE(fn fresnelSchlickRoughness(cosTheta : f32, f0 : vec3f, roughness : f32)
           ->vec3f {
             return f0
                    + (max(vec3f(1.0 - roughness), f0) - f0)
                        * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
           });

/* Radical Inverse Van Der Corpus shader function */
static const char* radical_inverse_vdc_wgsl = CODE(
  // http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
  // efficient VanDerCorpus calculation.
  fn radicalInverseVdC(bits : u32)->f32 {
    var result = bits;
    result     = (bits << 16u) | (bits >> 16u);
    result = ((result & 0x55555555u) << 1u) | ((result & 0xAAAAAAAAu) >> 1u);
    result = ((result & 0x33333333u) << 2u) | ((result & 0xCCCCCCCCu) >> 2u);
    result = ((result & 0x0F0F0F0Fu) << 4u) | ((result & 0xF0F0F0F0u) >> 4u);
    result = ((result & 0x00FF00FFu) << 8u) | ((result & 0xFF00FF00u) >> 8u);
    return f32(result) * 2.3283064365386963e-10;
  });

/* Hammersley shader function */
static const char* hammersley_wgsl
  = CODE(fn hammersley(i : u32, n : u32)->vec2f {
      return vec2f(f32(i) / f32(n), radicalInverseVdC(i));
    });

/* Importance Sample GGX shader function */
static const char* importance_sample_ggx_wgsl
  = CODE(fn importanceSampleGGX(xi : vec2f, n : vec3f, roughness : f32)->vec3f {
      let a = roughness * roughness;

      let phi      = 2.0 * PI * xi.x;
      let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
      let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

      // from spherical coordinates to cartesian coordinates - halfway vector
      let h = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

      // from tangent-space H vector to world-space sample vector
      let up : vec3f = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0),
                              abs(n.z) < 0.999);
      let tangent    = normalize(cross(up, n));
      let bitangent  = cross(n, tangent);

      let sampleVec = tangent * h.x + bitangent * h.y + n * h.z;
      return normalize(sampleVec);
    });

/* Tone mapping - Reinhard (currently unused, ACES is used by default) */
/*
static const char* tone_mapping_reinhard_wgsl
  = CODE(fn toneMapping(color : vec3f)->vec3f {
      return color / (color + vec3f(1.0));
    });
*/

/* Tone mapping - Uncharted 2 (currently unused) */
/*
static const char* tone_mapping_uncharted2_wgsl = CODE(
  fn uncharted2Helper(x : vec3f)
    ->vec3f {
      let a = 0.15;
      let b = 0.50;
      let c = 0.10;
      let d = 0.20;
      let e = 0.02;
      let f = 0.30;

      return (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f;
    }

  fn toneMapping(color : vec3f)
    ->vec3f {
      let w            = 11.2;
      let exposureBias = 2.0;
      let current      = uncharted2Helper(exposureBias * color);
      let whiteScale   = 1 / uncharted2Helper(vec3f(w));
      return current * whiteScale;
    });
*/

/* Tone mapping - ACES */
static const char* tone_mapping_aces_wgsl
  = CODE(fn toneMapping(color : vec3f)->vec3f {
      let a = 2.51;
      let b = 0.03;
      let c = 2.43;
      let d = 0.59;
      let e = 0.14;

      return (color * (a * color + b)) / (color * (c * color + d) + e);
    });

/* Tone mapping - Lottes (currently unused) */
/*
static const char* tone_mapping_lottes_wgsl
  = CODE(fn toneMapping(color : vec3f)->vec3f {
      let a      = vec3f(1.6);
      let d      = vec3f(0.977);
      let hdrMax = vec3f(8.0);
      let midIn  = vec3f(0.18);
      let midOut = vec3f(0.267);

      let b = (-pow(midIn, a) + pow(hdrMax, a) * midOut)
              / ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
      let c = (pow(hdrMax, a * d) * pow(midIn, a)
               - pow(hdrMax, a) * pow(midIn, a * d) * midOut)
              / ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

      return pow(color, a) / (pow(color, a * d) * b + c);
    });
*/

/* -------------------------------------------------------------------------- *
 * GLTF Type Definitions and Enums
 * -------------------------------------------------------------------------- */

/* Shader attribute locations */
typedef enum shader_location_t {
  SHADER_LOCATION_POSITION   = 0,
  SHADER_LOCATION_NORMAL     = 1,
  SHADER_LOCATION_TEXCOORD_0 = 2,
  SHADER_LOCATION_TANGENT    = 3,
} shader_location_t;

/* GLTF Component types */
typedef enum gltf_component_type_t {
  GLTF_COMPONENT_TYPE_BYTE           = 5120,
  GLTF_COMPONENT_TYPE_UNSIGNED_BYTE  = 5121,
  GLTF_COMPONENT_TYPE_SHORT          = 5122,
  GLTF_COMPONENT_TYPE_UNSIGNED_SHORT = 5123,
  GLTF_COMPONENT_TYPE_UNSIGNED_INT   = 5125,
  GLTF_COMPONENT_TYPE_FLOAT          = 5126,
} gltf_component_type_t;

/* GLTF Accessor types */
typedef enum gltf_accessor_type_t {
  GLTF_ACCESSOR_TYPE_SCALAR,
  GLTF_ACCESSOR_TYPE_VEC2,
  GLTF_ACCESSOR_TYPE_VEC3,
  GLTF_ACCESSOR_TYPE_VEC4,
} gltf_accessor_type_t;

/* Alpha mode */
typedef enum gltf_alpha_mode_t {
  GLTF_ALPHA_MODE_OPAQUE,
  GLTF_ALPHA_MODE_MASK,
  GLTF_ALPHA_MODE_BLEND,
} gltf_alpha_mode_t;

/* -------------------------------------------------------------------------- *
 * Renderer Utility Functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Get number of components for accessor type
 */
static uint32_t get_num_components_for_type(gltf_accessor_type_t type)
{
  switch (type) {
    case GLTF_ACCESSOR_TYPE_SCALAR:
      return 1;
    case GLTF_ACCESSOR_TYPE_VEC2:
      return 2;
    case GLTF_ACCESSOR_TYPE_VEC3:
      return 3;
    case GLTF_ACCESSOR_TYPE_VEC4:
      return 4;
    default:
      return 0;
  }
}

/**
 * @brief Get component type size in bytes
 */
static uint32_t get_component_type_size(gltf_component_type_t component_type)
{
  switch (component_type) {
    case GLTF_COMPONENT_TYPE_BYTE:
    case GLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      return 1;
    case GLTF_COMPONENT_TYPE_SHORT:
    case GLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      return 2;
    case GLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case GLTF_COMPONENT_TYPE_FLOAT:
      return 4;
    default:
      return 0;
  }
}

/**
 * @brief Get GPU vertex format for accessor
 */
static WGPUVertexFormat
get_gpu_vertex_format(gltf_component_type_t component_type,
                      gltf_accessor_type_t type, bool normalized)
{
  const uint32_t count = get_num_components_for_type(type);

  switch (component_type) {
    case GLTF_COMPONENT_TYPE_BYTE:
      if (normalized) {
        return (count == 2) ? WGPUVertexFormat_Snorm8x2 :
               (count == 4) ? WGPUVertexFormat_Snorm8x4 :
                              WGPUVertexFormat_Sint8x2; /* Fallback */
      }
      return (count == 2) ? WGPUVertexFormat_Sint8x2 :
             (count == 4) ? WGPUVertexFormat_Sint8x4 :
                            WGPUVertexFormat_Sint8x2; /* Fallback */

    case GLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      if (normalized) {
        return (count == 2) ? WGPUVertexFormat_Unorm8x2 :
               (count == 4) ? WGPUVertexFormat_Unorm8x4 :
                              WGPUVertexFormat_Uint8x2; /* Fallback */
      }
      return (count == 2) ? WGPUVertexFormat_Uint8x2 :
             (count == 4) ? WGPUVertexFormat_Uint8x4 :
                            WGPUVertexFormat_Uint8x2; /* Fallback */

    case GLTF_COMPONENT_TYPE_SHORT:
      if (normalized) {
        return (count == 2) ? WGPUVertexFormat_Snorm16x2 :
               (count == 4) ? WGPUVertexFormat_Snorm16x4 :
                              WGPUVertexFormat_Sint16x2; /* Fallback */
      }
      return (count == 2) ? WGPUVertexFormat_Sint16x2 :
             (count == 4) ? WGPUVertexFormat_Sint16x4 :
                            WGPUVertexFormat_Sint16x2; /* Fallback */

    case GLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      if (normalized) {
        return (count == 2) ? WGPUVertexFormat_Unorm16x2 :
               (count == 4) ? WGPUVertexFormat_Unorm16x4 :
                              WGPUVertexFormat_Uint8x2;
      }
      return (count == 2) ? WGPUVertexFormat_Uint16x2 :
             (count == 4) ? WGPUVertexFormat_Uint16x4 :
                            WGPUVertexFormat_Uint8x2;

    case GLTF_COMPONENT_TYPE_UNSIGNED_INT:
      return (count == 1) ? WGPUVertexFormat_Uint32 :
             (count == 2) ? WGPUVertexFormat_Uint32x2 :
             (count == 3) ? WGPUVertexFormat_Uint32x3 :
             (count == 4) ? WGPUVertexFormat_Uint32x4 :
                            WGPUVertexFormat_Uint8x2;

    case GLTF_COMPONENT_TYPE_FLOAT:
      return (count == 1) ? WGPUVertexFormat_Float32 :
             (count == 2) ? WGPUVertexFormat_Float32x2 :
             (count == 3) ? WGPUVertexFormat_Float32x3 :
             (count == 4) ? WGPUVertexFormat_Float32x4 :
                            WGPUVertexFormat_Uint8x2;

    default:
      return WGPUVertexFormat_Uint8x2;
  }
}

/**
 * @brief Get GPU index format for component type
 */
static WGPUIndexFormat
get_gpu_index_format(gltf_component_type_t component_type)
{
  switch (component_type) {
    case GLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      return WGPUIndexFormat_Uint16;
    case GLTF_COMPONENT_TYPE_UNSIGNED_INT:
      return WGPUIndexFormat_Uint32;
    default:
      return WGPUIndexFormat_Undefined;
  }
}

/**
 * @brief Get GPU address mode for wrapping mode
 */
static WGPUAddressMode get_gpu_address_mode(uint32_t wrap_mode)
{
  switch (wrap_mode) {
    case 33071: /* CLAMP_TO_EDGE */
      return WGPUAddressMode_ClampToEdge;
    case 33648: /* MIRRORED_REPEAT */
      return WGPUAddressMode_MirrorRepeat;
    case 10497: /* REPEAT */
      return WGPUAddressMode_Repeat;
    default:
      return WGPUAddressMode_Repeat;
  }
}

/**
 * @brief Create a sampler from GLTF sampler descriptor
 */
static WGPUSampler create_sampler_from_gltf(wgpu_context_t* wgpu_context,
                                            uint32_t mag_filter,
                                            uint32_t min_filter,
                                            uint32_t wrap_s, uint32_t wrap_t)
{
  WGPUSamplerDescriptor sampler_desc = {
    .addressModeU = get_gpu_address_mode(wrap_s),
    .addressModeV = get_gpu_address_mode(wrap_t),
    .addressModeW = WGPUAddressMode_Repeat,
    .magFilter
    = (mag_filter == 9729) ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest,
    .minFilter    = WGPUFilterMode_Nearest,
    .mipmapFilter = WGPUMipmapFilterMode_Nearest,
  };

  /* Handle minification filter */
  switch (min_filter) {
    case 9728: /* NEAREST */
      sampler_desc.minFilter = WGPUFilterMode_Nearest;
      break;
    case 9729: /* LINEAR */
    case 9985: /* LINEAR_MIPMAP_NEAREST */
      sampler_desc.minFilter = WGPUFilterMode_Linear;
      break;
    case 9986: /* NEAREST_MIPMAP_LINEAR */
      sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Linear;
      break;
    case 9987: /* LINEAR_MIPMAP_LINEAR */
    default:
      sampler_desc.minFilter    = WGPUFilterMode_Linear;
      sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Linear;
      break;
  }

  return wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
}

/**
 * @brief Create default sampler
 */
static WGPUSampler create_default_sampler(wgpu_context_t* wgpu_context)
{
  WGPUSamplerDescriptor sampler_desc = {
    .label         = STRVIEW("Default sampler"),
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
    .addressModeW  = WGPUAddressMode_Repeat,
    .magFilter     = WGPUFilterMode_Linear,
    .minFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUMipmapFilterMode_Linear,
    .maxAnisotropy = 1,
  };

  return wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
}

/**
 * @brief Create a solid color texture
 */
static WGPUTexture create_solid_color_texture(wgpu_context_t* wgpu_context,
                                              float r, float g, float b,
                                              float a)
{
  const uint8_t data[4] = {(uint8_t)(r * 255.0f), (uint8_t)(g * 255.0f),
                           (uint8_t)(b * 255.0f), (uint8_t)(a * 255.0f)};

  WGPUTextureDescriptor texture_desc = {
    .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width  = 1,
      .height = 1,
      .depthOrArrayLayers = 1,
    },
    .format = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = texture,
                          .mipLevel = 0,
                          .origin   = (WGPUOrigin3D){0, 0, 0},
                          .aspect   = WGPUTextureAspect_All,
                        },
                        data, 4,
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = 4,
                          .rowsPerImage = 1,
                        },
                        &texture_desc.size);

  return texture;
}

/* -------------------------------------------------------------------------- *
 * Irradiance Map Generation
 * -------------------------------------------------------------------------- */

/**
 * @brief Generate irradiance map from environment cubemap
 */
static WGPUTexture generate_irradiance_map(wgpu_context_t* wgpu_context,
                                           WGPUTexture cubemap_texture,
                                           uint32_t size)
{
  /* Create irradiance texture */
  WGPUTextureDescriptor irradiance_desc = {
    .label = STRVIEW("irradiance map"),
    .usage = WGPUTextureUsage_TextureBinding
           | WGPUTextureUsage_CopyDst
           | WGPUTextureUsage_RenderAttachment,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width = size,
      .height = size,
      .depthOrArrayLayers = 6,
    },
    .format = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture irradiance_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &irradiance_desc);

  /* Create depth texture */
  WGPUTextureDescriptor depth_desc = {
    .label = STRVIEW("irradiance map depth"),
    .usage = WGPUTextureUsage_RenderAttachment,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width = size,
      .height = size,
      .depthOrArrayLayers = 1,
    },
    .format = wgpu_context->depth_stencil_format,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_desc);
  WGPUTextureView depth_view = wgpuTextureCreateView(depth_texture, NULL);

  /* Fragment shader for irradiance convolution */
  const char* fragment_shader_wgsl = CODE(
    @group(0) @binding(1) var environmentMap : texture_cube<f32>;
    @group(0) @binding(2) var ourSampler : sampler;

    const PI = 3.14159265359;

    @fragment fn main(@location(0) worldPosition : vec4f)->@location(0) vec4f {
      let normal     = normalize(worldPosition.xyz);
      var irradiance = vec3f(0.0, 0.0, 0.0);

      var up    = vec3f(0.0, 1.0, 0.0);
      let right = normalize(cross(up, normal));
      up        = normalize(cross(normal, right));

      var sampleDelta = 0.025;
      var nrSamples   = 0.0;
      for (var phi : f32 = 0.0; phi < 2.0 * PI; phi = phi + sampleDelta) {
        for (var theta : f32 = 0.0; theta < 0.5 * PI;
             theta           = theta + sampleDelta) {
          let tangentSample : vec3f = vec3f(sin(theta) * cos(phi),
                                            sin(theta) * sin(phi), cos(theta));
          let sampleVec = tangentSample.x * right + tangentSample.y * up
                          + tangentSample.z * normal;

          irradiance
            = irradiance
              + textureSample(environmentMap, ourSampler, sampleVec).rgb
                  * cos(theta) * sin(theta);
          nrSamples = nrSamples + 1.0;
        }
      }
      irradiance = PI * irradiance * (1.0 / nrSamples);

      return vec4f(irradiance, 1.0);
    });

  /* Create shaders */
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, cubemap_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  /* Create vertex buffer */
  WGPUBuffer vertex_buffer = create_buffer_with_data(
    wgpu_context, cube_vertex_array, sizeof(cube_vertex_array),
    WGPUBufferUsage_Vertex);

  /* Create sampler */
  WGPUSampler sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Irradiance - Sampler"),
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });

  /* Create uniform buffer */
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("irradiance uniform"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4),
    });

  /* Create pipeline */
  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("irradiance map pipeline"),
      .vertex = (WGPUVertexState){
        .module = vert_shader_module,
        .entryPoint = STRVIEW("main"),
        .bufferCount = 1,
        .buffers = &(WGPUVertexBufferLayout){
          .arrayStride = 4 * sizeof(float),
          .stepMode = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes = &(WGPUVertexAttribute){
            .shaderLocation = 0,
            .offset = 0,
            .format = WGPUVertexFormat_Float32x4,
          },
        },
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode = WGPUCullMode_None,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = frag_shader_module,
        .entryPoint = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = WGPUTextureFormat_RGBA8Unorm,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    }
  );

  /* Create cubemap view */
  WGPUTextureView cubemap_view = wgpuTextureCreateView(
    cubemap_texture, &(WGPUTextureViewDescriptor){
                       .dimension       = WGPUTextureViewDimension_Cube,
                       .mipLevelCount   = 1,
                       .arrayLayerCount = 6,
                     });

  /* Create bind group */
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label = STRVIEW("irradiance bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
      .entryCount = 3,
      .entries = (WGPUBindGroupEntry[]){
        {.binding = 0, .buffer = uniform_buffer, .offset = 0, .size = sizeof(mat4)},
        {.binding = 1, .textureView = cubemap_view},
        {.binding = 2, .sampler = sampler},
      },
    }
  );

  /* Setup projection matrix */
  mat4 projection;
  glm_perspective(GLM_PI_2f, 1.0f, 0.1f, 10.0f, projection);

  /* Render each face */
  for (uint32_t face = 0; face < 6; face++) {
    mat4 mvp;
    glm_mat4_mul(projection, cubemap_view_matrices_inverted[face], mvp);
    wgpuQueueWriteBuffer(wgpu_context->queue, uniform_buffer, 0, mvp,
                         sizeof(mat4));

    WGPUTextureView face_view = wgpuTextureCreateView(
      irradiance_texture, &(WGPUTextureViewDescriptor){
                            .label           = STRVIEW("irradiance face view"),
                            .format          = WGPUTextureFormat_RGBA8Unorm,
                            .dimension       = WGPUTextureViewDimension_2D,
                            .baseMipLevel    = 0,
                            .mipLevelCount   = 1,
                            .baseArrayLayer  = face,
                            .arrayLayerCount = 1,
                          });

    WGPUCommandEncoder command_encoder
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPURenderPassEncoder pass_encoder = wgpuCommandEncoderBeginRenderPass(
      command_encoder,
      &(WGPURenderPassDescriptor){
        .label = STRVIEW("irradiance render pass"),
        .colorAttachmentCount = 1,
        .colorAttachments = &(WGPURenderPassColorAttachment){
          .view = face_view,
          .loadOp = WGPULoadOp_Clear,
          .storeOp = WGPUStoreOp_Store,
          .clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0},
          .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        },
        .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
          .view = depth_view,
          .depthLoadOp = WGPULoadOp_Clear,
          .depthStoreOp = WGPUStoreOp_Store,
          .depthClearValue = 1.0f,
          .stencilLoadOp = WGPULoadOp_Clear,
          .stencilStoreOp = WGPUStoreOp_Store,
          .stencilClearValue = 0,
        },
      }
    );

    wgpuRenderPassEncoderSetPipeline(pass_encoder, pipeline);
    wgpuRenderPassEncoderSetViewport(pass_encoder, 0, 0, size, size, 0, 1);
    wgpuRenderPassEncoderSetVertexBuffer(pass_encoder, 0, vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(pass_encoder, 0, bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(pass_encoder, 36, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass_encoder);

    WGPUCommandBuffer command = wgpuCommandEncoderFinish(command_encoder, NULL);
    wgpuQueueSubmit(wgpu_context->queue, 1, &command);

    WGPU_RELEASE_RESOURCE(CommandBuffer, command)
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass_encoder)
    WGPU_RELEASE_RESOURCE(CommandEncoder, command_encoder)
    WGPU_RELEASE_RESOURCE(TextureView, face_view)
  }

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(TextureView, cubemap_view)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer)
  WGPU_RELEASE_RESOURCE(Sampler, sampler)
  WGPU_RELEASE_RESOURCE(Buffer, vertex_buffer)
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module)
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module)
  WGPU_RELEASE_RESOURCE(TextureView, depth_view)
  WGPU_RELEASE_RESOURCE(Texture, depth_texture)

  return irradiance_texture;
}

/* -------------------------------------------------------------------------- *
 * Prefilter Map Generation
 * -------------------------------------------------------------------------- */

/**
 * @brief Generate prefiltered environment map for specular IBL
 */
static WGPUTexture generate_prefilter_map(wgpu_context_t* wgpu_context,
                                          WGPUTexture cubemap_texture,
                                          uint32_t size, uint32_t levels)
{
  /* Create prefilter texture with mipmaps */
  WGPUTextureDescriptor prefilter_desc = {
    .label = STRVIEW("prefilter map"),
    .usage = WGPUTextureUsage_TextureBinding
           | WGPUTextureUsage_CopyDst
           | WGPUTextureUsage_RenderAttachment,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width = size,
      .height = size,
      .depthOrArrayLayers = 6,
    },
    .format = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = levels,
    .sampleCount = 1,
  };

  WGPUTexture prefilter_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &prefilter_desc);

  /* Create depth texture with mipmaps */
  WGPUTextureDescriptor depth_desc = {
    .label = STRVIEW("prefilter map depth"),
    .usage = WGPUTextureUsage_RenderAttachment,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width = size,
      .height = size,
      .depthOrArrayLayers = 1,
    },
    .format = wgpu_context->depth_stencil_format,
    .mipLevelCount = levels,
    .sampleCount = 1,
  };

  WGPUTexture depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_desc);

  /* Vertex shader with roughness uniform */
  const char* vertex_shader_wgsl = CODE(
    struct VSOut {
      @builtin(position) position: vec4f,
      @location(0) worldPosition: vec4f,
    };

    struct Uniforms {
      modelViewProjectionMatrix: mat4x4f,
      roughness: f32,
    };

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    @vertex
    fn main(@location(0) position: vec4f) -> VSOut {
      var output: VSOut;
      output.position = uniforms.modelViewProjectionMatrix * position;
      output.worldPosition = position;
      return output;
    }
  );

  /* Fragment shader for prefiltered environment map */
  char fragment_shader_wgsl[16384];
  snprintf(fragment_shader_wgsl, sizeof(fragment_shader_wgsl), CODE(
    struct Uniforms {
      modelViewProjectionMatrix: mat4x4f,
      roughness: f32,
    };

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var environmentMap: texture_cube<f32>;
    @group(0) @binding(2) var environmentSampler: sampler;

    const PI = 3.14159265359;

    %s
    %s
    %s
    %s

    @fragment
    fn main(@location(0) worldPosition: vec4f) -> @location(0) vec4f {
      var n = normalize(worldPosition.xyz);

      // Make the simplifying assumption that V equals R equals the normal
      let r = n;
      let v = r;

      let SAMPLE_COUNT: u32 = 4096u;
      var prefilteredColor = vec3f(0.0, 0.0, 0.0);
      var totalWeight = 0.0;

      for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        // Generates a sample vector that's biased towards the preferred alignment
        // direction (importance sampling).
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importanceSampleGGX(xi, n, uniforms.roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);

        let nDotL = max(dot(n, l), 0.0);

        if(nDotL > 0.0) {
          // sample from the environment's mip level based on roughness/pdf
          let d = distributionGGX(n, h, uniforms.roughness);
          let nDotH = max(dot(n, h), 0.0);
          let hDotV = max(dot(h, v), 0.0);
          let pdf = d * nDotH / (4.0 * hDotV) + 0.0001;

          let resolution = %u.0; // resolution of source cubemap (per face)
          let saTexel = 4.0 * PI / (6.0 * resolution * resolution);
          let saSample = 1.0 / (f32(SAMPLE_COUNT) * pdf + 0.0001);

          let mipLevel = select(0.5 * log2(saSample / saTexel), 0.0, uniforms.roughness == 0.0);

          prefilteredColor += textureSampleLevel(environmentMap, environmentSampler, l, mipLevel).rgb * nDotL;
          totalWeight += nDotL;
        }
      }

      prefilteredColor = prefilteredColor / totalWeight;
      return vec4f(prefilteredColor, 1.0);
    }
  ), distribution_ggx_wgsl, radical_inverse_vdc_wgsl, hammersley_wgsl, importance_sample_ggx_wgsl, size);

  /* Create shaders */
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  /* Create vertex buffer */
  WGPUBuffer vertex_buffer = create_buffer_with_data(
    wgpu_context, cube_vertex_array, sizeof(cube_vertex_array),
    WGPUBufferUsage_Vertex);

  /* Create sampler */
  WGPUSampler sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("prefilter sampler"),
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });

  /* Create uniform buffer (mat4x4 + roughness + padding) */
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("prefilter uniform"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(float) * (16 + 4), /* mat4 + roughness + padding */
    });

  /* Create pipeline */
  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("prefilter map pipeline"),
      .vertex = (WGPUVertexState){
        .module = vert_shader_module,
        .entryPoint = STRVIEW("main"),
        .bufferCount = 1,
        .buffers = &(WGPUVertexBufferLayout){
          .arrayStride = 4 * sizeof(float),
          .stepMode = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes = &(WGPUVertexAttribute){
            .shaderLocation = 0,
            .offset = 0,
            .format = WGPUVertexFormat_Float32x4,
          },
        },
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode = WGPUCullMode_None,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = frag_shader_module,
        .entryPoint = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = WGPUTextureFormat_RGBA8Unorm,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    }
  );

  /* Create cubemap view (source cubemap only has 1 mip level) */
  WGPUTextureView cubemap_view = wgpuTextureCreateView(
    cubemap_texture, &(WGPUTextureViewDescriptor){
                       .dimension       = WGPUTextureViewDimension_Cube,
                       .mipLevelCount   = 1,
                       .arrayLayerCount = 6,
                     });

  /* Setup projection matrix */
  mat4 projection;
  glm_perspective(GLM_PI_2f, 1.0f, 0.1f, 10.0f, projection);

  /* Render each mip level */
  for (uint32_t mip = 0; mip < levels; mip++) {
    uint32_t mip_width  = size >> mip;
    uint32_t mip_height = size >> mip;
    float roughness     = (float)mip / (float)(levels - 1);

    /* Create bind group for this mip level */
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label = STRVIEW("prefilter bind group"),
        .layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
        .entryCount = 3,
        .entries = (WGPUBindGroupEntry[]){
          {.binding = 0, .buffer = uniform_buffer, .offset = 0, .size = sizeof(float) * (16 + 4)},
          {.binding = 1, .textureView = cubemap_view},
          {.binding = 2, .sampler = sampler},
        },
      }
    );

    /* Create depth view for this mip level */
    WGPUTextureView depth_view
      = wgpuTextureCreateView(depth_texture, &(WGPUTextureViewDescriptor){
                                               .baseMipLevel    = mip,
                                               .mipLevelCount   = 1,
                                               .arrayLayerCount = 1,
                                             });

    /* Render each cubemap face */
    for (uint32_t face = 0; face < 6; face++) {
      /* Compute MVP matrix */
      mat4 mvp;
      glm_mat4_mul(projection, cubemap_view_matrices_inverted[face], mvp);

      /* Write uniforms (MVP + roughness + padding) */
      float uniform_data[20]; /* 16 for mat4 + 4 for roughness and padding */
      memcpy(uniform_data, mvp, sizeof(mat4));
      uniform_data[16] = roughness;
      uniform_data[17] = 0.0f;
      uniform_data[18] = 0.0f;
      uniform_data[19] = 0.0f;
      wgpuQueueWriteBuffer(wgpu_context->queue, uniform_buffer, 0, uniform_data,
                           sizeof(uniform_data));

      /* Create face view */
      WGPUTextureView face_view = wgpuTextureCreateView(
        prefilter_texture, &(WGPUTextureViewDescriptor){
                             .label           = STRVIEW("prefilter face view"),
                             .format          = WGPUTextureFormat_RGBA8Unorm,
                             .dimension       = WGPUTextureViewDimension_2D,
                             .baseMipLevel    = mip,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = face,
                             .arrayLayerCount = 1,
                           });

      /* Render pass */
      WGPUCommandEncoder encoder
        = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
      WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
        encoder,
        &(WGPURenderPassDescriptor){
          .label = STRVIEW("prefilter render pass"),
          .colorAttachmentCount = 1,
          .colorAttachments = &(WGPURenderPassColorAttachment){
            .view = face_view,
            .loadOp = WGPULoadOp_Clear,
            .storeOp = WGPUStoreOp_Store,
            .clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0},
            .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
          },
          .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
            .view = depth_view,
            .depthLoadOp = WGPULoadOp_Clear,
            .depthStoreOp = WGPUStoreOp_Store,
            .depthClearValue = 1.0f,
            .stencilLoadOp = WGPULoadOp_Clear,
            .stencilStoreOp = WGPUStoreOp_Store,
            .stencilClearValue = 0,
          },
        }
      );

      wgpuRenderPassEncoderSetPipeline(pass, pipeline);
      wgpuRenderPassEncoderSetViewport(pass, 0, 0, mip_width, mip_height, 0, 1);
      wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex_buffer, 0,
                                           WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
      wgpuRenderPassEncoderDraw(pass, 36, 1, 0, 0);
      wgpuRenderPassEncoderEnd(pass);

      WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
      wgpuQueueSubmit(wgpu_context->queue, 1, &command);

      WGPU_RELEASE_RESOURCE(CommandBuffer, command)
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
      WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)
      WGPU_RELEASE_RESOURCE(TextureView, face_view)
    }

    WGPU_RELEASE_RESOURCE(TextureView, depth_view)
    WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  }

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(TextureView, cubemap_view)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer)
  WGPU_RELEASE_RESOURCE(Sampler, sampler)
  WGPU_RELEASE_RESOURCE(Buffer, vertex_buffer)
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module)
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module)
  WGPU_RELEASE_RESOURCE(Texture, depth_texture)

  return prefilter_texture;
}

/* -------------------------------------------------------------------------- *
 * Hash Function
 * -------------------------------------------------------------------------- */

/**
 * @brief Simple hash function for strings
 * @param value String to hash
 * @return 32-bit hash value
 */
static uint32_t hash_string(const char* value)
{
  if (!value) {
    return 0;
  }

  uint32_t hash = 0;
  for (size_t i = 0; value[i] != '\0'; i++) {
    uint8_t ch = (uint8_t)value[i];
    hash       = ((hash << 5) - hash) + ch;
    /* hash is already 32-bit (uint32_t) */
  }

  return hash;
}

/* -------------------------------------------------------------------------- *
 * HDR Image Loading using stb_image
 * -------------------------------------------------------------------------- */

/* stb_image already included at top of file - removed duplicate include */

/* HDR image data structure */
typedef struct {
  uint32_t width;
  uint32_t height;
  float exposure;
  float gamma;
  float* data; /* RGBA32F data */
} hdr_image_t;

/**
 * @brief Load HDR image from memory buffer
 */
static bool load_hdr_image(const void* buffer, size_t buffer_size,
                           hdr_image_t* out_image)
{
  if (!buffer || !out_image) {
    return false;
  }

  memset(out_image, 0, sizeof(hdr_image_t));

  int width, height, channels;
  float* data
    = stbi_loadf_from_memory((const unsigned char*)buffer, (int)buffer_size,
                             &width, &height, &channels, 4);

  if (!data) {
    return false;
  }

  out_image->width    = (uint32_t)width;
  out_image->height   = (uint32_t)height;
  out_image->exposure = 1.0f;
  out_image->gamma    = 1.0f;
  out_image->data     = data;

  return true;
}

/**
 * @brief Free HDR image data
 */
static void free_hdr_image(hdr_image_t* image)
{
  if (image && image->data) {
    stbi_image_free(image->data);
    image->data = NULL;
  }
}

/* -------------------------------------------------------------------------- *
 * Cubemap Utilities - Vertex Shader and Conversion
 * -------------------------------------------------------------------------- */

/* Cubemap vertex shader */
static const char* cubemap_vertex_shader_wgsl = CODE(
  struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) worldPosition: vec4f,
  };

  struct Uniforms {
    modelViewProjectionMatrix: mat4x4f,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;

  @vertex
  fn main(@location(0) position: vec4f) -> VSOut {
    var output: VSOut;
    output.position = uniforms.modelViewProjectionMatrix * position;
    output.worldPosition = position;
    return output;
  }
);

/**
 * @brief Convert equirectangular HDR image to cubemap texture
 */
static WGPUTexture
convert_equirectangular_to_cubemap(wgpu_context_t* wgpu_context,
                                   hdr_image_t* hdr, uint32_t size)
{
  /* Create vertex buffer */
  WGPUBuffer cubemap_vertices_buffer = create_buffer_with_data(
    wgpu_context, cube_vertex_array, sizeof(cube_vertex_array),
    WGPUBufferUsage_Vertex);

  /* Create cubemap texture */
  WGPUTextureDescriptor cubemap_desc = {
    .label = STRVIEW("cubemap from equirectangular"),
    .usage = WGPUTextureUsage_TextureBinding
           | WGPUTextureUsage_CopyDst
           | WGPUTextureUsage_RenderAttachment,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width = size,
      .height = size,
      .depthOrArrayLayers = 6,
    },
    .format = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount = 1,
    .viewFormatCount = 0,
    .viewFormats = NULL,
  };

  WGPUTexture cubemap_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &cubemap_desc);

  /* Create source equirectangular texture */
  WGPUTextureDescriptor equirect_desc = {
    .label = STRVIEW("source equirectangular texture"),
    .usage = WGPUTextureUsage_RenderAttachment
           | WGPUTextureUsage_TextureBinding
           | WGPUTextureUsage_CopyDst,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width = hdr->width,
      .height = hdr->height,
      .depthOrArrayLayers = 1,
    },
    .format = WGPUTextureFormat_RGBA16Float,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture equirect_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &equirect_desc);

  /* Upload HDR data to equirectangular texture */
  /* Convert float32 HDR data to float16 for texture upload */
  const size_t pixel_count  = hdr->width * hdr->height * 4;
  const size_t float16_size = pixel_count * sizeof(uint16_t);
  uint16_t* float16_data    = (uint16_t*)malloc(float16_size);

  float* hdr_floats = (float*)hdr->data;
  float max_val     = 0.0f;
  for (size_t i = 0; i < pixel_count; i++) {
    float16_data[i] = float32_to_float16(hdr_floats[i]);
    if (hdr_floats[i] > max_val)
      max_val = hdr_floats[i];
  }

  printf(
    "Uploading HDR data to equirect texture: %dx%d, float16_size=%zu bytes\n",
    hdr->width, hdr->height, float16_size);
  printf("HDR max value: %.2f (should be > 1.0 for HDR)\n", max_val);

  wgpuQueueWriteTexture(
    wgpu_context->queue,
    &(WGPUTexelCopyTextureInfo){
      .texture  = equirect_texture,
      .mipLevel = 0,
      .origin   = (WGPUOrigin3D){0, 0, 0},
      .aspect   = WGPUTextureAspect_All,
    },
    float16_data, float16_size,
    &(WGPUTexelCopyBufferLayout){
      .offset = 0,
      .bytesPerRow
      = 8 * hdr->width, /* RGBA16Float = 4 channels * 2 bytes = 8 bytes/pixel */
      .rowsPerImage = hdr->height,
    },
    &equirect_desc.size);

  free(float16_data);

  /* Create depth texture */
  WGPUTextureDescriptor depth_desc = {
    .label = STRVIEW("cubemap depth texture"),
    .usage = WGPUTextureUsage_RenderAttachment,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width = size,
      .height = size,
      .depthOrArrayLayers = 1,
    },
    .format = wgpu_context->depth_stencil_format,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_desc);
  WGPUTextureView depth_view = wgpuTextureCreateView(depth_texture, NULL);

  /* Fragment shader for equirectangular sampling */
  const char* fragment_shader_wgsl = CODE(
    @group(0) @binding(1) var ourTexture: texture_2d<f32>;
    @group(0) @binding(2) var ourSampler: sampler;

    const inverseAtan = vec2f(0.1591, 0.3183);

    fn sampleSphericalMap(v: vec3f) -> vec2f {
      var uv = vec2f(atan2(v.z, v.x), asin(v.y));
      uv *= inverseAtan;
      uv += 0.5;
      return uv;
    }

    @fragment
    fn main(@location(0) worldPosition: vec4f) -> @location(0) vec4f {
      let uv = sampleSphericalMap(normalize(worldPosition.xyz));
      var color = textureSample(ourTexture, ourSampler, uv).rgb;
      return vec4f(color, 1);
    }
  );

  /* Create shaders */
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, cubemap_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  /* Create pipeline - continuing in next chunk to avoid length limit */
  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("renderToCubemap"),
      .vertex = (WGPUVertexState){
        .module = vert_shader_module,
        .entryPoint = STRVIEW("main"),
        .bufferCount = 1,
        .buffers = &(WGPUVertexBufferLayout){
          .arrayStride = 4 * sizeof(float),
          .stepMode = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes = &(WGPUVertexAttribute){
            .shaderLocation = 0,
            .offset = 0,
            .format = WGPUVertexFormat_Float32x4,
          },
        },
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode = WGPUCullMode_None,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = frag_shader_module,
        .entryPoint = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = WGPUTextureFormat_RGBA8Unorm,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    }
  );

  /* Create sampler and uniform buffer */
  WGPUSampler sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("equirect sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });

  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("cubemap uniform"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4),
    });

  /* Setup projection matrix */
  mat4 projection;
  glm_perspective(GLM_PI_2f, 1.0f, 0.1f, 10.0f, projection);

  WGPUTextureView equirect_view = wgpuTextureCreateView(equirect_texture, NULL);

  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label = STRVIEW("equirect bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
      .entryCount = 3,
      .entries = (WGPUBindGroupEntry[]){
        {.binding = 0, .buffer = uniform_buffer, .offset = 0, .size = sizeof(mat4)},
        {.binding = 1, .textureView = equirect_view},
        {.binding = 2, .sampler = sampler},
      },
    }
  );

  /* Render each cubemap face */
  printf("Rendering %d cubemap faces...\n", 6);
  for (uint32_t face = 0; face < 6; face++) {
    mat4 mvp;
    glm_mat4_mul(projection, cubemap_view_matrices[face], mvp);
    wgpuQueueWriteBuffer(wgpu_context->queue, uniform_buffer, 0, mvp,
                         sizeof(mat4));

    if (face == 0) {
      printf("Face 0 MVP matrix first row: [%.2f, %.2f, %.2f, %.2f]\n",
             mvp[0][0], mvp[0][1], mvp[0][2], mvp[0][3]);
      printf("Equirect texture: %p, Sampler: %p\n", (void*)equirect_texture,
             (void*)sampler);
    }

    WGPUTextureView face_view = wgpuTextureCreateView(
      cubemap_texture, &(WGPUTextureViewDescriptor){
                         .label           = STRVIEW("cubemap face view"),
                         .format          = WGPUTextureFormat_RGBA8Unorm,
                         .dimension       = WGPUTextureViewDimension_2D,
                         .baseMipLevel    = 0,
                         .mipLevelCount   = 1,
                         .baseArrayLayer  = face,
                         .arrayLayerCount = 1,
                       });

    WGPUCommandEncoder command_Encoder
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPURenderPassEncoder pass_encoder = wgpuCommandEncoderBeginRenderPass(
      command_Encoder,
      &(WGPURenderPassDescriptor){
        .label = STRVIEW("cubemap render pass"),
        .colorAttachmentCount = 1,
        .colorAttachments = &(WGPURenderPassColorAttachment){
          .view = face_view,
          .loadOp = WGPULoadOp_Clear,
          .storeOp = WGPUStoreOp_Store,
          .clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0},
          .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        },
        .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
          .view = depth_view,
          .depthLoadOp = WGPULoadOp_Clear,
          .depthStoreOp = WGPUStoreOp_Store,
          .depthClearValue = 1.0f,
          .stencilLoadOp = WGPULoadOp_Clear,
          .stencilStoreOp = WGPUStoreOp_Store,
          .stencilClearValue = 0,
        },
      }
    );

    wgpuRenderPassEncoderSetPipeline(pass_encoder, pipeline);
    wgpuRenderPassEncoderSetViewport(pass_encoder, 0, 0, size, size, 0, 1);
    wgpuRenderPassEncoderSetVertexBuffer(
      pass_encoder, 0, cubemap_vertices_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(pass_encoder, 0, bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(pass_encoder, 36, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass_encoder);

    WGPUCommandBuffer command = wgpuCommandEncoderFinish(command_Encoder, NULL);
    wgpuQueueSubmit(wgpu_context->queue, 1, &command);

    WGPU_RELEASE_RESOURCE(CommandBuffer, command)
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass_encoder)
    WGPU_RELEASE_RESOURCE(CommandEncoder, command_Encoder)
    WGPU_RELEASE_RESOURCE(TextureView, face_view)
  }

  printf("Cubemap conversion complete\n");

  /* Wait for all commands to complete */
  wgpuDeviceTick(wgpu_context->device);
  printf("Device tick complete - cubemap should be ready\n");

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(TextureView, equirect_view)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer)
  WGPU_RELEASE_RESOURCE(Sampler, sampler)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module)
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module)
  WGPU_RELEASE_RESOURCE(Buffer, cubemap_vertices_buffer)
  WGPU_RELEASE_RESOURCE(TextureView, depth_view)
  WGPU_RELEASE_RESOURCE(Texture, depth_texture)
  WGPU_RELEASE_RESOURCE(Texture, equirect_texture)

  return cubemap_texture;
}

/* -------------------------------------------------------------------------- *
 * PBR Shader Creation
 * -------------------------------------------------------------------------- */

/* Maximum shader source size */
#define MAX_SHADER_SOURCE_SIZE (32 * 1024)

/**
 * @brief Create PBR shader with conditional compilation
 */
static char* create_pbr_shader(bool has_uvs, bool has_tangents,
                               bool use_alpha_cutoff, uint32_t shadow_map_size)
{
  static char shader_source[MAX_SHADER_SOURCE_SIZE];
  static char temp_buffer[MAX_SHADER_SOURCE_SIZE];

  /* Build vertex input struct based on mesh attributes */
  char vertex_input_struct[512];
  if (has_uvs && has_tangents) {
    snprintf(vertex_input_struct, sizeof(vertex_input_struct),
             "struct VertexInput {\n"
             "  @location(%d) position: vec3f,\n"
             "  @location(%d) normal: vec3f,\n"
             "  @location(%d) uv: vec2f,\n"
             "  @location(%d) tangent: vec4f,\n"
             "}\n",
             SHADER_LOCATION_POSITION, SHADER_LOCATION_NORMAL,
             SHADER_LOCATION_TEXCOORD_0, SHADER_LOCATION_TANGENT);
  }
  else if (has_uvs) {
    snprintf(vertex_input_struct, sizeof(vertex_input_struct),
             "struct VertexInput {\n"
             "  @location(%d) position: vec3f,\n"
             "  @location(%d) normal: vec3f,\n"
             "  @location(%d) uv: vec2f,\n"
             "}\n",
             SHADER_LOCATION_POSITION, SHADER_LOCATION_NORMAL,
             SHADER_LOCATION_TEXCOORD_0);
  }
  else if (has_tangents) {
    snprintf(vertex_input_struct, sizeof(vertex_input_struct),
             "struct VertexInput {\n"
             "  @location(%d) position: vec3f,\n"
             "  @location(%d) normal: vec3f,\n"
             "  @location(%d) tangent: vec4f,\n"
             "}\n",
             SHADER_LOCATION_POSITION, SHADER_LOCATION_NORMAL,
             SHADER_LOCATION_TANGENT);
  }
  else {
    snprintf(vertex_input_struct, sizeof(vertex_input_struct),
             "struct VertexInput {\n"
             "  @location(%d) position: vec3f,\n"
             "  @location(%d) normal: vec3f,\n"
             "}\n",
             SHADER_LOCATION_POSITION, SHADER_LOCATION_NORMAL);
  }

  /* Build vertex output struct */
  char vertex_output_struct[512];
  if (has_tangents) {
    snprintf(vertex_output_struct, sizeof(vertex_output_struct),
             "struct VertexOutput {\n"
             "  @builtin(position) position: vec4f,\n"
             "  @location(0) normal: vec3f,\n"
             "  @location(1) uv: vec2f,\n"
             "  @location(2) worldPosition: vec3f,\n"
             "  @location(3) shadowPosition: vec3f,\n"
             "  @location(4) tangent: vec4f,\n"
             "}\n");
  }
  else {
    snprintf(vertex_output_struct, sizeof(vertex_output_struct),
             "struct VertexOutput {\n"
             "  @builtin(position) position: vec4f,\n"
             "  @location(0) normal: vec3f,\n"
             "  @location(1) uv: vec2f,\n"
             "  @location(2) worldPosition: vec3f,\n"
             "  @location(3) shadowPosition: vec3f,\n"
             "}\n");
  }

  /* Build UV assignment in vertex shader */
  char uv_assignment[128];
  if (has_uvs) {
    snprintf(uv_assignment, sizeof(uv_assignment), "output.uv = input.uv;");
  }
  else {
    snprintf(uv_assignment, sizeof(uv_assignment), "output.uv = vec2f(0.0);");
  }

  /* Build tangent assignment in vertex shader */
  char tangent_assignment[256];
  if (has_tangents) {
    snprintf(tangent_assignment, sizeof(tangent_assignment),
             "output.tangent = vec4f((models[instance] * "
             "vec4f(input.tangent.xyz, 0.0)).xyz, input.tangent.w);");
  }
  else {
    snprintf(tangent_assignment, sizeof(tangent_assignment), "");
  }

  /* Build normal calculation in fragment shader */
  char normal_calc[512];
  if (has_tangents) {
    snprintf(normal_calc, sizeof(normal_calc),
             "var normalSample = textureSample(normalTexture, normalSampler, "
             "input.uv).rgb;\n"
             "  normalSample = normalize(normalSample * 2.0 - 1.0);\n"
             "  var n = normalize(input.normal);\n"
             "  let t = normalize(input.tangent.xyz);\n"
             "  let b = cross(n, t) * input.tangent.w;\n"
             "  let tbn = mat3x3f(t, b, n);\n"
             "  n = normalize(tbn * normalSample);\n");
  }
  else {
    snprintf(normal_calc, sizeof(normal_calc),
             "var normalSample = textureSample(normalTexture, normalSampler, "
             "input.uv).rgb;\n"
             "  normalSample = normalize(normalSample * 2.0 - 1.0);\n"
             "  let n = normalize(input.normal);\n");
  }

  /* Build alpha cutoff code */
  char alpha_cutoff_code[256];
  if (use_alpha_cutoff) {
    snprintf(alpha_cutoff_code, sizeof(alpha_cutoff_code),
             "if (baseColor.a < material.alphaCutoff) { discard; }");
  }
  else {
    snprintf(alpha_cutoff_code, sizeof(alpha_cutoff_code), "");
  }

  /* Build the shader template in chunks to avoid ISO C99 string length limit */
  // clang-format off
  const char* shader_part1 = CODE(
    struct Scene {
      cameraProjection: mat4x4f,
      cameraView: mat4x4f,
      cameraPosition: vec3f,
      lightPosition: vec3f,
      lightColor: vec3f,
      lightViewProjection: mat4x4f,
    };

    struct Material {
      baseColorFactor: vec4f,
      alphaCutoff: f32,
    };

    @group(0) @binding(0) var<uniform> scene: Scene;
    @group(1) @binding(0) var<storage> models: array<mat4x4f>;

    // Material
    @group(2) @binding(0) var<uniform> material: Material;
    @group(2) @binding(1) var albedoSampler: sampler;
    @group(2) @binding(2) var albedoTexture: texture_2d<f32>;
    @group(2) @binding(3) var normalSampler: sampler;
    @group(2) @binding(4) var normalTexture: texture_2d<f32>;
    @group(2) @binding(5) var roughnessMetallicSampler: sampler;
    @group(2) @binding(6) var roughnessMetallicTexture: texture_2d<f32>;
    @group(2) @binding(7) var aoSampler: sampler;
    @group(2) @binding(8) var aoTexture: texture_2d<f32>;
    @group(2) @binding(9) var emissiveSampler: sampler;
    @group(2) @binding(10) var emissiveTexture: texture_2d<f32>;

    // PBR textures
    @group(3) @binding(0) var samplerBRDF: sampler;
    @group(3) @binding(1) var samplerGeneral: sampler;
    @group(3) @binding(2) var brdfLUT: texture_2d<f32>;
    @group(3) @binding(3) var irradianceMap: texture_cube<f32>;
    @group(3) @binding(4) var prefilterMap: texture_cube<f32>;
    @group(3) @binding(5) var shadowMap: texture_depth_2d;
    @group(3) @binding(6) var shadowSampler: sampler_comparison;

    VERTEX_INPUT_STRUCT

    VERTEX_OUTPUT_STRUCT

    @vertex
    fn vertexMain(input: VertexInput, @builtin(instance_index) instance: u32) -> VertexOutput {
      let positionFromLight = scene.lightViewProjection * models[instance] * vec4f(input.position, 1.0);

      var output: VertexOutput;
      output.position = scene.cameraProjection * scene.cameraView * models[instance] * vec4f(input.position, 1.0);
      output.normal = normalize((models[instance] * vec4f(input.normal, 0.0)).xyz);
      output.worldPosition = (models[instance] * vec4f(input.position, 1.0)).xyz;
      output.shadowPosition = vec3f(
        positionFromLight.xy * vec2f(0.5, -0.5) + vec2f(0.5),
        positionFromLight.z
      );

      UV_ASSIGNMENT
      TANGENT_ASSIGNMENT

      return output;
    }

    DISTRIBUTION_GGX
    GEOMETRY_SCHLICK_GGX
    GEOMETRY_SMITH
    FRESNEL_SCHLICK
    FRESNEL_SCHLICK_ROUGHNESS
    TONE_MAPPING

    const MAX_REFLECTION_LOD = 4.0;
    const PI = 3.14159265359;
  );

  const char* shader_part2 = CODE(
    fn calculateShadows(shadowPosition: vec3f) -> f32 {
      var visibility = 0.0;
      let oneOverShadowDepthTextureSize = 1.0 / SHADOW_MAP_SIZE;
      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let offset = vec2f(vec2(x, y)) * oneOverShadowDepthTextureSize;
          visibility += textureSampleCompare(
            shadowMap,
            shadowSampler,
            shadowPosition.xy + offset,
            shadowPosition.z - 0.005
          );
        }
      }
      visibility /= 9.0;
      return visibility;
    }

    @fragment
    fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
      var visibility = calculateShadows(input.shadowPosition);
      visibility = 1.0;

      let baseColor = textureSample(albedoTexture, albedoSampler, input.uv) * material.baseColorFactor;

      ALPHA_CUTOFF_CODE

      let ao = textureSample(aoTexture, aoSampler, input.uv).r;
      let albedo = baseColor.rgb;

      let roughnessMetallic = textureSample(roughnessMetallicTexture, roughnessMetallicSampler, input.uv);
      let metallic = roughnessMetallic.b;
      let roughness = roughnessMetallic.g;
      let emissive = textureSample(emissiveTexture, emissiveSampler, input.uv).rgb;

      NORMAL_CALC

      let v = normalize(scene.cameraPosition - input.worldPosition);
      let r = reflect(-v, n);
      let f0 = mix(vec3f(0.04), albedo, metallic);

      var lo = vec3f(0.0);
  );

  const char* shader_part3 = CODE(
      {
        let l = normalize(scene.lightPosition - input.worldPosition);
        let h = normalize(v + l);
        let distance = length(scene.lightPosition - input.worldPosition);
        let attenuation = 1.0 / (distance * distance);
        let radiance = scene.lightColor * attenuation;

        let d = distributionGGX(n, h, roughness);
        let g = geometrySmith(n, v, l, roughness);
        let f = fresnelSchlick(max(dot(h, v), 0.0), f0);

        let numerator = d * g * f;
        let denominator = 4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0) + 0.00001;
        let specular = numerator / denominator;

        let kS = f;
        var kD = vec3f(1.0) - kS;
        kD *= 1.0 - metallic;

        let nDotL = max(dot(n, l), 0.00001);
        lo += (kD * albedo / PI + specular) * radiance * nDotL * visibility;
      }

      let f = fresnelSchlickRoughness(max(dot(n, v), 0.00001), f0, roughness);
      let kS = f;
      var kD = vec3f(1.0) - kS;
      kD *= 1.0 - metallic;

      let irradiance = textureSample(irradianceMap, samplerGeneral, n).rgb;
      let diffuse = irradiance * albedo;

      let prefilteredColor = textureSampleLevel(prefilterMap, samplerGeneral, r, roughness * MAX_REFLECTION_LOD).rgb;
      let brdf = textureSample(brdfLUT, samplerBRDF, vec2f(max(dot(n, v), 0.0), roughness)).rg;
      let specular = prefilteredColor * (f * brdf.x + brdf.y);

      let ambient = (kD * diffuse + specular) * ao;

      var color = ambient + lo + emissive;
      color = toneMapping(color);
      color = pow(color, vec3f(1.0 / 2.2));
      return vec4f(color, 1.0);
    }
  );
  // clang-format on

  /* Combine shader parts into one buffer */
  snprintf(shader_source, sizeof(shader_source), "%s%s%s", shader_part1,
           shader_part2, shader_part3);

  /* Replace shader location constants */
  char location_str[32];
  snprintf(location_str, sizeof(location_str), "%d", SHADER_LOCATION_POSITION);
  str_replace_all(shader_source, "SHADER_LOCATION_POSITION", location_str,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  snprintf(location_str, sizeof(location_str), "%d", SHADER_LOCATION_NORMAL);
  str_replace_all(shader_source, "SHADER_LOCATION_NORMAL", location_str,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  snprintf(location_str, sizeof(location_str), "%d",
           SHADER_LOCATION_TEXCOORD_0);
  str_replace_all(shader_source, "SHADER_LOCATION_TEXCOORD_0", location_str,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  snprintf(location_str, sizeof(location_str), "%d", SHADER_LOCATION_TANGENT);
  str_replace_all(shader_source, "SHADER_LOCATION_TANGENT", location_str,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  /* Replace struct definitions */
  str_replace_all(shader_source, "VERTEX_INPUT_STRUCT", vertex_input_struct,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "VERTEX_OUTPUT_STRUCT", vertex_output_struct,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  /* Replace code blocks */
  str_replace_all(shader_source, "UV_ASSIGNMENT", uv_assignment, temp_buffer,
                  sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "TANGENT_ASSIGNMENT", tangent_assignment,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "NORMAL_CALC", normal_calc, temp_buffer,
                  sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "ALPHA_CUTOFF_CODE", alpha_cutoff_code,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  /* Replace shadow map size */
  char size_str[32];
  snprintf(size_str, sizeof(size_str), "%u.0", shadow_map_size);
  str_replace_all(shader_source, "SHADOW_MAP_SIZE", size_str, temp_buffer,
                  sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  /* Replace PBR shader functions - IMPORTANT: Replace longer names first! */
  str_replace_all(shader_source, "FRESNEL_SCHLICK_ROUGHNESS",
                  fresnel_schlick_roughness_wgsl, temp_buffer,
                  sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "FRESNEL_SCHLICK", fresnel_schlick_wgsl,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "GEOMETRY_SCHLICK_GGX",
                  geometry_schlick_ggx_wgsl, temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "GEOMETRY_SMITH", geometry_smith_wgsl,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "DISTRIBUTION_GGX", distribution_ggx_wgsl,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "TONE_MAPPING", tone_mapping_aces_wgsl,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  return shader_source;
}

/* -------------------------------------------------------------------------- *
 * BRDF Convolution - LUT Generation
 * -------------------------------------------------------------------------- */

/* Quad vertices for BRDF LUT rendering */
static const float brdf_quad_vertices[] = {
  /* clang-format off */
  /* position        uv */
  -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
   1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
   1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
  -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
   1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
  -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
  /* clang-format on */
};

/* BRDF LUT vertex shader */
static const char* brdf_lut_vertex_shader_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) Position : vec4f, @location(0) uv : vec2f,
  }

  @vertex fn main(@location(0) position : vec3f, @location(1) uv : vec2f)
    ->VertexOutput {
      var output : VertexOutput;
      output.Position = vec4f(position, 1.0);
      output.uv       = uv;
      return output;
    });

/* BRDF LUT fragment shader */
static const char* brdf_lut_fragment_shader_wgsl = CODE(
  const PI : f32 = 3.14159265359;

  // http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
  // efficient VanDerCorpus calculation.
  fn radicalInverseVdC(bits : u32)
    ->f32 {
      var result = bits;
      result     = (bits << 16u) | (bits >> 16u);
      result = ((result & 0x55555555u) << 1u) | ((result & 0xAAAAAAAAu) >> 1u);
      result = ((result & 0x33333333u) << 2u) | ((result & 0xCCCCCCCCu) >> 2u);
      result = ((result & 0x0F0F0F0Fu) << 4u) | ((result & 0xF0F0F0F0u) >> 4u);
      result = ((result & 0x00FF00FFu) << 8u) | ((result & 0xFF00FF00u) >> 8u);
      return f32(result) * 2.3283064365386963e-10;
    }

  fn hammersley(i : u32, n : u32)
    ->vec2f { return vec2f(f32(i) / f32(n), radicalInverseVdC(i)); }

  fn importanceSampleGGX(xi : vec2f, n : vec3f, roughness : f32)
    ->vec3f {
      let a = roughness * roughness;

      let phi      = 2.0 * PI * xi.x;
      let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
      let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

      // from spherical coordinates to cartesian coordinates - halfway vector
      let h = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

      // from tangent-space H vector to world-space sample vector
      let up : vec3f = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0),
                              abs(n.z) < 0.999);
      let tangent    = normalize(cross(up, n));
      let bitangent  = cross(n, tangent);

      let sampleVec = tangent * h.x + bitangent * h.y + n * h.z;
      return normalize(sampleVec);
    }

  fn geometrySmith(n : vec3f, v : vec3f, l : vec3f, roughness : f32)
    ->f32 {
      let nDotV = max(dot(n, v), 0.0);
      let nDotL = max(dot(n, l), 0.0);
      let ggx2  = geometrySchlickGGX(nDotV, roughness);
      let ggx1  = geometrySchlickGGX(nDotL, roughness);
      return ggx1 * ggx2;
    }

  // This one is different from the standard PBR version
  fn geometrySchlickGGX(nDotV : f32, roughness : f32)
    ->f32 {
      let a = roughness;
      let k = (a * a) / 2.0;

      let nom   = nDotV;
      let denom = nDotV * (1.0 - k) + k;

      return nom / denom;
    }

  fn integrateBRDF(NdotV : f32, roughness : f32)
    ->vec2f {
      var V : vec3f;
      V.x = sqrt(1.0 - NdotV * NdotV);
      V.y = 0.0;
      V.z = NdotV;

      var A : f32 = 0.0;
      var B : f32 = 0.0;

      let N = vec3f(0.0, 0.0, 1.0);

      let SAMPLE_COUNT : u32 = 1024u;
      for (var i : u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let Xi : vec2f = hammersley(i, SAMPLE_COUNT);
        let H : vec3f  = importanceSampleGGX(Xi, N, roughness);
        let L : vec3f  = normalize(2.0 * dot(V, H) * H - V);

        let NdotL : f32 = max(L.z, 0.0);
        let NdotH : f32 = max(H.z, 0.0);
        let VdotH : f32 = max(dot(V, H), 0.0);

        if (NdotL > 0.0) {
          let G : f32     = geometrySmith(N, V, L, roughness);
          let G_Vis : f32 = (G * VdotH) / (NdotH * NdotV);
          let Fc : f32    = pow(1.0 - VdotH, 5.0);

          A += (1.0 - Fc) * G_Vis;
          B += Fc * G_Vis;
        }
      }
      A /= f32(SAMPLE_COUNT);
      B /= f32(SAMPLE_COUNT);
      return vec2f(A, B);
    }

  @fragment fn main(@location(0) uv : vec2f)
    ->@location(0) vec2f {
      let result = integrateBRDF(uv.x, 1 - uv.y);
      return result;
    });

/**
 * @brief Generate BRDF convolution LUT texture
 */
static WGPUTexture generate_brdf_lut(wgpu_context_t* wgpu_context,
                                     uint32_t size)
{
  /* Create the BRDF LUT texture */
  WGPUTextureDescriptor texture_desc = {
    .label = STRVIEW("BRDF LUT"),
    .usage = WGPUTextureUsage_RenderAttachment
           | WGPUTextureUsage_TextureBinding
           | WGPUTextureUsage_CopyDst,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width  = size,
      .height = size,
      .depthOrArrayLayers = 1,
    },
    .format = WGPUTextureFormat_RG16Float,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  /* Create shaders */
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, brdf_lut_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, brdf_lut_fragment_shader_wgsl);

  /* Create render pipeline */
  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("BRDF convolution pipeline"),
      .vertex = (WGPUVertexState){
        .module = vert_shader_module,
        .entryPoint = STRVIEW("main"),
        .bufferCount = 1,
        .buffers = &(WGPUVertexBufferLayout){
          .arrayStride = 5 * sizeof(float),
          .stepMode = WGPUVertexStepMode_Vertex,
          .attributeCount = 2,
          .attributes = (WGPUVertexAttribute[]){
            {
              .shaderLocation = 0,
              .offset = 0,
              .format = WGPUVertexFormat_Float32x3,
            },
            {
              .shaderLocation = 1,
              .offset = 3 * sizeof(float),
              .format = WGPUVertexFormat_Float32x2,
            },
          },
        },
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode = WGPUCullMode_None,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = frag_shader_module,
        .entryPoint = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = WGPUTextureFormat_RG16Float,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    }
  );

  /* Create depth texture */
  WGPUTextureDescriptor depth_texture_desc = {
    .label = STRVIEW("BRDF LUT depth"),
    .usage = WGPUTextureUsage_RenderAttachment,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
     .width  = size,
     .height = size,
     .depthOrArrayLayers = 1,
    },
    .format = wgpu_context->depth_stencil_format,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_texture_desc);
  WGPUTextureView depth_texture_view
    = wgpuTextureCreateView(depth_texture, NULL);

  /* Create vertex buffer */
  WGPUBuffer vertex_buffer = create_buffer_with_data(
    wgpu_context, brdf_quad_vertices, sizeof(brdf_quad_vertices),
    WGPUBufferUsage_Vertex);

  /* Render the BRDF LUT */
  WGPUTextureView texture_view = wgpuTextureCreateView(texture, NULL);

  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("BRDF LUT command encoder"),
                          });

  WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
    encoder,
    &(WGPURenderPassDescriptor){
      .label = STRVIEW("BRDF convolution"),
      .colorAttachmentCount = 1,
      .colorAttachments = &(WGPURenderPassColorAttachment){
        .view = texture_view,
        .loadOp = WGPULoadOp_Clear,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      },
      .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
        .view = depth_texture_view,
        .depthLoadOp = WGPULoadOp_Clear,
        .depthStoreOp = WGPUStoreOp_Store,
        .depthClearValue = 1.0f,
        .stencilLoadOp = WGPULoadOp_Clear,
        .stencilStoreOp = WGPUStoreOp_Store,
        .stencilClearValue = 0,
      },
    }
  );

  wgpuRenderPassEncoderSetPipeline(pass, pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
  wgpuRenderPassEncoderEnd(pass);

  WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command);

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(CommandBuffer, command)
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)
  WGPU_RELEASE_RESOURCE(TextureView, texture_view)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module)
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module)
  WGPU_RELEASE_RESOURCE(Buffer, vertex_buffer)
  WGPU_RELEASE_RESOURCE(TextureView, depth_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, depth_texture)

  return texture;
}

/**
 * @brief Create a roughness-metallic texture
 */
static WGPUTexture
create_roughness_metallic_texture(wgpu_context_t* wgpu_context, float roughness,
                                  float metallic)
{
  const uint8_t data[4]
    = {0, (uint8_t)(roughness * 255.0f), (uint8_t)(metallic * 255.0f), 0};

  WGPUTextureDescriptor texture_desc = {
    .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width  = 1,
      .height = 1,
      .depthOrArrayLayers = 1,
    },
    .format = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = texture,
                          .mipLevel = 0,
                          .origin   = (WGPUOrigin3D){0, 0, 0},
                          .aspect   = WGPUTextureAspect_All,
                        },
                        data, 4,
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = 4,
                          .rowsPerImage = 1,
                        },
                        &texture_desc.size);

  return texture;
}

/* -------------------------------------------------------------------------- *
 * Utility Functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Align value to a multiple
 * @param value Value to align
 * @param multiple Multiple to align to
 * @return Aligned value
 */
static uint32_t align_to(uint32_t value, uint32_t multiple)
{
  return ((value + multiple - 1) / multiple) * multiple;
}

/* -------------------------------------------------------------------------- *
 * WebGPU Example - GLTF PBR IBL Application State
 * -------------------------------------------------------------------------- */

/* State struct to hold all application data */
/* -------------------------------------------------------------------------- *
 * GLTF Rendering Structures
 * -------------------------------------------------------------------------- */

/**
 * @brief Material data for PBR rendering
 */
typedef struct {
  WGPUBindGroup bind_group;
  WGPUBuffer uniform_buffer;
  vec4 base_color_factor;
  float alpha_cutoff;
  bool has_alpha_cutoff;
} gltf_material_t;

/**
 * @brief Primitive rendering data
 */
typedef struct {
  WGPURenderPipeline pipeline;
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  uint32_t index_count;
  WGPUIndexFormat index_format;
  gltf_material_t* material;
  uint32_t instance_count;
  uint32_t instance_offset;
  /* Pipeline creation parameters (for deferred creation) */
  bool has_uvs;
  bool has_tangents;
  bool use_alpha_cutoff;
  bool pipeline_created;
} gltf_primitive_t;

/**
 * @brief Mesh data containing multiple primitives
 */
typedef struct {
  gltf_primitive_t* primitives;
  uint32_t primitive_count;
} gltf_mesh_t;

/* -------------------------------------------------------------------------- *
 * Main State Structure
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* GLTF data */
  cgltf_data* gltf_data;
  uint8_t* gltf_buffer;
  size_t gltf_buffer_size;

  /* GLTF rendering data */
  gltf_mesh_t* meshes;
  uint32_t mesh_count;
  gltf_material_t* materials;
  uint32_t material_count;
  WGPUBuffer instance_buffer;
  uint32_t total_instances;

  /* Textures */
  WGPUTexture cubemap_texture;
  WGPUTexture irradiance_map;
  WGPUTexture prefilter_map;
  WGPUTexture brdf_lut;

  /* GLTF textures */
  WGPUTexture* gltf_textures;
  uint32_t gltf_texture_count;

  /* HDR loading */
  uint8_t* hdr_buffer;
  size_t hdr_buffer_size;

  /* Rendering pipeline */
  WGPURenderPipeline render_pipeline;
  WGPUBindGroupLayout scene_bind_group_layout;
  WGPUBindGroupLayout instance_bind_group_layout;
  WGPUBindGroupLayout material_bind_group_layout;
  WGPUBindGroupLayout pbr_bind_group_layout;
  WGPUBindGroup scene_bind_group;
  WGPUBindGroup instance_bind_group;
  WGPUBindGroup pbr_bind_group;
  WGPUBuffer scene_uniform_buffer;
  WGPUBuffer vertex_buffer;
  uint32_t vertex_count;

  /* PBR samplers */
  WGPUSampler brdf_sampler;
  WGPUSampler shadow_sampler;

  /* Temporary test renderer (TODO: remove when full PBR pipeline is
   * implemented) */
  WGPUBindGroupLayout camera_bind_group_layout;
  WGPUBindGroup camera_bind_group;
  WGPUBuffer camera_uniform_buffer;

  /* Default textures */
  WGPUTexture default_white_texture;
  WGPUTexture default_normal_texture;
  WGPUTexture default_roughness_metallic_texture;
  WGPUTexture placeholder_shadow_map;
  WGPUSampler default_sampler;

  /* Skybox rendering */
  WGPURenderPipeline skybox_pipeline;
  WGPUBindGroupLayout skybox_bind_group_layout;
  WGPUBindGroup skybox_bind_group;
  WGPUBuffer skybox_uniform_buffer;
  WGPUBuffer skybox_vertex_buffer;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI Settings */
  struct {
    uint32_t cubemap_size;
    uint32_t irradiance_map_size;
    uint32_t prefilter_map_size;
    uint32_t roughness_levels;
    uint32_t brdf_lut_size;
    uint32_t sample_count;
    uint32_t shadow_map_size;
  } settings;

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1, 0.2, 0.3, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .settings = {
    .cubemap_size = 512,
    .irradiance_map_size = 32,
    .prefilter_map_size = 128,
    .roughness_levels = 5,
    .brdf_lut_size = 512,
    .sample_count = 1,
    .shadow_map_size = 4096,
  },
  .initialized = false,
};

/* -------------------------------------------------------------------------- *
 * Forward Declarations
 * -------------------------------------------------------------------------- */

static void init_pbr_bind_group(wgpu_context_t* wgpu_context);
static void init_skybox(wgpu_context_t* wgpu_context);
static WGPURenderPipeline create_pbr_pipeline(wgpu_context_t* wgpu_context,
                                              bool has_uvs, bool has_tangents,
                                              bool use_alpha_cutoff);
static gltf_material_t create_material_bind_group(wgpu_context_t* wgpu_context,
                                                  cgltf_material* material);

/* -------------------------------------------------------------------------- *
 * Mipmap Generation
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUSampler sampler;
  WGPURenderPipeline pipeline;
  WGPUBindGroupLayout bind_group_layout;
} mipmap_generator_t;

static mipmap_generator_t mipmap_gen = {0};

/**
 * @brief Initialize mipmap generator
 */
static void init_mipmap_generator(wgpu_context_t* wgpu_context)
{
  /* Create sampler */
  mipmap_gen.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("mip generator sampler"),
                            .minFilter     = WGPUFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });

  /* Shader module for mipmap generation */
  // clang-format off
  const char* mipmap_shader_wgsl = CODE(
    struct VSOutput {
      @builtin(position) position: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex fn vs(
      @builtin(vertex_index) vertexIndex: u32
    ) -> VSOutput {
      var pos = array<vec2f, 6>(
        vec2f(0.0,  0.0), // center
        vec2f(1.0,  0.0), // right, center
        vec2f(0.0,  1.0), // center, top

        vec2f(0.0,  1.0), // center, top
        vec2f(1.0,  0.0), // right, center
        vec2f(1.0,  1.0), // right, top
      );

      var vsOutput: VSOutput;
      let xy = pos[vertexIndex];
      vsOutput.position = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
      vsOutput.uv = vec2f(xy.x, 1.0 - xy.y);
      return vsOutput;
    }

    @group(0) @binding(0) var ourSampler: sampler;
    @group(0) @binding(1) var ourTexture: texture_2d<f32>;

    @fragment fn fs(fsInput: VSOutput) -> @location(0) vec4f {
      return textureSample(ourTexture, ourSampler, fsInput.uv);
    }
  );
  // clang-format on

  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, mipmap_shader_wgsl);

  /* Create pipeline */
  mipmap_gen.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("mip level generator"),
      .layout = NULL,
      .vertex = (WGPUVertexState){
        .module = shader_module,
        .entryPoint = STRVIEW("vs"),
      },
      .fragment = &(WGPUFragmentState){
        .module = shader_module,
        .entryPoint = STRVIEW("fs"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = WGPUTextureFormat_RGBA8Unorm,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
    }
  );

  mipmap_gen.bind_group_layout
    = wgpuRenderPipelineGetBindGroupLayout(mipmap_gen.pipeline, 0);

  WGPU_RELEASE_RESOURCE(ShaderModule, shader_module)
}

/**
 * @brief Generate mipmaps for a texture
 */
static void generate_mipmaps(wgpu_context_t* wgpu_context, WGPUTexture texture,
                             uint32_t base_width, uint32_t base_height,
                             uint32_t mip_level_count)
{
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("mip gen encoder"),
                          });

  uint32_t width  = base_width;
  uint32_t height = base_height;

  for (uint32_t current_mip = 1; current_mip < mip_level_count; current_mip++) {
    uint32_t next_width  = width > 1 ? width / 2 : 1;
    uint32_t next_height = height > 1 ? height / 2 : 1;

    WGPUTextureView src_view
      = wgpuTextureCreateView(texture, &(WGPUTextureViewDescriptor){
                                         .baseMipLevel    = current_mip - 1,
                                         .mipLevelCount   = 1,
                                         .arrayLayerCount = 1,
                                       });

    WGPUTextureView dst_view
      = wgpuTextureCreateView(texture, &(WGPUTextureViewDescriptor){
                                         .baseMipLevel    = current_mip,
                                         .mipLevelCount   = 1,
                                         .arrayLayerCount = 1,
                                       });

    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label = STRVIEW("mipmap generator"),
        .layout = mipmap_gen.bind_group_layout,
        .entryCount = 2,
        .entries = (WGPUBindGroupEntry[]){
          {.binding = 0, .sampler = mipmap_gen.sampler},
          {.binding = 1, .textureView = src_view},
        },
      }
    );

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      encoder,
      &(WGPURenderPassDescriptor){
        .colorAttachmentCount = 1,
        .colorAttachments = &(WGPURenderPassColorAttachment){
          .view = dst_view,
          .loadOp = WGPULoadOp_Clear,
          .storeOp = WGPUStoreOp_Store,
          .clearValue = (WGPUColor){0.0, 0.0, 0.0, 0.0},
          .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        },
      }
    );

    wgpuRenderPassEncoderSetPipeline(pass, mipmap_gen.pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);

    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
    WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
    WGPU_RELEASE_RESOURCE(TextureView, dst_view)
    WGPU_RELEASE_RESOURCE(TextureView, src_view)

    width  = next_width;
    height = next_height;
  }

  WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command);

  WGPU_RELEASE_RESOURCE(CommandBuffer, command)
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)
}

/**
 * @brief Cleanup mipmap generator
 */
static void cleanup_mipmap_generator(void)
{
  WGPU_RELEASE_RESOURCE(BindGroupLayout, mipmap_gen.bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, mipmap_gen.pipeline)
  WGPU_RELEASE_RESOURCE(Sampler, mipmap_gen.sampler)
}

/* -------------------------------------------------------------------------- *
 * GLTF Loading Utilities
 * -------------------------------------------------------------------------- */

/**
 * @brief Calculate number of mip levels for a texture
 */
static uint32_t calculate_mip_levels(uint32_t width, uint32_t height)
{
  uint32_t levels = 1;
  uint32_t size   = width > height ? width : height;
  while (size > 1) {
    size /= 2;
    levels++;
  }
  return levels;
}

/**
 * @brief Load a file into memory synchronously
 */
static bool load_file_sync(const char* path, uint8_t** buffer, size_t* size)
{
  FILE* file = fopen(path, "rb");
  if (!file) {
    printf("Failed to open file: %s\n", path);
    return false;
  }

  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  *buffer = (uint8_t*)malloc(file_size);
  if (!*buffer) {
    fclose(file);
    return false;
  }

  size_t read_size = fread(*buffer, 1, file_size, file);
  fclose(file);

  if (read_size != (size_t)file_size) {
    free(*buffer);
    *buffer = NULL;
    return false;
  }

  *size = file_size;
  printf("Loaded file: %s (%zu bytes)\n", path, *size);
  return true;
}

/**
 * @brief Callback for GLTF file loading
 */
static void gltf_fetch_callback(const sfetch_response_t* response)
{
  if (response->fetched) {
    state.gltf_buffer = (uint8_t*)malloc(response->data.size);
    if (state.gltf_buffer) {
      memcpy(state.gltf_buffer, response->data.ptr, response->data.size);
      state.gltf_buffer_size = response->data.size;
    }
  }
}

/**
 * @brief Callback for HDR file loading
 */
static void hdr_fetch_callback(const sfetch_response_t* response)
{
  if (response->fetched) {
    state.hdr_buffer = (uint8_t*)malloc(response->data.size);
    if (state.hdr_buffer) {
      memcpy(state.hdr_buffer, response->data.ptr, response->data.size);
      state.hdr_buffer_size = response->data.size;
    }
  }
}

/**
 * @brief Load GLTF model asynchronously
 */
static void load_gltf_model(const char* path)
{
  sfetch_send(&(sfetch_request_t){
    .path = path,
    .callback = gltf_fetch_callback,
    .buffer = {
      .ptr = NULL,
      .size = 0,
    },
  });
}

/**
 * @brief Load HDR environment asynchronously
 */
static void load_hdr_environment(const char* path)
{
  sfetch_send(&(sfetch_request_t){
    .path = path,
    .callback = hdr_fetch_callback,
    .buffer = {
      .ptr = NULL,
      .size = 0,
    },
  });
}

/**
 * @brief Process loaded GLTF data
 */
static void process_gltf_data(wgpu_context_t* wgpu_context)
{
  if (!state.gltf_buffer || state.gltf_data || state.meshes) {
    return; /* Already processed or not loaded yet */
  }

  /* Parse GLTF data */
  cgltf_options options = {0};
  cgltf_result result   = cgltf_parse(&options, state.gltf_buffer,
                                      state.gltf_buffer_size, &state.gltf_data);

  if (result != cgltf_result_success) {
    return;
  }

  /* Load buffers */
  result = cgltf_load_buffers(&options, state.gltf_data, NULL);
  if (result != cgltf_result_success) {
    cgltf_free(state.gltf_data);
    state.gltf_data = NULL;
    return;
  }

  /* Create textures from images */
  if (state.gltf_data->images_count > 0) {
    state.gltf_textures = (WGPUTexture*)calloc(state.gltf_data->images_count,
                                               sizeof(WGPUTexture));
    state.gltf_texture_count = (uint32_t)state.gltf_data->images_count;

    for (size_t i = 0; i < state.gltf_data->images_count; i++) {
      cgltf_image* image = &state.gltf_data->images[i];

      /* Load image data */
      int width, height, channels;
      uint8_t* pixels = NULL;

      if (image->buffer_view) {
        cgltf_buffer_view* view = image->buffer_view;
        uint8_t* data           = (uint8_t*)view->buffer->data + view->offset;
        pixels = stbi_load_from_memory(data, (int)view->size, &width, &height,
                                       &channels, 4);
      }

      if (pixels) {
        /* Calculate mip levels */
        uint32_t mip_levels = calculate_mip_levels(width, height);
        /* Create texture */
        WGPUTextureDescriptor tex_desc = {
          .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                 | WGPUTextureUsage_RenderAttachment,
          .dimension = WGPUTextureDimension_2D,
          .size = (WGPUExtent3D){
            .width = (uint32_t)width,
            .height = (uint32_t)height,
            .depthOrArrayLayers = 1,
          },
          .format = WGPUTextureFormat_RGBA8Unorm,
          .mipLevelCount = mip_levels,
          .sampleCount = 1,
        };

        state.gltf_textures[i]
          = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);

        /* Upload texture data */
        wgpuQueueWriteTexture(wgpu_context->queue,
                              &(WGPUTexelCopyTextureInfo){
                                .texture  = state.gltf_textures[i],
                                .mipLevel = 0,
                                .origin   = (WGPUOrigin3D){0, 0, 0},
                                .aspect   = WGPUTextureAspect_All,
                              },
                              pixels, width * height * 4,
                              &(WGPUTexelCopyBufferLayout){
                                .offset       = 0,
                                .bytesPerRow  = width * 4,
                                .rowsPerImage = (uint32_t)height,
                              },
                              &tex_desc.size);

        /* Generate mipmaps */
        if (mip_levels > 1) {
          generate_mipmaps(wgpu_context, state.gltf_textures[i], width, height,
                           mip_levels);
        }
        stbi_image_free(pixels);
      }
    }
  }

  /* Create materials */
  if (state.gltf_data->materials_count > 0) {
    state.material_count = (uint32_t)state.gltf_data->materials_count;
    state.materials
      = (gltf_material_t*)calloc(state.material_count, sizeof(gltf_material_t));
    for (size_t i = 0; i < state.gltf_data->materials_count; i++) {
      state.materials[i] = create_material_bind_group(
        wgpu_context, &state.gltf_data->materials[i]);
    }
  }

  /* Count total instances needed (one per mesh for now) */
  state.total_instances = (uint32_t)state.gltf_data->meshes_count;

  /* Create instance buffer with identity matrices for now */
  if (state.total_instances > 0) {
    size_t instance_data_size = state.total_instances * sizeof(mat4);
    mat4* instance_matrices
      = (mat4*)calloc(state.total_instances, sizeof(mat4));

    for (uint32_t i = 0; i < state.total_instances; i++) {
      glm_mat4_identity(instance_matrices[i]);
    }

    state.instance_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label            = STRVIEW("instance buffer"),
        .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size             = instance_data_size,
        .mappedAtCreation = false,
      });

    wgpuQueueWriteBuffer(wgpu_context->queue, state.instance_buffer, 0,
                         instance_matrices, instance_data_size);
    free(instance_matrices);
  }

  /* Create meshes and primitives */
  if (state.gltf_data->meshes_count > 0) {
    state.mesh_count = (uint32_t)state.gltf_data->meshes_count;
    state.meshes = (gltf_mesh_t*)calloc(state.mesh_count, sizeof(gltf_mesh_t));

    for (size_t mesh_idx = 0; mesh_idx < state.gltf_data->meshes_count;
         mesh_idx++) {
      cgltf_mesh* mesh = &state.gltf_data->meshes[mesh_idx];
      state.meshes[mesh_idx].primitive_count = (uint32_t)mesh->primitives_count;
      state.meshes[mesh_idx].primitives      = (gltf_primitive_t*)calloc(
        mesh->primitives_count, sizeof(gltf_primitive_t));

      for (size_t prim_idx = 0; prim_idx < mesh->primitives_count; prim_idx++) {
        cgltf_primitive* primitive = &mesh->primitives[prim_idx];
        gltf_primitive_t* gpu_prim
          = &state.meshes[mesh_idx].primitives[prim_idx];

        /* Check what attributes we have */
        bool has_uvs = false, has_tangents = false;
        cgltf_accessor *pos_accessor = NULL, *norm_accessor = NULL;
        cgltf_accessor *uv_accessor = NULL, *tan_accessor = NULL;

        for (size_t attr_idx = 0; attr_idx < primitive->attributes_count;
             attr_idx++) {
          cgltf_attribute* attr = &primitive->attributes[attr_idx];
          if (attr->type == cgltf_attribute_type_position)
            pos_accessor = attr->data;
          else if (attr->type == cgltf_attribute_type_normal)
            norm_accessor = attr->data;
          else if (attr->type == cgltf_attribute_type_texcoord) {
            uv_accessor = attr->data;
            has_uvs     = true;
          }
          else if (attr->type == cgltf_attribute_type_tangent) {
            tan_accessor = attr->data;
            has_tangents = true;
          }
        }

        if (!pos_accessor || !norm_accessor) {
          continue; /* Skip invalid primitive */
        }

        /* Build interleaved vertex buffer */
        size_t vertex_count  = pos_accessor->count;
        size_t vertex_stride = 3 + 3; /* pos + normal */
        if (has_uvs)
          vertex_stride += 2;
        if (has_tangents)
          vertex_stride += 4;

        float* vertices
          = (float*)calloc(vertex_count * vertex_stride, sizeof(float));

        for (size_t v = 0; v < vertex_count; v++) {
          size_t offset = 0;

          /* Position */
          float pos[3];
          cgltf_accessor_read_float(pos_accessor, v, pos, 3);
          memcpy(&vertices[v * vertex_stride + offset], pos, 3 * sizeof(float));
          offset += 3;

          /* Normal */
          float norm[3];
          cgltf_accessor_read_float(norm_accessor, v, norm, 3);
          memcpy(&vertices[v * vertex_stride + offset], norm,
                 3 * sizeof(float));
          offset += 3;

          /* UV */
          if (has_uvs) {
            float uv[2];
            cgltf_accessor_read_float(uv_accessor, v, uv, 2);
            memcpy(&vertices[v * vertex_stride + offset], uv,
                   2 * sizeof(float));
            offset += 2;
          }

          /* Tangent */
          if (has_tangents) {
            float tan[4];
            cgltf_accessor_read_float(tan_accessor, v, tan, 4);
            memcpy(&vertices[v * vertex_stride + offset], tan,
                   4 * sizeof(float));
            offset += 4;
          }
        }

        /* Create vertex buffer */
        size_t vertex_buffer_size
          = vertex_count * vertex_stride * sizeof(float);
        gpu_prim->vertex_buffer = create_buffer_with_data(
          wgpu_context, vertices, vertex_buffer_size, WGPUBufferUsage_Vertex);
        free(vertices);

        /* Create index buffer */
        if (primitive->indices) {
          cgltf_accessor* indices = primitive->indices;
          gpu_prim->index_count   = (uint32_t)indices->count;

          if (indices->component_type == cgltf_component_type_r_16u) {
            gpu_prim->index_format = WGPUIndexFormat_Uint16;
            uint16_t* index_data
              = (uint16_t*)calloc(indices->count, sizeof(uint16_t));
            for (size_t i = 0; i < indices->count; i++) {
              index_data[i] = (uint16_t)cgltf_accessor_read_index(indices, i);
            }
            gpu_prim->index_buffer = create_buffer_with_data(
              wgpu_context, index_data, indices->count * sizeof(uint16_t),
              WGPUBufferUsage_Index);
            free(index_data);
          }
          else {
            gpu_prim->index_format = WGPUIndexFormat_Uint32;
            uint32_t* index_data
              = (uint32_t*)calloc(indices->count, sizeof(uint32_t));
            for (size_t i = 0; i < indices->count; i++) {
              index_data[i] = (uint32_t)cgltf_accessor_read_index(indices, i);
            }
            gpu_prim->index_buffer = create_buffer_with_data(
              wgpu_context, index_data, indices->count * sizeof(uint32_t),
              WGPUBufferUsage_Index);
            free(index_data);
          }
        }

        /* Assign material */
        if (primitive->material && state.materials) {
          size_t mat_idx = primitive->material - state.gltf_data->materials;
          if (mat_idx < state.material_count) {
            gpu_prim->material = &state.materials[mat_idx];
          }
        }

        /* Store pipeline creation parameters for later */
        gpu_prim->has_uvs      = has_uvs;
        gpu_prim->has_tangents = has_tangents;
        gpu_prim->use_alpha_cutoff
          = primitive->material
            && primitive->material->alpha_mode == cgltf_alpha_mode_mask;
        gpu_prim->pipeline_created = false;
        gpu_prim->pipeline         = NULL; /* Will be created later */

        /* Set instance data */
        gpu_prim->instance_count  = 1;
        gpu_prim->instance_offset = (uint32_t)mesh_idx;
      }
    }
  }
}

/**
 * @brief Process loaded HDR data and generate IBL textures
 */
static void process_hdr_data(wgpu_context_t* wgpu_context)
{
  if (!state.hdr_buffer || state.cubemap_texture) {
    return;
  }

  /* Load HDR image */
  hdr_image_t hdr;
  if (!load_hdr_image(state.hdr_buffer, state.hdr_buffer_size, &hdr)) {
    return;
  }

  /* Convert to cubemap */
  printf("Converting HDR to cubemap (size: %d)...\n",
         state.settings.cubemap_size);
  state.cubemap_texture = convert_equirectangular_to_cubemap(
    wgpu_context, &hdr, state.settings.cubemap_size);
  printf("Cubemap created: %p\n", (void*)state.cubemap_texture);

  /* Generate irradiance map */
  printf("Generating irradiance map (size: %d)...\n",
         state.settings.irradiance_map_size);
  state.irradiance_map = generate_irradiance_map(
    wgpu_context, state.cubemap_texture, state.settings.irradiance_map_size);
  printf("Irradiance map created: %p\n", (void*)state.irradiance_map);

  /* Generate prefilter map */
  printf("Generating prefilter map (size: %d, levels: %d)...\n",
         state.settings.prefilter_map_size, state.settings.roughness_levels);
  state.prefilter_map = generate_prefilter_map(
    wgpu_context, state.cubemap_texture, state.settings.prefilter_map_size,
    state.settings.roughness_levels);
  printf("Prefilter map created: %p\n", (void*)state.prefilter_map);

  /* Free HDR data */
  free_hdr_image(&hdr);

  /* Initialize skybox rendering now that cubemap is ready */
  printf("Initializing skybox...\n");
  init_skybox(wgpu_context);
  printf("Skybox initialized\n");

  /* Initialize PBR bind group now that all IBL textures are ready */
  init_pbr_bind_group(wgpu_context);

  /* Create pipelines for GLTF primitives now that IBL textures exist */
  if (state.meshes) {
    for (uint32_t mesh_idx = 0; mesh_idx < state.mesh_count; mesh_idx++) {
      gltf_mesh_t* mesh = &state.meshes[mesh_idx];
      for (uint32_t prim_idx = 0; prim_idx < mesh->primitive_count;
           prim_idx++) {
        gltf_primitive_t* prim = &mesh->primitives[prim_idx];
        if (!prim->pipeline_created) {
          prim->pipeline
            = create_pbr_pipeline(wgpu_context, prim->has_uvs,
                                  prim->has_tangents, prim->use_alpha_cutoff);
          prim->pipeline_created = true;
        }
      }
    }
  }
}

/**
 * @brief Create simple cube geometry for testing
 */
static void init_cube_geometry(wgpu_context_t* wgpu_context)
{
  /* Complete cube vertices (position + normal + texcoord) */
  /* Each face: 2 triangles = 6 vertices */
  static const float cube_vertices[] = {
    /* Front face (+Z) */
    -1.0f,
    -1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f, /*  0 */
    1.0f,
    -1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    0.0f, /*  1 */
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f, /*  2 */
    -1.0f,
    -1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f, /*  3 */
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f, /*  4 */
    -1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    1.0f, /*  5 */

    /* Back face (-Z) */
    1.0f,
    -1.0f,
    -1.0f,
    0.0f,
    0.0f,
    -1.0f,
    0.0f,
    0.0f, /*  6 */
    -1.0f,
    -1.0f,
    -1.0f,
    0.0f,
    0.0f,
    -1.0f,
    1.0f,
    0.0f, /*  7 */
    -1.0f,
    1.0f,
    -1.0f,
    0.0f,
    0.0f,
    -1.0f,
    1.0f,
    1.0f, /*  8 */
    1.0f,
    -1.0f,
    -1.0f,
    0.0f,
    0.0f,
    -1.0f,
    0.0f,
    0.0f, /*  9 */
    -1.0f,
    1.0f,
    -1.0f,
    0.0f,
    0.0f,
    -1.0f,
    1.0f,
    1.0f, /* 10 */
    1.0f,
    1.0f,
    -1.0f,
    0.0f,
    0.0f,
    -1.0f,
    0.0f,
    1.0f, /* 11 */

    /* Right face (+X) */
    1.0f,
    -1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f, /* 12 */
    1.0f,
    -1.0f,
    -1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f, /* 13 */
    1.0f,
    1.0f,
    -1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f, /* 14 */
    1.0f,
    -1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f, /* 15 */
    1.0f,
    1.0f,
    -1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f, /* 16 */
    1.0f,
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f, /* 17 */

    /* Left face (-X) */
    -1.0f,
    -1.0f,
    -1.0f,
    -1.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f, /* 18 */
    -1.0f,
    -1.0f,
    1.0f,
    -1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f, /* 19 */
    -1.0f,
    1.0f,
    1.0f,
    -1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f, /* 20 */
    -1.0f,
    -1.0f,
    -1.0f,
    -1.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f, /* 21 */
    -1.0f,
    1.0f,
    1.0f,
    -1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f, /* 22 */
    -1.0f,
    1.0f,
    -1.0f,
    -1.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f, /* 23 */

    /* Top face (+Y) */
    -1.0f,
    1.0f,
    1.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f, /* 24 */
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    1.0f,
    0.0f,
    1.0f,
    0.0f, /* 25 */
    1.0f,
    1.0f,
    -1.0f,
    0.0f,
    1.0f,
    0.0f,
    1.0f,
    1.0f, /* 26 */
    -1.0f,
    1.0f,
    1.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f, /* 27 */
    1.0f,
    1.0f,
    -1.0f,
    0.0f,
    1.0f,
    0.0f,
    1.0f,
    1.0f, /* 28 */
    -1.0f,
    1.0f,
    -1.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f, /* 29 */

    /* Bottom face (-Y) */
    -1.0f,
    -1.0f,
    -1.0f,
    0.0f,
    -1.0f,
    0.0f,
    0.0f,
    0.0f, /* 30 */
    1.0f,
    -1.0f,
    -1.0f,
    0.0f,
    -1.0f,
    0.0f,
    1.0f,
    0.0f, /* 31 */
    1.0f,
    -1.0f,
    1.0f,
    0.0f,
    -1.0f,
    0.0f,
    1.0f,
    1.0f, /* 32 */
    -1.0f,
    -1.0f,
    -1.0f,
    0.0f,
    -1.0f,
    0.0f,
    0.0f,
    0.0f, /* 33 */
    1.0f,
    -1.0f,
    1.0f,
    0.0f,
    -1.0f,
    0.0f,
    1.0f,
    1.0f, /* 34 */
    -1.0f,
    -1.0f,
    1.0f,
    0.0f,
    -1.0f,
    0.0f,
    0.0f,
    1.0f, /* 35 */
  };

  state.vertex_count = 36; /* 6 faces * 6 vertices */

  state.vertex_buffer = create_buffer_with_data(
    wgpu_context, cube_vertices, sizeof(cube_vertices), WGPUBufferUsage_Vertex);
}

/**
 * @brief Initialize default textures and sampler
 */
static void init_default_textures(wgpu_context_t* wgpu_context)
{
  /* Create default white texture */
  state.default_white_texture
    = create_solid_color_texture(wgpu_context, 255, 255, 255, 255);

  /* Create default normal texture (pointing up in tangent space) */
  state.default_normal_texture
    = create_solid_color_texture(wgpu_context, 128, 128, 255, 255);

  /* Create default sampler */
  state.default_sampler = create_default_sampler(wgpu_context);

  /* Create placeholder shadow map (1x1 depth texture for now) */
  state.placeholder_shadow_map = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Placeholder shadow map"),
      .size = (WGPUExtent3D){
        .width = 1,
        .height = 1,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_Depth32Float,
      .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment,
      .mipLevelCount = 1,
      .sampleCount = 1,
      .dimension = WGPUTextureDimension_2D,
    }
  );

  /* Create default roughness/metallic texture */
  /* Format: R=unused, G=roughness(0.5), B=metallic(0.0), A=unused */
  /* This makes materials appear as dielectric with medium roughness */
  uint8_t roughness_metallic_data[4]     = {0, 128, 0, 255}; /* G=0.5, B=0.0 */
  WGPUTexture roughness_metallic_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Default roughness/metallic texture"),
      .size  = (WGPUExtent3D){.width = 1, .height = 1, .depthOrArrayLayers = 1},
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
    });
  wgpuQueueWriteTexture(
    wgpu_context->queue,
    &(WGPUTexelCopyTextureInfo){
      .texture  = roughness_metallic_texture,
      .mipLevel = 0,
      .origin   = (WGPUOrigin3D){0, 0, 0},
      .aspect   = WGPUTextureAspect_All,
    },
    roughness_metallic_data, sizeof(roughness_metallic_data),
    &(WGPUTexelCopyBufferLayout){
      .offset       = 0,
      .bytesPerRow  = 4,
      .rowsPerImage = 1,
    },
    &(WGPUExtent3D){.width = 1, .height = 1, .depthOrArrayLayers = 1});

  /* Store the default roughness/metallic texture in state */
  state.default_roughness_metallic_texture = roughness_metallic_texture;

  /* Create BRDF LUT sampler */
  state.brdf_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("BRDF LUT sampler"),
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .maxAnisotropy = 1,
                          });

  /* Create shadow comparison sampler */
  state.shadow_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Shadow sampler"),
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .compare       = WGPUCompareFunction_LessEqual,
                            .maxAnisotropy = 1,
                          });
}

/**
 * @brief Initialize bind group layouts for PBR rendering
 */
static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Scene bind group layout (camera and lighting uniforms) */
  state.scene_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label = STRVIEW("scene bind group layout"),
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry){
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(float) * (16 + 16 + 4 + 4 + 4 + 16),
        },
      },
    }
  );

  /* Instance bind group layout (for instanced rendering) */
  state.instance_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label = STRVIEW("instance bind group layout"),
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry){
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_ReadOnlyStorage,
        },
      },
    }
  );

  /* Material bind group layout (PBR textures per material) */
  state.material_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label = STRVIEW("material bind group layout"),
      .entryCount = 11,
      .entries = (WGPUBindGroupLayoutEntry[]){
        /* Material uniforms */
        {
          .binding = 0,
          .visibility = WGPUShaderStage_Fragment,
          .buffer = (WGPUBufferBindingLayout){
            .type = WGPUBufferBindingType_Uniform,
          },
        },
        /* Albedo sampler + texture */
        {
          .binding = 1,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = (WGPUSamplerBindingLayout){
            .type = WGPUSamplerBindingType_Filtering,
          },
        },
        {
          .binding = 2,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
        /* Normal sampler + texture */
        {
          .binding = 3,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = (WGPUSamplerBindingLayout){
            .type = WGPUSamplerBindingType_Filtering,
          },
        },
        {
          .binding = 4,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
        /* Roughness/Metallic sampler + texture */
        {
          .binding = 5,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = (WGPUSamplerBindingLayout){
            .type = WGPUSamplerBindingType_Filtering,
          },
        },
        {
          .binding = 6,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
        /* AO sampler + texture */
        {
          .binding = 7,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = (WGPUSamplerBindingLayout){
            .type = WGPUSamplerBindingType_Filtering,
          },
        },
        {
          .binding = 8,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
        /* Emissive sampler + texture */
        {
          .binding = 9,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = (WGPUSamplerBindingLayout){
            .type = WGPUSamplerBindingType_Filtering,
          },
        },
        {
          .binding = 10,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
      },
    }
  );

  /* PBR bind group layout (IBL textures) */
  state.pbr_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label = STRVIEW("PBR bind group layout"),
      .entryCount = 7,
      .entries = (WGPUBindGroupLayoutEntry[]){
        /* Default sampler */
        {
          .binding = 0,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = (WGPUSamplerBindingLayout){
            .type = WGPUSamplerBindingType_Filtering,
          },
        },
        /* BRDF LUT sampler */
        {
          .binding = 1,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = (WGPUSamplerBindingLayout){
            .type = WGPUSamplerBindingType_Filtering,
          },
        },
        /* BRDF LUT texture */
        {
          .binding = 2,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
        /* Irradiance map cubemap */
        {
          .binding = 3,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_Cube,
          },
        },
        /* Prefilter map cubemap */
        {
          .binding = 4,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_Cube,
          },
        },
        /* Shadow map depth texture */
        {
          .binding = 5,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Depth,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
        /* Shadow sampler comparison */
        {
          .binding = 6,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = (WGPUSamplerBindingLayout){
            .type = WGPUSamplerBindingType_Comparison,
          },
        },
      },
    }
  );
}

/**
 * @brief Initialize scene uniform buffer and bind group
 */
static void init_scene_uniforms(wgpu_context_t* wgpu_context)
{
  /* Scene uniforms: projection, view, camera position, light direction, light
   * color, light matrix */
  state.scene_uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("scene uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(float) * (16 + 16 + 4 + 4 + 4 + 16),
    });

  /* Create scene bind group */
  state.scene_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label = STRVIEW("scene bind group"),
      .layout = state.scene_bind_group_layout,
      .entryCount = 1,
      .entries = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer = state.scene_uniform_buffer,
        .size = sizeof(float) * (16 + 16 + 4 + 4 + 4 + 16),
      },
    }
  );
}

/**
 * @brief Initialize PBR bind group with IBL textures
 */
static void init_pbr_bind_group(wgpu_context_t* wgpu_context)
{
  /* Wait until IBL textures are generated */
  if (!state.brdf_lut || !state.irradiance_map || !state.prefilter_map) {
    return;
  }

  /* Create texture views */
  WGPUTextureView brdf_view = wgpuTextureCreateView(
    state.brdf_lut, &(WGPUTextureViewDescriptor){
                      .label           = STRVIEW("BRDF LUT view"),
                      .format          = WGPUTextureFormat_RG16Float,
                      .dimension       = WGPUTextureViewDimension_2D,
                      .mipLevelCount   = 1,
                      .arrayLayerCount = 1,
                    });

  WGPUTextureView irradiance_view = wgpuTextureCreateView(
    state.irradiance_map, &(WGPUTextureViewDescriptor){
                            .label           = STRVIEW("irradiance map view"),
                            .format          = WGPUTextureFormat_RGBA8Unorm,
                            .dimension       = WGPUTextureViewDimension_Cube,
                            .mipLevelCount   = 1,
                            .arrayLayerCount = 6,
                          });

  WGPUTextureView prefilter_view = wgpuTextureCreateView(
    state.prefilter_map, &(WGPUTextureViewDescriptor){
                           .label           = STRVIEW("prefilter map view"),
                           .format          = WGPUTextureFormat_RGBA8Unorm,
                           .dimension       = WGPUTextureViewDimension_Cube,
                           .mipLevelCount   = state.settings.roughness_levels,
                           .arrayLayerCount = 6,
                         });

  /* Create PBR bind group */
  /* Create depth-only view for placeholder shadow map */
  WGPUTextureView shadow_map_view
    = wgpuTextureCreateView(state.placeholder_shadow_map,
                            &(WGPUTextureViewDescriptor){
                              .label  = STRVIEW("Placeholder shadow map view"),
                              .format = WGPUTextureFormat_Depth32Float,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .aspect          = WGPUTextureAspect_DepthOnly,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                            });

  state.pbr_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label = STRVIEW("PBR bind group"),
      .layout = state.pbr_bind_group_layout,
      .entryCount = 7,
      .entries = (WGPUBindGroupEntry[]){
        {
          .binding = 0,
          .sampler = state.default_sampler,
        },
        {
          .binding = 1,
          .sampler = state.brdf_sampler,
        },
        {
          .binding = 2,
          .textureView = brdf_view,
        },
        {
          .binding = 3,
          .textureView = irradiance_view,
        },
        {
          .binding = 4,
          .textureView = prefilter_view,
        },
        {
          .binding = 5,
          .textureView = shadow_map_view,
        },
        {
          .binding = 6,
          .sampler = state.shadow_sampler,
        },
      },
    }
  );

  /* Note: views are owned by bind group, don't release them separately */
}

/**
 * @brief Initialize instance bind group
 */
static void init_instance_bind_group(wgpu_context_t* wgpu_context)
{
  /* Wait until instance buffer is created */
  if (!state.instance_buffer) {
    return;
  }

  /* Create instance bind group */
  state.instance_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label = STRVIEW("instance bind group"),
      .layout = state.instance_bind_group_layout,
      .entryCount = 1,
      .entries = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer = state.instance_buffer,
        .size = state.total_instances * sizeof(mat4),
      },
    }
  );
}

/**
 * @brief Initialize camera uniform buffer (legacy, use init_scene_uniforms
 * instead)
 */
static void init_camera_uniforms(wgpu_context_t* wgpu_context)
{
  /* This function is kept for backward compatibility but scene_uniforms should
   * be used */
  UNUSED_VAR(wgpu_context);
  /* Camera uniforms are now part of scene_uniform_buffer */
  /* See init_scene_uniforms() */
}

/**
 * @brief Initialize render pipeline (TEMPORARY - Simple test renderer)
 * TODO: Replace with full PBR pipeline using create_pbr_shader()
 */
static void init_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* TEMPORARY: This is a simple test pipeline, not the full PBR renderer */

  /* Create temporary uniform buffer for testing */
  state.camera_uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("temp camera uniforms"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(float) * 32, /* 2 mat4s */
    });

  /* Create temporary bind group layout for testing */
  state.camera_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label = STRVIEW("temp camera bind group layout"),
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry){
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(float) * 32, /* 2 mat4s */
        },
      },
    }
  );

  /* Create simple vertex shader */
  const char* vertex_shader_wgsl = CODE(
    struct Uniforms {
      projection: mat4x4f,
      view: mat4x4f,
    };

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct VertexInput {
      @location(0) position: vec3f,
      @location(1) normal: vec3f,
      @location(2) texcoord: vec2f,
    };

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) normal: vec3f,
      @location(1) texcoord: vec2f,
    };

    @vertex
    fn main(input: VertexInput) -> VertexOutput {
      var output: VertexOutput;
      output.position = uniforms.projection * uniforms.view * vec4f(input.position, 1.0);
      output.normal = input.normal;
      output.texcoord = input.texcoord;
      return output;
    }
  );

  /* Create simple fragment shader - bright color for visibility */
  const char* fragment_shader_wgsl = CODE(
    @fragment fn main(@location(0) normal : vec3f,
                      @location(1) texcoord : vec2f)
      ->@location(0) vec4f {
        /* Bright colors for debugging visibility */
        let n     = normalize(normal);
        let light = max(dot(n, normalize(vec3f(1.0, 1.0, 1.0))), 0.2);
        /* Mix of red and lighting for clear visibility */
        return vec4f(vec3f(1.0, 0.2, 0.2) * light + vec3f(0.3, 0.0, 0.0), 1.0);
      });

  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  /* Create pipeline */
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts = &state.camera_bind_group_layout,
                          });

  state.render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("Render - Pipeline"),
      .layout = pipeline_layout,
      .vertex = (WGPUVertexState){
        .module = vert_shader_module,
        .entryPoint = STRVIEW("main"),
        .bufferCount = 1,
        .buffers = &(WGPUVertexBufferLayout){
          .arrayStride = 8 * sizeof(float),
          .stepMode = WGPUVertexStepMode_Vertex,
          .attributeCount = 3,
          .attributes = (WGPUVertexAttribute[]){
            {.shaderLocation = 0, .offset = 0, .format = WGPUVertexFormat_Float32x3},
            {.shaderLocation = 1, .offset = 3 * sizeof(float), .format = WGPUVertexFormat_Float32x3},
            {.shaderLocation = 2, .offset = 6 * sizeof(float), .format = WGPUVertexFormat_Float32x2},
          },
        },
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode = WGPUCullMode_None,  /* Disable culling for debugging */
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = frag_shader_module,
        .entryPoint = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    }
  );

  /* Create bind group */
  state.camera_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label = STRVIEW("Camera - Bind group"),
      .layout = state.camera_bind_group_layout,
      .entryCount = 1,
      .entries = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer = state.camera_uniform_buffer,
        .size = sizeof(mat4) * 2,
      },
    }
  );

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module)
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module)
}

/**
 * @brief Create a PBR render pipeline
 * @param wgpu_context WebGPU context
 * @param has_uvs Whether the mesh has UV coordinates
 * @param has_tangents Whether the mesh has tangent vectors
 * @param use_alpha_cutoff Whether to use alpha cutoff for transparency
 * @return Created render pipeline
 */
static WGPURenderPipeline create_pbr_pipeline(wgpu_context_t* wgpu_context,
                                              bool has_uvs, bool has_tangents,
                                              bool use_alpha_cutoff)
{
  /* Generate PBR shader code */
  char* shader_code = create_pbr_shader(has_uvs, has_tangents, use_alpha_cutoff,
                                        state.settings.shadow_map_size);

  /* Create shader module */
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, shader_code);

  if (!shader_module) {
    return NULL;
  }

  /* Create pipeline layout with all bind group layouts */
  WGPUBindGroupLayout layouts[4] = {
    state.scene_bind_group_layout,
    state.instance_bind_group_layout,
    state.material_bind_group_layout,
    state.pbr_bind_group_layout,
  };

  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("PBR - Pipeline layout"),
                            .bindGroupLayoutCount = 4,
                            .bindGroupLayouts     = layouts,
                          });

  /* Define vertex attributes based on what the mesh has */
  WGPUVertexAttribute attributes[4];
  uint32_t attr_count = 0;
  uint64_t offset     = 0;

  /* Position (always present) */
  attributes[attr_count++] = (WGPUVertexAttribute){
    .shaderLocation = SHADER_LOCATION_POSITION,
    .offset         = offset,
    .format         = WGPUVertexFormat_Float32x3,
  };
  offset += 3 * sizeof(float);

  /* Normal (always present) */
  attributes[attr_count++] = (WGPUVertexAttribute){
    .shaderLocation = SHADER_LOCATION_NORMAL,
    .offset         = offset,
    .format         = WGPUVertexFormat_Float32x3,
  };
  offset += 3 * sizeof(float);

  /* UV coordinates (if present) */
  if (has_uvs) {
    attributes[attr_count++] = (WGPUVertexAttribute){
      .shaderLocation = SHADER_LOCATION_TEXCOORD_0,
      .offset         = offset,
      .format         = WGPUVertexFormat_Float32x2,
    };
    offset += 2 * sizeof(float);
  }

  /* Tangent (if present) */
  if (has_tangents) {
    attributes[attr_count++] = (WGPUVertexAttribute){
      .shaderLocation = SHADER_LOCATION_TANGENT,
      .offset         = offset,
      .format         = WGPUVertexFormat_Float32x4,
    };
    offset += 4 * sizeof(float);
  }

  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = offset,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = attr_count,
    .attributes     = attributes,
  };

  /* Create render pipeline */
  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("PBR - Render pipeline"),
      .layout = pipeline_layout,
      .vertex = (WGPUVertexState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("vertexMain"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .primitive = (WGPUPrimitiveState){
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = use_alpha_cutoff ? WGPUCullMode_None : WGPUCullMode_Back,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format            = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = state.settings.sample_count,
        .mask  = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
          .blend = use_alpha_cutoff ? NULL : &(WGPUBlendState){
            .color = (WGPUBlendComponent){
              .operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_SrcAlpha,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
            },
            .alpha = (WGPUBlendComponent){
              .operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
            },
          },
        },
      },
    });

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(ShaderModule, shader_module)

  return pipeline;
}

/**
 * @brief Create a material bind group
 * @param wgpu_context WebGPU context
 * @param material CGLTF material data
 * @return Created material structure
 */
static gltf_material_t create_material_bind_group(wgpu_context_t* wgpu_context,
                                                  cgltf_material* material)
{
  gltf_material_t mat = {0};

  /* Set base color factor */
  if (material && material->has_pbr_metallic_roughness) {
    memcpy(mat.base_color_factor,
           material->pbr_metallic_roughness.base_color_factor, sizeof(vec4));
  }
  else {
    mat.base_color_factor[0] = 1.0f;
    mat.base_color_factor[1] = 1.0f;
    mat.base_color_factor[2] = 1.0f;
    mat.base_color_factor[3] = 1.0f;
  }

  /* Set alpha cutoff */
  if (material && material->alpha_mode == cgltf_alpha_mode_mask) {
    mat.has_alpha_cutoff = true;
    mat.alpha_cutoff     = material->alpha_cutoff;
  }
  else {
    mat.has_alpha_cutoff = false;
    mat.alpha_cutoff     = 0.5f;
  }

  /* Create material uniform buffer */
  struct {
    vec4 base_color_factor;
    float alpha_cutoff;
    float padding[3];
  } material_uniforms;

  memcpy(material_uniforms.base_color_factor, mat.base_color_factor,
         sizeof(vec4));
  material_uniforms.alpha_cutoff = mat.alpha_cutoff;

  mat.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Material - Uniform buffer"),
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = sizeof(material_uniforms),
      .mappedAtCreation = false,
    });

  wgpuQueueWriteBuffer(wgpu_context->queue, mat.uniform_buffer, 0,
                       &material_uniforms, sizeof(material_uniforms));

  /* Get textures (using defaults if not present) */
  WGPUTextureView albedo_view = wgpuTextureCreateView(
    state.default_white_texture, &(WGPUTextureViewDescriptor){
                                   .label     = STRVIEW("default albedo view"),
                                   .format    = WGPUTextureFormat_RGBA8Unorm,
                                   .dimension = WGPUTextureViewDimension_2D,
                                   .arrayLayerCount = 1,
                                   .mipLevelCount   = 1,
                                 });

  WGPUTextureView normal_view = wgpuTextureCreateView(
    state.default_normal_texture, &(WGPUTextureViewDescriptor){
                                    .label     = STRVIEW("default normal view"),
                                    .format    = WGPUTextureFormat_RGBA8Unorm,
                                    .dimension = WGPUTextureViewDimension_2D,
                                    .arrayLayerCount = 1,
                                    .mipLevelCount   = 1,
                                  });

  WGPUTextureView roughness_metallic_view = wgpuTextureCreateView(
    state.default_roughness_metallic_texture,
    &(WGPUTextureViewDescriptor){
      .label           = STRVIEW("default roughness metallic view"),
      .format          = WGPUTextureFormat_RGBA8Unorm,
      .dimension       = WGPUTextureViewDimension_2D,
      .arrayLayerCount = 1,
      .mipLevelCount   = 1,
    });

  /* Create material bind group */
  mat.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("material bind group"),
      .layout     = state.material_bind_group_layout,
      .entryCount = 11,
      .entries = (WGPUBindGroupEntry[]){
        /* Material uniforms */
        {.binding = 0, .buffer = mat.uniform_buffer, .size = sizeof(material_uniforms)},
        /* Albedo sampler + texture */
        {.binding = 1, .sampler = state.default_sampler},
        {.binding = 2, .textureView = albedo_view},
        /* Normal sampler + texture */
        {.binding = 3, .sampler = state.default_sampler},
        {.binding = 4, .textureView = normal_view},
        /* Roughness/Metallic sampler + texture */
        {.binding = 5, .sampler = state.default_sampler},
        {.binding = 6, .textureView = roughness_metallic_view},
        /* AO sampler + texture */
        {.binding = 7, .sampler = state.default_sampler},
        {.binding = 8, .textureView = albedo_view},
        /* Emissive sampler + texture */
        {.binding = 9, .sampler = state.default_sampler},
        {.binding = 10, .textureView = albedo_view},
      },
    });

  /* Note: views are owned by bind group */

  return mat;
}

/* -------------------------------------------------------------------------- *
 * Initialization Functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Initialize IBL textures (cubemap, irradiance, prefilter, BRDF LUT)
 */
static void init_ibl_textures(wgpu_context_t* wgpu_context)
{
  /* Initialize cubemap view matrices (needed for IBL texture generation) */
  init_cubemap_view_matrices();

  /* Initialize camera */
  /* Initialize camera (pitch, yaw, distance) */
  /* Pitch 0.3 radians (~17°) for slight top-down view */
  /* Yaw PI/4 (~45°) for angled view */
  camera_init(&state.camera, 0.3f, GLM_PI_4f, 8.0f);

  /* Generate BRDF LUT first (doesn't depend on other textures) */
  state.brdf_lut
    = generate_brdf_lut(wgpu_context, state.settings.brdf_lut_size);

  /* Note: HDR loading is asynchronous via load_hdr_environment()
   * The cubemap, irradiance, and prefilter maps are generated in
   * process_hdr_data() when the HDR file finishes loading.
   * See: load_hdr_environment() -> hdr_fetch_callback() -> process_hdr_data()
   */
}

/**
 * @brief Initialize skybox rendering pipeline
 */
static void init_skybox(wgpu_context_t* wgpu_context)
{
  if (!state.cubemap_texture) {
    return; /* Wait until cubemap is loaded */
  }

  /* Create skybox vertex buffer (cube vertices) */
  static const float cube_vertices[]
    = {/* positions */
       -1.0f, 1.0f,  -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
       1.0f,  -1.0f, -1.0f, 1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,
       1.0f,  1.0f,  -1.0f, 1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,

       -1.0f, -1.0f, 1.0f,  1.0f, -1.0f, -1.0f, -1.0f, 1.0f,
       -1.0f, 1.0f,  -1.0f, 1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,
       -1.0f, 1.0f,  1.0f,  1.0f, -1.0f, -1.0f, 1.0f,  1.0f,

       1.0f,  -1.0f, -1.0f, 1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,
       1.0f,  1.0f,  1.0f,  1.0f, 1.0f,  1.0f,  1.0f,  1.0f,
       1.0f,  1.0f,  -1.0f, 1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,

       -1.0f, -1.0f, 1.0f,  1.0f, -1.0f, 1.0f,  1.0f,  1.0f,
       1.0f,  1.0f,  1.0f,  1.0f, 1.0f,  1.0f,  1.0f,  1.0f,
       1.0f,  -1.0f, 1.0f,  1.0f, -1.0f, -1.0f, 1.0f,  1.0f,

       -1.0f, 1.0f,  -1.0f, 1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,
       1.0f,  1.0f,  1.0f,  1.0f, 1.0f,  1.0f,  1.0f,  1.0f,
       -1.0f, 1.0f,  1.0f,  1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,

       -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,  1.0f,
       1.0f,  -1.0f, -1.0f, 1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,
       -1.0f, -1.0f, 1.0f,  1.0f, 1.0f,  -1.0f, 1.0f,  1.0f};

  state.skybox_vertex_buffer = create_buffer_with_data(
    wgpu_context, cube_vertices, sizeof(cube_vertices), WGPUBufferUsage_Vertex);

  /* Create skybox uniform buffer (view + projection matrices) */
  state.skybox_uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("skybox uniform buffer"),
      .size  = 64 * sizeof(float), /* 2 * mat4 */
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    });

  /* Create skybox bind group layout */
  state.skybox_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label = STRVIEW("skybox bind group layout"),
      .entryCount = 3,
      .entries = (WGPUBindGroupLayoutEntry[]){
        {
          .binding = 0,
          .visibility = WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout){
            .sampleType = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_Cube,
          },
        },
        {
          .binding = 1,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = (WGPUSamplerBindingLayout){
            .type = WGPUSamplerBindingType_Filtering,
          },
        },
        {
          .binding = 2,
          .visibility = WGPUShaderStage_Vertex,
          .buffer = (WGPUBufferBindingLayout){
            .type = WGPUBufferBindingType_Uniform,
          },
        },
      },
    }
  );

  /* Create cubemap view for skybox - use full resolution cubemap, not
   * irradiance */
  printf("Creating skybox with cubemap texture: %p (512x512 RGBA8Unorm)\n",
         (void*)state.cubemap_texture);
  WGPUTextureView cubemap_view = wgpuTextureCreateView(
    state.cubemap_texture, &(WGPUTextureViewDescriptor){
                             .dimension       = WGPUTextureViewDimension_Cube,
                             .baseMipLevel    = 0,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = 0,
                             .arrayLayerCount = 6,
                           });

  /* Create skybox bind group */
  state.skybox_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label = STRVIEW("skybox bind group"),
      .layout = state.skybox_bind_group_layout,
      .entryCount = 3,
      .entries = (WGPUBindGroupEntry[]){
        {
          .binding = 0,
          .textureView = cubemap_view,
        },
        {
          .binding = 1,
          .sampler = state.default_sampler,
        },
        {
          .binding = 2,
          .buffer = state.skybox_uniform_buffer,
          .size = 64 * sizeof(float),
        },
      },
    }
  );

  /* Create skybox shader */
  const char* skybox_shader_wgsl = CODE(
    struct Uniforms {
      view: mat4x4f,
      projection: mat4x4f,
    }

    @group(0) @binding(0) var skyboxTexture: texture_cube<f32>;
    @group(0) @binding(1) var skyboxSampler: sampler;
    @group(0) @binding(2) var<uniform> uniforms: Uniforms;

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) texCoord: vec3f,
    }

    @vertex
    fn vertexMain(@location(0) position: vec4f) -> VertexOutput {
      var output: VertexOutput;
      var copy = uniforms.view;
      /* Reset translation to keep skybox centered */
      copy[3][0] = 0.0;
      copy[3][1] = 0.0;
      copy[3][2] = 0.0;
      output.position = (uniforms.projection * copy * position).xyww;
      /* Use position directly as cubemap direction */
      output.texCoord = position.xyz;
      return output;
    }

    /* ACES tone mapping */
    fn toneMapping(color: vec3f) -> vec3f {
      let a = 2.51;
      let b = 0.03;
      let c = 2.43;
      let d = 0.59;
      let e = 0.14;
      return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
    }

    @fragment
    fn fragmentMain(@location(0) texCoord: vec3f) -> @location(0) vec4f {
      /* Sample cubemap directly with texCoord from vertex shader */
      var color = textureSample(skyboxTexture, skyboxSampler, texCoord).rgb;
      
      color = toneMapping(color);
      color = pow(color, vec3f(1.0 / 2.2));
      return vec4f(color, 1);
    }
  );

  WGPUShaderModule skybox_shader
    = wgpu_create_shader_module(wgpu_context->device, skybox_shader_wgsl);

  /* Create skybox pipeline */
  state.skybox_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("skybox pipeline"),
      .layout = wgpuDeviceCreatePipelineLayout(
        wgpu_context->device,
        &(WGPUPipelineLayoutDescriptor){
          .bindGroupLayoutCount = 1,
          .bindGroupLayouts = &state.skybox_bind_group_layout,
        }
      ),
      .vertex = (WGPUVertexState){
        .module = skybox_shader,
        .entryPoint = STRVIEW("vertexMain"),
        .bufferCount = 1,
        .buffers = &(WGPUVertexBufferLayout){
          .arrayStride = 4 * sizeof(float),
          .attributeCount = 1,
          .attributes = &(WGPUVertexAttribute){
            .shaderLocation = 0,
            .offset = 0,
            .format = WGPUVertexFormat_Float32x4,
          },
        },
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_None,  /* No culling like TypeScript reference */
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = true,  /* Enable depth write like TypeScript reference */
        .depthCompare = WGPUCompareFunction_LessEqual,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = skybox_shader,
        .entryPoint = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    }
  );

  WGPU_RELEASE_RESOURCE(ShaderModule, skybox_shader)
}

/**
 * @brief Initialize the example
 */
static int init(wgpu_context_t* wgpu_context)
{
  if (state.initialized) {
    return EXIT_SUCCESS;
  }

  /* Initialize sokol fetch for async file loading */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 8,
    .num_channels = 2,
    .num_lanes    = 4,
  });

  /* Initialize mipmap generator */
  init_mipmap_generator(wgpu_context);

  /* Initialize IBL textures */
  init_ibl_textures(wgpu_context);

  /* Initialize default textures */
  init_default_textures(wgpu_context);

  /* Initialize bind group layouts */
  init_bind_group_layouts(wgpu_context);

  /* Initialize scene uniforms */
  init_scene_uniforms(wgpu_context);

  /* Initialize cube geometry */
  init_cube_geometry(wgpu_context);

  /* Initialize render pipeline */
  init_render_pipeline(wgpu_context);

  /* Load GLTF model synchronously */
  const char* gltf_path
    = "../../src/examples/pbr-webgpu/public/assets/helmet-flipped.glb";
  load_file_sync(gltf_path, &state.gltf_buffer, &state.gltf_buffer_size);

  /* Load HDR environment synchronously */
  const char* hdr_path
    = "../../src/examples/pbr-webgpu/public/assets/venice_sunset_1k.hdr";
  if (load_file_sync(hdr_path, &state.hdr_buffer, &state.hdr_buffer_size)) {
    printf("HDR file loaded, will process in first frame\n");
  }

  state.initialized = true;

  return EXIT_SUCCESS;
}

/**
 * @brief Update uniform buffers (camera matrices, lighting, etc.)
 */
static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update camera view matrix */
  mat4 view_matrix;
  camera_get_view(&state.camera, view_matrix);

  /* Update projection matrix based on window size */
  mat4 projection;
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  glm_perspective(GLM_PI_4f, aspect, 0.1f, 100.0f, projection);

  /* Get camera position */
  vec3 camera_position;
  camera_get_position(&state.camera, camera_position);

  /* Scene uniforms structure:
   * mat4 projection (16 floats)
   * mat4 view (16 floats)
   * vec3 cameraPosition + padding (4 floats)
   * vec3 lightDirection + padding (4 floats)
   * vec3 lightColor + padding (4 floats)
   * mat4 lightMatrix (16 floats)
   * Total: 60 floats
   */
  float scene_uniforms[60];

  /* Copy projection matrix */
  memcpy(&scene_uniforms[0], projection, sizeof(mat4));

  /* Copy view matrix */
  memcpy(&scene_uniforms[16], view_matrix, sizeof(mat4));

  /* Camera position */
  scene_uniforms[32] = camera_position[0];
  scene_uniforms[33] = camera_position[1];
  scene_uniforms[34] = camera_position[2];
  scene_uniforms[35] = 0.0f; /* padding */

  /* Light direction (normalized) */
  vec3 light_dir = {1.0f, 1.0f, 1.0f};
  glm_normalize(light_dir);
  scene_uniforms[36] = light_dir[0];
  scene_uniforms[37] = light_dir[1];
  scene_uniforms[38] = light_dir[2];
  scene_uniforms[39] = 0.0f; /* padding */

  /* Light color */
  scene_uniforms[40] = 1.0f; /* R */
  scene_uniforms[41] = 1.0f; /* G */
  scene_uniforms[42] = 1.0f; /* B */
  scene_uniforms[43] = 1.0f; /* padding/intensity */

  /* Light matrix (identity for now, used for shadow mapping) */
  mat4 light_matrix = GLM_MAT4_IDENTITY_INIT;
  memcpy(&scene_uniforms[44], light_matrix, sizeof(mat4));

  /* Write to scene uniform buffer */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.scene_uniform_buffer, 0,
                       scene_uniforms, sizeof(scene_uniforms));

  /* TEMPORARY: Also write to camera uniform buffer for test renderer */
  if (state.camera_uniform_buffer) {
    float camera_uniforms[32]; /* 2 mat4s */
    memcpy(camera_uniforms, projection, sizeof(mat4));
    memcpy(camera_uniforms + 16, view_matrix, sizeof(mat4));
    wgpuQueueWriteBuffer(wgpu_context->queue, state.camera_uniform_buffer, 0,
                         camera_uniforms, sizeof(camera_uniforms));
  }
}

/**
 * @brief Frame rendering function
 */
static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  static int frame_count = 0;

  /* Process async file loading */
  sfetch_dowork();

  /* Process loaded GLTF data */
  process_gltf_data(wgpu_context);

  /* Process loaded HDR data */
  process_hdr_data(wgpu_context);

  /* Initialize instance bind group after GLTF is loaded */
  if (state.instance_buffer && !state.instance_bind_group) {
    init_instance_bind_group(wgpu_context);
  }

  /* Update uniform buffers */
  update_uniform_buffers(wgpu_context);

  /* Update render pass attachments */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  /* Render scene */
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(encoder, &state.render_pass_descriptor);

  /* Re-enable model rendering to test if IBL textures work */
  /* Render GLTF models if loaded AND IBL textures are ready */
  if (state.meshes && state.scene_bind_group && state.instance_bind_group
      && state.pbr_bind_group && state.cubemap_texture && state.irradiance_map
      && state.prefilter_map && state.brdf_lut) {
    for (uint32_t mesh_idx = 0; mesh_idx < state.mesh_count; mesh_idx++) {
      gltf_mesh_t* mesh = &state.meshes[mesh_idx];
      for (uint32_t prim_idx = 0; prim_idx < mesh->primitive_count;
           prim_idx++) {
        gltf_primitive_t* prim = &mesh->primitives[prim_idx];

        if (!prim->pipeline || !prim->vertex_buffer || !prim->index_buffer) {
          continue;
        }

        /* Set pipeline and bind groups */
        wgpuRenderPassEncoderSetPipeline(pass, prim->pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, state.scene_bind_group, 0,
                                          NULL);
        wgpuRenderPassEncoderSetBindGroup(pass, 1, state.instance_bind_group, 0,
                                          NULL);
        if (prim->material && prim->material->bind_group) {
          wgpuRenderPassEncoderSetBindGroup(pass, 2, prim->material->bind_group,
                                            0, NULL);
        }
        wgpuRenderPassEncoderSetBindGroup(pass, 3, state.pbr_bind_group, 0,
                                          NULL);

        /* Set vertex and index buffers */
        wgpuRenderPassEncoderSetVertexBuffer(pass, 0, prim->vertex_buffer, 0,
                                             WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetIndexBuffer(
          pass, prim->index_buffer, prim->index_format, 0, WGPU_WHOLE_SIZE);

        /* Draw indexed with instances */
        wgpuRenderPassEncoderDrawIndexed(pass, prim->index_count,
                                         prim->instance_count, 0, 0,
                                         prim->instance_offset);
      }
    }
  }
  /* Fallback: render test cube if GLTF not loaded yet */
  else if (state.render_pipeline && state.vertex_buffer) {
    wgpuRenderPassEncoderSetPipeline(pass, state.render_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.camera_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(pass, state.vertex_count, 1, 0, 0);
  }

  /* Render skybox if available */
  if (state.skybox_pipeline && state.skybox_bind_group) {
    static int skybox_render_count = 0;
    if (skybox_render_count == 0) {
      printf(
        "First skybox render: pipeline=%p, bind_group=%p, vertex_buffer=%p\n",
        (void*)state.skybox_pipeline, (void*)state.skybox_bind_group,
        (void*)state.skybox_vertex_buffer);
    }
    skybox_render_count++;

    /* Update skybox uniforms */
    mat4 view_matrix;
    camera_get_view(&state.camera, view_matrix);

    mat4 projection;
    float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
    glm_perspective(GLM_PI_4f, aspect, 0.1f, 100.0f, projection);

    float skybox_uniforms[32]; /* 2 mat4s */
    memcpy(&skybox_uniforms[0], view_matrix, sizeof(mat4));
    memcpy(&skybox_uniforms[16], projection, sizeof(mat4));
    wgpuQueueWriteBuffer(wgpu_context->queue, state.skybox_uniform_buffer, 0,
                         skybox_uniforms, sizeof(skybox_uniforms));

    wgpuRenderPassEncoderSetPipeline(pass, state.skybox_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.skybox_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.skybox_vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(pass, 36, 1, 0, 0);
  }
  else if (frame_count < 5) {
    printf("Frame %d: Skybox not rendering - pipeline=%p, bind_group=%p\n",
           frame_count, (void*)state.skybox_pipeline,
           (void*)state.skybox_bind_group);
  }

  frame_count++;
  (void)frame_count; /* Track frame count for future use */

  wgpuRenderPassEncoderEnd(pass);
  WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command);

  WGPU_RELEASE_RESOURCE(CommandBuffer, command)
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)

  return EXIT_SUCCESS;
}

/**
 * @brief Input event callback for camera control
 */
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* event)
{
  UNUSED_VAR(wgpu_context);

  switch (event->type) {
    case INPUT_EVENT_TYPE_MOUSE_SCROLL:
      camera_handle_mouse_wheel(&state.camera, event->scroll_y);
      break;
    case INPUT_EVENT_TYPE_MOUSE_DOWN:
      if (event->mouse_button == BUTTON_LEFT) {
        camera_handle_mouse_down(&state.camera, event->mouse_x, event->mouse_y);
      }
      break;
    case INPUT_EVENT_TYPE_MOUSE_UP:
      if (event->mouse_button == BUTTON_LEFT) {
        camera_handle_mouse_up(&state.camera);
      }
      break;
    case INPUT_EVENT_TYPE_MOUSE_MOVE:
      camera_handle_mouse_move(&state.camera, event->mouse_x, event->mouse_y);
      break;
    case INPUT_EVENT_TYPE_RESIZED:
      /* Window resize handled automatically by framework */
      break;
    default:
      break;
  }
}

/**
 * @brief Cleanup and release resources
 */
static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Shutdown sokol fetch */
  sfetch_shutdown();

  /* Free GLTF data */
  if (state.gltf_data) {
    cgltf_free(state.gltf_data);
    state.gltf_data = NULL;
  }
  if (state.gltf_buffer) {
    free(state.gltf_buffer);
    state.gltf_buffer = NULL;
  }

  /* Free HDR buffer */
  if (state.hdr_buffer) {
    free(state.hdr_buffer);
    state.hdr_buffer = NULL;
  }

  /* Free GLTF meshes and primitives */
  if (state.meshes) {
    for (uint32_t i = 0; i < state.mesh_count; i++) {
      if (state.meshes[i].primitives) {
        for (uint32_t j = 0; j < state.meshes[i].primitive_count; j++) {
          WGPU_RELEASE_RESOURCE(RenderPipeline,
                                state.meshes[i].primitives[j].pipeline)
          WGPU_RELEASE_RESOURCE(Buffer,
                                state.meshes[i].primitives[j].vertex_buffer)
          WGPU_RELEASE_RESOURCE(Buffer,
                                state.meshes[i].primitives[j].index_buffer)
        }
        free(state.meshes[i].primitives);
      }
    }
    free(state.meshes);
    state.meshes = NULL;
  }

  /* Free GLTF materials */
  if (state.materials) {
    for (uint32_t i = 0; i < state.material_count; i++) {
      WGPU_RELEASE_RESOURCE(BindGroup, state.materials[i].bind_group)
      WGPU_RELEASE_RESOURCE(Buffer, state.materials[i].uniform_buffer)
    }
    free(state.materials);
    state.materials = NULL;
  }

  /* Release instance buffer */
  WGPU_RELEASE_RESOURCE(Buffer, state.instance_buffer)

  /* Release GLTF textures */
  if (state.gltf_textures) {
    for (uint32_t i = 0; i < state.gltf_texture_count; i++) {
      WGPU_RELEASE_RESOURCE(Texture, state.gltf_textures[i])
    }
    free(state.gltf_textures);
    state.gltf_textures = NULL;
  }

  /* Release rendering resources */
  WGPU_RELEASE_RESOURCE(BindGroup, state.scene_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.instance_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.pbr_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.scene_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.scene_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.instance_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.material_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.pbr_bind_group_layout)
  WGPU_RELEASE_RESOURCE(Sampler, state.brdf_sampler)
  WGPU_RELEASE_RESOURCE(Sampler, state.shadow_sampler)

  /* Release temporary test renderer resources */
  WGPU_RELEASE_RESOURCE(BindGroup, state.camera_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.camera_uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.camera_bind_group_layout)

  /* Release default textures */
  WGPU_RELEASE_RESOURCE(Texture, state.default_white_texture)
  WGPU_RELEASE_RESOURCE(Texture, state.default_normal_texture)
  WGPU_RELEASE_RESOURCE(Texture, state.default_roughness_metallic_texture)
  WGPU_RELEASE_RESOURCE(Texture, state.placeholder_shadow_map)
  WGPU_RELEASE_RESOURCE(Sampler, state.default_sampler)

  /* Release skybox resources */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.skybox_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.skybox_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.skybox_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.skybox_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.skybox_vertex_buffer)

  /* Release textures */
  WGPU_RELEASE_RESOURCE(Texture, state.cubemap_texture)
  WGPU_RELEASE_RESOURCE(Texture, state.irradiance_map)
  WGPU_RELEASE_RESOURCE(Texture, state.prefilter_map)
  WGPU_RELEASE_RESOURCE(Texture, state.brdf_lut)

  /* Cleanup mipmap generator */
  cleanup_mipmap_generator();
}

/**
 * @brief Main entry point
 */
int main(void)
{
  /* Ensure stdout is not buffered so printf appears immediately */
  setvbuf(stdout, NULL, _IONBF, 0);

  printf("=== GLTF PBR IBL Example Starting ===\n");
  fflush(stdout);

  wgpu_start(&(wgpu_desc_t){
    .title          = "GLTF PBR with IBL",
    .init_cb        = init,
    .frame_cb       = frame,
    .input_event_cb = input_event_cb,
    .shutdown_cb    = shutdown,
  });

  return EXIT_SUCCESS;
}
