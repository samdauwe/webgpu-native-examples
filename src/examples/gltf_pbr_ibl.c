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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - PBR with IBL (Physically Based Rendering with Image Based
 * Lighting)
 *
 * This example demonstrates Physically Based Rendering with Image Based
 * Lighting using WebGPU.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/pbr
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

/* Tone mapping functions */
static const char* tone_mapping_aces_wgsl;
static const char* tone_mapping_reinhard_wgsl;
static const char* tone_mapping_uncharted2_wgsl;
static const char* tone_mapping_lottes_wgsl;

/* Full PBR shader */
static const char* pbr_shader_wgsl;

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
static void camera_init(camera_t* camera, float pitch, float yaw,
                        float distance)
{
  memset(camera, 0, sizeof(camera_t));

  glm_vec3_zero(camera->target);
  camera->pitch            = pitch;
  camera->yaw              = yaw;
  camera->distance         = distance > 0.0f ? distance : 10.0f;
  camera->scroll_direction = 0;
  camera->last_x           = 0.0f;
  camera->last_y           = 0.0f;
  camera->is_dragging      = false;
}

/**
 * @brief Handle mouse wheel event for zooming
 */
__attribute__((unused)) static void camera_handle_mouse_wheel(camera_t* camera,
                                                              float delta)
{
  camera->scroll_direction = (delta > 0.0f) ? 1 : ((delta < 0.0f) ? -1 : 0);

  const float zoom_speed = 0.5f;
  camera->distance -= camera->scroll_direction * zoom_speed;

  const float min_distance = 1.0f;
  camera->distance         = fmaxf(camera->distance, min_distance);
}

/**
 * @brief Handle mouse button down event
 */
__attribute__((unused)) static void camera_handle_mouse_down(camera_t* camera,
                                                             float x, float y)
{
  camera->is_dragging = true;
  camera->last_x      = x;
  camera->last_y      = y;
}

/**
 * @brief Handle mouse move event
 */
__attribute__((unused)) static void camera_handle_mouse_move(camera_t* camera,
                                                             float x, float y)
{
  if (!camera->is_dragging) {
    return;
  }

  const float dx = x - camera->last_x;
  const float dy = y - camera->last_y;

  camera->last_x = x;
  camera->last_y = y;

  camera->pitch -= dy * 0.003f;
  camera->yaw -= dx * 0.003f;
}

/**
 * @brief Handle mouse button up event
 */
__attribute__((unused)) static void camera_handle_mouse_up(camera_t* camera)
{
  camera->is_dragging = false;
}

/**
 * @brief Get camera position in world space
 */
static void camera_get_position(camera_t* camera, vec3 position)
{
  position[0] = cosf(camera->pitch) * cosf(camera->yaw);
  position[1] = sinf(camera->pitch);
  position[2] = cosf(camera->pitch) * sinf(camera->yaw);

  glm_vec3_scale(position, camera->distance, position);
  glm_vec3_add(position, camera->target, position);
}

/**
 * @brief Get camera view matrix
 */
static void camera_get_view(camera_t* camera, mat4 view)
{
  vec3 position, up = {0.0f, 1.0f, 0.0f};
  camera_get_position(camera, position);
  glm_lookat(position, camera->target, up, view);
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
static const mat4 cubemap_view_matrices[6] = {
  /* +X */ GLM_MAT4_IDENTITY_INIT,
  /* -X */ GLM_MAT4_IDENTITY_INIT,
  /* +Y */ GLM_MAT4_IDENTITY_INIT,
  /* -Y */ GLM_MAT4_IDENTITY_INIT,
  /* +Z */ GLM_MAT4_IDENTITY_INIT,
  /* -Z */ GLM_MAT4_IDENTITY_INIT,
};

/* Inverted cubemap view matrices */
static const mat4 cubemap_view_matrices_inverted[6] = {
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
__attribute__((unused)) static void init_cubemap_view_matrices(void)
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
  glm_mat4_inv(temp, (mat4*)cubemap_view_matrices[0]);

  /* -X face */
  target[0] = -1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, (mat4*)cubemap_view_matrices[1]);

  /* +Y face */
  target[0] = 0.0f;
  target[1] = -1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = -1.0f;
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, (mat4*)cubemap_view_matrices[2]);

  /* -Y face */
  target[0] = 0.0f;
  target[1] = 1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = 1.0f;
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, (mat4*)cubemap_view_matrices[3]);

  /* +Z face */
  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = 1.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, (mat4*)cubemap_view_matrices[4]);

  /* -Z face */
  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = -1.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, temp);
  glm_mat4_inv(temp, (mat4*)cubemap_view_matrices[5]);

  /* Initialize inverted matrices */
  target[0] = 1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, (mat4*)cubemap_view_matrices_inverted[0]);

  target[0] = -1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, (mat4*)cubemap_view_matrices_inverted[1]);

  target[0] = 0.0f;
  target[1] = 1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = -1.0f;
  glm_lookat(center, target, up, (mat4*)cubemap_view_matrices_inverted[2]);

  target[0] = 0.0f;
  target[1] = -1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = 1.0f;
  glm_lookat(center, target, up, (mat4*)cubemap_view_matrices_inverted[3]);

  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = 1.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, (mat4*)cubemap_view_matrices_inverted[4]);

  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = -1.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, (mat4*)cubemap_view_matrices_inverted[5]);
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
                    + (max(vec3(1.0 - roughness), f0) - f0)
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

/* Tone mapping - Reinhard */
__attribute__((unused)) static const char* tone_mapping_reinhard_wgsl
  = CODE(fn toneMapping(color : vec3f)->vec3f {
      return color / (color + vec3f(1.0));
    });

/* Tone mapping - Uncharted 2 */
__attribute__((unused)) static const char* tone_mapping_uncharted2_wgsl = CODE(
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

/* Tone mapping - Lottes */
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
__attribute__((unused)) static uint32_t
get_component_type_size(gltf_component_type_t component_type)
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
__attribute__((unused)) static WGPUVertexFormat
get_gpu_vertex_format(gltf_component_type_t component_type,
                      gltf_accessor_type_t type, bool normalized)
{
  const uint32_t count = get_num_components_for_type(type);

  switch (component_type) {
    case GLTF_COMPONENT_TYPE_BYTE:
      if (normalized) {
        return (count == 2) ? WGPUVertexFormat_Snorm8x2 :
               (count == 4) ? WGPUVertexFormat_Snorm8x4 :
                              WGPUVertexFormat_Uint8x2;
      }
      return (count == 2) ? WGPUVertexFormat_Sint8x2 :
             (count == 4) ? WGPUVertexFormat_Sint8x4 :
                            WGPUVertexFormat_Uint8x2;

    case GLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      if (normalized) {
        return (count == 2) ? WGPUVertexFormat_Unorm8x2 :
               (count == 4) ? WGPUVertexFormat_Unorm8x4 :
                              WGPUVertexFormat_Uint8x2;
      }
      return (count == 2) ? WGPUVertexFormat_Uint8x2 :
             (count == 4) ? WGPUVertexFormat_Uint8x4 :
                            WGPUVertexFormat_Uint8x2;

    case GLTF_COMPONENT_TYPE_SHORT:
      if (normalized) {
        return (count == 2) ? WGPUVertexFormat_Snorm16x2 :
               (count == 4) ? WGPUVertexFormat_Snorm16x4 :
                              WGPUVertexFormat_Uint8x2;
      }
      return (count == 2) ? WGPUVertexFormat_Sint16x2 :
             (count == 4) ? WGPUVertexFormat_Sint16x4 :
                            WGPUVertexFormat_Uint8x2;

    case GLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      if (normalized) {
        return (count == 2) ? WGPUVertexFormat_Unorm16x2 :
               (count == 4) ? WGPUVertexFormat_Unorm16x4 :
                              WGPUVertexFormat_Undefined;
      }
      return (count == 2) ? WGPUVertexFormat_Uint16x2 :
             (count == 4) ? WGPUVertexFormat_Uint16x4 :
                            WGPUVertexFormat_Undefined;

    case GLTF_COMPONENT_TYPE_UNSIGNED_INT:
      return (count == 1) ? WGPUVertexFormat_Uint32 :
             (count == 2) ? WGPUVertexFormat_Uint32x2 :
             (count == 3) ? WGPUVertexFormat_Uint32x3 :
             (count == 4) ? WGPUVertexFormat_Uint32x4 :
                            WGPUVertexFormat_Undefined;

    case GLTF_COMPONENT_TYPE_FLOAT:
      return (count == 1) ? WGPUVertexFormat_Float32 :
             (count == 2) ? WGPUVertexFormat_Float32x2 :
             (count == 3) ? WGPUVertexFormat_Float32x3 :
             (count == 4) ? WGPUVertexFormat_Float32x4 :
                            WGPUVertexFormat_Undefined;

    default:
      return WGPUVertexFormat_Undefined;
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
    .label        = "Default sampler",
    .addressModeU = WGPUAddressMode_Repeat,
    .addressModeV = WGPUAddressMode_Repeat,
    .addressModeW = WGPUAddressMode_Repeat,
    .magFilter    = WGPUFilterMode_Linear,
    .minFilter    = WGPUFilterMode_Linear,
    .mipmapFilter = WGPUMipmapFilterMode_Linear,
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

  WGPUImageCopyTexture destination = {
    .texture  = texture,
    .mipLevel = 0,
    .origin   = (WGPUOrigin3D){0, 0, 0},
    .aspect   = WGPUTextureAspect_All,
  };

  WGPUTextureDataLayout layout = {
    .offset       = 0,
    .bytesPerRow  = 4,
    .rowsPerImage = 1,
  };

  wgpuQueueWriteTexture(wgpu_context->queue, &destination, data, 4, &layout,
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
    .format = WGPUTextureFormat_Depth24Plus,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_desc);
  WGPUTextureView depth_view = wgpuTextureCreateView(depth_texture, NULL);

  /* Fragment shader for irradiance convolution */
  const char* fragment_shader = CODE(
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
  WGPUShaderModule vertex_shader = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = STRVIEW("irradiance vertex shader"),
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain.sType = WGPUSType_ShaderSourceWGSL,
        .code = cubemap_vertex_shader_wgsl,
      },
    }
  );

  WGPUShaderModule frag_shader = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = STRVIEW("irradiance fragment shader"),
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain.sType = WGPUSType_ShaderSourceWGSL,
        .code = fragment_shader,
      },
    }
  );

  /* Create vertex buffer */
  WGPUBuffer vertex_buffer = create_buffer_with_data(
    wgpu_context, cube_vertex_array, sizeof(cube_vertex_array),
    WGPUBufferUsage_Vertex);

  /* Create sampler */
  WGPUSampler sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label     = STRVIEW("irradiance sampler"),
                            .magFilter = WGPUFilterMode_Linear,
                            .minFilter = WGPUFilterMode_Linear,
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
      .label = STRVIEW("irradiance map pipeline"),
      .layout = NULL,
      .vertex = (WGPUVertexState){
        .module = vertex_shader,
        .entryPoint = "main",
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
        .format = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = frag_shader,
        .entryPoint = "main",
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
                       .dimension = WGPUTextureViewDimension_Cube,
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

    WGPUCommandEncoder encoder
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      encoder,
      &(WGPURenderPassDescriptor){
        .label = STRVIEW("irradiance render pass"),
        .colorAttachmentCount = 1,
        .colorAttachments = &(WGPURenderPassColorAttachment){
          .view = face_view,
          .loadOp = WGPULoadOp_Clear,
          .storeOp = WGPUStoreOp_Store,
          .clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0},
        },
        .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
          .view = depth_view,
          .depthLoadOp = WGPULoadOp_Clear,
          .depthStoreOp = WGPUStoreOp_Store,
          .depthClearValue = 1.0f,
        },
      }
    );

    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(pass, 36, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);

    WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(wgpu_context->queue, 1, &command);

    WGPU_RELEASE_RESOURCE(CommandBuffer, command)
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
    WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)
    WGPU_RELEASE_RESOURCE(TextureView, face_view)
  }

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(TextureView, cubemap_view)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer)
  WGPU_RELEASE_RESOURCE(Sampler, sampler)
  WGPU_RELEASE_RESOURCE(Buffer, vertex_buffer)
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader)
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_shader)
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
    .format = WGPUTextureFormat_Depth24Plus,
    .mipLevelCount = levels,
    .sampleCount = 1,
  };

  WGPUTexture depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_desc);

  /* Vertex shader with roughness uniform */
  const char* vertex_shader = CODE(
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
  char fragment_shader[16384];
  snprintf(fragment_shader, sizeof(fragment_shader), CODE(
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
  ), distribution_ggx_code, radical_inverse_vdc_code, hammersley_code, importance_sample_ggx_code, size);

  /* Create shaders */
  WGPUShaderModule vert_shader = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = STRVIEW("prefilter vertex shader"),
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain.sType = WGPUSType_ShaderSourceWGSL,
        .code = vertex_shader,
      },
    }
  );

  WGPUShaderModule frag_shader = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = STRVIEW("prefilter fragment shader"),
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain.sType = WGPUSType_ShaderSourceWGSL,
        .code = fragment_shader,
      },
    }
  );

  /* Create vertex buffer */
  WGPUBuffer vertex_buffer = create_buffer_with_data(
    wgpu_context, cube_vertex_array, sizeof(cube_vertex_array),
    WGPUBufferUsage_Vertex);

  /* Create sampler */
  WGPUSampler sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label     = STRVIEW("prefilter sampler"),
                            .magFilter = WGPUFilterMode_Linear,
                            .minFilter = WGPUFilterMode_Linear,
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
      .label = STRVIEW("prefilter map pipeline"),
      .layout = NULL,
      .vertex = (WGPUVertexState){
        .module = vert_shader,
        .entryPoint = "main",
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
        .format = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = frag_shader,
        .entryPoint = "main",
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
                       .dimension = WGPUTextureViewDimension_Cube,
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
                                               .baseMipLevel  = mip,
                                               .mipLevelCount = 1,
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
            .loadOp = WGPULoadOp_Load,
            .storeOp = WGPUStoreOp_Store,
            .clearValue = (WGPUColor){0.3, 0.3, 0.3, 1.0},
          },
          .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
            .view = depth_view,
            .depthLoadOp = WGPULoadOp_Clear,
            .depthStoreOp = WGPUStoreOp_Store,
            .depthClearValue = 1.0f,
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
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader)
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader)
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
    hash &= hash; /* Convert to 32bit integer */
  }

  return hash;
}

/* -------------------------------------------------------------------------- *
 * HDR Image Loading using stb_image
 * -------------------------------------------------------------------------- */

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

/* HDR image data structure */
typedef struct {
  uint32_t width;
  uint32_t height;
  float exposure;
  float gamma;
  float* data; /* RGBA32F data */
} hdr_image_t;

/**
 * @brief Load HDR image from file
 */
static bool load_hdr_image(const char* filename, hdr_image_t* out_image)
{
  if (!filename || !out_image) {
    return false;
  }

  memset(out_image, 0, sizeof(hdr_image_t));

  int width, height, channels;
  float* data = stbi_loadf(filename, &width, &height, &channels, 4);

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
  WGPUImageCopyTexture dest = {
    .texture  = equirect_texture,
    .mipLevel = 0,
    .origin   = (WGPUOrigin3D){0, 0, 0},
    .aspect   = WGPUTextureAspect_All,
  };

  WGPUTextureDataLayout data_layout = {
    .offset       = 0,
    .bytesPerRow  = 8 * hdr->width, /* 4 channels * 2 bytes (float16) */
    .rowsPerImage = hdr->height,
  };

  /* Convert float32 to float16 for upload - simplified version */
  const size_t data_size = hdr->width * hdr->height * 4 * sizeof(float);
  wgpuQueueWriteTexture(wgpu_context->queue, &dest, hdr->data, data_size,
                        &data_layout, &equirect_desc.size);

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
    .format = WGPUTextureFormat_Depth24Plus,
    .mipLevelCount = 1,
    .sampleCount = 1,
  };

  WGPUTexture depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_desc);
  WGPUTextureView depth_view = wgpuTextureCreateView(depth_texture, NULL);

  /* Fragment shader for equirectangular sampling */
  const char* fragment_shader = CODE(
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
  WGPUShaderModule vertex_shader = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = STRVIEW("cubemap vertex shader"),
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){
          .sType = WGPUSType_ShaderSourceWGSL,
        },
        .code = cubemap_vertex_shader_wgsl,
      },
    }
  );

  WGPUShaderModule frag_shader = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = STRVIEW("equirect fragment shader"),
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){
          .sType = WGPUSType_ShaderSourceWGSL,
        },
        .code = fragment_shader,
      },
    }
  );

  /* Create vertex buffer */
  WGPUBuffer vertex_buffer = create_buffer_with_data(
    wgpu_context, cube_vertex_array, sizeof(cube_vertex_array),
    WGPUBufferUsage_Vertex);

  /* Create pipeline - continuing in next chunk to avoid length limit */
  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("renderToCubemap"),
      .layout = NULL,
      .vertex = (WGPUVertexState){
        .module = vertex_shader,
        .entryPoint = "main",
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
        .format = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = frag_shader,
        .entryPoint = "main",
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
                            .label        = STRVIEW("equirect sampler"),
                            .addressModeU = WGPUAddressMode_ClampToEdge,
                            .addressModeV = WGPUAddressMode_ClampToEdge,
                            .addressModeW = WGPUAddressMode_ClampToEdge,
                            .magFilter    = WGPUFilterMode_Linear,
                            .minFilter    = WGPUFilterMode_Linear,
                            .mipmapFilter = WGPUMipmapFilterMode_Linear,
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

  /* Render each cubemap face */
  for (uint32_t face = 0; face < 6; face++) {
    mat4 mvp;
    glm_mat4_mul(projection, cubemap_view_matrices[face], mvp);
    wgpuQueueWriteBuffer(wgpu_context->queue, uniform_buffer, 0, mvp,
                         sizeof(mat4));

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

    WGPUCommandEncoder encoder
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      encoder,
      &(WGPURenderPassDescriptor){
        .label = STRVIEW("cubemap render pass"),
        .colorAttachmentCount = 1,
        .colorAttachments = &(WGPURenderPassColorAttachment){
          .view = face_view,
          .loadOp = WGPULoadOp_Clear,
          .storeOp = WGPUStoreOp_Store,
          .clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0},
        },
        .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
          .view = depth_view,
          .depthLoadOp = WGPULoadOp_Clear,
          .depthStoreOp = WGPUStoreOp_Store,
          .depthClearValue = 1.0f,
        },
      }
    );

    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(pass, 36, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);

    WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(wgpu_context->queue, 1, &command);

    WGPU_RELEASE_RESOURCE(CommandBuffer, command)
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
    WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)
    WGPU_RELEASE_RESOURCE(TextureView, face_view)
    WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  }

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(TextureView, equirect_view)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer)
  WGPU_RELEASE_RESOURCE(Sampler, sampler)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader)
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_shader)
  WGPU_RELEASE_RESOURCE(Buffer, vertex_buffer)
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

  /* Build the shader template */
  const char* shader_template = CODE(
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

    struct VertexInput {
      @location(SHADER_LOCATION_POSITION) position: vec4f,
      @location(SHADER_LOCATION_NORMAL) normal: vec3f,
#if HAS_UVS
      @location(SHADER_LOCATION_TEXCOORD_0) uv: vec2f,
#endif
#if HAS_TANGENTS
      @location(SHADER_LOCATION_TANGENT) tangent: vec4f,
#endif
    }

    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) normal: vec3f,
      @location(1) uv: vec2f,
      @location(2) worldPosition: vec3f,
      @location(3) shadowPosition: vec3f,
#if HAS_TANGENTS
      @location(4) tangent: vec4f,
#endif
    };

    @vertex
    fn vertexMain(input: VertexInput, @builtin(instance_index) instance: u32) -> VertexOutput {
      let positionFromLight = scene.lightViewProjection * models[instance] * input.position;

      var output: VertexOutput;
      output.position = scene.cameraProjection * scene.cameraView * models[instance] * input.position;
      output.normal = normalize((models[instance] * vec4f(input.normal, 0.0)).xyz);
      output.worldPosition = (models[instance] * input.position).xyz;
      output.shadowPosition = vec3f(
        positionFromLight.xy * vec2f(0.5, -0.5) + vec2f(0.5),
        positionFromLight.z
      );

#if HAS_UVS
      output.uv = input.uv;
#else
      output.uv = vec2f(0);
#endif

#if HAS_TANGENTS
      output.tangent = models[instance] * input.tangent;
#endif

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

#if USE_ALPHA_CUTOFF
      if (baseColor.a < material.alphaCutoff) {
        discard;
      }
#endif

      let ao = textureSample(aoTexture, aoSampler, input.uv).r;
      let albedo = baseColor.rgb;

      let roughnessMetallic = textureSample(roughnessMetallicTexture, roughnessMetallicSampler, input.uv);
      let metallic = roughnessMetallic.b;
      let roughness = roughnessMetallic.g;
      let emissive = textureSample(emissiveTexture, emissiveSampler, input.uv).rgb;

      var normal = textureSample(normalTexture, normalSampler, input.uv).rgb;
      normal = normalize(normal * 2.0 - 1.0);

#if HAS_TANGENTS
      var n = normalize(input.normal);
      let t = normalize(input.tangent.xyz);
      let b = cross(n, t) * input.tangent.w;
      let tbn = mat3x3f(t, b, n);
      n = normalize(tbn * normal);
#else
      let n = normalize(input.normal);
#endif

      let v = normalize(scene.cameraPosition - input.worldPosition);
      let r = reflect(-v, n);
      let f0 = mix(vec3f(0.04), albedo, metallic);

      var lo = vec3f(0.0);
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

  /* Copy template to working buffer */
  snprintf(shader_source, sizeof(shader_source), "%s", shader_template);

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

  /* Replace conditional compilation flags */
  str_replace_all(shader_source, "HAS_UVS", has_uvs ? "1" : "0", temp_buffer,
                  sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "HAS_TANGENTS", has_tangents ? "1" : "0",
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "USE_ALPHA_CUTOFF",
                  use_alpha_cutoff ? "1" : "0", temp_buffer,
                  sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  /* Replace shadow map size */
  char size_str[32];
  snprintf(size_str, sizeof(size_str), "%u.0", shadow_map_size);
  str_replace_all(shader_source, "SHADOW_MAP_SIZE", size_str, temp_buffer,
                  sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  /* Replace PBR shader functions */
  str_replace_all(shader_source, "DISTRIBUTION_GGX", distribution_ggx_wgsl,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "GEOMETRY_SCHLICK_GGX",
                  geometry_schlick_ggx_wgsl, temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "GEOMETRY_SMITH", geometry_smith_wgsl,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "FRESNEL_SCHLICK", fresnel_schlick_wgsl,
                  temp_buffer, sizeof(temp_buffer));
  snprintf(shader_source, sizeof(shader_source), "%s", temp_buffer);

  str_replace_all(shader_source, "FRESNEL_SCHLICK_ROUGHNESS",
                  fresnel_schlick_roughness_wgsl, temp_buffer,
                  sizeof(temp_buffer));
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
    .label = "BRDF LUT",
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

  /* Create depth texture */
  WGPUTextureDescriptor depth_texture_desc = {
    .label = "BRDF LUT depth",
    .usage = WGPUTextureUsage_RenderAttachment,
    .dimension = WGPUTextureDimension_2D,
    .size = (WGPUExtent3D){
      .width  = size,
      .height = size,
      .depthOrArrayLayers = 1,
    },
    .format = WGPUTextureFormat_Depth24Plus,
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

  /* Create shaders */
  WGPUShaderModule vertex_shader = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = "BRDF LUT vertex shader",
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){
          .sType = WGPUSType_ShaderSourceWGSL,
        },
        .code = brdf_lut_vertex_shader_wgsl,
      },
    }
  );

  WGPUShaderModule fragment_shader = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = "BRDF LUT fragment shader",
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){
          .sType = WGPUSType_ShaderSourceWGSL,
        },
        .code = brdf_lut_fragment_shader_wgsl,
      },
    }
  );

  /* Create render pipeline */
  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = "BRDF convolution pipeline",
      .layout = NULL,
      .vertex = (WGPUVertexState){
        .module = vertex_shader,
        .entryPoint = "main",
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
        .format = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = fragment_shader,
        .entryPoint = "main",
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = WGPUTextureFormat_RG16Float,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    }
  );

  /* Render the BRDF LUT */
  WGPUTextureView texture_view = wgpuTextureCreateView(texture, NULL);

  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = "BRDF LUT command encoder",
                          });

  WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
    encoder,
    &(WGPURenderPassDescriptor){
      .label = "BRDF convolution",
      .colorAttachmentCount = 1,
      .colorAttachments = &(WGPURenderPassColorAttachment){
        .view = texture_view,
        .loadOp = WGPULoadOp_Clear,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0},
      },
      .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
        .view = depth_texture_view,
        .depthLoadOp = WGPULoadOp_Clear,
        .depthStoreOp = WGPUStoreOp_Store,
        .depthClearValue = 1.0f,
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
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_shader)
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_shader)
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

  WGPUImageCopyTexture destination = {
    .texture  = texture,
    .mipLevel = 0,
    .origin   = (WGPUOrigin3D){0, 0, 0},
    .aspect   = WGPUTextureAspect_All,
  };

  WGPUTextureDataLayout layout = {
    .offset       = 0,
    .bytesPerRow  = 4,
    .rowsPerImage = 1,
  };

  wgpuQueueWriteTexture(wgpu_context->queue, &destination, data, 4, &layout,
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
static struct {
  /* Camera */
  camera_t camera;

  /* GLTF data */
  cgltf_data* gltf_data;
  uint8_t* gltf_buffer;
  size_t gltf_buffer_size;

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
  WGPUBindGroupLayout camera_bind_group_layout;
  WGPUBindGroupLayout material_bind_group_layout;
  WGPUBindGroup camera_bind_group;
  WGPUBuffer camera_uniform_buffer;
  WGPUBuffer vertex_buffer;
  uint32_t vertex_count;

  /* Default textures */
  WGPUTexture default_white_texture;
  WGPUTexture default_normal_texture;
  WGPUSampler default_sampler;

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
    uint32_t roughness_levels;
uint32_t brdf_lut_size;
}
settings;

WGPUBool initialized;
}
state = {
  .settings = {
    .cubemap_size = 512,
    .irradiance_map_size = 32,
    .prefilter_map_size = 128,
    .roughness_levels = 5,
    .brdf_lut_size = 512,
    .sample_count = 4,
    .shadow_map_size = 4096,
  },
  .initialized = false,
};

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
                            .label     = STRVIEW("mip generator sampler"),
                            .minFilter = WGPUFilterMode_Linear,
                          });

  /* Shader module for mipmap generation */
  const char* mipmap_shader = CODE(
    struct VSOutput {
      @builtin(position) position : vec4f, @location(0) uv : vec2f,
    };

    @vertex fn vs(@builtin(vertex_index) vertexIndex : u32)
      ->VSOutput {
        var pos = array<vec2f, 6>(vec2f(0.0, 0.0), // center
                                  vec2f(1.0, 0.0), // right, center
                                  vec2f(0.0, 1.0), // center, top

                                  vec2f(0.0, 1.0), // center, top
                                  vec2f(1.0, 0.0), // right, center
                                  vec2f(1.0, 1.0), // right, top
        );

        var vsOutput : VSOutput;
        let xy            = pos[vertexIndex];
        vsOutput.position = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
        vsOutput.uv       = vec2f(xy.x, 1.0 - xy.y);
        return vsOutput;
      }

    @group(0) @binding(0) var ourSampler : sampler;
    @group(0) @binding(1) var ourTexture : texture_2d<f32>;

    @fragment fn fs(fsInput : VSOutput)
      ->@location(0)
        vec4f { return textureSample(ourTexture, ourSampler, fsInput.uv); });

  WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = STRVIEW("mip generator shader"),
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain.sType = WGPUSType_ShaderSourceWGSL,
        .code = mipmap_shader,
      },
    }
  );

  /* Create pipeline */
  mipmap_gen.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("mip level generator"),
      .layout = NULL,
      .vertex = (WGPUVertexState){
        .module = shader_module,
        .entryPoint = "vs",
      },
      .fragment = &(WGPUFragmentState){
        .module = shader_module,
        .entryPoint = "fs",
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = WGPUTextureFormat_RGBA8Unorm,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
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
                                         .baseMipLevel  = current_mip - 1,
                                         .mipLevelCount = 1,
                                       });

    WGPUTextureView dst_view
      = wgpuTextureCreateView(texture, &(WGPUTextureViewDescriptor){
                                         .baseMipLevel  = current_mip,
                                         .mipLevelCount = 1,
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
  if (!state.gltf_buffer || state.gltf_data) {
    return;
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
        /* Image data is in buffer view */
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
          .usage = WGPUTextureUsage_TextureBinding 
                 | WGPUTextureUsage_CopyDst 
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
        WGPUImageCopyTexture destination = {
          .texture  = state.gltf_textures[i],
          .mipLevel = 0,
          .origin   = (WGPUOrigin3D){0, 0, 0},
          .aspect   = WGPUTextureAspect_All,
        };

        WGPUTextureDataLayout layout = {
          .offset       = 0,
          .bytesPerRow  = width * 4,
          .rowsPerImage = (uint32_t)height,
        };

        wgpuQueueWriteTexture(wgpu_context->queue, &destination, pixels,
                              width * height * 4, &layout, &tex_desc.size);

        /* Generate mipmaps */
        if (mip_levels > 1) {
          generate_mipmaps(wgpu_context, state.gltf_textures[i], width, height,
                           mip_levels);
        }

        stbi_image_free(pixels);
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
  hdr_image_t hdr = load_hdr_image(state.hdr_buffer, state.hdr_buffer_size);
  if (!hdr.data) {
    return;
  }

  /* Convert to cubemap */
  state.cubemap_texture = convert_equirectangular_to_cubemap(
    wgpu_context, &hdr, state.settings.cubemap_size);

  /* Generate irradiance map */
  state.irradiance_map = generate_irradiance_map(
    wgpu_context, state.cubemap_texture, state.settings.irradiance_map_size);

  /* Generate prefilter map */
  state.prefilter_map = generate_prefilter_map(
    wgpu_context, state.cubemap_texture, state.settings.prefilter_map_size,
    state.settings.roughness_levels);

  /* Free HDR data */
  free_hdr_image(&hdr);
}

/**
 * @brief Create simple cube geometry for testing
 */
static void init_cube_geometry(wgpu_context_t* wgpu_context)
{
  /* Simple cube vertices (position + normal + texcoord) */
  static const float cube_vertices[] = {
    /* Front face */
    -1.0f, -1.0f, 1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f,
    0.0f,  1.0f,  1.0f, 0.0f, 1.0f,  1.0f, 1.0f, 0.0f, 0.0f, 1.0f,  1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  1.0f, 0.0f,
    0.0f,  1.0f,  1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,  0.0f, 1.0f,
    /* Add other faces similarly... for now just front face */
  };

  state.vertex_count = 6;

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
}

/**
 * @brief Initialize camera uniform buffer
 */
static void init_camera_uniforms(wgpu_context_t* wgpu_context)
{
  /* Camera uniforms: projection + view matrices */
  state.camera_uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("camera uniforms"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4) * 2, /* projection + view */
    });
}

/**
 * @brief Initialize render pipeline
 */
static void init_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* Create bind group layouts */
  state.camera_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label = STRVIEW("camera bind group layout"),
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry){
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(mat4) * 2,
        },
      },
    }
  );

  /* Create simple vertex shader */
  const char* vertex_shader = CODE(
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

  /* Create simple fragment shader */
  const char* fragment_shader
    = CODE(@fragment fn main(@location(0) normal : vec3f,
                             @location(1) texcoord : vec2f)
             ->@location(0) vec4f {
               let n     = normalize(normal);
               let light = max(dot(n, normalize(vec3f(1.0, 1.0, 1.0))), 0.0);
               return vec4f(vec3f(0.5) * light + vec3f(0.2), 1.0);
             });

  WGPUShaderModule vs_module = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = STRVIEW("vertex shader"),
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain.sType = WGPUSType_ShaderSourceWGSL,
        .code = vertex_shader,
      },
    }
  );

  WGPUShaderModule fs_module = wgpuDeviceCreateShaderModule(
    wgpu_context->device,
    &(WGPUShaderModuleDescriptor){
      .label = STRVIEW("fragment shader"),
      .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain.sType = WGPUSType_ShaderSourceWGSL,
        .code = fragment_shader,
      },
    }
  );

  /* Create pipeline */
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts = &state.camera_bind_group_layout,
                          });

  state.render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("render pipeline"),
      .layout = pipeline_layout,
      .vertex = (WGPUVertexState){
        .module = vs_module,
        .entryPoint = "main",
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
        .cullMode = WGPUCullMode_Back,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = ~0u,
      },
      .fragment = &(WGPUFragmentState){
        .module = fs_module,
        .entryPoint = "main",
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
      .label = STRVIEW("camera bind group"),
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
  WGPU_RELEASE_RESOURCE(ShaderModule, fs_module)
  WGPU_RELEASE_RESOURCE(ShaderModule, vs_module)
}

/* -------------------------------------------------------------------------- *
 * Initialization Functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Initialize IBL textures (cubemap, irradiance, prefilter, BRDF LUT)
 */
static void init_ibl_textures(wgpu_context_t* wgpu_context)
{
  /* Initialize camera */
  camera_init(&state.camera, 0.0f, 0.0f, 5.0f);

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

  /* Initialize cubemap shared data */
  init_cubemap_shared_data();

  /* Initialize mipmap generator */
  init_mipmap_generator(wgpu_context);

  /* Initialize IBL textures */
  init_ibl_textures(wgpu_context);

  /* Initialize default textures */
  init_default_textures(wgpu_context);

  /* Initialize cube geometry */
  init_cube_geometry(wgpu_context);

  /* Initialize camera uniforms */
  init_camera_uniforms(wgpu_context);

  /* Initialize render pipeline */
  init_render_pipeline(wgpu_context);

  /* Start loading GLTF model */
  load_gltf_model("assets/helmet-flipped.glb");

  /* Start loading HDR environment */
  load_hdr_environment("assets/venice_sunset_1k.hdr");

  state.initialized = true;

  return EXIT_SUCCESS;
}

/**
 * @brief Update uniform buffers (camera matrices, etc.)
 */
static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update camera view matrix */
  camera_update_view_matrix(&state.camera);

  /* Update projection matrix based on window size */
  mat4 projection;
  float aspect
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;
  glm_perspective(GLM_PI_4f, aspect, 0.1f, 100.0f, projection);

  /* Write matrices to uniform buffer */
  float matrices[32]; /* 2 mat4s */
  memcpy(matrices, projection, sizeof(mat4));
  memcpy(matrices + 16, state.camera.view_matrix, sizeof(mat4));

  wgpuQueueWriteBuffer(wgpu_context->queue, state.camera_uniform_buffer, 0,
                       matrices, sizeof(matrices));
}

/**
 * @brief Frame rendering function
 */
static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async file loading */
  sfetch_dowork();

  /* Process loaded GLTF data */
  process_gltf_data(wgpu_context);

  /* Process loaded HDR data */
  process_hdr_data(wgpu_context);

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

  if (state.render_pipeline && state.vertex_buffer) {
    wgpuRenderPassEncoderSetPipeline(pass, state.render_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.camera_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(pass, state.vertex_count, 1, 0, 0);
  }

  wgpuRenderPassEncoderEnd(pass);
  WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command);

  WGPU_RELEASE_RESOURCE(CommandBuffer, command)
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)

  return EXIT_SUCCESS;
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

  /* Release GLTF textures */
  if (state.gltf_textures) {
    for (uint32_t i = 0; i < state.gltf_texture_count; i++) {
      WGPU_RELEASE_RESOURCE(Texture, state.gltf_textures[i])
    }
    free(state.gltf_textures);
    state.gltf_textures = NULL;
  }

  /* Release rendering resources */
  WGPU_RELEASE_RESOURCE(BindGroup, state.camera_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.camera_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.camera_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.material_bind_group_layout)

  /* Release default textures */
  WGPU_RELEASE_RESOURCE(Texture, state.default_white_texture)
  WGPU_RELEASE_RESOURCE(Texture, state.default_normal_texture)
  WGPU_RELEASE_RESOURCE(Sampler, state.default_sampler)

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
  wgpu_start(&(wgpu_desc_t){
    .title       = "GLTF PBR with IBL",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader Code Implementations
 * -------------------------------------------------------------------------- */

/* Distribution GGX */
static const char* distribution_ggx_code
  = CODE(fn distributionGGX(N : vec3f, H : vec3f, roughness : f32)->f32 {
      let a      = roughness * roughness;
      let a2     = a * a;
      let NdotH  = max(dot(N, H), 0.0);
      let NdotH2 = NdotH * NdotH;

      let nom   = a2;
      var denom = (NdotH2 * (a2 - 1.0) + 1.0);
      denom     = PI * denom * denom;

      return nom / denom;
    });

/* Geometry Schlick GGX */
static const char* geometry_schlick_ggx_code
  = CODE(fn geometrySchlickGGX(NdotV : f32, roughness : f32)->f32 {
      let r = (roughness + 1.0);
      let k = (r * r) / 8.0;

      let nom   = NdotV;
      let denom = NdotV * (1.0 - k) + k;

      return nom / denom;
    });

/* Geometry Smith */
static const char* geometry_smith_code = CODE(
  fn geometrySmith(N : vec3f, V : vec3f, L : vec3f, roughness : f32)->f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2  = geometrySchlickGGX(NdotV, roughness);
    let ggx1  = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
  });

/* Fresnel Schlick */
static const char* fresnel_schlick_code
  = CODE(fn fresnelSchlick(cosTheta : f32, F0 : vec3f)->vec3f {
      return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    });

/* Fresnel Schlick Roughness */
static const char* fresnel_schlick_roughness_code
  = CODE(fn fresnelSchlickRoughness(cosTheta : f32, F0 : vec3f, roughness : f32)
           ->vec3f {
             return F0
                    + (max(vec3f(1.0 - roughness), F0) - F0)
                        * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
           });

/* Radical Inverse VdC */
static const char* radical_inverse_vdc_code
  = CODE(fn radicalInverseVdC(inputBits : u32)->f32 {
      var bits = inputBits;
      bits     = (bits << 16u) | (bits >> 16u);
      bits     = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
      bits     = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
      bits     = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
      bits     = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
      return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
    });

/* Hammersley */
static const char* hammersley_code
  = CODE(fn hammersley(i : u32, N : u32)->vec2f {
      return vec2f(f32(i) / f32(N), radicalInverseVdC(i));
    });

/* Importance Sample GGX */
static const char* importance_sample_ggx_code
  = CODE(fn importanceSampleGGX(Xi : vec2f, N : vec3f, roughness : f32)->vec3f {
      let a = roughness * roughness;

      let phi      = 2.0 * PI * Xi.x;
      let cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
      let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

      // from spherical coordinates to cartesian coordinates
      let H = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

      // from tangent-space vector to world-space sample vector
      let up
        = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(N.z) < 0.999);
      let tangent   = normalize(cross(up, N));
      let bitangent = cross(N, tangent);

      let sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
      return normalize(sampleVec);
    });

/* Tone Mapping - ACES */
static const char* tone_mapping_aces_code
  = CODE(fn toneMapACES(color : vec3f)->vec3f {
      let a = 2.51;
      let b = 0.03;
      let c = 2.43;
      let d = 0.59;
      let e = 0.14;
      return clamp((color * (a * color + b)) / (color * (c * color + d) + e),
                   vec3f(0.0), vec3f(1.0));
    });

/* Tone Mapping - Reinhard */
static const char* tone_mapping_reinhard_code
  = CODE(fn toneMapReinhard(color : vec3f)->vec3f {
      return color / (color + vec3f(1.0));
    });

/* Tone Mapping - Uncharted 2 */
static const char* tone_mapping_uncharted2_code = CODE(
  fn uncharted2Tonemap(x : vec3f)
    ->vec3f {
      let A = 0.15;
      let B = 0.50;
      let C = 0.10;
      let D = 0.20;
      let E = 0.02;
      let F = 0.30;
      return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F))
             - E / F;
    }

  fn toneMapUncharted2(color : vec3f)
    ->vec3f {
      let W            = vec3f(11.2);
      let exposureBias = 2.0;
      let curr         = uncharted2Tonemap(exposureBias * color);
      let whiteScale   = vec3f(1.0) / uncharted2Tonemap(W);
      return curr * whiteScale;
    });

/* Tone Mapping - Lottes */
static const char* tone_mapping_lottes_code
  = CODE(fn toneMapLottes(color : vec3f)->vec3f {
      let a      = vec3f(1.6);
      let d      = vec3f(0.977);
      let hdrMax = 8.0;
      let midIn  = 0.18;
      let midOut = 0.267;

      let b = (-pow(midIn, a) + pow(hdrMax, a) * midOut)
              / ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
      let c = (pow(hdrMax, a * d) * pow(midIn, a)
               - pow(hdrMax, a) * pow(midIn, a * d) * midOut)
              / ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

      return pow(color, a) / (pow(color, a * d) * b + c);
    });
