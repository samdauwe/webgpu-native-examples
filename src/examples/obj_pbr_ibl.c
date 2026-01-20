#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

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
 * WebGPU Example - PBR with IBL (Physically Based Rendering with Image Based
 * Lighting) - OBJ Loader
 *
 * This example demonstrates Physically Based Rendering with Image Based
 * Lighting using WebGPU, loading OBJ files and HDR environment maps.
 *
 * Features:
 * - OBJ file loading for 3D models
 * - HDR environment map loading
 * - Equirectangular to cubemap conversion
 * - Irradiance map generation
 * - Prefilter map generation for specular IBL
 * - BRDF lookup table generation
 * - Interactive camera controls
 *
 * Ref:
 * https://github.com/tchayen/pbr-webgpu
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define LIGHT_COUNT 4
#define COUNT_X 6
#define COUNT_Y 2
#define CUBEMAP_SIZE 512
#define IRRADIANCE_MAP_SIZE 32
#define PREFILTER_MAP_SIZE 256
#define ROUGHNESS_LEVELS 5
#define SAMPLE_COUNT 4

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
 * WGSL Shader Code - Forward Declarations
 * -------------------------------------------------------------------------- */

static const char* distribution_ggx_wgsl;
static const char* geometry_schlick_ggx_wgsl;
static const char* geometry_smith_wgsl;
static const char* fresnel_schlick_wgsl;
static const char* fresnel_schlick_roughness_wgsl;
static const char* tone_mapping_lottes_wgsl;

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
 * OBJ File Parser
 * -------------------------------------------------------------------------- */

typedef struct {
  uint32_t vertex_index;
  uint32_t uv_index;
  uint32_t normal_index;
} obj_face_vertex_t;

typedef struct {
  obj_face_vertex_t vertices[3]; /* Triangle face */
} obj_face_t;

typedef struct {
  vec3* vertices;
  uint32_t vertex_count;
  vec2* uvs;
  uint32_t uv_count;
  vec3* normals;
  uint32_t normal_count;
  obj_face_t* faces;
  uint32_t face_count;
} obj_data_t;

/**
 * @brief Parse OBJ file from string content
 */
static bool parse_obj_file(const char* content, obj_data_t* obj)
{
  if (!content || !obj) {
    return false;
  }

  /* Initialize counters */
  obj->vertex_count = 0;
  obj->uv_count     = 0;
  obj->normal_count = 0;
  obj->face_count   = 0;

  /* First pass: count elements */
  const char* line = content;
  while (*line) {
    if (line[0] == 'v' && line[1] == ' ') {
      obj->vertex_count++;
    }
    else if (line[0] == 'v' && line[1] == 't') {
      obj->uv_count++;
    }
    else if (line[0] == 'v' && line[1] == 'n') {
      obj->normal_count++;
    }
    else if (line[0] == 'f' && line[1] == ' ') {
      obj->face_count++;
    }
    /* Move to next line */
    while (*line && *line != '\n') {
      line++;
    }
    if (*line == '\n') {
      line++;
    }
  }

  /* Allocate memory */
  obj->vertices = (vec3*)malloc(obj->vertex_count * sizeof(vec3));
  obj->uvs      = (vec2*)malloc(obj->uv_count * sizeof(vec2));
  obj->normals  = (vec3*)malloc(obj->normal_count * sizeof(vec3));
  obj->faces    = (obj_face_t*)malloc(obj->face_count * sizeof(obj_face_t));

  if (!obj->vertices || !obj->uvs || !obj->normals || !obj->faces) {
    return false;
  }

  /* Second pass: parse data */
  line            = content;
  uint32_t v_idx  = 0;
  uint32_t vt_idx = 0;
  uint32_t vn_idx = 0;
  uint32_t f_idx  = 0;
  char line_buffer[256];

  while (*line) {
    /* Copy line to buffer */
    size_t len = 0;
    while (*line && *line != '\n' && len < sizeof(line_buffer) - 1) {
      line_buffer[len++] = *line++;
    }
    line_buffer[len] = '\0';

    if (*line == '\n') {
      line++;
    }

    /* Parse line */
    if (line_buffer[0] == 'v' && line_buffer[1] == ' ') {
      /* Vertex position */
      float x, y, z;
      if (sscanf(line_buffer, "v %f %f %f", &x, &y, &z) == 3) {
        obj->vertices[v_idx][0] = x;
        obj->vertices[v_idx][1] = y;
        obj->vertices[v_idx][2] = z;
        v_idx++;
      }
    }
    else if (line_buffer[0] == 'v' && line_buffer[1] == 't') {
      /* Texture coordinate */
      float u, v;
      if (sscanf(line_buffer, "vt %f %f", &u, &v) == 2) {
        obj->uvs[vt_idx][0] = u;
        obj->uvs[vt_idx][1] = v;
        vt_idx++;
      }
    }
    else if (line_buffer[0] == 'v' && line_buffer[1] == 'n') {
      /* Normal */
      float x, y, z;
      if (sscanf(line_buffer, "vn %f %f %f", &x, &y, &z) == 3) {
        obj->normals[vn_idx][0] = x;
        obj->normals[vn_idx][1] = y;
        obj->normals[vn_idx][2] = z;
        vn_idx++;
      }
    }
    else if (line_buffer[0] == 'f' && line_buffer[1] == ' ') {
      /* Face (triangle) */
      uint32_t v1, v2, v3, vt1, vt2, vt3, vn1, vn2, vn3;
      if (sscanf(line_buffer, "f %u/%u/%u %u/%u/%u %u/%u/%u", &v1, &vt1, &vn1,
                 &v2, &vt2, &vn2, &v3, &vt3, &vn3)
          == 9) {
        obj->faces[f_idx].vertices[0].vertex_index = v1 - 1;
        obj->faces[f_idx].vertices[0].uv_index     = vt1 - 1;
        obj->faces[f_idx].vertices[0].normal_index = vn1 - 1;

        obj->faces[f_idx].vertices[1].vertex_index = v2 - 1;
        obj->faces[f_idx].vertices[1].uv_index     = vt2 - 1;
        obj->faces[f_idx].vertices[1].normal_index = vn2 - 1;

        obj->faces[f_idx].vertices[2].vertex_index = v3 - 1;
        obj->faces[f_idx].vertices[2].uv_index     = vt3 - 1;
        obj->faces[f_idx].vertices[2].normal_index = vn3 - 1;

        f_idx++;
      }
    }
  }

  return true;
}

/**
 * @brief Clean up OBJ data
 */
static void obj_data_cleanup(obj_data_t* obj)
{
  if (obj) {
    if (obj->vertices) {
      free(obj->vertices);
      obj->vertices = NULL;
    }
    if (obj->uvs) {
      free(obj->uvs);
      obj->uvs = NULL;
    }
    if (obj->normals) {
      free(obj->normals);
      obj->normals = NULL;
    }
    if (obj->faces) {
      free(obj->faces);
      obj->faces = NULL;
    }
    obj->vertex_count = 0;
    obj->uv_count     = 0;
    obj->normal_count = 0;
    obj->face_count   = 0;
  }
}

/* -------------------------------------------------------------------------- *
 * Lights Configuration
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 position;
  float padding1;
  vec3 color;
  float padding2;
} light_t;

static light_t lights[LIGHT_COUNT] = {
  {.position = {-10.0f, 10.0f, 10.0f}, .color = {100.0f, 100.0f, 100.0f}},
  {.position = {10.0f, 10.0f, 10.0f}, .color = {100.0f, 100.0f, 100.0f}},
  {.position = {-10.0f, -10.0f, 10.0f}, .color = {100.0f, 100.0f, 100.0f}},
  {.position = {10.0f, -10.0f, 10.0f}, .color = {100.0f, 100.0f, 100.0f}},
};

/* -------------------------------------------------------------------------- *
 * HDR Image Structure
 * -------------------------------------------------------------------------- */

typedef struct {
  uint32_t width;
  uint32_t height;
  uint16_t* data; /* Float16 data (RGBA) */
} hdr_image_t;

/* -------------------------------------------------------------------------- *
 * Main state structure
 * -------------------------------------------------------------------------- */

static struct {
  camera_t camera;
  obj_data_t obj;
  hdr_image_t hdr;

  /* File loading */
  uint8_t obj_file_buffer[1024 * 1024];     /* 1MB for OBJ file */
  uint8_t hdr_file_buffer[4 * 1024 * 1024]; /* 4MB for HDR file */
  bool obj_loaded;
  bool hdr_loaded;
  bool pipelines_created;

  /* Textures */
  WGPUTexture cubemap_texture;
  WGPUTextureView cubemap_texture_view;
  WGPUTexture irradiance_map;
  WGPUTextureView irradiance_map_view;
  WGPUTexture prefilter_map;
  WGPUTextureView prefilter_map_view;
  WGPUTexture brdf_lookup;
  WGPUTextureView brdf_lookup_view;
  WGPUTexture color_texture;
  WGPUTextureView color_texture_view;
  WGPUTexture depth_texture;
  WGPUTextureView depth_texture_view;

  /* Buffers */
  WGPUBuffer position_buffer;
  WGPUBuffer uniform_buffer;
  WGPUBuffer matrix_buffer;
  WGPUBuffer lights_buffer;
  WGPUBuffer cubemap_vertices_buffer;
  WGPUBuffer cubemap_uniform_buffer;
  WGPUBuffer view_projection_buffer;
  WGPUBuffer skybox_uniform_buffer; /* view + projection for skybox */

  /* Bind groups and layouts */
  WGPUBindGroup uniform_bind_group;         /* PBR group 0 */
  WGPUBindGroup matrix_bind_group;          /* PBR group 1 */
  WGPUBindGroup texture_bind_group;         /* PBR group 2 */
  WGPUBindGroup skybox_uniform_bind_group;  /* Skybox group 0 */
  WGPUBindGroup cubemap_uniform_bind_group; /* Skybox group 1 */

  /* Pipelines */
  WGPURenderPipeline pipeline;
  WGPURenderPipeline skybox_pipeline;

  /* Samplers */
  WGPUSampler sampler;
  WGPUSampler sampler_brdf;
  WGPUSampler main_sampler;

  /* Cubemap view matrices */
  mat4 cubemap_view_matrices[6];

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

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
    .stencilLoadOp     = WGPULoadOp_Undefined,
    .stencilStoreOp    = WGPUStoreOp_Undefined,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
};

/* Placeholder main functions - will be implemented */
static int init(wgpu_context_t* wgpu_context);
static int frame(wgpu_context_t* wgpu_context);
static void shutdown(wgpu_context_t* wgpu_context);
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event);

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "OBJ PBR with IBL",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader Code Implementations
 * -------------------------------------------------------------------------- */

/* PBR Distribution GGX function */
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

/* PBR Geometry Schlick GGX function */
static const char* geometry_schlick_ggx_wgsl
  = CODE(fn geometrySchlickGGX(nDotV : f32, roughness : f32)->f32 {
      let r = (roughness + 1.0);
      let k = (r * r) / 8.0;
      return nDotV / (nDotV * (1.0 - k) + k);
    });

/* PBR Geometry Smith function */
static const char* geometry_smith_wgsl = CODE(
  fn geometrySmith(n : vec3f, v : vec3f, l : vec3f, roughness : f32)->f32 {
    let nDotV = max(dot(n, v), 0.0);
    let nDotL = max(dot(n, l), 0.0);
    let ggx2  = geometrySchlickGGX(nDotV, roughness);
    let ggx1  = geometrySchlickGGX(nDotL, roughness);
    return ggx1 * ggx2;
  });

/* PBR Fresnel Schlick function */
static const char* fresnel_schlick_wgsl
  = CODE(fn fresnelSchlick(cosTheta : f32, f0 : vec3f)->vec3f {
      return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    });

/* PBR Fresnel Schlick Roughness function */
static const char* fresnel_schlick_roughness_wgsl
  = CODE(fn fresnelSchlickRoughness(cosTheta : f32, f0 : vec3f, roughness : f32)
           ->vec3f {
             return f0
                    + (max(vec3(1.0 - roughness), f0) - f0)
                        * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
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
/* clang-format on */

/* PBR Vertex Shader */
/* clang-format off */
static const char* pbr_vertex_shader_wgsl = CODE(
  struct VSOut {
    @builtin(position) Position: vec4f,
    @location(0) normal: vec3f,
    @location(1) uv: vec2f,
    @location(2) @interpolate(flat) instanceIndex: u32,
    @location(3) worldPosition: vec3f,
  };

  @group(1) @binding(0) var<uniform> modelMatrices: array<mat4x4f, 12>;
  @group(1) @binding(1) var<uniform> viewProjectionMatrix: mat4x4f;

  @vertex
  fn main(
    @builtin(instance_index) instanceIndex: u32,
    @location(0) inPosition: vec3f,
    @location(1) inNormal: vec3f,
    @location(2) inUV: vec2f,
  ) -> VSOut {
    var vsOut: VSOut;
    vsOut.Position = viewProjectionMatrix * modelMatrices[instanceIndex] * vec4f(inPosition, 1);
    vsOut.normal = inNormal;
    vsOut.uv = inUV;
    vsOut.worldPosition = (modelMatrices[instanceIndex] * vec4f(inPosition, 1)).xyz;
    vsOut.instanceIndex = instanceIndex;
    return vsOut;
  }
);
/* clang-format on */

/* PBR Fragment Shader - Built dynamically with PBR functions */
static char pbr_fragment_shader_wgsl[8192]; /* Large buffer for shader code */

/**
 * @brief Build PBR fragment shader with all PBR functions
 */
static void build_pbr_fragment_shader(void)
{
  /* clang-format off */
  snprintf(pbr_fragment_shader_wgsl, sizeof(pbr_fragment_shader_wgsl),
           CODE(struct Uniforms {
                  cameraPosition : vec3f,
                }

                struct Light {
                  position : vec3f,
                  padding1 : f32,
                  color : vec3f,
                  padding2 : f32,
                }

                @group(0) @binding(0) var<uniform> uni
                : Uniforms;
                @group(0) @binding(1) var<uniform> lights
                : array<Light, 4>;

                @group(2) @binding(0) var ourSampler
                : sampler;
                @group(2) @binding(1) var samplerBRDF
                : sampler;
                @group(2) @binding(2) var brdfLUT
                : texture_2d<f32>;
                @group(2) @binding(3) var irradianceMap
                : texture_cube<f32>;
                @group(2) @binding(4) var prefilterMap
                : texture_cube<f32>;

                const PI                 = 3.14159265359;
                const MAX_REFLECTION_LOD = 4.0;

                %s

                %s

                %s

                %s

                %s

                %s

                @fragment fn main(@location(0) normal
                                  : vec3f, @location(1) uv
                                  : vec2f,
                                  @location(2) @interpolate(flat) instanceIndex
                                  : u32, @location(3) worldPosition
                                  : vec3f, )
                    ->@location(0) vec4f {
                  let ao       = 1.0;
                  let albedo   = select(vec3f(0.957, 0.792, 0.407), vec3f(1, 0, 0),
                                      instanceIndex < 6);
                  let metallic = select(1.0, 0.0, instanceIndex < 6);
                  let roughness = f32(instanceIndex) % 6 / 6;

                  let n = normalize(normal);
                  let v = normalize(uni.cameraPosition - worldPosition);
                  let r = reflect(-v, n);

                  let f0 = mix(vec3f(0.04), albedo, metallic);

                  var lo = vec3f(0.0);

                  for (var i = 0; i < 4; i++) {
                    let l = normalize(lights[i].position - worldPosition);
                    let h = normalize(v + l);

                    let distance    = length(lights[i].position - worldPosition);
                    let attenuation = 1.0 / (distance * distance);
                    let radiance    = lights[i].color * attenuation;

                    let d = distributionGGX(n, h, roughness);
                    let g = geometrySmith(n, v, l, roughness);
                    let f = fresnelSchlick(max(dot(h, v), 0.0), f0);

                    let numerator = d * g * f;
                    let denominator =
                        4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0) + 0.00001;
                    let specular = numerator / denominator;

                    let kS = f;
                    var kD = vec3f(1.0) - kS;
                    kD *= 1.0 - metallic;

                    let nDotL = max(dot(n, l), 0.00001);
                    lo += (kD * albedo / PI + specular) * radiance * nDotL;
                  }

                  let f = fresnelSchlickRoughness(max(dot(n, v), 0.00001), f0, roughness);
                  let kS = f;
                  var kD = vec3f(1.0) - kS;
                  kD *= 1.0 - metallic;

                  let irradiance = textureSample(irradianceMap, ourSampler, n).rgb;
                  let diffuse    = irradiance * albedo;

                  let prefilteredColor =
                      textureSampleLevel(prefilterMap, ourSampler, r,
                                         roughness * MAX_REFLECTION_LOD)
                          .rgb;
                  let brdf =
                      textureSample(brdfLUT, samplerBRDF,
                                    vec2f(max(dot(n, v), 0.0), roughness))
                          .rg;
                  let specular = prefilteredColor * (f * brdf.x + brdf.y);

                  let ambient = (kD * diffuse + specular) * ao;

                  var color = ambient + lo;
                  color     = toneMapping(color);
                  color     = pow(color, vec3f(1.0 / 2.2));
                  return vec4f(color, 1.0);
                }),
           distribution_ggx_wgsl, geometry_schlick_ggx_wgsl,
           geometry_smith_wgsl, fresnel_schlick_wgsl,
           fresnel_schlick_roughness_wgsl, tone_mapping_lottes_wgsl);
  /* clang-format on */
}

/* Skybox Vertex Shader */
/* clang-format off */
static const char* skybox_vertex_shader_wgsl = CODE(
  struct Uniforms {
    view: mat4x4f,
    projection: mat4x4f,
  }

  @binding(0) @group(0) var<uniform> uniforms: Uniforms;

  struct VertexOutput {
    @builtin(position) Position: vec4f,
    @location(0) fragmentPosition: vec4f,
  }

  @vertex
  fn main(@location(0) position: vec4f) -> VertexOutput {
    var output: VertexOutput;

    var copy = uniforms.view;
    copy[3][0] = 0.0;
    copy[3][1] = 0.0;
    copy[3][2] = 0.0;

    output.Position = (uniforms.projection * copy * position).xyww;
    output.fragmentPosition = 0.5 * (position + vec4(1.0, 1.0, 1.0, 1.0));
    return output;
  }
);

/* Skybox Fragment Shader */
/* clang-format off */
static const char* skybox_fragment_shader_wgsl = CODE(
  @group(1) @binding(0) var myTexture: texture_cube<f32>;
  @group(1) @binding(1) var mySampler: sampler;

  fn toneMapping(color: vec3f) -> vec3f {
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
  }

  @fragment
  fn main(@location(0) fragmentPosition: vec4f) -> @location(0) vec4f {
    var cubemapVec = fragmentPosition.xyz - vec3(0.5);
    var color = textureSample(myTexture, mySampler, cubemapVec).rgb;
    color = toneMapping(color);
    color = pow(color, vec3f(1.0 / 2.2));
    return vec4f(color, 1);
  }
);

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
   1.0f,  1.0f, -1.0f, 1.0f,
   1.0f, -1.0f, -1.0f, 1.0f,
  -1.0f,  1.0f, -1.0f, 1.0f,
  /* clang-format on */
};

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

/**
 * @brief Initialize cubemap view matrices
 * 
 * Note on matrix conventions:
 * TypeScript Mat4.lookAt(pos, target, up) creates a matrix with orientation + position,
 * NOT a standard view matrix. Calling .invert() on it gives the view matrix.
 * 
 * CGLM glm_lookat(eye, center, up) returns the standard view matrix directly.
 * 
 * cubemapViewMatrices (TypeScript): lookAt().invert() = view matrix
 * cubemapViewMatricesInverted (TypeScript): lookAt() = inverse of view matrix
 * 
 * So for C99:
 * - cubemap_view_matrices = glm_lookat result (already the view matrix)
 * - cubemap_view_matrices_inverted = inverse of glm_lookat result
 */
static void init_cubemap_view_matrices(void)
{
  vec3 center = {0.0f, 0.0f, 0.0f};
  vec3 target, up;

  /* TypeScript cubemapViewMatrices - used for equirectangular to cubemap conversion */
  /* +X face */
  target[0] = 1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices[0]);

  /* -X face */
  target[0] = -1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices[1]);

  /* +Y face */
  target[0] = 0.0f;
  target[1] = -1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = -1.0f;
  glm_lookat(center, target, up, cubemap_view_matrices[2]);

  /* -Y face */
  target[0] = 0.0f;
  target[1] = 1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = 1.0f;
  glm_lookat(center, target, up, cubemap_view_matrices[3]);

  /* +Z face */
  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = 1.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices[4]);

  /* -Z face */
  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = -1.0f;
  up[0]     = 0.0f;
  up[1]     = -1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices[5]);

  /* TypeScript cubemapViewMatricesInverted - used for irradiance and prefilter maps */
  /* These are the inverses of glm_lookat (i.e., inverse of view matrix) */
  /* +X face */
  target[0] = 1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[0]);
  glm_mat4_inv(cubemap_view_matrices_inverted[0], cubemap_view_matrices_inverted[0]);

  /* -X face */
  target[0] = -1.0f;
  target[1] = 0.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[1]);
  glm_mat4_inv(cubemap_view_matrices_inverted[1], cubemap_view_matrices_inverted[1]);

  /* +Y face */
  target[0] = 0.0f;
  target[1] = 1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = -1.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[2]);
  glm_mat4_inv(cubemap_view_matrices_inverted[2], cubemap_view_matrices_inverted[2]);

  /* -Y face */
  target[0] = 0.0f;
  target[1] = -1.0f;
  target[2] = 0.0f;
  up[0]     = 0.0f;
  up[1]     = 0.0f;
  up[2]     = 1.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[3]);
  glm_mat4_inv(cubemap_view_matrices_inverted[3], cubemap_view_matrices_inverted[3]);

  /* +Z face */
  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = 1.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[4]);
  glm_mat4_inv(cubemap_view_matrices_inverted[4], cubemap_view_matrices_inverted[4]);

  /* -Z face */
  target[0] = 0.0f;
  target[1] = 0.0f;
  target[2] = -1.0f;
  up[0]     = 0.0f;
  up[1]     = 1.0f;
  up[2]     = 0.0f;
  glm_lookat(center, target, up, cubemap_view_matrices_inverted[5]);
  glm_mat4_inv(cubemap_view_matrices_inverted[5], cubemap_view_matrices_inverted[5]);
}

/* -------------------------------------------------------------------------- *
 * HDR File Parsing Using stb_image
 * -------------------------------------------------------------------------- */

/**
 * @brief Load HDR file using stb_image and convert to float16
 */
static bool load_hdr_file(const uint8_t* buffer, size_t buffer_size,
                          hdr_image_t* hdr)
{
  if (!buffer || !hdr || buffer_size == 0) {
    return false;
  }

  /* Load HDR using stb_image */
  int width, height, channels;
  float* hdr_data = stbi_loadf_from_memory(buffer, (int)buffer_size, &width,
                                           &height, &channels, 4);
  if (!hdr_data) {
    printf("Failed to load HDR: %s\n", stbi_failure_reason());
    return false;
  }

  /* Convert float32 to float16 */
  hdr->data = (uint16_t*)malloc(width * height * 4 * sizeof(uint16_t));
  if (!hdr->data) {
    stbi_image_free(hdr_data);
    return false;
  }

  for (int i = 0; i < width * height * 4; ++i) {
    hdr->data[i] = float32_to_float16(hdr_data[i]);
  }

  stbi_image_free(hdr_data);

  hdr->width  = (uint32_t)width;
  hdr->height = (uint32_t)height;

  return true;
}

/* -------------------------------------------------------------------------- *
 * File Loading Callbacks
 * -------------------------------------------------------------------------- */

/**
 * @brief Callback for OBJ file loading
 */
static void obj_fetch_callback(const sfetch_response_t* response)
{
  if (response->fetched) {
    const char* content = (const char*)response->data.ptr;
    if (parse_obj_file(content, &state.obj)) {
      state.obj_loaded = true;
      printf("OBJ file loaded: %u vertices, %u faces\n", state.obj.vertex_count,
             state.obj.face_count);
    }
    else {
      printf("Failed to parse OBJ file\n");
    }
  }
  else if (response->failed) {
    printf("Failed to load OBJ file: error %d\n", response->error_code);
  }
}

/**
 * @brief Callback for HDR file loading
 */
static void hdr_fetch_callback(const sfetch_response_t* response)
{
  if (response->fetched) {
    if (load_hdr_file(response->data.ptr, response->data.size, &state.hdr)) {
      state.hdr_loaded = true;
      printf("HDR file loaded: %ux%u\n", state.hdr.width, state.hdr.height);
    }
    else {
      printf("Failed to load HDR file\n");
    }
  }
  else if (response->failed) {
    printf("Failed to load HDR file: error %d\n", response->error_code);
  }
}

/* -------------------------------------------------------------------------- *
 * Initialization Functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Initialize file loading
 */
static void init_file_loading(void)
{
  /* Start loading OBJ file */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/sphere.obj",
    .callback = obj_fetch_callback,
    .buffer   = SFETCH_RANGE(state.obj_file_buffer),
  });

  /* Start loading HDR file */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/venice_sunset_1k.hdr",
    .callback = hdr_fetch_callback,
    .buffer   = SFETCH_RANGE(state.hdr_file_buffer),
  });
}

/**
 * @brief Create a buffer with initial data
 */
static WGPUBuffer create_buffer_with_data(wgpu_context_t* wgpu_context,
                                          const void* data, uint64_t size,
                                          WGPUBufferUsage usage)
{
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

/**
 * @brief Initialize samplers
 */
static void init_samplers(wgpu_context_t* wgpu_context)
{
  /* Main sampler for textures */
  state.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("PBR - Main sampler"),
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_Repeat,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });

  /* Copy main sampler for compatibility */
  state.main_sampler = state.sampler;

  /* BRDF sampler */
  state.sampler_brdf = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("PBR - BRDF sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
}

/**
 * @brief Initialize uniform buffers
 */
static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Uniform buffer for camera position */
  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("PBR - Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = 4 * sizeof(float), /* vec3 + padding */
    });

  /* Lights buffer */
  state.lights_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("PBR - Lights buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(light_t) * LIGHT_COUNT,
    });

  /* Write lights data */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.lights_buffer, 0, lights,
                       sizeof(light_t) * LIGHT_COUNT);

  /* Matrix buffer for instances */
  state.matrix_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("PBR - Matrix buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4) * COUNT_X * COUNT_Y,
    });

  /* View projection buffer */
  state.view_projection_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("PBR - View projection buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4),
    });

  /* Cubemap uniform buffer */
  state.cubemap_uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("PBR - Cubemap uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4) * 2, /* view + projection */
    });

  /* Skybox uniform buffer (for rendering the skybox in main scene) */
  state.skybox_uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Skybox - Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4) * 2, /* view + projection */
    });
}

/**
 * @brief Create vertex buffer from OBJ data
 */
static void create_obj_vertex_buffer(wgpu_context_t* wgpu_context)
{
  if (!state.obj_loaded || state.obj.face_count == 0) {
    return;
  }

  const uint32_t vertex_count      = state.obj.face_count * 3;
  const uint32_t floats_per_vertex = 8;
  const uint64_t buffer_size = vertex_count * floats_per_vertex * sizeof(float);

  float* buffer_data = (float*)malloc(buffer_size);
  if (!buffer_data) {
    return;
  }

  uint32_t buffer_idx = 0;
  for (uint32_t i = 0; i < state.obj.face_count; ++i) {
    for (uint32_t j = 0; j < 3; ++j) {
      obj_face_vertex_t* fv     = &state.obj.faces[i].vertices[j];
      buffer_data[buffer_idx++] = state.obj.vertices[fv->vertex_index][0];
      buffer_data[buffer_idx++] = state.obj.vertices[fv->vertex_index][1];
      buffer_data[buffer_idx++] = state.obj.vertices[fv->vertex_index][2];
      buffer_data[buffer_idx++] = state.obj.normals[fv->normal_index][0];
      buffer_data[buffer_idx++] = state.obj.normals[fv->normal_index][1];
      buffer_data[buffer_idx++] = state.obj.normals[fv->normal_index][2];
      buffer_data[buffer_idx++] = state.obj.uvs[fv->uv_index][0];
      buffer_data[buffer_idx++] = state.obj.uvs[fv->uv_index][1];
    }
  }

  state.position_buffer = create_buffer_with_data(
    wgpu_context, buffer_data, buffer_size, WGPUBufferUsage_Vertex);

  free(buffer_data);

  state.cubemap_vertices_buffer = create_buffer_with_data(
    wgpu_context, cube_vertex_array, sizeof(cube_vertex_array),
    WGPUBufferUsage_Vertex);
}

/**
 * @brief Initialize render textures
 */
static void init_render_textures(wgpu_context_t* wgpu_context)
{
  state.color_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("PBR - Color texture"),
      .size  = (WGPUExtent3D){
         .width              = wgpu_context->width,
         .height             = wgpu_context->height,
         .depthOrArrayLayers = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = SAMPLE_COUNT,
      .dimension     = WGPUTextureDimension_2D,
      .format        = wgpu_context->render_format,
      .usage         = WGPUTextureUsage_RenderAttachment,
    });
  state.color_texture_view = wgpuTextureCreateView(state.color_texture, NULL);

  state.depth_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("PBR - Depth texture"),
      .size  = (WGPUExtent3D){
         .width              = wgpu_context->width,
         .height             = wgpu_context->height,
         .depthOrArrayLayers = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = SAMPLE_COUNT,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24Plus,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
    });
  state.depth_texture_view = wgpuTextureCreateView(state.depth_texture, NULL);
}

/**
 * @brief Create render pipelines
 */
static void init_pipelines(wgpu_context_t* wgpu_context)
{
  build_pbr_fragment_shader();

  WGPUShaderModule pbr_vert
    = wgpu_create_shader_module(wgpu_context->device, pbr_vertex_shader_wgsl);
  WGPUShaderModule pbr_frag
    = wgpu_create_shader_module(wgpu_context->device, pbr_fragment_shader_wgsl);
  WGPUShaderModule skybox_vert = wgpu_create_shader_module(
    wgpu_context->device, skybox_vertex_shader_wgsl);
  WGPUShaderModule skybox_frag = wgpu_create_shader_module(
    wgpu_context->device, skybox_fragment_shader_wgsl);

  WGPUBlendState blend_state = wgpu_create_blend_state(true);
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });

  WGPUVertexAttribute pbr_attrs[3] = {
    {.shaderLocation = 0, .offset = 0, .format = WGPUVertexFormat_Float32x3},
    {.shaderLocation = 1, .offset = 12, .format = WGPUVertexFormat_Float32x3},
    {.shaderLocation = 2, .offset = 24, .format = WGPUVertexFormat_Float32x2},
  };

  WGPUVertexBufferLayout pbr_layout = {
    .arrayStride    = 32,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 3,
    .attributes     = pbr_attrs,
  };

  state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("PBR pipeline"),
      .layout = NULL,
      .vertex =
        {
          .module      = pbr_vert,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = 1,
          .buffers     = &pbr_layout,
        },
      .fragment =
        &(WGPUFragmentState){
          .module      = pbr_frag,
          .entryPoint  = STRVIEW("main"),
          .targetCount = 1,
          .targets =
            &(WGPUColorTargetState){
              .format    = wgpu_context->render_format,
              .blend     = &blend_state,
              .writeMask = WGPUColorWriteMask_All,
            },
        },
      .primitive =
        {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CW,
          .cullMode  = WGPUCullMode_None,
        },
      .depthStencil = &depth_stencil_state,
      .multisample  = {.count = SAMPLE_COUNT, .mask = 0xFFFFFFFF},
    });

  WGPUVertexAttribute skybox_attrs[1] = {
    {.shaderLocation = 0, .offset = 0, .format = WGPUVertexFormat_Float32x4},
  };

  WGPUVertexBufferLayout skybox_layout = {
    .arrayStride    = 16,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 1,
    .attributes     = skybox_attrs,
  };

  WGPUDepthStencilState skybox_depth
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  skybox_depth.depthCompare = WGPUCompareFunction_LessEqual;

  state.skybox_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Skybox pipeline"),
      .layout = NULL,
      .vertex =
        {
          .module      = skybox_vert,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = 1,
          .buffers     = &skybox_layout,
        },
      .fragment =
        &(WGPUFragmentState){
          .module      = skybox_frag,
          .entryPoint  = STRVIEW("main"),
          .targetCount = 1,
          .targets =
            &(WGPUColorTargetState){
              .format    = wgpu_context->render_format,
              .blend     = &blend_state,
              .writeMask = WGPUColorWriteMask_All,
            },
        },
      .primitive =
        {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CW,
          .cullMode  = WGPUCullMode_None,
        },
      .depthStencil = &skybox_depth,
      .multisample  = {.count = SAMPLE_COUNT, .mask = 0xFFFFFFFF},
    });

  wgpuShaderModuleRelease(pbr_vert);
  wgpuShaderModuleRelease(pbr_frag);
  wgpuShaderModuleRelease(skybox_vert);
  wgpuShaderModuleRelease(skybox_frag);

  state.pipelines_created = true;
}

/* ---------------------------------------------------------------------------
 * Bind Groups Creation
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Create bind groups for PBR and skybox rendering
 */
static void create_bind_groups(wgpu_context_t* wgpu_context)
{
  /* PBR Bind Group 0: Uniforms (camera position + lights) */
  WGPUBindGroupEntry pbr_group0_entries[2] = {
    {.binding = 0, .buffer = state.uniform_buffer, .size = 16},
    {.binding = 1, .buffer = state.lights_buffer, .size = LIGHT_COUNT * 32},
  };

  state.uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("PBR uniform bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline, 0),
      .entryCount = 2,
      .entries    = pbr_group0_entries,
    });

  /* PBR Bind Group 1: Matrices (model matrices + view projection) */
  WGPUBindGroupEntry pbr_group1_entries[2] = {
    {.binding = 0,
     .buffer  = state.matrix_buffer,
     .size    = COUNT_X * COUNT_Y * sizeof(mat4)},
    {.binding = 1,
     .buffer  = state.view_projection_buffer,
     .size    = sizeof(mat4)},
  };

  state.matrix_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("PBR matrix bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline, 1),
      .entryCount = 2,
      .entries    = pbr_group1_entries,
    });

  /* PBR Bind Group 2: Textures (samplers + BRDF LUT + irradiance + prefilter)
   */
  WGPUBindGroupEntry pbr_group2_entries[5] = {
    {.binding = 0, .sampler = state.sampler},
    {.binding = 1, .sampler = state.sampler_brdf},
    {.binding = 2, .textureView = state.brdf_lookup_view},
    {.binding = 3, .textureView = state.irradiance_map_view},
    {.binding = 4, .textureView = state.prefilter_map_view},
  };

  state.texture_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("PBR texture bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline, 2),
      .entryCount = 5,
      .entries    = pbr_group2_entries,
    });

  /* Skybox Bind Group 0: Uniforms (view + projection matrices) */
  WGPUBindGroupEntry skybox_group0_entries[1] = {
    {.binding = 0,
     .buffer  = state.skybox_uniform_buffer,
     .size    = sizeof(mat4) * 2},
  };

  WGPUBindGroupLayout skybox_layout0
    = wgpuRenderPipelineGetBindGroupLayout(state.skybox_pipeline, 0);

  state.skybox_uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Skybox uniform bind group"),
                            .layout     = skybox_layout0,
                            .entryCount = 1,
                            .entries    = skybox_group0_entries,
                          });

  /* Skybox Bind Group 1: Cubemap texture + sampler (use irradiance map like TypeScript) */
  WGPUBindGroupEntry skybox_group1_entries[2] = {
    {.binding = 0, .textureView = state.irradiance_map_view},
    {.binding = 1, .sampler = state.sampler},
  };

  state.cubemap_uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Skybox cubemap bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(state.skybox_pipeline, 1),
      .entryCount = 2,
      .entries    = skybox_group1_entries,
    });

  wgpuBindGroupLayoutRelease(skybox_layout0);
}

/* ---------------------------------------------------------------------------
 * IBL Processing - Convert HDR to cubemap and generate IBL textures
 * ---------------------------------------------------------------------------
 */

/* Convert equirectangular HDR to cubemap */
static void convert_equirectangular_to_cubemap(wgpu_context_t* wgpu_context)
{
  /* Compute view-projection matrices for each cubemap face */
  mat4 projection;
  glm_perspective(GLM_PI_2f, 1.0f, 0.1f, 10.0f, projection);

  /* Pre-compute view*projection for each face (stored separately for rendering) */
  mat4 cubemap_mvp_matrices[6];
  for (uint32_t i = 0; i < 6; ++i) {
    /* TypeScript: view.multiply(projection) which is view * projection in row-major */
    /* CGLM column-major: projection * view gives the same result for shader consumption */
    glm_mat4_mul(projection, cubemap_view_matrices[i], cubemap_mvp_matrices[i]);
  }

  /* Create cubemap texture */
  state.cubemap_texture = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label = STRVIEW("Cubemap texture"),
                            .usage = WGPUTextureUsage_TextureBinding
                                     | WGPUTextureUsage_RenderAttachment,
                            .dimension     = WGPUTextureDimension_2D,
                            .size          = {.width              = CUBEMAP_SIZE,
                                              .height             = CUBEMAP_SIZE,
                                              .depthOrArrayLayers = 6},
                            .format        = WGPUTextureFormat_RGBA16Float,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                          });

  state.cubemap_texture_view = wgpuTextureCreateView(
    state.cubemap_texture, &(WGPUTextureViewDescriptor){
                             .label           = STRVIEW("Cubemap texture view"),
                             .format          = WGPUTextureFormat_RGBA16Float,
                             .dimension       = WGPUTextureViewDimension_Cube,
                             .baseMipLevel    = 0,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = 0,
                             .arrayLayerCount = 6,
                           });

  /* Create HDR texture from loaded data */
  WGPUTexture hdr_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label     = STRVIEW("HDR equirectangular texture"),
      .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size      = {.width              = state.hdr.width,
                    .height             = state.hdr.height,
                    .depthOrArrayLayers = 1},
      .format    = WGPUTextureFormat_RGBA16Float,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });

  /* Upload HDR data to texture */
  wgpuQueueWriteTexture(
    wgpu_context->queue,
    &(WGPUTexelCopyTextureInfo){
      .texture  = hdr_texture,
      .mipLevel = 0,
      .origin   = (WGPUOrigin3D){0, 0, 0},
      .aspect   = WGPUTextureAspect_All,
    },
    state.hdr.data, state.hdr.width * state.hdr.height * 4 * sizeof(uint16_t),
    &(WGPUTexelCopyBufferLayout){
      .offset       = 0,
      .bytesPerRow  = state.hdr.width * 4 * sizeof(uint16_t),
      .rowsPerImage = state.hdr.height,
    },
    &(WGPUExtent3D){.width              = state.hdr.width,
                    .height             = state.hdr.height,
                    .depthOrArrayLayers = 1});

  WGPUTextureView hdr_view = wgpuTextureCreateView(hdr_texture, NULL);

  /* Create shader for conversion */
  WGPUShaderModule conversion_shader = wgpu_create_shader_module(
    wgpu_context->device, CODE(
        @group(0) @binding(0) var equirectangular_sampler : sampler;
        @group(0) @binding(1) var equirectangular_texture : texture_2d<f32>;
        @group(0) @binding(2) var<uniform> view_projection : mat4x4<f32>;

        struct VertexOutput {
          @builtin(position) position : vec4<f32>,
          @location(0) local_pos : vec3<f32>,
        }

        @vertex
        fn vs_main(@location(0) position : vec4<f32>) -> VertexOutput {
          var output : VertexOutput;
          output.position = view_projection * position;
          output.local_pos = position.xyz;
          return output;
        }

        const INV_ATAN : vec2<f32> = vec2<f32>(0.1591, 0.3183);

        fn sample_spherical_map(v : vec3<f32>) -> vec2<f32> {
          var uv : vec2<f32> = vec2<f32>(atan2(v.z, v.x), asin(v.y));
          uv = uv * INV_ATAN + 0.5;
          return uv;
        }

        @fragment
        fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
          let uv = sample_spherical_map(normalize(input.local_pos));
          return textureSample(equirectangular_texture, equirectangular_sampler, uv);
        }
      ));

  /* Create bind group layout */
  WGPUBindGroupLayout conversion_bind_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Equirectangular to cubemap bind group layout"),
      .entryCount = 3,
      .entries    = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Fragment,
          .sampler    = {.type = WGPUSamplerBindingType_Filtering},
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Fragment,
          .texture = {
            .sampleType    = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
        {
          .binding    = 2,
          .visibility = WGPUShaderStage_Vertex,
          .buffer     = {.type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(mat4)},
        },
      },
    });

  /* Create bind group */
  WGPUBindGroup conversion_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Equirectangular to cubemap bind group"),
      .layout     = conversion_bind_layout,
      .entryCount = 3,
      .entries    = (WGPUBindGroupEntry[]){
        {.binding = 0, .sampler = state.main_sampler},
        {.binding = 1, .textureView = hdr_view},
        {.binding = 2, .buffer = state.cubemap_uniform_buffer, .size = sizeof(mat4)},
      },
    });

  /* Create pipeline */
  WGPUPipelineLayout conversion_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label = STRVIEW("Equirectangular to cubemap pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &conversion_bind_layout,
    });

  WGPUVertexAttribute conversion_attr = {
    .shaderLocation = 0,
    .offset         = 0,
    .format         = WGPUVertexFormat_Float32x4,
  };

  WGPURenderPipeline conversion_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Equirectangular to cubemap pipeline"),
      .layout = conversion_layout,
      .vertex = {
        .module      = conversion_shader,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &(WGPUVertexBufferLayout){
          .arrayStride    = 16,
          .stepMode       = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes     = &conversion_attr,
        },
      },
      .fragment = &(WGPUFragmentState){
        .module      = conversion_shader,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = WGPUTextureFormat_RGBA16Float,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .depthStencil = NULL,
      .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
    });

  /* Render to each cubemap face */
  for (uint32_t face = 0; face < 6; ++face) {
    WGPUTextureView face_view = wgpuTextureCreateView(
      state.cubemap_texture, &(WGPUTextureViewDescriptor){
                               .format          = WGPUTextureFormat_RGBA16Float,
                               .dimension       = WGPUTextureViewDimension_2D,
                               .baseMipLevel    = 0,
                               .mipLevelCount   = 1,
                               .baseArrayLayer  = face,
                               .arrayLayerCount = 1,
                             });

    wgpuQueueWriteBuffer(wgpu_context->queue, state.cubemap_uniform_buffer, 0,
                         cubemap_mvp_matrices[face], sizeof(mat4));

    WGPUCommandEncoder encoder
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      encoder, &(WGPURenderPassDescriptor){
        .colorAttachmentCount = 1,
        .colorAttachments     = &(WGPURenderPassColorAttachment){
          .view       = face_view,
          .loadOp     = WGPULoadOp_Clear,
          .storeOp    = WGPUStoreOp_Store,
          .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
          .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        },
      });

    wgpuRenderPassEncoderSetPipeline(pass, conversion_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, conversion_bind_group, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.cubemap_vertices_buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(pass, 36, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

    wgpuTextureViewRelease(face_view);
    wgpuCommandBufferRelease(command_buffer);
    wgpuRenderPassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
  }

  /* Cleanup */
  wgpuTextureViewRelease(hdr_view);
  wgpuTextureRelease(hdr_texture);
  wgpuRenderPipelineRelease(conversion_pipeline);
  wgpuPipelineLayoutRelease(conversion_layout);
  wgpuBindGroupRelease(conversion_bind_group);
  wgpuBindGroupLayoutRelease(conversion_bind_layout);
  wgpuShaderModuleRelease(conversion_shader);
}

/* Generate irradiance map from cubemap */
static void generate_irradiance_map(wgpu_context_t* wgpu_context)
{
  /* Compute view-projection matrices for each face using inverted matrices */
  mat4 projection;
  glm_perspective(GLM_PI_2f, 1.0f, 0.1f, 10.0f, projection);

  mat4 irradiance_mvp_matrices[6];
  for (uint32_t i = 0; i < 6; ++i) {
    glm_mat4_mul(projection, cubemap_view_matrices_inverted[i], irradiance_mvp_matrices[i]);
  }

  /* Create irradiance map texture */
  state.irradiance_map = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label = STRVIEW("Irradiance map texture"),
                            .usage = WGPUTextureUsage_TextureBinding
                                     | WGPUTextureUsage_RenderAttachment,
                            .dimension     = WGPUTextureDimension_2D,
                            .size          = {.width              = IRRADIANCE_MAP_SIZE,
                                              .height             = IRRADIANCE_MAP_SIZE,
                                              .depthOrArrayLayers = 6},
                            .format        = WGPUTextureFormat_RGBA16Float,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                          });

  state.irradiance_map_view = wgpuTextureCreateView(
    state.irradiance_map, &(WGPUTextureViewDescriptor){
                            .label           = STRVIEW("Irradiance map view"),
                            .format          = WGPUTextureFormat_RGBA16Float,
                            .dimension       = WGPUTextureViewDimension_Cube,
                            .baseMipLevel    = 0,
                            .mipLevelCount   = 1,
                            .baseArrayLayer  = 0,
                            .arrayLayerCount = 6,
                          });

  /* Create shader for irradiance convolution */
  WGPUShaderModule irradiance_shader = wgpu_create_shader_module(
    wgpu_context->device, CODE(
        @group(0) @binding(0) var environment_sampler : sampler;
        @group(0) @binding(1) var environment_map : texture_cube<f32>;
        @group(0) @binding(2) var<uniform> view_projection : mat4x4<f32>;

        struct VertexOutput {
          @builtin(position) position : vec4<f32>,
          @location(0) local_pos : vec3<f32>,
        }

        @vertex
        fn vs_main(@location(0) position : vec4<f32>) -> VertexOutput {
          var output : VertexOutput;
          output.position = view_projection * position;
          output.local_pos = position.xyz;
          return output;
        }

        const PI : f32 = 3.1415926535897932384626433832795;
        const SAMPLE_DELTA : f32 = 0.025;

        @fragment
        fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
          let normal = normalize(input.local_pos);
          var irradiance = vec3<f32>(0.0);

          let up = vec3<f32>(0.0, 1.0, 0.0);
          let right = normalize(cross(up, normal));
          let up_tangent = normalize(cross(normal, right));

          var num_samples = 0.0;
          var phi = 0.0;
          while (phi < 2.0 * PI) {
            var theta = 0.0;
            while (theta < 0.5 * PI) {
              let tangent_sample = vec3<f32>(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
              );
              let sample_vec = tangent_sample.x * right
                             + tangent_sample.y * up_tangent
                             + tangent_sample.z * normal;

              irradiance += textureSample(environment_map, environment_sampler, sample_vec).rgb
                          * cos(theta) * sin(theta);
              num_samples += 1.0;

              theta += SAMPLE_DELTA;
            }
            phi += SAMPLE_DELTA;
          }

          irradiance = PI * irradiance / num_samples;
          return vec4<f32>(irradiance, 1.0);
        }
      ));

  /* Create bind group layout */
  WGPUBindGroupLayout irradiance_bind_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Irradiance bind group layout"),
      .entryCount = 3,
      .entries    = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Fragment,
          .sampler    = {.type = WGPUSamplerBindingType_Filtering},
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Fragment,
          .texture = {
            .sampleType    = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_Cube,
          },
        },
        {
          .binding    = 2,
          .visibility = WGPUShaderStage_Vertex,
          .buffer     = {.type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(mat4)},
        },
      },
    });

  /* Create bind group */
  WGPUBindGroup irradiance_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Irradiance bind group"),
      .layout     = irradiance_bind_layout,
      .entryCount = 3,
      .entries    = (WGPUBindGroupEntry[]){
        {.binding = 0, .sampler = state.main_sampler},
        {.binding = 1, .textureView = state.cubemap_texture_view},
        {.binding = 2, .buffer = state.cubemap_uniform_buffer, .size = sizeof(mat4)},
      },
    });

  /* Create pipeline */
  WGPUPipelineLayout irradiance_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Irradiance pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &irradiance_bind_layout,
                          });

  WGPUVertexAttribute irradiance_attr = {
    .shaderLocation = 0,
    .offset         = 0,
    .format         = WGPUVertexFormat_Float32x4,
  };

  WGPURenderPipeline irradiance_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Irradiance pipeline"),
      .layout = irradiance_layout,
      .vertex = {
        .module      = irradiance_shader,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &(WGPUVertexBufferLayout){
          .arrayStride    = 16,
          .stepMode       = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes     = &irradiance_attr,
        },
      },
      .fragment = &(WGPUFragmentState){
        .module      = irradiance_shader,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = WGPUTextureFormat_RGBA16Float,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .depthStencil = NULL,
      .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
    });

  /* Render to each irradiance map face */
  for (uint32_t face = 0; face < 6; ++face) {
    WGPUTextureView face_view = wgpuTextureCreateView(
      state.irradiance_map, &(WGPUTextureViewDescriptor){
                              .format          = WGPUTextureFormat_RGBA16Float,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = face,
                              .arrayLayerCount = 1,
                            });

    wgpuQueueWriteBuffer(wgpu_context->queue, state.cubemap_uniform_buffer, 0,
                         irradiance_mvp_matrices[face], sizeof(mat4));

    WGPUCommandEncoder encoder
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      encoder, &(WGPURenderPassDescriptor){
        .colorAttachmentCount = 1,
        .colorAttachments     = &(WGPURenderPassColorAttachment){
          .view       = face_view,
          .loadOp     = WGPULoadOp_Clear,
          .storeOp    = WGPUStoreOp_Store,
          .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
          .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        },
      });

    wgpuRenderPassEncoderSetPipeline(pass, irradiance_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, irradiance_bind_group, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.cubemap_vertices_buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(pass, 36, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

    wgpuTextureViewRelease(face_view);
    wgpuCommandBufferRelease(command_buffer);
    wgpuRenderPassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
  }

  /* Cleanup */
  wgpuRenderPipelineRelease(irradiance_pipeline);
  wgpuPipelineLayoutRelease(irradiance_layout);
  wgpuBindGroupRelease(irradiance_bind_group);
  wgpuBindGroupLayoutRelease(irradiance_bind_layout);
  wgpuShaderModuleRelease(irradiance_shader);
}

/* Generate prefiltered environment map */
static void generate_prefilter_map(wgpu_context_t* wgpu_context)
{
  /* Compute view-projection matrices for each face using inverted matrices */
  mat4 projection;
  glm_perspective(GLM_PI_2f, 1.0f, 0.1f, 10.0f, projection);

  mat4 prefilter_mvp_matrices[6];
  for (uint32_t i = 0; i < 6; ++i) {
    glm_mat4_mul(projection, cubemap_view_matrices_inverted[i], prefilter_mvp_matrices[i]);
  }

  /* Create prefilter map texture with mipmaps */
  state.prefilter_map = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label = STRVIEW("Prefilter map texture"),
                            .usage = WGPUTextureUsage_TextureBinding
                                     | WGPUTextureUsage_RenderAttachment,
                            .dimension     = WGPUTextureDimension_2D,
                            .size          = {.width              = PREFILTER_MAP_SIZE,
                                              .height             = PREFILTER_MAP_SIZE,
                                              .depthOrArrayLayers = 6},
                            .format        = WGPUTextureFormat_RGBA16Float,
                            .mipLevelCount = ROUGHNESS_LEVELS,
                            .sampleCount   = 1,
                          });

  state.prefilter_map_view = wgpuTextureCreateView(
    state.prefilter_map, &(WGPUTextureViewDescriptor){
                           .label           = STRVIEW("Prefilter map view"),
                           .format          = WGPUTextureFormat_RGBA16Float,
                           .dimension       = WGPUTextureViewDimension_Cube,
                           .baseMipLevel    = 0,
                           .mipLevelCount   = ROUGHNESS_LEVELS,
                           .baseArrayLayer  = 0,
                           .arrayLayerCount = 6,
                         });

  /* Create shader for prefilter convolution */
  WGPUShaderModule prefilter_shader = wgpu_create_shader_module(
    wgpu_context->device, CODE(
        @group(0) @binding(0) var environment_sampler : sampler;
        @group(0) @binding(1) var environment_map : texture_cube<f32>;
        @group(0) @binding(2) var<uniform> view_projection : mat4x4<f32>;
        @group(0) @binding(3) var<uniform> roughness : f32;

        struct VertexOutput {
          @builtin(position) position : vec4<f32>,
          @location(0) local_pos : vec3<f32>,
        }

        @vertex
        fn vs_main(@location(0) position : vec4<f32>) -> VertexOutput {
          var output : VertexOutput;
          output.position = view_projection * position;
          output.local_pos = position.xyz;
          return output;
        }

        const PI : f32 = 3.1415926535897932384626433832795;

        fn radical_inverse_vdc(bits_in : u32) -> f32 {
          var bits = bits_in;
          bits = (bits << 16u) | (bits >> 16u);
          bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
          bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
          bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
          bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
          return f32(bits) * 2.3283064365386963e-10;
        }

        fn hammersley(i : u32, n : u32) -> vec2<f32> {
          return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
        }

        fn importance_sample_ggx(xi : vec2<f32>, n : vec3<f32>, roughness_val : f32) -> vec3<f32> {
          let a = roughness_val * roughness_val;
          let phi = 2.0 * PI * xi.x;
          let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
          let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

          let h = vec3<f32>(
            cos(phi) * sin_theta,
            sin(phi) * sin_theta,
            cos_theta
          );

          let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
          let tangent = normalize(cross(up, n));
          let bitangent = cross(n, tangent);

          return normalize(tangent * h.x + bitangent * h.y + n * h.z);
        }

        @fragment
        fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
          let n = normalize(input.local_pos);
          let r = n;
          let v = r;

          const SAMPLE_COUNT = 1024u;
          var prefiltered_color = vec3<f32>(0.0);
          var total_weight = 0.0;

          for (var i = 0u; i < SAMPLE_COUNT; i++) {
            let xi = hammersley(i, SAMPLE_COUNT);
            let h = importance_sample_ggx(xi, n, roughness);
            let l = normalize(2.0 * dot(v, h) * h - v);

            let n_dot_l = max(dot(n, l), 0.0);
            if (n_dot_l > 0.0) {
              prefiltered_color += textureSampleLevel(environment_map, environment_sampler, l, 0.0).rgb * n_dot_l;
              total_weight += n_dot_l;
            }
          }

          prefiltered_color = prefiltered_color / total_weight;
          return vec4<f32>(prefiltered_color, 1.0);
        }
      ));

  /* Create bind group layout */
  WGPUBindGroupLayout prefilter_bind_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Prefilter bind group layout"),
      .entryCount = 4,
      .entries    = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Fragment,
          .sampler    = {.type = WGPUSamplerBindingType_Filtering},
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Fragment,
          .texture = {
            .sampleType    = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_Cube,
          },
        },
        {
          .binding    = 2,
          .visibility = WGPUShaderStage_Vertex,
          .buffer     = {.type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(mat4)},
        },
        {
          .binding    = 3,
          .visibility = WGPUShaderStage_Fragment,
          .buffer     = {.type = WGPUBufferBindingType_Uniform, .minBindingSize = sizeof(float)},
        },
      },
    });

  /* Create roughness uniform buffer */
  WGPUBuffer roughness_buffer = create_buffer_with_data(
    wgpu_context, &(float){0.0f}, sizeof(float),
    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);

  /* Create bind group */
  WGPUBindGroup prefilter_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Prefilter bind group"),
      .layout     = prefilter_bind_layout,
      .entryCount = 4,
      .entries    = (WGPUBindGroupEntry[]){
        {.binding = 0, .sampler = state.main_sampler},
        {.binding = 1, .textureView = state.cubemap_texture_view},
        {.binding = 2, .buffer = state.cubemap_uniform_buffer, .size = sizeof(mat4)},
        {.binding = 3, .buffer = roughness_buffer, .size = sizeof(float)},
      },
    });

  /* Create pipeline */
  WGPUPipelineLayout prefilter_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Prefilter pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &prefilter_bind_layout,
                          });

  WGPUVertexAttribute prefilter_attr = {
    .shaderLocation = 0,
    .offset         = 0,
    .format         = WGPUVertexFormat_Float32x4,
  };

  WGPURenderPipeline prefilter_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Prefilter pipeline"),
      .layout = prefilter_layout,
      .vertex = {
        .module      = prefilter_shader,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &(WGPUVertexBufferLayout){
          .arrayStride    = 16,
          .stepMode       = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes     = &prefilter_attr,
        },
      },
      .fragment = &(WGPUFragmentState){
        .module      = prefilter_shader,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = WGPUTextureFormat_RGBA16Float,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .depthStencil = NULL,
      .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
    });

  /* Render to each mip level and face */
  for (uint32_t mip = 0; mip < ROUGHNESS_LEVELS; ++mip) {
    float roughness_val = (float)mip / (float)(ROUGHNESS_LEVELS - 1);
    wgpuQueueWriteBuffer(wgpu_context->queue, roughness_buffer, 0,
                         &roughness_val, sizeof(float));

    for (uint32_t face = 0; face < 6; ++face) {
      WGPUTextureView face_view = wgpuTextureCreateView(
        state.prefilter_map, &(WGPUTextureViewDescriptor){
                               .format          = WGPUTextureFormat_RGBA16Float,
                               .dimension       = WGPUTextureViewDimension_2D,
                               .baseMipLevel    = mip,
                               .mipLevelCount   = 1,
                               .baseArrayLayer  = face,
                               .arrayLayerCount = 1,
                             });

      wgpuQueueWriteBuffer(wgpu_context->queue, state.cubemap_uniform_buffer, 0,
                           prefilter_mvp_matrices[face], sizeof(mat4));

      WGPUCommandEncoder encoder
        = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
      WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
        encoder, &(WGPURenderPassDescriptor){
          .colorAttachmentCount = 1,
          .colorAttachments     = &(WGPURenderPassColorAttachment){
            .view       = face_view,
            .loadOp     = WGPULoadOp_Clear,
            .storeOp    = WGPUStoreOp_Store,
            .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
            .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
          },
        });

      wgpuRenderPassEncoderSetPipeline(pass, prefilter_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, prefilter_bind_group, 0, NULL);
      wgpuRenderPassEncoderSetVertexBuffer(
        pass, 0, state.cubemap_vertices_buffer, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDraw(pass, 36, 1, 0, 0);
      wgpuRenderPassEncoderEnd(pass);

      WGPUCommandBuffer command_buffer
        = wgpuCommandEncoderFinish(encoder, NULL);
      wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

      wgpuTextureViewRelease(face_view);
      wgpuCommandBufferRelease(command_buffer);
      wgpuRenderPassEncoderRelease(pass);
      wgpuCommandEncoderRelease(encoder);
    }
  }

  /* Cleanup */
  wgpuBufferRelease(roughness_buffer);
  wgpuRenderPipelineRelease(prefilter_pipeline);
  wgpuPipelineLayoutRelease(prefilter_layout);
  wgpuBindGroupRelease(prefilter_bind_group);
  wgpuBindGroupLayoutRelease(prefilter_bind_layout);
  wgpuShaderModuleRelease(prefilter_shader);
}

/* Generate BRDF lookup table */
static void generate_brdf_lut(wgpu_context_t* wgpu_context)
{
  const uint32_t lut_size = 512;

  /* Create BRDF LUT texture */
  state.brdf_lookup = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("BRDF LUT texture"),
      .usage
      = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment,
      .dimension = WGPUTextureDimension_2D,
      .size = {.width = lut_size, .height = lut_size, .depthOrArrayLayers = 1},
      .format        = WGPUTextureFormat_RG16Float,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });

  state.brdf_lookup_view = wgpuTextureCreateView(state.brdf_lookup, NULL);

  /* Create shader for BRDF convolution */
  WGPUShaderModule brdf_shader = wgpu_create_shader_module(
    wgpu_context->device,
    CODE(
      struct VertexOutput {
        @builtin(position) position : vec4<f32>, @location(0) uv : vec2<f32>,
      }

      @vertex fn vs_main(@builtin(vertex_index) vertex_index : u32)
        ->VertexOutput {
          var output : VertexOutput;
          let x           = f32((vertex_index << 1u) & 2u);
          let y           = f32(vertex_index & 2u);
          output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
          output.uv       = vec2<f32>(x, 1.0 - y);
          return output;
        }

      const PI : f32
      = 3.1415926535897932384626433832795;

      fn radical_inverse_vdc(bits_in : u32)
        ->f32 {
          var bits = bits_in;
          bits     = (bits << 16u) | (bits >> 16u);
          bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
          bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
          bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
          bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
          return f32(bits) * 2.3283064365386963e-10;
        }

      fn hammersley(i : u32, n : u32)
        ->vec2<f32> {
          return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
        }

      fn importance_sample_ggx(xi : vec2<f32>, n : vec3<f32>, roughness : f32)
        ->vec3<f32> {
          let a         = roughness * roughness;
          let phi       = 2.0 * PI * xi.x;
          let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
          let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

          let h
            = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

          let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0),
                          abs(n.z) < 0.999);
          let tangent   = normalize(cross(up, n));
          let bitangent = cross(n, tangent);

          return normalize(tangent * h.x + bitangent * h.y + n * h.z);
        }

      fn geometry_schlick_ggx(n_dot_v : f32, roughness : f32)
        ->f32 {
          let a = roughness;
          let k = (a * a) / 2.0;
          return n_dot_v / (n_dot_v * (1.0 - k) + k);
        }

      fn geometry_smith(n : vec3<f32>, v : vec3<f32>, l : vec3<f32>,
                        roughness : f32)
        ->f32 {
          let n_dot_v = max(dot(n, v), 0.0);
          let n_dot_l = max(dot(n, l), 0.0);
          let ggx2    = geometry_schlick_ggx(n_dot_v, roughness);
          let ggx1    = geometry_schlick_ggx(n_dot_l, roughness);
          return ggx1 * ggx2;
        }

      fn integrate_brdf(n_dot_v : f32, roughness : f32)
        ->vec2<f32> {
          let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
          var a = 0.0;
          var b = 0.0;
          let n = vec3<f32>(0.0, 0.0, 1.0);

          const SAMPLE_COUNT = 1024u;
          for (var i = 0u; i < SAMPLE_COUNT; i++) {
            let xi = hammersley(i, SAMPLE_COUNT);
            let h  = importance_sample_ggx(xi, n, roughness);
            let l  = normalize(2.0 * dot(v, h) * h - v);

            let n_dot_l = max(l.z, 0.0);
            let n_dot_h = max(h.z, 0.0);
            let v_dot_h = max(dot(v, h), 0.0);

            if (n_dot_l > 0.0) {
              let g     = geometry_smith(n, v, l, roughness);
              let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v);
              let fc    = pow(1.0 - v_dot_h, 5.0);

              a += (1.0 - fc) * g_vis;
              b += fc * g_vis;
            }
          }

          return vec2<f32>(a / f32(SAMPLE_COUNT), b / f32(SAMPLE_COUNT));
        }

      @fragment fn fs_main(input : VertexOutput)
        ->@location(0) vec4<f32> {
          let integrated_brdf = integrate_brdf(input.uv.x, 1.0 - input.uv.y);
          return vec4<f32>(integrated_brdf, 0.0, 0.0);
        }));

  /* Create pipeline */
  WGPURenderPipeline brdf_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("BRDF LUT pipeline"),
      .layout = NULL,
      .vertex = {
        .module     = brdf_shader,
        .entryPoint = STRVIEW("vs_main"),
      },
      .fragment = &(WGPUFragmentState){
        .module      = brdf_shader,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = WGPUTextureFormat_RG16Float,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CW,
        .cullMode  = WGPUCullMode_None,
      },
      .depthStencil = NULL,
      .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
    });

  /* Render BRDF LUT */
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
    encoder, &(WGPURenderPassDescriptor){
      .colorAttachmentCount = 1,
      .colorAttachments     = &(WGPURenderPassColorAttachment){
        .view       = state.brdf_lookup_view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      },
    });

  wgpuRenderPassEncoderSetPipeline(pass, brdf_pipeline);
  wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
  wgpuRenderPassEncoderEnd(pass);

  WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(command_buffer);
  wgpuRenderPassEncoderRelease(pass);
  wgpuCommandEncoderRelease(encoder);
  wgpuRenderPipelineRelease(brdf_pipeline);
  wgpuShaderModuleRelease(brdf_shader);
}

/* Stub implementations */
static int init(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera, 0.0f, glm_rad(90.0f), 20.0f);
  init_cubemap_view_matrices();
  init_samplers(wgpu_context);
  init_uniform_buffers(wgpu_context);
  init_render_textures(wgpu_context);

  /* Initialize sokol fetch for async file loading */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 8,
    .num_channels = 2,
    .num_lanes    = 4,
  });

  init_file_loading();
  state.initialized = true;
  return EXIT_SUCCESS;
}

/**
 * @brief Input event handler for camera control
 */
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  UNUSED_VAR(wgpu_context);

  if (input_event->type == INPUT_EVENT_TYPE_MOUSE_DOWN) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      camera_handle_mouse_down(&state.camera, (float)input_event->mouse_x,
                               (float)input_event->mouse_y);
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_UP) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      camera_handle_mouse_up(&state.camera);
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    camera_handle_mouse_move(&state.camera, (float)input_event->mouse_x,
                             (float)input_event->mouse_y);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_SCROLL) {
    camera_handle_mouse_wheel(&state.camera, input_event->scroll_y);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    /* Handle window resize - recreate render targets */
    /* TODO: Implement resize handling */
  }
}

static int frame(wgpu_context_t* wgpu_context)
{
  sfetch_dowork();

  if (!state.obj_loaded || !state.hdr_loaded) {
    return EXIT_SUCCESS;
  }

  /* Create pipelines and IBL textures after files are loaded */
  if (!state.pipelines_created) {
    create_obj_vertex_buffer(wgpu_context);
    convert_equirectangular_to_cubemap(wgpu_context);
    generate_irradiance_map(wgpu_context);
    generate_prefilter_map(wgpu_context);
    generate_brdf_lut(wgpu_context);
    init_pipelines(wgpu_context);
    create_bind_groups(wgpu_context);
    state.pipelines_created = true;
  }

  if (!state.pipelines_created) {
    return EXIT_SUCCESS;
  }

  /* Update camera and matrices */
  vec3 camera_position;
  camera_get_position(&state.camera, camera_position);

  mat4 view, projection;
  camera_get_view(&state.camera, view);
  /* Note: glm_lookat already returns the view matrix (inverse of camera matrix)
   * TypeScript Mat4.lookAt returns the camera matrix, then inverts it.
   * So no inversion needed here. */

  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;
  glm_perspective(glm_rad(45.0f), aspect_ratio, 0.1f, 100.0f, projection);

  mat4 view_projection;
  /* CGLM is column-major: VP = projection * view */
  glm_mat4_mul(projection, view, view_projection);

  /* Update PBR uniforms */
  float camera_uniform[4]
    = {camera_position[0], camera_position[1], camera_position[2], 0.0f};
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       camera_uniform, sizeof(camera_uniform));

  /* Update view projection for PBR */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.view_projection_buffer, 0,
                       view_projection, sizeof(mat4));

  /* Update skybox uniforms (view + projection) */
  mat4 skybox_uniforms[2];
  glm_mat4_copy(view, skybox_uniforms[0]);
  glm_mat4_copy(projection, skybox_uniforms[1]);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.skybox_uniform_buffer, 0,
                       skybox_uniforms, sizeof(skybox_uniforms));

  /* Update instance matrices */
  mat4 matrices[COUNT_X * COUNT_Y];
  const float distance = 2.8f;
  uint32_t idx         = 0;
  for (uint32_t y = 0; y < COUNT_Y; ++y) {
    for (uint32_t x = 0; x < COUNT_X; ++x) {
      float pos_x = (float)x * distance - (distance * (COUNT_X - 1)) / 2.0f;
      float pos_y = (float)y * distance - distance / 2.0f;
      glm_mat4_identity(matrices[idx]);
      glm_translate(matrices[idx], (vec3){pos_x, pos_y, 0.0f});
      idx++;
    }
  }
  wgpuQueueWriteBuffer(wgpu_context->queue, state.matrix_buffer, 0, matrices,
                       sizeof(matrices));

  /* Begin render pass */
  state.color_attachment.view          = state.color_texture_view;
  state.color_attachment.resolveTarget = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view  = state.depth_texture_view;

  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(encoder, &state.render_pass_descriptor);

  /* Render PBR objects */
  wgpuRenderPassEncoderSetPipeline(pass, state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.uniform_bind_group, 0, NULL);
  wgpuRenderPassEncoderSetBindGroup(pass, 1, state.matrix_bind_group, 0, NULL);
  wgpuRenderPassEncoderSetBindGroup(pass, 2, state.texture_bind_group, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.position_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(pass, state.obj.face_count * 3, COUNT_X * COUNT_Y,
                            0, 0);

  /* Render skybox */
  wgpuRenderPassEncoderSetPipeline(pass, state.skybox_pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.skybox_uniform_bind_group, 0,
                                    NULL);
  wgpuRenderPassEncoderSetBindGroup(pass, 1, state.cubemap_uniform_bind_group,
                                    0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.cubemap_vertices_buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(pass, 36, 1, 0, 0);

  wgpuRenderPassEncoderEnd(pass);
  WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  wgpuCommandBufferRelease(command_buffer);
  wgpuRenderPassEncoderRelease(pass);
  wgpuCommandEncoderRelease(encoder);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  obj_data_cleanup(&state.obj);

  /* Free HDR data */
  if (state.hdr.data) {
    free(state.hdr.data);
    state.hdr.data = NULL;
  }

  WGPU_RELEASE_RESOURCE(Buffer, state.position_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.matrix_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.lights_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cubemap_vertices_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cubemap_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.view_projection_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.skybox_uniform_buffer)

  WGPU_RELEASE_RESOURCE(Texture, state.cubemap_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.cubemap_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, state.irradiance_map)
  WGPU_RELEASE_RESOURCE(TextureView, state.irradiance_map_view)
  WGPU_RELEASE_RESOURCE(Texture, state.prefilter_map)
  WGPU_RELEASE_RESOURCE(TextureView, state.prefilter_map_view)
  WGPU_RELEASE_RESOURCE(Texture, state.brdf_lookup)
  WGPU_RELEASE_RESOURCE(TextureView, state.brdf_lookup_view)
  WGPU_RELEASE_RESOURCE(Texture, state.color_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.color_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_texture_view)

  /* Release bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.uniform_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.matrix_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.texture_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.skybox_uniform_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.cubemap_uniform_bind_group)

  /* Release pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.skybox_pipeline)

  /* Release samplers */
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler_brdf)
}
