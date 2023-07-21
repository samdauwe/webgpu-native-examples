#ifndef GLTF_MODEL_H
#define GLTF_MODEL_H

#include <stdint.h>

#include <cglm/cglm.h>

#include "api.h"

struct gltf_model_t;
struct wgpu_context_t;

// Changing this value here also requires changing it in the vertex shader
#define WGPU_GLTF_MAX_NUM_JOINTS 128u

#define WGPU_GLTF_VERTATTR_DESC(l, c)                                          \
  wgpu_gltf_get_vertex_attribute_description(l, c)

#define WGPU_GLTF_VERTEX_BUFFER_LAYOUT(name, ...)                              \
  uint64_t array_stride                       = wgpu_gltf_get_vertex_size();   \
  WGPUVertexAttribute vert_attr_desc_##name[] = {__VA_ARGS__};                 \
  WGPUVertexBufferLayout name##_vertex_buffer_layout                           \
    = WGPU_VERTBUFFERLAYOUT_DESC(array_stride, vert_attr_desc_##name);

/*
 * glTF model loading options
 */
typedef enum wgpu_gltf_file_loading_flags_enum_t {
  WGPU_GLTF_FileLoadingFlags_None                    = 0x00000000,
  WGPU_GLTF_FileLoadingFlags_PreTransformVertices    = 0x00000001,
  WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors = 0x00000002,
  WGPU_GLTF_FileLoadingFlags_FlipY                   = 0x00000004,
  WGPU_GLTF_FileLoadingFlags_DontLoadImages          = 0x00000008
} wgpu_gltf_file_loading_flags_enum_t;

/*
 * glTF model render options
 */
typedef enum wgpu_gltf_render_flags_enum_t {
  WGPU_GLTF_RenderFlags_BindImages              = 0x00000001,
  WGPU_GLTF_RenderFlags_RenderOpaqueNodes       = 0x00000002,
  WGPU_GLTF_RenderFlags_RenderAlphaMaskedNodes  = 0x00000004,
  WGPU_GLTF_RenderFlags_RenderAlphaBlendedNodes = 0x00000008
} wgpu_gltf_render_flags_enum_t;

/*
 * glTF default vertex layout with easy WebGPU mapping functions
 */
typedef enum wgpu_gltf_vertex_component_enum_t {
  WGPU_GLTF_VertexComponent_Position = 0,
  WGPU_GLTF_VertexComponent_Normal   = 1,
  WGPU_GLTF_VertexComponent_UV       = 2,
  WGPU_GLTF_VertexComponent_Color    = 3,
  WGPU_GLTF_VertexComponent_Tangent  = 4,
  WGPU_GLTF_VertexComponent_Joint0   = 5,
  WGPU_GLTF_VertexComponent_Weight0  = 6,
} wgpu_gltf_vertex_component_enum_t;

typedef enum wgpu_gltf_alpha_mode_enum_t {
  AlphaMode_OPAQUE = 0,
  AlphaMode_MASK   = 1,
  AlphaMode_BLEND  = 2,
} wgpu_gltf_alpha_mode_enum_t;

/*
 * glTF texture
 */
typedef struct wgpu_gltf_texture_t {
  wgpu_context_t* wgpu_context;
  texture_t wgpu_texture;
} wgpu_gltf_texture_t;

/*
 * glTF material
 */
typedef struct wgpu_gltf_material_t {
  wgpu_context_t* wgpu_context;
  wgpu_gltf_alpha_mode_enum_t alpha_mode;
  bool blend;
  bool double_sided;
  float alpha_cutoff;
  float metallic_factor;
  float roughness_factor;
  vec4 base_color_factor;
  vec4 emissive_factor;
  wgpu_gltf_texture_t* base_color_texture;
  wgpu_gltf_texture_t* metallic_roughness_texture;
  wgpu_gltf_texture_t* normal_texture;
  wgpu_gltf_texture_t* occlusion_texture;
  wgpu_gltf_texture_t* emissive_texture;
  struct {
    uint8_t base_color;
    uint8_t metallic_roughness;
    uint8_t specular_glossiness;
    uint8_t normal;
    uint8_t occlusion;
    uint8_t emissive;
  } tex_coord_sets;
  struct {
    wgpu_gltf_texture_t* specular_glossiness_texture;
    wgpu_gltf_texture_t* diffuse_texture;
    vec4 diffuse_factor;
    vec3 specular_factor;
  } extension;
  struct {
    bool metallic_roughness;
    bool specular_glossiness;
  } pbr_workflows;
  WGPUBindGroup bind_group;
  WGPURenderPipeline pipeline;
} wgpu_gltf_material_t;

typedef struct wgpu_gltf_materials_t {
  wgpu_gltf_material_t* materials;
  uint32_t material_count;
} wgpu_gltf_materials_t;

/**
 * @brief glTF model load options
 */
typedef struct wgpu_gltf_model_load_options_t {
  struct wgpu_context_t* wgpu_context;
  const char* filename;
  uint32_t file_loading_flags;
  float scale;
} wgpu_gltf_model_load_options_t;

/**
 * @brief glTF model creation/destruction
 */
struct gltf_model_t* wgpu_gltf_model_load_from_file(
  struct wgpu_gltf_model_load_options_t* load_options);
void wgpu_gltf_model_destroy(struct gltf_model_t* model);

/**
 * @brief Returns the vertex attribute description for the given shader location
 * and component.
 */
WGPUVertexAttribute wgpu_gltf_get_vertex_attribute_description(
  uint32_t shader_location, wgpu_gltf_vertex_component_enum_t component);

/** glTF helper functions */
uint64_t wgpu_gltf_get_vertex_size(void);
wgpu_gltf_materials_t wgpu_gltf_model_get_materials(void* model);
void wgpu_gltf_model_prepare_nodes_bind_group(
  struct gltf_model_t* model, WGPUBindGroupLayout bind_group_layout);
void wgpu_gltf_model_prepare_skins_bind_group(
  struct gltf_model_t* model, WGPUBindGroupLayout bind_group_layout);

/**
 *  @brief glTF model rendering
 */
typedef struct wgpu_gltf_model_render_options_t {
  uint32_t render_flags;
  uint32_t bind_mesh_model_set;
  uint32_t bind_image_set;
} wgpu_gltf_model_render_options_t;
void wgpu_gltf_model_draw(struct gltf_model_t* model,
                          wgpu_gltf_model_render_options_t render_options);
void gltf_model_update_animation(struct gltf_model_t* model, uint32_t index,
                                 float time);

#endif
