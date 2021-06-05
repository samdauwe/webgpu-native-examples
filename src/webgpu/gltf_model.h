#ifndef GLTF_MODEL_H
#define GLTF_MODEL_H

#include <stdint.h>

#include "api.h"

struct gltf_model_t;
struct wgpu_context_t;

#define WGPU_GLTF_VERTATTR_DESC(l, c)                                          \
  gltf_get_vertex_attribute_description(l, c)

#define WGPU_GLTF_VERTEX_BUFFER_LAYOUT(name, ...)                              \
  uint64_t array_stride                       = get_gltf_vertex_size();        \
  WGPUVertexAttribute vert_attr_desc_##name[] = {__VA_ARGS__};                 \
  WGPUVertexBufferLayout name##_vertex_buffer_layout                           \
    = WGPU_VERTBUFFERLAYOUT_DESC(array_stride, vert_attr_desc_##name);

/*
 * glTF model loading options
 */
typedef enum wgpu_gltf_file_loading_flags_enum {
  WGPU_GLTF_FileLoadingFlags_None                    = 0x00000000,
  WGPU_GLTF_FileLoadingFlags_PreTransformVertices    = 0x00000001,
  WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors = 0x00000002,
  WGPU_GLTF_FileLoadingFlags_FlipY                   = 0x00000004,
  WGPU_GLTF_FileLoadingFlags_DontLoadImages          = 0x00000008
} wgpu_gltf_file_loading_flags_enum;

/*
 * glTF default vertex layout with easy WebGPU mapping functions
 */
typedef enum wgpu_gltf_vertex_component_enum {
  WGPU_GLTF_VertexComponent_Position = 0,
  WGPU_GLTF_VertexComponent_Normal   = 1,
  WGPU_GLTF_VertexComponent_UV       = 2,
  WGPU_GLTF_VertexComponent_Color    = 3,
  WGPU_GLTF_VertexComponent_Tangent  = 4,
  WGPU_GLTF_VertexComponent_Joint0   = 5,
  WGPU_GLTF_VertexComponent_Weight0  = 6,
} wgpu_gltf_vertex_component_enum;

typedef struct wgpu_gltf_model_load_options_t {
  struct wgpu_context_t* wgpu_context;
  const char* filename;
  uint32_t file_loading_flags;
  float scale;
} wgpu_gltf_model_load_options_t;

/**
 *  @brief glTF model creation/destruction
 */
struct gltf_model_t* wgpu_gltf_model_load_from_file(
  struct wgpu_gltf_model_load_options_t* load_options);
void wgpu_gltf_model_destroy(struct gltf_model_t* model);

/**
 * @brief Returns the vertex attribute description for the given shader location
 * and component.
 */
WGPUVertexAttribute gltf_get_vertex_attribute_description(
  uint32_t shader_location, wgpu_gltf_vertex_component_enum component);

/**
 * @brief Returns the size of a glTF Vertex.
 */
uint64_t get_gltf_vertex_size();

/**
 *  @brief glTF model rendering
 */
void wgpu_gltf_model_draw(struct gltf_model_t* model, uint32_t render_flags,
                          uint32_t bind_image_set);

#endif
