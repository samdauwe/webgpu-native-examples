#ifndef TEXTURE_H
#define TEXTURE_H

#include "context.h"

typedef enum color_space_enum_t {
  COLOR_SPACE_UNDEFINED,
  COLOR_SPACE_SRGB,
  COLOR_SPACE_LINEAR,
} color_space_enum_t;

typedef struct texture_t {
  WGPUExtent3D size;
  uint32_t mip_level_count;
  WGPUTextureFormat format;
  WGPUTextureDimension dimension;
  WGPUTexture texture;
  WGPUTextureView view;
  WGPUSampler sampler;
} texture_t;

/* -------------------------------------------------------------------------- *
 * Helper functions
 * -------------------------------------------------------------------------- */

/* Copy image to texture */
void wgpu_image_to_texure(wgpu_context_t* wgpu_context, WGPUTexture texture,
                          void* pixels, WGPUExtent3D size, uint32_t channels);

/* Texture destruction */
void wgpu_destroy_texture(texture_t* texture);

/* -------------------------------------------------------------------------- *
 * WebGPU Mipmap Generator
 * @ref:
 * https://github.com/toji/webgpu-gltf-case-study/blob/main/samples/js/webgpu-mipmap-generator.js
 * -------------------------------------------------------------------------- */

/* Mip map generator */
typedef struct wgpu_mipmap_generator wgpu_mipmap_generator_t;

/* Mip map generator construction / destruction */
wgpu_mipmap_generator_t*
wgpu_mipmap_generator_create(wgpu_context_t* wgpu_context);
void wgpu_mipmap_generator_destroy(wgpu_mipmap_generator_t* mipmap_generator);

/* Mip map generator factory function */
WGPURenderPipeline wgpu_mipmap_generator_get_mipmap_pipeline(
  wgpu_mipmap_generator_t* mipmap_generator, WGPUTextureFormat format);

/**
 * @brief Generates mipmaps for the given GPUTexture from the data in level 0.
 *
 * @param {wgpu_mipmap_generator_t*} mipmap_generator - The mip map generator.
 * @param {WGPUTexture*} texture - Texture to generate mipmaps for.
 * @param {WGPUTextureDescriptor*} texture_desc - the texture description was
 * created with.
 * @returns {WGPUTexture} - The originally passed texture
 */
WGPUTexture
wgpu_mipmap_generator_generate_mipmap(wgpu_mipmap_generator_t* mipmap_generator,
                                      WGPUTexture texture,
                                      WGPUTextureDescriptor* texture_desc);

/* -------------------------------------------------------------------------- *
 * WebGPU Texture Client
 * -------------------------------------------------------------------------- */

typedef struct wgpu_texture_client_t {
  wgpu_context_t* wgpu_context;
  wgpu_mipmap_generator_t* wgpu_mipmap_generator;
  bool allow_compressed_formats;
  struct {
    WGPUTextureFormat values[4];
    size_t count;
  } uncompressed_format_list;
  struct {
    WGPUTextureFormat values[12];
    size_t count;
  } supported_format_list;
} wgpu_texture_client;

typedef struct wgpu_texture_load_options_t {
  const char* label;
  bool flip_y;
  bool generate_mipmaps;
  WGPUTextureUsage usage;
  WGPUTextureFormat format;
  WGPUAddressMode address_mode;
  color_space_enum_t color_space;
} wgpu_texture_load_options;

/* Texture client construction / destruction */
struct wgpu_texture_client_t*
wgpu_texture_client_create(wgpu_context_t* wgpu_context);
void wgpu_texture_client_destroy(struct wgpu_texture_client_t* texture_client);

/* Texture client functions*/

/**
 * @brief Returns a list of the WebGPU texture formats that this client can
 * support.
 * @param supported_formats output pointer to assign
 * @param count number of supported formats
 */
void get_supported_formats(struct wgpu_texture_client_t* texture_client,
                           WGPUTextureFormat* supported_formats, size_t* count);

/* -------------------------------------------------------------------------- *
 * Texture creation functions
 * -------------------------------------------------------------------------- */

/* Texture creation from memory */
texture_t
wgpu_create_texture_from_memory(wgpu_context_t* wgpu_context, void* data,
                                size_t data_size,
                                struct wgpu_texture_load_options_t* options);

/* Texture creation from file */
texture_t
wgpu_create_texture_from_file(wgpu_context_t* wgpu_context,
                              const char* filename,
                              struct wgpu_texture_load_options_t* options);

/* Texture cubemap creation from 6 individual image files */
texture_t wgpu_create_texture_cubemap_from_files(
  wgpu_context_t* wgpu_context, const char* filenames[6],
  struct wgpu_texture_load_options_t* options);

/* Texture creation with dimension 1x1 */
texture_t wgpu_create_empty_texture(wgpu_context_t* wgpu_context);

#endif /* TEXTURE_H */
