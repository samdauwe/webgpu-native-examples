#ifndef TEXTURE_H
#define TEXTURE_H

#include "context.h"

typedef struct texture_t {
  struct {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
  } size;
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

/* Texture creation with dimension 1x1 */
texture_t wgpu_create_empty_texture(wgpu_context_t* wgpu_context);

/* Texture destruction */
void wgpu_destroy_texture(texture_t* texture);

/* -------------------------------------------------------------------------- *
 * WebGPU Mipmap Generator
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
} wgpu_texture_client;

typedef struct wgpu_texture_load_options_t {
  bool generate_mipmaps;
  WGPUTextureUsage usage;
  WGPUTextureFormat format;
  WGPUAddressMode address_mode;
} wgpu_texture_load_options;

/* Texture client construction / destruction */
struct wgpu_texture_client_t*
wgpu_texture_client_create(wgpu_context_t* wgpu_context);
void wgpu_texture_client_destroy(struct wgpu_texture_client_t* texture_client);

/* -------------------------------------------------------------------------- *
 * Helper functions
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

#endif
