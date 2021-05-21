#ifndef TEXTURE_H
#define TEXTURE_H

#include "context.h"

typedef struct texture_t {
  struct {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
  } size;
  uint32_t channels;
  uint8_t* pixels;
  WGPUTexture texture;
  WGPUTextureView view;
  WGPUSampler sampler;
  bool generate_mipmaps;
  uint32_t mip_level_count;
  WGPUTextureFormat format;
} texture_t;

typedef struct texture_image_desc_t {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  WGPUTextureDimension dimension;
  WGPUTextureUsage usage;
  uint32_t channels;
  uint8_t* pixels;
  WGPUTexture texture;
  bool generate_mipmaps;
  uint32_t mip_level_count;
  WGPUTextureFormat format;
} texture_image_desc_t;

/* Copy image to texture */
void wgpu_image_to_texure(wgpu_context_t* wgpu_context,
                          texture_image_desc_t* desc);

/* KTX file loading */
texture_t wgpu_texture_load_from_ktx_file(wgpu_context_t* wgpu_context,
                                          const char* filename);

/* Image loading using stb */
texture_t wgpu_texture_load_with_stb(wgpu_context_t* wgpu_context,
                                     const char* filename,
                                     WGPUTextureUsageFlags texture_usage_flags);

/* Texture destruction */
void wgpu_destroy_texture(texture_t* texture);

/* Mip map generator */
typedef struct wgpu_mipmap_generator wgpu_mipmap_generator_t;

/* Mip map generator construction / destruction */
wgpu_mipmap_generator_t*
wgpu_mipmap_generator_create(wgpu_context_t* wgpu_context);
void wgpu_mipmap_generator_destroy(wgpu_mipmap_generator_t* mipmap_generator);

WGPURenderPipeline wgpu_mipmap_generator_get_mipmap_pipeline(
  wgpu_mipmap_generator_t* mipmap_generator, WGPUTextureFormat format);

/**
 * @brief Generates mipmaps for the given GPUTexture from the data in level 0.
 *
 * @param {wgpu_mipmap_generator_t*} mipmap_generator - The mip map generator.
 * @param {texture_image_desc_t*} texture_desc - the texture description was
 * created with.
 * @returns {WGPUTexture} - The originally passed texture
 */
WGPUTexture
wgpu_mipmap_generator_generate_mipmap(wgpu_mipmap_generator_t* mipmap_generator,
                                      texture_image_desc_t* texture_desc);

#endif
