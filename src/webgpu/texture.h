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
                                     const char* filename);

/* Texture destruction */
void wgpu_destroy_texture(texture_t* texture);

#endif
