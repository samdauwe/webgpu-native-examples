#include "texture.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../core/file.h"
#include "../core/log.h"
#include "../core/macro.h"
#include "shader.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough="
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
#include <ktx.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#define STB_IMAGE_IMPLEMENTATION
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <stb_image.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

void wgpu_image_to_texure(wgpu_context_t* wgpu_context,
                          texture_image_desc_t* desc)
{
  const uint64_t data_size
    = desc->width * desc->height * desc->channels * sizeof(uint8_t);
  wgpuQueueWriteTexture(wgpu_context->queue,
    &(WGPUTextureCopyView) {
      .texture = desc->texture,
      .mipLevel = 0,
      .origin = (WGPUOrigin3D) {
        .x = 0,
        .y = 0,
        .z = 0,
    },
    .aspect = WGPUTextureAspect_All,
    },
    desc->pixels, data_size,
    &(WGPUTextureDataLayout){
      .offset      = 0,
      .bytesPerRow = desc->width * desc->channels * sizeof(uint8_t),
      .rowsPerImage = desc->height,
    },
    &(WGPUExtent3D){
      .width  = desc->width,
      .height = desc->height,
      .depth  = 1,
    });
}

ktxResult load_ktx_file(const char* filename, ktxTexture** target)
{
  ktxResult result = KTX_SUCCESS;
  if (!file_exists(filename)) {
    log_fatal("Could not load texture from %s", filename);
  }
  result = ktxTexture_CreateFromNamedFile(
    filename, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, target);
  return result;
}

static bool is_power_of_2(int value)
{
  return (value & (value - 1)) == 0;
}

static int make_multiple_of_256(int value)
{
  int resized_width;
  if (value % 256 == 0) {
    resized_width = value;
  }
  else {
    resized_width = (value / 256 + 1) * 256;
  }
  return resized_width;
}

static void copy_padding_buffer(unsigned char* dst, unsigned char* src,
                                int width, int height, int padding)
{
  unsigned char* s = src;
  unsigned char* d = dst;
  for (int i = 0; i < height; ++i) {
    memcpy(d, s, width * 4);
    s += width * 4;
    d += padding * 4;
  }
}

/**
 * @brief Determines the number of mip levels needed for a full mip chain given
 * the width and height of texture level 0.
 * @param width width of texture level 0.
 * @param height height of texture level 0.
 * @return Ideal number of mip levels.
 */
static uint32_t calculate_mip_level_count(int width, int height)
{
  return (uint32_t)(floor((float)(log2(MIN(width, height))))) + 1;
}

static void destroy_image_data(uint8_t** pixel_vec, int width, int height)
{
  uint32_t mipmap_level = (uint32_t)(floor(log2(MAX(width, height)))) + 1;
  for (uint32_t i = 0; i < mipmap_level; ++i) {
    free(pixel_vec[i]);
  }
  free(pixel_vec);
}

static void generate_mipmap(uint8_t* input_pixels, int input_w, int input_h,
                            int input_stride_in_bytes, uint8_t*** output_pixels,
                            int output_w, int output_h,
                            int output_stride_in_bytes, int num_channels,
                            bool is_256_padding)
{
  uint32_t mipmap_level = (uint32_t)(floor(log2(MAX(output_w, output_h)))) + 1;
  uint8_t** mipmap_pixels
    = (unsigned char**)malloc(mipmap_level * sizeof(unsigned char**));
  *output_pixels = mipmap_pixels;
  int height     = output_h;
  int width      = output_w;

  if (!is_256_padding) {
    for (uint32_t i = 0; i < mipmap_level; ++i) {
      mipmap_pixels[i]
        = (unsigned char*)malloc(output_w * height * 4 * sizeof(char));
      stbir_resize_uint8(input_pixels, input_w, input_h, input_stride_in_bytes,
                         mipmap_pixels[i], width, height,
                         output_stride_in_bytes, num_channels);

      height >>= 1;
      width >>= 1;
      if (height == 0) {
        height = 1;
      }
    }
  }
  else {
    uint8_t* pixels
      = (unsigned char*)malloc(output_w * height * 4 * sizeof(char));

    for (uint32_t i = 0; i < mipmap_level; ++i) {
      mipmap_pixels[i]
        = (unsigned char*)malloc(output_w * height * 4 * sizeof(char));
      stbir_resize_uint8(input_pixels, input_w, input_h, input_stride_in_bytes,
                         pixels, width, height, output_stride_in_bytes,
                         num_channels);
      copy_padding_buffer(mipmap_pixels[i], pixels, width, height, output_w);

      height >>= 1;
      width >>= 1;
      if (height == 0) {
        height = 1;
      }
    }
    free(pixels);
  }
}

texture_t wgpu_texture_load_from_ktx_file(wgpu_context_t* wgpu_context,
                                          const char* filename)
{
  texture_t texture = {0};
  texture.format    = WGPUTextureFormat_RGBA8Unorm;

  ktxTexture* ktx_texture;
  ktxResult result = load_ktx_file(filename, &ktx_texture);
  assert(result == KTX_SUCCESS);

  ktx_uint8_t* ktx_texture_data = ktxTexture_GetData(ktx_texture);

  // WebGPU requires that the bytes per row is a multiple of 256
  uint32_t resized_width = make_multiple_of_256(ktx_texture->baseWidth);

  // Get properties required for using and upload texture data from the ktx
  // texture object
  texture.size.width
    = ktx_texture->isCubemap ? ktx_texture->baseWidth : resized_width;
  texture.size.height = ktx_texture->baseHeight;
  texture.size.depth  = ktx_texture->isCubemap ? 6u : 1u;
  texture.mip_level_count
    = ktx_texture->isCubemap ?
        1u :
        calculate_mip_level_count(ktx_texture->baseWidth,
                                  ktx_texture->baseHeight);

  WGPUTextureDescriptor texture_desc = {
    .size          = (WGPUExtent3D) {
      .width  = texture.size.width,
      .height = texture.size.height,
      .depth  = texture.size.depth,
     },
    .mipLevelCount = texture.mip_level_count,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = texture.format,
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_Sampled,
  };
  texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  if (ktx_texture->isCubemap) {
    // Create a host-visible staging buffer that contains the raw image data
    ktx_size_t ktx_texture_size              = ktxTexture_GetSize(ktx_texture);
    WGPUBufferDescriptor staging_buffer_desc = {
      .usage            = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
      .size             = ktx_texture_size,
      .mappedAtCreation = true,
    };
    WGPUBuffer staging_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &staging_buffer_desc);
    ASSERT(staging_buffer)

    // Copy texture data into staging buffer
    void* mapping
      = wgpuBufferGetMappedRange(staging_buffer, 0, ktx_texture_size);
    ASSERT(mapping)
    memcpy(mapping, ktx_texture_data, ktx_texture_size);
    wgpuBufferUnmap(staging_buffer);

    for (uint32_t face = 0; face < texture.size.depth; ++face) {
      for (uint32_t level = 0; level < texture.mip_level_count; ++level) {
        uint32_t width  = ktx_texture->baseWidth >> level;
        uint32_t height = ktx_texture->baseHeight >> level;

        ktx_size_t offset;
        KTX_error_code result
          = ktxTexture_GetImageOffset(ktx_texture, level, 0, face, &offset);
        assert(result == KTX_SUCCESS);

        // Upload statging buffer to texture
        wgpuCommandEncoderCopyBufferToTexture(cmd_encoder,
          // Source
          &(WGPUBufferCopyView) {
            .buffer = staging_buffer,
            .layout = (WGPUTextureDataLayout) {
              .offset = offset,
              .bytesPerRow = width * 4,
              .rowsPerImage= height,
            },
          },
          // Destination
          &(WGPUTextureCopyView){
            .texture = texture.texture,
            .mipLevel = level,
            .origin = (WGPUOrigin3D) {
              .x=0,
              .y=0,
              .z=face,
            },
            .aspect = WGPUTextureAspect_All,
          },
          // Copy size
          &(WGPUExtent3D){
            .width  = MAX(1u, width),
            .height = MAX(1u, height),
            .depth  = 1,
          });
      }
    }

    WGPU_RELEASE_RESOURCE(Buffer, staging_buffer);
  }
  else {
    // Generate Mipmap
    uint8_t** resized_vec = NULL;
    generate_mipmap(ktx_texture_data, ktx_texture->baseWidth,
                    ktx_texture->baseHeight, 0, &resized_vec, resized_width,
                    ktx_texture->baseHeight, 0, 4, true);
    fflush(stdout);

    // Setup buffer copy regions for each face including all of its mip levels
    // and copy the texture regions from the staging buffer into the texture
    for (uint32_t face = 0; face < texture.size.depth; ++face) {
      for (uint32_t level = 0; level < texture.mip_level_count; ++level) {
        uint32_t width  = texture.size.width >> level;
        uint32_t height = texture.size.height >> level;
        if (height == 0) {
          height = 1;
        }

        // Create a host-visible staging buffer that contains the raw image data
        size_t ktx_texture_size = texture.size.width * height * 4;
        WGPUBufferDescriptor staging_buffer_desc = {
          .usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
          .size  = ktx_texture_size,
          .mappedAtCreation = true,
        };
        WGPUBuffer staging_buffer
          = wgpuDeviceCreateBuffer(wgpu_context->device, &staging_buffer_desc);
        ASSERT(staging_buffer)

        // Copy texture data into staging buffer
        void* mapping
          = wgpuBufferGetMappedRange(staging_buffer, 0, ktx_texture_size);
        ASSERT(mapping)
        memcpy(mapping, resized_vec[level], ktx_texture_size);
        wgpuBufferUnmap(staging_buffer);

        // Upload statging buffer to texture
        wgpuCommandEncoderCopyBufferToTexture(cmd_encoder,
          // Source
          &(WGPUBufferCopyView) {
            .buffer = staging_buffer,
            .layout = (WGPUTextureDataLayout) {
              .offset = 0,
              .bytesPerRow = texture.size.width * 4,
              .rowsPerImage= height,
            },
          },
          // Destination
          &(WGPUTextureCopyView){
            .texture = texture.texture,
            .mipLevel = level,
            .origin = (WGPUOrigin3D) {
              .x=0,
              .y=0,
              .z=face,
            },
            .aspect = WGPUTextureAspect_All,
          },
          // Copy size
          &(WGPUExtent3D){
            .width  = MAX(1u, width),
            .height = MAX(1u, height),
            .depth  = 1,
          });

        WGPU_RELEASE_RESOURCE(Buffer, staging_buffer);
      }
    }
    // Free image data after upload to GPU
    destroy_image_data(resized_vec, resized_width, ktx_texture->baseHeight);
  }

  WGPUCommandBuffer command_buffer
    = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  // Sumbit commmand buffer and cleanup
  ASSERT(command_buffer != NULL)

  // Submit to the queue
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  // Release command buffer
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

  // Create texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .format          = texture.format,
    .dimension       = ktx_texture->isCubemap ? WGPUTextureViewDimension_Cube :
                                                WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = texture.mip_level_count,
    .baseArrayLayer  = 0,
    .arrayLayerCount = texture.size.depth, // Cube faces count as array layers
  };
  texture.view = wgpuTextureCreateView(texture.texture, &texture_view_dec);

  bool is_size_power_of_2
    = is_power_of_2(texture.size.width) && is_power_of_2(texture.size.height);
  WGPUFilterMode mipmapFilter = is_size_power_of_2 && !ktx_texture->isCubemap ?
                                  WGPUFilterMode_Linear :
                                  WGPUFilterMode_Nearest;

  // Create sampler
  WGPUSamplerDescriptor sampler_desc = {
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = mipmapFilter,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = (float)texture.mip_level_count,
    .maxAnisotropy = 1,
  };
  texture.sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);

  // Clean up staging resources
  ktxTexture_Destroy(ktx_texture);

  return texture;
}

texture_t wgpu_texture_load_with_stb(wgpu_context_t* wgpu_context,
                                     const char* filename)
{
  texture_t texture = {0};
  texture.format    = WGPUTextureFormat_RGBA8Unorm;

  const uint8_t comp_map[5] = {
    0, //
    1, //
    2, //
    4, //
    4  //
  };
  const uint32_t channels[5] = {
    STBI_default,    // only used for req_comp
    STBI_grey,       //
    STBI_grey_alpha, //
    STBI_rgb_alpha,  //
    STBI_rgb_alpha   //
  };

  int width  = 0;
  int height = 0;
  // Force loading 3 channel images to 4 channel by stb becasue Dawn doesn't
  // support 3 channel formats currently. The group is discussing on whether
  // webgpu shoud support 3 channel format.
  // https://github.com/gpuweb/gpuweb/issues/66#issuecomment-410021505
  int read_comps = 4;
  stbi_set_flip_vertically_on_load(false);
  stbi_uc* pixel_data = stbi_load(filename,            //
                                  &width,              //
                                  &height,             //
                                  &read_comps,         //
                                  channels[read_comps] //
  );
  if (!pixel_data) {
    log_debug("Couldn't load '%s'\n", filename);
  }
  ASSERT(pixel_data);
  uint8_t comps = comp_map[read_comps];
  log_debug("Loaded image %s (%d, %d, %d / %d)\n", filename, width, height,
            read_comps, comps);

  texture.size.width  = width;
  texture.size.height = height;
  texture.size.depth  = 1;
  texture.channels    = read_comps;

  WGPUExtent3D texture_size = {
    .width  = texture.size.width,
    .height = texture.size.height,
    .depth  = texture.size.depth,
  };

  WGPUTextureDescriptor tex_desc = {
    .usage         = WGPUTextureUsage_Sampled | WGPUTextureUsage_CopyDst,
    .dimension     = WGPUTextureDimension_2D,
    .size          = texture_size,
    .format        = texture.format,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };

  texture.texture = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .format          = texture.format,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  texture.view = wgpuTextureCreateView(texture.texture, &texture_view_dec);

  // Copy pixel data to texture
  wgpu_image_to_texure(wgpu_context, &(texture_image_desc_t){
                                       .width    = width,
                                       .height   = height,
                                       .channels = 4u,
                                       .pixels   = pixel_data,
                                       .texture  = texture.texture,
                                     });
  stbi_image_free(pixel_data);

  // Create the sampler
  WGPUSamplerDescriptor sampler_desc = {
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUFilterMode_Nearest,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  texture.sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);

  return texture;
}

void wgpu_destroy_texture(texture_t* texture)
{
  WGPU_RELEASE_RESOURCE(TextureView, texture->view)
  WGPU_RELEASE_RESOURCE(Texture, texture->texture)
  WGPU_RELEASE_RESOURCE(Sampler, texture->sampler)
}
