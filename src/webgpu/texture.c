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
    &(WGPUImageCopyTexture) {
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
      .width               = desc->width,
      .height              = desc->height,
      .depth               = 1,
      .depthOrArrayLayers  = 1,
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

/**
 * @brief Determines if the given value is a power of two.
 *
 * @param {int} n - Number to evaluate.
 * @returns {bool} - True if the number is a power of two.
 */
static bool is_power_of_2(int n)
{
  return (n & (n - 1)) == 0;
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
 *
 * @param {int} width width of texture level 0.
 * @param {int} height height of texture level 0.
 * @return {uint32_t} Ideal number of mip levels.
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
      .width               = texture.size.width,
      .height              = texture.size.height,
      .depth               = 1,
      .depthOrArrayLayers  = texture.size.depth,
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

        // Upload staging buffer to texture
        wgpuCommandEncoderCopyBufferToTexture(cmd_encoder,
          // Source
          &(WGPUImageCopyBuffer) {
            .buffer = staging_buffer,
            .layout = (WGPUTextureDataLayout) {
              .offset = offset,
              .bytesPerRow = width * 4,
              .rowsPerImage= height,
            },
          },
          // Destination
          &(WGPUImageCopyTexture){
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
            .width               = MAX(1u, width),
            .height              = MAX(1u, height),
            .depth               = 1,
            .depthOrArrayLayers  = 1,
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
          &(WGPUImageCopyBuffer) {
            .buffer = staging_buffer,
            .layout = (WGPUTextureDataLayout) {
              .offset = 0,
              .bytesPerRow = texture.size.width * 4,
              .rowsPerImage= height,
            },
          },
          // Destination
          &(WGPUImageCopyTexture){
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
            .width               = MAX(1u, width),
            .height              = MAX(1u, height),
            .depth               = 1,
            .depthOrArrayLayers  = 1,
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
                                     const char* filename,
                                     WGPUTextureUsageFlags texture_usage_flags)
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
    .width              = texture.size.width,
    .height             = texture.size.height,
    .depth              = texture.size.depth,
    .depthOrArrayLayers = texture.size.depth,
  };

  WGPUTextureDescriptor tex_desc = {
    .usage         = texture_usage_flags,
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
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
    .addressModeW  = WGPUAddressMode_Repeat,
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

/* -------------------------------------------------------------------------- *
 * WebGPU Mipmap Generator
 * -------------------------------------------------------------------------- */

#define NUMBER_OF_TEXTURE_FORMATS WGPUTextureFormat_R8BG8Biplanar420Unorm

struct wgpu_mipmap_generator {
  wgpu_context_t* wgpu_context;
  WGPUSampler sampler;
  // Pipeline for every texture format used.
  WGPUBindGroupLayout pipeline_layouts[(uint32_t)NUMBER_OF_TEXTURE_FORMATS];
  WGPURenderPipeline pipelines[(uint32_t)NUMBER_OF_TEXTURE_FORMATS];
  bool active_pipelines[(uint32_t)NUMBER_OF_TEXTURE_FORMATS];
  // Shaders are shared between all pipelines
  wgpu_shader_t vert_mipmap_shader;
  wgpu_shader_t frag_mipmap_shader;
};

wgpu_mipmap_generator_t*
wgpu_mipmap_generator_create(wgpu_context_t* wgpu_context)
{
  wgpu_mipmap_generator_t* mipmap_generator
    = (wgpu_mipmap_generator_t*)malloc(sizeof(wgpu_mipmap_generator_t));
  memset(mipmap_generator, 0, sizeof(wgpu_mipmap_generator_t));
  mipmap_generator->wgpu_context = wgpu_context;

  // Create sampler
  WGPUSamplerDescriptor sampler_desc = {
    .label         = "mip",
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Nearest,
    .mipmapFilter  = WGPUFilterMode_Nearest,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  mipmap_generator->sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);

  return mipmap_generator;
}

void wgpu_mipmap_generator_destroy(wgpu_mipmap_generator_t* mipmap_generator)
{
  WGPU_RELEASE_RESOURCE(Sampler, mipmap_generator->sampler)
  for (uint32_t i = 0; i < (uint32_t)NUMBER_OF_TEXTURE_FORMATS; ++i) {
    if (mipmap_generator->active_pipelines[i]) {
      WGPU_RELEASE_RESOURCE(RenderPipeline, mipmap_generator->pipelines[i])
      mipmap_generator->active_pipelines[i] = false;
    }
  }
  if (mipmap_generator->vert_mipmap_shader.module
      || mipmap_generator->frag_mipmap_shader.module) {
    wgpu_shader_release(&mipmap_generator->vert_mipmap_shader);
    wgpu_shader_release(&mipmap_generator->frag_mipmap_shader);
  }
  free(mipmap_generator);
}

WGPURenderPipeline wgpu_mipmap_generator_get_mipmap_pipeline(
  wgpu_mipmap_generator_t* mipmap_generator, WGPUTextureFormat format)
{
  uint32_t pipeline_index = (uint32_t)format;
  ASSERT(pipeline_index < (uint32_t)NUMBER_OF_TEXTURE_FORMATS)
  bool pipeline_exists = mipmap_generator->active_pipelines[pipeline_index];
  if (!pipeline_exists) {
    wgpu_context_t* wgpu_context = mipmap_generator->wgpu_context;

    // Rasterization state
    WGPURasterizationStateDescriptor rasterization_state_desc
      = wgpu_create_rasterization_state_descriptor(
        &(create_rasterization_state_desc_t){
          .front_face = WGPUFrontFace_CCW,
          .cull_mode  = WGPUCullMode_None,
        });

    // Color blend state
    WGPUColorStateDescriptor color_state_desc
      = wgpu_create_color_state_descriptor(&(create_color_state_desc_t){
        .format       = format,
        .enable_blend = false,
      });

    // Vertex state
    WGPUVertexStateDescriptor vertex_state = {
      .indexFormat = WGPUIndexFormat_Uint32,
    };

    // Shaders are shared between all pipelines, so only create once.
    if (!mipmap_generator->vert_mipmap_shader.module
        || !mipmap_generator->frag_mipmap_shader.module) {
      // Vertex shader
      mipmap_generator->vert_mipmap_shader = wgpu_shader_create(
        wgpu_context, &(wgpu_shader_desc_t){
                        // Vertex shader SPIR-V
                        .file = "shaders/blit/blit.vert.spv",
                      });
      // Fragment shader
      mipmap_generator->frag_mipmap_shader = wgpu_shader_create(
        wgpu_context, &(wgpu_shader_desc_t){
                        // Fragment shader SPIR-V
                        .file = "shaders/blit/blit.frag.spv",
                      });
    }

    // Create rendering pipeline using the specified states
    mipmap_generator->pipelines[pipeline_index]
      = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device,
        &(WGPURenderPipelineDescriptor){
          .label = "blit",
          // Vertex shader
          .vertexStage
          = mipmap_generator->vert_mipmap_shader.programmable_stage_descriptor,
          // Fragment shader
          .fragmentStage
          = &mipmap_generator->frag_mipmap_shader.programmable_stage_descriptor,
          // Rasterization state
          .rasterizationState     = &rasterization_state_desc,
          .primitiveTopology      = WGPUPrimitiveTopology_TriangleStrip,
          .colorStateCount        = 1,
          .colorStates            = &color_state_desc,
          .depthStencilState      = NULL,
          .vertexState            = &vertex_state,
          .sampleCount            = 1,
          .sampleMask             = 0xFFFFFFFF,
          .alphaToCoverageEnabled = false,
        });
    ASSERT(mipmap_generator->pipelines[pipeline_index]);

    // Store the bind group layout of the created pipeline
    mipmap_generator->pipeline_layouts[pipeline_index]
      = wgpuRenderPipelineGetBindGroupLayout(
        mipmap_generator->pipelines[pipeline_index], 0);
    ASSERT(mipmap_generator->pipeline_layouts[pipeline_index])

    // Update active pipeline state
    mipmap_generator->active_pipelines[pipeline_index] = true;
  }

  return mipmap_generator->pipelines[pipeline_index];
}

WGPUTexture
wgpu_mipmap_generator_generate_mipmap(wgpu_mipmap_generator_t* mipmap_generator,
                                      WGPUTexture texture,
                                      WGPUTextureDescriptor* texture_desc)
{
  WGPURenderPipeline pipeline = wgpu_mipmap_generator_get_mipmap_pipeline(
    mipmap_generator, texture_desc->format);

  if (texture_desc->dimension == WGPUTextureDimension_3D
      || texture_desc->dimension == WGPUTextureDimension_1D) {
    log_error(
      "Generating mipmaps for non-2d textures is currently unsupported!");
    return NULL;
  }

  wgpu_context_t* wgpu_context     = mipmap_generator->wgpu_context;
  WGPUTexture mip_texture          = texture;
  const uint32_t array_layer_count = texture_desc->size.depth > 0 ?
                                       texture_desc->size.depth :
                                       1; // Only valid for 2D textures.
  const uint32_t mip_level_count   = texture_desc->mipLevelCount;

  // If the texture was created with RENDER_ATTACHMENT usage we can render
  // directly between mip levels.
  const WGPUTextureUsage render_to_source
    = texture_desc->usage & WGPUTextureUsage_RenderAttachment;
  if (!render_to_source) {
    // Otherwise we have to use a separate texture to render into. It can be one
    // mip level smaller than the source texture, since we already have the top
    // level.
    const WGPUTextureDescriptor mip_texture_desc = {
      .size = (WGPUExtent3D) {
        .width              = ceil(texture_desc->size.width / 2.0f),
        .height             = ceil(texture_desc->size.height / 2.0f),
        .depth              = 1,
        .depthOrArrayLayers = array_layer_count,
      },
      .format        = texture_desc->format,
      .usage         = WGPUTextureUsage_CopySrc | WGPUTextureUsage_Sampled
                       | WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .mipLevelCount = texture_desc->mipLevelCount - 1,
      .sampleCount   = 1,
    };
    mip_texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &mip_texture_desc);
  }

  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  uint32_t pipeline_index = (uint32_t)texture_desc->format;
  WGPUBindGroupLayout bind_group_layout
    = mipmap_generator->pipeline_layouts[pipeline_index];

  const uint32_t views_count = array_layer_count * mip_level_count;
  WGPUTextureView* views
    = (WGPUTextureView*)calloc(views_count, sizeof(WGPUTextureView));
  const uint32_t bind_group_count = array_layer_count * (mip_level_count - 1);
  WGPUBindGroup* bind_groups
    = (WGPUBindGroup*)calloc(bind_group_count, sizeof(WGPUBindGroup));

  for (uint32_t array_layer = 0; array_layer < array_layer_count;
       ++array_layer) {
    uint32_t view_index = array_layer * mip_level_count;
    views[view_index]   = wgpuTextureCreateView(
      texture, &(WGPUTextureViewDescriptor){
                 .label           = "src_view",
                 .aspect          = WGPUTextureAspect_All,
                 .baseMipLevel    = 0,
                 .mipLevelCount   = 1,
                 .dimension       = WGPUTextureViewDimension_2D,
                 .baseArrayLayer  = array_layer,
                 .arrayLayerCount = 1,
               });

    uint32_t dst_mip_level = render_to_source ? 1 : 0;
    for (uint32_t i = 1; i < texture_desc->mipLevelCount; ++i) {
      const uint32_t target_mip = view_index + i;
      views[target_mip]         = wgpuTextureCreateView(
        mip_texture, &(WGPUTextureViewDescriptor){
                       .label           = "dst_view",
                       .aspect          = WGPUTextureAspect_All,
                       .baseMipLevel    = dst_mip_level++,
                       .mipLevelCount   = 1,
                       .dimension       = WGPUTextureViewDimension_2D,
                       .baseArrayLayer  = array_layer,
                       .arrayLayerCount = 1,
                     });

      const  WGPURenderPassColorAttachmentDescriptor color_attachment_desc
        = (WGPURenderPassColorAttachmentDescriptor){
           .view          = views[target_mip],
           .resolveTarget = NULL,
           .loadOp        = WGPULoadOp_Clear,
           .storeOp       = WGPUStoreOp_Store,
           .clearColor = (WGPUColor){
             .r = 0.0f,
             .g = 0.0f,
             .b = 0.0f,
             .a = 0.0f,
           },
        };
      WGPURenderPassEncoder pass_encoder = wgpuCommandEncoderBeginRenderPass(
        cmd_encoder, &(WGPURenderPassDescriptor){
                       .colorAttachmentCount   = 1,
                       .colorAttachments       = &color_attachment_desc,
                       .depthStencilAttachment = NULL,
                     });

      WGPUBindGroupEntry bg_entries[2] = {
        [0] = (WGPUBindGroupEntry){
          .binding     = 0,
          .textureView = views[target_mip - 1],
        },
        [1] = (WGPUBindGroupEntry){
          .binding = 1,
          .sampler = mipmap_generator->sampler,
        },
      };
      uint32_t bind_group_index = array_layer * (mip_level_count - 1) + i - 1;
      bind_groups[bind_group_index] = wgpuDeviceCreateBindGroup(
        wgpu_context->device, &(WGPUBindGroupDescriptor){
                                .layout     = bind_group_layout,
                                .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                                .entries    = bg_entries,
                              });

      wgpuRenderPassEncoderSetPipeline(pass_encoder, pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass_encoder, 0,
                                        bind_groups[bind_group_index], 0, NULL);
      wgpuRenderPassEncoderDraw(pass_encoder, 4, 1, 0, 0);
      wgpuRenderPassEncoderEndPass(pass_encoder);

      WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass_encoder)
    }
  }

  // If we didn't render to the source texture, finish by copying the mip
  // results from the temporary mipmap texture to the source.
  if (!render_to_source) {
    WGPUExtent3D mip_level_size = (WGPUExtent3D){
      .width              = ceil(texture_desc->size.width / 2.0f),
      .height             = ceil(texture_desc->size.height / 2.0f),
      .depth              = 1,
      .depthOrArrayLayers = array_layer_count,
    };

    for (uint32_t i = 1; i < texture_desc->mipLevelCount - 1; ++i) {
      wgpuCommandEncoderCopyTextureToTexture(cmd_encoder,
                                             // source
                                             &(WGPUImageCopyTexture){
                                               .texture  = mip_texture,
                                               .mipLevel = i - 1,
                                             },
                                             // destination
                                             &(WGPUImageCopyTexture){
                                               .texture  = texture,
                                               .mipLevel = i,
                                             },
                                             // copySize
                                             &mip_level_size);
      mip_level_size.width  = ceil(mip_level_size.width / 2.0f);
      mip_level_size.height = ceil(mip_level_size.height / 2.0f);
    }
  }

  WGPUCommandBuffer command_buffer
    = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  ASSERT(command_buffer != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  // Sumbit commmand buffer and cleanup
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

  if (!render_to_source) {
    WGPU_RELEASE_RESOURCE(Texture, mip_texture);
  }

  // Cleanup
  for (uint32_t i = 0; i < views_count; ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, views[i]);
  }
  free(views);
  for (uint32_t i = 0; i < bind_group_count; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, bind_groups[i]);
  }
  free(bind_groups);

  return texture;
}
