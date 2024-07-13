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

/* Basis Universal Supercompressed GPU Texture Codec */
#include <wgpu_basisu.h>

/* -------------------------------------------------------------------------- *
 * Helper functions
 * -------------------------------------------------------------------------- */

void wgpu_image_to_texure(wgpu_context_t* wgpu_context, WGPUTexture texture,
                          void* pixels, WGPUExtent3D size, uint32_t channels)
{
  const uint64_t data_size = size.width * size.height * size.depthOrArrayLayers
                             * channels * sizeof(uint8_t);
  wgpuQueueWriteTexture(wgpu_context->queue,
    &(WGPUImageCopyTexture) {
      .texture = texture,
      .mipLevel = 0,
      .origin = (WGPUOrigin3D) {
        .x = 0,
        .y = 0,
        .z = 0,
    },
    .aspect = WGPUTextureAspect_All,
    },
    pixels, data_size,
    &(WGPUTextureDataLayout){
      .offset       = 0,
      .bytesPerRow  = size.width * channels * sizeof(uint8_t),
      .rowsPerImage = size.height,
    },
    &(WGPUExtent3D){
      .width              = size.width,
      .height             = size.height,
      .depthOrArrayLayers = size.depthOrArrayLayers,
    });
}

void wgpu_destroy_texture(texture_t* texture)
{
  WGPU_RELEASE_RESOURCE(TextureView, texture->view)
  WGPU_RELEASE_RESOURCE(Texture, texture->texture)
  WGPU_RELEASE_RESOURCE(Sampler, texture->sampler)
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

static WGPUTextureFormat linear_to_sgrb_format(WGPUTextureFormat format)
{
  switch (format) {
    case WGPUTextureFormat_RGBA8Unorm:
      return WGPUTextureFormat_RGBA8UnormSrgb;
    case WGPUTextureFormat_BGRA8Unorm:
      return WGPUTextureFormat_BGRA8UnormSrgb;
    case WGPUTextureFormat_BC1RGBAUnorm:
      return WGPUTextureFormat_BC1RGBAUnormSrgb;
    case WGPUTextureFormat_BC2RGBAUnorm:
      return WGPUTextureFormat_BC2RGBAUnormSrgb;
    case WGPUTextureFormat_BC3RGBAUnorm:
      return WGPUTextureFormat_BC3RGBAUnormSrgb;
    case WGPUTextureFormat_BC7RGBAUnorm:
      return WGPUTextureFormat_BC7RGBAUnormSrgb;
    default:
      return format;
  }
}

static WGPUTextureFormat srgb_to_linear_format(WGPUTextureFormat format)
{
  switch (format) {
    case WGPUTextureFormat_RGBA8UnormSrgb:
      return WGPUTextureFormat_RGBA8Unorm;
    case WGPUTextureFormat_BGRA8UnormSrgb:
      return WGPUTextureFormat_BGRA8Unorm;
    case WGPUTextureFormat_BC1RGBAUnormSrgb:
      return WGPUTextureFormat_BC1RGBAUnorm;
    case WGPUTextureFormat_BC2RGBAUnormSrgb:
      return WGPUTextureFormat_BC2RGBAUnorm;
    case WGPUTextureFormat_BC3RGBAUnormSrgb:
      return WGPUTextureFormat_BC3RGBAUnorm;
    case WGPUTextureFormat_BC7RGBAUnormSrgb:
      return WGPUTextureFormat_BC7RGBAUnorm;
    default:
      return format;
  }
}

static WGPUTextureFormat format_for_color_space(WGPUTextureFormat format,
                                                color_space_enum_t colorSpace)
{
  switch (colorSpace) {
    case COLOR_SPACE_SRGB:
      return linear_to_sgrb_format(format);
    case COLOR_SPACE_LINEAR:
      return srgb_to_linear_format(format);
    default:
      return format;
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
  return (uint32_t)(floor((float)(log2(MAX(width, height))))) + 1;
}

/**
 * @brief Deallocates image data.
 */
static void destroy_image_data(uint8_t** pixel_vec, int width, int height)
{
  uint32_t mipmap_level = calculate_mip_level_count(width, height);
  for (uint32_t i = 0; i < mipmap_level; ++i) {
    free(pixel_vec[i]);
  }
  free(pixel_vec);
}

/**
 * @brief Generates mipmaps on the CPU.
 * @see https://github.com/webatintel/aquarium
 */
static void generate_mipmap(uint8_t* input_pixels, int input_w, int input_h,
                            int input_stride_in_bytes, uint8_t*** output_pixels,
                            int output_w, int output_h,
                            int output_stride_in_bytes, int num_channels,
                            bool is_256_padding)
{
  uint32_t mipmap_level = calculate_mip_level_count(output_w, output_h);
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
  // Vertex state and  Fragment state are shared between all pipelines
  WGPUVertexState vertex_state_desc;
  WGPUFragmentState fragment_state_desc;
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
    .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  mipmap_generator->sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(mipmap_generator->sampler != NULL);

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
  if (!mipmap_generator->vertex_state_desc.module
      || !mipmap_generator->fragment_state_desc.module) {
    WGPU_RELEASE_RESOURCE(ShaderModule,
                          mipmap_generator->vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule,
                          mipmap_generator->fragment_state_desc.module);
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

    // Primitive state
    WGPUPrimitiveState primitive_state_desc = {
      .topology         = WGPUPrimitiveTopology_TriangleStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint32,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .format    = format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex state and Fragment state are shared between all pipelines, so
    // only create once.
    if (!mipmap_generator->vertex_state_desc.module
        || !mipmap_generator->fragment_state_desc.module) {
      // clang-format off
      static const char* mipmap_shader_wgsl = CODE(
        var<private> pos : array<vec2<f32>, 3> = array<vec2<f32>, 3>(
          vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 3.0), vec2<f32>(3.0, -1.0)
        );

        struct VertexOutput {
          @builtin(position) position : vec4<f32>,
          @location(0) texCoord : vec2<f32>,
        }

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
          var output : VertexOutput;
          output.texCoord = pos[vertexIndex] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
          output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
          return output;
        }

        @group(0) @binding(0) var imgSampler : sampler;
        @group(0) @binding(1) var img : texture_2d<f32>;

        @fragment
        fn fragmentMain(@location(0) texCoord : vec2<f32>) -> @location(0) vec4<f32> {
          return textureSample(img, imgSampler, texCoord);
        }
      );
      // clang-format on

      // Vertex state
      mipmap_generator->vertex_state_desc = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader WGSL
                .wgsl_code.source = mipmap_shader_wgsl,
                .entry = "vertexMain",
              },
              .buffer_count = 0,
              .buffers = NULL,
            });
      // Fragment state
      mipmap_generator->fragment_state_desc = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader WGSL
                .wgsl_code.source = mipmap_shader_wgsl,
                .entry = "fragmentMain",
              },
              .target_count = 1,
              .targets = &color_target_state_desc,
            });
    }

    // Multisample state
    WGPUMultisampleState multisample_state_desc
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    mipmap_generator->pipelines[pipeline_index]
      = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device,
        &(WGPURenderPipelineDescriptor){
          .label       = "blit_render_pipeline",
          .primitive   = primitive_state_desc,
          .vertex      = mipmap_generator->vertex_state_desc,
          .fragment    = &mipmap_generator->fragment_state_desc,
          .multisample = multisample_state_desc,
        });
    ASSERT(mipmap_generator->pipelines[pipeline_index] != NULL);

    // Store the bind group layout of the created pipeline
    mipmap_generator->pipeline_layouts[pipeline_index]
      = wgpuRenderPipelineGetBindGroupLayout(
        mipmap_generator->pipelines[pipeline_index], 0);
    ASSERT(mipmap_generator->pipeline_layouts[pipeline_index] != NULL)

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
  const uint32_t array_layer_count = texture_desc->size.depthOrArrayLayers > 0 ?
                                       texture_desc->size.depthOrArrayLayers :
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
        .depthOrArrayLayers = array_layer_count,
      },
      .format        = texture_desc->format,
      .usage         = WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding
                       | WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .mipLevelCount = texture_desc->mipLevelCount - 1,
      .sampleCount   = 1,
    };
    mip_texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &mip_texture_desc);
    ASSERT(mip_texture != NULL);
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

      const  WGPURenderPassColorAttachment color_attachment_desc
        = (WGPURenderPassColorAttachment){
           .view          = views[target_mip],
           .depthSlice    = ~0,
           .resolveTarget = NULL,
           .loadOp        = WGPULoadOp_Clear,
           .storeOp       = WGPUStoreOp_Store,
           .clearValue = (WGPUColor){
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
          .binding = 0,
          .sampler = mipmap_generator->sampler,
        },
        [1] = (WGPUBindGroupEntry){
          .binding     = 1,
          .textureView = views[target_mip - 1],
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
      wgpuRenderPassEncoderDraw(pass_encoder, 3, 1, 0, 0);
      wgpuRenderPassEncoderEnd(pass_encoder);

      WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass_encoder)
    }
  }

  // If we didn't render to the source texture, finish by copying the mip
  // results from the temporary mipmap texture to the source.
  if (!render_to_source) {
    WGPUExtent3D mip_level_size = (WGPUExtent3D){
      .width              = ceil(texture_desc->size.width / 2.0f),
      .height             = ceil(texture_desc->size.height / 2.0f),
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

/* -------------------------------------------------------------------------- *
 * WebGPU Texture Client
 * -------------------------------------------------------------------------- */

typedef struct texture_result_t {
  WGPUTexture texture;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t mip_level_count;
  WGPUTextureFormat format;
  WGPUTextureDimension dimension;
} texture_result_t;

texture_result_t wgpu_texture_client_load_texture_from_memory(
  struct wgpu_texture_client_t* texture_client, void* data, size_t data_size,
  struct wgpu_texture_load_options_t* options)
{
  if (!texture_client->wgpu_context) {
    log_error("Cannot create new textures after object has been destroyed.");
    return (texture_result_t){0};
  }

  static const uint8_t comp_map[5] = {
    0, //
    1, //
    2, //
    4, //
    4  //
  };
  static const uint32_t channels[5] = {
    STBI_default,    // only used for req_comp
    STBI_grey,       //
    STBI_grey_alpha, //
    STBI_rgb_alpha,  //
    STBI_rgb_alpha   //
  };

  bool is_hdr = stbi_is_hdr_from_memory((stbi_uc*)data, data_size);
  int width = 0, height = 0, read_comps = 4;
  stbi_set_flip_vertically_on_load(options ? options->flip_y : false);
  uint8_t* pixel_data
    = is_hdr ? (uint8_t*)stbi_loadf_from_memory((stbi_uc*)data, data_size,
                                                &width, &height, &read_comps,
                                                channels[read_comps]) :
               (uint8_t*)stbi_load_from_memory((stbi_uc*)data, data_size,
                                               &width, &height, &read_comps,
                                               channels[read_comps]);

  if (pixel_data == NULL) {
    log_warn("Couldn't parse image data!");
    return (texture_result_t){0};
  }
  ASSERT(pixel_data);

  const uint8_t comps         = comp_map[read_comps];
  const bool generate_mipmaps = options ? options->generate_mipmaps : false;
  const uint32_t mip_level_count
    = generate_mipmaps ? calculate_mip_level_count(width, height) : 1u;

  const WGPUTextureUsage usage
    = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;

  WGPUExtent3D texture_size = {
    .width              = width,
    .height             = height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .usage         = usage,
    .dimension     = WGPUTextureDimension_2D,
    .size          = texture_size,
    .format        = options ? (options->format != WGPUTextureFormat_Undefined ?
                                  format_for_color_space(options->format,
                                                         options->color_space) :
                                  WGPUTextureFormat_RGBA8Unorm) :
                               WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = mip_level_count,
    .sampleCount   = 1,
  };
  WGPUTexture texture = wgpuDeviceCreateTexture(
    texture_client->wgpu_context->device, &texture_desc);

  // Copy pixel data to texture and free allocated memory
  wgpu_image_to_texure(texture_client->wgpu_context, texture, pixel_data,
                       texture_size, comps);
  free(pixel_data);

  if (generate_mipmaps) {
    texture = wgpu_mipmap_generator_generate_mipmap(
      texture_client->wgpu_mipmap_generator, texture, &texture_desc);
  }

  return (texture_result_t){
    .texture         = texture,
    .width           = texture_desc.size.width,
    .height          = texture_desc.size.height,
    .depth           = texture_desc.size.depthOrArrayLayers,
    .mip_level_count = texture_desc.mipLevelCount,
    .format          = texture_desc.format,
    .dimension       = texture_desc.dimension,
  };
}

typedef struct {
  int32_t image_width;
  int32_t image_height;
  int32_t channel_count;
  stbi_uc* pixel_data;
} stb_image_load_result_t;

static stb_image_load_result_t
stb_image_load_image_from_file(const char* filename, bool flip_y)
{
  static const uint8_t comp_map[5] = {
    0, //
    1, //
    2, //
    4, //
    4  //
  };
  static const uint32_t channels[5] = {
    STBI_default,    // only used for req_comp
    STBI_grey,       //
    STBI_grey_alpha, //
    STBI_rgb_alpha,  //
    STBI_rgb_alpha   //
  };

  int width = 0, height = 0;
  // Force loading 4 channel images to 3 channel by stb becasue Dawn doesn't
  // support 3 channel formats currently. The group is discussing on whether
  // webgpu shoud support 3 channel format.
  // https://github.com/gpuweb/gpuweb/issues/66#issuecomment-410021505
  int read_comps = 4;
  stbi_set_flip_vertically_on_load(flip_y);
  stbi_uc* pixel_data = stbi_load(filename,            //
                                  &width,              //
                                  &height,             //
                                  &read_comps,         //
                                  channels[read_comps] //
  );

  if (pixel_data == NULL) {
    log_error("Couldn't load '%s'\n", filename);
  }
  else {
    log_debug("Loaded image %s (%d, %d, %d / %d)\n", filename, width, height,
              read_comps, comp_map[read_comps]);
  }

  return (stb_image_load_result_t){
    .image_width   = width,
    .image_height  = height,
    .channel_count = comp_map[read_comps],
    .pixel_data    = pixel_data,
  };
}

static texture_result_t
wgpu_texture_load_with_stb(struct wgpu_texture_client_t* texture_client,
                           const char* filename,
                           struct wgpu_texture_load_options_t* options)
{
  if (!texture_client->wgpu_context) {
    log_error("Cannot create new textures after object has been destroyed.");
    return (texture_result_t){0};
  }

  const bool flip_y = options ? options->flip_y : false;
  stb_image_load_result_t image_load_result
    = stb_image_load_image_from_file(filename, flip_y);

  stbi_uc* pixel_data = image_load_result.pixel_data;
  if (pixel_data == NULL) {
    return (texture_result_t){0};
  }
  ASSERT(pixel_data);

  const int width             = image_load_result.image_width;
  const int height            = image_load_result.image_height;
  const int channel_count     = image_load_result.channel_count;
  const bool generate_mipmaps = options ? options->generate_mipmaps : false;
  const uint32_t mip_level_count
    = generate_mipmaps ? calculate_mip_level_count(width, height) : 1u;

  const WGPUTextureUsage usage
    = options ? (options->usage != WGPUTextureUsage_None ?
                   options->usage :
                   WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding) :
                WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;

  WGPUExtent3D texture_size = {
    .width              = width,
    .height             = height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .usage         = usage,
    .dimension     = WGPUTextureDimension_2D,
    .size          = texture_size,
    .format        = options ? (options->format != WGPUTextureFormat_Undefined ?
                                  format_for_color_space(options->format,
                                                         options->color_space) :
                                  WGPUTextureFormat_RGBA8Unorm) :
                               WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = mip_level_count,
    .sampleCount   = 1,
  };
  WGPUTexture texture = wgpuDeviceCreateTexture(
    texture_client->wgpu_context->device, &texture_desc);

  // Copy pixel data to texture and free allocated memory
  wgpu_image_to_texure(texture_client->wgpu_context, texture, pixel_data,
                       texture_size, channel_count);
  stbi_image_free(pixel_data);

  if (generate_mipmaps) {
    if (texture_client->wgpu_mipmap_generator == NULL) {
      texture_client->wgpu_mipmap_generator
        = wgpu_mipmap_generator_create(texture_client->wgpu_context);
    }
    texture = wgpu_mipmap_generator_generate_mipmap(
      texture_client->wgpu_mipmap_generator, texture, &texture_desc);
  }

  return (texture_result_t){
    .texture         = texture,
    .width           = texture_desc.size.width,
    .height          = texture_desc.size.height,
    .depth           = texture_desc.size.depthOrArrayLayers,
    .mip_level_count = texture_desc.mipLevelCount,
    .format          = texture_desc.format,
    .dimension       = texture_desc.dimension,
  };
}

static texture_result_t
wgpu_texture_cubemap_load_with_stb(wgpu_context_t* wgpu_context,
                                   const char* filenames[6],
                                   struct wgpu_texture_load_options_t* options)
{
  // Swap top and bottom when images should be flipped vertically
  const bool flip_y         = options ? options->flip_y : false;
  const uint16_t mapping[6] = {0, 1, flip_y ? 3 : 2, flip_y ? 2 : 3, 4, 5};

  // Load images into memory
  stb_image_load_result_t image_load_results[6] = {0};
  for (uint32_t face = 0; face < 6; ++face) {
    image_load_results[face]
      = stb_image_load_image_from_file(filenames[mapping[face]], flip_y);
    if (image_load_results[face].pixel_data == NULL) {
      // Free pixel data for already loaded images
      for (uint32_t prev_face = 0; prev_face < face; ++prev_face) {
        stbi_image_free(image_load_results[prev_face].pixel_data);
      }
      return (texture_result_t){0};
    }
  }

  // Use first image to determine the width and height of the image
  const uint32_t width           = image_load_results[0].image_width;
  const uint32_t height          = image_load_results[0].image_height;
  const uint32_t channel_count   = image_load_results[0].channel_count;
  const uint32_t depth           = 6u;
  const uint32_t mip_level_count = 1;
  const WGPUTextureFormat texture_format
    = options ?
        (options->format != WGPUTextureFormat_Undefined ?
           format_for_color_space(options->format, options->color_space) :
           WGPUTextureFormat_RGBA8Unorm) :
        WGPUTextureFormat_RGBA8Unorm;
  const size_t texture_size = width * height * channel_count * sizeof(uint8_t);

  // Create cubemap texture
  WGPUTextureDescriptor texture_desc = {
    .size          = (WGPUExtent3D) {
      .width               = width,
      .height              = height,
      .depthOrArrayLayers  = depth,
     },
    .mipLevelCount = mip_level_count,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = texture_format,
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
  };
  WGPUTexture texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create a host-visible staging buffers that contains the raw image data for
  // each face of the cubemap
  WGPUBuffer staging_buffers[6] = {0};
  for (uint32_t face = 0; face < depth; ++face) {
    WGPUBufferDescriptor staging_buffer_desc = {
      .usage            = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
      .size             = texture_size,
      .mappedAtCreation = true,
    };
    staging_buffers[face]
      = wgpuDeviceCreateBuffer(wgpu_context->device, &staging_buffer_desc);
    ASSERT(staging_buffers[face])
  }

  for (uint32_t face = 0; face < depth; ++face) {
    // Copy texture data into staging buffer
    void* mapping
      = wgpuBufferGetMappedRange(staging_buffers[face], 0, texture_size);
    ASSERT(mapping)
    memcpy(mapping, image_load_results[face].pixel_data, texture_size);
    wgpuBufferUnmap(staging_buffers[face]);

    // Upload staging buffer to texture
    wgpuCommandEncoderCopyBufferToTexture(cmd_encoder,
      // Source
      &(WGPUImageCopyBuffer) {
        .buffer = staging_buffers[face],
        .layout = (WGPUTextureDataLayout) {
          .offset       = 0,
          .bytesPerRow  = width * channel_count,
          .rowsPerImage = height,
        },
      },
      // Destination
      &(WGPUImageCopyTexture){
        .texture = texture,
        .mipLevel = 0,
        .origin = (WGPUOrigin3D) {
          .x = 0,
          .y = 0,
          .z = face,
        },
        .aspect = WGPUTextureAspect_All,
      },
      // Copy size
      &(WGPUExtent3D){
        .width              = width,
        .height             = height,
        .depthOrArrayLayers = 1,
      });
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

  // Clean up staging resources and pixel data
  for (uint32_t face = 0; face < depth; ++face) {
    WGPU_RELEASE_RESOURCE(Buffer, staging_buffers[face]);
    stbi_image_free(image_load_results[face].pixel_data);
  }

  return (texture_result_t){
    .texture         = texture,
    .width           = texture_desc.size.width,
    .height          = texture_desc.size.height,
    .depth           = texture_desc.size.depthOrArrayLayers,
    .mip_level_count = texture_desc.mipLevelCount,
    .format          = texture_desc.format,
    .dimension       = texture_desc.dimension,
  };
}

static ktxResult load_ktx_file(const char* filename, ktxTexture** target)
{
  ktxResult result = KTX_SUCCESS;
  if (!file_exists(filename)) {
    log_fatal("Could not load texture from %s", filename);
    return KTX_FILE_OPEN_FAILED;
  }
  result = ktxTexture_CreateFromNamedFile(
    filename, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, target);
  return result;
}

static texture_result_t
wgpu_texture_load_from_ktx_file(wgpu_context_t* wgpu_context,
                                const char* filename)
{
  ktxTexture* ktx_texture;
  ktxResult result = load_ktx_file(filename, &ktx_texture);
  assert(result == KTX_SUCCESS);

  ktx_uint8_t* ktx_texture_data = ktxTexture_GetData(ktx_texture);

  // WebGPU requires that the bytes per row is a multiple of 256
  uint32_t resized_width = make_multiple_of_256(ktx_texture->baseWidth);

  // Get properties required for using and upload texture data from the ktx
  // texture object
  uint32_t texture_width
    = ktx_texture->isCubemap ? ktx_texture->baseWidth : resized_width;
  uint32_t texture_height = ktx_texture->baseHeight;
  uint32_t texture_depth  = ktx_texture->isCubemap ? 6u : 1u;
  uint32_t numLevelsOffset /* bytesPerRow must multiple of 256. */
    = (uint32_t)(logf(256.0f / 4.0f) / logf(2.0f));
  uint32_t texture_mip_level_count
    = ktx_texture->isCubemap ?
        (ktx_texture->numLevels > 6u ?
           ktx_texture->numLevels - numLevelsOffset :
           1u) :
        calculate_mip_level_count(ktx_texture->baseWidth,
                                  ktx_texture->baseHeight);
  WGPUTextureFormat texture_format = WGPUTextureFormat_RGBA8Unorm;

  WGPUTextureDescriptor texture_desc = {
    .size          = (WGPUExtent3D) {
      .width               = texture_width,
      .height              = texture_height,
      .depthOrArrayLayers  = texture_depth,
     },
    .mipLevelCount = texture_mip_level_count,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = texture_format,
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
  };
  WGPUTexture texture
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

    for (uint32_t face = 0; face < texture_depth; ++face) {
      for (uint32_t level = 0; level < texture_mip_level_count; ++level) {
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
            .texture = texture,
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
            .depthOrArrayLayers  = 1,
          });
      }
    }

    WGPU_RELEASE_RESOURCE(Buffer, staging_buffer);
  }
  else { /* WGPUTextureDimension_2D */
    // Generate Mipmap
    uint8_t** resized_vec = NULL;
    generate_mipmap(ktx_texture_data, ktx_texture->baseWidth,
                    ktx_texture->baseHeight, 0, &resized_vec, resized_width,
                    ktx_texture->baseHeight, 0, 4, true);

    // Setup buffer copy regions for each face including all of its mip levels
    // and copy the texture regions from the staging buffer into the texture
    for (uint32_t face = 0; face < texture_depth; ++face) {
      for (uint32_t level = 0; level < texture_mip_level_count; ++level) {
        uint32_t width  = texture_width >> level;
        uint32_t height = texture_height >> level;
        if (height == 0) {
          height = 1;
        }

        // Create a host-visible staging buffer that contains the raw image data
        size_t ktx_texture_size                  = texture_width * height * 4;
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
              .bytesPerRow = texture_width * 4,
              .rowsPerImage= height,
            },
          },
          // Destination
          &(WGPUImageCopyTexture){
            .texture = texture,
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

  // Clean up staging resources
  ktxTexture_Destroy(ktx_texture);

  return (texture_result_t){
    .texture         = texture,
    .width           = texture_desc.size.width,
    .height          = texture_desc.size.height,
    .depth           = texture_desc.size.depthOrArrayLayers,
    .mip_level_count = texture_desc.mipLevelCount,
    .format          = texture_desc.format,
    .dimension       = texture_desc.dimension,
  };
}

static texture_result_t
wgpu_texture_load_from_basis_file(wgpu_context_t* wgpu_context,
                                  const char* filename)
{
  // Read file into memory
  if (!file_exists(filename)) {
    log_fatal("Could not load texture from %s", filename);
    return (texture_result_t){0};
  }

  file_read_result_t file_read_result = {0};
  read_file(filename, &file_read_result, false);

  struct wgpu_texture_client_t* texture_client = wgpu_context->texture_client;

  // Transcode file
  basisu_setup();
  basisu_transcode_result_t transcode_result = basisu_transcode(
    (basisu_data_t){
      .ptr  = file_read_result.data,
      .size = file_read_result.size,
    },
    &texture_client->supported_format_list.values,
    texture_client->supported_format_list.count, false);
  if (transcode_result.result_code != BASIS_TRANSCODE_RESULT_SUCCESS) {
    log_fatal("Could not transcode texture from %s", filename);
    return (texture_result_t){0};
  }

  // Create file
  basisu_image_desc_t* image_desc = &transcode_result.image_desc;
  WGPUTextureDescriptor texture_desc = {
    .size          = (WGPUExtent3D) {
      .width               = image_desc->width,
      .height              = image_desc->height,
      .depthOrArrayLayers  = 1,
     },
    .mipLevelCount = image_desc->level_count,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = image_desc->format,
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
  };
  WGPUTexture texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  for (uint32_t level = 0; level < image_desc->level_count; ++level) {
    uint32_t width  = image_desc->width >> level;
    uint32_t height = image_desc->height >> level;
    if (height == 0) {
      height = 1;
    }

    // Create a host-visible staging buffer that contains the raw image data
    size_t texture_size = image_desc->levels[level].size * 4;
    WGPUBufferDescriptor staging_buffer_desc = {
      .usage            = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
      .size             = texture_size,
      .mappedAtCreation = true,
    };
    WGPUBuffer staging_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &staging_buffer_desc);
    ASSERT(staging_buffer)

    // Copy texture data into staging buffer
    void* mapping = wgpuBufferGetMappedRange(staging_buffer, 0, texture_size);
    ASSERT(mapping)
    memcpy(mapping, image_desc->levels[level].ptr,
           image_desc->levels[level].size);
    wgpuBufferUnmap(staging_buffer);

    // Upload statging buffer to texture
    wgpuCommandEncoderCopyBufferToTexture(cmd_encoder,
      // Source
      &(WGPUImageCopyBuffer) {
        .buffer = staging_buffer,
        .layout = (WGPUTextureDataLayout) {
          .offset = 0,
          .bytesPerRow = texture_size / height,
          .rowsPerImage= height,
        },
      },
      // Destination
      &(WGPUImageCopyTexture){
        .texture = texture,
        .mipLevel = level,
        .origin = (WGPUOrigin3D) {
          .x=0,
          .y=0,
          .z=0,
        },
        .aspect = WGPUTextureAspect_All,
      },
      // Copy size
      &(WGPUExtent3D){
        .width               = MAX(1u, width),
        .height              = MAX(1u, height),
        .depthOrArrayLayers  = 1,
      });

    WGPU_RELEASE_RESOURCE(Buffer, staging_buffer);
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

  // Clean up staging resources
  basisu_free(image_desc);
  basisu_shutdown();
  free(file_read_result.data);

  return (texture_result_t){
    .texture         = texture,
    .width           = texture_desc.size.width,
    .height          = texture_desc.size.height,
    .depth           = texture_desc.size.depthOrArrayLayers,
    .mip_level_count = texture_desc.mipLevelCount,
    .format          = texture_desc.format,
    .dimension       = texture_desc.dimension,
  };
}

static texture_result_t wgpu_texture_client_load_texture_from_file(
  struct wgpu_texture_client_t* texture_client, const char* filename,
  struct wgpu_texture_load_options_t* options)
{
  if (filename_has_extension(filename, "jpg")
      || filename_has_extension(filename, "png")) {
    return wgpu_texture_load_with_stb(texture_client, filename, options);
  }
  else if (filename_has_extension(filename, "ktx")) {
    return wgpu_texture_load_from_ktx_file(texture_client->wgpu_context,
                                           filename);
  }
  else if (filename_has_extension(filename, "basis")) {
    return wgpu_texture_load_from_basis_file(texture_client->wgpu_context,
                                             filename);
  }

  return (texture_result_t){0};
}

static texture_t
wgpu_create_texture(wgpu_context_t* wgpu_context,
                    texture_result_t* texture_result,
                    struct wgpu_texture_load_options_t* options)
{
  ASSERT(texture_result);

  const bool is_cubemap = texture_result->depth == 6u;

  // Create texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .format = texture_result->format,
    .dimension
    = is_cubemap ? WGPUTextureViewDimension_Cube : WGPUTextureViewDimension_2D,
    .baseMipLevel   = 0,
    .mipLevelCount  = texture_result->mip_level_count,
    .baseArrayLayer = 0,
    .arrayLayerCount
    = texture_result->depth, // Cube faces count as array layers
  };
  WGPUTextureView texture_view
    = wgpuTextureCreateView(texture_result->texture, &texture_view_dec);

  const bool is_size_power_of_2 = is_power_of_2(texture_result->width)
                                  && is_power_of_2(texture_result->height);
  WGPUMipmapFilterMode mipmapFilter = is_size_power_of_2 && !is_cubemap ?
                                        WGPUMipmapFilterMode_Linear :
                                        WGPUMipmapFilterMode_Nearest;
  WGPUAddressMode address_mode
    = options ? options->address_mode : WGPUAddressMode_ClampToEdge;

  // Create sampler
  WGPUSamplerDescriptor sampler_desc = {
    .addressModeU  = address_mode,
    .addressModeV  = address_mode,
    .addressModeW  = address_mode,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = mipmapFilter,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = (float)texture_result->mip_level_count,
    .maxAnisotropy = 1,
  };
  WGPUSampler sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);

  return (texture_t){
    .size = {
      .width               = texture_result->width,
      .height              = texture_result->height,
      .depthOrArrayLayers  = texture_result->depth,
    },
    .mip_level_count = texture_result->mip_level_count,
    .format          = texture_result->format,
    .dimension       = texture_result->dimension,
    .texture         = texture_result->texture,
    .view            = texture_view,
    .sampler         = sampler,
  };
}

struct wgpu_texture_client_t*
wgpu_texture_client_create(wgpu_context_t* wgpu_context)
{
  struct wgpu_texture_client_t* texture_client
    = (struct wgpu_texture_client_t*)malloc(
      sizeof(struct wgpu_texture_client_t));
  memset(texture_client, 0, sizeof(struct wgpu_texture_client_t));
  texture_client->wgpu_context = wgpu_context;

  {
    static const WGPUTextureFormat uncompressed_format_list[4] = {
      WGPUTextureFormat_RGBA8Unorm,
      WGPUTextureFormat_RGBA8UnormSrgb,
      WGPUTextureFormat_BGRA8Unorm,
      WGPUTextureFormat_BGRA8UnormSrgb,
    };
    memcpy(texture_client->uncompressed_format_list.values,
           uncompressed_format_list, sizeof(uncompressed_format_list));
    texture_client->uncompressed_format_list.count = 4;
  }

  {
    static const WGPUTextureFormat supported_format_list[12] = {
      // Uncompressed format list
      WGPUTextureFormat_RGBA8Unorm,
      WGPUTextureFormat_RGBA8UnormSrgb,
      WGPUTextureFormat_BGRA8Unorm,
      WGPUTextureFormat_BGRA8UnormSrgb,
      // Compressed format list
      WGPUTextureFormat_BC1RGBAUnorm,
      WGPUTextureFormat_BC1RGBAUnormSrgb,
      WGPUTextureFormat_BC2RGBAUnorm,
      WGPUTextureFormat_BC2RGBAUnormSrgb,
      WGPUTextureFormat_BC3RGBAUnorm,
      WGPUTextureFormat_BC3RGBAUnormSrgb,
      WGPUTextureFormat_BC7RGBAUnorm,
      WGPUTextureFormat_BC7RGBAUnormSrgb,
    };
    memcpy(texture_client->supported_format_list.values, supported_format_list,
           sizeof(supported_format_list));
    texture_client->allow_compressed_formats = wgpuDeviceHasFeature(
      wgpu_context->device, WGPUFeatureName_TextureCompressionBC);
    texture_client->supported_format_list.count
      = texture_client->allow_compressed_formats ? 12 : 4;
  }

  return texture_client;
}

void wgpu_texture_client_destroy(struct wgpu_texture_client_t* texture_client)
{
  if (texture_client != NULL) {
    if (texture_client->wgpu_mipmap_generator != NULL) {
      wgpu_mipmap_generator_destroy(texture_client->wgpu_mipmap_generator);
      texture_client->wgpu_mipmap_generator = NULL;
    }
    free(texture_client);
    texture_client = NULL;
  }
}

void get_supported_formats(struct wgpu_texture_client_t* texture_client,
                           WGPUTextureFormat* supported_formats, size_t* count)
{
  UNUSED_VAR(supported_formats);

  if (texture_client->allow_compressed_formats) {
    supported_formats = texture_client->supported_format_list.values;
    *count            = texture_client->supported_format_list.count;
  }
  else {
    supported_formats = texture_client->uncompressed_format_list.values;
    *count            = texture_client->uncompressed_format_list.count;
  }
}

/* -------------------------------------------------------------------------- *
 * Helper functions
 * -------------------------------------------------------------------------- */

texture_t
wgpu_create_texture_from_memory(wgpu_context_t* wgpu_context, void* data,
                                size_t data_size,
                                struct wgpu_texture_load_options_t* options)
{
  if (wgpu_context->texture_client == NULL) {
    wgpu_create_texture_client(wgpu_context);
  }
  struct wgpu_texture_client_t* texture_client = wgpu_context->texture_client;

  texture_result_t texture_result
    = wgpu_texture_client_load_texture_from_memory(texture_client, data,
                                                   data_size, options);

  if (texture_result.texture) {
    return wgpu_create_texture(texture_client->wgpu_context, &texture_result,
                               options);
  }

  return (texture_t){0};
}

texture_t
wgpu_create_texture_from_file(wgpu_context_t* wgpu_context,
                              const char* filename,
                              struct wgpu_texture_load_options_t* options)
{
  if (wgpu_context->texture_client == NULL) {
    wgpu_create_texture_client(wgpu_context);
  }
  struct wgpu_texture_client_t* texture_client = wgpu_context->texture_client;

  texture_result_t texture_result = wgpu_texture_client_load_texture_from_file(
    texture_client, filename, options);

  if (texture_result.texture) {
    return wgpu_create_texture(texture_client->wgpu_context, &texture_result,
                               options);
  }

  return (texture_t){0};
}

texture_t wgpu_create_texture_cubemap_from_files(
  wgpu_context_t* wgpu_context, const char* filenames[6],
  struct wgpu_texture_load_options_t* options)
{
  texture_result_t texture_result
    = wgpu_texture_cubemap_load_with_stb(wgpu_context, filenames, options);

  if (texture_result.texture) {
    return wgpu_create_texture(wgpu_context, &texture_result, options);
  }

  return (texture_t){0};
}

texture_t wgpu_create_empty_texture(wgpu_context_t* wgpu_context)
{
  /* Create texture */
  WGPUExtent3D texture_size = {
    .width              = 1,
    .height             = 1,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
    .dimension     = WGPUTextureDimension_2D,
    .size          = texture_size,
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  WGPUTexture texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  /* Generate pixel data */
  size_t channels    = 4;
  size_t buffer_size = texture_size.width * texture_size.height * channels
                       * sizeof(unsigned char);
  unsigned char* buffer = malloc(buffer_size);
  memset(buffer, 0, buffer_size);

  /* Copy pixel data to texture */
  wgpu_image_to_texure(wgpu_context, texture, (void*)buffer, texture_size,
                       channels);
  free(buffer);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .format          = texture_desc.format,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  WGPUTextureView view = wgpuTextureCreateView(texture, &texture_view_dec);

  /* Create the texture sampler */
  WGPUSamplerDescriptor sampler_desc = {
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
    .addressModeW  = WGPUAddressMode_Repeat,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUMipmapFilterMode_Linear,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  WGPUSampler sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);

  return (texture_t){
    .size = {
      .width               = texture_size.width,
      .height              = texture_size.height,
      .depthOrArrayLayers  = texture_size.depthOrArrayLayers,
    },
    .mip_level_count = texture_desc.mipLevelCount,
    .format          = texture_desc.format,
    .dimension       = texture_desc.dimension,
    .texture         = texture,
    .view            = view,
    .sampler         = sampler,
  };
}
