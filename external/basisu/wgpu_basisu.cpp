//-----------------------------------------------------------------------------
//  wgpu_basisu.cpp
//-----------------------------------------------------------------------------
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif
#if !(TARGET_OS_IPHONE || defined(__EMSCRIPTEN__) || defined(__ANDROID__))
#define BASISD_SUPPORT_BC7 (0)
#define BASISD_SUPPORT_ATC (1)
#endif
#if defined(__ANDROID__)
#define BASISD_SUPPORT_PVRTC2 (0)
#endif
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#define BASISU_NO_ITERATOR_DEBUG_LEVEL (1)
#include "basisu_transcoder.cpp"
#include "wgpu_basisu.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <unordered_map>

struct WTTFormatMapItem {
  WGPUTextureFormat format = WGPUTextureFormat_Undefined;
  bool uncompressed        = false;
};

// clang-format off
static const std::unordered_map<basist::transcoder_texture_format, WTTFormatMapItem> WTT_FORMAT_MAP = {
  // Compressed formats
  {basist::transcoder_texture_format::cTFBC1_RGB, WTTFormatMapItem{WGPUTextureFormat_BC1RGBAUnorm}},
  {basist::transcoder_texture_format::cTFBC3_RGBA, WTTFormatMapItem{WGPUTextureFormat_BC3RGBAUnorm}},
  {basist::transcoder_texture_format::cTFBC7_RGBA, WTTFormatMapItem{WGPUTextureFormat_BC7RGBAUnorm}},
  {basist::transcoder_texture_format::cTFETC1_RGB, WTTFormatMapItem{WGPUTextureFormat_ETC2RGB8Unorm}},
  {basist::transcoder_texture_format::cTFETC2_RGBA, WTTFormatMapItem{WGPUTextureFormat_ETC2RGBA8Unorm}},
  {basist::transcoder_texture_format::cTFASTC_4x4_RGBA, WTTFormatMapItem{WGPUTextureFormat_ASTC4x4Unorm}},
  // Uncompressed formats
  {basist::transcoder_texture_format::cTFRGBA32, WTTFormatMapItem{WGPUTextureFormat_RGBA8Unorm, true}},
};
// clang-format on

static basist::etc1_global_selector_codebook* g_pGlobal_codebook = nullptr;

void basisu_setup(void)
{
  basist::basisu_transcoder_init();
  if (!g_pGlobal_codebook) {
    g_pGlobal_codebook = new basist::etc1_global_selector_codebook(
      basist::g_global_selector_cb_size, basist::g_global_selector_cb);
  }
}

void basisu_shutdown(void)
{
  if (g_pGlobal_codebook) {
    delete g_pGlobal_codebook;
    g_pGlobal_codebook = nullptr;
  }
}

bool basisu_is_initialized(void)
{
  return g_pGlobal_codebook != nullptr;
}

static bool QueryBasisTextureformatValue(
  const std::unordered_map<basist::transcoder_texture_format, bool>&
    supportedBasisFormats,
  basist::transcoder_texture_format format)
{
  return (supportedBasisFormats.find(format) == supportedBasisFormats.end()) ?
           false :
           supportedBasisFormats.at(format);
}

// clang-format off
static basist::transcoder_texture_format SelectBasisTextureformat(const std::unordered_map<basist::transcoder_texture_format, bool>& supportedBasisFormats, bool hasAlpha)
{
  basist::transcoder_texture_format basisFormat = basist::transcoder_texture_format::cTFRGBA32;

  if (hasAlpha) {
    if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFBC3_RGBA)) {
      basisFormat = basist::transcoder_texture_format::cTFBC3_RGBA;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFETC2_RGBA)) {
      basisFormat = basist::transcoder_texture_format::cTFETC2_RGBA;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFBC7_RGBA)) {
      basisFormat = basist::transcoder_texture_format::cTFBC7_RGBA;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFASTC_4x4_RGBA)) {
      basisFormat = basist::transcoder_texture_format::cTFASTC_4x4_RGBA;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFPVRTC1_4_RGBA)) {
      basisFormat = basist::transcoder_texture_format::cTFPVRTC1_4_RGBA;
    } else {
      // If we don't support any appropriate compressed formats transcode to
      // raw pixels. This is something of a last resort, because the GPU
      // upload will be significantly slower and take a lot more memory, but
      // at least it prevents you from needing to store a fallback JPG/PNG and
      // the download size will still likely be smaller.
      basisFormat = basist::transcoder_texture_format::cTFRGBA32;
    }
  } else {
    if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFBC1_RGB)) {
      basisFormat = basist::transcoder_texture_format::cTFBC1_RGB;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFETC1_RGB)) {
      // Should be the highest quality, so use when available.
      // http://richg42.blogspot.com/2018/05/basis-universal-gpu-texture-format.html
      basisFormat = basist::transcoder_texture_format::cTFETC1_RGB;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFBC7_RGBA)) {
      basisFormat = basist::transcoder_texture_format::cTFBC7_RGBA;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFETC2_RGBA)) {
      basisFormat = basist::transcoder_texture_format::cTFETC2_RGBA;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFASTC_4x4_RGBA)) {
      basisFormat = basist::transcoder_texture_format::cTFASTC_4x4_RGBA;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFPVRTC1_4_RGB)) {
      basisFormat = basist::transcoder_texture_format::cTFPVRTC1_4_RGB;
    } else if (QueryBasisTextureformatValue(supportedBasisFormats, basist::transcoder_texture_format::cTFRGB565)) {
      // See note on uncompressed transcode above.
      basisFormat = basist::transcoder_texture_format::cTFRGB565;
    } else {
      // See note on uncompressed transcode above.
      basisFormat = basist::transcoder_texture_format::cTFRGBA32;
    }
  }

  return basisFormat;
}
// clang-format on

basisu_transcode_result_t
basisu_transcode(basisu_data_t basisu_data,
                 WGPUTextureFormat (*supported_formats)[12],
                 uint32_t supported_format_count, bool mipmaps)
{

  basisu_transcode_result_t transcodeResult = {};
  transcodeResult.result_code = BASIS_TRANSCODE_RESULT_TRANSCODE_FAILURE;

  // The formats this device supports
  std::unordered_map<basist::transcoder_texture_format, bool>
    supportedBasisFormats;
  for (const auto& item : WTT_FORMAT_MAP) {
    const auto targetFormat             = item.first;
    const auto wttFormat                = item.second.format;
    supportedBasisFormats[targetFormat] = false;
    for (uint32_t i = 0; i < supported_format_count; ++i) {
      if ((*supported_formats)[i] == wttFormat) {
        supportedBasisFormats[targetFormat] = true;
        break;
      }
    }
  }

  const auto basisuDataSize = static_cast<uint32_t>(basisu_data.size);

  assert(g_pGlobal_codebook);
  basist::basisu_transcoder transcoder(g_pGlobal_codebook);
  if (!transcoder.validate_header(basisu_data.ptr, basisuDataSize)) {
    transcodeResult.result_code = BASIS_TRANSCODE_RESULT_INVALID_BASIS_HEADER;
    return transcodeResult;
  }
  if (!transcoder.start_transcoding(basisu_data.ptr, basisuDataSize)) {
    transcodeResult.result_code = BASIS_TRANSCODE_RESULT_TRANSCODE_FAILURE;
    return transcodeResult;
  }

  basist::basisu_image_info imageInfo;
  transcoder.get_image_info(basisu_data.ptr, basisuDataSize, imageInfo, 0);

  auto hasAlpha    = imageInfo.m_alpha_flag;
  auto levels      = imageInfo.m_total_levels;
  auto basisFormat = SelectBasisTextureformat(supportedBasisFormats, hasAlpha);

  if (levels == 0) {
    transcodeResult.result_code = BASIS_TRANSCODE_RESULT_INVALID_BASIS_DATA;
    return transcodeResult;
  }

  if (WTT_FORMAT_MAP.find(basisFormat) == WTT_FORMAT_MAP.end()) {
    transcodeResult.result_code
      = BASIS_TRANSCODE_RESULT_UNSUPPORTED_TRANSCODE_FORMAT;
    return transcodeResult;
  }

  const auto& wttFormat = WTT_FORMAT_MAP.at(basisFormat);

  // If we're not using compressed textures or we've been explicitly instructed
  // to not unpack mipmaps only transcode a single level.
  if (wttFormat.uncompressed || !mipmaps) {
    levels = 1;
  }

  // Set image info
  transcodeResult.image_desc.format      = wttFormat.format;
  transcodeResult.image_desc.width       = imageInfo.m_width;
  transcodeResult.image_desc.height      = imageInfo.m_height;
  transcodeResult.image_desc.level_count = levels;

  // Transcode each mip level
  uint32_t descW, descH, blocks;
  for (uint32_t level = 0; level < levels; level++) {
    // reset per level
    bool success = false;
    if (transcoder.get_image_level_desc(basisu_data.ptr, basisuDataSize, 0,
                                        level, descW, descH, blocks)) {
      uint32_t decSize = basis_get_bytes_per_block_or_pixel(basisFormat);
      if (basis_transcoder_format_is_uncompressed(basisFormat)) {
        decSize *= descW * descH;
      }
      else {
        decSize *= blocks;
      }
      if (void* decBuf = malloc(decSize)) {
        if (basis_transcoder_format_is_uncompressed(basisFormat)) {
          // note that blocks becomes total number of pixels for RGB/RGBA
          blocks = descW * descH;
        }
        if (transcoder.transcode_image_level(basisu_data.ptr, basisuDataSize, 0,
                                             level, decBuf, blocks,
                                             basisFormat)) {
          transcodeResult.image_desc.levels[level].ptr  = decBuf;
          transcodeResult.image_desc.levels[level].size = decSize;
          success                                       = true;
        }
      }
    }
    if (!success) {
      basisu_free(&transcodeResult.image_desc);
      transcodeResult.result_code = BASIS_TRANSCODE_RESULT_TRANSCODE_FAILURE;
      return transcodeResult;
    }
  }
  transcodeResult.result_code = BASIS_TRANSCODE_RESULT_SUCCESS;

  return transcodeResult;
}

void basisu_free(const basisu_image_desc_t* desc)
{
  assert(desc);
  for (uint32_t i = 0; i < desc->level_count; ++i) {
    if (desc->levels[i].ptr) {
      free((void*)desc->levels[i].ptr);
    }
  }
}
