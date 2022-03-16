#pragma once
/*
 * wgpu_basisu.h -- C-API wrapper glue code for Basis Universal
 */
#include <stdbool.h>
#include <stdint.h>

#include <dawn/webgpu.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* Various compile-time constants */
enum {
  // Basis transcode result
  BASIS_TRANSCODE_RESULT_SUCCESS                      = 0u,
  BASIS_TRANSCODE_RESULT_INVALID_BASIS_HEADER         = 1u,
  BASIS_TRANSCODE_RESULT_INVALID_BASIS_DATA           = 2u,
  BASIS_TRANSCODE_RESULT_TRANSCODE_FAILURE            = 3u,
  BASIS_TRANSCODE_RESULT_UNSUPPORTED_TRANSCODE_FORMAT = 4u,
  // Upper limit(s)
  BASISU_MAX_MIPMAPS = 16u,
};

/*
 * basisu_data_t is a pointer-size-pair struct used to pass memory blobs.
 */
typedef struct basisu_data_t {
  const void* ptr;
  size_t size;
} basisu_data_t;

typedef struct basisu_image_desc_t {
  WGPUTextureFormat format;
  uint32_t width;
  uint32_t height;
  uint32_t level_count; // Number of mipmaps
  basisu_data_t levels[BASISU_MAX_MIPMAPS];
} basisu_image_desc_t;

typedef struct basisu_transcode_result_t {
  uint32_t result_code;
  basisu_image_desc_t image_desc;
} basisu_transcode_result_t;

/* Basis Universal Setup/Shudown */
void basisu_setup(void);
void basisu_shutdown(void);
bool basisu_is_initialized(void);

/* Basis Universal transcoding */
basisu_transcode_result_t
basisu_transcode(basisu_data_t basisu_data,
                 WGPUTextureFormat (*supported_formats)[12],
                 uint32_t supported_format_count, bool mipmaps);
void basisu_free(const basisu_image_desc_t* desc);

#if defined(__cplusplus)
} // extern "C"
#endif
