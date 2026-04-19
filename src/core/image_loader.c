/* -------------------------------------------------------------------------- *
 * Image Loader - Implementation
 *
 * Thin wrapper around stb_image. This is the only compilation unit in the
 * project that instantiates the stb_image implementation.
 *
 * Reference:
 * https://github.com/nothings/stb/blob/master/stb_image.h
 * -------------------------------------------------------------------------- */

#include "core/image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <string.h>

/* -------------------------------------------------------------------------- *
 * Struct-based loading functions
 * -------------------------------------------------------------------------- */

bool image_load_from_memory(const uint8_t* buffer, int length,
                            int desired_channels, image_t* out_image)
{
  if (!buffer || !out_image) {
    return false;
  }

  memset(out_image, 0, sizeof(*out_image));
  out_image->format = IMAGE_FORMAT_LDR;

  out_image->pixels.u8 = stbi_load_from_memory(
    buffer, length, &out_image->width, &out_image->height, &out_image->channels,
    desired_channels);

  return out_image->pixels.u8 != NULL;
}

bool image_load_from_file(const char* filename, int desired_channels,
                          image_t* out_image)
{
  if (!filename || !out_image) {
    return false;
  }

  memset(out_image, 0, sizeof(*out_image));
  out_image->format = IMAGE_FORMAT_LDR;

  out_image->pixels.u8
    = stbi_load(filename, &out_image->width, &out_image->height,
                &out_image->channels, desired_channels);

  return out_image->pixels.u8 != NULL;
}

bool image_load_hdr_from_memory(const uint8_t* buffer, int length,
                                int desired_channels, image_t* out_image)
{
  if (!buffer || !out_image) {
    return false;
  }

  memset(out_image, 0, sizeof(*out_image));
  out_image->format = IMAGE_FORMAT_HDR;

  out_image->pixels.f32 = stbi_loadf_from_memory(
    buffer, length, &out_image->width, &out_image->height, &out_image->channels,
    desired_channels);

  return out_image->pixels.f32 != NULL;
}

bool image_load_hdr_from_file(const char* filename, int desired_channels,
                              image_t* out_image)
{
  if (!filename || !out_image) {
    return false;
  }

  memset(out_image, 0, sizeof(*out_image));
  out_image->format = IMAGE_FORMAT_HDR;

  out_image->pixels.f32
    = stbi_loadf(filename, &out_image->width, &out_image->height,
                 &out_image->channels, desired_channels);

  return out_image->pixels.f32 != NULL;
}

/* -------------------------------------------------------------------------- *
 * Direct pointer-returning functions
 * -------------------------------------------------------------------------- */

uint8_t* image_pixels_from_memory(const uint8_t* buffer, int length, int* width,
                                  int* height, int* channels,
                                  int desired_channels)
{
  return stbi_load_from_memory(buffer, length, width, height, channels,
                               desired_channels);
}

uint8_t* image_pixels_from_file(const char* filename, int* width, int* height,
                                int* channels, int desired_channels)
{
  return stbi_load(filename, width, height, channels, desired_channels);
}

float* image_pixels_hdr_from_memory(const uint8_t* buffer, int length,
                                    int* width, int* height, int* channels,
                                    int desired_channels)
{
  return stbi_loadf_from_memory(buffer, length, width, height, channels,
                                desired_channels);
}

float* image_pixels_hdr_from_file(const char* filename, int* width, int* height,
                                  int* channels, int desired_channels)
{
  return stbi_loadf(filename, width, height, channels, desired_channels);
}

uint16_t* image_pixels_16_from_file(const char* filename, int* width,
                                    int* height, int* channels,
                                    int desired_channels)
{
  return stbi_load_16(filename, width, height, channels, desired_channels);
}

/* -------------------------------------------------------------------------- *
 * Query functions
 * -------------------------------------------------------------------------- */

bool image_get_info_from_memory(const uint8_t* buffer, int length, int* width,
                                int* height, int* channels)
{
  if (!buffer) {
    return false;
  }

  return stbi_info_from_memory(buffer, length, width, height, channels) != 0;
}

const char* image_failure_reason(void)
{
  return stbi_failure_reason();
}

/* -------------------------------------------------------------------------- *
 * Configuration functions
 * -------------------------------------------------------------------------- */

void image_set_flip_vertically_on_load(bool flip)
{
  stbi_set_flip_vertically_on_load(flip ? 1 : 0);
}

/* -------------------------------------------------------------------------- *
 * Cleanup
 * -------------------------------------------------------------------------- */

void image_free(void* pixel_data)
{
  if (pixel_data) {
    stbi_image_free(pixel_data);
  }
}

void image_destroy(image_t* image)
{
  if (image) {
    if (image->pixels.raw) {
      stbi_image_free(image->pixels.raw);
    }
    memset(image, 0, sizeof(*image));
  }
}
