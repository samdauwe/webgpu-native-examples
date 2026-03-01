/* -------------------------------------------------------------------------- *
 * Image Loader
 *
 * Thin wrapper around stb_image providing a clean C99 API for loading images
 * from memory buffers and file paths. Supports LDR (8-bit) and HDR (float)
 * image formats.
 *
 * This is the single compilation unit for stb_image in the project. No other
 * source file should define STB_IMAGE_IMPLEMENTATION.
 * -------------------------------------------------------------------------- */

#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------------------- *
 * Types
 * -------------------------------------------------------------------------- */

/* Decoded LDR image (8-bit per channel) */
typedef struct {
  uint8_t* pixels; /* Decoded pixel data (RGBA if desired_channels == 4) */
  int width;       /* Image width in pixels */
  int height;      /* Image height in pixels */
  int channels;    /* Actual number of channels in the source image */
} image_t;

/* Decoded HDR image (32-bit float per channel) */
typedef struct {
  float* pixels; /* Decoded pixel data (RGBAF if desired_channels == 4) */
  int width;     /* Image width in pixels */
  int height;    /* Image height in pixels */
  int channels;  /* Actual number of channels in the source image */
} hdr_image_t;

/* -------------------------------------------------------------------------- *
 * Loading functions (struct-based)
 * -------------------------------------------------------------------------- */

/**
 * @brief Load an LDR image from a memory buffer into an image_t struct.
 *
 * @param buffer           Pointer to the encoded image data (PNG, JPEG, etc.)
 * @param length           Size of the buffer in bytes.
 * @param desired_channels Number of channels to decode to (e.g. 4 for RGBA).
 *                         Pass 0 to use the source channel count.
 * @param out_image        Receives the decoded image data on success.
 * @return true on success, false on failure.
 */
bool image_load_from_memory(const uint8_t* buffer, int length,
                            int desired_channels, image_t* out_image);

/**
 * @brief Load an LDR image from a file path into an image_t struct.
 *
 * @param filename         Path to the image file.
 * @param desired_channels Number of channels to decode to (e.g. 4 for RGBA).
 *                         Pass 0 to use the source channel count.
 * @param out_image        Receives the decoded image data on success.
 * @return true on success, false on failure.
 */
bool image_load_from_file(const char* filename, int desired_channels,
                          image_t* out_image);

/**
 * @brief Load an HDR (float) image from a memory buffer into an hdr_image_t.
 *
 * @param buffer           Pointer to the encoded HDR image data (.hdr, etc.)
 * @param length           Size of the buffer in bytes.
 * @param desired_channels Number of channels to decode to (e.g. 4 for RGBA).
 *                         Pass 0 to use the source channel count.
 * @param out_image        Receives the decoded HDR image data on success.
 * @return true on success, false on failure.
 */
bool image_load_hdr_from_memory(const uint8_t* buffer, int length,
                                int desired_channels, hdr_image_t* out_image);

/* -------------------------------------------------------------------------- *
 * Loading functions (direct pointer-returning)
 *
 * These return raw pixel pointers directly, mirroring the stb_image calling
 * convention. Use image_free() to release the returned pixel data.
 * -------------------------------------------------------------------------- */

/**
 * @brief Load an LDR image from a memory buffer, returning the pixel pointer.
 */
uint8_t* image_pixels_from_memory(const uint8_t* buffer, int length, int* width,
                                  int* height, int* channels,
                                  int desired_channels);

/**
 * @brief Load an LDR image from a file path, returning the pixel pointer.
 */
uint8_t* image_pixels_from_file(const char* filename, int* width, int* height,
                                int* channels, int desired_channels);

/**
 * @brief Load an HDR (float) image from a memory buffer, returning the pixel
 *        pointer.
 */
float* image_pixels_hdr_from_memory(const uint8_t* buffer, int length,
                                    int* width, int* height, int* channels,
                                    int desired_channels);

/* -------------------------------------------------------------------------- *
 * Query functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Get image dimensions and channel count without decoding.
 *
 * @param buffer   Pointer to the encoded image data.
 * @param length   Size of the buffer in bytes.
 * @param width    Receives the image width.
 * @param height   Receives the image height.
 * @param channels Receives the number of channels.
 * @return true on success, false on failure.
 */
bool image_get_info_from_memory(const uint8_t* buffer, int length, int* width,
                                int* height, int* channels);

/**
 * @brief Get a brief description of the last load failure.
 *
 * @return A static string describing the failure, or NULL if no failure.
 */
const char* image_failure_reason(void);

/* -------------------------------------------------------------------------- *
 * Configuration functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Set whether images should be flipped vertically on load.
 *
 * @param flip true to flip, false for default orientation.
 */
void image_set_flip_vertically_on_load(bool flip);

/* -------------------------------------------------------------------------- *
 * Cleanup
 * -------------------------------------------------------------------------- */

/**
 * @brief Free pixel data returned by any of the load functions.
 *
 * @param pixel_data Pointer to the pixel data (image_t.pixels or
 *                   hdr_image_t.pixels). Safe to call with NULL.
 */
void image_free(void* pixel_data);

#endif /* IMAGE_LOADER_H */
