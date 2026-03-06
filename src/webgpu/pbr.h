#ifndef PBR_H_
#define PBR_H_

#include "webgpu/wgpu_common.h"
#include <stdbool.h>
#include <webgpu/webgpu.h>

/* -------------------------------------------------------------------------- *
 * WebGPU environment map (IBL) helper functions
 *
 * Complete Image-Based Lighting pipeline:
 *   HDR panorama (.hdr) → equirectangular cubemap → mipmapped cubemap
 *   → irradiance cubemap + prefiltered specular cubemap + BRDF LUT
 *
 * Usage example:
 *
 *  // Load HDR panorama from file
 *  wgpu_environment_t env = {0};
 *  wgpu_environment_load_from_file(&env, "assets/environments/pisa.hdr");
 *
 *  // Create all IBL textures from the loaded panorama
 *  wgpu_ibl_textures_t ibl = {0};
 *  wgpu_ibl_textures_desc_t ibl_desc = {
 *    .environment_size = 1024,     // cubemap face size
 *    .irradiance_size  = 64,       // irradiance face size
 *    .prefiltered_size = 512,      // specular prefiltered face size
 *    .brdf_lut_size    = 128,      // BRDF LUT size
 *    .num_samples      = 1024,     // Monte Carlo samples
 *  };
 *  wgpu_ibl_textures_from_environment(ctx, &env, &ibl_desc, &ibl);
 *
 *  // Use ibl.environment_cubemap, ibl.irradiance_cubemap,
 *  //     ibl.prefiltered_cubemap, ibl.brdf_lut in your bind groups
 *
 *  // Cleanup
 *  wgpu_ibl_textures_destroy(&ibl);
 *  wgpu_environment_release(&env);
 * -------------------------------------------------------------------------- */

/**
 * @brief CPU-side HDR environment data loaded from an equirectangular
 *        panorama (.hdr). No GPU resources are created at this stage.
 */
typedef struct wgpu_environment_t {
  float* data;     /* RGBA float pixel data (owned, allocated by stbi) */
  uint32_t width;  /* Panorama width (must be 2× height) */
  uint32_t height; /* Panorama height */
  float rotation;  /* Y-axis rotation angle in radians */
} wgpu_environment_t;

/**
 * @brief Configuration for IBL texture generation.
 *        All sizes are per-face dimensions (square). Zero = use defaults.
 */
typedef struct wgpu_ibl_textures_desc_t {
  uint32_t environment_size; /* Cubemap face size (default: 1024) */
  uint32_t irradiance_size;  /* Irradiance cubemap face size (default: 64) */
  uint32_t prefiltered_size; /* Prefiltered specular face size (default: 512)*/
  uint32_t brdf_lut_size;    /* BRDF integration LUT size (default: 128) */
  uint32_t num_samples;      /* Monte Carlo sample count (default: 1024) */
} wgpu_ibl_textures_desc_t;

/**
 * @brief GPU-side IBL textures produced from an environment map.
 *        All textures use RGBA16Float format (except noted).
 */
typedef struct wgpu_ibl_textures_t {
  WGPUTexture environment_cubemap;  /* Mipmapped env cubemap (6 faces) */
  WGPUTextureView environment_view; /* Cube view of env cubemap */
  WGPUTexture irradiance_cubemap;   /* Diffuse irradiance (6 faces) */
  WGPUTextureView irradiance_view;  /* Cube view of irradiance */
  WGPUTexture prefiltered_cubemap;  /* Specular prefiltered (6 faces) */
  WGPUTextureView prefiltered_view; /* Cube view of prefiltered */
  WGPUTexture brdf_lut;             /* 2D BRDF integration LUT */
  WGPUTextureView brdf_lut_view;    /* 2D view of BRDF LUT */
  WGPUSampler environment_sampler;  /* Trilinear, repeat */
  WGPUSampler brdf_lut_sampler;     /* Linear, clamp-to-edge */
  uint32_t prefiltered_mip_levels;  /* Mip count of prefiltered cubemap */
} wgpu_ibl_textures_t;

/* --- Environment loading (CPU-only, no GPU) --- */

/**
 * @brief Loads an equirectangular HDR panorama from file.
 *        The image must have a 2:1 aspect ratio. Images wider than 4096px
 *        are bilinearly downsampled to 4096×2048.
 * @return true on success, false on failure.
 */
bool wgpu_environment_load_from_file(wgpu_environment_t* env,
                                     const char* filepath);

/**
 * @brief Loads an equirectangular HDR panorama from memory.
 * @return true on success, false on failure.
 */
bool wgpu_environment_load_from_memory(wgpu_environment_t* env,
                                       const uint8_t* data, uint32_t size);

/**
 * @brief Releases CPU-side environment pixel data.
 */
void wgpu_environment_release(wgpu_environment_t* env);

/* --- IBL texture generation (GPU compute) --- */

/**
 * @brief Generates all IBL textures from a loaded environment panorama.
 *        Performs the full pipeline: upload panorama → cubemap conversion →
 *        mipmap generation → irradiance/specular/BRDF LUT computation.
 *        All operations are submitted to the GPU queue synchronously.
 *
 * @param wgpu_context  The WebGPU context
 * @param env           Loaded environment data (CPU-side HDR pixels)
 * @param desc          IBL generation parameters (NULL = use defaults)
 * @param ibl           Output IBL textures (caller must destroy with
 *                      wgpu_ibl_textures_destroy)
 * @return true on success, false on failure.
 */
bool wgpu_ibl_textures_from_environment(wgpu_context_t* wgpu_context,
                                        const wgpu_environment_t* env,
                                        const wgpu_ibl_textures_desc_t* desc,
                                        wgpu_ibl_textures_t* ibl);

/**
 * @brief Releases all GPU resources held by the IBL textures.
 */
void wgpu_ibl_textures_destroy(wgpu_ibl_textures_t* ibl);

#endif /* PBR_H_ */
