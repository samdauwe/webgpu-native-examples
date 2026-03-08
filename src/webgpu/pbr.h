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
 *  wgpu_environment_load_from_file(
 *    &env,
 *    "assets/textures/environments/pisa.hdr"
 *  );
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

/* -------------------------------------------------------------------------- *
 * Shared WGSL PBR shader snippets
 *
 * These are commonly-needed PBR/lighting WGSL functions provided as C string
 * constants.  Each snippet is a self-contained set of WGSL functions that can
 * be concatenated (e.g. with string literal juxtaposition or at runtime) into
 * a complete shader module together with the caller's own bindings, structs,
 * and entry points.
 *
 * Usage example (compile-time concatenation):
 *
 *   static const char* my_shader = CODE(
 *     @group(0) @binding(0) var<uniform> exposure : f32;
 *     @group(0) @binding(1) var<uniform> gamma    : f32;
 *   )
 *   WGPU_PBR_WGSL_SRGB           // adds SRGBtoLINEAR, linearToSRGB
 *   WGPU_PBR_WGSL_TONE_MAPPING   // adds toneMapPBRNeutral, Uncharted2, …
 *   CODE(
 *     @fragment fn fs_main() -> @location(0) vec4f {
 *       var c = SRGBtoLINEAR(…);
 *       c = toneMap(c, exposure, gamma, 0);
 *       return vec4f(c, 1.0);
 *     }
 *   );
 *
 * Alternatively, reference the const-char* variables at runtime:
 *
 *   const char* parts[] = {
 *     my_bindings_code,
 *     wgpu_pbr_wgsl_srgb,
 *     wgpu_pbr_wgsl_tone_mapping,
 *     my_main_code,
 *   };
 *   // concatenate parts → create shader module
 * -------------------------------------------------------------------------- */

/**
 * sRGB ↔ linear colour-space conversion (accurate piecewise functions).
 * Provides: SRGBtoLINEAR(vec4f) → vec4f, linearToSRGB(vec3f) → vec3f
 */
extern const char* wgpu_pbr_wgsl_srgb;

/**
 * Compile-time string-literal version (for static concatenation with CODE()).
 */
#define WGPU_PBR_WGSL_SRGB                                                     \
  "fn SRGBtoLINEAR(srgbIn : vec4f) -> vec4f {"                                 \
  "  let bLess = step(vec3f(0.04045), srgbIn.xyz);"                            \
  "  let linOut = mix(srgbIn.xyz / vec3f(12.92),"                              \
  "    pow((srgbIn.xyz + vec3f(0.055)) / vec3f(1.055), vec3f(2.4)), bLess);"   \
  "  return vec4f(linOut, srgbIn.w);"                                          \
  "}"                                                                          \
  "fn linearToSRGB(linIn : vec3f) -> vec3f {"                                  \
  "  let cutoff = step(vec3f(0.0031308), linIn);"                              \
  "  let higher = vec3f(1.055) * pow(linIn, vec3f(1.0/2.4)) - vec3f(0.055);"   \
  "  let lower  = linIn * vec3f(12.92);"                                       \
  "  return mix(lower, higher, cutoff);"                                       \
  "}"

/**
 * Tone-mapping functions (PBR Neutral, Uncharted2, Reinhard).
 * Provides: toneMapPBRNeutral(vec3f) → vec3f
 *           Uncharted2Tonemap(vec3f) → vec3f
 *           toneMap(color: vec3f, exposure: f32, gamma: f32, type: i32) → vec3f
 *             type: 0 = PBR Neutral, 1 = Uncharted2 filmic, 2 = Reinhard
 */
extern const char* wgpu_pbr_wgsl_tone_mapping;

#define WGPU_PBR_WGSL_TONE_MAPPING                                             \
  "fn Uncharted2Tonemap(x : vec3f) -> vec3f {"                                 \
  "  let A=0.15; let B=0.50; let C=0.10;"                                      \
  "  let D=0.20; let E=0.02; let F=0.30;"                                      \
  "  return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;"                          \
  "}"                                                                          \
  "fn toneMapPBRNeutral(colorIn : vec3f) -> vec3f {"                           \
  "  let startCompression = 0.8 - 0.04;"                                       \
  "  let desaturation = 0.15;"                                                 \
  "  let x = min(colorIn.r, min(colorIn.g, colorIn.b));"                       \
  "  let offset = select(0.04, x - 6.25*x*x, x < 0.08);"                       \
  "  var color = colorIn - offset;"                                            \
  "  let peak = max(color.r, max(color.g, color.b));"                          \
  "  if (peak < startCompression) { return color; }"                           \
  "  let d = 1.0 - startCompression;"                                          \
  "  let newPeak = 1.0 - d*d/(peak+d-startCompression);"                       \
  "  color = color * (newPeak / peak);"                                        \
  "  let g = 1.0 - 1.0/(desaturation*(peak-newPeak)+1.0);"                     \
  "  return mix(color, newPeak * vec3f(1.0), g);"                              \
  "}"                                                                          \
  "fn toneMap(colorIn: vec3f, exposure: f32, gamma: f32, tmType: i32)"         \
  "    -> vec3f {"                                                             \
  "  let invGamma = 1.0 / gamma;"                                              \
  "  var c = colorIn * exposure;"                                              \
  "  if (tmType == 1) {"                                                       \
  "    let W = 11.2;"                                                          \
  "    c = Uncharted2Tonemap(c) * (1.0/Uncharted2Tonemap(vec3f(W)));"          \
  "  } else if (tmType == 2) {"                                                \
  "    c = c / (c + vec3f(1.0));"                                              \
  "  } else {"                                                                 \
  "    c = toneMapPBRNeutral(c);"                                              \
  "  }"                                                                        \
  "  return pow(c, vec3f(invGamma));"                                          \
  "}"

/**
 * Cook-Torrance specular micro-facet BRDF components.
 * Provides:
 *   FSchlick(f0, f90, VdotH) → vec3f            Fresnel (Schlick approx)
 *   VGGX(NdotL, NdotV, alphaRoughness) → f32    Smith-GGX visibility
 *   DGGX(NdotH, alphaRoughness) → f32           GGX normal distribution
 *   BRDFLambertian(f0,f90,diffColor,sw,VdotH)   Lambertian diffuse w/ Fresnel
 *   BRDFSpecularGGX(f0,f90,aR,sw,VdotH,NdotL,NdotV,NdotH) Specular BRDF
 *
 * Requires: const pi = 3.141592653589793; (defined by caller)
 */
extern const char* wgpu_pbr_wgsl_brdf;

#define WGPU_PBR_WGSL_BRDF                                                     \
  "fn FSchlick(f0: vec3f, f90: vec3f, vDotH: f32) -> vec3f {"                  \
  "  return f0 + (f90-f0)*pow(clamp(1.0-vDotH, 0.0, 1.0), 5.0);"               \
  "}"                                                                          \
  "fn VGGX(nDotL: f32, nDotV: f32, alphaRoughness: f32) -> f32 {"              \
  "  let a2 = alphaRoughness * alphaRoughness;"                                \
  "  let ggxV = nDotL * sqrt(nDotV*nDotV*(1.0-a2)+a2);"                        \
  "  let ggxL = nDotV * sqrt(nDotL*nDotL*(1.0-a2)+a2);"                        \
  "  let ggx = ggxV + ggxL;"                                                   \
  "  if (ggx > 0.0) { return 0.5 / ggx; } return 0.0;"                         \
  "}"                                                                          \
  "fn DGGX(nDotH: f32, alphaRoughness: f32) -> f32 {"                          \
  "  let a2 = alphaRoughness * alphaRoughness;"                                \
  "  let f = (nDotH*nDotH)*(a2-1.0)+1.0;"                                      \
  "  return a2 / (3.141592653589793 * f * f);"                                 \
  "}"                                                                          \
  "fn BRDFLambertian(f0: vec3f, f90: vec3f, diffuseColor: vec3f,"              \
  "    specularWeight: f32, vDotH: f32) -> vec3f {"                            \
  "  return (1.0-specularWeight*FSchlick(f0,f90,vDotH))"                       \
  "    * (diffuseColor / 3.141592653589793);"                                  \
  "}"                                                                          \
  "fn BRDFSpecularGGX(f0: vec3f, f90: vec3f, alphaRoughness: f32,"             \
  "    specularWeight: f32, vDotH: f32, nDotL: f32,"                           \
  "    nDotV: f32, nDotH: f32) -> vec3f {"                                     \
  "  let F = FSchlick(f0, f90, vDotH);"                                        \
  "  let V = VGGX(nDotL, nDotV, alphaRoughness);"                              \
  "  let D = DGGX(nDotH, alphaRoughness);"                                     \
  "  return specularWeight * F * V * D;"                                       \
  "}"

/**
 * IBL multi-scattering Fresnel (Fdez-Agüera approximation).
 * Provides:
 *   getIBLGGXFresnel(N, V, roughness, F0, specularWeight,
 *                    brdfLUT_tex, brdfLUT_sampler) → vec3f
 *
 * The function reads the BRDF LUT texture directly.  The caller must ensure
 * that the texture and sampler bindings are in scope (any binding slot).
 *
 * NOTE: this snippet references global bindings by name; callers must have
 * `iblBRDFIntegrationLUTTexture` and `iblBRDFIntegrationLUTSampler` in scope.
 */
extern const char* wgpu_pbr_wgsl_ibl_fresnel;

#endif /* PBR_H_ */
