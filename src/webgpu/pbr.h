#ifndef PBR_H
#define PBR_H

#include "../webgpu/gltf_model.h"
#include "texture.h"

/**
 * @brief Generates a BRDF integration map used as a look-up-table (stores
 * roughness / NdotV)
 */
texture_t pbr_generate_brdf_lut(wgpu_context_t* wgpu_context);

/**
 * @brief Generates an irradiance cube map from the environment cube map.
 */
texture_t pbr_generate_irradiance_cube(wgpu_context_t* wgpu_context,
                                       struct gltf_model_t* skybox,
                                       texture_t* skybox_texture);

/**
 * @brief Generates a prefiltered environment cubemap.
 * @see
 * https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
 */
texture_t pbr_generate_prefiltered_env_cube(wgpu_context_t* wgpu_context,
                                            struct gltf_model_t* skybox,
                                            texture_t* skybox_texture);

#endif
