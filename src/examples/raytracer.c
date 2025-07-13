#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Raytracer
 *
 * WebGPU demo featuring realtime path tracing via WebGPU compute shaders.
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Material
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/Material.ts
 * -------------------------------------------------------------------------- */

/* Material type enumeration */
typedef enum material_type_enun {
  MATERIAL_TYPE_EMISSIVE_MATERIAL   = 0,
  MATERIAL_TYPE_REFLECTIVE_MATERIAL = 1,
  MATERIAL_TYPE_DIELECTRIC_MATERIAL = 2,
  MATERIAL_TYPE_LAMBERTIAN_MATERIAL = 3,
} material_type_enun;

/* Material structure */
typedef struct {
  vec4 albedo;
  material_type_enun mtl_type;
  float reflection_ratio;
  float reflection_gloss;
  float refraction_index;
} material_t;

static void material_init(material_t* this, vec4 albedo,
                          material_type_enun mtl_type, float reflection_ratio,
                          float reflection_gloss, float refraction_index)
{
  glm_vec4_copy(albedo, this->albedo);
  this->mtl_type         = mtl_type;
  this->reflection_ratio = reflection_ratio;
  this->reflection_gloss = reflection_gloss;
  this->refraction_index = refraction_index;
}

static void material_init_default(material_t* this, vec4 albedo)
{
  material_init(this, albedo, MATERIAL_TYPE_LAMBERTIAN_MATERIAL, 0.0f, 1.0f,
                1.0f);
}
