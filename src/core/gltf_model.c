/* -------------------------------------------------------------------------- *
 * glTF 2.0 Model - Implementation
 *
 * Full C99 glTF 2.0 model loader following the official specification.
 * Uses cgltf for parsing and stb_image for image decoding.
 *
 * Reference implementations:
 * - Vulkan-glTF-PBR (Sascha Willems)
 * - webgpu-gltf-viewer
 *
 * Reference:
 * https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
 * -------------------------------------------------------------------------- */
#include "core/gltf_model.h"
#include "core/image_loader.h"

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * Internal helper macros
 * -------------------------------------------------------------------------- */

static void gltf_log_error(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  fprintf(stderr, "[glTF ERROR] ");
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
}

static void gltf_log_warn(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  fprintf(stderr, "[glTF WARN] ");
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
}

#define GLTF_SAFE_FREE(ptr)                                                    \
  do {                                                                         \
    if ((ptr)) {                                                               \
      free((ptr));                                                             \
      (ptr) = NULL;                                                            \
    }                                                                          \
  } while (0)

#define GLTF_MIN(a, b) ((a) < (b) ? (a) : (b))
#define GLTF_MAX(a, b) ((a) > (b) ? (a) : (b))

/* -------------------------------------------------------------------------- *
 * Internal loader context
 *
 * Tracks buffer positions during geometry loading to build a single
 * shared vertex/index buffer.
 * -------------------------------------------------------------------------- */

typedef struct {
  gltf_vertex_t* vertex_buffer;
  uint32_t* index_buffer;
  size_t vertex_pos;
  size_t index_pos;
} gltf_loader_info_t;

/* -------------------------------------------------------------------------- *
 * Bounding box helpers
 * -------------------------------------------------------------------------- */

void gltf_bounding_box_get_aabb(const gltf_bounding_box_t* bb, mat4 m,
                                gltf_bounding_box_t* dest)
{
  vec3 min_val, max_val;

  /* Transform the 8 OBB corners and find the AABB.
   * Optimized: decompose transform to find min/max per axis. */
  vec3 center;
  glm_vec3_add((float*)bb->min, (float*)bb->max, center);
  glm_vec3_scale(center, 0.5f, center);

  vec3 half_extent;
  glm_vec3_sub((float*)bb->max, (float*)bb->min, half_extent);
  glm_vec3_scale(half_extent, 0.5f, half_extent);

  /* Transform center */
  vec4 center4 = {center[0], center[1], center[2], 1.0f};
  vec4 transformed_center;
  glm_mat4_mulv(m, center4, transformed_center);

  /* Compute new half-extents using abs of the rotation/scale matrix */
  vec3 new_half;
  for (int i = 0; i < 3; i++) {
    new_half[i] = fabsf(m[0][i]) * half_extent[0]
                  + fabsf(m[1][i]) * half_extent[1]
                  + fabsf(m[2][i]) * half_extent[2];
  }

  glm_vec3_sub((float*)transformed_center, new_half, min_val);
  glm_vec3_add((float*)transformed_center, new_half, max_val);

  glm_vec3_copy(min_val, dest->min);
  glm_vec3_copy(max_val, dest->max);
  dest->valid = true;
}

/* -------------------------------------------------------------------------- *
 * String helpers
 * -------------------------------------------------------------------------- */

static void safe_strncpy(char* dest, const char* src, size_t max_len)
{
  if (src) {
    size_t len = strlen(src);
    if (len >= max_len) {
      len = max_len - 1;
    }
    memcpy(dest, src, len);
    dest[len] = '\0';
  }
  else {
    dest[0] = '\0';
  }
}

/* -------------------------------------------------------------------------- *
 * Path helpers
 * -------------------------------------------------------------------------- */

static void extract_directory(const char* filepath, char* dir, size_t dir_size)
{
  safe_strncpy(dir, filepath, dir_size);
  char* last_sep = strrchr(dir, '/');
  if (!last_sep) {
    last_sep = strrchr(dir, '\\');
  }
  if (last_sep) {
    last_sep[1] = '\0';
  }
  else {
    dir[0] = '\0';
  }
}

static void build_path(const char* dir, const char* filename, char* out,
                       size_t out_size)
{
  if (dir[0] == '\0') {
    safe_strncpy(out, filename, out_size);
  }
  else {
    snprintf(out, out_size, "%s%s", dir, filename);
  }
}

/* -------------------------------------------------------------------------- *
 * Texture loading
 * -------------------------------------------------------------------------- */

static void load_texture_samplers(const cgltf_data* gltf_data,
                                  gltf_model_t* model)
{
  model->texture_count = 0;

  /* First pass: count textures */
  uint32_t tex_count = (uint32_t)gltf_data->textures_count;
  if (tex_count == 0) {
    return;
  }

  model->textures = (gltf_texture_t*)calloc(tex_count, sizeof(gltf_texture_t));
  if (!model->textures) {
    gltf_log_error("Failed to allocate textures array");
    return;
  }
  model->texture_count = tex_count;

  for (uint32_t i = 0; i < tex_count; i++) {
    gltf_texture_t* tex     = &model->textures[i];
    const cgltf_texture* gt = &gltf_data->textures[i];

    /* Name */
    safe_strncpy(tex->name, gt->name, GLTF_MODEL_MAX_NAME_LENGTH);

    /* Default sampler values per spec: REPEAT wrapping */
    tex->sampler.mag_filter = GltfFilter_Linear;
    tex->sampler.min_filter = GltfFilter_LinearMipmapLinear;
    tex->sampler.wrap_s     = GltfWrap_Repeat;
    tex->sampler.wrap_t     = GltfWrap_Repeat;

    /* Override with actual sampler if present */
    if (gt->sampler) {
      const cgltf_sampler* gs = gt->sampler;
      if (gs->mag_filter != 0) {
        tex->sampler.mag_filter = (gltf_filter_enum)gs->mag_filter;
      }
      if (gs->min_filter != 0) {
        tex->sampler.min_filter = (gltf_filter_enum)gs->min_filter;
      }
      tex->sampler.wrap_s = (gltf_wrap_enum)gs->wrap_s;
      tex->sampler.wrap_t = (gltf_wrap_enum)gs->wrap_t;
    }

    tex->mip_levels = 1;
  }
}

static void load_texture_images(const cgltf_data* gltf_data,
                                gltf_model_t* model)
{
  for (uint32_t i = 0; i < model->texture_count; i++) {
    gltf_texture_t* tex     = &model->textures[i];
    const cgltf_texture* gt = &gltf_data->textures[i];

    if (!gt->image) {
      continue;
    }

    const cgltf_image* img = gt->image;
    int w = 0, h = 0;

    if (img->buffer_view) {
      /* Image embedded in buffer (GLB or data URI) */
      const uint8_t* buf_data
        = (const uint8_t*)cgltf_buffer_view_data(img->buffer_view);
      if (buf_data) {
        image_t decoded = {0};
        if (image_load_from_memory(buf_data, (int)img->buffer_view->size, 4,
                                   &decoded)) {
          tex->data = decoded.pixels.u8;
          w         = decoded.width;
          h         = decoded.height;
        }
      }
    }
    else if (img->uri) {
      /* External image file */
      char image_path[GLTF_MODEL_MAX_URI_LENGTH];
      build_path(model->file_path, img->uri, image_path, sizeof(image_path));
      image_t decoded = {0};
      if (image_load_from_file(image_path, 4, &decoded)) {
        tex->data = decoded.pixels.u8;
        w         = decoded.width;
        h         = decoded.height;
      }
    }

    if (tex->data) {
      tex->width    = (uint32_t)w;
      tex->height   = (uint32_t)h;
      tex->channels = 4; /* Always request RGBA */
    }
    else {
      gltf_log_warn("Failed to load texture: %s",
                    img->uri ? img->uri : "(embedded)");
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Material loading
 * -------------------------------------------------------------------------- */

static int32_t get_texture_index(const cgltf_data* gltf_data,
                                 const cgltf_texture* tex)
{
  if (!tex) {
    return -1;
  }
  return (int32_t)(tex - gltf_data->textures);
}

static void load_materials(const cgltf_data* gltf_data, gltf_model_t* model)
{
  uint32_t mat_count = (uint32_t)gltf_data->materials_count;

  /* Allocate +1 for default material at the end */
  model->materials
    = (gltf_material_t*)calloc(mat_count + 1, sizeof(gltf_material_t));
  if (!model->materials) {
    gltf_log_error("Failed to allocate materials array");
    model->material_count = 0;
    return;
  }

  for (uint32_t i = 0; i < mat_count; i++) {
    const cgltf_material* gm = &gltf_data->materials[i];
    gltf_material_t* mat     = &model->materials[i];

    /* Name */
    safe_strncpy(mat->name, gm->name, GLTF_MODEL_MAX_NAME_LENGTH);
    mat->index = (int32_t)i;

    /* --- Initialize all texture indices to "no texture" --- */
    mat->base_color_tex_index          = -1;
    mat->metallic_roughness_tex_index  = -1;
    mat->normal_tex_index              = -1;
    mat->occlusion_tex_index           = -1;
    mat->emissive_tex_index            = -1;
    mat->diffuse_tex_index             = -1;
    mat->specular_glossiness_tex_index = -1;

    /* --- Default values per glTF 2.0 spec --- */
    glm_vec4_one(mat->base_color_factor); /* [1,1,1,1] */
    mat->metallic_factor    = 1.0f;
    mat->roughness_factor   = 1.0f;
    mat->normal_scale       = 1.0f;
    mat->occlusion_strength = 1.0f;
    mat->alpha_mode         = GltfAlphaMode_Opaque;
    mat->alpha_cutoff       = 0.5f;
    mat->double_sided       = false;
    mat->unlit              = false;
    mat->emissive_strength  = 1.0f;
    mat->pbr_workflow       = GltfPbrWorkflow_MetallicRoughness;
    glm_vec4_zero(mat->emissive_factor);
    glm_vec4_one(mat->diffuse_factor);
    glm_vec3_zero(mat->specular_factor);
    mat->glossiness_factor = 1.0f;

    /* Zero-init tex coord sets (all default to TEXCOORD_0) */
    memset(&mat->tex_coord_sets, 0, sizeof(mat->tex_coord_sets));

    /* --- PBR Metallic-Roughness (core spec) --- */
    if (gm->has_pbr_metallic_roughness) {
      const cgltf_pbr_metallic_roughness* pbr = &gm->pbr_metallic_roughness;

      glm_vec4_copy((float*)pbr->base_color_factor, mat->base_color_factor);
      mat->metallic_factor  = pbr->metallic_factor;
      mat->roughness_factor = pbr->roughness_factor;

      if (pbr->base_color_texture.texture) {
        mat->base_color_tex_index
          = get_texture_index(gltf_data, pbr->base_color_texture.texture);
        mat->tex_coord_sets.base_color
          = (uint8_t)pbr->base_color_texture.texcoord;
      }
      if (pbr->metallic_roughness_texture.texture) {
        mat->metallic_roughness_tex_index = get_texture_index(
          gltf_data, pbr->metallic_roughness_texture.texture);
        mat->tex_coord_sets.metallic_roughness
          = (uint8_t)pbr->metallic_roughness_texture.texcoord;
      }
    }

    /* --- Normal texture --- */
    if (gm->normal_texture.texture) {
      mat->normal_tex_index
        = get_texture_index(gltf_data, gm->normal_texture.texture);
      mat->normal_scale          = gm->normal_texture.scale;
      mat->tex_coord_sets.normal = (uint8_t)gm->normal_texture.texcoord;
    }

    /* --- Occlusion texture --- */
    if (gm->occlusion_texture.texture) {
      mat->occlusion_tex_index
        = get_texture_index(gltf_data, gm->occlusion_texture.texture);
      mat->occlusion_strength       = gm->occlusion_texture.scale;
      mat->tex_coord_sets.occlusion = (uint8_t)gm->occlusion_texture.texcoord;
    }

    /* --- Emissive --- */
    if (gm->emissive_texture.texture) {
      mat->emissive_tex_index
        = get_texture_index(gltf_data, gm->emissive_texture.texture);
      mat->tex_coord_sets.emissive = (uint8_t)gm->emissive_texture.texcoord;
    }
    mat->emissive_factor[0] = gm->emissive_factor[0];
    mat->emissive_factor[1] = gm->emissive_factor[1];
    mat->emissive_factor[2] = gm->emissive_factor[2];
    mat->emissive_factor[3] = 0.0f;

    /* --- Alpha mode --- */
    switch (gm->alpha_mode) {
      case cgltf_alpha_mode_mask:
        mat->alpha_mode = GltfAlphaMode_Mask;
        break;
      case cgltf_alpha_mode_blend:
        mat->alpha_mode = GltfAlphaMode_Blend;
        break;
      default:
        mat->alpha_mode = GltfAlphaMode_Opaque;
        break;
    }
    mat->alpha_cutoff = gm->alpha_cutoff;
    mat->double_sided = gm->double_sided;
    mat->unlit        = gm->unlit;

    /* --- KHR_materials_pbrSpecularGlossiness extension --- */
    if (gm->has_pbr_specular_glossiness) {
      mat->pbr_workflow = GltfPbrWorkflow_SpecularGlossiness;
      const cgltf_pbr_specular_glossiness* sg = &gm->pbr_specular_glossiness;

      glm_vec4_copy((float*)sg->diffuse_factor, mat->diffuse_factor);
      glm_vec3_copy((float*)sg->specular_factor, mat->specular_factor);
      mat->glossiness_factor = sg->glossiness_factor;

      if (sg->diffuse_texture.texture) {
        mat->diffuse_tex_index
          = get_texture_index(gltf_data, sg->diffuse_texture.texture);
        mat->tex_coord_sets.diffuse = (uint8_t)sg->diffuse_texture.texcoord;
      }
      if (sg->specular_glossiness_texture.texture) {
        mat->specular_glossiness_tex_index = get_texture_index(
          gltf_data, sg->specular_glossiness_texture.texture);
        mat->tex_coord_sets.specular_glossiness
          = (uint8_t)sg->specular_glossiness_texture.texcoord;
      }
    }

    /* --- KHR_materials_emissive_strength extension --- */
    if (gm->has_emissive_strength) {
      mat->emissive_strength = gm->emissive_strength.emissive_strength;
    }
  }

  /* --- Default material at index mat_count --- */
  {
    gltf_material_t* def = &model->materials[mat_count];
    def->index           = (int32_t)mat_count;
    safe_strncpy(def->name, "default", GLTF_MODEL_MAX_NAME_LENGTH);
    glm_vec4_one(def->base_color_factor);
    def->metallic_factor    = 1.0f;
    def->roughness_factor   = 1.0f;
    def->normal_scale       = 1.0f;
    def->occlusion_strength = 1.0f;
    def->alpha_mode         = GltfAlphaMode_Opaque;
    def->alpha_cutoff       = 0.5f;
    def->double_sided       = false;
    def->unlit              = false;
    def->emissive_strength  = 1.0f;
    def->pbr_workflow       = GltfPbrWorkflow_MetallicRoughness;
    glm_vec4_zero(def->emissive_factor);
    glm_vec4_one(def->diffuse_factor);
    glm_vec3_zero(def->specular_factor);
    def->glossiness_factor             = 1.0f;
    def->base_color_tex_index          = -1;
    def->metallic_roughness_tex_index  = -1;
    def->normal_tex_index              = -1;
    def->occlusion_tex_index           = -1;
    def->emissive_tex_index            = -1;
    def->diffuse_tex_index             = -1;
    def->specular_glossiness_tex_index = -1;
    memset(&def->tex_coord_sets, 0, sizeof(def->tex_coord_sets));
  }

  model->material_count = mat_count + 1; /* includes default */
}

/* -------------------------------------------------------------------------- *
 * Geometry pre-counting
 *
 * First pass over the node tree to determine total vertex and index counts
 * so we can pre-allocate single contiguous buffers.
 * -------------------------------------------------------------------------- */

static void get_node_vertex_index_count(const cgltf_node* node,
                                        size_t* vertex_count,
                                        size_t* index_count)
{
  if (node->mesh) {
    const cgltf_mesh* mesh = node->mesh;
    for (cgltf_size p = 0; p < mesh->primitives_count; p++) {
      const cgltf_primitive* prim = &mesh->primitives[p];

      /* Find POSITION accessor for vertex count */
      for (cgltf_size a = 0; a < prim->attributes_count; a++) {
        if (prim->attributes[a].type == cgltf_attribute_type_position) {
          *vertex_count += prim->attributes[a].data->count;
          break;
        }
      }

      /* Index count */
      if (prim->indices) {
        *index_count += prim->indices->count;
      }
    }
  }

  for (cgltf_size c = 0; c < node->children_count; c++) {
    get_node_vertex_index_count(node->children[c], vertex_count, index_count);
  }
}

/* -------------------------------------------------------------------------- *
 * Node loading
 * -------------------------------------------------------------------------- */

static gltf_node_t* alloc_node(void)
{
  gltf_node_t* node = (gltf_node_t*)calloc(1, sizeof(gltf_node_t));
  if (!node) {
    gltf_log_error("Failed to allocate node");
    return NULL;
  }

  /* Default transform values per spec */
  glm_vec3_zero(node->translation);
  glm_quat_identity(node->rotation);
  glm_vec3_one(node->scale);
  glm_mat4_identity(node->matrix);
  glm_mat4_identity(node->cached_local_matrix);
  glm_mat4_identity(node->cached_world_matrix);
  node->skin_index        = -1;
  node->has_matrix        = false;
  node->use_cached_matrix = false;

  return node;
}

/* Compute local matrix from TRS or raw matrix */
void gltf_node_get_local_matrix(const gltf_node_t* node, mat4 dest)
{
  if (node->has_matrix) {
    /* Use raw matrix and apply TRS on top:
     * local = T * R * S * M (per Vulkan reference impl) */
    mat4 t_mat, r_mat, s_mat, tr_mat, trs_mat;
    glm_translate_make(t_mat, (float*)node->translation);
    glm_quat_mat4((float*)node->rotation, r_mat);
    glm_scale_make(s_mat, (float*)node->scale);
    glm_mat4_mul(t_mat, r_mat, tr_mat);
    glm_mat4_mul(tr_mat, s_mat, trs_mat);
    glm_mat4_mul(trs_mat, (vec4*)node->matrix, dest);
  }
  else {
    /* Compose from TRS: local = T * R * S */
    mat4 t_mat, r_mat, s_mat, tr_mat;
    glm_translate_make(t_mat, (float*)node->translation);
    glm_quat_mat4((float*)node->rotation, r_mat);
    glm_scale_make(s_mat, (float*)node->scale);
    glm_mat4_mul(t_mat, r_mat, tr_mat);
    glm_mat4_mul(tr_mat, s_mat, dest);
  }
}

/* Walk parent chain to compute world matrix */
void gltf_node_get_world_matrix(const gltf_node_t* node, mat4 dest)
{
  mat4 local;
  gltf_node_get_local_matrix(node, local);

  const gltf_node_t* p = node->parent;
  if (p) {
    mat4 parent_world;
    gltf_node_get_world_matrix(p, parent_world);
    glm_mat4_mul(parent_world, local, dest);
  }
  else {
    glm_mat4_copy(local, dest);
  }
}

/* Recursively update node and children */
void gltf_node_update(gltf_node_t* node)
{
  /* Compute world matrix */
  gltf_node_get_world_matrix(node, node->cached_world_matrix);

  /* Update skinning joint matrices if this node has a skinned mesh */
  if (node->mesh && node->skin) {
    gltf_skin_t* skin = node->skin;
    gltf_mesh_t* mesh = node->mesh;

    /* Inverse of the node's world transform */
    mat4 inverse_transform;
    glm_mat4_inv(node->cached_world_matrix, inverse_transform);

    uint32_t joint_count
      = GLTF_MIN(skin->joint_count, GLTF_MODEL_MAX_NUM_JOINTS);
    mesh->joint_count = joint_count;

    for (uint32_t j = 0; j < joint_count; j++) {
      gltf_node_t* joint_node = skin->joints[j];
      mat4 joint_world;
      gltf_node_get_world_matrix(joint_node, joint_world);

      /* jointMatrix = inverseTransform * jointWorld * inverseBindMatrix */
      mat4 joint_mat, temp;
      glm_mat4_mul(joint_world, skin->inverse_bind_matrices[j], temp);
      glm_mat4_mul(inverse_transform, temp, joint_mat);

      glm_mat4_copy(joint_mat, mesh->joint_matrices[j]);
    }
  }

  /* Recurse children */
  for (uint32_t c = 0; c < node->child_count; c++) {
    gltf_node_update(node->children[c]);
  }
}

/* Load a single node and its children recursively */
static void load_node(gltf_node_t* parent, const cgltf_node* gltf_node,
                      uint32_t node_index, const cgltf_data* gltf_data,
                      gltf_model_t* model, gltf_loader_info_t* loader_info,
                      float global_scale)
{
  gltf_node_t* node = alloc_node();
  if (!node) {
    return;
  }

  node->index  = node_index;
  node->parent = parent;
  safe_strncpy(node->name, gltf_node->name, GLTF_MODEL_MAX_NAME_LENGTH);

  /* --- Transform --- */
  if (gltf_node->has_translation) {
    glm_vec3_copy((float*)gltf_node->translation, node->translation);
  }
  if (gltf_node->has_rotation) {
    /* cgltf stores rotation as [x, y, z, w], cglm versor is [x, y, z, w] */
    /* Use memcpy instead of glm_vec4_copy to avoid SSE alignment issues
     * with cgltf data which may not be 16-byte aligned. */
    memcpy(node->rotation, gltf_node->rotation, sizeof(vec4));
  }
  if (gltf_node->has_scale) {
    glm_vec3_copy((float*)gltf_node->scale, node->scale);
  }
  if (gltf_node->has_matrix) {
    /* cgltf stores column-major, cglm mat4 is column-major */
    memcpy(node->matrix, gltf_node->matrix, sizeof(mat4));
    node->has_matrix = true;
  }

  /* --- Skin index --- */
  if (gltf_node->skin) {
    node->skin_index = (int32_t)(gltf_node->skin - gltf_data->skins);
  }

  /* --- Children --- */
  if (gltf_node->children_count > 0) {
    node->children
      = (gltf_node_t**)calloc(gltf_node->children_count, sizeof(gltf_node_t*));
    node->child_count = (uint32_t)gltf_node->children_count;
  }

  /* --- Register in linear node list --- */
  model->linear_nodes[model->linear_node_count++] = node;

  /* --- Mesh --- */
  if (gltf_node->mesh) {
    const cgltf_mesh* gltf_mesh = gltf_node->mesh;

    gltf_mesh_t* mesh = (gltf_mesh_t*)calloc(1, sizeof(gltf_mesh_t));
    if (!mesh) {
      gltf_log_error("Failed to allocate mesh");
      return;
    }

    glm_mat4_identity(mesh->matrix);
    gltf_node_get_local_matrix(node, mesh->matrix);
    mesh->index = (uint32_t)(gltf_mesh - gltf_data->meshes);

    /* --- Morph target weights --- */
    if (gltf_mesh->weights_count > 0) {
      mesh->morph_target_count = (uint32_t)gltf_mesh->weights_count;
      mesh->morph_weights
        = (float*)calloc(mesh->morph_target_count, sizeof(float));
      if (mesh->morph_weights) {
        memcpy(mesh->morph_weights, gltf_mesh->weights,
               mesh->morph_target_count * sizeof(float));
      }
      /* Node weights override mesh weights per spec */
      if (gltf_node->weights_count > 0 && mesh->morph_weights) {
        uint32_t copy_count = GLTF_MIN((uint32_t)gltf_node->weights_count,
                                       mesh->morph_target_count);
        memcpy(mesh->morph_weights, gltf_node->weights,
               copy_count * sizeof(float));
      }
    }

    /* --- Primitives --- */
    mesh->primitive_count = (uint32_t)gltf_mesh->primitives_count;
    mesh->primitives      = (gltf_primitive_t*)calloc(mesh->primitive_count,
                                                      sizeof(gltf_primitive_t));
    if (!mesh->primitives) {
      gltf_log_error("Failed to allocate primitives");
      free(mesh);
      return;
    }

    for (uint32_t p = 0; p < mesh->primitive_count; p++) {
      const cgltf_primitive* gltf_prim = &gltf_mesh->primitives[p];
      gltf_primitive_t* prim           = &mesh->primitives[p];

      /* Primitive topology mode */
      switch (gltf_prim->type) {
        case cgltf_primitive_type_points:
          prim->mode = GltfPrimitiveMode_Points;
          break;
        case cgltf_primitive_type_lines:
          prim->mode = GltfPrimitiveMode_Lines;
          break;
        case cgltf_primitive_type_line_loop:
          prim->mode = GltfPrimitiveMode_LineLoop;
          break;
        case cgltf_primitive_type_line_strip:
          prim->mode = GltfPrimitiveMode_LineStrip;
          break;
        case cgltf_primitive_type_triangle_strip:
          prim->mode = GltfPrimitiveMode_TriangleStrip;
          break;
        case cgltf_primitive_type_triangle_fan:
          prim->mode = GltfPrimitiveMode_TriangleFan;
          break;
        default:
          prim->mode = GltfPrimitiveMode_Triangles;
          break;
      }

      /* Material index */
      if (gltf_prim->material) {
        prim->material_index
          = (int32_t)(gltf_prim->material - gltf_data->materials);
      }
      else {
        /* Use the default material (last in the array) */
        prim->material_index = (int32_t)gltf_data->materials_count;
      }

      /* --- Read vertex attributes --- */
      uint32_t vertex_start = (uint32_t)loader_info->vertex_pos;
      uint32_t vertex_count = 0;

      /* Attribute data pointers */
      const float* pos_data     = NULL;
      const float* normal_data  = NULL;
      const float* uv0_data     = NULL;
      const float* uv1_data     = NULL;
      const float* tangent_data = NULL;
      const float* color_data   = NULL;
      const void* joints_data   = NULL;
      const float* weights_data = NULL;

      cgltf_component_type joints_component_type = cgltf_component_type_invalid;
      uint32_t color_components                  = 0;

      /* Byte stride for each attribute */
      size_t pos_stride     = 0;
      size_t normal_stride  = 0;
      size_t uv0_stride     = 0;
      size_t uv1_stride     = 0;
      size_t tangent_stride = 0;
      size_t color_stride   = 0;
      size_t joints_stride  = 0;
      size_t weights_stride = 0;

      for (cgltf_size a = 0; a < gltf_prim->attributes_count; a++) {
        const cgltf_attribute* attr    = &gltf_prim->attributes[a];
        const cgltf_accessor* accessor = attr->data;

        if (!accessor || !accessor->buffer_view) {
          continue;
        }

        const uint8_t* buf_base
          = (const uint8_t*)cgltf_buffer_view_data(accessor->buffer_view);
        if (!buf_base) {
          continue;
        }

        const uint8_t* attr_data_ptr = buf_base + accessor->offset;
        size_t stride                = accessor->stride;

        switch (attr->type) {
          case cgltf_attribute_type_position:
            pos_data     = (const float*)attr_data_ptr;
            pos_stride   = stride;
            vertex_count = (uint32_t)accessor->count;
            /* Bounding box from accessor min/max */
            if (accessor->has_min && accessor->has_max) {
              prim->bb.valid = true;
              glm_vec3_copy((float*)accessor->min, prim->bb.min);
              glm_vec3_copy((float*)accessor->max, prim->bb.max);
              /* Apply global scale */
              glm_vec3_scale(prim->bb.min, global_scale, prim->bb.min);
              glm_vec3_scale(prim->bb.max, global_scale, prim->bb.max);
            }
            break;

          case cgltf_attribute_type_normal:
            normal_data   = (const float*)attr_data_ptr;
            normal_stride = stride;
            break;

          case cgltf_attribute_type_texcoord:
            if (attr->index == 0) {
              uv0_data   = (const float*)attr_data_ptr;
              uv0_stride = stride;
            }
            else if (attr->index == 1) {
              uv1_data   = (const float*)attr_data_ptr;
              uv1_stride = stride;
            }
            break;

          case cgltf_attribute_type_tangent:
            tangent_data   = (const float*)attr_data_ptr;
            tangent_stride = stride;
            break;

          case cgltf_attribute_type_color:
            color_data       = (const float*)attr_data_ptr;
            color_stride     = stride;
            color_components = (uint32_t)cgltf_num_components(accessor->type);
            break;

          case cgltf_attribute_type_joints:
            joints_data           = (const void*)attr_data_ptr;
            joints_stride         = stride;
            joints_component_type = accessor->component_type;
            break;

          case cgltf_attribute_type_weights:
            weights_data   = (const float*)attr_data_ptr;
            weights_stride = stride;
            break;

          default:
            break;
        }
      }

      /* Fill vertex buffer */
      for (uint32_t v = 0; v < vertex_count; v++) {
        gltf_vertex_t* vert
          = &loader_info->vertex_buffer[loader_info->vertex_pos];

        /* Position (required) */
        if (pos_data) {
          const float* pos
            = (const float*)((const uint8_t*)pos_data + v * pos_stride);
          glm_vec3_scale((float*)pos, global_scale, vert->position);
        }

        /* Normal */
        if (normal_data) {
          const float* nrm
            = (const float*)((const uint8_t*)normal_data + v * normal_stride);
          glm_vec3_normalize_to((float*)nrm, vert->normal);
        }
        else {
          glm_vec3_zero(vert->normal);
        }

        /* UV0 */
        if (uv0_data) {
          const float* uv
            = (const float*)((const uint8_t*)uv0_data + v * uv0_stride);
          glm_vec2_copy((float*)uv, vert->uv0);
        }
        else {
          glm_vec2_zero(vert->uv0);
        }

        /* UV1 */
        if (uv1_data) {
          const float* uv
            = (const float*)((const uint8_t*)uv1_data + v * uv1_stride);
          glm_vec2_copy((float*)uv, vert->uv1);
        }
        else {
          glm_vec2_zero(vert->uv1);
        }

        /* Tangent */
        if (tangent_data) {
          const float* tan
            = (const float*)((const uint8_t*)tangent_data + v * tangent_stride);
          glm_vec4_copy((float*)tan, vert->tangent);
        }
        else {
          glm_vec4_zero(vert->tangent);
        }

        /* Color */
        if (color_data) {
          const float* col
            = (const float*)((const uint8_t*)color_data + v * color_stride);
          if (color_components == 3) {
            vert->color[0] = col[0];
            vert->color[1] = col[1];
            vert->color[2] = col[2];
            vert->color[3] = 1.0f;
          }
          else {
            glm_vec4_copy((float*)col, vert->color);
          }
        }
        else {
          glm_vec4_one(vert->color);
        }

        /* Joints */
        if (joints_data) {
          const uint8_t* jdata
            = (const uint8_t*)joints_data + v * joints_stride;
          if (joints_component_type == cgltf_component_type_r_16u) {
            const uint16_t* j16 = (const uint16_t*)jdata;
            vert->joint0[0]     = (uint32_t)j16[0];
            vert->joint0[1]     = (uint32_t)j16[1];
            vert->joint0[2]     = (uint32_t)j16[2];
            vert->joint0[3]     = (uint32_t)j16[3];
          }
          else {
            /* UNSIGNED_BYTE */
            vert->joint0[0] = (uint32_t)jdata[0];
            vert->joint0[1] = (uint32_t)jdata[1];
            vert->joint0[2] = (uint32_t)jdata[2];
            vert->joint0[3] = (uint32_t)jdata[3];
          }
        }
        else {
          memset(vert->joint0, 0, sizeof(vert->joint0));
        }

        /* Weights */
        if (weights_data) {
          const float* w
            = (const float*)((const uint8_t*)weights_data + v * weights_stride);
          glm_vec4_copy((float*)w, vert->weight0);
        }
        else {
          glm_vec4_zero(vert->weight0);
        }

        loader_info->vertex_pos++;
      }

      /* --- Read indices --- */
      prim->vertex_count = vertex_count;

      if (gltf_prim->indices) {
        const cgltf_accessor* idx_acc = gltf_prim->indices;
        prim->has_indices             = true;
        prim->index_count             = (uint32_t)idx_acc->count;
        prim->first_index             = (uint32_t)loader_info->index_pos;

        for (cgltf_size idx = 0; idx < idx_acc->count; idx++) {
          size_t index_val = cgltf_accessor_read_index(idx_acc, idx);
          loader_info->index_buffer[loader_info->index_pos++]
            = (uint32_t)(vertex_start + index_val);
        }
      }
      else {
        prim->has_indices = false;
        prim->first_index = 0;
        prim->index_count = 0;
      }

      /* Update mesh bounding box from primitive */
      if (prim->bb.valid) {
        if (!mesh->bb.valid) {
          glm_vec3_copy(prim->bb.min, mesh->bb.min);
          glm_vec3_copy(prim->bb.max, mesh->bb.max);
          mesh->bb.valid = true;
        }
        else {
          glm_vec3_minv(mesh->bb.min, prim->bb.min, mesh->bb.min);
          glm_vec3_maxv(mesh->bb.max, prim->bb.max, mesh->bb.max);
        }
      }
    } /* end primitives loop */

    node->mesh = mesh;
  } /* end mesh loading */

  /* --- Add to parent's children --- */
  if (parent) {
    for (uint32_t c = 0; c < parent->child_count; c++) {
      if (parent->children[c] == NULL) {
        parent->children[c] = node;
        break;
      }
    }
  }

  /* --- Recurse children --- */
  for (cgltf_size c = 0; c < gltf_node->children_count; c++) {
    uint32_t child_index
      = (uint32_t)(gltf_node->children[c] - gltf_data->nodes);
    load_node(node, gltf_node->children[c], child_index, gltf_data, model,
              loader_info, global_scale);
  }
}

/* -------------------------------------------------------------------------- *
 * Skin loading
 * -------------------------------------------------------------------------- */

static void load_skins(const cgltf_data* gltf_data, gltf_model_t* model)
{
  uint32_t skin_count = (uint32_t)gltf_data->skins_count;
  if (skin_count == 0) {
    return;
  }

  model->skins = (gltf_skin_t*)calloc(skin_count, sizeof(gltf_skin_t));
  if (!model->skins) {
    gltf_log_error("Failed to allocate skins array");
    return;
  }
  model->skin_count = skin_count;

  for (uint32_t i = 0; i < skin_count; i++) {
    const cgltf_skin* gs = &gltf_data->skins[i];
    gltf_skin_t* skin    = &model->skins[i];

    safe_strncpy(skin->name, gs->name, GLTF_MODEL_MAX_NAME_LENGTH);

    /* Skeleton root */
    if (gs->skeleton) {
      uint32_t skel_index = (uint32_t)(gs->skeleton - gltf_data->nodes);
      skin->skeleton_root = gltf_model_find_node(model, skel_index);
    }

    /* Joints */
    skin->joint_count = (uint32_t)gs->joints_count;
    if (skin->joint_count > GLTF_MODEL_MAX_NUM_JOINTS) {
      gltf_log_warn("Skin '%s' has %u joints, max supported is %u", skin->name,
                    skin->joint_count, GLTF_MODEL_MAX_NUM_JOINTS);
    }

    skin->joints
      = (gltf_node_t**)calloc(skin->joint_count, sizeof(gltf_node_t*));
    if (!skin->joints) {
      gltf_log_error("Failed to allocate joints array");
      continue;
    }

    for (uint32_t j = 0; j < skin->joint_count; j++) {
      uint32_t joint_index = (uint32_t)(gs->joints[j] - gltf_data->nodes);
      skin->joints[j]      = gltf_model_find_node(model, joint_index);
    }

    /* Inverse bind matrices */
    skin->inverse_bind_matrices
      = (mat4*)calloc(skin->joint_count, sizeof(mat4));
    if (!skin->inverse_bind_matrices) {
      gltf_log_error("Failed to allocate inverse bind matrices");
      continue;
    }

    if (gs->inverse_bind_matrices) {
      const cgltf_accessor* ibm_acc = gs->inverse_bind_matrices;
      for (uint32_t j = 0; j < skin->joint_count; j++) {
        cgltf_accessor_read_float(ibm_acc, j,
                                  (float*)skin->inverse_bind_matrices[j], 16);
      }
    }
    else {
      /* Default to identity matrices per spec */
      for (uint32_t j = 0; j < skin->joint_count; j++) {
        glm_mat4_identity(skin->inverse_bind_matrices[j]);
      }
    }
  }

  /* Assign skin pointers to nodes */
  for (uint32_t n = 0; n < model->linear_node_count; n++) {
    gltf_node_t* node = model->linear_nodes[n];
    if (node->skin_index >= 0
        && (uint32_t)node->skin_index < model->skin_count) {
      node->skin = &model->skins[node->skin_index];
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Animation loading
 * -------------------------------------------------------------------------- */

static void load_animations(const cgltf_data* gltf_data, gltf_model_t* model)
{
  uint32_t anim_count = (uint32_t)gltf_data->animations_count;
  if (anim_count == 0) {
    return;
  }

  model->animations
    = (gltf_animation_t*)calloc(anim_count, sizeof(gltf_animation_t));
  if (!model->animations) {
    gltf_log_error("Failed to allocate animations array");
    return;
  }
  model->animation_count = anim_count;

  for (uint32_t i = 0; i < anim_count; i++) {
    const cgltf_animation* ga = &gltf_data->animations[i];
    gltf_animation_t* anim    = &model->animations[i];

    safe_strncpy(anim->name, ga->name, GLTF_MODEL_MAX_NAME_LENGTH);
    anim->start_time = FLT_MAX;
    anim->end_time   = -FLT_MAX;

    /* --- Samplers --- */
    anim->sampler_count = (uint32_t)ga->samplers_count;
    anim->samplers      = (gltf_animation_sampler_t*)calloc(
      anim->sampler_count, sizeof(gltf_animation_sampler_t));
    if (!anim->samplers) {
      gltf_log_error("Failed to allocate animation samplers");
      continue;
    }

    for (uint32_t s = 0; s < anim->sampler_count; s++) {
      const cgltf_animation_sampler* gs = &ga->samplers[s];
      gltf_animation_sampler_t* sampler = &anim->samplers[s];

      /* Interpolation type */
      switch (gs->interpolation) {
        case cgltf_interpolation_type_step:
          sampler->interpolation = GltfInterpolation_Step;
          break;
        case cgltf_interpolation_type_cubic_spline:
          sampler->interpolation = GltfInterpolation_CubicSpline;
          break;
        default:
          sampler->interpolation = GltfInterpolation_Linear;
          break;
      }

      /* Input timestamps */
      {
        const cgltf_accessor* input = gs->input;
        sampler->input_count        = (uint32_t)input->count;
        sampler->inputs = (float*)calloc(sampler->input_count, sizeof(float));
        if (sampler->inputs) {
          cgltf_accessor_unpack_floats(input, sampler->inputs,
                                       sampler->input_count);
          /* Update animation time range */
          for (uint32_t t = 0; t < sampler->input_count; t++) {
            float time_val = sampler->inputs[t];
            if (time_val < anim->start_time) {
              anim->start_time = time_val;
            }
            if (time_val > anim->end_time) {
              anim->end_time = time_val;
            }
          }
        }
      }

      /* Output values */
      {
        const cgltf_accessor* output = gs->output;

        if (sampler->interpolation == GltfInterpolation_CubicSpline) {
          /* For cubic spline, store raw floats: 3 * count * num_components
           * Layout per keyframe: [in-tangent, value, out-tangent] */
          uint32_t num_components
            = (uint32_t)cgltf_num_components(output->type);
          sampler->output_count = (uint32_t)output->count * num_components;
          sampler->outputs
            = (float*)calloc(sampler->output_count, sizeof(float));
          if (sampler->outputs) {
            cgltf_accessor_unpack_floats(output, sampler->outputs,
                                         sampler->output_count);
          }
        }
        else {
          /* For LINEAR and STEP, store as vec4 for uniform access */
          sampler->output_vec4_count = (uint32_t)output->count;
          sampler->outputs_vec4
            = (vec4*)calloc(sampler->output_vec4_count, sizeof(vec4));
          if (sampler->outputs_vec4) {
            uint32_t num_components
              = (uint32_t)cgltf_num_components(output->type);
            for (uint32_t o = 0; o < sampler->output_vec4_count; o++) {
              float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
              cgltf_accessor_read_float(output, o, values, num_components);
              glm_vec4_copy(values, sampler->outputs_vec4[o]);
            }
          }
        }
      }
    }

    /* --- Channels --- */
    anim->channel_count = (uint32_t)ga->channels_count;
    anim->channels      = (gltf_animation_channel_t*)calloc(
      anim->channel_count, sizeof(gltf_animation_channel_t));
    if (!anim->channels) {
      gltf_log_error("Failed to allocate animation channels");
      continue;
    }

    for (uint32_t c = 0; c < anim->channel_count; c++) {
      const cgltf_animation_channel* gc = &ga->channels[c];
      gltf_animation_channel_t* channel = &anim->channels[c];

      /* Target path */
      switch (gc->target_path) {
        case cgltf_animation_path_type_translation:
          channel->path = GltfAnimPath_Translation;
          break;
        case cgltf_animation_path_type_rotation:
          channel->path = GltfAnimPath_Rotation;
          break;
        case cgltf_animation_path_type_scale:
          channel->path = GltfAnimPath_Scale;
          break;
        case cgltf_animation_path_type_weights:
          channel->path = GltfAnimPath_Weights;
          break;
        default:
          channel->path = GltfAnimPath_Translation;
          break;
      }

      /* Sampler index */
      channel->sampler_index = (uint32_t)(gc->sampler - ga->samplers);

      /* Target node */
      if (gc->target_node) {
        uint32_t target_index = (uint32_t)(gc->target_node - gltf_data->nodes);
        channel->node         = gltf_model_find_node(model, target_index);
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Animation update
 * -------------------------------------------------------------------------- */

/**
 * @brief Hermite cubic spline interpolation per glTF 2.0 Appendix C.
 *
 * v(t) = (2t³ - 3t² + 1)v_k + t_d(t³ - 2t² + t)b_k
 *      + (-2t³ + 3t²)v_{k+1} + t_d(t³ - t²)a_{k+1}
 */
static void cubic_spline_interpolation(const float* outputs, uint32_t index,
                                       float t, uint32_t stride, float td,
                                       float* result)
{
  /* Each keyframe has stride*3 floats: [in-tangent, value, out-tangent] */
  const float* vk  = &outputs[index * stride * 3 + stride]; /* value k      */
  const float* bk  = &outputs[index * stride * 3];          /* in-tangent k  */
  const float* vk1 = &outputs[(index + 1) * stride * 3 + stride]; /* val k+1*/
  const float* ak1
    = &outputs[(index + 1) * stride * 3 + stride * 2]; /* out k+1*/

  float t2 = t * t;
  float t3 = t2 * t;

  for (uint32_t i = 0; i < stride; i++) {
    result[i] = (2.0f * t3 - 3.0f * t2 + 1.0f) * vk[i]
                + td * (t3 - 2.0f * t2 + t) * bk[i]
                + (-2.0f * t3 + 3.0f * t2) * vk1[i] + td * (t3 - t2) * ak1[i];
  }
}

void gltf_model_update_animation(gltf_model_t* model, uint32_t animation_index,
                                 float time)
{
  if (!model || animation_index >= model->animation_count) {
    return;
  }

  gltf_animation_t* anim = &model->animations[animation_index];
  float duration         = anim->end_time - anim->start_time;

  /* Wrap time to animation range */
  if (duration > 0.0f) {
    time = fmodf(time - anim->start_time, duration) + anim->start_time;
  }

  for (uint32_t c = 0; c < anim->channel_count; c++) {
    gltf_animation_channel_t* channel = &anim->channels[c];
    gltf_animation_sampler_t* sampler = &anim->samplers[channel->sampler_index];

    if (!channel->node || sampler->input_count == 0) {
      continue;
    }

    /* Find the keyframe interval */
    uint32_t i0 = 0;
    for (uint32_t k = 0; k < sampler->input_count - 1; k++) {
      if (time >= sampler->inputs[k] && time <= sampler->inputs[k + 1]) {
        i0 = k;
        break;
      }
    }

    /* Clamp to last keyframe if beyond range */
    if (time > sampler->inputs[sampler->input_count - 1]) {
      i0 = sampler->input_count - 2;
    }
    if (sampler->input_count == 1) {
      i0 = 0;
    }

    /* Compute interpolation factor */
    float t0 = sampler->inputs[i0];
    float t1 = (i0 + 1 < sampler->input_count) ? sampler->inputs[i0 + 1] : t0;
    float td = t1 - t0;
    float u  = (td > 0.0f) ? (time - t0) / td : 0.0f;
    u        = GLTF_MAX(0.0f, GLTF_MIN(1.0f, u));

    gltf_node_t* node = channel->node;

    switch (channel->path) {
      case GltfAnimPath_Translation: {
        if (sampler->interpolation == GltfInterpolation_CubicSpline) {
          float result[3];
          cubic_spline_interpolation(sampler->outputs, i0, u, 3, td, result);
          glm_vec3_copy(result, node->translation);
        }
        else if (sampler->interpolation == GltfInterpolation_Step) {
          glm_vec3_copy(sampler->outputs_vec4[i0], node->translation);
        }
        else {
          /* LINEAR */
          glm_vec3_lerp(sampler->outputs_vec4[i0],
                        sampler->outputs_vec4[i0 + 1], u, node->translation);
        }
        break;
      }

      case GltfAnimPath_Rotation: {
        if (sampler->interpolation == GltfInterpolation_CubicSpline) {
          float result[4];
          cubic_spline_interpolation(sampler->outputs, i0, u, 4, td, result);
          glm_quat_normalize_to(result, node->rotation);
        }
        else if (sampler->interpolation == GltfInterpolation_Step) {
          glm_vec4_copy(sampler->outputs_vec4[i0], node->rotation);
        }
        else {
          /* LINEAR: slerp for quaternions */
          glm_quat_slerp(sampler->outputs_vec4[i0],
                         sampler->outputs_vec4[i0 + 1], u, node->rotation);
        }
        break;
      }

      case GltfAnimPath_Scale: {
        if (sampler->interpolation == GltfInterpolation_CubicSpline) {
          float result[3];
          cubic_spline_interpolation(sampler->outputs, i0, u, 3, td, result);
          glm_vec3_copy(result, node->scale);
        }
        else if (sampler->interpolation == GltfInterpolation_Step) {
          glm_vec3_copy(sampler->outputs_vec4[i0], node->scale);
        }
        else {
          /* LINEAR */
          glm_vec3_lerp(sampler->outputs_vec4[i0],
                        sampler->outputs_vec4[i0 + 1], u, node->scale);
        }
        break;
      }

      case GltfAnimPath_Weights: {
        if (!node->mesh || !node->mesh->morph_weights) {
          break;
        }
        uint32_t num_weights = node->mesh->morph_target_count;
        if (sampler->interpolation == GltfInterpolation_CubicSpline) {
          /* Cubic spline for weights: stride = num_weights */
          float result[GLTF_MODEL_MAX_NUM_JOINTS];
          uint32_t clamped = GLTF_MIN(num_weights, GLTF_MODEL_MAX_NUM_JOINTS);
          cubic_spline_interpolation(sampler->outputs, i0, u, clamped, td,
                                     result);
          memcpy(node->mesh->morph_weights, result, clamped * sizeof(float));
        }
        else if (sampler->interpolation == GltfInterpolation_Step) {
          /* Step: use outputs_vec4 which stores scalar weights sequentially */
          for (uint32_t w = 0; w < num_weights; w++) {
            uint32_t out_idx = i0 * num_weights + w;
            if (out_idx < sampler->output_vec4_count) {
              node->mesh->morph_weights[w] = sampler->outputs_vec4[out_idx][0];
            }
          }
        }
        else {
          /* LINEAR */
          for (uint32_t w = 0; w < num_weights; w++) {
            uint32_t idx0 = i0 * num_weights + w;
            uint32_t idx1 = (i0 + 1) * num_weights + w;
            if (idx0 < sampler->output_vec4_count
                && idx1 < sampler->output_vec4_count) {
              node->mesh->morph_weights[w]
                = glm_lerp(sampler->outputs_vec4[idx0][0],
                           sampler->outputs_vec4[idx1][0], u);
            }
          }
        }
        break;
      }
    }
  }

  /* Update all node transforms after animation */
  for (uint32_t n = 0; n < model->node_count; n++) {
    gltf_node_update(model->nodes[n]);
  }
}

/* -------------------------------------------------------------------------- *
 * Scene dimensions
 * -------------------------------------------------------------------------- */

static void calculate_bounding_box(gltf_node_t* node, gltf_node_t* parent)
{
  gltf_bounding_box_t parent_bvh;
  memset(&parent_bvh, 0, sizeof(parent_bvh));

  if (parent) {
    parent_bvh = parent->bvh;
  }

  if (node->mesh && node->mesh->bb.valid) {
    mat4 world_mat;
    gltf_node_get_world_matrix(node, world_mat);

    gltf_bounding_box_get_aabb(&node->mesh->bb, world_mat, &node->aabb);

    if (parent_bvh.valid) {
      node->bvh.min[0] = GLTF_MIN(parent_bvh.min[0], node->aabb.min[0]);
      node->bvh.min[1] = GLTF_MIN(parent_bvh.min[1], node->aabb.min[1]);
      node->bvh.min[2] = GLTF_MIN(parent_bvh.min[2], node->aabb.min[2]);
      node->bvh.max[0] = GLTF_MAX(parent_bvh.max[0], node->aabb.max[0]);
      node->bvh.max[1] = GLTF_MAX(parent_bvh.max[1], node->aabb.max[1]);
      node->bvh.max[2] = GLTF_MAX(parent_bvh.max[2], node->aabb.max[2]);
    }
    else {
      glm_vec3_copy(node->aabb.min, node->bvh.min);
      glm_vec3_copy(node->aabb.max, node->bvh.max);
    }
    node->bvh.valid = true;
  }

  for (uint32_t c = 0; c < node->child_count; c++) {
    calculate_bounding_box(node->children[c], node);
  }
}

void gltf_model_get_scene_dimensions(gltf_model_t* model)
{
  /* Initialize to extreme values */
  glm_vec3_fill(model->dimensions.min, FLT_MAX);
  glm_vec3_fill(model->dimensions.max, -FLT_MAX);

  /* Calculate bounding boxes for all nodes */
  for (uint32_t n = 0; n < model->node_count; n++) {
    calculate_bounding_box(model->nodes[n], NULL);
  }

  /* Find global min/max from all nodes */
  for (uint32_t n = 0; n < model->linear_node_count; n++) {
    gltf_node_t* node = model->linear_nodes[n];
    if (node->bvh.valid) {
      glm_vec3_minv(model->dimensions.min, node->bvh.min,
                    model->dimensions.min);
      glm_vec3_maxv(model->dimensions.max, node->bvh.max,
                    model->dimensions.max);
    }
  }

  /* Compute derived values */
  glm_vec3_sub(model->dimensions.max, model->dimensions.min,
               model->dimensions.size);
  glm_vec3_add(model->dimensions.min, model->dimensions.max,
               model->dimensions.center);
  glm_vec3_scale(model->dimensions.center, 0.5f, model->dimensions.center);
  model->dimensions.radius
    = glm_vec3_distance(model->dimensions.min, model->dimensions.max) * 0.5f;
}

/* -------------------------------------------------------------------------- *
 * Node finding
 * -------------------------------------------------------------------------- */

gltf_node_t* gltf_model_find_node(gltf_model_t* model, uint32_t index)
{
  for (uint32_t n = 0; n < model->linear_node_count; n++) {
    if (model->linear_nodes[n]->index == index) {
      return model->linear_nodes[n];
    }
  }
  return NULL;
}

/* -------------------------------------------------------------------------- *
 * Extensions loading
 * -------------------------------------------------------------------------- */

static void load_extensions(const cgltf_data* gltf_data, gltf_model_t* model)
{
  uint32_t ext_count = (uint32_t)gltf_data->extensions_used_count;
  if (ext_count == 0) {
    return;
  }

  model->extensions = (char**)calloc(ext_count, sizeof(char*));
  if (!model->extensions) {
    return;
  }
  model->extension_count = ext_count;

  for (uint32_t i = 0; i < ext_count; i++) {
    size_t len           = strlen(gltf_data->extensions_used[i]) + 1;
    model->extensions[i] = (char*)calloc(len, sizeof(char));
    if (model->extensions[i]) {
      safe_strncpy(model->extensions[i], gltf_data->extensions_used[i], len);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Core loading implementation
 * -------------------------------------------------------------------------- */

static bool load_model_internal(gltf_model_t* model,
                                const cgltf_data* gltf_data, float scale)
{
  /* --- Load extensions --- */
  load_extensions(gltf_data, model);

  /* --- Load texture samplers and images --- */
  load_texture_samplers(gltf_data, model);
  load_texture_images(gltf_data, model);

  /* --- Load materials --- */
  load_materials(gltf_data, model);

  /* --- Pre-count vertices and indices --- */
  size_t total_vertex_count = 0;
  size_t total_index_count  = 0;

  const cgltf_scene* scene = gltf_data->scene;
  if (!scene && gltf_data->scenes_count > 0) {
    scene = &gltf_data->scenes[0];
  }
  if (!scene) {
    gltf_log_error("No scene found in glTF file");
    return false;
  }

  for (cgltf_size n = 0; n < scene->nodes_count; n++) {
    get_node_vertex_index_count(scene->nodes[n], &total_vertex_count,
                                &total_index_count);
  }

  /* --- Allocate geometry buffers --- */
  gltf_loader_info_t loader_info = {0};

  if (total_vertex_count > 0) {
    loader_info.vertex_buffer
      = (gltf_vertex_t*)calloc(total_vertex_count, sizeof(gltf_vertex_t));
    if (!loader_info.vertex_buffer) {
      gltf_log_error("Failed to allocate vertex buffer (%zu vertices)",
                     total_vertex_count);
      return false;
    }
  }

  if (total_index_count > 0) {
    loader_info.index_buffer
      = (uint32_t*)calloc(total_index_count, sizeof(uint32_t));
    if (!loader_info.index_buffer) {
      gltf_log_error("Failed to allocate index buffer (%zu indices)",
                     total_index_count);
      GLTF_SAFE_FREE(loader_info.vertex_buffer);
      return false;
    }
  }

  /* --- Allocate linear node list (upper bound: all nodes) --- */
  model->linear_nodes
    = (gltf_node_t**)calloc(gltf_data->nodes_count, sizeof(gltf_node_t*));
  if (!model->linear_nodes) {
    gltf_log_error("Failed to allocate linear nodes array");
    GLTF_SAFE_FREE(loader_info.vertex_buffer);
    GLTF_SAFE_FREE(loader_info.index_buffer);
    return false;
  }

  /* --- Load scene nodes --- */
  model->node_count = (uint32_t)scene->nodes_count;
  model->nodes = (gltf_node_t**)calloc(model->node_count, sizeof(gltf_node_t*));
  if (!model->nodes) {
    gltf_log_error("Failed to allocate scene nodes array");
    GLTF_SAFE_FREE(loader_info.vertex_buffer);
    GLTF_SAFE_FREE(loader_info.index_buffer);
    return false;
  }

  for (cgltf_size n = 0; n < scene->nodes_count; n++) {
    uint32_t node_index = (uint32_t)(scene->nodes[n] - gltf_data->nodes);
    load_node(NULL, scene->nodes[n], node_index, gltf_data, model, &loader_info,
              scale);

    /* Store root node pointer */
    model->nodes[n] = model->linear_nodes[model->linear_node_count - 1];
    /* Walk up to find the actual root just loaded (the one without parent) */
    gltf_node_t* root = model->nodes[n];
    while (root->parent) {
      root = root->parent;
    }
    model->nodes[n] = root;
  }

  /* --- Transfer geometry ownership to model --- */
  model->vertices     = loader_info.vertex_buffer;
  model->vertex_count = (uint32_t)loader_info.vertex_pos;
  model->indices      = loader_info.index_buffer;
  model->index_count  = (uint32_t)loader_info.index_pos;

  /* --- Load skins (needs nodes to be loaded first) --- */
  load_skins(gltf_data, model);

  /* --- Load animations (needs nodes to be loaded first) --- */
  load_animations(gltf_data, model);

  /* --- Initial node update for skinning --- */
  for (uint32_t n = 0; n < model->node_count; n++) {
    gltf_node_update(model->nodes[n]);
  }

  /* --- Compute scene dimensions --- */
  gltf_model_get_scene_dimensions(model);

  return true;
}

/* -------------------------------------------------------------------------- *
 * Public API implementation
 * -------------------------------------------------------------------------- */

bool gltf_model_load_from_file(gltf_model_t* model, const char* filename,
                               float scale)
{
  if (!model || !filename) {
    gltf_log_error("Invalid arguments to gltf_model_load_from_file");
    return false;
  }

  memset(model, 0, sizeof(gltf_model_t));

  /* Extract directory path for relative URI resolution */
  extract_directory(filename, model->file_path, sizeof(model->file_path));

  /* Parse glTF file */
  cgltf_options options = {0};
  cgltf_data* gltf_data = NULL;
  cgltf_result result   = cgltf_parse_file(&options, filename, &gltf_data);

  if (result != cgltf_result_success) {
    gltf_log_error("Failed to parse glTF file: %s (error: %d)", filename,
                   (int)result);
    memset(model, 0, sizeof(gltf_model_t));
    return false;
  }

  /* Load binary buffers */
  result = cgltf_load_buffers(&options, gltf_data, filename);
  if (result != cgltf_result_success) {
    gltf_log_error("Failed to load glTF buffers: %s (error: %d)", filename,
                   (int)result);
    cgltf_free(gltf_data);
    memset(model, 0, sizeof(gltf_model_t));
    return false;
  }

  /* Validate */
  result = cgltf_validate(gltf_data);
  if (result != cgltf_result_success) {
    gltf_log_warn("glTF validation warning: %s (error: %d)", filename,
                  (int)result);
    /* Continue despite validation warnings */
  }

  /* Load model */
  bool success = load_model_internal(model, gltf_data, scale);

  /* Free cgltf data (all data has been copied) */
  cgltf_free(gltf_data);

  if (!success) {
    gltf_model_destroy(model);
    return false;
  }

  return true;
}

bool gltf_model_load_from_memory(gltf_model_t* model, const void* data,
                                 size_t size, const char* base_path,
                                 float scale)
{
  if (!model || !data || size == 0) {
    gltf_log_error("Invalid arguments to gltf_model_load_from_memory");
    return false;
  }

  memset(model, 0, sizeof(gltf_model_t));

  if (base_path) {
    extract_directory(base_path, model->file_path, sizeof(model->file_path));
  }

  /* Parse from memory */
  cgltf_options options = {0};
  cgltf_data* gltf_data = NULL;
  cgltf_result result   = cgltf_parse(&options, data, size, &gltf_data);

  if (result != cgltf_result_success) {
    gltf_log_error("Failed to parse glTF from memory (error: %d)", (int)result);
    memset(model, 0, sizeof(gltf_model_t));
    return false;
  }

  /* For GLB, binary buffer is embedded; for glTF, try to load external buffers
   */
  if (base_path) {
    result = cgltf_load_buffers(&options, gltf_data, base_path);
    if (result != cgltf_result_success) {
      gltf_log_warn("Failed to load external buffers (error: %d)", (int)result);
    }
  }

  /* Validate */
  result = cgltf_validate(gltf_data);
  if (result != cgltf_result_success) {
    gltf_log_warn("glTF validation warning (error: %d)", (int)result);
  }

  /* Load model */
  bool success = load_model_internal(model, gltf_data, scale);

  /* Free cgltf data */
  cgltf_free(gltf_data);

  if (!success) {
    gltf_model_destroy(model);
    return false;
  }

  return true;
}

/* -------------------------------------------------------------------------- *
 * Cleanup
 * -------------------------------------------------------------------------- */

static void destroy_node(gltf_node_t* node)
{
  if (!node) {
    return;
  }

  /* Destroy children recursively */
  for (uint32_t c = 0; c < node->child_count; c++) {
    destroy_node(node->children[c]);
  }

  /* Free mesh */
  if (node->mesh) {
    GLTF_SAFE_FREE(node->mesh->primitives);
    GLTF_SAFE_FREE(node->mesh->morph_weights);
    free(node->mesh);
    node->mesh = NULL;
  }

  /* Free children array */
  GLTF_SAFE_FREE(node->children);

  /* Free node itself */
  free(node);
}

void gltf_model_destroy(gltf_model_t* model)
{
  if (!model) {
    return;
  }

  /* Destroy nodes (recursive) */
  if (model->nodes) {
    for (uint32_t n = 0; n < model->node_count; n++) {
      destroy_node(model->nodes[n]);
    }
    GLTF_SAFE_FREE(model->nodes);
  }

  /* Free linear nodes array (nodes themselves already freed above) */
  GLTF_SAFE_FREE(model->linear_nodes);

  /* Free geometry buffers */
  GLTF_SAFE_FREE(model->vertices);
  GLTF_SAFE_FREE(model->indices);

  /* Free textures */
  if (model->textures) {
    for (uint32_t i = 0; i < model->texture_count; i++) {
      if (model->textures[i].data) {
        image_free(model->textures[i].data);
        model->textures[i].data = NULL;
      }
    }
    GLTF_SAFE_FREE(model->textures);
  }

  /* Free materials */
  GLTF_SAFE_FREE(model->materials);

  /* Free skins */
  if (model->skins) {
    for (uint32_t i = 0; i < model->skin_count; i++) {
      GLTF_SAFE_FREE(model->skins[i].joints);
      GLTF_SAFE_FREE(model->skins[i].inverse_bind_matrices);
    }
    GLTF_SAFE_FREE(model->skins);
  }

  /* Free animations */
  if (model->animations) {
    for (uint32_t i = 0; i < model->animation_count; i++) {
      gltf_animation_t* anim = &model->animations[i];
      if (anim->samplers) {
        for (uint32_t s = 0; s < anim->sampler_count; s++) {
          GLTF_SAFE_FREE(anim->samplers[s].inputs);
          GLTF_SAFE_FREE(anim->samplers[s].outputs_vec4);
          GLTF_SAFE_FREE(anim->samplers[s].outputs);
        }
        GLTF_SAFE_FREE(anim->samplers);
      }
      GLTF_SAFE_FREE(anim->channels);
    }
    GLTF_SAFE_FREE(model->animations);
  }

  /* Free extensions */
  if (model->extensions) {
    for (uint32_t i = 0; i < model->extension_count; i++) {
      GLTF_SAFE_FREE(model->extensions[i]);
    }
    GLTF_SAFE_FREE(model->extensions);
  }

  /* Zero the struct */
  memset(model, 0, sizeof(gltf_model_t));
}
