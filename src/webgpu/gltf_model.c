#include "gltf_model.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <cgltf.h>

#include "../core/file.h"
#include "../core/log.h"
#include "../core/macro.h"

/*
 * Forward declarations
 */
struct gltf_model_t;
struct gltf_node_t;
static struct gltf_node_t*
gltf_model_node_from_index(struct gltf_model_t* model, uint32_t index);
static void gltf_model_get_scene_dimensions(struct gltf_model_t* model);

/*
 * glTF enums
 */
typedef wgpu_gltf_alpha_mode_enum_t alpha_mode_enum;

/*
 * Bounding box
 */
typedef struct bounding_box_t {
  vec3 min;
  vec3 max;
  bool valid;
} bounding_box_t;

static void bounding_box_init(bounding_box_t* bounding_box, vec3 min, vec3 max)
{
  glm_vec3_copy(min, bounding_box->min);
  glm_vec3_copy(max, bounding_box->max);
  bounding_box->valid = false;
}

static vec3* vec3_min(vec3* x, vec3* y)
{
  return ((*y)[0] < (*x)[0] && (*y)[1] < (*x)[1] && (*y)[2] < (*x)[2]) ? y : x;
}

static vec3* vec3_max(vec3* x, vec3* y)
{
  return ((*x)[0] < (*y)[0] && (*x)[1] < (*y)[1] && (*x)[2] < (*y)[2]) ? y : x;
}

static void bounding_get_aabb(bounding_box_t* bounding_box, mat4 m,
                              bounding_box_t* dest)
{
  vec3 min = {m[0][3], m[0][3], m[0][3]};
  vec3 max = GLM_VEC3_ZERO_INIT;
  glm_vec3_copy(min, max);
  vec3 v0 = GLM_VEC3_ZERO_INIT, v1 = GLM_VEC3_ZERO_INIT;

  vec3 right = {m[0][0], m[0][0], m[0][0]};
  glm_vec3_scale(right, bounding_box->min[0], v0);
  glm_vec3_scale(right, bounding_box->max[0], v1);
  glm_vec3_add(min, *vec3_min(&v0, &v1), min);
  glm_vec3_add(max, *vec3_max(&v0, &v1), max);

  vec3 up = {m[0][1], m[0][1], m[0][1]};
  glm_vec3_scale(up, bounding_box->min[1], v0);
  glm_vec3_scale(up, bounding_box->max[1], v1);
  glm_vec3_add(min, *vec3_min(&v0, &v1), min);
  glm_vec3_add(max, *vec3_max(&v0, &v1), max);

  vec3 back = {m[0][2], m[0][2], m[0][2]};
  glm_vec3_scale(back, bounding_box->min[2], v0);
  glm_vec3_scale(back, bounding_box->max[2], v1);
  glm_vec3_add(min, *vec3_min(&v0, &v1), min);
  glm_vec3_add(max, *vec3_max(&v0, &v1), max);

  bounding_box_init(dest, min, max);
}

/*
 * glTF texture sampler
 */
typedef struct gltf_texture_sampler_t {
  WGPUFilterMode mag_filter;
  WGPUFilterMode min_filter;
  WGPUAddressMode address_mode_u;
  WGPUAddressMode address_mode_v;
  WGPUAddressMode address_mode_w;
} gltf_texture_sampler_t;

static WGPUAddressMode get_wgpu_wrap_mode(int32_t wrap_mode)
{
  switch (wrap_mode) {
    case 10497:
      return WGPUAddressMode_Repeat;
    case 33071:
      return WGPUAddressMode_ClampToEdge;
    case 33648:
      return WGPUAddressMode_MirrorRepeat;
    default:
      break;
  }

  log_warn("Unknown wrap mode for get_wgpu_wrap_mode: %d", wrap_mode);
  return WGPUAddressMode_Repeat;
}

static WGPUFilterMode get_wgpu_filter_mode(int32_t filterMode)
{
  switch (filterMode) {
    case 9728:
      return WGPUFilterMode_Nearest;
    case 9729:
      return WGPUFilterMode_Linear;
    case 9984:
      return WGPUFilterMode_Nearest;
    case 9985:
      return WGPUFilterMode_Nearest;
    case 9986:
      return WGPUFilterMode_Linear;
    case 9987:
      return WGPUFilterMode_Linear;
    default:
      break;
  }

  log_warn("Unknown filter mode for get_wgpu_filter_mode: %d", filterMode);
  return WGPUFilterMode_Nearest;
}

/*
 * glTF texture loading
 */
typedef wgpu_gltf_texture_t gltf_texture_t;

static void gltf_texture_init(gltf_texture_t* texture,
                              wgpu_context_t* wgpu_context)
{
  texture->wgpu_context = wgpu_context;
}

static void gltf_texture_destroy(gltf_texture_t* texture)
{
  if (!texture) {
    return;
  }

  wgpu_destroy_texture(&texture->wgpu_texture);
}

static void get_relative_file_path(const char* base_path, const char* new_path,
                                   char* result)
{
  snprintf(result, strlen(base_path) + 1, "%s", base_path);
  char* insert_point = strrchr(result, '/');
  if (insert_point) {
    insert_point++;
  }
  else {
    insert_point = result;
  }
  snprintf(insert_point, strlen(new_path) + 1, "%s", new_path);
}

static void gltf_texture_from_gltf_image(const char* model_uri,
                                         gltf_texture_t* texture,
                                         cgltf_image* gltf_image)
{
  ASSERT(texture && texture->wgpu_context != NULL);

  if (gltf_image->uri != NULL) {
    /* Load image data from file */
    char image_uri[STRMAX];
    get_relative_file_path(model_uri, gltf_image->uri, image_uri);
    if (filename_has_extension(image_uri, "jpg")
        || filename_has_extension(image_uri, "png")
        || filename_has_extension(image_uri, "ktx")) {
      texture->wgpu_texture = wgpu_create_texture_from_file(
        texture->wgpu_context, image_uri,
        &(struct wgpu_texture_load_options_t){
          .generate_mipmaps = true,
          .address_mode     = WGPUAddressMode_Repeat,
        });
    }
  }
  else if (gltf_image->buffer_view) {
    /* Load image data from memory */
    texture->wgpu_texture = wgpu_create_texture_from_memory(
      texture->wgpu_context,
      (void*)((uint8_t*)gltf_image->buffer_view->buffer->data
              + gltf_image->buffer_view->offset),
      gltf_image->buffer_view->size, NULL);
  }
}

/*
 * glTF material
 */
typedef wgpu_gltf_material_t gltf_material_t;

static void gltf_material_init(gltf_material_t* material,
                               wgpu_context_t* wgpu_context)
{
  ASSERT(material != NULL);

  material->wgpu_context     = wgpu_context;
  material->alpha_mode       = AlphaMode_OPAQUE;
  material->blend            = false;
  material->double_sided     = false;
  material->alpha_cutoff     = 1.0f;
  material->metallic_factor  = 1.0f;
  material->roughness_factor = 1.0f;
  glm_vec4_one(material->base_color_factor);
  glm_vec4_one(material->emissive_factor);
  material->base_color_texture                    = NULL;
  material->metallic_roughness_texture            = NULL;
  material->normal_texture                        = NULL;
  material->occlusion_texture                     = NULL;
  material->emissive_texture                      = NULL;
  material->tex_coord_sets.base_color             = 0;
  material->tex_coord_sets.metallic_roughness     = 0;
  material->tex_coord_sets.specular_glossiness    = 0;
  material->tex_coord_sets.normal                 = 0;
  material->tex_coord_sets.occlusion              = 0;
  material->tex_coord_sets.emissive               = 0;
  material->extension.specular_glossiness_texture = NULL;
  material->extension.diffuse_texture             = NULL;
  glm_vec4_one(material->extension.diffuse_factor);
  glm_vec3_zero(material->extension.specular_factor);
  material->pbr_workflows.metallic_roughness  = true;
  material->pbr_workflows.specular_glossiness = false;
  material->bind_group                        = NULL;
  material->pipeline                          = NULL;
}

static void gltf_material_destroy(gltf_material_t* material)
{
  WGPU_RELEASE_RESOURCE(BindGroup, material->bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, material->pipeline)
}

/*
 * glTF primitive
 */
typedef struct gltf_primitive_t {
  uint32_t first_index;
  uint32_t index_count;
  uint32_t first_vertex;
  uint32_t vertex_count;
  gltf_material_t* material;
  bool has_indices;
  bounding_box_t bb;
} gltf_primitive_t;

static void gltf_primitive_init(gltf_primitive_t* primitive,
                                uint32_t first_index, uint32_t index_count,
                                gltf_material_t* material)
{
  primitive->first_index  = first_index;
  primitive->index_count  = index_count;
  primitive->first_vertex = 0;
  primitive->vertex_count = 0;
  primitive->material     = material;
  primitive->has_indices  = index_count > 0;
  bounding_box_init(&primitive->bb, GLM_VEC3_ZERO, GLM_VEC3_ZERO);
}

static void gltf_primitive_set_bounding_box(gltf_primitive_t* primitive,
                                            vec3 min, vec3 max)
{
  glm_vec3_copy(min, primitive->bb.min);
  glm_vec3_copy(max, primitive->bb.max);
  primitive->bb.valid = true;
}

/*
 * glTF mesh
 */
typedef struct gltf_mesh_t {
  wgpu_context_t* wgpu_context;
  gltf_primitive_t* primitives;
  uint32_t primitive_count;
  char name[STRMAX];
  bounding_box_t bb;
  bounding_box_t aabb;
  struct {
    wgpu_buffer_t buffer;
    WGPUBindGroup bind_group;
  } uniform_buffer;
  struct {
    mat4 matrix;
    mat4 joint_matrix[WGPU_GLTF_MAX_NUM_JOINTS];
    float joint_count;
  } uniform_block;
} gltf_mesh_t;

static void gltf_mesh_init(gltf_mesh_t* mesh, wgpu_context_t* wgpu_context,
                           mat4 matrix)
{
  memset(mesh, 0, sizeof(gltf_mesh_t));

  mesh->wgpu_context = wgpu_context;
  glm_mat4_copy(matrix, mesh->uniform_block.matrix);
  mesh->uniform_buffer.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Object vertex shader uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(mesh->uniform_block),
                    .initial.data = &mesh->uniform_block,
                  });
}

static void gltf_mesh_destroy(gltf_mesh_t* mesh)
{
  wgpu_destroy_buffer(&mesh->uniform_buffer.buffer);
  WGPU_RELEASE_RESOURCE(BindGroup, mesh->uniform_buffer.bind_group);

  if (mesh->primitives != NULL) {
    free(mesh->primitives);
  }
}

static void gltf_mesh_set_bounding_box(gltf_mesh_t* mesh, vec3 min, vec3 max)
{
  glm_vec3_copy(min, mesh->bb.min);
  glm_vec3_copy(max, mesh->bb.max);
  mesh->bb.valid = true;
}

/*
 * glTF skin
 */
typedef struct gltf_skin_t {
  char name[STRMAX];
  struct gltf_node_t* skeleton_root;
  mat4* inverse_bind_matrices;
  uint32_t inverse_bind_matrix_count;
  struct gltf_node_t** joints;
  uint32_t current_joint_index;
  uint32_t joint_count;
  struct {
    WGPUBuffer buffer;
    uint64_t size;
    WGPUBindGroup bind_group;
  } ssbo;
} gltf_skin_t;

static void gltf_skin_destroy(gltf_skin_t* skin)
{
  if (skin->inverse_bind_matrix_count > 0) {
    free(skin->inverse_bind_matrices);
  }
  if (skin->joint_count > 0) {
    free(skin->joints);
  }
}

/*
 * glTF node
 */
typedef struct gltf_node_t {
  struct gltf_node_t* parent;
  uint32_t index;
  struct gltf_node_t** children;
  uint32_t current_child_index;
  uint32_t child_count;
  mat4 matrix;
  char name[STRMAX];
  gltf_mesh_t* mesh;
  gltf_skin_t* skin;
  int32_t skin_index;
  vec3 translation;
  vec3 scale;
  versor rotation;
  bounding_box_t bvh;
  bounding_box_t aabb;
} gltf_node_t;

static void gltf_node_init(gltf_node_t* node)
{
  node->parent              = NULL;
  node->children            = NULL;
  node->current_child_index = 0;
  node->child_count         = 0;
  node->mesh                = NULL;
  node->skin                = NULL;
  node->skin_index          = -1;
  glm_mat4_identity(node->matrix);
  glm_vec3_zero(node->translation);
  glm_vec3_one(node->scale);
  glm_quat_identity(node->rotation);
  bounding_box_init(&node->bvh, GLM_VEC3_ZERO, GLM_VEC3_ZERO);
  bounding_box_init(&node->aabb, GLM_VEC3_ZERO, GLM_VEC3_ZERO);
}

static void glm_cast_versor_to_mat3(versor q, mat3* result)
{
  float qxx = q[0] * q[0];
  float qyy = q[1] * q[1];
  float qzz = q[2] * q[2];
  float qxz = q[0] * q[2];
  float qxy = q[0] * q[1];
  float qyz = q[1] * q[2];
  float qwx = q[3] * q[0];
  float qwy = q[3] * q[1];
  float qwz = q[3] * q[2];

  (*result)[0][0] = 1.0f - 2.0f * (qyy + qzz);
  (*result)[0][1] = 2.0f * (qxy + qwz);
  (*result)[0][2] = 2.0f * (qxz - qwy);

  (*result)[1][0] = 2.0f * (qxy - qwz);
  (*result)[1][1] = 1.0f - 2.0f * (qxx + qzz);
  (*result)[1][2] = 2.0f * (qyz + qwx);

  (*result)[2][0] = 2.0f * (qxz + qwy);
  (*result)[2][1] = 2.0f * (qyz - qwx);
  (*result)[2][2] = 1.0f - 2.0f * (qxx + qyy);
}

static void glm_cast_versor_to_mat4(versor q, mat4* result)
{
  mat3 m3 = GLM_MAT3_ZERO_INIT;
  glm_cast_versor_to_mat3(q, &m3);
  glm_mat4_identity(*result);
  glm_mat4_ins3(m3, *result);
}

/*
 * Get a node's local matrix from the current translation, rotation and scale
 * values.These are calculated from the current animation an need to be
 * calculated dynamically.
 */
static void gltf_node_get_local_matrix(gltf_node_t* node, mat4* dest)
{
  mat4 mat_translate = GLM_MAT4_IDENTITY_INIT;
  glm_translate(mat_translate, node->translation);
  mat4 mat_scale = GLM_MAT4_IDENTITY_INIT;
  glm_scale(mat_scale, node->scale);
  mat4 mat_rotation = GLM_MAT4_IDENTITY_INIT;
  glm_cast_versor_to_mat4(node->rotation, &mat_rotation);
  glm_mat4_mulN(
    (mat4*[]){&mat_translate, &mat_rotation, &mat_scale, &node->matrix}, 4,
    *dest);
}

/*
 * Traverse the node hierarchy to the top-most parent to get the local matrix of
 * the given node
 */
static void gltf_node_get_matrix(gltf_node_t* node, mat4* dest)
{
  gltf_node_get_local_matrix(node, dest);
  gltf_node_t* p    = node->parent;
  mat4 local_matrix = GLM_MAT4_ZERO_INIT;
  while (p != NULL) {
    gltf_node_get_local_matrix(p, &local_matrix);
    glm_mat4_mul(local_matrix, *dest, *dest);
    p = p->parent;
  }
}

static void gltf_node_update(wgpu_context_t* wgpu_context, gltf_node_t* node)
{
  if (node->mesh != NULL) {
    mat4 m = GLM_MAT4_ZERO_INIT;
    gltf_node_get_local_matrix(node, &m);
    if (node->skin != NULL) {
      gltf_skin_t* skin = node->skin;
      glm_mat4_copy(m, node->mesh->uniform_block.matrix);
      // Update the joint matrices
      mat4 inverse_transform = GLM_MAT4_ZERO_INIT;
      glm_mat4_inv(m, inverse_transform);
      size_t num_joints = MIN(skin->joint_count, WGPU_GLTF_MAX_NUM_JOINTS);
      for (size_t i = 0; i < num_joints; ++i) {
        gltf_node_t* joint_node = skin->joints[i];
        mat4 joint_mat          = GLM_MAT4_ZERO_INIT;
        mat4 joint_node_mat     = GLM_MAT4_ZERO_INIT;
        gltf_node_get_matrix(joint_node, &joint_node_mat);
        glm_mat4_mul(joint_node_mat, skin->inverse_bind_matrices[i], joint_mat);
        glm_mat4_mul(inverse_transform, joint_mat, joint_mat);
        glm_mat4_copy(joint_mat, node->mesh->uniform_block.joint_matrix[i]);
      }
      node->mesh->uniform_block.joint_count = (float)skin->joint_count;
      wgpu_queue_write_buffer(
        wgpu_context, node->mesh->uniform_buffer.buffer.buffer, 0,
        &node->mesh->uniform_block, sizeof(node->mesh->uniform_buffer));
    }
    else {
      wgpu_queue_write_buffer(wgpu_context,
                              node->mesh->uniform_buffer.buffer.buffer, 0, &m,
                              sizeof(mat4));
    }
  }

  for (uint32_t i = 0; i < node->child_count; ++i) {
    gltf_node_update(wgpu_context, node->children[i]);
  }
}

static void gltf_node_destroy(gltf_node_t* node)
{
  if (node->children != NULL) {
    free(node->children);
  }
}

typedef enum path_type_enum {
  PathType_TRANSLATION = 0,
  PathType_ROTATION    = 1,
  PathType_SCALE       = 2,
} path_type_enum;

/*
 * glTF Animation channel
 */
typedef struct gltf_animation_channel_t {
  path_type_enum path;
  gltf_node_t* node;
  uint32_t sampler_index;
  bool is_valid;
} gltf_animation_channel_t;

static void gltf_animation_channel_init(gltf_animation_channel_t* channel)
{
  channel->node          = NULL;
  channel->sampler_index = 0;
  channel->is_valid      = false;
}

typedef enum interpolation_type_enum {
  InterpolationType_LINEAR      = 0,
  InterpolationType_STEP        = 1,
  InterpolationType_CUBICSPLINE = 2,
} interpolation_type_enum;

/*
 * glTF animation sampler
 */
typedef struct gltf_animation_sampler_t {
  interpolation_type_enum interpolation;
  float* inputs;
  uint32_t input_count;
  vec4* outputs_vec4;
  uint32_t outputs_vec4_count;
} gltf_animation_sampler_t;

static void gltf_animation_sampler_init(gltf_animation_sampler_t* sampler)
{
  sampler->inputs      = NULL;
  sampler->input_count = 0;

  sampler->outputs_vec4       = NULL;
  sampler->outputs_vec4_count = 0;
}

/* glTF animation */
typedef struct gltf_animation_t {
  char name[STRMAX];
  gltf_animation_sampler_t* samplers;
  uint32_t sampler_count;
  gltf_animation_channel_t* channels;
  uint32_t channel_count;
  float start;
  float end;
} gltf_animation_t;

static void gltf_animation_init(gltf_animation_t* animation)
{
  animation->samplers      = NULL;
  animation->sampler_count = 0;

  animation->channels      = NULL;
  animation->channel_count = 0;

  animation->start = FLT_MAX;
  animation->end   = FLT_MIN;
}

/* glTF Vertex */
typedef struct gltf_vertex_t {
  vec3 pos;
  vec3 normal;
  vec2 uv;
  vec4 color;
  vec4 joint0;
  vec4 weight0;
  vec4 tangent;
} gltf_vertex_t;

WGPUVertexAttribute wgpu_gltf_get_vertex_attribute_description(
  uint32_t shader_location, wgpu_gltf_vertex_component_enum_t component)
{
  switch (component) {
    case WGPU_GLTF_VertexComponent_Position:
      return (WGPUVertexAttribute){
        .shaderLocation = shader_location,
        .format         = WGPUVertexFormat_Float32x3,
        .offset         = offsetof(gltf_vertex_t, pos),
      };
    case WGPU_GLTF_VertexComponent_Normal:
      return (WGPUVertexAttribute){
        .shaderLocation = shader_location,
        .format         = WGPUVertexFormat_Float32x3,
        .offset         = offsetof(gltf_vertex_t, normal),
      };
    case WGPU_GLTF_VertexComponent_UV:
      return (WGPUVertexAttribute){
        .shaderLocation = shader_location,
        .format         = WGPUVertexFormat_Float32x2,
        .offset         = offsetof(gltf_vertex_t, uv),
      };
    case WGPU_GLTF_VertexComponent_Color:
      return (WGPUVertexAttribute){
        .shaderLocation = shader_location,
        .format         = WGPUVertexFormat_Float32x4,
        .offset         = offsetof(gltf_vertex_t, color),
      };
    case WGPU_GLTF_VertexComponent_Tangent:
      return (WGPUVertexAttribute){
        .shaderLocation = shader_location,
        .format         = WGPUVertexFormat_Float32x4,
        .offset         = offsetof(gltf_vertex_t, tangent),
      };
    case WGPU_GLTF_VertexComponent_Joint0:
      return (WGPUVertexAttribute){
        .shaderLocation = shader_location,
        .format         = WGPUVertexFormat_Float32x4,
        .offset         = offsetof(gltf_vertex_t, joint0),
      };
    case WGPU_GLTF_VertexComponent_Weight0:
      return (WGPUVertexAttribute){
        .shaderLocation = shader_location,
        .format         = WGPUVertexFormat_Float32x4,
        .offset         = offsetof(gltf_vertex_t, weight0),
      };
    default:
      return (WGPUVertexAttribute){0};
  }
}

uint64_t wgpu_gltf_get_vertex_size(void)
{
  return sizeof(gltf_vertex_t);
}

/*
 * glTF model loading and rendering class
 */
typedef struct gltf_model_t {
  wgpu_context_t* wgpu_context;
  char uri[STRMAX];

  wgpu_buffer_t vertices;
  wgpu_buffer_t indices;

  mat4 aabb;

  gltf_node_t* nodes;
  uint32_t node_count;

  gltf_node_t** linear_nodes;
  uint32_t linear_node_count;

  gltf_skin_t* skins;
  uint32_t skin_count;

  gltf_texture_t* textures;
  uint32_t texture_count;

  gltf_texture_t* empty_texture;

  gltf_texture_sampler_t* texture_samplers;
  uint32_t texture_sampler_count;

  gltf_material_t* materials;
  uint32_t material_count;

  gltf_mesh_t* meshes;
  uint32_t mesh_count;

  gltf_animation_t* animations;
  uint32_t animation_count;

  struct {
    vec3 min;
    vec3 max;
  } dimensions;

  bool buffers_bound;
  char path[STRMAX];
} gltf_model_t;

/*
 * In this WebGPU glTF model, each texture is represented by a single image,
 * therefore WebGPU texture = glTF image. This function maps a glTF texture to a
 * WebGPU texture (= glTF image).
 */
static gltf_texture_t* gltf_model_get_texture(gltf_model_t* model,
                                              cgltf_data* data,
                                              cgltf_texture* texture)
{
  uint32_t index = texture->image - data->images;
  if (index < model->texture_count) {
    return &model->textures[index];
  }
  return NULL;
}

static void gltf_model_create_empty_texture(gltf_model_t* model)
{
  gltf_texture_t* empty_texture = calloc(1, sizeof(gltf_texture_t));
  model->empty_texture          = empty_texture;
  empty_texture->wgpu_context   = model->wgpu_context;
  empty_texture->wgpu_texture = wgpu_create_empty_texture(model->wgpu_context);
}

/*
 * glTF model loading and rendering class
 */
static void gltf_model_init(gltf_model_t* model,
                            struct wgpu_gltf_model_load_options_t* options)
{
  model->wgpu_context = options->wgpu_context;

  snprintf(model->uri, strlen(options->filename) + 1, "%s", options->filename);

  glm_mat4_zero(model->aabb);

  model->nodes      = NULL;
  model->node_count = 0;

  model->linear_nodes      = NULL;
  model->linear_node_count = 0;

  model->skins      = NULL;
  model->skin_count = 0;

  model->textures      = NULL;
  model->texture_count = 0;

  model->empty_texture = NULL;

  model->texture_samplers      = NULL;
  model->texture_sampler_count = 0;

  model->materials      = NULL;
  model->material_count = 0;

  model->meshes     = NULL;
  model->mesh_count = 0;

  model->animations      = NULL;
  model->animation_count = 0;

  glm_vec3_copy((vec3){FLT_MAX, FLT_MAX, FLT_MAX}, model->dimensions.min);
  glm_vec3_copy((vec3){-FLT_MAX, -FLT_MAX, -FLT_MAX}, model->dimensions.max);
}

/*
 * Release all WebGPU resources acquired for the model
 */
void wgpu_gltf_model_destroy(gltf_model_t* model)
{
  if (model == NULL) {
    return;
  }

  WGPU_RELEASE_RESOURCE(Buffer, model->vertices.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, model->indices.buffer);

  if (model->skin_count > 0) {
    for (uint32_t i = 0; i < model->skin_count; ++i) {
      gltf_skin_destroy(&model->skins[i]);
    }
    free(model->skins);
  }

  for (uint32_t i = 0; i < model->texture_count; ++i) {
    gltf_texture_destroy(&model->textures[i]);
  }
  free(model->textures);

  if (model->texture_count > 0) {
    free(model->texture_samplers);
  }

  for (uint32_t i = 0; i < model->mesh_count; ++i) {
    gltf_mesh_destroy(&model->meshes[i]);
  }
  free(model->meshes);

  for (uint32_t i = 0; i < model->node_count; ++i) {
    gltf_node_destroy(&model->nodes[i]);
  }
  free(model->nodes);
  free(model->linear_nodes);

  gltf_texture_destroy(model->empty_texture);
  free(model->empty_texture);

  for (uint32_t i = 0; i < model->material_count; ++i) {
    gltf_material_destroy(&model->materials[i]);
  }
  free(model->materials);

  free(model);
}

static void gltf_model_load_node(gltf_model_t* model, cgltf_node* parent,
                                 cgltf_node* node, cgltf_data* data,
                                 gltf_vertex_t** vertices,
                                 uint32_t* vertex_count, uint32_t** indices,
                                 uint32_t* index_count, float global_scale)
{
  gltf_node_t* new_node = &model->nodes[node - data->nodes];
  gltf_node_init(new_node);
  new_node->index = (int32_t)(node - data->nodes);
  if (node->name) {
    snprintf(new_node->name, strlen(node->name) + 1, "%s", node->name);
  }
  new_node->skin_index  = (int32_t)(node->skin - data->skins);
  new_node->children    = calloc(node->children_count, sizeof(gltf_node_t*));
  new_node->child_count = node->children_count;

  // Generate local node matrix
  // It's either made up from translation, rotation, scale or a 4x4 matrix
  if (node->has_translation) {
    glm_vec3_copy(node->translation, new_node->translation);
  }
  if (node->has_rotation) {
    versor q = GLM_VEC4_ONE_INIT;
    memcpy(q, node->rotation, sizeof(node->rotation));
    glm_quat_copy(q, new_node->rotation);
  }
  if (node->has_scale) {
    glm_vec3_copy(node->scale, new_node->scale);
  }
  if (node->has_matrix) {
    memcpy(new_node->matrix, node->matrix, sizeof(node->matrix));
    if (global_scale != 1.0f) {
      // Not support yet
    }
  }

  // Node with children
  if (node->children_count > 0) {
    for (cgltf_size i = 0, len = node->children_count; i < len; ++i) {
      gltf_model_load_node(model, node, node->children[i], data, vertices,
                           vertex_count, indices, index_count, global_scale);
    }
  }

  // If the node contains mesh data, we load vertices and indices from the
  // buffers. In glTF this is done via accessors and buffer views.
  if (node->mesh != NULL) {
    cgltf_mesh* mesh      = node->mesh;
    gltf_mesh_t* new_mesh = &model->meshes[node->mesh - data->meshes];
    gltf_mesh_init(new_mesh, model->wgpu_context, new_node->matrix);
    if (mesh->name) {
      snprintf(new_mesh->name, strlen(mesh->name) + 1, "%s", mesh->name);
    }

    new_mesh->primitive_count = (uint32_t)mesh->primitives_count;
    new_mesh->primitives
      = mesh->primitives_count > 0 ?
          calloc(new_mesh->primitive_count, sizeof(*new_mesh->primitives)) :
          NULL;

    // Iterate through all primitives of this node's mesh
    for (uint32_t i = 0; i < mesh->primitives_count; ++i) {
      cgltf_primitive* primitive = &mesh->primitives[i];
      if (primitive->indices == NULL) {
        continue;
      }
      uint32_t index_start       = *index_count;
      uint32_t vertex_start      = *vertex_count;
      uint32_t prim_index_count  = 0;
      uint32_t prim_vertex_count = 0;
      vec3 pos_min               = GLM_VEC3_ZERO_INIT;
      vec3 pos_max               = GLM_VEC3_ZERO_INIT;
      bool has_skin              = false;

      // Vertices
      {
        const float* buffer_pos       = NULL;
        const float* buffer_normals   = NULL;
        const float* buffer_texcoords = NULL;
        const float* buffer_colors    = NULL;
        const float* buffer_tangents  = NULL;
        uint32_t num_color_components = 0;
        const uint16_t* buffer_joints = NULL;
        const float* buffer_weights   = NULL;

        cgltf_accessor* pos_accessor = NULL;

        for (uint32_t j = 0; j < primitive->attributes_count; ++j) {
          // Get buffer data for vertex normals
          if (primitive->attributes[j].type == cgltf_attribute_type_position) {
            pos_accessor                = primitive->attributes[j].data;
            cgltf_buffer_view* pos_view = pos_accessor->buffer_view;
            buffer_pos
              = (float*)&((unsigned char*)pos_view->buffer
                            ->data)[pos_accessor->offset + pos_view->offset];
            if (pos_accessor->has_min) {
              glm_vec3_copy(
                (vec3){
                  pos_accessor->min[0],
                  pos_accessor->min[1],
                  pos_accessor->min[2],
                },
                pos_min);
            }
            if (pos_accessor->has_max) {
              glm_vec3_copy(
                (vec3){
                  pos_accessor->max[0],
                  pos_accessor->max[1],
                  pos_accessor->max[2],
                },
                pos_max);
            }
          }
          // Get buffer data for vertex normals
          if (primitive->attributes[j].type == cgltf_attribute_type_normal) {
            cgltf_accessor* normal_accessor = primitive->attributes[j].data;
            cgltf_buffer_view* normal_view  = normal_accessor->buffer_view;
            buffer_normals                  = (float*)&(
              (unsigned char*)normal_view->buffer
                ->data)[normal_accessor->offset + normal_view->offset];
          }
          // Get buffer data for vertex texture coordinates
          if (primitive->attributes[j].type == cgltf_attribute_type_texcoord) {
            cgltf_accessor* texcoord_accessor = primitive->attributes[j].data;
            cgltf_buffer_view* texcoord_view  = texcoord_accessor->buffer_view;
            buffer_texcoords                  = (float*)&(
              ((unsigned char*)texcoord_view->buffer
                 ->data)[texcoord_accessor->offset + texcoord_view->offset]);
          }
          // Get buffer data for vertex colors
          if (primitive->attributes[j].type == cgltf_attribute_type_color) {
            cgltf_accessor* color_accessor = primitive->attributes[j].data;
            cgltf_buffer_view* color_view  = color_accessor->buffer_view;
            // Color buffer are either of type vec3 or vec4
            num_color_components
              = color_accessor->type == cgltf_type_vec3 ? 3 : 4;
            buffer_colors = (float*)&(
              ((unsigned char*)color_view->buffer
                 ->data)[color_accessor->offset + color_view->offset]);
          }
          // Get buffer data for vertex tangents
          if (primitive->attributes[j].type == cgltf_attribute_type_tangent) {
            cgltf_accessor* tangent_accessor = primitive->attributes[j].data;
            cgltf_buffer_view* tangent_view  = tangent_accessor->buffer_view;
            buffer_tangents                  = (float*)&(
              ((unsigned char*)tangent_view->buffer
                 ->data)[tangent_accessor->offset + tangent_view->offset]);
          }

          // Skinning
          // Get vertex joint indices
          if (primitive->attributes[j].type == cgltf_attribute_type_joints) {
            cgltf_accessor* joint_accessor = primitive->attributes[j].data;
            cgltf_buffer_view* joint_view  = joint_accessor->buffer_view;
            buffer_joints                  = (uint16_t*)&(
              ((unsigned char*)joint_view->buffer
                 ->data)[joint_accessor->offset + joint_view->offset]);
          }
          // Get vertex joint weights
          if (primitive->attributes[j].type == cgltf_attribute_type_weights) {
            cgltf_accessor* weight_accessor = primitive->attributes[j].data;
            cgltf_buffer_view* weight_view  = weight_accessor->buffer_view;
            buffer_weights                  = (float*)&(
              ((unsigned char*)weight_view->buffer
                 ->data)[weight_accessor->offset + weight_view->offset]);
          }
        }

        has_skin = (buffer_joints != NULL && buffer_weights != NULL);

        // Position attribute is required
        ASSERT(pos_accessor != NULL);

        prim_vertex_count = (uint32_t)pos_accessor->count;

        *vertex_count += prim_vertex_count;
        *vertices = realloc(*vertices, (*vertex_count) * sizeof(gltf_vertex_t));

        // Append data to model's vertex buffer
        for (uint32_t v = 0; v < pos_accessor->count; ++v) {
          gltf_vertex_t vert = {0};
          memcpy(&vert.pos, &buffer_pos[v * 3], sizeof(vec3));
          if (buffer_normals != NULL) {
            memcpy(&vert.normal, &buffer_normals[v * 3], sizeof(vec3));
          }
          if (buffer_texcoords != NULL) {
            memcpy(&vert.uv, &buffer_texcoords[v * 2], sizeof(vec2));
          }
          if (buffer_colors) {
            switch (num_color_components) {
              case 3: {
                glm_vec4_one(vert.color);
                vec3 tmp_vec3 = GLM_VEC3_ZERO_INIT;
                memcpy(&tmp_vec3, &buffer_colors[v * 3], sizeof(vec3));
                glm_vec3_copy(tmp_vec3, vert.color);
              } break;
              case 4:
                memcpy(&vert.color, &buffer_colors[v * 4], sizeof(vec4));
                break;
            }
          }
          else {
            glm_vec4_one(vert.color);
          }
          if (buffer_tangents) {
            memcpy(&vert.tangent, &buffer_tangents[v * 4], sizeof(vec4));
          }
          if (has_skin) {
            uint16_t tmp_joint[4] = {0};
            memcpy(&tmp_joint, &buffer_joints[v * 4], sizeof(tmp_joint));
            glm_vec4_copy(
              (vec4){tmp_joint[0], tmp_joint[1], tmp_joint[2], tmp_joint[3]},
              vert.joint0);
          }
          if (has_skin) {
            memcpy(&vert.weight0, &buffer_weights[v * 4], sizeof(vec4));
          }
          (*vertices)[(*vertex_count) - pos_accessor->count + v] = vert;
        }
      }

      // Indices
      {
        cgltf_accessor* accessor       = primitive->indices;
        cgltf_buffer_view* buffer_view = accessor->buffer_view;
        cgltf_buffer* buffer           = buffer_view->buffer;

        prim_index_count = (uint32_t)accessor->count;

        *index_count += prim_index_count;
        *indices = realloc(*indices, (*index_count) * sizeof(**indices));

        // glTF supports different component types of indices
        switch (accessor->component_type) {
          case cgltf_component_type_r_32u: {
            uint32_t* buf = calloc(accessor->count, sizeof(*buf));
            memcpy(buf,
                   &((unsigned char*)
                       buffer->data)[accessor->offset + buffer_view->offset],
                   accessor->count * sizeof(*buf));
            for (size_t index = 0; index < accessor->count; index++) {
              (*indices)[(*index_count) - prim_index_count + index]
                = buf[index] + vertex_start;
            }
            free(buf);
            break;
          }
          case cgltf_component_type_r_16u: {
            uint16_t* buf = calloc(accessor->count, sizeof(*buf));
            memcpy(buf,
                   &((unsigned char*)
                       buffer->data)[accessor->offset + buffer_view->offset],
                   accessor->count * sizeof(*buf));
            for (size_t index = 0; index < accessor->count; index++) {
              (*indices)[(*index_count) - prim_index_count + index]
                = buf[index] + vertex_start;
            }
            free(buf);
            break;
          }
          case cgltf_component_type_r_8u: {
            uint8_t* buf = calloc(accessor->count, sizeof(*buf));
            memcpy(buf,
                   &((unsigned char*)
                       buffer->data)[accessor->offset + buffer_view->offset],
                   accessor->count * sizeof(*buf));
            for (size_t index = 0; index < accessor->count; index++) {
              (*indices)[(*index_count) - prim_index_count + index]
                = buf[index] + vertex_start;
            }
            free(buf);
            break;
          }
          default: {
            assert(false);
          }
        }
      }
      gltf_primitive_t new_primitive = {0};
      gltf_primitive_init(
        &new_primitive, index_start, prim_index_count,
        &model->materials[primitive->material - data->materials]);
      new_primitive.first_vertex = vertex_start;
      new_primitive.vertex_count = prim_vertex_count;
      gltf_primitive_set_bounding_box(&new_primitive, pos_min, pos_max);
      new_mesh->primitives[i] = new_primitive;
    }
    // Mesh BB from BBs of primitives
    for (uint32_t pi = 0; pi < new_mesh->primitive_count; ++pi) {
      gltf_primitive_t* p = &new_mesh->primitives[pi];
      if (p->bb.valid && !new_mesh->bb.valid) {
        memcpy(&new_mesh->bb, &p->bb, sizeof(p->bb));
        new_mesh->bb.valid = true;
      }
      glm_vec3_copy(*vec3_min(&new_mesh->bb.min, &p->bb.min), new_mesh->bb.min);
      glm_vec3_copy(*vec3_min(&new_mesh->bb.max, &p->bb.max), new_mesh->bb.max);
    }
    if (node->mesh != NULL) {
      new_node->mesh = &model->meshes[node->mesh - data->meshes];
    }
  }
  if (parent != NULL) {
    new_node->parent = &model->nodes[parent - data->nodes];
    new_node->parent->children[new_node->parent->current_child_index++]
      = new_node;
  }
  model->linear_nodes[model->linear_node_count++] = new_node;
}

static void gltf_model_load_skins(gltf_model_t* model, cgltf_data* data)
{
  model->skin_count = (uint32_t)data->skins_count;
  model->skins      = model->skin_count > 0 ?
                        calloc(model->skin_count, sizeof(*model->skins)) :
                        NULL;
  for (uint32_t i = 0; i < model->skin_count; ++i) {
    cgltf_skin* skin      = &data->skins[i];
    gltf_skin_t* new_skin = &model->skins[i];
    snprintf(new_skin->name, strlen(skin->name) + 1, "%s", skin->name);

    // Find the root node of the skeleton
    if (skin->skeleton != NULL) {
      new_skin->skeleton_root
        = gltf_model_node_from_index(model, skin->skeleton - data->nodes);
    }

    // Find joint nodes
    new_skin->joint_count = (uint32_t)skin->joints_count;
    new_skin->joints
      = new_skin->joint_count > 0 ?
          calloc(new_skin->joint_count, sizeof(*new_skin->joints)) :
          NULL;
    for (uint32_t j = 0; j < new_skin->joint_count; ++j) {
      gltf_node_t* node
        = gltf_model_node_from_index(model, skin->joints[i] - data->nodes);
      if (node != NULL) {
        new_skin->joints[new_skin->current_joint_index++] = node;
      }
    }

    // Get the inverse bind matrices from the buffer associated to this skin
    if (skin->inverse_bind_matrices != NULL) {
      cgltf_accessor* accessor            = skin->inverse_bind_matrices;
      cgltf_buffer_view* buffer_view      = accessor->buffer_view;
      cgltf_buffer* buffer                = buffer_view->buffer;
      new_skin->inverse_bind_matrix_count = (uint32_t)accessor->count;

      if (new_skin->inverse_bind_matrix_count > 0) {
        new_skin->inverse_bind_matrices
          = calloc(new_skin->inverse_bind_matrix_count,
                   sizeof(*new_skin->inverse_bind_matrices));
        memcpy(new_skin->inverse_bind_matrices,
               (mat4*)&((unsigned char*)
                          buffer->data)[accessor->offset + buffer_view->offset],
               new_skin->inverse_bind_matrix_count
                 * sizeof(*new_skin->inverse_bind_matrices));
      }
    }
  }
}

static void gltf_model_load_images(gltf_model_t* model, cgltf_data* data)
{
  model->texture_count = (uint32_t)data->images_count;
  model->textures      = model->texture_count > 0 ?
                           calloc(model->texture_count, sizeof(*model->textures)) :
                           NULL;
  for (uint32_t i = 0; i < model->texture_count; ++i) {
    cgltf_image* image      = &data->images[i];
    gltf_texture_t* texture = &model->textures[i];
    gltf_texture_init(texture, model->wgpu_context);
    gltf_texture_from_gltf_image(model->uri, texture, image);
  }
  // Create an empty texture to be used for empty material images
  gltf_model_create_empty_texture(model);
}

static void gltf_model_load_texture_samplers(gltf_model_t* model,
                                             cgltf_data* data)
{
  model->texture_sampler_count = (uint32_t)data->samplers_count;
  model->texture_samplers
    = model->texture_sampler_count > 0 ?
        calloc(model->texture_sampler_count, sizeof(*model->texture_samplers)) :
        NULL;
  for (uint32_t i = 0; i < model->texture_sampler_count; ++i) {
    cgltf_sampler* smpl                     = &data->samplers[i];
    gltf_texture_sampler_t* texture_sampler = &model->texture_samplers[i];
    texture_sampler->min_filter     = get_wgpu_filter_mode(smpl->min_filter);
    texture_sampler->mag_filter     = get_wgpu_filter_mode(smpl->mag_filter);
    texture_sampler->address_mode_u = get_wgpu_wrap_mode(smpl->wrap_s);
    texture_sampler->address_mode_v = get_wgpu_wrap_mode(smpl->wrap_t);
    texture_sampler->address_mode_w = texture_sampler->address_mode_v;
  }
}

static void gltf_model_load_materials(gltf_model_t* model, cgltf_data* data)
{
  model->material_count = (uint32_t)data->materials_count + 1;
  model->materials
    = model->material_count > 0 ?
        calloc(model->material_count, sizeof(*model->materials)) :
        NULL;
  for (uint32_t i = 0; i < data->materials_count; ++i) {
    cgltf_material* mat       = &data->materials[i];
    gltf_material_t* material = &model->materials[i];
    gltf_material_init(material, model->wgpu_context);

    // Metallic roughness workflow
    if (mat->has_pbr_metallic_roughness) {
      cgltf_pbr_metallic_roughness* mr_config = &mat->pbr_metallic_roughness;
      if (mr_config->base_color_texture.texture != NULL) {
        material->base_color_texture = gltf_model_get_texture(
          model, data, mr_config->base_color_texture.texture);
      }
      if (mr_config->metallic_roughness_texture.texture != NULL) {
        material->metallic_roughness_texture = gltf_model_get_texture(
          model, data, mr_config->metallic_roughness_texture.texture);
      }
      material->roughness_factor = mr_config->roughness_factor;
      material->metallic_factor  = mr_config->metallic_factor;
      const float* c             = mr_config->base_color_factor;
      memcpy(material->base_color_factor, (vec4){c[0], c[1], c[2], c[3]},
             sizeof(vec4));
    }
    if (mat->normal_texture.texture != NULL) {
      material->normal_texture
        = gltf_model_get_texture(model, data, mat->normal_texture.texture);
    }
    else {
      material->normal_texture = model->empty_texture;
    }
    if (mat->emissive_texture.texture != NULL) {
      material->emissive_texture
        = gltf_model_get_texture(model, data, mat->emissive_texture.texture);
    }
    if (mat->occlusion_texture.texture != NULL) {
      material->occlusion_texture
        = gltf_model_get_texture(model, data, mat->occlusion_texture.texture);
    }
    if (mat->alpha_mode == cgltf_alpha_mode_blend) {
      material->alpha_mode = AlphaMode_BLEND;
      material->blend      = true;
    }
    if (mat->alpha_mode == cgltf_alpha_mode_mask) {
      material->alpha_mode = AlphaMode_MASK;
      material->blend      = true;
    }
    material->alpha_cutoff = mat->alpha_cutoff;
    material->double_sided = mat->double_sided;
  }
  // Push a default material at the end of the list for meshes with no material
  // assigned
  gltf_material_init(&model->materials[model->material_count - 1],
                     model->wgpu_context);
}

/*
 * Load the animations from the glTF model
 */
static void gltf_model_load_animations(gltf_model_t* model, cgltf_data* data)
{
  model->animation_count = (uint32_t)data->animations_count;
  model->animations
    = model->animation_count > 0 ?
        calloc(model->animation_count, sizeof(*model->animations)) :
        NULL;
  for (uint32_t i = 0; i < model->animation_count; ++i) {
    cgltf_animation* anim       = &data->animations[i];
    gltf_animation_t* animation = &model->animations[i];
    gltf_animation_init(animation);
    if (anim->name) {
      snprintf(animation->name, strlen(anim->name) + 1, "%s", anim->name);
    }

    // Samplers
    animation->sampler_count = (uint32_t)anim->samplers_count;
    animation->samplers
      = animation->sampler_count > 0 ?
          calloc(animation->sampler_count, sizeof(*animation->samplers)) :
          NULL;
    for (uint32_t j = 0; j < animation->sampler_count; ++j) {
      cgltf_animation_sampler* samp     = &anim->samplers[j];
      gltf_animation_sampler_t* sampler = &animation->samplers[j];
      gltf_animation_sampler_init(sampler);

      if (samp->interpolation == cgltf_interpolation_type_linear) {
        sampler->interpolation = InterpolationType_LINEAR;
      }
      if (samp->interpolation == cgltf_interpolation_type_step) {
        sampler->interpolation = InterpolationType_STEP;
      }
      if (samp->interpolation == cgltf_interpolation_type_cubic_spline) {
        sampler->interpolation = InterpolationType_CUBICSPLINE;
      }

      // Read sampler input time values
      {
        cgltf_accessor* accessor       = samp->input;
        cgltf_buffer_view* buffer_view = accessor->buffer_view;
        cgltf_buffer* buffer           = buffer_view->buffer;

        ASSERT(accessor->component_type == cgltf_component_type_r_32f);

        sampler->input_count = (uint32_t)accessor->count;
        sampler->inputs
          = sampler->input_count > 0 ?
              calloc(sampler->input_count, sizeof(*sampler->inputs)) :
              NULL;

        float* buf = calloc(accessor->count, sizeof(float));
        memcpy(buf,
               (float*)&(((unsigned char*)buffer
                            ->data)[accessor->offset + buffer_view->offset]),
               accessor->count * sizeof(*buf));
        for (size_t index = 0; index < accessor->count; index++) {
          sampler->inputs[index] = buf[index];
        }
        free(buf);

        // Adjust animation's start and end times
        for (uint32_t k = 0; k < sampler->input_count; ++k) {
          float input = sampler->inputs[k];
          if (input < animation->start) {
            animation->start = input;
          };
          if (input > animation->end) {
            animation->end = input;
          }
        }
      }

      // Read sampler keyframe output translate/rotate/scale values
      {
        cgltf_accessor* accessor       = samp->output;
        cgltf_buffer_view* buffer_view = accessor->buffer_view;
        cgltf_buffer* buffer           = buffer_view->buffer;

        ASSERT(accessor->component_type == cgltf_component_type_r_32f);

        switch (accessor->type) {
          case cgltf_type_vec3: {
            sampler->outputs_vec4_count = (uint32_t)accessor->count;
            sampler->outputs_vec4       = calloc(sampler->outputs_vec4_count,
                                                 sizeof(*sampler->outputs_vec4));

            vec3* buf = calloc(accessor->count, sizeof(vec3));
            memcpy(buf,
                   (vec3*)&(((unsigned char*)buffer
                               ->data)[accessor->offset + buffer_view->offset]),
                   accessor->count * sizeof(*buf));

            for (size_t index = 0; index < accessor->count; ++index) {
              glm_vec4_zero(sampler->outputs_vec4[i]);
              glm_vec4_copy3(buf[index], sampler->outputs_vec4[i]);
            }

            free(buf);
          } break;
          case cgltf_type_vec4: {
            sampler->outputs_vec4_count = (uint32_t)accessor->count;
            sampler->outputs_vec4       = calloc(sampler->outputs_vec4_count,
                                                 sizeof(*sampler->outputs_vec4));

            vec4* buf = calloc(accessor->count, sizeof(vec4));
            memcpy(buf,
                   (vec4*)&(((unsigned char*)buffer
                               ->data)[accessor->offset + buffer_view->offset]),
                   accessor->count * sizeof(*buf));

            for (size_t index = 0; index < accessor->count; ++index) {
              glm_vec4_copy(buf[index], sampler->outputs_vec4[i]);
            }

            free(buf);
          } break;
          default: {
            log_warn("unknown type");
            break;
          }
        }
      }
    }

    // Channels
    animation->channel_count = (uint32_t)anim->channels_count;
    animation->channels
      = calloc(animation->channel_count, sizeof(*animation->channels));
    for (uint32_t j = 0; j < animation->channel_count; ++j) {
      cgltf_animation_channel* chan     = &anim->channels[j];
      gltf_animation_channel_t* channel = &animation->channels[j];
      gltf_animation_channel_init(channel);

      if (chan->target_path == cgltf_animation_path_type_rotation) {
        channel->path = PathType_ROTATION;
      }
      if (chan->target_path == cgltf_animation_path_type_translation) {
        channel->path = PathType_TRANSLATION;
      }
      if (chan->target_path == cgltf_animation_path_type_scale) {
        channel->path = PathType_SCALE;
      }
      if (chan->target_path == cgltf_animation_path_type_weights) {
        log_warn("Weights not yet supported, skipping channel!");
        continue;
      }
      channel->sampler_index = chan->sampler - anim->samplers;
      channel->node
        = gltf_model_node_from_index(model, chan->target_node - data->nodes);
      if (!channel->node) {
        continue;
      }

      channel->is_valid = true;
    }
  }
}

gltf_model_t* wgpu_gltf_model_load_from_file(
  struct wgpu_gltf_model_load_options_t* load_options)
{
  uint32_t file_loading_flags = load_options->file_loading_flags;

  gltf_model_t* gltf_model = NULL;

  cgltf_options options = {0};
  cgltf_data* gltf_data = NULL;
  cgltf_result result
    = cgltf_parse_file(&options, load_options->filename, &gltf_data);

  // Vertex buffer & Index buffer
  gltf_vertex_t* vertices = NULL;
  uint32_t* indices       = NULL;

  if (result == cgltf_result_success) {
    cgltf_result buffers_result
      = cgltf_load_buffers(&options, gltf_data, load_options->filename);

    if (buffers_result == cgltf_result_success) {
      gltf_model = calloc(1, sizeof(gltf_model_t));
      gltf_model_init(gltf_model, load_options);

      // Load samplers and images
      if (!(file_loading_flags & WGPU_GLTF_FileLoadingFlags_DontLoadImages)) {
        gltf_model_load_texture_samplers(gltf_model, gltf_data);
        gltf_model_load_images(gltf_model, gltf_data);
      }

      // Load materials
      gltf_model_load_materials(gltf_model, gltf_data);

      // If there is no default scene specified, then the default is the first
      // one. It is not an error for a glTF file to have zero scenes.
      const cgltf_scene* scene
        = gltf_data->scene ? gltf_data->scene : gltf_data->scenes;
      if (!scene) {
        return NULL;
      }

      // Vertex buffer & Index buffer
      gltf_model->vertices.count = 0;
      gltf_model->indices.count  = 0;

      // Nodes and meshes
      gltf_model->node_count = (uint32_t)gltf_data->nodes_count;
      gltf_model->nodes = calloc(gltf_model->node_count, sizeof(gltf_node_t));

      gltf_model->linear_nodes
        = calloc(gltf_model->node_count, sizeof(gltf_node_t*));

      gltf_model->mesh_count = (uint32_t)gltf_data->meshes_count;
      gltf_model->meshes = calloc(gltf_model->mesh_count, sizeof(gltf_mesh_t));

      // Recursively create all nodes.
      for (cgltf_size i = 0, len = scene->nodes_count; i < len; ++i) {
        gltf_model_load_node(gltf_model, NULL, scene->nodes[i], gltf_data,
                             &vertices, &gltf_model->vertices.count, &indices,
                             &gltf_model->indices.count, load_options->scale);
      }

      // Load animations
      if (gltf_data->animations_count > 0) {
        gltf_model_load_animations(gltf_model, gltf_data);
      }

      // Load skins
      gltf_model_load_skins(gltf_model, gltf_data);

      // Assign skins and initial pose
      for (uint32_t i = 0; i < gltf_model->linear_node_count; ++i) {
        gltf_node_t* node = gltf_model->linear_nodes[i];
        // Assign skins
        if (node->skin_index > -1) {
          node->skin = &gltf_model->skins[(uint32_t)node->skin_index];
        }
        // Initial pose
        if (node->mesh != NULL) {
          gltf_node_update(gltf_model->wgpu_context, node);
        }
      }
    }
  }
  else {
    log_error("Could not load gltf file: %s, error: %d\n",
              load_options->filename, result);
    return NULL;
  }

  ASSERT(gltf_model != NULL);

  // Pre-Calculations for requested features
  if ((file_loading_flags & WGPU_GLTF_FileLoadingFlags_PreTransformVertices)
      || (file_loading_flags
          & WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors)
      || (file_loading_flags & WGPU_GLTF_FileLoadingFlags_FlipY)) {
    const bool preTransform
      = file_loading_flags & WGPU_GLTF_FileLoadingFlags_PreTransformVertices;
    const bool preMultiplyColor
      = file_loading_flags & WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors;
    const bool flipY = file_loading_flags & WGPU_GLTF_FileLoadingFlags_FlipY;
    for (uint32_t n = 0; n < gltf_model->linear_node_count; ++n) {
      gltf_node_t* node = gltf_model->linear_nodes[n];
      if (node->mesh != NULL) {
        mat4 local_matrix = GLM_MAT4_ZERO_INIT;
        gltf_node_get_matrix(node, &local_matrix);
        for (uint32_t p = 0; p < node->mesh->primitive_count; ++p) {
          gltf_primitive_t* primitive = &node->mesh->primitives[p];
          for (uint32_t i = 0; i < primitive->vertex_count; ++i) {
            gltf_vertex_t* vertex = &vertices[primitive->first_vertex + i];
            // Pre-transform vertex positions by node-hierarchy
            if (preTransform) {
              // Vertex position
              vec4 vertex_pos_tmp = GLM_VEC4_ZERO_INIT;
              glm_vec4(vertex->pos, 1.0f, vertex_pos_tmp);
              glm_mat4_mulv(local_matrix, vertex_pos_tmp, vertex_pos_tmp);
              glm_vec3(vertex_pos_tmp, vertex->pos);
              // Vertex normal
              mat3 local_matrix_tmp = GLM_MAT3_ZERO_INIT;
              glm_mat4_pick3(local_matrix, local_matrix_tmp);
              vec3 mulv_result;
              glm_mat3_mulv(local_matrix_tmp, vertex->normal, mulv_result);
              glm_normalize_to(mulv_result, vertex->normal);
            }
            // Flip Y-Axis of vertex positions
            if (flipY) {
              vertex->pos[1] *= -1.0f;
              vertex->normal[1] *= -1.0f;
            }
            // Pre-Multiply vertex colors with material base color
            if (preMultiplyColor) {
              glm_vec4_mul(primitive->material->base_color_factor,
                           vertex->color, vertex->color);
            }
          }
        }
      }
    }
  }

  // Vertex and index buffers
  size_t vertex_buffer_size
    = gltf_model->vertices.count * sizeof(gltf_vertex_t);
  size_t index_buffer_size = gltf_model->indices.count * sizeof(uint32_t);

  assert((vertex_buffer_size > 0) && (index_buffer_size > 0));

  // Create vertex buffer
  gltf_model->vertices.buffer
    = wgpu_create_buffer_from_data(load_options->wgpu_context, vertices,
                                   vertex_buffer_size, WGPUBufferUsage_Vertex);

  // Create index buffer
  gltf_model->indices.buffer
    = wgpu_create_buffer_from_data(load_options->wgpu_context, indices,
                                   index_buffer_size, WGPUBufferUsage_Index);

  if (vertices != NULL) {
    free(vertices);
  }
  if (indices != NULL) {
    free(indices);
  }

  // Get scene dimensions
  gltf_model_get_scene_dimensions(gltf_model);

  // Cleanup
  cgltf_free(gltf_data);

  return gltf_model;
}

static void gltf_model_bind_buffers(gltf_model_t* model)
{
  wgpu_context_t* wgpu_context = model->wgpu_context;
  wgpuRenderPassEncoderSetVertexBuffer(
    wgpu_context->rpass_enc, 0, model->vertices.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, model->indices.buffer, WGPUIndexFormat_Uint32, 0,
    WGPU_WHOLE_SIZE);
  // model->buffers_bound = true;
}

static void
gltf_model_draw_node(gltf_model_t* model, gltf_node_t* node,
                     wgpu_gltf_model_render_options_t render_options)
{
  uint32_t render_flags = render_options.render_flags;

  if (node->mesh && node->mesh->primitive_count > 0) {
    if (node->mesh->uniform_buffer.bind_group) {
      wgpuRenderPassEncoderSetBindGroup(
        model->wgpu_context->rpass_enc, render_options.bind_mesh_model_set,
        node->mesh->uniform_buffer.bind_group, 0, 0);
    }
    for (uint32_t i = 0; i < node->mesh->primitive_count; ++i) {
      gltf_primitive_t* primitive = &node->mesh->primitives[i];
      bool skip                   = false;
      gltf_material_t* material   = primitive->material;
      if (render_flags & WGPU_GLTF_RenderFlags_RenderOpaqueNodes) {
        skip = (material->alpha_mode != AlphaMode_OPAQUE);
      }
      if (render_flags & WGPU_GLTF_RenderFlags_RenderAlphaMaskedNodes) {
        skip = (material->alpha_mode != AlphaMode_MASK);
      }
      if (render_flags & WGPU_GLTF_RenderFlags_RenderAlphaBlendedNodes) {
        skip = (material->alpha_mode != AlphaMode_BLEND);
      }
      if (!skip) {
        // Bind the pipeline for the node's material if present
        if (material->pipeline) {
          wgpuRenderPassEncoderSetPipeline(model->wgpu_context->rpass_enc,
                                           material->pipeline);
        }
        if ((render_flags & WGPU_GLTF_RenderFlags_BindImages)
            && material->bind_group) {
          wgpuRenderPassEncoderSetBindGroup(model->wgpu_context->rpass_enc,
                                            render_options.bind_image_set,
                                            material->bind_group, 0, 0);
        }
        wgpuRenderPassEncoderDrawIndexed(model->wgpu_context->rpass_enc,
                                         primitive->index_count, 1,
                                         primitive->first_index, 0, 0);
      }
    }
  }
  for (uint32_t i = 0; i < node->child_count; ++i) {
    gltf_model_draw_node(model, node->children[i], render_options);
  }
}

// Draw the glTF scene starting at the top-level-nodes
void wgpu_gltf_model_draw(gltf_model_t* model,
                          wgpu_gltf_model_render_options_t render_options)
{
  if (!model->buffers_bound) {
    // All vertices and indices are stored in single buffers, so we only need to
    // bind once
    gltf_model_bind_buffers(model);
  }
  // Render all nodes at top-level
  for (uint32_t i = 0; i < model->node_count; ++i) {
    gltf_model_draw_node(model, &model->nodes[i], render_options);
  }
}

wgpu_gltf_materials_t wgpu_gltf_model_get_materials(void* model)
{
  return (wgpu_gltf_materials_t){
    .materials      = ((gltf_model_t*)model)->materials,
    .material_count = ((gltf_model_t*)model)->material_count,
  };
}

static void
gltf_model_prepare_node_bind_group(gltf_model_t* model, gltf_node_t* node,
                                   WGPUBindGroupLayout bind_group_layout)
{
  if (node->mesh != NULL) {
    WGPUBindGroupDescriptor bg_desc = {
      .layout     = bind_group_layout,
      .entryCount = 1,
      .entries    = &(WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = node->mesh->uniform_buffer.buffer.buffer,
        .offset  = 0,
        .size    =  node->mesh->uniform_buffer.buffer.size,
      },
    };
    node->mesh->uniform_buffer.bind_group
      = wgpuDeviceCreateBindGroup(model->wgpu_context->device, &bg_desc);
    ASSERT(node->mesh->uniform_buffer.bind_group != NULL)
  }
  for (uint32_t i = 0; i < node->child_count; ++i) {
    gltf_model_prepare_node_bind_group(model, node->children[i],
                                       bind_group_layout);
  }
}

void wgpu_gltf_model_prepare_nodes_bind_group(
  gltf_model_t* model, WGPUBindGroupLayout bind_group_layout)
{
  for (uint32_t i = 0; i < model->node_count; ++i) {
    gltf_model_prepare_node_bind_group(model, &model->nodes[i],
                                       bind_group_layout);
  }
}

static void
gltf_model_prepare_skin_bind_group(gltf_model_t* model, gltf_skin_t* skin,
                                   WGPUBindGroupLayout bind_group_layout)
{
  WGPUBindGroupDescriptor bg_desc = {
      .layout     = bind_group_layout,
      .entryCount = 1,
      .entries    = &(WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = skin->ssbo.buffer,
        .offset  = 0,
        .size    =  skin->ssbo.size,
      },
    };
  skin->ssbo.bind_group
    = wgpuDeviceCreateBindGroup(model->wgpu_context->device, &bg_desc);
  ASSERT(skin->ssbo.bind_group != NULL)
}

void wgpu_gltf_model_prepare_skins_bind_group(
  gltf_model_t* model, WGPUBindGroupLayout bind_group_layout)
{
  for (uint32_t i = 0; i < model->skin_count; ++i) {
    gltf_model_prepare_skin_bind_group(model, &model->skins[i],
                                       bind_group_layout);
  }
}

static void gltf_model_node_calculate_bounding_box(gltf_node_t* node,
                                                   gltf_node_t* parent)
{
  if (node->mesh) {
    if (node->mesh->bb.valid) {
      mat4 node_matrix;
      gltf_node_get_matrix(node, &node_matrix);
      bounding_get_aabb(&node->mesh->bb, node_matrix, &node->aabb);
      if (node->child_count == 0) {
        glm_vec3_copy(node->aabb.min, node->bvh.min);
        glm_vec3_copy(node->aabb.max, node->bvh.max);
        node->bvh.valid = true;
      }
    }
  }

  if (parent) {
    bounding_box_t* parent_bvh = &parent->bvh;
    glm_vec3_copy(*vec3_min(&parent_bvh->min, &node->bvh.min), node->bvh.min);
    glm_vec3_copy(*vec3_min(&parent_bvh->max, &node->bvh.max), node->bvh.max);
  }

  for (uint32_t i = 0; i < node->child_count; ++i) {
    gltf_model_node_calculate_bounding_box(node->children[i], node);
  }
}

static void gltf_model_get_scene_dimensions(gltf_model_t* model)
{
  // Calculate binary volume hierarchy for all nodes in the scene
  for (uint32_t i = 0; i < model->linear_node_count; ++i) {
    gltf_model_node_calculate_bounding_box(model->linear_nodes[i], NULL);
  }

  glm_vec3_copy((vec3){FLT_MAX, FLT_MAX, FLT_MAX}, model->dimensions.min);
  glm_vec3_copy((vec3){-FLT_MAX, -FLT_MAX, -FLT_MAX}, model->dimensions.max);

  for (uint32_t i = 0; i < model->linear_node_count; ++i) {
    gltf_node_t* node = model->linear_nodes[i];
    if (node->bvh.valid) {
      glm_vec3_copy(*vec3_min(&model->dimensions.min, &node->bvh.min),
                    model->dimensions.min);
      glm_vec3_copy(*vec3_max(&model->dimensions.max, &node->bvh.max),
                    model->dimensions.max);
    }
  }

  // Calculate scene aabb
  glm_mat4_identity(model->aabb);
  vec3 scale = {
    model->dimensions.max[0] - model->dimensions.min[0], // x
    model->dimensions.max[1] - model->dimensions.min[1], // y
    model->dimensions.max[2] - model->dimensions.min[2]  // z
  };
  glm_scale(model->aabb, scale);
  model->aabb[3][0] = model->dimensions.min[0];
  model->aabb[3][1] = model->dimensions.min[1];
  model->aabb[3][2] = model->dimensions.min[2];
}

void gltf_model_update_animation(gltf_model_t* model, uint32_t index,
                                 float time)
{
  if (model->animation_count == 0) {
    log_warn(".glTF does not contain animations.");
    return;
  }
  if ((int32_t)index > ((int32_t)model->animation_count) - 1) {
    log_warn("No animation with index %u", index);
    return;
  }
  gltf_animation_t* animation = &model->animations[index];

  bool updated = false;
  for (uint32_t c = 0; c < animation->channel_count; ++c) {
    gltf_animation_channel_t* channel = &animation->channels[c];
    gltf_animation_sampler_t* sampler
      = &animation->samplers[channel->sampler_index];
    if (sampler->input_count > sampler->outputs_vec4_count) {
      continue;
    }

    for (uint32_t i = 0; i < sampler->input_count - 1; ++i) {
      if ((time >= sampler->inputs[i]) && (time <= sampler->inputs[i + 1])) {
        float u = MAX(0.0f, time - sampler->inputs[i])
                  / (sampler->inputs[i + 1] - sampler->inputs[i]);
        if (u <= 1.0f) {
          switch (channel->path) {
            case PathType_TRANSLATION: {
              vec4 trans = GLM_VEC4_ZERO_INIT;
              glm_vec4_mix(sampler->outputs_vec4[i],
                           sampler->outputs_vec4[i + 1], u, trans);
              glm_vec3_copy((vec3){trans[0], trans[1], trans[2]},
                            channel->node->translation);
              break;
            }
            case PathType_SCALE: {
              vec4 trans = GLM_VEC4_ZERO_INIT;
              glm_vec4_mix(sampler->outputs_vec4[i],
                           sampler->outputs_vec4[i + 1], u, trans);
              glm_vec3_copy((vec3){trans[0], trans[1], trans[2]},
                            channel->node->scale);
              break;
            }
            case PathType_ROTATION: {
              versor q1  = GLM_VEC4_ZERO_INIT;
              q1[0]      = sampler->outputs_vec4[i][0];
              q1[1]      = sampler->outputs_vec4[i][1];
              q1[2]      = sampler->outputs_vec4[i][2];
              q1[3]      = sampler->outputs_vec4[i][3];
              versor q2  = GLM_VEC4_ZERO_INIT;
              q2[0]      = sampler->outputs_vec4[i + 1][0];
              q2[1]      = sampler->outputs_vec4[i + 1][1];
              q2[2]      = sampler->outputs_vec4[i + 1][2];
              q2[3]      = sampler->outputs_vec4[i + 1][3];
              versor tmp = GLM_VEC4_ZERO_INIT;
              glm_quat_slerp(q1, q2, u, tmp);
              glm_quat_normalize(tmp);
              glm_vec4_copy(tmp, channel->node->rotation);
              break;
            }
          }
          updated = true;
        }
      }
    }
  }
  if (updated) {
    for (uint32_t i = 0; i < model->node_count; ++i) {
      gltf_node_update(model->wgpu_context, &model->nodes[i]);
    }
  }
}

/*
 * Helper functions for locating glTF nodes
 */
static gltf_node_t* gltf_model_find_node(gltf_model_t* model,
                                         gltf_node_t* parent, uint32_t index)
{
  gltf_node_t* node_found = NULL;
  if (parent->index == index) {
    return parent;
  }
  for (uint32_t i = 0; i < parent->child_count; ++i) {
    node_found = gltf_model_find_node(model, parent->children[i], index);
    if (node_found) {
      break;
    }
  }
  return node_found;
}

static gltf_node_t* gltf_model_node_from_index(gltf_model_t* model,
                                               uint32_t index)
{
  gltf_node_t* node_found = NULL;
  for (uint32_t i = 0; i < model->node_count; ++i) {
    node_found = gltf_model_find_node(model, &model->nodes[i], index);
    if (node_found) {
      break;
    }
  }
  return node_found;
}
