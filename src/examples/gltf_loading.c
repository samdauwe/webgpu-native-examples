#include "example_base.h"

#include <string.h>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <stb_image.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - glTF Model Rendering
 *
 * Shows how to load and display a simple scene from a glTF file
 * Note that this isn't a complete glTF loader and only basic functions are
 * shown here This means no complex materials, no animations, no skins, etc. For
 * details on how glTF 2.0 works, see the official spec at
 * https://github.com/KhronosGroup/glTF/tree/master/specification/2.0
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/gltfloading
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * glTF Model
 * -------------------------------------------------------------------------- */

// Loader options
typedef struct gltf_model_load_options_t {
  struct wgpu_context_t* wgpu_context;
  const char* filename;
  int flip_uvs;
} gltf_model_load_options_t;

// Bounding box
typedef struct gltf_dimensions_t {
  vec3 min;
  vec3 max;
  bool valid;
} gltf_dimensions_t;

static void gltf_dimensions_init(gltf_dimensions_t* dimensions)
{
  glm_vec3_copy((vec3){FLT_MAX, FLT_MAX, FLT_MAX}, dimensions->min);
  glm_vec3_copy((vec3){-FLT_MAX, -FLT_MAX, -FLT_MAX}, dimensions->min);
  dimensions->valid = false;
}

// The vertex layout for the samples' model
typedef struct gltf_vertex_t {
  vec3 pos;
  vec3 normal;
  vec2 uv;
  vec3 color;
} gltf_vertex_t;

// Single vertex buffer for all primitives
typedef struct gltf_vertices_t {
  WGPUBuffer buffer;
  uint32_t count;
} gltf_vertices_t;

// Single index buffer for all primitives
typedef struct gltf_indices_t {
  WGPUBuffer buffer;
  uint32_t count;
} gltf_indices_t;

// The following structures roughly represent the glTF scene structure
// To keep things simple, they only contain those properties that are required
// for this sample
struct gltf_material_t;
struct gltf_node_t;

// A primitive contains the data for a single draw call
typedef struct gltf_primitive_t {
  uint32_t first_index;
  uint32_t index_count;
  int32_t material_index;
  bool has_indices;
  gltf_dimensions_t dimensions;
} gltf_primitive_t;

static void gltf_primitive_init(gltf_primitive_t* primitive,
                                uint32_t first_index, uint32_t index_count,
                                int32_t material_index)
{
  primitive->first_index    = first_index;
  primitive->index_count    = index_count;
  primitive->material_index = material_index;
  primitive->has_indices    = index_count > 0;
  gltf_dimensions_init(&primitive->dimensions);
}

static void gltf_primitive_set_dimensions(gltf_primitive_t* primitive, vec3 min,
                                          vec3 max)
{
  gltf_dimensions_t* dim = &primitive->dimensions;
  glm_vec3_copy(min, dim->min);
  glm_vec3_copy(max, dim->max);
  dim->valid = true;
}

// Contains the node's (optional) geometry and can be made up of an arbitrary
// number of primitives
typedef struct gltf_mesh_t {
  gltf_primitive_t* primitives;
  uint32_t primitive_count;
  mat4 matrix;
  WGPUBuffer uniform_buffer;
  WGPUBindGroup bind_group;
} gltf_mesh_t;

static void gltf_mesh_init(gltf_mesh_t* mesh, wgpu_context_t* wgpu_context,
                           mat4 matrix)
{
  mesh->primitives      = NULL;
  mesh->primitive_count = 0;
  glm_mat4_copy(matrix, mesh->matrix);

  mesh->uniform_buffer = wgpu_create_buffer_from_data(
    wgpu_context, &mesh->matrix, sizeof(mat4), WGPUBufferUsage_Uniform);
}

static void gltf_mesh_destroy(gltf_mesh_t* mesh)
{
  WGPU_RELEASE_RESOURCE(Buffer, mesh->uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, mesh->bind_group)

  if (mesh->primitives != NULL) {
    free(mesh->primitives);
  }
}

// A node represents an object in the glTF scene graph
typedef struct gltf_node_t {
  struct gltf_node_t* parent;
  struct gltf_node_t** children;
  uint32_t current_child_index;
  uint32_t child_count;
  gltf_mesh_t* mesh;
  mat4 matrix;
} gltf_node_t;

static void gltf_node_init(gltf_node_t* node)
{
  node->parent              = NULL;
  node->children            = NULL;
  node->current_child_index = 0;
  node->child_count         = 0;
  node->mesh                = NULL;
  glm_mat4_identity(node->matrix);
}

static void gltf_node_destroy(gltf_node_t* node)
{
  if (node->children != NULL) {
    free(node->children);
  }
}

// A glTF material stores information in e.g. the texture that is attached to it
// and colors
typedef struct gltf_material_t {
  vec4 base_color_factor;
  uint32_t base_color_image_index;
} gltf_material_t;

// Contains the texture for a single glTF image
// Images may be reused by texture objects and are as such separated
typedef struct gltf_image_t {
  texture_t texture;
  // We also store (and create) a bind group that's used to access this texture
  // from the fragment shader
  WGPUBindGroup bind_group;
} gltf_image_t;

static void gltf_image_destroy(gltf_image_t* image)
{
  wgpu_destroy_texture(&image->texture);
  WGPU_RELEASE_RESOURCE(BindGroup, image->bind_group)
}

// A glTF texture stores a reference to the image and a sampler
// In this sample, we are only interested in the image
typedef struct gltf_texture_t {
  int32_t image_index;
} gltf_texture_t;

// Contains everything required to render a glTF model
// This class is heavily simplified (compared to glTF's feature set) but retains
// the basic glTF structure
typedef struct gltf_model_t {
  wgpu_context_t* wgpu_context;

  char uri[STRMAX];
  bool flip_uvs;

  gltf_vertices_t vertices;
  gltf_indices_t indices;

  /* Model data */
  gltf_image_t* images;
  uint32_t image_count;

  gltf_texture_t* textures;
  uint32_t texture_count;

  gltf_material_t* materials;
  uint32_t material_count;

  gltf_node_t* nodes;
  uint32_t node_count;

  gltf_mesh_t* meshes;
  uint32_t mesh_count;
} gltf_model_t;

static void gltf_model_init(gltf_model_t* model,
                            gltf_model_load_options_t* options)
{
  model->wgpu_context = options->wgpu_context;

  snprintf(model->uri, strlen(options->filename) + 1, "%s", options->filename);
  model->flip_uvs = options->flip_uvs;

  model->vertices = (gltf_vertices_t){0};
  model->indices  = (gltf_indices_t){0};

  model->images      = NULL;
  model->image_count = 0;

  model->textures      = NULL;
  model->texture_count = 0;

  model->materials      = NULL;
  model->material_count = 0;

  model->nodes      = NULL;
  model->node_count = 0;

  model->meshes     = NULL;
  model->mesh_count = 0;
}

static void gltf_model_destroy(gltf_model_t* model)
{
  if (model == NULL) {
    return;
  }

  WGPU_RELEASE_RESOURCE(Buffer, model->vertices.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, model->indices.buffer);

  for (uint32_t i = 0; i < model->mesh_count; i++) {
    gltf_mesh_destroy(&model->meshes[i]);
  }
  free(model->meshes);

  for (uint32_t i = 0; i < model->node_count; i++) {
    gltf_node_destroy(&model->nodes[i]);
  }
  free(model->nodes);

  for (uint32_t i = 0; i < model->image_count; i++) {
    gltf_image_destroy(&model->images[i]);
  }
  free(model->images);
  free(model->textures);
  free(model->materials);

  free(model);
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

/*
 * glTF loading functions
 *
 * The following functions take a glTF input model loaded via cglTF and convert
 * all required data into our own structure
 */

void gltf_model_load_images(gltf_model_t* model, cgltf_data* data)
{
  wgpu_context_t* wgpu_context = model->wgpu_context;

  // Images can be stored inside the glTF (which is the case for the sample
  // model), so instead of directly loading them from disk, we fetch them from
  // the glTF loader and upload the buffers
  model->image_count = (uint32_t)data->images_count;
  model->images      = calloc(model->image_count, sizeof(*model->images));
  for (uint32_t i = 0; i < model->image_count; ++i) {
    cgltf_image* image = &data->images[i];
    gltf_image_t* img  = &model->images[i];
    // Load texture from image file
    char image_uri[STRMAX] = {0};
    get_relative_file_path(model->uri, image->uri, image_uri);
    img->texture = wgpu_create_texture_from_file(wgpu_context, image_uri, NULL);
  }
}

void gltf_model_load_textures(gltf_model_t* model, cgltf_data* data)
{
  model->texture_count = (uint32_t)data->textures_count;
  model->textures      = calloc(model->texture_count, sizeof(*model->textures));
  for (uint32_t i = 0; i < model->texture_count; ++i) {
    cgltf_texture* texture = &data->textures[i];
    gltf_texture_t* tex    = &model->textures[i];
    tex->image_index       = texture->image - data->images;
  }
}

void gltf_model_load_materials(gltf_model_t* model, cgltf_data* data)
{
  model->material_count = (uint32_t)data->materials_count;
  model->materials = calloc(model->material_count, sizeof(*model->materials));
  for (uint32_t i = 0; i < data->materials_count; ++i) {
    // We only read the most basic properties required for our sample
    cgltf_material* material = &data->materials[i];
    gltf_material_t* mat     = &model->materials[i];
    ASSERT(material->has_pbr_metallic_roughness);
    // Get the base color factor
    cgltf_pbr_metallic_roughness mr_config = material->pbr_metallic_roughness;
    const float* c                         = mr_config.base_color_factor;
    memcpy(mat->base_color_factor, (vec4){c[0], c[1], c[2], c[3]},
           sizeof(vec4));
    // Get base color image index
    if (mr_config.base_color_texture.texture != NULL) {
      mat->base_color_image_index
        = mr_config.base_color_texture.texture->image - data->images;
    }
  }
}

static void gltf_model_load_node(gltf_model_t* model, cgltf_node* parent,
                                 cgltf_node* node, cgltf_data* data,
                                 gltf_vertex_t** vertices,
                                 uint32_t* vertex_count, uint32_t** indices,
                                 uint32_t* index_count, bool flip_uvs)
{
  gltf_node_t* new_node = &model->nodes[node - data->nodes];
  gltf_node_init(new_node);
  new_node->children    = malloc(node->children_count * sizeof(gltf_node_t*));
  new_node->child_count = node->children_count;

  if (parent != NULL) {
    new_node->parent = &model->nodes[parent - data->nodes];
    new_node->parent->children[new_node->parent->current_child_index++]
      = new_node;
  }

  // Get the local node matrix
  // It's either made up from translation, rotation, scale or a 4x4 matrix
  if (node->has_translation) {
    vec3 translation = GLM_VEC3_ZERO_INIT;
    glm_vec3_copy(node->translation, translation);
    glm_translate(new_node->matrix, translation);
  }
  if (node->has_rotation) {
    versor q      = GLM_VEC4_ZERO_INIT;
    mat4 rotation = GLM_MAT4_IDENTITY_INIT;
    memcpy(q, node->rotation, sizeof(node->rotation));
    q[1] += -1.0f; // flip Y to match Vulkan Y-axis to keep other code the same
    glm_quat_mat4(q, rotation);
    glm_mat4_mul(new_node->matrix, rotation, new_node->matrix);
  }
  if (node->has_scale) {
    vec3 scale = GLM_VEC3_ZERO_INIT;
    glm_vec3_copy(node->scale, scale);
    glm_scale(new_node->matrix, scale);
  }
  if (node->has_matrix) {
    memcpy(new_node->matrix, node->matrix, sizeof(node->matrix));
  }

  // Load node's children
  if (node->children_count > 0) {
    for (cgltf_size i = 0, len = node->children_count; i < len; ++i) {
      gltf_model_load_node(model, node, node->children[i], data, vertices,
                           vertex_count, indices, index_count, flip_uvs);
    }
  }

  // If the node contains mesh data, we load vertices and indices from the
  // buffers In glTF this is done via accessors and buffer views
  if (node->mesh != NULL) {
    cgltf_mesh* mesh      = node->mesh;
    gltf_mesh_t* new_mesh = &model->meshes[node->mesh - data->meshes];
    gltf_mesh_init(new_mesh, model->wgpu_context, new_node->matrix);

    new_mesh->primitive_count = (uint32_t)mesh->primitives_count;
    new_mesh->primitives
      = calloc(new_mesh->primitive_count, sizeof(*new_mesh->primitives));

    for (uint32_t i = 0; i < mesh->primitives_count; ++i) {
      cgltf_primitive* primitive = &mesh->primitives[i];

      if (primitive->indices == NULL) {
        continue;
      }

      uint32_t index_start      = *index_count;
      uint32_t vertex_start     = *vertex_count;
      uint32_t prim_index_count = 0;

      vec3 pos_min = GLM_VEC3_ZERO_INIT;
      vec3 pos_max = GLM_VEC3_ZERO_INIT;

      /* Vertices */
      {
        float* buffer_pos       = NULL;
        float* buffer_normals   = NULL;
        float* buffer_texcoords = NULL;

        cgltf_accessor* pos_accessor      = NULL;
        cgltf_accessor* normal_accessor   = NULL;
        cgltf_accessor* texcoord_accessor = NULL;

        for (uint32_t j = 0; j < primitive->attributes_count; j++) {
          if (primitive->attributes[j].type == cgltf_attribute_type_position) {
            // Get buffer data for vertex positions
            pos_accessor                = primitive->attributes[j].data;
            cgltf_buffer_view* pos_view = pos_accessor->buffer_view;
            buffer_pos
              = (float*)&((unsigned char*)pos_view->buffer
                            ->data)[pos_accessor->offset + pos_view->offset];
          }
          if (primitive->attributes[j].type == cgltf_attribute_type_normal) {
            // Get buffer data for vertex normals
            normal_accessor                = primitive->attributes[j].data;
            cgltf_buffer_view* normal_view = normal_accessor->buffer_view;
            buffer_normals                 = (float*)&(
              (unsigned char*)normal_view->buffer
                ->data)[normal_accessor->offset + normal_view->offset];
          }
          // Get buffer data for vertex texture coordinates
          if (primitive->attributes[j].type == cgltf_attribute_type_texcoord) {
            texcoord_accessor                = primitive->attributes[j].data;
            cgltf_buffer_view* texcoord_view = texcoord_accessor->buffer_view;
            buffer_texcoords                 = (float*)&(
              ((unsigned char*)texcoord_view->buffer
                 ->data)[texcoord_accessor->offset + texcoord_view->offset]);
          }
        }

        ASSERT(pos_accessor != NULL);

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

        *vertex_count += (uint32_t)pos_accessor->count;
        *vertices = realloc(*vertices, (*vertex_count) * sizeof(gltf_vertex_t));

        for (uint32_t v = 0; v < pos_accessor->count; ++v) {
          gltf_vertex_t vert = {0};
          memcpy(&vert.pos, &buffer_pos[v * 3], sizeof(vec3));
          if (normal_accessor != NULL) {
            memcpy(&vert.normal, &buffer_normals[v * 3], sizeof(vec3));
          }
          if (texcoord_accessor != NULL) {
            memcpy(&vert.uv, &buffer_texcoords[v * 2], sizeof(vec2));
          }
          if (flip_uvs) {
            vert.uv[1] = 1.0f - vert.uv[1]; // flip y
          }
          glm_vec3_one(vert.color);
          (*vertices)[(*vertex_count) - pos_accessor->count + v] = vert;
        }
      }

      /* Indices */
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
            ASSERT(false);
          }
        }

        gltf_primitive_t new_primitive = {0};
        gltf_primitive_init(&new_primitive, index_start, prim_index_count,
                            primitive->material - data->materials);
        gltf_primitive_set_dimensions(&new_primitive, pos_min, pos_max);
        new_mesh->primitives[i] = new_primitive;
      }

      if (node->mesh != NULL) {
        new_node->mesh = &model->meshes[node->mesh - data->meshes];
      }
    }
  }
}

gltf_model_t* gltf_model_load_from_file(gltf_model_load_options_t* load_options)
{
  gltf_model_t* gltf_model = NULL;

  cgltf_options options = {0};
  cgltf_data* gltf_data = NULL;
  cgltf_result result
    = cgltf_parse_file(&options, load_options->filename, &gltf_data);
  if (result == cgltf_result_success) {
    cgltf_result buffers_result
      = cgltf_load_buffers(&options, gltf_data, load_options->filename);

    if (buffers_result == cgltf_result_success) {
      gltf_model = malloc(sizeof(gltf_model_t));
      gltf_model_init(gltf_model, load_options);

      // Load images
      gltf_model_load_images(gltf_model, gltf_data);

      // Load materials
      gltf_model_load_materials(gltf_model, gltf_data);

      // Load textures
      gltf_model_load_textures(gltf_model, gltf_data);

      // If there is no default scene specified, then the default is the first
      // one. It is not an error for a glTF file to have zero scenes.
      const cgltf_scene* scene
        = gltf_data->scene ? gltf_data->scene : gltf_data->scenes;
      if (!scene) {
        return NULL;
      }

      // Nodes and meshes
      gltf_model->node_count = (uint32_t)gltf_data->nodes_count;
      gltf_model->nodes = calloc(gltf_model->node_count, sizeof(gltf_node_t));

      gltf_model->mesh_count = (uint32_t)gltf_data->meshes_count;
      gltf_model->meshes = calloc(gltf_model->mesh_count, sizeof(gltf_mesh_t));

      gltf_vertex_t* vertices    = NULL;
      gltf_model->vertices.count = 0u;
      uint32_t* indices          = NULL;
      gltf_model->indices.count  = 0u;

      // Recursively create all nodes.
      for (cgltf_size i = 0, len = scene->nodes_count; i < len; ++i) {
        gltf_model_load_node(gltf_model, NULL, scene->nodes[i], gltf_data,
                             &vertices, &gltf_model->vertices.count, &indices,
                             &gltf_model->indices.count,
                             load_options->flip_uvs);
      }

      // Vertex and index buffers
      size_t vertex_buffer_size
        = gltf_model->vertices.count * sizeof(gltf_vertex_t);
      size_t index_buffer_size = gltf_model->indices.count * sizeof(uint32_t);

      ASSERT((vertex_buffer_size > 0) && (index_buffer_size > 0));

      // Create vertex buffer
      gltf_model->vertices.buffer = wgpu_create_buffer_from_data(
        load_options->wgpu_context, vertices, vertex_buffer_size,
        WGPUBufferUsage_Vertex);

      // Create index buffer
      gltf_model->indices.buffer = wgpu_create_buffer_from_data(
        load_options->wgpu_context, indices, index_buffer_size,
        WGPUBufferUsage_Index);

      if (vertices != NULL) {
        free(vertices);
      }
      if (indices != NULL) {
        free(indices);
      }
    }

    // Cleanup
    cgltf_free(gltf_data);
  }
  else {
    log_error("Could not load gltf file: %s, error: %d\n",
              load_options->filename, result);
    return NULL;
  }

  return gltf_model;
}

static void gltf_model_setup_material_bind_groups(
  gltf_model_t* gltf_model, WGPUBindGroupLayout texture_bind_group_layout)
{
  for (uint32_t i = 0; i < gltf_model->image_count; ++i) {
    gltf_image_t* image = &gltf_model->images[i];

    WGPUBindGroupEntry bg_entries[2] = {
        [0] = (WGPUBindGroupEntry) {
          /* Binding 0: Texture view */
          .binding     = 0,
          .textureView = image->texture.view,
        },
        [1] = (WGPUBindGroupEntry) {
          /* Binding 1: Texture sampler */
          .binding = 1,
          .sampler = image->texture.sampler,
        }
      };
    image->bind_group = wgpuDeviceCreateBindGroup(
      gltf_model->wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = "Material bind group",
        .layout     = texture_bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(image->bind_group != NULL)
  }
}

static void gltf_model_setup_mesh_bind_groups(
  gltf_model_t* gltf_model, WGPUBindGroupLayout model_data_bind_group_layout)
{
  for (uint32_t i = 0; i < gltf_model->mesh_count; ++i) {
    gltf_mesh_t* mesh = &gltf_model->meshes[i];

    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Mesh bind group",
      .layout     = model_data_bind_group_layout,
      .entryCount = 1,
      .entries = &(WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = mesh->uniform_buffer,
        .offset  = 0,
        .size    = sizeof(mat4),
      },
    };

    mesh->bind_group
      = wgpuDeviceCreateBindGroup(gltf_model->wgpu_context->device, &bg_desc);
    ASSERT(mesh->bind_group != NULL)
  }
}

// Draw a single node including child nodes (if present)
static void gltf_model_draw_node(gltf_model_t* gltf_model, gltf_node_t* node)
{
  if (node->mesh && node->mesh->primitive_count > 0) {
    // Traverse the node hierarchy to the top-most parent to get the final
    // matrix of the current node
    mat4 node_matrix = GLM_MAT4_ZERO_INIT;
    glm_mat4_copy(node->matrix, node_matrix);
    gltf_node_t* current_parent = node->parent;
    while (current_parent != NULL) {
      glm_mat4_mul(current_parent->matrix, node_matrix, node_matrix);
      current_parent = current_parent->parent;
    }
    // Pass the final matrix to the vertex shader
    wgpuRenderPassEncoderSetBindGroup(gltf_model->wgpu_context->rpass_enc, 2,
                                      node->mesh->bind_group, 0, 0);
    wgpu_queue_write_buffer(gltf_model->wgpu_context,
                            node->mesh->uniform_buffer, 0, &node_matrix,
                            sizeof(mat4));
    for (uint32_t i = 0; i < node->mesh->primitive_count; ++i) {
      gltf_primitive_t* primitive = &node->mesh->primitives[i];
      if (primitive->index_count > 0) {
        // Get the texture index for this primitive
        gltf_material_t* prim_material
          = &gltf_model->materials[primitive->material_index];
        gltf_texture_t* texture
          = &gltf_model->textures[prim_material->base_color_image_index];
        // Set the bind group for the current primitive's texture
        wgpuRenderPassEncoderSetBindGroup(
          gltf_model->wgpu_context->rpass_enc, 1,
          gltf_model->images[texture->image_index].bind_group, 0, 0);
        wgpuRenderPassEncoderDrawIndexed(gltf_model->wgpu_context->rpass_enc,
                                         primitive->index_count, 1,
                                         primitive->first_index, 0, 0);
      }
    }
  }
  for (uint32_t i = 0; i < node->child_count; ++i) {
    gltf_model_draw_node(gltf_model, node->children[i]);
  }
}

// Draw the glTF scene starting at the top-level-nodes
static void gltf_model_draw(gltf_model_t* gltf_model)
{
  wgpu_context_t* wgpu_context = gltf_model->wgpu_context;
  // All vertices and indices are stored in single buffers, so we only need to
  // bind once
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       gltf_model->vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, gltf_model->indices.buffer, WGPUIndexFormat_Uint32,
    0, WGPU_WHOLE_SIZE);
  // Render all nodes at top-level
  for (uint32_t i = 0; i < gltf_model->node_count; ++i) {
    gltf_model_draw_node(gltf_model, &gltf_model->nodes[i]);
  }
}

/* -------------------------------------------------------------------------- *
 * glTF loading example
 * -------------------------------------------------------------------------- */

static struct ubo_scene_t {
  mat4 projection;
  mat4 model;
  vec4 lightPos;
} ubo_scene = {
  .lightPos = {5.0f, 5.0f, -5.0f, 1.0f},
};

// Uniform buffers
static WGPUBuffer scene_uniform_buffer = NULL;

// Bind group layouts
static WGPUBindGroupLayout ubo_scene_bind_group_layout  = NULL; // UBOScene
static WGPUBindGroupLayout model_data_bind_group_layout = NULL; // ModelData
static WGPUBindGroupLayout texture_bind_group_layout    = NULL; // Texture

// Bind group
static WGPUBindGroup ubo_scene_bind_group = NULL;

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Pipelines
static WGPURenderPipeline solid_pipeline = NULL;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

// The glTF model
static gltf_model_t* gltf_model = NULL;

// Other variables
static const char* example_title = "glTF Model Rendering";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, -0.1f, -1.0f});
  camera_set_rotation(context->camera, (vec3){0.0f, -135.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
  camera_set_rotation_speed(context->camera, 0.5f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  gltf_model = gltf_model_load_from_file(&(gltf_model_load_options_t){
    .wgpu_context = wgpu_context,
    .filename     = "models/FlightHelmet/glTF/FlightHelmet.gltf",
    .flip_uvs     = false,
  });
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Pass matrices to the shaders
  glm_mat4_copy(context->camera->matrices.perspective, ubo_scene.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_scene.model);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, scene_uniform_buffer, 0,
                          &ubo_scene, sizeof(ubo_scene));
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Vertex shader uniform buffer block */
  scene_uniform_buffer = wgpuDeviceCreateBuffer(
    context->wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = "Scene uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_scene),
    });

  update_uniform_buffers(context);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Attachment is acquired in render loop */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.25f,
        .g = 0.25f,
        .b = 0.25f,
        .a = 1.0f,
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layouts */

  // Bind group for uniform UBOScene
  {
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "UBOScene uniform - Bind group layout",
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Vertex shader) => UBOScene
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(ubo_scene),
        },
        .sampler = {0},
      }
    };
    ubo_scene_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(ubo_scene_bind_group_layout != NULL);
  }

  // Bind group for uniform UBOScene material texture
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: texture2D (Fragment shader) => baseColorTexture
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: sampler (Fragment shader) => defaultSampler
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type  =WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "UBOScene material texture - Uniform bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    texture_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(texture_bind_group_layout != NULL);
  }

  // Bind group for uniform ModelData
  {
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "ModelData - Uniform bind group",
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Vertex shader) => modelData
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(mat4),
        },
        .sampler = {0},
      }
    };
    model_data_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(model_data_bind_group_layout != NULL);
  }

  // Bind Group
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "UBO scene - Bind group",
    .layout     = ubo_scene_bind_group_layout,
    .entryCount = 1,
    .entries = &(WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = scene_uniform_buffer,
      .offset  = 0,
      .size    = sizeof(ubo_scene),
    },
  };
  ubo_scene_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(ubo_scene_bind_group != NULL);

  // Create the pipeline layout
  WGPUBindGroupLayout bind_group_layouts[3] = {
    ubo_scene_bind_group_layout,  // set 0
    texture_bind_group_layout,    // set 1
    model_data_bind_group_layout, // set 2
  };
  // Pipeline layout
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .label                = "Render pipeline layout",
    .bindGroupLayoutCount = 3,
    .bindGroupLayouts     = bind_group_layouts,
  };
  pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                   &pipeline_layout_desc);
  ASSERT(pipeline_layout != NULL);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(gltf_loading, sizeof(gltf_vertex_t),
                            /* Attribute location 0: Position */
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                                               offsetof(gltf_vertex_t, pos)),
                            /* Attribute location 1: Normal */
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                                               offsetof(gltf_vertex_t, normal)),
                            /* Attribute location 2: Texture coordinates (UV) */
                            WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x2,
                                               offsetof(gltf_vertex_t, uv)),
                            /* Attribute location 3: Color */
                            WGPU_VERTATTR_DESC(3, WGPUVertexFormat_Float32x3,
                                               offsetof(gltf_vertex_t, color)));

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader SPIR-V */
              .label = "Mesh - Vertex shader SPIR-V",
              .file  = "shaders/gltf_loading/mesh.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &gltf_loading_vertex_buffer_layout,
          });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader SPIR-V */
              .label = "Mesh - Fragment shader SPIR-V",
              .file  = "shaders/gltf_loading/mesh.frag.spv",
            },
            .target_count = 1,
            .targets      = &color_target_state,
          });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Create rendering pipeline using the specified states */
  solid_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Solid mesh - Render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    load_assets(context->wgpu_context);
    setup_camera(context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    gltf_model_setup_material_bind_groups(gltf_model,
                                          texture_bind_group_layout);
    gltf_model_setup_mesh_bind_groups(gltf_model, model_data_bind_group_layout);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, solid_pipeline);

  /* Bind scene matrices descriptor to set 0 */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    ubo_scene_bind_group, 0, 0);
  gltf_model_draw(gltf_model);

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, NULL);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_context_t* wgpu_context)
{
  /* Get next image in the swap chain (back/front buffer) */
  wgpu_swap_chain_get_current_image(wgpu_context);

  /* Create command buffer */
  WGPUCommandBuffer command_buffer = build_command_buffer(wgpu_context);
  ASSERT(command_buffer != NULL);

  /* Submit command buffer to the queue */
  wgpu_flush_command_buffers(wgpu_context, &command_buffer, 1);

  /* Present the current buffer to the swap chain */
  wgpu_swap_chain_present(wgpu_context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context->wgpu_context);
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_uniform_buffers(context);
}

/* Clean up used WebGPU resources */
static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  gltf_model_destroy(gltf_model);
  WGPU_RELEASE_RESOURCE(Buffer, scene_uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, ubo_scene_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, model_data_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, texture_bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, solid_pipeline)
}

void example_gltf_loading(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title   = example_title,
     .overlay = true,
     .vsync   = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
