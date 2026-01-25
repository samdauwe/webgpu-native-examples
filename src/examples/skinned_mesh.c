#include "meshes.h"
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Skinned Mesh
 *
 * This example demonstrates basic GLTF loading and mesh skinning, ported from
 * https://webgl2fundamentals.org/webgl/lessons/webgl-skinning.html. Mesh data,
 * per vertex attributes, and skin inverseBindMatrices are taken from the JSON
 * parsed from the binary output of the .glb file. Animations are generated
 * programmatically, with animated joint matrices updated and passed to shaders
 * per frame via uniform buffers.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/skinnedMesh
 * https://webgl2fundamentals.org/webgl/lessons/webgl-skinning.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* gltf_vertex_shader_wgsl;
static const char* gltf_fragment_shader_wgsl;
static const char* grid_vertex_shader_wgsl;
static const char* grid_fragment_shader_wgsl;
static const char* skybox_vertex_shader_wgsl;
static const char* skybox_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define MAX_JOINTS (64)
#define MAT4X4_BYTES (64)

/* Skybox constants */
#define SKYBOX_FACES (6)
#define SKYBOX_FACE_WIDTH (2048)
#define SKYBOX_FACE_HEIGHT (2048)
#define SKYBOX_FACE_BYTES (SKYBOX_FACE_WIDTH * SKYBOX_FACE_HEIGHT * 4)

/* Render modes */
typedef enum render_mode_t {
  RENDER_MODE_NORMAL  = 0,
  RENDER_MODE_JOINTS  = 1,
  RENDER_MODE_WEIGHTS = 2,
} render_mode_t;

/* Skin modes */
typedef enum skin_mode_t {
  SKIN_MODE_ON  = 0,
  SKIN_MODE_OFF = 1,
} skin_mode_t;

/* Object types */
typedef enum object_type_t {
  OBJECT_TYPE_WHALE        = 0,
  OBJECT_TYPE_SKINNED_GRID = 1,
} object_type_t;

/* -------------------------------------------------------------------------- *
 * GLTF Structures
 * -------------------------------------------------------------------------- */

/* GLTF Vertex */
typedef struct gltf_vertex_t {
  vec3 position;
  vec3 normal;
  vec2 texcoord;
  uint8_t joints[4];
  vec4 weights;
} gltf_vertex_t;

/* GLTF Primitive */
typedef struct gltf_primitive_t {
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  uint32_t vertex_count;
  uint32_t index_count;
  WGPURenderPipeline pipeline;
} gltf_primitive_t;

/* GLTF Node */
typedef struct gltf_node_t {
  char name[64];
  int32_t parent_index;
  vec3 translation;
  versor rotation;
  vec3 scale;
  mat4 local_matrix;
  mat4 world_matrix;
  int32_t mesh_index;
  int32_t skin_index;
  WGPUBuffer uniform_buffer;
  WGPUBindGroup bind_group;
} gltf_node_t;

/* GLTF Skin */
typedef struct gltf_skin_t {
  char name[64];
  uint32_t* joints;
  uint32_t joint_count;
  mat4* inverse_bind_matrices;
  WGPUBuffer joint_matrices_buffer;
  WGPUBuffer inverse_bind_matrices_buffer;
  WGPUBindGroup bind_group;
} gltf_skin_t;

/* GLTF Mesh */
typedef struct gltf_mesh_t {
  char name[64];
  gltf_primitive_t* primitives;
  uint32_t primitive_count;
} gltf_mesh_t;

/* GLTF Scene */
typedef struct gltf_scene_t {
  gltf_node_t* nodes;
  uint32_t node_count;
  gltf_mesh_t* meshes;
  uint32_t mesh_count;
  gltf_skin_t* skins;
  uint32_t skin_count;
  uint8_t* glb_buffer;
  size_t glb_buffer_size;
  cgltf_data* gltf_data;
} gltf_scene_t;

/* -------------------------------------------------------------------------- *
 * Grid Structures
 * -------------------------------------------------------------------------- */

/* Grid buffers */
typedef struct grid_buffers_t {
  wgpu_buffer_t positions;
  wgpu_buffer_t joints;
  wgpu_buffer_t weights;
  wgpu_buffer_t indices;
  uint32_t index_count;
} grid_buffers_t;

/* Bone collection */
typedef struct bone_collection_t {
  mat4 transforms[5];
  mat4 bind_poses[5];
  mat4 bind_poses_inv[5];
  uint32_t bone_count;
} bone_collection_t;

/* -------------------------------------------------------------------------- *
 * State Structure
 * -------------------------------------------------------------------------- */

static struct {
  /* Scene and objects */
  gltf_scene_t whale_scene;
  grid_buffers_t grid_buffers;
  bone_collection_t grid_bones;

  /* Uniforms */
  struct {
    mat4 projection;
    mat4 view;
    mat4 model;
  } camera_matrices;
  struct {
    WGPUBuffer buffer;
    WGPUBindGroup bind_group;
  } camera_uniform;
  struct {
    uint32_t render_mode;
    uint32_t skin_mode;
    WGPUBuffer buffer;
    WGPUBindGroup bind_group;
  } general_uniforms;

  /* Pipeline resources */
  WGPUBindGroupLayout camera_bind_group_layout;
  WGPUBindGroupLayout general_bind_group_layout;
  WGPUBindGroupLayout node_bind_group_layout;
  WGPUBindGroupLayout skin_bind_group_layout;
  WGPURenderPipeline grid_pipeline;

  /* Skybox resources */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
    WGPUSampler sampler;
    WGPUBuffer vertex_buffer;
    WGPUBuffer uniform_buffer;
    WGPUBindGroup bind_group;
    WGPUBindGroupLayout bind_group_layout;
    WGPURenderPipeline pipeline;
    uint8_t face_pixels[SKYBOX_FACES][SKYBOX_FACE_BYTES];
    int load_count;
    bool is_dirty;
    bool initialized;
  } skybox;

  /* Render pass */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth_texture;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI settings */
  struct {
    float camera_x;
    float camera_y;
    float camera_z;
    float object_scale;
    float angle;
    float speed;
    object_type_t object_type;
    render_mode_t render_mode;
    skin_mode_t skin_mode;
    bool skybox_enabled;
  } settings;

  /* Animation */
  mat4 orig_matrices[MAX_JOINTS];
  bool orig_matrices_initialized[MAX_JOINTS];

  /* Original TRS values for animation */
  vec3 orig_translations[MAX_JOINTS];
  versor orig_rotations[MAX_JOINTS];
  vec3 orig_scales[MAX_JOINTS];

  /* File loading */
  bool glb_loaded;
  bool initialized;

  /* Frame timing */
  uint64_t last_frame_time;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.3, 0.3, 0.3, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .settings = {
    .camera_x        = 0.0f,
    .camera_y        = -5.1f,
    .camera_z        = -14.6f,
    .object_scale    = 1.0f,
    .angle           = 0.2f,
    .speed           = 50.0f,
    .object_type     = OBJECT_TYPE_WHALE,
    .render_mode     = RENDER_MODE_NORMAL,
    .skin_mode       = SKIN_MODE_ON,
    .skybox_enabled  = true,
  },
  .glb_loaded  = false,
  .initialized = false,
};

/* -------------------------------------------------------------------------- *
 * Helper Functions
 * -------------------------------------------------------------------------- */

static void anim_skinned_grid(mat4* bone_transforms, float angle)
{
  /* Match TypeScript:
   * mat4.rotateZ(m, angle, boneTransforms[0]);  // boneTransforms[0] =
   * rotateZ(m) mat4.translate(boneTransforms[0], vec3.create(4, 0, 0), m);  //
   * m = translate(boneTransforms[0]) mat4.rotateZ(m, angle, boneTransforms[1]);
   * // boneTransforms[1] = rotateZ(m) mat4.translate(boneTransforms[1],
   * vec3.create(4, 0, 0), m);  // m = translate(boneTransforms[1])
   * mat4.rotateZ(m, angle, boneTransforms[2]);  // boneTransforms[2] =
   * rotateZ(m)
   */
  mat4 m;
  glm_mat4_identity(m);

  /* Bone 0: rotate m, store in bone_transforms[0] */
  glm_rotate_z(m, angle, bone_transforms[0]);
  /* Then translate bone_transforms[0], store in m for next bone */
  glm_mat4_copy(bone_transforms[0], m);
  glm_translate(m, (vec3){4.0f, 0.0f, 0.0f});

  /* Bone 1: rotate m, store in bone_transforms[1] */
  glm_rotate_z(m, angle, bone_transforms[1]);
  /* Then translate bone_transforms[1], store in m for next bone */
  glm_mat4_copy(bone_transforms[1], m);
  glm_translate(m, (vec3){4.0f, 0.0f, 0.0f});

  /* Bone 2: just rotate m, store in bone_transforms[2] */
  glm_rotate_z(m, angle, bone_transforms[2]);
}

static void create_bone_collection(bone_collection_t* collection,
                                   uint32_t num_bones)
{
  collection->bone_count = num_bones;

  /* Initialize transforms and bind poses */
  for (uint32_t i = 0; i < num_bones; ++i) {
    glm_mat4_identity(collection->transforms[i]);
    glm_mat4_identity(collection->bind_poses[i]);
  }

  /* Get initial bind pose positions */
  anim_skinned_grid(collection->bind_poses, 0.0f);

  /* Calculate inverse bind poses */
  for (uint32_t i = 0; i < num_bones; ++i) {
    glm_mat4_inv(collection->bind_poses[i], collection->bind_poses_inv[i]);
  }
}

static void update_node_matrix(gltf_node_t* node)
{
  /* Compose local matrix from TRS
   * TypeScript does: T * R * S order
   * - Start with identity
   * - Scale it
   * - Rotate the result
   * - Translate the result
   */
  mat4 scale_mat, rotation_mat;

  /* 1. Create scale matrix */
  glm_mat4_identity(scale_mat);
  glm_scale(scale_mat, node->scale);

  /* 2. Create rotation matrix from quaternion */
  glm_quat_mat4(node->rotation, rotation_mat);

  /* 3. Multiply: rotation * scale */
  glm_mat4_mul(rotation_mat, scale_mat, node->local_matrix);

  /* 4. Translate the result */
  glm_translate(node->local_matrix, node->translation);
}

static void update_world_matrix(gltf_scene_t* scene, gltf_node_t* node,
                                mat4 parent_matrix)
{
  /* Update world matrix */
  if (parent_matrix) {
    glm_mat4_mul(parent_matrix, node->local_matrix, node->world_matrix);
  }
  else {
    glm_mat4_copy(node->local_matrix, node->world_matrix);
  }

  /* Find child nodes and update their world matrices */
  for (uint32_t i = 0; i < scene->node_count; ++i) {
    if (scene->nodes[i].parent_index == (int32_t)i) {
      continue; /* Skip self */
    }
    if (scene->nodes[i].parent_index >= 0) {
      gltf_node_t* parent = &scene->nodes[scene->nodes[i].parent_index];
      if (parent == node) {
        update_world_matrix(scene, &scene->nodes[i], node->world_matrix);
      }
    }
  }
}

static void anim_whale_skin(wgpu_context_t* wgpu_context, gltf_skin_t* skin,
                            float angle)
{
  UNUSED_VAR(wgpu_context);

  for (uint32_t i = 0; i < skin->joint_count; ++i) {
    uint32_t joint_index = skin->joints[i];

    /* Bounds check */
    if (joint_index >= state.whale_scene.node_count) {
      continue;
    }

    gltf_node_t* node      = &state.whale_scene.nodes[joint_index];
    mat4* orig_matrix      = &state.orig_matrices[joint_index];
    bool* orig_matrix_init = &state.orig_matrices_initialized[joint_index];

    /* Store original TRS values on first run */
    if (!*orig_matrix_init) {
      glm_mat4_copy(node->local_matrix, *orig_matrix);
      glm_vec3_copy(node->translation, state.orig_translations[joint_index]);
      glm_quat_copy(node->rotation, state.orig_rotations[joint_index]);
      glm_vec3_copy(node->scale, state.orig_scales[joint_index]);
      *orig_matrix_init = true;
    }

    /* TypeScript approach: rotate the original matrix, then extract TRS
     * MATCHING TypeScript exactly:
     * - mat4.getTranslation(m) -> translation from column 3
     * - mat4.getScaling(m) -> scale from column lengths
     * - quat.fromMat(m) -> quaternion from matrix (NOT normalized!) */
    mat4 m;
    glm_mat4_copy(*orig_matrix, m);

    /* Apply rotations based on joint index */
    if (joint_index == 1 || joint_index == 0) {
      glm_rotate_y(m, -angle, m);
    }
    else if (joint_index == 3 || joint_index == 4) {
      glm_rotate_x(m, (joint_index == 3) ? angle : -angle, m);
    }
    else {
      glm_rotate_z(m, angle, m);
    }

    /* Extract translation from column 3 (matches mat4.getTranslation) */
    node->translation[0] = m[3][0];
    node->translation[1] = m[3][1];
    node->translation[2] = m[3][2];

    /* Extract scale from column lengths (matches mat4.getScaling) */
    node->scale[0] = glm_vec3_norm((vec3){m[0][0], m[0][1], m[0][2]});
    node->scale[1] = glm_vec3_norm((vec3){m[1][0], m[1][1], m[1][2]});
    node->scale[2] = glm_vec3_norm((vec3){m[2][0], m[2][1], m[2][2]});

    /* Extract quaternion from matrix
     * Note: TypeScript uses quat.fromMat(m) which works on the scaled matrix
     * We need to normalize the matrix columns before extracting quaternion */
    mat4 normalized;
    glm_mat4_copy(m, normalized);

    /* Normalize the rotation part (first 3 columns) by dividing by scale */
    if (node->scale[0] > 0.0001f) {
      normalized[0][0] /= node->scale[0];
      normalized[0][1] /= node->scale[0];
      normalized[0][2] /= node->scale[0];
    }
    if (node->scale[1] > 0.0001f) {
      normalized[1][0] /= node->scale[1];
      normalized[1][1] /= node->scale[1];
      normalized[1][2] /= node->scale[1];
    }
    if (node->scale[2] > 0.0001f) {
      normalized[2][0] /= node->scale[2];
      normalized[2][1] /= node->scale[2];
      normalized[2][2] /= node->scale[2];
    }

    /* Now extract quaternion from normalized rotation matrix */
    glm_mat4_quat(normalized, node->rotation);

    /* Rebuild local matrix from TRS */
    update_node_matrix(node);
  }
}

static void update_skin_buffers(wgpu_context_t* wgpu_context, gltf_skin_t* skin,
                                uint32_t skinned_mesh_node_index)
{
  mat4 joint_matrices[MAX_JOINTS];

  /* Get the inverse of the skinned mesh node's world matrix */
  mat4 global_world_inverse;
  glm_mat4_inv(state.whale_scene.nodes[skinned_mesh_node_index].world_matrix,
               global_world_inverse);

  /* Calculate joint matrices: globalWorldInverse * joint.worldMatrix */
  for (uint32_t i = 0; i < skin->joint_count; ++i) {
    uint32_t joint_index = skin->joints[i];

    /* Bounds check */
    if (joint_index >= state.whale_scene.node_count) {
      glm_mat4_identity(joint_matrices[i]);
      continue;
    }

    gltf_node_t* node = &state.whale_scene.nodes[joint_index];

    /* joint_matrix = globalWorldInverse * joint.worldMatrix */
    glm_mat4_mul(global_world_inverse, node->world_matrix, joint_matrices[i]);
  }

  /* Upload to GPU */
  wgpuQueueWriteBuffer(wgpu_context->queue, skin->joint_matrices_buffer, 0,
                       joint_matrices, skin->joint_count * sizeof(mat4));
}

/* -------------------------------------------------------------------------- *
 * GLTF Loading Functions
 * -------------------------------------------------------------------------- */

static void parse_gltf_nodes(gltf_scene_t* scene, cgltf_data* data)
{
  scene->node_count = (uint32_t)data->nodes_count;
  scene->nodes = (gltf_node_t*)calloc(scene->node_count, sizeof(gltf_node_t));

  for (uint32_t i = 0; i < scene->node_count; ++i) {
    cgltf_node* src_node = &data->nodes[i];
    gltf_node_t* node    = &scene->nodes[i];

    /* Copy name */
    if (src_node->name) {
      strncpy(node->name, src_node->name, sizeof(node->name) - 1);
    }

    /* Find parent index */
    node->parent_index = -1;
    if (src_node->parent) {
      for (uint32_t j = 0; j < scene->node_count; ++j) {
        if (&data->nodes[j] == src_node->parent) {
          node->parent_index = (int32_t)j;
          break;
        }
      }
    }

    /* Get transform */
    if (src_node->has_matrix) {
      /* Matrix provided directly */
      memcpy(node->local_matrix, src_node->matrix, sizeof(mat4));
      vec4 translation_v4;
      mat4 rotation_mat;
      glm_decompose(node->local_matrix, translation_v4, rotation_mat,
                    node->scale);
      glm_vec3_copy(translation_v4, node->translation);
      glm_mat4_quat(rotation_mat, node->rotation);
    }
    else {
      /* Use TRS */
      if (src_node->has_translation) {
        memcpy(node->translation, src_node->translation, sizeof(vec3));
      }
      else {
        glm_vec3_zero(node->translation);
      }

      if (src_node->has_rotation) {
        memcpy(node->rotation, src_node->rotation, sizeof(versor));
      }
      else {
        glm_quat_identity(node->rotation);
      }

      if (src_node->has_scale) {
        memcpy(node->scale, src_node->scale, sizeof(vec3));
      }
      else {
        glm_vec3_one(node->scale);
      }

      update_node_matrix(node);
    }

    /* Mesh index */
    node->mesh_index = -1;
    if (src_node->mesh) {
      for (size_t j = 0; j < data->meshes_count; ++j) {
        if (&data->meshes[j] == src_node->mesh) {
          node->mesh_index = (int32_t)j;
          break;
        }
      }
    }

    /* Skin index */
    node->skin_index = -1;
    if (src_node->skin) {
      for (size_t j = 0; j < data->skins_count; ++j) {
        if (&data->skins[j] == src_node->skin) {
          node->skin_index = (int32_t)j;
          break;
        }
      }
    }

    /* Initialize world matrix */
    glm_mat4_identity(node->world_matrix);
  }
}

static void parse_gltf_skins(wgpu_context_t* wgpu_context, gltf_scene_t* scene,
                             cgltf_data* data)
{
  scene->skin_count = (uint32_t)data->skins_count;
  if (scene->skin_count == 0) {
    return;
  }

  scene->skins = (gltf_skin_t*)calloc(scene->skin_count, sizeof(gltf_skin_t));

  for (uint32_t i = 0; i < scene->skin_count; ++i) {
    cgltf_skin* src_skin = &data->skins[i];
    gltf_skin_t* skin    = &scene->skins[i];

    /* Copy name */
    if (src_skin->name) {
      strncpy(skin->name, src_skin->name, sizeof(skin->name) - 1);
    }

    /* Parse joints */
    skin->joint_count = (uint32_t)src_skin->joints_count;
    skin->joints      = (uint32_t*)calloc(skin->joint_count, sizeof(uint32_t));

    for (uint32_t j = 0; j < skin->joint_count; ++j) {
      cgltf_node* joint_node = src_skin->joints[j];
      /* Find node index */
      for (uint32_t k = 0; k < scene->node_count; ++k) {
        if (&data->nodes[k] == joint_node) {
          skin->joints[j] = k;
          break;
        }
      }
    }

    /* Parse inverse bind matrices */
    skin->inverse_bind_matrices
      = (mat4*)calloc(skin->joint_count, sizeof(mat4));

    if (src_skin->inverse_bind_matrices) {
      cgltf_accessor* accessor = src_skin->inverse_bind_matrices;
      const uint8_t* data_ptr
        = (const uint8_t*)cgltf_buffer_view_data(accessor->buffer_view);
      data_ptr += accessor->offset;

      for (uint32_t j = 0; j < skin->joint_count; ++j) {
        memcpy(skin->inverse_bind_matrices[j], data_ptr + j * sizeof(mat4),
               sizeof(mat4));
      }
    }

    /* Create GPU buffers */
    skin->joint_matrices_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label            = STRVIEW("Joint matrices buffer"),
        .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size             = skin->joint_count * sizeof(mat4),
        .mappedAtCreation = false,
      });

    skin->inverse_bind_matrices_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label            = STRVIEW("Inverse bind matrices buffer"),
        .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size             = skin->joint_count * sizeof(mat4),
        .mappedAtCreation = false,
      });

    /* Upload inverse bind matrices */
    wgpuQueueWriteBuffer(
      wgpu_context->queue, skin->inverse_bind_matrices_buffer, 0,
      skin->inverse_bind_matrices, skin->joint_count * sizeof(mat4));
  }
}

static void parse_gltf_meshes(wgpu_context_t* wgpu_context, gltf_scene_t* scene,
                              cgltf_data* data)
{
  scene->mesh_count = (uint32_t)data->meshes_count;
  if (scene->mesh_count == 0) {
    return;
  }

  scene->meshes = (gltf_mesh_t*)calloc(scene->mesh_count, sizeof(gltf_mesh_t));

  for (uint32_t i = 0; i < scene->mesh_count; ++i) {
    cgltf_mesh* src_mesh = &data->meshes[i];
    gltf_mesh_t* mesh    = &scene->meshes[i];

    /* Copy name */
    if (src_mesh->name) {
      strncpy(mesh->name, src_mesh->name, sizeof(mesh->name) - 1);
    }

    /* Parse primitives */
    mesh->primitive_count = (uint32_t)src_mesh->primitives_count;
    mesh->primitives      = (gltf_primitive_t*)calloc(mesh->primitive_count,
                                                      sizeof(gltf_primitive_t));

    for (uint32_t j = 0; j < mesh->primitive_count; ++j) {
      cgltf_primitive* src_prim = &src_mesh->primitives[j];
      gltf_primitive_t* prim    = &mesh->primitives[j];

      /* Find accessors */
      cgltf_accessor* position_accessor = NULL;
      cgltf_accessor* normal_accessor   = NULL;
      cgltf_accessor* texcoord_accessor = NULL;
      cgltf_accessor* joints_accessor   = NULL;
      cgltf_accessor* weights_accessor  = NULL;

      for (size_t k = 0; k < src_prim->attributes_count; ++k) {
        cgltf_attribute* attr = &src_prim->attributes[k];
        if (attr->type == cgltf_attribute_type_position) {
          position_accessor = attr->data;
        }
        else if (attr->type == cgltf_attribute_type_normal) {
          normal_accessor = attr->data;
        }
        else if (attr->type == cgltf_attribute_type_texcoord) {
          texcoord_accessor = attr->data;
        }
        else if (attr->type == cgltf_attribute_type_joints) {
          joints_accessor = attr->data;
        }
        else if (attr->type == cgltf_attribute_type_weights) {
          weights_accessor = attr->data;
        }
      }

      if (!position_accessor) {
        continue; /* Skip if no positions */
      }

      prim->vertex_count = (uint32_t)position_accessor->count;

      /* Build vertex buffer */
      gltf_vertex_t* vertices
        = (gltf_vertex_t*)calloc(prim->vertex_count, sizeof(gltf_vertex_t));

      for (uint32_t v = 0; v < prim->vertex_count; ++v) {
        /* Position */
        cgltf_accessor_read_float(position_accessor, v, vertices[v].position,
                                  3);

        /* Normal */
        if (normal_accessor) {
          cgltf_accessor_read_float(normal_accessor, v, vertices[v].normal, 3);
        }

        /* Texcoord */
        if (texcoord_accessor) {
          cgltf_accessor_read_float(texcoord_accessor, v, vertices[v].texcoord,
                                    2);
        }

        /* Joints */
        if (joints_accessor) {
          uint32_t joints[4] = {0};
          cgltf_accessor_read_uint(joints_accessor, v, joints, 4);
          vertices[v].joints[0] = (uint8_t)joints[0];
          vertices[v].joints[1] = (uint8_t)joints[1];
          vertices[v].joints[2] = (uint8_t)joints[2];
          vertices[v].joints[3] = (uint8_t)joints[3];
        }

        /* Weights */
        if (weights_accessor) {
          cgltf_accessor_read_float(weights_accessor, v, vertices[v].weights,
                                    4);
        }
      }

      /* Create vertex buffer */
      prim->vertex_buffer = wgpu_create_buffer(
        wgpu_context,
        &(wgpu_buffer_desc_t){
          .label        = "GLTF vertex buffer",
          .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
          .size         = prim->vertex_count * sizeof(gltf_vertex_t),
          .initial.data = vertices,
        });

      free(vertices);

      /* Parse indices */
      if (src_prim->indices) {
        cgltf_accessor* index_accessor = src_prim->indices;
        prim->index_count              = (uint32_t)index_accessor->count;

        uint16_t* indices
          = (uint16_t*)calloc(prim->index_count, sizeof(uint16_t));

        for (uint32_t idx = 0; idx < prim->index_count; ++idx) {
          uint32_t index_value = 0;
          cgltf_accessor_read_uint(index_accessor, idx, &index_value, 1);
          indices[idx] = (uint16_t)index_value;
        }

        /* Create index buffer */
        prim->index_buffer = wgpu_create_buffer(
          wgpu_context,
          &(wgpu_buffer_desc_t){
            .label        = "GLTF index buffer",
            .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
            .size         = prim->index_count * sizeof(uint16_t),
            .initial.data = indices,
          });

        free(indices);
      }
    }
  }
}

static void load_gltf_scene(wgpu_context_t* wgpu_context, gltf_scene_t* scene,
                            const uint8_t* buffer, size_t size)
{
  /* Parse GLTF */
  cgltf_options options = {0};
  cgltf_result result = cgltf_parse(&options, buffer, size, &scene->gltf_data);

  if (result != cgltf_result_success) {
    fprintf(stderr, "Failed to parse GLTF: %d\n", result);
    return;
  }

  /* Load buffers - for GLB files, buffers are already embedded */
  result = cgltf_load_buffers(&options, scene->gltf_data, NULL);
  if (result != cgltf_result_success) {
    fprintf(stderr, "Failed to load GLTF buffers: %d\n", result);
    cgltf_free(scene->gltf_data);
    scene->gltf_data = NULL;
    return;
  }

  /* Parse data */
  parse_gltf_nodes(scene, scene->gltf_data);
  parse_gltf_skins(wgpu_context, scene, scene->gltf_data);
  parse_gltf_meshes(wgpu_context, scene, scene->gltf_data);
}

/* Callback for asynchronously loading the GLB file */
static void glb_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    fprintf(stderr, "Failed to load GLB file, error: %d\n",
            response->error_code);
    return;
  }

  /* Store buffer */
  state.whale_scene.glb_buffer_size = response->data.size;
  state.whale_scene.glb_buffer
    = (uint8_t*)malloc(state.whale_scene.glb_buffer_size);
  memcpy(state.whale_scene.glb_buffer, response->data.ptr,
         state.whale_scene.glb_buffer_size);

  state.glb_loaded = true;
}

/* -------------------------------------------------------------------------- *
 * Grid Functions
 * -------------------------------------------------------------------------- */

/* clang-format off */
/* Grid vertex data - 2D grid matching TypeScript gridData.ts */
static const float grid_vertices[] = {
  /* B0 */       0.0f,  1.0f,   0.0f, -1.0f,
  /* CONNECTOR */ 2.0f,  1.0f,   2.0f, -1.0f,
  /* B1 */       4.0f,  1.0f,   4.0f, -1.0f,
  /* CONNECTOR */ 6.0f,  1.0f,   6.0f, -1.0f,
  /* B2 */       8.0f,  1.0f,   8.0f, -1.0f,
  /* CONNECTOR */ 10.0f, 1.0f,  10.0f, -1.0f,
  /* B3 */       12.0f, 1.0f,  12.0f, -1.0f,
};

/* Joint indices (4 per vertex) */
static const uint32_t grid_joints[] = {
  0, 0, 0, 0,  /* Vertex 0 */   0, 0, 0, 0,  /* Vertex 1 */
  0, 1, 0, 0,  /* Vertex 2 */   0, 1, 0, 0,  /* Vertex 3 */
  1, 0, 0, 0,  /* Vertex 4 */   1, 0, 0, 0,  /* Vertex 5 */
  1, 2, 0, 0,  /* Vertex 6 */   1, 2, 0, 0,  /* Vertex 7 */
  2, 0, 0, 0,  /* Vertex 8 */   2, 0, 0, 0,  /* Vertex 9 */
  1, 2, 3, 0,  /* Vertex 10 */  1, 2, 3, 0,  /* Vertex 11 */
  2, 3, 0, 0,  /* Vertex 12 */  2, 3, 0, 0,  /* Vertex 13 */
};

/* Weights (4 per vertex) */
static const float grid_weights[] = {
  1.0f, 0.0f, 0.0f, 0.0f,  /* Vertex 0 */   1.0f, 0.0f, 0.0f, 0.0f,  /* Vertex 1 */
  0.5f, 0.5f, 0.0f, 0.0f,  /* Vertex 2 */   0.5f, 0.5f, 0.0f, 0.0f,  /* Vertex 3 */
  1.0f, 0.0f, 0.0f, 0.0f,  /* Vertex 4 */   1.0f, 0.0f, 0.0f, 0.0f,  /* Vertex 5 */
  0.5f, 0.5f, 0.0f, 0.0f,  /* Vertex 6 */   0.5f, 0.5f, 0.0f, 0.0f,  /* Vertex 7 */
  1.0f, 0.0f, 0.0f, 0.0f,  /* Vertex 8 */   1.0f, 0.0f, 0.0f, 0.0f,  /* Vertex 9 */
  0.5f, 0.5f, 0.0f, 0.0f,  /* Vertex 10 */  0.5f, 0.5f, 0.0f, 0.0f,  /* Vertex 11 */
  1.0f, 0.0f, 0.0f, 0.0f,  /* Vertex 12 */  1.0f, 0.0f, 0.0f, 0.0f,  /* Vertex 13 */
};

/* Line indices for grid rendering */
static const uint16_t grid_indices[] = {
  /* B0 */       0, 1,  0, 2,  1, 3,
  /* CONNECTOR */ 2, 3,  2, 4,  3, 5,
  /* B1 */       4, 5,  4, 6,  5, 7,
  /* CONNECTOR */ 6, 7,  6, 8,  7, 9,
  /* B2 */       8, 9,  8, 10, 9, 11,
  /* CONNECTOR */ 10, 11, 10, 12, 11, 13,
  /* B3 */       12, 13,
};
/* clang-format on */

static void init_grid_buffers(wgpu_context_t* wgpu_context)
{
  state.grid_buffers.positions = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Grid positions buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(grid_vertices),
                    .initial.data = grid_vertices,
                  });

  state.grid_buffers.joints = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Grid joints buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(grid_joints),
                    .initial.data = grid_joints,
                  });

  state.grid_buffers.weights = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Grid weights buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(grid_weights),
                    .initial.data = grid_weights,
                  });

  state.grid_buffers.indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Grid indices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(grid_indices),
                    .initial.data = grid_indices,
                  });

  state.grid_buffers.index_count
    = sizeof(grid_indices) / sizeof(grid_indices[0]);
}

/* -------------------------------------------------------------------------- *
 * Bind Group Layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Camera bind group layout */
  state.camera_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Camera bind group layout"),
      .entryCount = 1,
      .entries    = &(WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = MAT4X4_BYTES * 3,
        },
      },
    });

  /* General uniforms bind group layout */
  state.general_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("General uniforms bind group layout"),
      .entryCount = 1,
      .entries    = &(WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer     = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uint32_t) * 2,
        },
      },
    });

  /* Node uniforms bind group layout */
  state.node_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Node uniforms bind group layout"),
      .entryCount = 1,
      .entries    = &(WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = MAT4X4_BYTES,
        },
      },
    });

  /* Skin bind group layout */
  WGPUBindGroupLayoutEntry skin_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer     = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = MAT4X4_BYTES,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex,
      .buffer     = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = MAT4X4_BYTES,
      },
    },
  };

  state.skin_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Skin bind group layout"),
                            .entryCount = 2,
                            .entries    = skin_entries,
                          });
}

/* -------------------------------------------------------------------------- *
 * Uniform Buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Camera uniform buffer */
  state.camera_uniform.buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Camera uniform buffer"),
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = MAT4X4_BYTES * 3,
      .mappedAtCreation = false,
    });

  state.camera_uniform.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Camera bind group"),
      .layout = state.camera_bind_group_layout,
      .entryCount = 1,
      .entries    = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.camera_uniform.buffer,
        .size    = MAT4X4_BYTES * 3,
      },
    });

  /* General uniforms buffer */
  state.general_uniforms.render_mode = state.settings.render_mode;
  state.general_uniforms.skin_mode   = state.settings.skin_mode;

  state.general_uniforms.buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("General uniforms buffer"),
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = sizeof(uint32_t) * 2,
      .mappedAtCreation = false,
    });

  uint32_t general_uniform_data[2]
    = {state.general_uniforms.render_mode, state.general_uniforms.skin_mode};
  wgpuQueueWriteBuffer(wgpu_context->queue, state.general_uniforms.buffer, 0,
                       general_uniform_data, sizeof(general_uniform_data));

  state.general_uniforms.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("General uniforms bind group"),
      .layout = state.general_bind_group_layout,
      .entryCount = 1,
      .entries    = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.general_uniforms.buffer,
        .size    = sizeof(uint32_t) * 2,
      },
    });
}

/* -------------------------------------------------------------------------- *
 * Pipelines
 * -------------------------------------------------------------------------- */

static void init_gltf_pipeline(wgpu_context_t* wgpu_context,
                               gltf_primitive_t* primitive)
{
  /* Vertex buffer layout */
  WGPUVertexAttribute vertex_attributes[5] = {
    [0] = (WGPUVertexAttribute){
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, position),
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute){
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, normal),
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute){
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = offsetof(gltf_vertex_t, texcoord),
      .shaderLocation = 2,
    },
    [3] = (WGPUVertexAttribute){
      .format         = WGPUVertexFormat_Uint8x4,
      .offset         = offsetof(gltf_vertex_t, joints),
      .shaderLocation = 3,
    },
    [4] = (WGPUVertexAttribute){
      .format         = WGPUVertexFormat_Float32x4,
      .offset         = offsetof(gltf_vertex_t, weights),
      .shaderLocation = 4,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 5,
    .attributes     = vertex_attributes,
  };

  /* Shader modules */
  WGPUShaderModule vertex_shader
    = wgpu_create_shader_module(wgpu_context->device, gltf_vertex_shader_wgsl);

  WGPUShaderModule fragment_shader = wgpu_create_shader_module(
    wgpu_context->device, gltf_fragment_shader_wgsl);

  /* Pipeline layout */
  WGPUBindGroupLayout bind_group_layouts[4] = {
    state.camera_bind_group_layout,
    state.general_bind_group_layout,
    state.node_bind_group_layout,
    state.skin_bind_group_layout,
  };

  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("GLTF pipeline layout"),
                            .bindGroupLayoutCount = 4,
                            .bindGroupLayouts     = bind_group_layouts,
                          });

  /* Create pipeline */
  primitive->pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("GLTF render pipeline"),
      .layout = pipeline_layout,
      .primitive
      = (WGPUPrimitiveState){
        .topology         = WGPUPrimitiveTopology_TriangleList,
        .stripIndexFormat = WGPUIndexFormat_Undefined,
        .frontFace        = WGPUFrontFace_CCW,
        .cullMode         = WGPUCullMode_Back,
      },
      .vertex
      = (WGPUVertexState){
        .module      = vertex_shader,
        .entryPoint  = STRVIEW("vertexMain"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .depthStencil
      = &(WGPUDepthStencilState){
        .format            = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample
      = (WGPUMultisampleState){
        .count = 1,
        .mask  = ~0u,
      },
      .fragment
      = &(WGPUFragmentState){
        .module      = fragment_shader,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets
        = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    });

  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_shader)
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_shader)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
}

static void init_grid_pipeline(wgpu_context_t* wgpu_context)
{
  /* Vertex buffer layouts */
  WGPUVertexBufferLayout vertex_buffer_layouts[3] = {
    [0] = (WGPUVertexBufferLayout){
      .arrayStride    = sizeof(float) * 2,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes
      = &(WGPUVertexAttribute){
        .format         = WGPUVertexFormat_Float32x2,
        .offset         = 0,
        .shaderLocation = 0,
      },
    },
    [1] = (WGPUVertexBufferLayout){
      .arrayStride    = sizeof(uint32_t) * 4,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes
      = &(WGPUVertexAttribute){
        .format         = WGPUVertexFormat_Uint32x4,
        .offset         = 0,
        .shaderLocation = 1,
      },
    },
    [2] = (WGPUVertexBufferLayout){
      .arrayStride    = sizeof(float) * 4,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes
      = &(WGPUVertexAttribute){
        .format         = WGPUVertexFormat_Float32x4,
        .offset         = 0,
        .shaderLocation = 2,
      },
    },
  };

  /* Shader modules */
  WGPUShaderModule vertex_shader
    = wgpu_create_shader_module(wgpu_context->device, grid_vertex_shader_wgsl);

  WGPUShaderModule fragment_shader = wgpu_create_shader_module(
    wgpu_context->device, grid_fragment_shader_wgsl);

  /* Pipeline layout */
  WGPUBindGroupLayout bind_group_layouts[3] = {
    state.camera_bind_group_layout,
    state.general_bind_group_layout,
    state.skin_bind_group_layout,
  };

  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Grid pipeline layout"),
                            .bindGroupLayoutCount = 3,
                            .bindGroupLayouts     = bind_group_layouts,
                          });

  /* Create pipeline - no depth testing for grid (matches TypeScript) */
  state.grid_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Grid render pipeline"),
      .layout = pipeline_layout,
      .primitive
      = (WGPUPrimitiveState){
        .topology         = WGPUPrimitiveTopology_LineList,
        .stripIndexFormat = WGPUIndexFormat_Undefined,
        .frontFace        = WGPUFrontFace_CCW,
        .cullMode         = WGPUCullMode_None,
      },
      .vertex
      = (WGPUVertexState){
        .module      = vertex_shader,
        .entryPoint  = STRVIEW("vertexMain"),
        .bufferCount = 3,
        .buffers     = vertex_buffer_layouts,
      },
      .multisample
      = (WGPUMultisampleState){
        .count = 1,
        .mask  = ~0u,
      },
      .fragment
      = &(WGPUFragmentState){
        .module      = fragment_shader,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets
        = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    });

  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_shader)
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_shader)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
}

/* -------------------------------------------------------------------------- *
 * Skybox Loading and Initialization
 * -------------------------------------------------------------------------- */

static void skybox_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Skybox face fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* Decode the image data */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* decoded_pixels    = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);

  if (decoded_pixels) {
    assert(img_width == SKYBOX_FACE_WIDTH);
    assert(img_height == SKYBOX_FACE_HEIGHT);
    memcpy((void*)response->buffer.ptr, decoded_pixels, SKYBOX_FACE_BYTES);
    stbi_image_free(decoded_pixels);
    ++state.skybox.load_count;

    /* Mark texture as dirty if all faces are loaded */
    if (state.skybox.load_count == SKYBOX_FACES) {
      state.skybox.is_dirty = true;
    }
  }
}

static void init_skybox_texture(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Load the 6 ocean cubemap faces */
  static const char* ocean_cubemap_paths[SKYBOX_FACES] = {
    "assets/textures/cubemaps/ocean_cube_px.jpg", /* Right  (+X) */
    "assets/textures/cubemaps/ocean_cube_nx.jpg", /* Left   (-X) */
    "assets/textures/cubemaps/ocean_cube_py.jpg", /* Top    (+Y) */
    "assets/textures/cubemaps/ocean_cube_ny.jpg", /* Bottom (-Y) */
    "assets/textures/cubemaps/ocean_cube_pz.jpg", /* Back   (+Z) */
    "assets/textures/cubemaps/ocean_cube_nz.jpg", /* Front  (-Z) */
  };

  /* Reset load count */
  state.skybox.load_count = 0;

  /* Start fetching all cubemap faces */
  for (int i = 0; i < SKYBOX_FACES; i++) {
    sfetch_send(&(sfetch_request_t){
      .path     = ocean_cubemap_paths[i],
      .callback = skybox_fetch_callback,
      .buffer   = SFETCH_RANGE(state.skybox.face_pixels[i]),
    });
  }

  state.skybox.is_dirty = true;
}

static void update_skybox_texture(wgpu_context_t* wgpu_context)
{
  if (!state.skybox.is_dirty || state.skybox.load_count != SKYBOX_FACES) {
    return;
  }

  /* Create the cubemap texture if not yet created */
  if (!state.skybox.texture) {
    state.skybox.texture = wgpuDeviceCreateTexture(
      wgpu_context->device,
      &(WGPUTextureDescriptor){
        .label = STRVIEW("Skybox cubemap texture"),
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size = (WGPUExtent3D){
          .width = SKYBOX_FACE_WIDTH,
          .height = SKYBOX_FACE_HEIGHT,
          .depthOrArrayLayers = SKYBOX_FACES,
        },
        .format = WGPUTextureFormat_RGBA8Unorm,
        .mipLevelCount = 1,
        .sampleCount = 1,
      });

    state.skybox.view = wgpuTextureCreateView(
      state.skybox.texture, &(WGPUTextureViewDescriptor){
                              .label           = STRVIEW("Skybox cubemap view"),
                              .format          = WGPUTextureFormat_RGBA8Unorm,
                              .dimension       = WGPUTextureViewDimension_Cube,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = SKYBOX_FACES,
                            });

    state.skybox.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = STRVIEW("Skybox sampler"),
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .magFilter     = WGPUFilterMode_Linear,
                              .minFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                              .maxAnisotropy = 1,
                            });
  }

  /* Upload the face data to the texture */
  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create staging buffers for each face */
  WGPUBuffer staging_buffers[SKYBOX_FACES] = {0};
  for (uint32_t face = 0; face < SKYBOX_FACES; ++face) {
    WGPUBufferDescriptor staging_buffer_desc = {
      .usage            = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
      .size             = SKYBOX_FACE_BYTES,
      .mappedAtCreation = true,
    };
    staging_buffers[face]
      = wgpuDeviceCreateBuffer(wgpu_context->device, &staging_buffer_desc);
    ASSERT(staging_buffers[face])
  }

  for (uint32_t face = 0; face < SKYBOX_FACES; ++face) {
    /* Copy texture data into staging buffer */
    void* mapping
      = wgpuBufferGetMappedRange(staging_buffers[face], 0, SKYBOX_FACE_BYTES);
    ASSERT(mapping)
    memcpy(mapping, state.skybox.face_pixels[face], SKYBOX_FACE_BYTES);
    wgpuBufferUnmap(staging_buffers[face]);

    /* Upload staging buffer to texture */
    wgpuCommandEncoderCopyBufferToTexture(cmd_encoder,
      /* Source */
      &(WGPUTexelCopyBufferInfo) {
        .buffer = staging_buffers[face],
        .layout = (WGPUTexelCopyBufferLayout) {
          .offset       = 0,
          .bytesPerRow  = SKYBOX_FACE_WIDTH * 4,
          .rowsPerImage = SKYBOX_FACE_HEIGHT,
        },
      },
      /* Destination */
      &(WGPUTexelCopyTextureInfo){
        .texture  = state.skybox.texture,
        .mipLevel = 0,
        .origin = (WGPUOrigin3D) {
            .x = 0,
            .y = 0,
            .z = face,
        },
        .aspect = WGPUTextureAspect_All,
      },
      /* Size */
      &(WGPUExtent3D){
        .width = SKYBOX_FACE_WIDTH,
        .height = SKYBOX_FACE_HEIGHT,
        .depthOrArrayLayers = 1,
      });
  }

  /* Execute the command and cleanup staging buffers */
  WGPUCommandBuffer command = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command);

  for (uint32_t face = 0; face < SKYBOX_FACES; ++face) {
    WGPU_RELEASE_RESOURCE(Buffer, staging_buffers[face])
  }
  WGPU_RELEASE_RESOURCE(CommandBuffer, command)
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  state.skybox.is_dirty = false;

  /* Now create the bind group with the loaded texture */

  WGPUBindGroupEntry bind_group_entries[3] = {
    [0] = {
      .binding = 0,
      .textureView = state.skybox.view,
    },
    [1] = {
      .binding = 1,
      .sampler = state.skybox.sampler,
    },
    [2] = {
      .binding = 2,
      .buffer = state.skybox.uniform_buffer,
      .offset = 0,
      .size = 128,
    },
  };

  state.skybox.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Skybox bind group"),
      .layout     = state.skybox.bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bind_group_entries),
      .entries    = bind_group_entries,
    });
}

static void init_skybox_buffers(wgpu_context_t* wgpu_context)
{
  /* Create skybox vertex buffer - full cube with triangulated faces (36
   * vertices) */
  static const float cube_vertices[] = {
    /* clang-format off */
    /* Front face */
    -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,

    /* Back face */
     1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,

    /* Left face */
    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,

    /* Right face */
     1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,

    /* Top face */
    -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,

    /* Bottom face */
    -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,
    /* clang-format on */
  };

  state.skybox.vertex_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Skybox vertex buffer"),
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
      .size             = sizeof(cube_vertices),
      .mappedAtCreation = true,
    });

  /* Copy vertex data */
  void* vertex_mapping = wgpuBufferGetMappedRange(state.skybox.vertex_buffer, 0,
                                                  sizeof(cube_vertices));
  memcpy(vertex_mapping, cube_vertices, sizeof(cube_vertices));
  wgpuBufferUnmap(state.skybox.vertex_buffer);

  /* Create uniform buffer for skybox matrices */
  state.skybox.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Skybox uniform buffer"),
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = 128, /* 2 mat4s (view + projection) */
    });

  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = {
      .binding = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = {
        .sampleType = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_Cube,
        .multisampled = false,
      },
    },
    [1] = {
      .binding = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = {
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
    [2] = {
      .binding = 2,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = {
        .type = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize = 128,
      },
    },
  };

  state.skybox.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Skybox bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
}

static void init_skybox_pipeline(wgpu_context_t* wgpu_context)
{
  if (!state.skybox.texture) {
    return; /* Wait until texture is loaded */
  }

  /* Create pipeline layout */
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Skybox pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts = &state.skybox.bind_group_layout,
                          });

  /* Create shader modules */
  WGPUShaderModule vertex_shader = wgpu_create_shader_module(
    wgpu_context->device, skybox_vertex_shader_wgsl);
  WGPUShaderModule fragment_shader = wgpu_create_shader_module(
    wgpu_context->device, skybox_fragment_shader_wgsl);

  /* Create render pipeline */
  state.skybox.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("Skybox render pipeline"),
      .layout = pipeline_layout,
      .vertex = {
        .module = vertex_shader,
        .entryPoint = STRVIEW("vertexMain"),
        .bufferCount = 1,
        .buffers = &(WGPUVertexBufferLayout){
          .arrayStride = 3 * sizeof(float),
          .stepMode = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes = &(WGPUVertexAttribute){
            .offset = 0,
            .shaderLocation = 0,
            .format = WGPUVertexFormat_Float32x3,
          },
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .stripIndexFormat = WGPUIndexFormat_Undefined,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode = WGPUCullMode_None, /* Don't cull skybox faces */
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = false, /* Don't write to depth buffer */
        .depthCompare = WGPUCompareFunction_LessEqual,
      },
      .multisample = {
        .count = 1,
        .mask = ~0u,
        .alphaToCoverageEnabled = false,
      },
      .fragment = &(WGPUFragmentState){
        .module = fragment_shader,
        .entryPoint = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = wgpu_context->render_format,
          .blend = NULL, /* No blending */
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    });

  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_shader)
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_shader)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)

  state.skybox.initialized = true;
}

/* -------------------------------------------------------------------------- *
 * Depth Texture
 * -------------------------------------------------------------------------- */

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  state.depth_texture.texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label         = STRVIEW("Depth texture"),
      .usage         = WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .size          = (WGPUExtent3D){
        .width              = wgpu_context->width,
        .height             = wgpu_context->height,
        .depthOrArrayLayers = 1,
      },
      .format        = WGPUTextureFormat_Depth24Plus,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });

  state.depth_texture.view
    = wgpuTextureCreateView(state.depth_texture.texture, NULL);

  state.depth_stencil_attachment.view = state.depth_texture.view;
}

static void resize_depth_texture(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_texture.view)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_texture.texture)

  init_depth_texture(wgpu_context);
}

/* -------------------------------------------------------------------------- *
 * Node Bind Groups
 * -------------------------------------------------------------------------- */

static void init_node_bind_groups(wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < state.whale_scene.node_count; ++i) {
    gltf_node_t* node = &state.whale_scene.nodes[i];

    if (node->mesh_index < 0) {
      continue; /* Skip nodes without meshes */
    }

    /* Create uniform buffer for node */
    node->uniform_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label            = STRVIEW("Node uniform buffer"),
        .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size             = sizeof(mat4),
        .mappedAtCreation = false,
      });

    /* Create bind group */
    node->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label  = STRVIEW("Node bind group"),
        .layout = state.node_bind_group_layout,
        .entryCount = 1,
        .entries    = &(WGPUBindGroupEntry){
          .binding = 0,
          .buffer  = node->uniform_buffer,
          .size    = sizeof(mat4),
        },
      });
  }
}

/* -------------------------------------------------------------------------- *
 * Skin Bind Groups
 * -------------------------------------------------------------------------- */

static void init_skin_bind_groups(wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < state.whale_scene.skin_count; ++i) {
    gltf_skin_t* skin = &state.whale_scene.skins[i];

    WGPUBindGroupEntry entries[2] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = skin->joint_matrices_buffer,
        .size    = skin->joint_count * sizeof(mat4),
      },
      [1] = (WGPUBindGroupEntry){
        .binding = 1,
        .buffer  = skin->inverse_bind_matrices_buffer,
        .size    = skin->joint_count * sizeof(mat4),
      },
    };

    skin->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Skin bind group"),
                              .layout     = state.skin_bind_group_layout,
                              .entryCount = 2,
                              .entries    = entries,
                            });
  }
}

/* -------------------------------------------------------------------------- *
 * Grid Bones
 * -------------------------------------------------------------------------- */

static WGPUBuffer grid_joint_buffer        = NULL;
static WGPUBuffer grid_inverse_bind_buffer = NULL;
static WGPUBindGroup grid_bone_bind_group  = NULL;

static void init_grid_bones(wgpu_context_t* wgpu_context)
{
  /* Create bone collection */
  create_bone_collection(&state.grid_bones, 5);

  /* Create buffers */
  grid_joint_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Grid joint uniform buffer"),
      .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
      .size             = 5 * sizeof(mat4),
      .mappedAtCreation = false,
    });

  grid_inverse_bind_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Grid inverse bind uniform buffer"),
      .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
      .size             = 5 * sizeof(mat4),
      .mappedAtCreation = false,
    });

  /* Upload inverse bind matrices */
  for (uint32_t i = 0; i < state.grid_bones.bone_count; ++i) {
    wgpuQueueWriteBuffer(wgpu_context->queue, grid_inverse_bind_buffer,
                         i * sizeof(mat4), state.grid_bones.bind_poses_inv[i],
                         sizeof(mat4));
  }

  /* Create bind group */
  WGPUBindGroupEntry entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = grid_joint_buffer,
      .size    = 5 * sizeof(mat4),
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = grid_inverse_bind_buffer,
      .size    = 5 * sizeof(mat4),
    },
  };

  grid_bone_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Grid bone bind group"),
                            .layout     = state.skin_bind_group_layout,
                            .entryCount = 2,
                            .entries    = entries,
                          });
}

/* -------------------------------------------------------------------------- *
 * Update Functions
 * -------------------------------------------------------------------------- */

static void update_camera_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  if (state.settings.object_type == OBJECT_TYPE_WHALE) {
    glm_perspective((2.0f * PI) / 5.0f, aspect, 0.1f, 100.0f,
                    state.camera_matrices.projection);
  }
  else {
    glm_ortho(-20.0f, 20.0f, -10.0f, 10.0f, -100.0f, 100.0f,
              state.camera_matrices.projection);
  }

  /* View matrix */
  glm_mat4_identity(state.camera_matrices.view);
  if (state.settings.object_type == OBJECT_TYPE_SKINNED_GRID) {
    glm_translate(state.camera_matrices.view,
                  (vec3){state.settings.camera_x * state.settings.object_scale,
                         state.settings.camera_y * state.settings.object_scale,
                         state.settings.camera_z});
  }
  else {
    glm_translate(state.camera_matrices.view,
                  (vec3){state.settings.camera_x, state.settings.camera_y,
                         state.settings.camera_z});
  }

  /* Model matrix */
  glm_mat4_identity(state.camera_matrices.model);
  glm_scale(state.camera_matrices.model,
            (vec3){state.settings.object_scale, state.settings.object_scale,
                   state.settings.object_scale});

  if (state.settings.object_type == OBJECT_TYPE_WHALE) {
    const float time = stm_sec(stm_now());
    glm_rotate_y(state.camera_matrices.model, time * 0.5f,
                 state.camera_matrices.model);
  }

  /* Upload to GPU */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.camera_uniform.buffer, 0,
                       state.camera_matrices.projection, sizeof(mat4));
  wgpuQueueWriteBuffer(wgpu_context->queue, state.camera_uniform.buffer,
                       sizeof(mat4), state.camera_matrices.view, sizeof(mat4));
  wgpuQueueWriteBuffer(wgpu_context->queue, state.camera_uniform.buffer,
                       sizeof(mat4) * 2, state.camera_matrices.model,
                       sizeof(mat4));
}

static void update_grid_bones(wgpu_context_t* wgpu_context, float angle)
{
  /* Animate bones */
  anim_skinned_grid(state.grid_bones.transforms, angle);

  /* Upload to GPU */
  for (uint32_t i = 0; i < state.grid_bones.bone_count; ++i) {
    wgpuQueueWriteBuffer(wgpu_context->queue, grid_joint_buffer,
                         i * sizeof(mat4), state.grid_bones.transforms[i],
                         sizeof(mat4));
  }
}

/* -------------------------------------------------------------------------- *
 * Initialization
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (state.initialized) {
    return EXIT_SUCCESS;
  }

  /* Initialize sokol time */
  stm_setup();

  /* Initialize ImGui overlay */
  imgui_overlay_init(wgpu_context);

  /* Initialize sokol fetch */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 8,
    .num_channels = 2,
    .num_lanes    = 4,
    .logger.func  = slog_func,
  });

  /* Initialize bind group layouts */
  init_bind_group_layouts(wgpu_context);

  /* Initialize uniform buffers */
  init_uniform_buffers(wgpu_context);

  /* Initialize depth texture */
  init_depth_texture(wgpu_context);

  /* Initialize grid */
  init_grid_buffers(wgpu_context);
  init_grid_bones(wgpu_context);
  init_grid_pipeline(wgpu_context);

  /* Initialize skybox buffers and layout early */
  init_skybox_buffers(wgpu_context);

  /* Initialize skybox texture loading */
  init_skybox_texture(wgpu_context);

  /* Load GLB file asynchronously */
  static uint8_t glb_file_buffer[8 * 1024 * 1024]; /* 8MB buffer */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/models/whale.glb",
    .callback = glb_fetch_callback,
    .buffer   = SFETCH_RANGE(glb_file_buffer),
  });

  state.initialized = true;

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  /* Set window position closer to upper left corner */
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);
  igBegin("Skinned Mesh Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Object selector */
  const char* object_items[2] = {"Whale", "Skinned Grid"};
  int32_t object_index        = (int32_t)state.settings.object_type;
  if (imgui_overlay_combo_box("Object", &object_index, object_items, 2)) {
    state.settings.object_type = (object_type_t)object_index;
    /* Update camera based on object type */
    if (state.settings.object_type == OBJECT_TYPE_SKINNED_GRID) {
      state.settings.camera_x     = -10.0f;
      state.settings.camera_y     = 0.0f;
      state.settings.object_scale = 1.27f;
    }
    else {
      if (state.settings.skin_mode == SKIN_MODE_OFF) {
        state.settings.camera_x = 0.0f;
        state.settings.camera_y = 0.0f;
        state.settings.camera_z = -11.0f;
      }
      else {
        state.settings.camera_x = 0.0f;
        state.settings.camera_y = -5.1f;
        state.settings.camera_z = -14.6f;
      }
    }
  }

  /* Render mode */
  const char* render_mode_items[3] = {"Normal", "Joints", "Weights"};
  int32_t render_mode_index        = (int32_t)state.settings.render_mode;
  if (imgui_overlay_combo_box("Render Mode", &render_mode_index,
                              render_mode_items, 3)) {
    state.settings.render_mode         = (render_mode_t)render_mode_index;
    state.general_uniforms.render_mode = state.settings.render_mode;
    uint32_t render_mode_data[1]       = {state.general_uniforms.render_mode};
    wgpuQueueWriteBuffer(wgpu_context->queue, state.general_uniforms.buffer, 0,
                         render_mode_data, sizeof(uint32_t));
  }

  /* Skin mode */
  const char* skin_mode_items[2] = {"On", "Off"};
  int32_t skin_mode_index        = (int32_t)state.settings.skin_mode;
  if (imgui_overlay_combo_box("Skin Mode", &skin_mode_index, skin_mode_items,
                              2)) {
    state.settings.skin_mode         = (skin_mode_t)skin_mode_index;
    state.general_uniforms.skin_mode = state.settings.skin_mode;

    if (state.settings.object_type == OBJECT_TYPE_WHALE) {
      if (state.settings.skin_mode == SKIN_MODE_OFF) {
        state.settings.camera_x = 0.0f;
        state.settings.camera_y = 0.0f;
        state.settings.camera_z = -11.0f;
      }
      else {
        state.settings.camera_x = 0.0f;
        state.settings.camera_y = -5.1f;
        state.settings.camera_z = -14.6f;
      }
    }

    uint32_t skin_mode_data[1] = {state.general_uniforms.skin_mode};
    wgpuQueueWriteBuffer(wgpu_context->queue, state.general_uniforms.buffer, 4,
                         skin_mode_data, sizeof(uint32_t));
  }

  /* Skybox settings */
  if (igCollapsingHeaderBoolPtr("Environment", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    igCheckbox("Enable Skybox", &state.settings.skybox_enabled);
  }

  /* Animation settings */
  if (igCollapsingHeaderBoolPtr("Animation", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_slider_float("Angle", &state.settings.angle, 0.05f, 0.5f,
                               "%.2f");
    imgui_overlay_slider_float("Speed", &state.settings.speed, 10.0f, 100.0f,
                               "%.0f");
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Frame Rendering
 * -------------------------------------------------------------------------- */

static int frame(wgpu_context_t* wgpu_context)
{
  /* Process file loading */
  sfetch_dowork();

  /* Update skybox texture when all faces are loaded */
  update_skybox_texture(wgpu_context);

  /* Initialize skybox pipeline when texture and bind group are ready */
  if (state.skybox.texture && state.skybox.bind_group
      && !state.skybox.initialized) {
    init_skybox_pipeline(wgpu_context);
  }

  /* Calculate delta time for ImGui */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);

  /* Render GUI controls */
  render_gui(wgpu_context);

  /* Initialize GLTF scene when loaded */
  if (state.glb_loaded && !state.whale_scene.gltf_data) {
    load_gltf_scene(wgpu_context, &state.whale_scene,
                    state.whale_scene.glb_buffer,
                    state.whale_scene.glb_buffer_size);

    if (state.whale_scene.gltf_data) {
      /* Create pipelines for each mesh */
      for (uint32_t i = 0; i < state.whale_scene.mesh_count; ++i) {
        gltf_mesh_t* mesh = &state.whale_scene.meshes[i];
        for (uint32_t j = 0; j < mesh->primitive_count; ++j) {
          init_gltf_pipeline(wgpu_context, &mesh->primitives[j]);
        }
      }

      /* Create bind groups */
      init_node_bind_groups(wgpu_context);
      init_skin_bind_groups(wgpu_context);
    }
  }

  /* Update camera matrices */
  update_camera_matrices(wgpu_context);

  /* Calculate animation */
  const float t     = (stm_sec(stm_now()) / 20.0f) * state.settings.speed;
  const float angle = sinf(t) * state.settings.angle;

  /* Update grid bones */
  if (state.settings.object_type == OBJECT_TYPE_SKINNED_GRID) {
    update_grid_bones(wgpu_context, angle);
  }

  /* Update whale animation */
  if (state.whale_scene.gltf_data && state.whale_scene.nodes
      && state.settings.object_type == OBJECT_TYPE_WHALE) {
    /* Update world matrices for all nodes */
    for (uint32_t i = 0; i < state.whale_scene.node_count; ++i) {
      gltf_node_t* node = &state.whale_scene.nodes[i];
      if (node->parent_index < 0) {
        update_world_matrix(&state.whale_scene, node, NULL);
      }
    }

    /* Animate skin */
    if (state.whale_scene.skin_count > 0) {
      anim_whale_skin(wgpu_context, &state.whale_scene.skins[0], angle);

      /* Note: We do NOT call update_world_matrix after animation to match
       * TypeScript behavior which uses stale world matrices. The animation
       * updates source TRS but skin.update uses worldMatrix computed before
       * animation. On next frame, updateWorldMatrix will incorporate the new
       * TRS values. */

      /* Update skin buffers - find the node that uses skin 0 */
      uint32_t skinned_node_index = 0;
      for (uint32_t i = 0; i < state.whale_scene.node_count; ++i) {
        if (state.whale_scene.nodes[i].skin_index == 0) {
          skinned_node_index = i;
          break;
        }
      }
      update_skin_buffers(wgpu_context, &state.whale_scene.skins[0],
                          skinned_node_index);

      /* Upload node world matrices */
      for (uint32_t i = 0; i < state.whale_scene.node_count; ++i) {
        gltf_node_t* node = &state.whale_scene.nodes[i];
        if (node->uniform_buffer) {
          wgpuQueueWriteBuffer(wgpu_context->queue, node->uniform_buffer, 0,
                               node->world_matrix, sizeof(mat4));
        }
      }
    }
  }

  /* Render pass */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth_texture.view;

  /* Update background clear color based on skybox setting */
  if (state.settings.skybox_enabled) {
    /* Dark background to let skybox show */
    state.color_attachment.clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0};
  }
  else {
    /* Original gray background */
    state.color_attachment.clearValue = (WGPUColor){0.3, 0.3, 0.3, 1.0};
  }

  /* Update background clear color based on skybox setting */
  if (state.settings.skybox_enabled) {
    /* Dark background to let skybox show */
    state.color_attachment.clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0};
  }
  else {
    /* Original gray background */
    state.color_attachment.clearValue = (WGPUColor){0.3, 0.3, 0.3, 1.0};
  }

  WGPUCommandEncoder command_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  if (state.settings.object_type == OBJECT_TYPE_WHALE
      && state.whale_scene.gltf_data) {
    /* GLTF render pass with depth testing */
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
      command_encoder, &state.render_pass_descriptor);

    /* Render skybox first (if enabled and available) */
    if (state.settings.skybox_enabled && state.skybox.pipeline
        && state.skybox.bind_group) {
      /* Update skybox uniforms */
      mat4 view_matrix, projection_matrix;
      glm_mat4_copy(state.camera_matrices.view, view_matrix);
      glm_mat4_copy(state.camera_matrices.projection, projection_matrix);

      float skybox_uniforms[32]; /* 2 mat4s: view + projection */
      memcpy(&skybox_uniforms[0], view_matrix, sizeof(mat4));
      memcpy(&skybox_uniforms[16], projection_matrix, sizeof(mat4));
      wgpuQueueWriteBuffer(wgpu_context->queue, state.skybox.uniform_buffer, 0,
                           skybox_uniforms, sizeof(skybox_uniforms));

      wgpuRenderPassEncoderSetPipeline(render_pass, state.skybox.pipeline);
      wgpuRenderPassEncoderSetBindGroup(render_pass, 0, state.skybox.bind_group,
                                        0, NULL);
      wgpuRenderPassEncoderSetVertexBuffer(
        render_pass, 0, state.skybox.vertex_buffer, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDraw(render_pass, 36, 1, 0,
                                0); /* 36 vertices for cube */
    }

    /* Render whale mesh */
    for (uint32_t i = 0; i < state.whale_scene.node_count; ++i) {
      gltf_node_t* node = &state.whale_scene.nodes[i];

      if (node->mesh_index < 0) {
        continue; /* Skip nodes without meshes */
      }

      gltf_mesh_t* mesh = &state.whale_scene.meshes[node->mesh_index];

      for (uint32_t j = 0; j < mesh->primitive_count; ++j) {
        gltf_primitive_t* prim = &mesh->primitives[j];

        wgpuRenderPassEncoderSetPipeline(render_pass, prim->pipeline);
        wgpuRenderPassEncoderSetBindGroup(
          render_pass, 0, state.camera_uniform.bind_group, 0, NULL);
        wgpuRenderPassEncoderSetBindGroup(
          render_pass, 1, state.general_uniforms.bind_group, 0, NULL);
        wgpuRenderPassEncoderSetBindGroup(render_pass, 2, node->bind_group, 0,
                                          NULL);

        /* Set skin bind group if node has skin */
        if (node->skin_index >= 0) {
          gltf_skin_t* skin = &state.whale_scene.skins[node->skin_index];
          wgpuRenderPassEncoderSetBindGroup(render_pass, 3, skin->bind_group, 0,
                                            NULL);
        }

        wgpuRenderPassEncoderSetVertexBuffer(
          render_pass, 0, prim->vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);

        if (prim->index_buffer.buffer) {
          wgpuRenderPassEncoderSetIndexBuffer(
            render_pass, prim->index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
            WGPU_WHOLE_SIZE);
          wgpuRenderPassEncoderDrawIndexed(render_pass, prim->index_count, 1, 0,
                                           0, 0);
        }
        else {
          wgpuRenderPassEncoderDraw(render_pass, prim->vertex_count, 1, 0, 0);
        }
      }
    }

    wgpuRenderPassEncoderEnd(render_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass)
  }
  else if (state.settings.object_type == OBJECT_TYPE_SKINNED_GRID) {
    /* Grid render pass with depth testing for skybox */
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
      command_encoder, &state.render_pass_descriptor);

    /* Render skybox first (if enabled and available) */
    if (state.settings.skybox_enabled && state.skybox.pipeline
        && state.skybox.bind_group) {
      /* Update skybox uniforms */
      mat4 view_matrix, projection_matrix;
      glm_mat4_copy(state.camera_matrices.view, view_matrix);
      glm_mat4_copy(state.camera_matrices.projection, projection_matrix);

      float skybox_uniforms[32]; /* 2 mat4s: view + projection */
      memcpy(&skybox_uniforms[0], view_matrix, sizeof(mat4));
      memcpy(&skybox_uniforms[16], projection_matrix, sizeof(mat4));
      wgpuQueueWriteBuffer(wgpu_context->queue, state.skybox.uniform_buffer, 0,
                           skybox_uniforms, sizeof(skybox_uniforms));

      wgpuRenderPassEncoderSetPipeline(render_pass, state.skybox.pipeline);
      wgpuRenderPassEncoderSetBindGroup(render_pass, 0, state.skybox.bind_group,
                                        0, NULL);
      wgpuRenderPassEncoderSetVertexBuffer(
        render_pass, 0, state.skybox.vertex_buffer, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDraw(render_pass, 36, 1, 0,
                                0); /* 36 vertices for cube */
    }

    /* Render skinned grid */
    wgpuRenderPassEncoderSetPipeline(render_pass, state.grid_pipeline);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                      state.camera_uniform.bind_group, 0, NULL);
    wgpuRenderPassEncoderSetBindGroup(
      render_pass, 1, state.general_uniforms.bind_group, 0, NULL);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 2, grid_bone_bind_group, 0,
                                      NULL);

    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 0, state.grid_buffers.positions.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 1, state.grid_buffers.joints.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 2, state.grid_buffers.weights.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      render_pass, state.grid_buffers.indices.buffer, WGPUIndexFormat_Uint16, 0,
      WGPU_WHOLE_SIZE);

    wgpuRenderPassEncoderDrawIndexed(
      render_pass, state.grid_buffers.index_count, 1, 0, 0, 0);

    wgpuRenderPassEncoderEnd(render_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass)
  }

  WGPUCommandBuffer command_buffer
    = wgpuCommandEncoderFinish(command_encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, command_encoder)

  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Event Handlers
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    resize_depth_texture(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Cleanup
 * -------------------------------------------------------------------------- */

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Shutdown ImGui overlay */
  imgui_overlay_shutdown();

  /* Free GLTF scene */
  if (state.whale_scene.nodes) {
    for (uint32_t i = 0; i < state.whale_scene.node_count; ++i) {
      WGPU_RELEASE_RESOURCE(Buffer, state.whale_scene.nodes[i].uniform_buffer)
      WGPU_RELEASE_RESOURCE(BindGroup, state.whale_scene.nodes[i].bind_group)
    }
    free(state.whale_scene.nodes);
  }

  if (state.whale_scene.meshes) {
    for (uint32_t i = 0; i < state.whale_scene.mesh_count; ++i) {
      gltf_mesh_t* mesh = &state.whale_scene.meshes[i];
      if (mesh->primitives) {
        for (uint32_t j = 0; j < mesh->primitive_count; ++j) {
          wgpu_destroy_buffer(&mesh->primitives[j].vertex_buffer);
          wgpu_destroy_buffer(&mesh->primitives[j].index_buffer);
          WGPU_RELEASE_RESOURCE(RenderPipeline, mesh->primitives[j].pipeline)
        }
        free(mesh->primitives);
      }
    }
    free(state.whale_scene.meshes);
  }

  if (state.whale_scene.skins) {
    for (uint32_t i = 0; i < state.whale_scene.skin_count; ++i) {
      gltf_skin_t* skin = &state.whale_scene.skins[i];
      free(skin->joints);
      free(skin->inverse_bind_matrices);
      WGPU_RELEASE_RESOURCE(Buffer, skin->joint_matrices_buffer)
      WGPU_RELEASE_RESOURCE(Buffer, skin->inverse_bind_matrices_buffer)
      WGPU_RELEASE_RESOURCE(BindGroup, skin->bind_group)
    }
    free(state.whale_scene.skins);
  }

  if (state.whale_scene.gltf_data) {
    cgltf_free(state.whale_scene.gltf_data);
  }

  if (state.whale_scene.glb_buffer) {
    free(state.whale_scene.glb_buffer);
  }

  /* Free grid resources */
  wgpu_destroy_buffer(&state.grid_buffers.positions);
  wgpu_destroy_buffer(&state.grid_buffers.joints);
  wgpu_destroy_buffer(&state.grid_buffers.weights);
  wgpu_destroy_buffer(&state.grid_buffers.indices);
  WGPU_RELEASE_RESOURCE(Buffer, grid_joint_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, grid_inverse_bind_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, grid_bone_bind_group)

  /* Free uniforms */
  WGPU_RELEASE_RESOURCE(Buffer, state.camera_uniform.buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.camera_uniform.bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.general_uniforms.buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.general_uniforms.bind_group)

  /* Free pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.grid_pipeline)

  /* Free bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.camera_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.general_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.node_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.skin_bind_group_layout)

  /* Free skybox resources */
  WGPU_RELEASE_RESOURCE(Buffer, state.skybox.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.skybox.uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.skybox.bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.skybox.bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.skybox.pipeline)
  WGPU_RELEASE_RESOURCE(Sampler, state.skybox.sampler)
  WGPU_RELEASE_RESOURCE(TextureView, state.skybox.view)
  WGPU_RELEASE_RESOURCE(Texture, state.skybox.texture)

  /* Free depth texture */
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_texture.view)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_texture.texture)

  /* Shutdown sokol fetch */
  sfetch_shutdown();
}

/* -------------------------------------------------------------------------- *
 * Main Entry Point
 * -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  wgpu_start(&(wgpu_desc_t){
    .title          = "WebGPU Skinned Mesh",
    .width          = 1280,
    .height         = 720,
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders - GLTF
 * -------------------------------------------------------------------------- */

/* clang-format off */
static const char* gltf_vertex_shader_wgsl = CODE(
  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) texcoord: vec2f,
    @location(3) joints: vec4u,
    @location(4) weights: vec4f,
  };

  struct VertexOutput {
    @builtin(position) Position: vec4f,
    @location(0) normal: vec3f,
    @location(1) joints: vec4f,
    @location(2) weights: vec4f,
  };

  struct CameraUniforms {
    proj_matrix: mat4x4f,
    view_matrix: mat4x4f,
    model_matrix: mat4x4f,
  };

  struct GeneralUniforms {
    render_mode: u32,
    skin_mode: u32,
  };

  struct NodeUniforms {
    world_matrix: mat4x4f,
  };

  @group(0) @binding(0) var<uniform> camera_uniforms: CameraUniforms;
  @group(1) @binding(0) var<uniform> general_uniforms: GeneralUniforms;
  @group(2) @binding(0) var<uniform> node_uniforms: NodeUniforms;
  @group(3) @binding(0) var<storage, read> joint_matrices: array<mat4x4f>;
  @group(3) @binding(1) var<storage, read> inverse_bind_matrices: array<mat4x4f>;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Compute joint_matrices * inverse_bind_matrices
    let joint0 = joint_matrices[input.joints[0]] * inverse_bind_matrices[input.joints[0]];
    let joint1 = joint_matrices[input.joints[1]] * inverse_bind_matrices[input.joints[1]];
    let joint2 = joint_matrices[input.joints[2]] * inverse_bind_matrices[input.joints[2]];
    let joint3 = joint_matrices[input.joints[3]] * inverse_bind_matrices[input.joints[3]];

    // Compute influence of joint based on weight
    let skin_matrix =
      joint0 * input.weights[0] +
      joint1 * input.weights[1] +
      joint2 * input.weights[2] +
      joint3 * input.weights[3];

    // Position of the vertex relative to our world
    let world_position = vec4f(input.position.x, input.position.y, input.position.z, 1.0);

    // Vertex position with model rotation, skinning, and the mesh's node transformation applied.
    let skinned_position = camera_uniforms.model_matrix * skin_matrix * node_uniforms.world_matrix * world_position;

    // Vertex position with only the model rotation applied.
    let rotated_position = camera_uniforms.model_matrix * world_position;

    // Determine which position to use based on whether skinMode is turned on or off.
    let transformed_position = select(
      rotated_position,
      skinned_position,
      general_uniforms.skin_mode == 0u
    );

    // Apply the camera and projection matrix transformations to our transformed position
    output.Position = camera_uniforms.proj_matrix * camera_uniforms.view_matrix * transformed_position;
    output.normal = input.normal;

    // Convert u32 joint data to f32s to prevent flat interpolation error
    output.joints = vec4f(f32(input.joints[0]), f32(input.joints[1]), f32(input.joints[2]), f32(input.joints[3]));
    output.weights = input.weights;

    return output;
  }
);
/* clang-format on */

/* clang-format off */
static const char* gltf_fragment_shader_wgsl = CODE(
  struct GeneralUniforms {
    render_mode: u32,
    skin_mode: u32,
  };

  @group(1) @binding(0) var<uniform> general_uniforms: GeneralUniforms;

  struct VertexOutput {
    @builtin(position) Position: vec4f,
    @location(0) normal: vec3f,
    @location(1) joints: vec4f,
    @location(2) weights: vec4f,
  };

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    switch general_uniforms.render_mode {
      case 1u: {
        return input.joints;
      }
      case 2u: {
        return input.weights;
      }
      default: {
        return vec4f(input.normal, 1.0);
      }
    }
  }
);
/* clang-format on */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders - Grid
 * -------------------------------------------------------------------------- */

/* clang-format off */
static const char* grid_vertex_shader_wgsl = CODE(
  struct VertexInput {
    @location(0) position: vec2f,
    @location(1) joints: vec4u,
    @location(2) weights: vec4f,
  };

  struct VertexOutput {
    @builtin(position) Position: vec4f,
    @location(0) color: vec4f,
  };

  struct CameraUniforms {
    proj_matrix: mat4x4f,
    view_matrix: mat4x4f,
    model_matrix: mat4x4f,
  };

  struct GeneralUniforms {
    render_mode: u32,
    skin_mode: u32,
  };

  @group(0) @binding(0) var<uniform> camera_uniforms: CameraUniforms;
  @group(1) @binding(0) var<uniform> general_uniforms: GeneralUniforms;
  @group(2) @binding(0) var<storage, read> joint_matrices: array<mat4x4f>;
  @group(2) @binding(1) var<storage, read> inverse_bind_matrices: array<mat4x4f>;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Compute joint_matrices * inverse_bind_matrices
    let joint0 = joint_matrices[input.joints[0]] * inverse_bind_matrices[input.joints[0]];
    let joint1 = joint_matrices[input.joints[1]] * inverse_bind_matrices[input.joints[1]];
    let joint2 = joint_matrices[input.joints[2]] * inverse_bind_matrices[input.joints[2]];
    let joint3 = joint_matrices[input.joints[3]] * inverse_bind_matrices[input.joints[3]];

    // Compute influence of joint based on weight
    let skin_matrix =
      joint0 * input.weights[0] +
      joint1 * input.weights[1] +
      joint2 * input.weights[2] +
      joint3 * input.weights[3];

    let world_position = vec4f(input.position.x, input.position.y, 0.0, 1.0);
    let transformed_position = skin_matrix * world_position;

    output.Position = camera_uniforms.proj_matrix * camera_uniforms.view_matrix * transformed_position;
    output.color = vec4f(1.0, 1.0, 1.0, 1.0);

    return output;
  }
);
/* clang-format on */

/* clang-format off */
static const char* grid_fragment_shader_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) Position: vec4f,
    @location(0) color: vec4f,
  };

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
  }
);
/* clang-format on */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders - Skybox
 * -------------------------------------------------------------------------- */

/* clang-format off */
static const char* skybox_vertex_shader_wgsl = CODE(
  struct Uniforms {
    view: mat4x4f,
    projection: mat4x4f,
  }

  @group(0) @binding(2) var<uniform> uniforms: Uniforms;

  struct VertexOutput {
    @builtin(position) Position: vec4f,
    @location(0) worldPosition: vec3f,
  }

  @vertex
  fn vertexMain(@location(0) position: vec3f) -> VertexOutput {
    var output: VertexOutput;
    var view = uniforms.view;
    /* Remove translation from view matrix to keep skybox centered */
    view[3][0] = 0.0;
    view[3][1] = 0.0;
    view[3][2] = 0.0;
    let pos = uniforms.projection * view * vec4f(position, 1.0);
    output.Position = pos.xyww; /* Set z = w for max depth */
    output.worldPosition = position;
    return output;
  }
);

static const char* skybox_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var skyboxTexture: texture_cube<f32>;
  @group(0) @binding(1) var skyboxSampler: sampler;

  @fragment
  fn fragmentMain(@location(0) worldPosition: vec3f) -> @location(0) vec4f {
    let color = textureSample(skyboxTexture, skyboxSampler, worldPosition).rgb;
    return vec4f(color, 1.0);
  }
);
/* clang-format on */
