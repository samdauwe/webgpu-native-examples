/* -------------------------------------------------------------------------- *
 * glTF 2.0 Model - Header
 *
 * Full C99 glTF 2.0 model loader following the official specification.
 * This module is API-agnostic (no WebGPU/Vulkan dependencies) and provides
 * parsed geometry, materials, textures, animations, skins, and morph targets
 * ready for consumption by any rendering backend.
 *
 * Reference:
 * https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
 *
 * Uses cgltf for parsing:
 * https://github.com/jkuhlmann/cgltf
 * -------------------------------------------------------------------------- */

#ifndef GLTF_MODEL_H
#define GLTF_MODEL_H

#include <cglm/cglm.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define GLTF_MODEL_MAX_NUM_JOINTS 128u
#define GLTF_MODEL_MAX_NAME_LENGTH 256u
#define GLTF_MODEL_MAX_URI_LENGTH 512u

/* -------------------------------------------------------------------------- *
 * Forward declarations
 * -------------------------------------------------------------------------- */

typedef struct gltf_node_t gltf_node_t;
typedef struct gltf_model_t gltf_model_t;

/* -------------------------------------------------------------------------- *
 * Enumerations
 * -------------------------------------------------------------------------- */

/**
 * @brief Alpha rendering modes per glTF 2.0 spec section 3.9.2.
 */
typedef enum gltf_alpha_mode_enum {
  GltfAlphaMode_Opaque = 0, /* Default: fully opaque, alpha ignored          */
  GltfAlphaMode_Mask   = 1, /* Binary: >= alphaCutoff opaque, else discarded  */
  GltfAlphaMode_Blend  = 2  /* Standard Porter-Duff "over" compositing        */
} gltf_alpha_mode_enum;

/**
 * @brief PBR workflow type.
 */
typedef enum gltf_pbr_workflow_enum {
  GltfPbrWorkflow_MetallicRoughness  = 0, /* Default per glTF 2.0 core spec  */
  GltfPbrWorkflow_SpecularGlossiness = 1  /* KHR_materials_pbrSpecGloss ext  */
} gltf_pbr_workflow_enum;

/**
 * @brief Animation channel target paths per glTF 2.0 spec section 3.8.2.
 */
typedef enum gltf_anim_path_enum {
  GltfAnimPath_Translation = 0,
  GltfAnimPath_Rotation    = 1,
  GltfAnimPath_Scale       = 2,
  GltfAnimPath_Weights     = 3
} gltf_anim_path_enum;

/**
 * @brief Animation interpolation types per glTF 2.0 spec section 3.8.3.
 */
typedef enum gltf_interpolation_enum {
  GltfInterpolation_Linear      = 0, /* Default. Slerp for rotations         */
  GltfInterpolation_Step        = 1, /* Constant until next keyframe          */
  GltfInterpolation_CubicSpline = 2  /* Hermite cubic spline interpolation    */
} gltf_interpolation_enum;

/**
 * @brief Primitive topology modes per glTF 2.0 spec section 3.7.2.1.
 */
typedef enum gltf_primitive_mode_enum {
  GltfPrimitiveMode_Points        = 0,
  GltfPrimitiveMode_Lines         = 1,
  GltfPrimitiveMode_LineLoop      = 2,
  GltfPrimitiveMode_LineStrip     = 3,
  GltfPrimitiveMode_Triangles     = 4, /* Default */
  GltfPrimitiveMode_TriangleStrip = 5,
  GltfPrimitiveMode_TriangleFan   = 6
} gltf_primitive_mode_enum;

/**
 * @brief Texture sampler filter modes per glTF 2.0 spec section 3.8.4.
 */
typedef enum gltf_filter_enum {
  GltfFilter_Undefined            = 0,
  GltfFilter_Nearest              = 9728,
  GltfFilter_Linear               = 9729,
  GltfFilter_NearestMipmapNearest = 9984,
  GltfFilter_LinearMipmapNearest  = 9985,
  GltfFilter_NearestMipmapLinear  = 9986,
  GltfFilter_LinearMipmapLinear   = 9987
} gltf_filter_enum;

/**
 * @brief Texture sampler wrapping modes per glTF 2.0 spec section 3.8.4.
 */
typedef enum gltf_wrap_enum {
  GltfWrap_ClampToEdge    = 33071,
  GltfWrap_MirroredRepeat = 33648,
  GltfWrap_Repeat         = 10497 /* Default */
} gltf_wrap_enum;

/* -------------------------------------------------------------------------- *
 * Bounding Box
 * -------------------------------------------------------------------------- */

typedef struct gltf_bounding_box_t {
  vec3 min;
  vec3 max;
  bool valid;
} gltf_bounding_box_t;

/* -------------------------------------------------------------------------- *
 * Texture Sampler
 * -------------------------------------------------------------------------- */

typedef struct gltf_texture_sampler_t {
  gltf_filter_enum mag_filter;
  gltf_filter_enum min_filter;
  gltf_wrap_enum wrap_s;
  gltf_wrap_enum wrap_t;
} gltf_texture_sampler_t;

/* -------------------------------------------------------------------------- *
 * Texture
 *
 * Holds CPU-side image data after loading. The rendering backend is
 * responsible for uploading to the GPU.
 * -------------------------------------------------------------------------- */

typedef struct gltf_texture_t {
  char name[GLTF_MODEL_MAX_NAME_LENGTH];
  uint8_t* data; /* RGBA pixel data (owned, free with gltf_model_destroy) */
  uint32_t width;
  uint32_t height;
  uint32_t channels; /* Number of channels (typically 4 = RGBA)              */
  uint32_t mip_levels; /* Number of mip levels (1 = no mipmaps) */
  gltf_texture_sampler_t sampler;
} gltf_texture_t;

/* -------------------------------------------------------------------------- *
 * Material
 *
 * PBR material properties following glTF 2.0 spec section 3.9.
 * Supports both metallic-roughness and specular-glossiness workflows.
 * -------------------------------------------------------------------------- */

typedef struct gltf_material_t {
  char name[GLTF_MODEL_MAX_NAME_LENGTH];

  /* --- PBR metallic-roughness workflow (default) --- */
  vec4 base_color_factor; /* Default: [1, 1, 1, 1]                  */
  float metallic_factor;  /* Default: 1.0                           */
  float roughness_factor; /* Default: 1.0                           */

  /* --- PBR specular-glossiness workflow (extension) --- */
  vec4 diffuse_factor;     /* Default: [1, 1, 1, 1]                  */
  vec3 specular_factor;    /* Default: [1, 1, 1]                     */
  float glossiness_factor; /* Default: 1.0                           */

  /* --- Common material properties --- */
  vec4 emissive_factor;            /* Default: [0, 0, 0, 0] (w unused)       */
  float emissive_strength;         /* KHR_materials_emissive_strength, def 1  */
  float normal_scale;              /* Scale for normal map XY, default 1.0   */
  float occlusion_strength;        /* Strength for AO, default 1.0           */
  gltf_alpha_mode_enum alpha_mode; /* Default: OPAQUE                        */
  float alpha_cutoff;              /* Default: 0.5 (used when alpha_mode=MASK)*/
  bool double_sided;               /* Default: false                          */
  bool unlit;                      /* KHR_materials_unlit extension           */

  /* --- KHR_materials_clearcoat extension --- */
  bool has_clearcoat;
  float clearcoat_factor;           /* Default: 0.0                           */
  float clearcoat_roughness_factor; /* Default: 0.0                           */

  /* --- KHR_materials_sheen extension --- */
  bool has_sheen;
  vec3 sheen_color_factor;      /* Default: [0, 0, 0]                     */
  float sheen_roughness_factor; /* Default: 0.0                           */

  /* --- Workflow selection --- */
  gltf_pbr_workflow_enum pbr_workflow;

  /* --- Texture indices (-1 = no texture) --- */
  int32_t base_color_tex_index;
  int32_t metallic_roughness_tex_index;
  int32_t normal_tex_index;
  int32_t occlusion_tex_index;
  int32_t emissive_tex_index;
  int32_t diffuse_tex_index; /* specular-glossiness workflow          */
  int32_t specular_glossiness_tex_index;

  /* --- Per-texture UV set selection --- */
  struct {
    uint8_t base_color; /* Default: 0 → TEXCOORD_0              */
    uint8_t metallic_roughness;
    uint8_t normal;
    uint8_t occlusion;
    uint8_t emissive;
    uint8_t diffuse;
    uint8_t specular_glossiness;
  } tex_coord_sets;

  /* --- Index of this material in the model's materials array --- */
  int32_t index;
} gltf_material_t;

/* -------------------------------------------------------------------------- *
 * Vertex
 *
 * Interleaved vertex layout supporting all glTF 2.0 standard attributes.
 * -------------------------------------------------------------------------- */

typedef struct gltf_vertex_t {
  vec3 position; /* POSITION    (required)                                   */
  vec3 normal;   /* NORMAL      (optional, default [0,0,0])                  */
  vec2 uv0;      /* TEXCOORD_0  (optional, default [0,0])                    */
  vec2 uv1;      /* TEXCOORD_1  (optional, default [0,0])                    */
  vec4 tangent;  /* TANGENT     (optional, xyzw, w=±1 handedness)            */
  vec4 color;    /* COLOR_0     (optional, default [1,1,1,1])                */
  uint32_t joint0[4]; /* JOINTS_0  (optional, indices into skin.joints)       */
  vec4 weight0; /* WEIGHTS_0   (optional, must sum ≈ 1.0)                   */
} gltf_vertex_t;

/* -------------------------------------------------------------------------- *
 * Primitive
 *
 * A drawable unit within a mesh, referencing a material and a range in the
 * shared vertex/index buffers.
 * -------------------------------------------------------------------------- */

typedef struct gltf_primitive_t {
  uint32_t first_index;   /* Start index in the model's index buffer         */
  uint32_t index_count;   /* Number of indices to draw                       */
  uint32_t vertex_count;  /* Number of vertices (for non-indexed draws)      */
  int32_t material_index; /* Index into model's materials (-1 = default)     */
  bool has_indices;       /* Whether this primitive is indexed               */
  gltf_primitive_mode_enum mode; /* Topology mode (default: TRIANGLES)        */
  gltf_bounding_box_t bb;        /* Object-space bounding box                 */
} gltf_primitive_t;

/* -------------------------------------------------------------------------- *
 * Mesh
 *
 * A collection of primitives. Each mesh can have morph target weights and
 * joint matrices for skinned rendering.
 * -------------------------------------------------------------------------- */

typedef struct gltf_mesh_t {
  gltf_primitive_t* primitives;
  uint32_t primitive_count;

  /* --- Bounding boxes --- */
  gltf_bounding_box_t bb;   /* Union of all primitive bounding boxes          */
  gltf_bounding_box_t aabb; /* World-space axis-aligned bounding box          */

  /* --- Skinning data (filled at runtime by animation update) --- */
  mat4 joint_matrices[GLTF_MODEL_MAX_NUM_JOINTS];
  uint32_t joint_count;

  /* --- Morph target weights --- */
  float* morph_weights; /* Current morph weights, count = target_count  */
  uint32_t morph_target_count;

  /* --- Local transform from node --- */
  mat4 matrix;

  /* --- Index for identification --- */
  uint32_t index;
} gltf_mesh_t;

/* -------------------------------------------------------------------------- *
 * Skin
 *
 * Skeletal animation data per glTF 2.0 spec section 3.7.4.
 * -------------------------------------------------------------------------- */

typedef struct gltf_skin_t {
  char name[GLTF_MODEL_MAX_NAME_LENGTH];
  gltf_node_t* skeleton_root;  /* Root joint node (optional)        */
  mat4* inverse_bind_matrices; /* Per-joint IBMs (owned)            */
  gltf_node_t** joints;        /* Array of joint node pointers      */
  uint32_t joint_count;
} gltf_skin_t;

/* -------------------------------------------------------------------------- *
 * Node
 *
 * Scene graph node per glTF 2.0 spec section 3.7.
 * Supports both matrix and TRS decomposition. When animated, TRS must be used.
 * -------------------------------------------------------------------------- */

struct gltf_node_t {
  char name[GLTF_MODEL_MAX_NAME_LENGTH];
  uint32_t index;

  /* --- Hierarchy --- */
  gltf_node_t* parent;
  gltf_node_t** children;
  uint32_t child_count;

  /* --- Transform (TRS components) --- */
  vec3 translation; /* Default: [0, 0, 0]               */
  versor rotation;  /* Default: [0, 0, 0, 1] (xyzw)     */
  vec3 scale;       /* Default: [1, 1, 1]               */
  mat4 matrix;      /* Raw 4x4 matrix if provided       */
  bool has_matrix;  /* true if node uses matrix form     */

  /* --- Cached world transform --- */
  mat4 cached_local_matrix;
  mat4 cached_world_matrix;
  bool use_cached_matrix;

  /* --- Associated resources --- */
  gltf_mesh_t* mesh;  /* NULL if no mesh attached          */
  gltf_skin_t* skin;  /* NULL if no skin attached          */
  int32_t skin_index; /* -1 if no skin                     */

  /* --- Bounding volumes --- */
  gltf_bounding_box_t bvh;
  gltf_bounding_box_t aabb;
};

/* -------------------------------------------------------------------------- *
 * Animation Sampler
 *
 * Holds keyframe timing and output data for an animation channel.
 * Supports LINEAR, STEP, and CUBICSPLINE interpolation.
 * -------------------------------------------------------------------------- */

typedef struct gltf_animation_sampler_t {
  gltf_interpolation_enum interpolation; /* Default: LINEAR                  */
  float* inputs;                         /* Keyframe timestamps (seconds)    */
  uint32_t input_count;

  /* Output data as vec4 for T/R/S channels */
  vec4* outputs_vec4;
  uint32_t output_vec4_count;

  /* Raw float outputs for cubic spline (in-tangent, value, out-tangent)     */
  float* outputs;
  uint32_t output_count;
} gltf_animation_sampler_t;

/* -------------------------------------------------------------------------- *
 * Animation Channel
 *
 * Links an animation sampler to a target node property.
 * -------------------------------------------------------------------------- */

typedef struct gltf_animation_channel_t {
  gltf_anim_path_enum path;
  gltf_node_t* node;      /* Target node                      */
  uint32_t sampler_index; /* Index into parent animation      */
} gltf_animation_channel_t;

/* -------------------------------------------------------------------------- *
 * Animation
 *
 * A set of channels and samplers defining a named animation clip.
 * -------------------------------------------------------------------------- */

typedef struct gltf_animation_t {
  char name[GLTF_MODEL_MAX_NAME_LENGTH];
  gltf_animation_sampler_t* samplers;
  uint32_t sampler_count;
  gltf_animation_channel_t* channels;
  uint32_t channel_count;
  float start_time; /* Earliest keyframe timestamp      */
  float end_time;   /* Latest keyframe timestamp        */
} gltf_animation_t;

/* -------------------------------------------------------------------------- *
 * Scene dimensions
 * -------------------------------------------------------------------------- */

typedef struct gltf_dimensions_t {
  vec3 min;
  vec3 max;
  vec3 size;
  vec3 center;
  float radius; /* Bounding sphere radius           */
} gltf_dimensions_t;

/* -------------------------------------------------------------------------- *
 * glTF Model
 *
 * Root container for all parsed glTF 2.0 data. Owns all allocated memory.
 * Call gltf_model_destroy() to free everything.
 * -------------------------------------------------------------------------- */

struct gltf_model_t {
  /* --- Geometry data (single shared buffer approach) --- */
  gltf_vertex_t* vertices;
  uint32_t vertex_count;
  uint32_t* indices;
  uint32_t index_count;

  /* --- Scene graph --- */
  gltf_node_t** nodes; /* Root scene nodes (tree roots)    */
  uint32_t node_count;
  gltf_node_t** linear_nodes; /* Flat list of ALL nodes           */
  uint32_t linear_node_count;

  /* --- Resources --- */
  gltf_texture_t* textures;
  uint32_t texture_count;
  gltf_material_t* materials;
  uint32_t material_count;
  gltf_skin_t* skins;
  uint32_t skin_count;
  gltf_animation_t* animations;
  uint32_t animation_count;

  /* --- Scene bounds --- */
  gltf_dimensions_t dimensions;

  /* --- File path for resolving relative URIs --- */
  char file_path[GLTF_MODEL_MAX_URI_LENGTH];

  /* --- Extensions used --- */
  char** extensions;
  uint32_t extension_count;
};

/* -------------------------------------------------------------------------- *
 * Public API
 * -------------------------------------------------------------------------- */

/**
 * @brief Load a glTF 2.0 model from a file (.gltf or .glb).
 *
 * Parses the file, loads all buffers and images, builds the scene graph,
 * and prepares interleaved vertex/index data. Textures are loaded as CPU-side
 * RGBA pixel data.
 *
 * @param model      Pointer to an uninitialized model struct.
 * @param filename   Path to the .gltf or .glb file.
 * @param scale      Global scale factor applied to all positions.
 * @return true on success, false on error (model is zeroed on failure).
 */
bool gltf_model_load_from_file(gltf_model_t* model, const char* filename,
                               float scale);

/**
 * @brief Load a glTF 2.0 model from memory (e.g., embedded GLB data).
 *
 * @param model      Pointer to an uninitialized model struct.
 * @param data       Pointer to the GLB/glTF data in memory.
 * @param size       Size of the data in bytes.
 * @param base_path  Base path for resolving relative URIs (can be NULL).
 * @param scale      Global scale factor applied to all positions.
 * @return true on success, false on error.
 */
bool gltf_model_load_from_memory(gltf_model_t* model, const void* data,
                                 size_t size, const char* base_path,
                                 float scale);

/**
 * @brief Free all memory owned by the model.
 *
 * After calling this, the model struct is zeroed and safe to reuse.
 *
 * @param model  Pointer to the model to destroy.
 */
void gltf_model_destroy(gltf_model_t* model);

/**
 * @brief Update an animation by index at the given time.
 *
 * Applies keyframe interpolation to all animated nodes, then updates the
 * entire node hierarchy (transforms, skinning matrices).
 *
 * @param model           Pointer to the loaded model.
 * @param animation_index Index of the animation to update.
 * @param time            Current time in seconds (will be wrapped to the
 *                        animation's [start, end] range).
 */
void gltf_model_update_animation(gltf_model_t* model, uint32_t animation_index,
                                 float time);

/**
 * @brief Recompute the scene's bounding dimensions.
 *
 * Traverses all nodes and updates the model's dimensions struct.
 *
 * @param model  Pointer to the loaded model.
 */
void gltf_model_get_scene_dimensions(gltf_model_t* model);

/**
 * @brief Get the local transform matrix for a node.
 *
 * Computes T * R * S or uses the raw matrix, depending on the node's
 * transform type.
 *
 * @param node    Pointer to the node.
 * @param dest    Output 4x4 matrix.
 */
void gltf_node_get_local_matrix(const gltf_node_t* node, mat4 dest);

/**
 * @brief Get the world transform matrix for a node.
 *
 * Walks up the parent chain multiplying local matrices.
 *
 * @param node    Pointer to the node.
 * @param dest    Output 4x4 matrix.
 */
void gltf_node_get_world_matrix(const gltf_node_t* node, mat4 dest);

/**
 * @brief Update a node's cached matrices and recursively update children.
 *
 * Recomputes skin joint matrices if the node has a skinned mesh.
 *
 * @param node  Pointer to the node to update.
 */
void gltf_node_update(gltf_node_t* node);

/**
 * @brief Find a node by its index in the model's linear node list.
 *
 * @param model  Pointer to the loaded model.
 * @param index  The node index to search for.
 * @return Pointer to the node, or NULL if not found.
 */
gltf_node_t* gltf_model_find_node(gltf_model_t* model, uint32_t index);

/**
 * @brief Compute AABB from an OBB transformed by a matrix.
 *
 * @param bb    The object-space bounding box.
 * @param m     The transformation matrix.
 * @param dest  Output axis-aligned bounding box.
 */
void gltf_bounding_box_get_aabb(const gltf_bounding_box_t* bb, mat4 m,
                                gltf_bounding_box_t* dest);

#endif /* GLTF_MODEL_H */
