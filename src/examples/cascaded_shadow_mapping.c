#include "webgpu/wgpu_common.h"

#include "webgpu/imgui_overlay.h"

#include "core/camera.h"
#include "core/gltf_model.h"

#include <cglm/cglm.h>
#include <cglm/clipspace/ortho_rh_zo.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

/* cimgui */
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <math.h>
#include <stdbool.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Cascaded shadow mapping
 *
 * Cascaded shadow mapping for directional light sources. The camera frustum is
 * split into multiple sub-frustums, each receiving its own full-resolution
 * shadow map stored as a layer of a depth texture array. The fragment shader
 * selects the appropriate cascade layer based on the view-space depth. A
 * configurable lambda controls the logarithmic/uniform split ratio. PCF
 * (Percentage Closer Filtering) and cascade coloring overlays are togglable
 * through the ImGui-based UI.
 *
 * Ref:
 *  https://github.com/SaschaWillems/Vulkan/tree/master/examples
 *  /shadowmappingcascade
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */
static const char* depth_pass_shader_wgsl;
static const char* scene_shader_wgsl;
static const char* debug_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define SHADOW_MAP_DIM (2048)
#define SHADOW_MAP_CASCADE_COUNT (4)

#define NUM_TREE_POSITIONS (5)

/* Per-draw stride in the dynamic push-const buffer.          *
 * Must be ≥ sizeof(push_const_ubo_t) and a multiple of       *
 * minUniformBufferOffsetAlignment (256 on all Dawn targets). */
#define PUSH_CONST_STRIDE (256u)

/* Total pre-baked draw slots:                              *
 * 4 cascades × 6 draws (terrain+5 trees) = 24 (depth pass) *
 * + 6 draws (scene pass) + 1 (debug pass) = 31             */
#define PUSH_CONST_NUM_DRAWS (31u)

/* Maximum materials per model (terrain has 1, tree has 2 leaf+bark) */
#define MAX_MATERIALS_PER_MODEL (8)

/* -------------------------------------------------------------------------- *
 * Uniform buffer structures
 * -------------------------------------------------------------------------- */

/* Vertex UBO: projection, view, model, lightDir */
typedef struct {
  mat4 projection;
  mat4 view;
  mat4 model;
  vec4 light_dir; /* xyz = dir, w = pad */
} uniform_data_vertex_t;

/* Fragment UBO: cascade splits, inverse view, lightDir, flags */
typedef struct {
  vec4 cascade_splits; /* x,y,z,w = 4 split depths */
  mat4 inverse_view;
  vec4 light_dir; /* xyz = dir, w = pad */
  int32_t color_cascades;
  int32_t _pad[3];
} uniform_data_fragment_t;

/* Per-object UBO passed to depth and scene shaders for position offset and
   cascade selection. Replaces Vulkan push constants. */
typedef struct {
  vec4 position; /* xyz = offset, w = 0 */
  uint32_t cascade_index;
  uint32_t _pad[3];
} push_const_ubo_t;

/* -------------------------------------------------------------------------- *
 * Cascade data
 * -------------------------------------------------------------------------- */

typedef struct {
  float split_depth;
  mat4 view_proj_matrix;
} cascade_t;

/* Per-material GPU texture + bind group */
typedef struct {
  WGPUTexture texture;
  WGPUTextureView view;
  WGPUBindGroup bind_group;
} model_mat_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  camera_t camera;

  /* Animation timer */
  float timer;
  float animation_speed;
  uint64_t last_frame_time;

  /* Models */
  gltf_model_t terrain;
  gltf_model_t tree;
  bool terrain_loaded;
  bool tree_loaded;

  struct {
    WGPUBuffer vertex_buffer;
    WGPUBuffer index_buffer;
  } terrain_buffers, tree_buffers;

  /* Per-material GPU textures (one entry per GLTF material in each model) */
  model_mat_t terrain_mats[MAX_MATERIALS_PER_MODEL];
  model_mat_t tree_mats[MAX_MATERIALS_PER_MODEL];

  /* Fallback 1×1 white texture used for materials without a base colour map */
  WGPUTexture default_texture;
  WGPUTextureView default_texture_view;
  WGPUSampler texture_sampler;

  /* Cascaded shadow map */
  struct {
    WGPUTexture depth_texture; /* 2D array, 4 layers */
    WGPUTextureView full_view; /* all layers (for sampling) */
    WGPUTextureView cascade_views[SHADOW_MAP_CASCADE_COUNT]; /* per-layer */
    WGPUSampler comparison_sampler; /* shadow comparison sampler */
  } shadow_map;

  cascade_t cascades[SHADOW_MAP_CASCADE_COUNT];

  /* Main pass depth texture */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth;

  /* Uniform buffers */
  struct {
    WGPUBuffer vertex;
    WGPUBuffer fragment;
    WGPUBuffer cascade_vp; /* 4 × mat4 */
    WGPUBuffer push_const; /* per-draw UBO (push const replacement) */
  } uniform_buffers;

  uniform_data_vertex_t ubo_vertex;
  uniform_data_fragment_t ubo_fragment;

  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout
      scene; /* set 0: vertex/fragment UBOs + shadow map + cascade VP */
    WGPUBindGroupLayout depth;      /* set 0: cascade VP + push const UBO */
    WGPUBindGroupLayout push_const; /* set 1 for scene: push const UBO */
    WGPUBindGroupLayout material;   /* set 2 for scene: colorMap + sampler */
  } bind_group_layouts;

  /* Pipeline layouts */
  struct {
    WGPUPipelineLayout scene;
    WGPUPipelineLayout depth;
    WGPUPipelineLayout debug;
  } pipeline_layouts;

  /* Bind groups */
  struct {
    WGPUBindGroup scene;      /* scene pass set 0 */
    WGPUBindGroup depth;      /* depth pass set 0 */
    WGPUBindGroup push_const; /* shared set 1 (or set for depth) */
    WGPUBindGroup debug;      /* same as scene for debug viz */
  } bind_groups;

  /* Pipelines */
  struct {
    WGPURenderPipeline depth_pass;       /* depth-only shadow generation */
    WGPURenderPipeline scene_shadow;     /* scene, no PCF */
    WGPURenderPipeline scene_shadow_pcf; /* scene with PCF */
    WGPURenderPipeline debug;            /* shadow map debug visualization */
  } pipelines;

  /* Shadow render pass (depth-only) */
  struct {
    WGPURenderPassDepthStencilAttachment depth_att;
    WGPURenderPassDescriptor descriptor;
  } shadow_pass;

  /* Main render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI / settings */
  struct {
    float cascade_split_lambda;
    bool color_cascades;
    bool display_depth_map;
    int32_t depth_map_cascade_index;
    bool filter_pcf;
  } settings;

  /* Camera clip planes */
  float z_near;
  float z_far;

  /* Light */
  vec3 light_pos;

  /* Tree positions (Vulkan Y negated for WebGPU) */
  vec4 tree_positions[NUM_TREE_POSITIONS];

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.2f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Undefined,
    .stencilStoreOp    = WGPUStoreOp_Undefined,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .shadow_pass = {
    .depth_att = {
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Undefined,
      .stencilStoreOp    = WGPUStoreOp_Undefined,
    },
    .descriptor = {
      .colorAttachmentCount   = 0,
      .colorAttachments       = NULL,
      .depthStencilAttachment = &state.shadow_pass.depth_att,
    },
  },
  /* Defaults */
  .z_near = 0.5f,
  .z_far  = 48.0f,
  .animation_speed = 0.00625f, /* Vulkan base 0.25 * example 0.025 */
  .timer           = 0.2f,
  .settings = {
    .cascade_split_lambda    = 0.95f,
    .color_cascades          = false,
    .display_depth_map       = false,
    .depth_map_cascade_index = 0,
    .filter_pcf              = false,
  },
};

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_models(void)
{
  gltf_model_desc_t desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
  };

  state.terrain_loaded = gltf_model_load_from_file_ext(
    &state.terrain, "assets/models/terrain_gridlines.gltf", 1.0f, &desc);

  state.tree_loaded = gltf_model_load_from_file_ext(
    &state.tree, "assets/models/oaktree.gltf", 1.0f, &desc);
}

static void create_model_buffers_for(struct wgpu_context_t* wgpu_context,
                                     gltf_model_t* model, WGPUBuffer* vb_out,
                                     WGPUBuffer* ib_out)
{
  WGPUDevice device = wgpu_context->device;
  size_t vb_size    = model->vertex_count * sizeof(gltf_vertex_t);
  size_t ib_size    = model->index_count * sizeof(uint32_t);

  *vb_out = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Model - Vertex buffer"),
              .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
              .size  = vb_size,
              .mappedAtCreation = true,
            });
  memcpy(wgpuBufferGetMappedRange(*vb_out, 0, vb_size), model->vertices,
         vb_size);
  wgpuBufferUnmap(*vb_out);

  if (model->index_count > 0 && model->indices) {
    *ib_out = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Model - Index buffer"),
                .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                .size  = ib_size,
                .mappedAtCreation = true,
              });
    memcpy(wgpuBufferGetMappedRange(*ib_out, 0, ib_size), model->indices,
           ib_size);
    wgpuBufferUnmap(*ib_out);
  }
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  if (state.terrain_loaded) {
    create_model_buffers_for(wgpu_context, &state.terrain,
                             &state.terrain_buffers.vertex_buffer,
                             &state.terrain_buffers.index_buffer);
  }
  if (state.tree_loaded) {
    create_model_buffers_for(wgpu_context, &state.tree,
                             &state.tree_buffers.vertex_buffer,
                             &state.tree_buffers.index_buffer);
  }
}

static void draw_model(WGPURenderPassEncoder pass, gltf_model_t* model,
                       WGPUBuffer vb, WGPUBuffer ib,
                       model_mat_t mat_array[MAX_MATERIALS_PER_MODEL])
{
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vb, 0, WGPU_WHOLE_SIZE);
  if (ib) {
    wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
  }

  for (uint32_t n = 0; n < model->linear_node_count; n++) {
    gltf_node_t* node = model->linear_nodes[n];
    if (!node->mesh)
      continue;
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; p++) {
      gltf_primitive_t* prim = &mesh->primitives[p];

      /* Bind per-material texture at set 2 (scene pass only) */
      if (mat_array) {
        uint32_t mat_idx
          = (prim->material_index >= 0
             && (uint32_t)prim->material_index < MAX_MATERIALS_PER_MODEL) ?
              (uint32_t)prim->material_index :
              0u;
        if (mat_array[mat_idx].bind_group) {
          wgpuRenderPassEncoderSetBindGroup(
            pass, 2, mat_array[mat_idx].bind_group, 0, NULL);
        }
      }

      if (prim->has_indices && prim->index_count > 0) {
        wgpuRenderPassEncoderDrawIndexed(pass, prim->index_count, 1,
                                         prim->first_index, 0, 0);
      }
      else if (prim->vertex_count > 0) {
        wgpuRenderPassEncoderDraw(pass, prim->vertex_count, 1, 0, 0);
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Shadow map texture (layered depth array)
 * -------------------------------------------------------------------------- */

static void init_shadow_map(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Create layered depth texture (4 layers) */
  state.shadow_map.depth_texture = wgpuDeviceCreateTexture(
    device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Cascade Shadow Map"),
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
      .dimension = WGPUTextureDimension_2D,
      .size      = {SHADOW_MAP_DIM, SHADOW_MAP_DIM, SHADOW_MAP_CASCADE_COUNT},
      .format    = WGPUTextureFormat_Depth32Float,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });

  /* Full array view for sampling in the fragment shader */
  state.shadow_map.full_view
    = wgpuTextureCreateView(state.shadow_map.depth_texture,
                            &(WGPUTextureViewDescriptor){
                              .label         = STRVIEW("Shadow Map Full View"),
                              .format        = WGPUTextureFormat_Depth32Float,
                              .dimension     = WGPUTextureViewDimension_2DArray,
                              .baseMipLevel  = 0,
                              .mipLevelCount = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = SHADOW_MAP_CASCADE_COUNT,
                              .aspect          = WGPUTextureAspect_DepthOnly,
                            });

  /* Per-cascade views for rendering (single layer each) */
  for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
    state.shadow_map.cascade_views[i]
      = wgpuTextureCreateView(state.shadow_map.depth_texture,
                              &(WGPUTextureViewDescriptor){
                                .label         = STRVIEW("Cascade View"),
                                .format        = WGPUTextureFormat_Depth32Float,
                                .dimension     = WGPUTextureViewDimension_2D,
                                .baseMipLevel  = 0,
                                .mipLevelCount = 1,
                                .baseArrayLayer  = i,
                                .arrayLayerCount = 1,
                                .aspect          = WGPUTextureAspect_DepthOnly,
                              });
  }

  /* Comparison sampler for hardware shadow testing (gives free 2×2 bilinear
   * PCF when the GPU linearly interpolates comparison results). */
  state.shadow_map.comparison_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Shadow Comparison - Sampler"),
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .compare       = WGPUCompareFunction_Less,
              .maxAnisotropy = 1,
            });
}

/* -------------------------------------------------------------------------- *
 * Main pass depth texture
 * -------------------------------------------------------------------------- */

static void init_depth_texture(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  if (state.depth.view) {
    WGPU_RELEASE_RESOURCE(TextureView, state.depth.view);
  }
  if (state.depth.texture) {
    WGPU_RELEASE_RESOURCE(Texture, state.depth.texture);
  }

  state.depth.texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label         = STRVIEW("Main Depth"),
              .usage         = WGPUTextureUsage_RenderAttachment,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {wgpu_context->width, wgpu_context->height, 1},
              .format        = WGPUTextureFormat_Depth32Float,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });

  state.depth.view = wgpuTextureCreateView(
    state.depth.texture, &(WGPUTextureViewDescriptor){
                           .label           = STRVIEW("Main Depth View"),
                           .format          = WGPUTextureFormat_Depth32Float,
                           .dimension       = WGPUTextureViewDimension_2D,
                           .baseMipLevel    = 0,
                           .mipLevelCount   = 1,
                           .baseArrayLayer  = 0,
                           .arrayLayerCount = 1,
                           .aspect          = WGPUTextureAspect_DepthOnly,
                         });
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.uniform_buffers.vertex = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Vertex UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(uniform_data_vertex_t),
            });

  state.uniform_buffers.fragment = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Fragment UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(uniform_data_fragment_t),
            });

  state.uniform_buffers.cascade_vp = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Cascade VP UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(mat4) * SHADOW_MAP_CASCADE_COUNT,
            });

  /* Large dynamic push-const buffer: PUSH_CONST_NUM_DRAWS entries,     *
   * each PUSH_CONST_STRIDE bytes apart (256-byte minOffsetAlignment).  */
  state.uniform_buffers.push_const = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Push Const UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = PUSH_CONST_NUM_DRAWS * PUSH_CONST_STRIDE,
            });
}

/* -------------------------------------------------------------------------- *
 * Light animation
 * -------------------------------------------------------------------------- */

static void update_light(void)
{
  float angle  = glm_rad(state.timer * 360.0f);
  float radius = 20.0f;

  /* Vulkan Y is down; WebGPU Y is up → negate Y */
  state.light_pos[0] = cosf(angle) * radius;
  state.light_pos[1] = radius; /* Vulkan: -radius → WebGPU: +radius */
  state.light_pos[2] = sinf(angle) * radius;
}

/* -------------------------------------------------------------------------- *
 * Cascade update algorithm (logarithmic/uniform hybrid split)
 * -------------------------------------------------------------------------- */

static void update_cascades(void)
{
  float cascade_splits[SHADOW_MAP_CASCADE_COUNT];

  float near_clip  = state.z_near;
  float far_clip   = state.z_far;
  float clip_range = far_clip - near_clip;
  float min_z      = near_clip;
  float max_z      = near_clip + clip_range;
  float range      = max_z - min_z;
  float ratio      = max_z / min_z;

  float lambda = state.settings.cascade_split_lambda;

  for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
    float p           = (float)(i + 1) / (float)SHADOW_MAP_CASCADE_COUNT;
    float log_val     = min_z * powf(ratio, p);
    float uni_val     = min_z + range * p;
    float d           = lambda * (log_val - uni_val) + uni_val;
    cascade_splits[i] = (d - near_clip) / clip_range;
  }

  float last_split_dist = 0.0f;
  for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
    float split_dist = cascade_splits[i];

    /* 8 NDC frustum corners: near plane z=0, far plane z=1 (WebGPU) */
    vec3 frustum_corners[8] = {
      {-1.0f, 1.0f, 0.0f},  {1.0f, 1.0f, 0.0f},   {1.0f, -1.0f, 0.0f},
      {-1.0f, -1.0f, 0.0f}, {-1.0f, 1.0f, 1.0f},  {1.0f, 1.0f, 1.0f},
      {1.0f, -1.0f, 1.0f},  {-1.0f, -1.0f, 1.0f},
    };

    /* Transform from NDC to world space */
    mat4 vp, inv_cam;
    glm_mat4_mul(state.camera.matrices.perspective, state.camera.matrices.view,
                 vp);
    glm_mat4_inv(vp, inv_cam);

    for (uint32_t j = 0; j < 8; j++) {
      vec4 pt = {frustum_corners[j][0], frustum_corners[j][1],
                 frustum_corners[j][2], 1.0f};
      vec4 world;
      glm_mat4_mulv(inv_cam, pt, world);
      glm_vec3_divs(world, world[3], frustum_corners[j]);
    }

    /* Slice the frustum between lastSplitDist and splitDist */
    for (uint32_t j = 0; j < 4; j++) {
      vec3 dist;
      glm_vec3_sub(frustum_corners[j + 4], frustum_corners[j], dist);
      vec3 far_pt, near_pt;
      glm_vec3_scale(dist, split_dist, far_pt);
      glm_vec3_add(frustum_corners[j], far_pt, frustum_corners[j + 4]);
      glm_vec3_scale(dist, last_split_dist, near_pt);
      glm_vec3_add(frustum_corners[j], near_pt, frustum_corners[j]);
    }

    /* Compute frustum center */
    vec3 center = {0.0f, 0.0f, 0.0f};
    for (uint32_t j = 0; j < 8; j++) {
      glm_vec3_add(center, frustum_corners[j], center);
    }
    glm_vec3_divs(center, 8.0f, center);

    /* Compute bounding sphere radius */
    float radius = 0.0f;
    for (uint32_t j = 0; j < 8; j++) {
      float d2 = glm_vec3_distance(frustum_corners[j], center);
      if (d2 > radius)
        radius = d2;
    }
    radius = ceilf(radius * 16.0f) / 16.0f;

    vec3 max_extents = {radius, radius, radius};
    vec3 min_extents;
    glm_vec3_negate_to(max_extents, min_extents);

    /* Light direction and matrices */
    vec3 light_dir;
    glm_vec3_negate_to(state.light_pos, light_dir);
    glm_vec3_normalize(light_dir);

    vec3 eye;
    vec3 scaled_dir;
    glm_vec3_scale(light_dir, -min_extents[2], scaled_dir);
    glm_vec3_sub(center, scaled_dir, eye);

    mat4 light_view, light_ortho;
    vec3 up = {0.0f, 1.0f, 0.0f};
    glm_lookat(eye, center, up, light_view);
    glm_ortho_rh_zo(min_extents[0], max_extents[0], min_extents[1],
                    max_extents[1], 0.0f, max_extents[2] - min_extents[2],
                    light_ortho);

    /* ---- Texel snapping: stabilize shadow map when camera moves ---- *
     * Snap the light-space origin to shadow texel boundaries so that   *
     * sub-texel camera movement doesn't shift the whole shadow map.    */
    {
      mat4 shadow_matrix;
      glm_mat4_mul(light_ortho, light_view, shadow_matrix);

      vec4 shadow_origin = {0.0f, 0.0f, 0.0f, 1.0f};
      vec4 shadow_origin_proj;
      glm_mat4_mulv(shadow_matrix, shadow_origin, shadow_origin_proj);

      float half_dim = (float)SHADOW_MAP_DIM * 0.5f;
      shadow_origin_proj[0] *= half_dim;
      shadow_origin_proj[1] *= half_dim;

      float rounded_x = roundf(shadow_origin_proj[0]);
      float rounded_y = roundf(shadow_origin_proj[1]);

      float offset_x = (rounded_x - shadow_origin_proj[0]) / half_dim;
      float offset_y = (rounded_y - shadow_origin_proj[1]) / half_dim;

      light_ortho[3][0] += offset_x;
      light_ortho[3][1] += offset_y;
    }

    /* Store cascade data */
    state.cascades[i].split_depth
      = (near_clip + split_dist * clip_range) * -1.0f;
    glm_mat4_mul(light_ortho, light_view, state.cascades[i].view_proj_matrix);

    last_split_dist = cascade_splits[i];
  }
}

/* -------------------------------------------------------------------------- *
 * Update uniform buffers
 * -------------------------------------------------------------------------- */

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUQueue queue = wgpu_context->queue;

  /* Cascade view-projection matrices */
  mat4 cascade_vp[SHADOW_MAP_CASCADE_COUNT];
  for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
    glm_mat4_copy(state.cascades[i].view_proj_matrix, cascade_vp[i]);
  }
  wgpuQueueWriteBuffer(queue, state.uniform_buffers.cascade_vp, 0, cascade_vp,
                       sizeof(cascade_vp));

  /* Vertex UBO */
  glm_mat4_copy(state.camera.matrices.perspective, state.ubo_vertex.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo_vertex.view);
  glm_mat4_identity(state.ubo_vertex.model);

  vec3 neg_light;
  glm_vec3_negate_to(state.light_pos, neg_light);
  glm_vec3_normalize(neg_light);
  glm_vec3_copy(neg_light, state.ubo_vertex.light_dir);
  state.ubo_vertex.light_dir[3] = 0.0f;

  wgpuQueueWriteBuffer(queue, state.uniform_buffers.vertex, 0,
                       &state.ubo_vertex, sizeof(state.ubo_vertex));

  /* Fragment UBO */
  for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
    state.ubo_fragment.cascade_splits[i] = state.cascades[i].split_depth;
  }
  mat4 inv_view;
  glm_mat4_inv(state.camera.matrices.view, inv_view);
  glm_mat4_copy(inv_view, state.ubo_fragment.inverse_view);
  glm_vec3_copy(neg_light, state.ubo_fragment.light_dir);
  state.ubo_fragment.light_dir[3]   = 0.0f;
  state.ubo_fragment.color_cascades = state.settings.color_cascades ? 1 : 0;

  wgpuQueueWriteBuffer(queue, state.uniform_buffers.fragment, 0,
                       &state.ubo_fragment, sizeof(state.ubo_fragment));
}

/* -------------------------------------------------------------------------- *
 * Write push constant UBO (per-draw update)
 * -------------------------------------------------------------------------- */

/* Write one push-const slot at byte offset slot*PUSH_CONST_STRIDE.     */
static void write_push_const_slot(WGPUQueue queue, uint32_t slot,
                                  const vec4 position, uint32_t cascade_index)
{
  push_const_ubo_t pc;
  glm_vec4_copy((float*)position, pc.position);
  pc.cascade_index = cascade_index;
  pc._pad[0] = pc._pad[1] = pc._pad[2] = 0;

  wgpuQueueWriteBuffer(queue, state.uniform_buffers.push_const,
                       (uint64_t)slot * PUSH_CONST_STRIDE, &pc, sizeof(pc));
}

/*
 * Pre-upload all per-draw push-const data before recording any render commands.
 * Slot layout:
 *   0 .. 4*6-1  = depth pass cascades (c*6 + 0..5 for terrain, tree0..4)
 *   24..29      = scene pass (terrain, tree0..4)
 *   30          = debug pass
 */
static void upload_all_push_const_data(WGPUQueue queue)
{
  vec4 zero = {0.0f, 0.0f, 0.0f, 0.0f};

  /* Depth pass – one batch per cascade */
  for (uint32_t c = 0; c < SHADOW_MAP_CASCADE_COUNT; c++) {
    uint32_t base = c * (1u + NUM_TREE_POSITIONS);
    write_push_const_slot(queue, base + 0, zero, c);
    for (uint32_t t = 0; t < NUM_TREE_POSITIONS; t++) {
      write_push_const_slot(queue, base + 1 + t, state.tree_positions[t], c);
    }
  }

  /* Scene pass */
  uint32_t scene_base = SHADOW_MAP_CASCADE_COUNT * (1u + NUM_TREE_POSITIONS);
  write_push_const_slot(queue, scene_base + 0, zero, 0);
  for (uint32_t t = 0; t < NUM_TREE_POSITIONS; t++) {
    write_push_const_slot(queue, scene_base + 1 + t, state.tree_positions[t],
                          0);
  }

  /* Debug pass */
  uint32_t debug_slot = scene_base + 1 + NUM_TREE_POSITIONS;
  write_push_const_slot(queue, debug_slot, zero,
                        (uint32_t)state.settings.depth_map_cascade_index);
}

/* -------------------------------------------------------------------------- *
 * Material textures
 * -------------------------------------------------------------------------- */

/* Create GPU textures & bind groups for all materials in a GLTF model.
 * Results are stored in mat_array (indexed by GLTF material index).
 * mat_array must have at least MAX_MATERIALS_PER_MODEL entries. */
static void
create_model_material_textures(struct wgpu_context_t* wgpu_context,
                               gltf_model_t* model,
                               model_mat_t mat_array[MAX_MATERIALS_PER_MODEL])
{
  WGPUDevice device = wgpu_context->device;

  uint32_t count = model->material_count < MAX_MATERIALS_PER_MODEL ?
                     model->material_count :
                     MAX_MATERIALS_PER_MODEL;

  for (uint32_t i = 0; i < count; i++) {
    gltf_material_t* mat = &model->materials[i];

    WGPUTextureView tex_view = state.default_texture_view;

    if (mat->base_color_tex_index >= 0
        && (uint32_t)mat->base_color_tex_index < model->texture_count) {
      const gltf_texture_t* tex = &model->textures[mat->base_color_tex_index];
      if (tex->data && tex->width > 0 && tex->height > 0) {
        mat_array[i].texture = wgpuDeviceCreateTexture(
          device,
          &(WGPUTextureDescriptor){
            .label = STRVIEW("Material Texture"),
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
            .dimension     = WGPUTextureDimension_2D,
            .size          = {tex->width, tex->height, 1},
            .format        = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1,
            .sampleCount   = 1,
          });

        wgpuQueueWriteTexture(wgpu_context->queue,
                              &(WGPUTexelCopyTextureInfo){
                                .texture  = mat_array[i].texture,
                                .mipLevel = 0,
                                .aspect   = WGPUTextureAspect_All,
                              },
                              tex->data, (size_t)4 * tex->width * tex->height,
                              &(WGPUTexelCopyBufferLayout){
                                .offset       = 0,
                                .bytesPerRow  = 4 * tex->width,
                                .rowsPerImage = tex->height,
                              },
                              &(WGPUExtent3D){tex->width, tex->height, 1});

        mat_array[i].view = wgpuTextureCreateView(
          mat_array[i].texture, &(WGPUTextureViewDescriptor){
                                  .label         = STRVIEW("Material Tex View"),
                                  .format        = WGPUTextureFormat_RGBA8Unorm,
                                  .dimension     = WGPUTextureViewDimension_2D,
                                  .baseMipLevel  = 0,
                                  .mipLevelCount = 1,
                                  .baseArrayLayer  = 0,
                                  .arrayLayerCount = 1,
                                });

        tex_view = mat_array[i].view;
      }
    }

    WGPUBindGroupEntry bg_entries[2] = {
      [0] = {.binding = 0, .textureView = tex_view},
      [1] = {.binding = 1, .sampler = state.texture_sampler},
    };
    mat_array[i].bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Material BG"),
                .layout     = state.bind_group_layouts.material,
                .entryCount = ARRAY_SIZE(bg_entries),
                .entries    = bg_entries,
              });
  }
}

static void create_material_textures(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Shared linear sampler */
  state.texture_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Texture Sampler"),
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_Repeat,
              .addressModeW  = WGPUAddressMode_Repeat,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
              .lodMinClamp   = 0.0f,
              .lodMaxClamp   = 1.0f,
              .maxAnisotropy = 1,
            });

  /* 1×1 white default texture */
  uint8_t white[4] = {255, 255, 255, 255};
  state.default_texture
    = wgpuDeviceCreateTexture(device, &(WGPUTextureDescriptor){
                                        .label = STRVIEW("Default White"),
                                        .usage = WGPUTextureUsage_TextureBinding
                                                 | WGPUTextureUsage_CopyDst,
                                        .dimension = WGPUTextureDimension_2D,
                                        .size      = {1, 1, 1},
                                        .format = WGPUTextureFormat_RGBA8Unorm,
                                        .mipLevelCount = 1,
                                        .sampleCount   = 1,
                                      });
  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture = state.default_texture,
                          .aspect  = WGPUTextureAspect_All,
                        },
                        white, 4,
                        &(WGPUTexelCopyBufferLayout){
                          .bytesPerRow  = 4,
                          .rowsPerImage = 1,
                        },
                        &(WGPUExtent3D){1, 1, 1});
  state.default_texture_view = wgpuTextureCreateView(
    state.default_texture, &(WGPUTextureViewDescriptor){
                             .label           = STRVIEW("Default White View"),
                             .format          = WGPUTextureFormat_RGBA8Unorm,
                             .dimension       = WGPUTextureViewDimension_2D,
                             .baseMipLevel    = 0,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = 0,
                             .arrayLayerCount = 1,
                           });

  if (state.terrain_loaded) {
    create_model_material_textures(wgpu_context, &state.terrain,
                                   state.terrain_mats);
  }
  if (state.tree_loaded) {
    create_model_material_textures(wgpu_context, &state.tree, state.tree_mats);
  }
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene layout (set 0) – 5 bindings:
     0: vertex UBO (vert)
     1: shadow map 2D depth array (frag)
     2: fragment UBO (frag)
     3: cascade VP matrices (vert+frag)
     4: shadow comparison sampler (frag) */
  {
    WGPUBindGroupLayoutEntry entries[5] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = {.type = WGPUBufferBindingType_Uniform,
                       .minBindingSize = sizeof(uniform_data_vertex_t)},
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Depth,
                       .viewDimension = WGPUTextureViewDimension_2DArray,
                       .multisampled  = false},
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {.type = WGPUBufferBindingType_Uniform,
                       .minBindingSize = sizeof(uniform_data_fragment_t)},
      },
      [3] = {
        .binding    = 3,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer     = {.type = WGPUBufferBindingType_Uniform,
                       .minBindingSize = sizeof(mat4) * SHADOW_MAP_CASCADE_COUNT},
      },
      [4] = {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Comparison},
      },
    };
    state.bind_group_layouts.scene = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Scene BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Depth pass layout (set 0) – 1 binding:
     0: cascade VP UBO */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = {.type = WGPUBufferBindingType_Uniform,
                       .minBindingSize = sizeof(mat4) * SHADOW_MAP_CASCADE_COUNT},
      },
    };
    state.bind_group_layouts.depth = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Depth BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Push const layout (used as set 1 in both scene and depth) – 1 binding:
     0: push const UBO (vert+frag) – dynamic offset */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                       .hasDynamicOffset = true,
                       .minBindingSize   = sizeof(push_const_ubo_t)},
      },
    };
    state.bind_group_layouts.push_const = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("PushConst BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Material layout (set 2 for scene pass) – 2 bindings:
     0: texture_2d (base colour map), 1: sampler */
  {
    WGPUBindGroupLayoutEntry entries[2] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D,
                       .multisampled  = false},
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Filtering},
      },
    };
    state.bind_group_layouts.material = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Material BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Pipeline layouts
 * -------------------------------------------------------------------------- */

static void init_pipeline_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene: set 0 = scene, set 1 = push_const, set 2 = material */
  {
    WGPUBindGroupLayout layouts[3] = {
      state.bind_group_layouts.scene,
      state.bind_group_layouts.push_const,
      state.bind_group_layouts.material,
    };
    state.pipeline_layouts.scene = wgpuDeviceCreatePipelineLayout(
      device, &(WGPUPipelineLayoutDescriptor){
                .label                = STRVIEW("Scene PipelineLayout"),
                .bindGroupLayoutCount = ARRAY_SIZE(layouts),
                .bindGroupLayouts     = layouts,
              });
  }

  /* Depth: set 0 = depth, set 1 = push_const, set 2 = material (alpha test) */
  {
    WGPUBindGroupLayout layouts[3] = {
      state.bind_group_layouts.depth,
      state.bind_group_layouts.push_const,
      state.bind_group_layouts.material,
    };
    state.pipeline_layouts.depth = wgpuDeviceCreatePipelineLayout(
      device, &(WGPUPipelineLayoutDescriptor){
                .label                = STRVIEW("Depth PipelineLayout"),
                .bindGroupLayoutCount = ARRAY_SIZE(layouts),
                .bindGroupLayouts     = layouts,
              });
  }

  /* Debug: set 0 = scene, set 1 = push_const (no material group needed) */
  {
    WGPUBindGroupLayout layouts[2] = {
      state.bind_group_layouts.scene,
      state.bind_group_layouts.push_const,
    };
    state.pipeline_layouts.debug = wgpuDeviceCreatePipelineLayout(
      device, &(WGPUPipelineLayoutDescriptor){
                .label                = STRVIEW("Debug PipelineLayout"),
                .bindGroupLayoutCount = ARRAY_SIZE(layouts),
                .bindGroupLayouts     = layouts,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene (set 0): vertex UBO, shadow map array, fragment UBO, cascade VP,
   *                 shadow comparison sampler */
  {
    WGPUBindGroupEntry entries[5] = {
      [0] = {.binding = 0,
             .buffer  = state.uniform_buffers.vertex,
             .size    = sizeof(uniform_data_vertex_t)},
      [1] = {.binding = 1, .textureView = state.shadow_map.full_view},
      [2] = {.binding = 2,
             .buffer  = state.uniform_buffers.fragment,
             .size    = sizeof(uniform_data_fragment_t)},
      [3] = {.binding = 3,
             .buffer  = state.uniform_buffers.cascade_vp,
             .size    = sizeof(mat4) * SHADOW_MAP_CASCADE_COUNT},
      [4] = {.binding = 4, .sampler = state.shadow_map.comparison_sampler},
    };
    state.bind_groups.scene = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Scene BG"),
                .layout     = state.bind_group_layouts.scene,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
    /* Debug uses the same bind group */
    state.bind_groups.debug = state.bind_groups.scene;
  }

  /* Depth (set 0): cascade VP only */
  {
    WGPUBindGroupEntry entries[1] = {
      [0] = {.binding = 0,
             .buffer  = state.uniform_buffers.cascade_vp,
             .size    = sizeof(mat4) * SHADOW_MAP_CASCADE_COUNT},
    };
    state.bind_groups.depth = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Depth BG"),
                .layout     = state.bind_group_layouts.depth,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Push const (set 1): shared for both passes */
  {
    WGPUBindGroupEntry entries[1] = {
      [0] = {.binding = 0,
             .buffer  = state.uniform_buffers.push_const,
             .offset  = 0,
             .size    = sizeof(push_const_ubo_t)},
    };
    state.bind_groups.push_const = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("PushConst BG"),
                .layout     = state.bind_group_layouts.push_const,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ===== Depth pass pipeline ===== */
  {
    /* Vertex layout: position + uv (for alpha test in frag) */
    WGPUVertexAttribute depth_attrs[] = {
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x2,
       .offset         = offsetof(gltf_vertex_t, uv0)},
    };
    WGPUVertexBufferLayout depth_vb = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(depth_attrs),
      .attributes     = depth_attrs,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, depth_pass_shader_wgsl);

    WGPUDepthStencilState depth_stencil = {
      .format              = WGPUTextureFormat_Depth32Float,
      .depthWriteEnabled   = WGPUOptionalBool_True,
      .depthCompare        = WGPUCompareFunction_LessEqual,
      .depthBias           = 0,
      .depthBiasSlopeScale = 0.0f,
      .depthBiasClamp      = 0.0f,
    };

    state.pipelines.depth_pass = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Depth Pass Pipeline"),
        .layout = state.pipeline_layouts.depth,
        .vertex = (WGPUVertexState){
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &depth_vb,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Back,
        },
        .depthStencil = &depth_stencil,
        .multisample  = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = shader,
          .entryPoint  = STRVIEW("fs_main"),
          .targetCount = 0,
          .targets     = NULL,
        },
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
  }

  /* ===== Scene pipelines (with/without PCF) ===== */
  {
    WGPUVertexAttribute scene_attrs[] = {
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x2,
       .offset         = offsetof(gltf_vertex_t, uv0)},
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, color)},
      {.shaderLocation = 3,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal)},
    };
    WGPUVertexBufferLayout scene_vb = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(scene_attrs),
      .attributes     = scene_attrs,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, scene_shader_wgsl);

    WGPUBlendState blend              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = WGPUTextureFormat_Depth32Float,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
    };

    /* No PCF */
    WGPUConstantEntry no_pcf_const = {
      .key   = STRVIEW("enablePCF"),
      .value = 0.0,
    };

    state.pipelines.scene_shadow = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Scene Shadow Pipeline"),
        .layout = state.pipeline_layouts.scene,
        .vertex = (WGPUVertexState){
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &scene_vb,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_None,
        },
        .depthStencil = &depth_stencil,
        .multisample  = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module        = shader,
          .entryPoint    = STRVIEW("fs_main"),
          .targetCount   = 1,
          .targets       = &color_target,
          .constantCount = 1,
          .constants     = &no_pcf_const,
        },
      });

    /* With PCF */
    WGPUConstantEntry pcf_const = {
      .key   = STRVIEW("enablePCF"),
      .value = 1.0,
    };

    state.pipelines.scene_shadow_pcf = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Scene Shadow PCF Pipeline"),
        .layout = state.pipeline_layouts.scene,
        .vertex = (WGPUVertexState){
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &scene_vb,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_None,
        },
        .depthStencil = &depth_stencil,
        .multisample  = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module        = shader,
          .entryPoint    = STRVIEW("fs_main"),
          .targetCount   = 1,
          .targets       = &color_target,
          .constantCount = 1,
          .constants     = &pcf_const,
        },
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
  }

  /* ===== Debug shadow map pipeline ===== */
  {
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, debug_shader_wgsl);

    WGPUBlendState blend              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = WGPUTextureFormat_Depth32Float,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
    };

    state.pipelines.debug = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Debug Pipeline"),
        .layout = state.pipeline_layouts.debug,
        .vertex = (WGPUVertexState){
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 0,
          .buffers     = NULL,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Back,
        },
        .depthStencil = &depth_stencil,
        .multisample  = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = shader,
          .entryPoint  = STRVIEW("fs_main"),
          .targetCount = 1,
          .targets     = &color_target,
        },
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
  }
}

/* -------------------------------------------------------------------------- *
 * Render scene (terrain + trees) for a given render pass
 * -------------------------------------------------------------------------- */

static void render_scene(WGPURenderPassEncoder pass, uint32_t base_slot,
                         bool use_materials)
{
  /* Terrain (slot base_slot+0) */
  if (state.terrain_loaded) {
    uint32_t offset = (base_slot + 0) * PUSH_CONST_STRIDE;
    wgpuRenderPassEncoderSetBindGroup(pass, 1, state.bind_groups.push_const, 1,
                                      &offset);
    draw_model(pass, &state.terrain, state.terrain_buffers.vertex_buffer,
               state.terrain_buffers.index_buffer,
               use_materials ? state.terrain_mats : NULL);
  }

  /* Trees (slots base_slot+1 .. base_slot+NUM_TREE_POSITIONS) */
  if (state.tree_loaded) {
    for (uint32_t t = 0; t < NUM_TREE_POSITIONS; t++) {
      uint32_t offset = (base_slot + 1 + t) * PUSH_CONST_STRIDE;
      wgpuRenderPassEncoderSetBindGroup(pass, 1, state.bind_groups.push_const,
                                        1, &offset);
      draw_model(pass, &state.tree, state.tree_buffers.vertex_buffer,
                 state.tree_buffers.index_buffer,
                 use_materials ? state.tree_mats : NULL);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    if (igSliderFloat("Split lambda", &state.settings.cascade_split_lambda,
                      0.1f, 1.0f, "%.2f", 0)) {
      update_cascades();
    }
    igCheckbox("Color cascades", &state.settings.color_cascades);
    igCheckbox("Display depth map", &state.settings.display_depth_map);
    if (state.settings.display_depth_map) {
      igSliderInt("Cascade", &state.settings.depth_map_cascade_index, 0,
                  SHADOW_MAP_CASCADE_COUNT - 1, "%d");
    }
    igCheckbox("PCF filtering", &state.settings.filter_pcf);
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input event callback
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  /* Skip camera input when ImGui captures the mouse */
  if (!imgui_overlay_want_capture_mouse()) {
    camera_on_input_event(&state.camera, input_event);
  }

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context)
    return EXIT_FAILURE;

  stm_setup();

  /* Camera setup: FirstPerson type */
  camera_init(&state.camera);
  state.camera.type           = CameraType_FirstPerson;
  state.camera.movement_speed = 2.5f;
  camera_set_position(&state.camera,
                      (vec3)VKY_TO_WGPU_VEC3(-0.12f, -1.14f, -2.25f));
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-17.0f, 7.0f, 0.0f));
  camera_set_perspective(&state.camera, 45.0f,
                         (float)wgpu_context->width
                           / (float)wgpu_context->height,
                         state.z_near, state.z_far);
  /* Convert perspective to WebGPU [0,1] depth range (glm_perspective uses
   * OpenGL [-1,1] by default and CGLM_FORCE_DEPTH_ZERO_TO_ONE is not set). */
  projection_matrix_convert_clip_space_near_z(
    &state.camera.matrices.perspective, ClipSpaceNearZ_Zero,
    ClipSpaceNearZ_NegativeOne);

  /* Tree positions (Vulkan Y negated for WebGPU) */
  glm_vec4_copy((vec4){0.0f, 0.0f, 0.0f, 0.0f}, state.tree_positions[0]);
  glm_vec4_copy((vec4){1.25f, -0.25f, 1.25f, 0.0f}, state.tree_positions[1]);
  glm_vec4_copy((vec4){-1.25f, 0.2f, 1.25f, 0.0f}, state.tree_positions[2]);
  glm_vec4_copy((vec4){1.25f, -0.1f, -1.25f, 0.0f}, state.tree_positions[3]);
  glm_vec4_copy((vec4){-1.25f, 0.25f, -1.25f, 0.0f}, state.tree_positions[4]);

  /* Load assets */
  load_models();
  create_model_buffers(wgpu_context);

  /* Textures */
  init_shadow_map(wgpu_context);
  init_depth_texture(wgpu_context);

  /* Initial light & cascades */
  update_light();
  update_cascades();

  /* Buffers */
  init_uniform_buffers(wgpu_context);

  /* GPU setup */
  init_bind_group_layouts(wgpu_context);
  create_material_textures(wgpu_context); /* after BGL, before pipelines */
  init_pipeline_layouts(wgpu_context);
  init_bind_groups(wgpu_context);
  init_pipelines(wgpu_context);

  /* GUI */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized)
    return EXIT_FAILURE;

  /* Timing */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Animate */
  state.timer += delta_time * state.animation_speed;
  if (state.timer > 1.0f)
    state.timer -= 1.0f;

  /* Camera & light */
  camera_update(&state.camera, delta_time);
  update_light();
  update_cascades();
  update_uniform_buffers(wgpu_context);

  /* Pre-upload all per-draw push-const data BEFORE encoding render commands */
  upload_all_push_const_data(wgpu_context->queue);

  /* ImGui new frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* Begin command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("Frame Cmd Enc"),
                          });

  /* ===== Pass 1: Shadow map generation (one pass per cascade) ===== */
  for (uint32_t c = 0; c < SHADOW_MAP_CASCADE_COUNT; c++) {
    state.shadow_pass.depth_att.view = state.shadow_map.cascade_views[c];

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.shadow_pass.descriptor);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)SHADOW_MAP_DIM,
                                     (float)SHADOW_MAP_DIM, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, SHADOW_MAP_DIM,
                                        SHADOW_MAP_DIM);

    wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.depth_pass);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_groups.depth, 0,
                                      NULL);

    /* base_slot = c * (1 + NUM_TREE_POSITIONS) */
    render_scene(pass, c * (1u + NUM_TREE_POSITIONS), true);

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass);
  }

  /* ===== Pass 2: Scene rendering ===== */
  {
    state.color_attachment.view         = wgpu_context->swapchain_view;
    state.depth_stencil_attachment.view = state.depth.view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)wgpu_context->width,
                                     (float)wgpu_context->height, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, wgpu_context->width,
                                        wgpu_context->height);

    /* Debug shadow map OR scene – mutually exclusive */
    if (state.settings.display_depth_map) {
      wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.debug);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_groups.debug, 0,
                                        NULL);

      /* debug slot is last pre-uploaded slot */
      uint32_t debug_slot = SHADOW_MAP_CASCADE_COUNT * (1u + NUM_TREE_POSITIONS)
                            + 1u + NUM_TREE_POSITIONS;
      uint32_t offset = debug_slot * PUSH_CONST_STRIDE;
      wgpuRenderPassEncoderSetBindGroup(pass, 1, state.bind_groups.push_const,
                                        1, &offset);

      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    }
    else {
      /* Scene with shadows */
      WGPURenderPipeline pipeline = state.settings.filter_pcf ?
                                      state.pipelines.scene_shadow_pcf :
                                      state.pipelines.scene_shadow;
      wgpuRenderPassEncoderSetPipeline(pass, pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_groups.scene, 0,
                                        NULL);

      /* scene base_slot */
      uint32_t scene_base
        = SHADOW_MAP_CASCADE_COUNT * (1u + NUM_TREE_POSITIONS);
      render_scene(pass, scene_base, true);
    }

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass);
  }

  /* Submit */
  WGPUCommandBuffer cmd_buf
    = wgpuCommandEncoderFinish(cmd_enc, &(WGPUCommandBufferDescriptor){
                                          .label = STRVIEW("Frame CmdBuf"),
                                        });
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buf);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buf);

  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Shadow map */
  for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
    WGPU_RELEASE_RESOURCE(TextureView, state.shadow_map.cascade_views[i]);
  }
  WGPU_RELEASE_RESOURCE(TextureView, state.shadow_map.full_view);
  WGPU_RELEASE_RESOURCE(Texture, state.shadow_map.depth_texture);
  WGPU_RELEASE_RESOURCE(Sampler, state.shadow_map.comparison_sampler);

  /* Depth */
  WGPU_RELEASE_RESOURCE(TextureView, state.depth.view);
  WGPU_RELEASE_RESOURCE(Texture, state.depth.texture);

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.vertex);
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.fragment);
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.cascade_vp);
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.push_const);

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.depth_pass);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.scene_shadow);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.scene_shadow_pcf);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.debug);

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.scene);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.depth);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.debug);

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.scene);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.depth);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.push_const);

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.scene);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.depth);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.push_const);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.material);

  /* Material textures */
  for (uint32_t i = 0; i < MAX_MATERIALS_PER_MODEL; i++) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.terrain_mats[i].bind_group);
    WGPU_RELEASE_RESOURCE(TextureView, state.terrain_mats[i].view);
    WGPU_RELEASE_RESOURCE(Texture, state.terrain_mats[i].texture);
    WGPU_RELEASE_RESOURCE(BindGroup, state.tree_mats[i].bind_group);
    WGPU_RELEASE_RESOURCE(TextureView, state.tree_mats[i].view);
    WGPU_RELEASE_RESOURCE(Texture, state.tree_mats[i].texture);
  }
  WGPU_RELEASE_RESOURCE(Sampler, state.texture_sampler);
  WGPU_RELEASE_RESOURCE(TextureView, state.default_texture_view);
  WGPU_RELEASE_RESOURCE(Texture, state.default_texture);

  /* Model buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain_buffers.vertex_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain_buffers.index_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.tree_buffers.vertex_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.tree_buffers.index_buffer);

  /* Models */
  gltf_model_destroy(&state.terrain);
  gltf_model_destroy(&state.tree);
}

/* -------------------------------------------------------------------------- *
 * Main
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Cascaded Shadow Mapping",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off

/* Depth pass shader: vertex transforms + alpha-test fragment (for tree leaves) */
static const char* depth_pass_shader_wgsl = CODE(
  const CASCADE_COUNT = 4u;

  struct CascadeVP {
    matrices : array<mat4x4f, 4>,
  }

  struct PushConst {
    position     : vec4f,
    cascadeIndex : u32,
  }

  @group(0) @binding(0) var<uniform> cascadeVP : CascadeVP;
  @group(1) @binding(0) var<uniform> pc : PushConst;
  @group(2) @binding(0) var colorMap    : texture_2d<f32>;
  @group(2) @binding(1) var colorSampler : sampler;

  struct VSOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  }

  @vertex
  fn vs_main(@location(0) inPos : vec3f,
             @location(1) inUV  : vec2f) -> VSOutput {
    var out : VSOutput;
    let pos = inPos + pc.position.xyz;
    out.position = cascadeVP.matrices[pc.cascadeIndex] * vec4f(pos, 1.0);
    out.uv = inUV;
    return out;
  }

  @fragment
  fn fs_main(in : VSOutput) {
    let alpha = textureSample(colorMap, colorSampler, in.uv).a;
    if (alpha < 0.5) {
      discard;
    }
  }
);

/* Scene shader: shadow-mapped rendering with cascaded lookup */
static const char* scene_shader_wgsl = CODE(
  const CASCADE_COUNT = 4u;
  const ambient : f32 = 0.3;

  struct VertexUBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
    lightDir   : vec4f,
  }

  struct FragmentUBO {
    cascadeSplits : vec4f,
    inverseView   : mat4x4f,
    lightDir      : vec4f,
    colorCascades : i32,
  }

  struct CascadeVP {
    matrices : array<mat4x4f, 4>,
  }

  struct PushConst {
    position     : vec4f,
    cascadeIndex : u32,
  }

  @group(0) @binding(0) var<uniform> ubo : VertexUBO;
  @group(0) @binding(1) var shadowMap : texture_depth_2d_array;
  @group(0) @binding(2) var<uniform> uboFrag : FragmentUBO;
  @group(0) @binding(3) var<uniform> cascadeVP : CascadeVP;
  @group(0) @binding(4) var shadowSampler : sampler_comparison;
  @group(1) @binding(0) var<uniform> pc : PushConst;
  @group(2) @binding(0) var colorMap    : texture_2d<f32>;
  @group(2) @binding(1) var colorSampler : sampler;

  override enablePCF : u32 = 0u;

  struct VSOutput {
    @builtin(position) position : vec4f,
    @location(0) normal   : vec3f,
    @location(1) color    : vec4f,
    @location(2) viewPos  : vec3f,
    @location(3) worldPos : vec3f,
    @location(4) uv       : vec2f,
  }

  /* WebGPU NDC bias matrix: maps xy from [-1,1] to [0,1], Y-flip */
  const biasMat = mat4x4f(
    0.5,  0.0, 0.0, 0.0,
    0.0, -0.5, 0.0, 0.0,
    0.0,  0.0, 1.0, 0.0,
    0.5,  0.5, 0.0, 1.0
  );

  @vertex
  fn vs_main(
    @location(0) inPos    : vec3f,
    @location(1) inUV     : vec2f,
    @location(2) inColor  : vec4f,
    @location(3) inNormal : vec3f,
  ) -> VSOutput {
    var out : VSOutput;
    out.color  = inColor;
    out.normal = inNormal;
    out.uv     = inUV;
    let pos = inPos + pc.position.xyz;
    out.worldPos = pos;
    out.viewPos  = (ubo.view * vec4f(pos, 1.0)).xyz;
    out.position = ubo.projection * ubo.view * ubo.model * vec4f(pos, 1.0);
    return out;
  }

  /* Shadow test via comparison sampler — returns a value in [0,1]:
   * 0.0 = fully in shadow, 1.0 = fully lit, with free 2×2 bilinear PCF. */
  fn textureProj(shadowCoord : vec4f, offset : vec2f,
                 cascadeIndex : u32) -> f32 {
    let bias : f32 = 0.005;
    if (shadowCoord.z >= 0.0 && shadowCoord.z <= 1.0 && shadowCoord.w > 0.0) {
      let uv = shadowCoord.xy + offset;
      let visibility = textureSampleCompareLevel(
        shadowMap, shadowSampler, uv, i32(cascadeIndex),
        shadowCoord.z - bias);
      return ambient + (1.0 - ambient) * visibility;
    }
    return 1.0;
  }

  fn filterPCF(sc : vec4f, cascadeIndex : u32) -> f32 {
    let texelSize : f32 = 1.0 / 2048.0;
    let scale : f32 = 1.5;
    var shadowFactor : f32 = 0.0;
    var count : i32 = 0;

    for (var x : i32 = -1; x <= 1; x++) {
      for (var y : i32 = -1; y <= 1; y++) {
        let off = vec2f(f32(x), f32(y)) * texelSize * scale;
        shadowFactor += textureProj(sc, off, cascadeIndex);
        count++;
      }
    }
    return shadowFactor / f32(count);
  }

  @fragment
  fn fs_main(in : VSOutput) -> @location(0) vec4f {
    /* Sample base colour texture; multiply by vertex colour (usually white) */
    let texColor = textureSample(colorMap, colorSampler, in.uv);
    /* Alpha cutout for leaves / transparent geometry */
    if (texColor.a < 0.5) {
      discard;
    }
    let color = texColor * in.color;

    /* Determine cascade index from view-space Z */
    var cascadeIndex : u32 = 0u;
    for (var i : u32 = 0u; i < CASCADE_COUNT - 1u; i++) {
      if (in.viewPos.z < uboFrag.cascadeSplits[i]) {
        cascadeIndex = i + 1u;
      }
    }

    /* Shadow coord via bias * cascade VP * worldPos */
    let shadowCoord = (biasMat * cascadeVP.matrices[cascadeIndex])
                      * vec4f(in.worldPos, 1.0);
    let sc = shadowCoord / shadowCoord.w;

    var shadow : f32;
    if (enablePCF == 1u) {
      shadow = filterPCF(sc, cascadeIndex);
    } else {
      shadow = textureProj(sc, vec2f(0.0), cascadeIndex);
    }

    /* Directional lighting */
    let N = normalize(in.normal);
    let L = normalize(-uboFrag.lightDir.xyz);
    let diffuse = max(dot(N, L), ambient);
    var outColor = vec3f(max(diffuse * color.rgb, vec3f(0.0)));
    outColor *= shadow;

    /* Cascade coloring */
    if (uboFrag.colorCascades == 1) {
      switch (cascadeIndex) {
        case 0u: { outColor *= vec3f(1.0, 0.25, 0.25); }
        case 1u: { outColor *= vec3f(0.25, 1.0, 0.25); }
        case 2u: { outColor *= vec3f(0.25, 0.25, 1.0); }
        case 3u: { outColor *= vec3f(1.0, 1.0, 0.25); }
        default: {}
      }
    }

    return vec4f(outColor, color.a);
  }
);

/* Debug shader: full-screen quad showing a single cascade depth */
static const char* debug_shader_wgsl = CODE(
  struct PushConst {
    position     : vec4f,
    cascadeIndex : u32,
  }

  @group(0) @binding(1) var shadowMap : texture_depth_2d_array;
  @group(1) @binding(0) var<uniform> pc : PushConst;

  struct VSOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  }

  @vertex
  fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VSOutput {
    var out : VSOutput;
    let u = f32((vertexIndex << 1u) & 2u);
    let v = f32(vertexIndex & 2u);
    out.uv = vec2f(u, 1.0 - v);
    out.position = vec4f(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VSOutput) -> @location(0) vec4f {
    let texSize = 2048;
    let tc = clamp(
      vec2i(in.uv * vec2f(f32(texSize))),
      vec2i(0), vec2i(texSize - 1));
    let depth = textureLoad(shadowMap, tc, i32(pc.cascadeIndex), 0);
    return vec4f(vec3f(depth), 1.0);
  }
);

// clang-format on
