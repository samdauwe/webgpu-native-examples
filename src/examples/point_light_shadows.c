#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include "core/camera.h"
#include "core/gltf_model.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <stdbool.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * Point light shadows using a dynamic shadow cube map
 *
 * Ported from Sascha Willems' Vulkan example "shadowmappingomni"
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/shadowmappingomni
 *
 * Renders omni-directional shadows from a point light by rendering the scene
 * from the light's perspective into each face of a cube map (storing distance
 * from the light). The main scene pass samples this cube map to determine
 * shadowing.
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Forward-declared shader strings (defined at end of file)
 * -------------------------------------------------------------------------- */

static const char* offscreen_shader_wgsl;
static const char* scene_shader_wgsl;
static const char* cubemap_display_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

/* Shadow cube map resolution per face */
#define SHADOW_MAP_DIM (1024)

/* Depth range for shadow maps */
#define Z_NEAR (0.1f)
#define Z_FAR (1024.0f)

/* Shadow comparison epsilon / opacity */
#define SHADOW_EPSILON (0.15f)
#define SHADOW_OPACITY (0.5f)

/* Model file paths */
static const char* scene_model_path = "assets/models/shadowscene_fire.gltf";
static const char* cube_model_path  = "assets/models/cube.gltf";

/* Number of cube map faces */
#define NUM_CUBE_FACES 6

/* -------------------------------------------------------------------------- *
 * Uniform data structures (must match WGSL layout)
 * -------------------------------------------------------------------------- */

/*
 * Shared UBO structure for both offscreen and scene passes.
 * Layout matches a 4x4 mat4 + mat4 + mat4 + vec4 = 208 bytes.
 * Aligned to 16 bytes per std140.
 */
typedef struct {
  mat4 projection;
  mat4 view;
  mat4 model;
  vec4 light_pos;
} uniform_data_t;

/*
 * Per-face view matrix for the offscreen cube map pass.
 * Passed via a separate uniform buffer (WebGPU equivalent of push constants).
 */
typedef struct {
  mat4 face_view;
} face_view_uniform_t;

/* -------------------------------------------------------------------------- *
 * Global state
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Models */
  gltf_model_t scene_model;
  gltf_model_t cube_model;
  bool scene_loaded;
  bool cube_loaded;

  struct {
    WGPUBuffer vertex_buffer;
    WGPUBuffer index_buffer;
  } scene_buffers, cube_buffers;

  /* Shadow cube map */
  struct {
    WGPUTexture texture;
    WGPUTextureView cube_view; /* Cube view for scene sampling */
    WGPUTextureView face_views[NUM_CUBE_FACES]; /* Per-face 2D views */
    WGPUSampler sampler;
  } shadow_cube_map;

  /* Offscreen depth attachment (shared by all faces) */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } offscreen_depth;

  /* Main pass depth texture */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth;

  /* Uniform buffers */
  struct {
    WGPUBuffer scene;
    WGPUBuffer offscreen;
    WGPUBuffer face_views[NUM_CUBE_FACES]; /* Per-face view matrix */
  } uniform_buffers;

  /* Uniform data */
  uniform_data_t ubo_scene;
  uniform_data_t ubo_offscreen;
  face_view_uniform_t face_view_data[NUM_CUBE_FACES];

  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout offscreen;
    WGPUBindGroupLayout scene;
  } bind_group_layouts;

  /* Pipeline layouts */
  struct {
    WGPUPipelineLayout offscreen;
    WGPUPipelineLayout scene;
  } pipeline_layouts;

  /* Bind groups */
  struct {
    WGPUBindGroup offscreen[NUM_CUBE_FACES]; /* One per face */
    WGPUBindGroup scene;
  } bind_groups;

  /* Render pipelines */
  struct {
    WGPURenderPipeline offscreen;
    WGPURenderPipeline scene;
    WGPURenderPipeline cubemap_display;
  } pipelines;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Light */
  vec4 light_pos;

  /* Animation */
  float timer;
  float timer_speed;
  uint64_t last_frame_time;

  /* Settings */
  struct {
    bool display_cube_map;
    bool paused;
  } settings;

  WGPUBool initialized;
} state = {
  /* Render pass defaults */
  .color_attachment = {
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.025f, 0.025f, 0.025f, 1.0f},
  },
  .depth_stencil_attachment = {
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
    .stencilLoadOp   = WGPULoadOp_Clear,
    .stencilStoreOp  = WGPUStoreOp_Store,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  /* Vulkan lightPos = (0, -2.5, 0, 1). Negate Y for WebGPU. */
  .light_pos   = {0.0f, 2.5f, 0.0f, 1.0f},
  .timer_speed = 0.125f, /* Vulkan: 0.25 * 0.5 */
};

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_models(void)
{
  const gltf_model_desc_t desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
    /* No FlipY for WebGPU */
  };

  state.scene_loaded = gltf_model_load_from_file_ext(
    &state.scene_model, scene_model_path, 1.0f, &desc);
  if (!state.scene_loaded) {
    printf("Failed to load scene model: %s\n", scene_model_path);
  }

  state.cube_loaded = gltf_model_load_from_file_ext(
    &state.cube_model, cube_model_path, 1.0f, &desc);
  if (!state.cube_loaded) {
    printf("Failed to load cube model: %s\n", cube_model_path);
  }
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Helper to create vertex + index buffers for a model */
  struct {
    gltf_model_t* model;
    bool loaded;
    WGPUBuffer* vb;
    WGPUBuffer* ib;
  } models[] = {
    {&state.scene_model, state.scene_loaded, &state.scene_buffers.vertex_buffer,
     &state.scene_buffers.index_buffer},
    {&state.cube_model, state.cube_loaded, &state.cube_buffers.vertex_buffer,
     &state.cube_buffers.index_buffer},
  };

  for (uint32_t i = 0; i < ARRAY_SIZE(models); i++) {
    if (!models[i].loaded) {
      continue;
    }

    gltf_model_t* m = models[i].model;
    size_t vb_size  = m->vertex_count * sizeof(gltf_vertex_t);
    size_t ib_size  = m->index_count * sizeof(uint32_t);

    /* Vertex buffer */
    *models[i].vb = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Model Vertex Buffer"),
                .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                .size  = vb_size,
                .mappedAtCreation = true,
              });
    void* vdata = wgpuBufferGetMappedRange(*models[i].vb, 0, vb_size);
    memcpy(vdata, m->vertices, vb_size);
    wgpuBufferUnmap(*models[i].vb);

    /* Index buffer */
    if (m->index_count > 0) {
      *models[i].ib = wgpuDeviceCreateBuffer(
        device, &(WGPUBufferDescriptor){
                  .label = STRVIEW("Model Index Buffer"),
                  .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                  .size  = ib_size,
                  .mappedAtCreation = true,
                });
      void* idata = wgpuBufferGetMappedRange(*models[i].ib, 0, ib_size);
      memcpy(idata, m->indices, ib_size);
      wgpuBufferUnmap(*models[i].ib);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Draw model helper
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass, gltf_model_t* model,
                       WGPUBuffer vb, WGPUBuffer ib)
{
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vb, 0, WGPU_WHOLE_SIZE);
  if (ib) {
    wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
  }

  for (uint32_t n = 0; n < model->linear_node_count; n++) {
    gltf_node_t* node = model->linear_nodes[n];
    if (!node->mesh) {
      continue;
    }
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; p++) {
      gltf_primitive_t* prim = &mesh->primitives[p];
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
 * Shadow cube map texture
 * -------------------------------------------------------------------------- */

static void init_shadow_cube_map(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Create the cube map texture: 6 layers, R32Float, RenderAttachment +
   * TextureBinding */
  state.shadow_cube_map.texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("Shadow Cube Map"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {SHADOW_MAP_DIM, SHADOW_MAP_DIM, NUM_CUBE_FACES},
              .format        = WGPUTextureFormat_R32Float,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });

  /* Cube view (all 6 layers) for sampling in the scene pass */
  state.shadow_cube_map.cube_view = wgpuTextureCreateView(
    state.shadow_cube_map.texture, &(WGPUTextureViewDescriptor){
                                     .label  = STRVIEW("Shadow Cube Map View"),
                                     .format = WGPUTextureFormat_R32Float,
                                     .dimension = WGPUTextureViewDimension_Cube,
                                     .baseMipLevel    = 0,
                                     .mipLevelCount   = 1,
                                     .baseArrayLayer  = 0,
                                     .arrayLayerCount = NUM_CUBE_FACES,
                                   });

  /* Per-face 2D views for rendering into each face */
  for (uint32_t i = 0; i < NUM_CUBE_FACES; i++) {
    state.shadow_cube_map.face_views[i] = wgpuTextureCreateView(
      state.shadow_cube_map.texture, &(WGPUTextureViewDescriptor){
                                       .label     = STRVIEW("Shadow Face View"),
                                       .format    = WGPUTextureFormat_R32Float,
                                       .dimension = WGPUTextureViewDimension_2D,
                                       .baseMipLevel    = 0,
                                       .mipLevelCount   = 1,
                                       .baseArrayLayer  = i,
                                       .arrayLayerCount = 1,
                                     });
  }

  /* Sampler: linear filtering, clamp to border (white) */
  state.shadow_cube_map.sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Shadow Cube Sampler"),
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .maxAnisotropy = 1,
            });
}

/* -------------------------------------------------------------------------- *
 * Offscreen depth attachment (shared by all 6 face passes)
 * -------------------------------------------------------------------------- */

static void init_offscreen_depth(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.offscreen_depth.texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label         = STRVIEW("Offscreen Depth"),
              .usage         = WGPUTextureUsage_RenderAttachment,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {SHADOW_MAP_DIM, SHADOW_MAP_DIM, 1},
              .format        = WGPUTextureFormat_Depth32Float,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });

  state.offscreen_depth.view = wgpuTextureCreateView(
    state.offscreen_depth.texture, &(WGPUTextureViewDescriptor){
                                     .label  = STRVIEW("Offscreen Depth View"),
                                     .format = WGPUTextureFormat_Depth32Float,
                                     .dimension = WGPUTextureViewDimension_2D,
                                     .baseMipLevel    = 0,
                                     .mipLevelCount   = 1,
                                     .baseArrayLayer  = 0,
                                     .arrayLayerCount = 1,
                                     .aspect = WGPUTextureAspect_DepthOnly,
                                   });
}

/* -------------------------------------------------------------------------- *
 * Main pass depth texture (recreated on resize)
 * -------------------------------------------------------------------------- */

static void init_depth_texture(struct wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.depth.view);
  WGPU_RELEASE_RESOURCE(Texture, state.depth.texture);

  WGPUDevice device = wgpu_context->device;
  uint32_t w        = (uint32_t)wgpu_context->width;
  uint32_t h        = (uint32_t)wgpu_context->height;

  state.depth.texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label         = STRVIEW("Main Depth"),
              .usage         = WGPUTextureUsage_RenderAttachment,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = WGPUTextureFormat_Depth24PlusStencil8,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });

  state.depth.view = wgpuTextureCreateView(
    state.depth.texture, &(WGPUTextureViewDescriptor){
                           .format    = WGPUTextureFormat_Depth24PlusStencil8,
                           .dimension = WGPUTextureViewDimension_2D,
                           .baseMipLevel    = 0,
                           .mipLevelCount   = 1,
                           .baseArrayLayer  = 0,
                           .arrayLayerCount = 1,
                         });
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene UBO */
  state.uniform_buffers.scene = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Scene UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(uniform_data_t),
            });

  /* Offscreen UBO */
  state.uniform_buffers.offscreen = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Offscreen UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(uniform_data_t),
            });

  /* Per-face view matrix buffers */
  for (uint32_t i = 0; i < NUM_CUBE_FACES; i++) {
    state.uniform_buffers.face_views[i] = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Face View UBO"),
                .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                .size  = sizeof(face_view_uniform_t),
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Cube map face view matrices
 *
 * WebGPU uses Y-up, Z-out-of-screen (right-handed) with clip Y going up.
 * The cube map faces in WebGPU correspond to:
 *   +X: look in +X direction, up = -Y (hardware convention for cube maps)
 *   -X: look in -X direction, up = -Y
 *   +Y: look in +Y direction, up = +Z
 *   -Y: look in -Y direction, up = -Z
 *   +Z: look in +Z direction, up = -Y
 *   -Z: look in -Z direction, up = -Y
 *
 * These match the standard cube map face orientation expected by
 * texture_cube sampling in WGSL.
 * -------------------------------------------------------------------------- */

static void compute_face_view_matrices(void)
{
  vec3 center = {0.0f, 0.0f, 0.0f};

  /* The view matrices look from origin towards each face direction.
   * The model matrix already translates world to light-centered coords. */
  struct {
    vec3 target;
    vec3 up;
  } face_dirs[NUM_CUBE_FACES] = {
    /* +X */ {{+1.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}},
    /* -X */ {{-1.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}},
    /* +Y */ {{0.0f, +1.0f, 0.0f}, {0.0f, 0.0f, +1.0f}},
    /* -Y */ {{0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},
    /* +Z */ {{0.0f, 0.0f, +1.0f}, {0.0f, -1.0f, 0.0f}},
    /* -Z */ {{0.0f, 0.0f, -1.0f}, {0.0f, -1.0f, 0.0f}},
  };

  for (uint32_t i = 0; i < NUM_CUBE_FACES; i++) {
    glm_lookat(center, face_dirs[i].target, face_dirs[i].up,
               state.face_view_data[i].face_view);
  }
}

/* -------------------------------------------------------------------------- *
 * Update uniform buffers
 * -------------------------------------------------------------------------- */

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUQueue queue = wgpu_context->queue;

  /* Animate light position (small orbit in XZ plane) */
  if (!state.settings.paused) {
    /*
     * Vulkan: lightPos.x = sin(rad(timer*360)) * 0.15
     *         lightPos.z = cos(rad(timer*360)) * 0.15
     *         lightPos.y stays at -2.5 (Vulkan Y-down, so -2.5 = above)
     * WebGPU: Y is negated, so light_pos.y = 2.5 (above)
     */
    float rad          = glm_rad(state.timer * 360.0f);
    state.light_pos[0] = sinf(rad) * 0.15f;
    state.light_pos[2] = cosf(rad) * 0.15f;
    /* Y stays at the initial value (+2.5 for WebGPU) */
  }

  /* === Offscreen UBO (shadow cube map generation) === */
  {
    /* 90° FOV perspective for cube face */
    glm_perspective(GLM_PI_2f, 1.0f, Z_NEAR, Z_FAR,
                    state.ubo_offscreen.projection);

    /* Flip Y: cube map face rendering in WebGPU needs Y-flip because
     * WebGPU clip Y-up doesn't match the texel layout (row 0 = top).
     * Vulkan's Y-down naturally matches, so Vulkan doesn't need this. */
    state.ubo_offscreen.projection[1][1] *= -1.0f;

    /* View is identity (actual face view comes from per-face uniform) */
    glm_mat4_identity(state.ubo_offscreen.view);

    /* Model matrix: translate world so light is at origin */
    glm_mat4_identity(state.ubo_offscreen.model);
    glm_translate(
      state.ubo_offscreen.model,
      (vec3){-state.light_pos[0], -state.light_pos[1], -state.light_pos[2]});

    glm_vec4_copy(state.light_pos, state.ubo_offscreen.light_pos);

    wgpuQueueWriteBuffer(queue, state.uniform_buffers.offscreen, 0,
                         &state.ubo_offscreen, sizeof(uniform_data_t));
  }

  /* === Per-face view matrices === */
  compute_face_view_matrices();
  for (uint32_t i = 0; i < NUM_CUBE_FACES; i++) {
    wgpuQueueWriteBuffer(queue, state.uniform_buffers.face_views[i], 0,
                         &state.face_view_data[i], sizeof(face_view_uniform_t));
  }

  /* === Scene UBO (main pass) === */
  {
    glm_mat4_copy(state.camera.matrices.perspective,
                  state.ubo_scene.projection);
    glm_mat4_copy(state.camera.matrices.view, state.ubo_scene.view);
    glm_mat4_identity(state.ubo_scene.model);
    glm_vec4_copy(state.light_pos, state.ubo_scene.light_pos);

    wgpuQueueWriteBuffer(queue, state.uniform_buffers.scene, 0,
                         &state.ubo_scene, sizeof(uniform_data_t));
  }
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Offscreen: binding 0 = UBO (vertex), binding 1 = face view UBO (vertex) */
  {
    WGPUBindGroupLayoutEntry entries[2] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniform_data_t),
        },
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(face_view_uniform_t),
        },
      },
    };

    state.bind_group_layouts.offscreen = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Offscreen BGL"),
                .entryCount = 2,
                .entries    = entries,
              });
  }

  /* Scene: binding 0 = UBO, binding 1 = shadow cube texture,
   *        binding 2 = shadow cube sampler */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniform_data_t),
        },
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
        },
      },
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
    };

    state.bind_group_layouts.scene = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Scene BGL"),
                .entryCount = 3,
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

  state.pipeline_layouts.offscreen = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Offscreen Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.bind_group_layouts.offscreen,
            });

  state.pipeline_layouts.scene = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Scene Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.bind_group_layouts.scene,
            });
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Offscreen: one bind group per face (different face_view uniform) */
  for (uint32_t i = 0; i < NUM_CUBE_FACES; i++) {
    WGPUBindGroupEntry entries[2] = {
      {
        .binding = 0,
        .buffer  = state.uniform_buffers.offscreen,
        .offset  = 0,
        .size    = sizeof(uniform_data_t),
      },
      {
        .binding = 1,
        .buffer  = state.uniform_buffers.face_views[i],
        .offset  = 0,
        .size    = sizeof(face_view_uniform_t),
      },
    };

    state.bind_groups.offscreen[i] = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Offscreen Bind Group"),
                .layout     = state.bind_group_layouts.offscreen,
                .entryCount = 2,
                .entries    = entries,
              });
  }

  /* Scene bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      {
        .binding = 0,
        .buffer  = state.uniform_buffers.scene,
        .offset  = 0,
        .size    = sizeof(uniform_data_t),
      },
      {
        .binding     = 1,
        .textureView = state.shadow_cube_map.cube_view,
      },
      {
        .binding = 2,
        .sampler = state.shadow_cube_map.sampler,
      },
    };

    state.bind_groups.scene = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Scene Bind Group"),
                .layout     = state.bind_group_layouts.scene,
                .entryCount = 3,
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ===== Offscreen pipeline (shadow map, renders distance to R32Float) =====
   */
  {
    WGPUVertexAttribute attrs[] = {
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
    };

    WGPUVertexBufferLayout vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(attrs),
      .attributes     = attrs,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, offscreen_shader_wgsl);

    WGPUColorTargetState target = {
      .format    = WGPUTextureFormat_R32Float,
      .blend     = NULL, /* R32Float does not support blending */
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = WGPUTextureFormat_Depth32Float,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
    };

    state.pipelines.offscreen = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Offscreen Pipeline"),
        .layout = state.pipeline_layouts.offscreen,
        .vertex = (WGPUVertexState){
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &vb_layout,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CW, /* CW due to projection Y-flip */
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
          .targets     = &target,
        },
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
  }

  /* ===== Scene pipeline (shadow mapped lighting) ===== */
  {
    WGPUVertexAttribute attrs[] = {
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, color)},
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal)},
    };

    WGPUVertexBufferLayout vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(attrs),
      .attributes     = attrs,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, scene_shader_wgsl);

    WGPUBlendState blend        = wgpu_create_blend_state(false);
    WGPUColorTargetState target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = WGPUTextureFormat_Depth24PlusStencil8,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    state.pipelines.scene = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Scene Pipeline"),
        .layout = state.pipeline_layouts.scene,
        .vertex = (WGPUVertexState){
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &vb_layout,
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
          .targets     = &target,
        },
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
  }

  /* ===== Cubemap display pipeline (debug visualization) ===== */
  {
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, cubemap_display_shader_wgsl);

    WGPUBlendState blend        = wgpu_create_blend_state(false);
    WGPUColorTargetState target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = WGPUTextureFormat_Depth24PlusStencil8,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    state.pipelines.cubemap_display = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Cubemap Display Pipeline"),
        .layout = state.pipeline_layouts.scene,
        .vertex = (WGPUVertexState){
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 0,
          .buffers     = NULL,
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
          .module      = shader,
          .entryPoint  = STRVIEW("fs_main"),
          .targetCount = 1,
          .targets     = &target,
        },
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
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
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    igCheckbox("Display shadow cube render target",
               &state.settings.display_cube_map);
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
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
    camera_update_aspect_ratio(&state.camera, (float)wgpu_context->width
                                                / (float)wgpu_context->height);
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();

  /* Camera setup:
   * Vulkan: pos(0, 0.5, -15), rot(-20.5, -673, 0), FOV 45, lookat
   * Apply porting guide: negate pos Y, negate rot X */
  camera_init(&state.camera);
  state.camera.type      = CameraType_LookAt;
  state.camera.invert_dx = true;
  state.camera.invert_dy = true;
  camera_set_position(&state.camera,
                      (vec3)VKY_TO_WGPU_VEC3(0.0f, 0.5f, -15.0f));
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-20.5f, -673.0f, 0.0f));
  camera_set_perspective(
    &state.camera, 45.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, Z_NEAR, Z_FAR);

  /* Load models synchronously */
  load_models();
  create_model_buffers(wgpu_context);

  /* Shadow cube map */
  init_shadow_cube_map(wgpu_context);
  init_offscreen_depth(wgpu_context);

  /* Main pass depth */
  init_depth_texture(wgpu_context);

  /* Uniform buffers */
  init_uniform_buffers(wgpu_context);

  /* Bind group / pipeline layouts */
  init_bind_group_layouts(wgpu_context);
  init_pipeline_layouts(wgpu_context);

  /* Bind groups */
  init_bind_groups(wgpu_context);

  /* Pipelines */
  init_pipelines(wgpu_context);

  /* ImGui */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Timing */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Update animation timer (timerSpeed * 0.5 from Vulkan) */
  state.timer += delta_time * state.timer_speed;
  if (state.timer > 1.0f) {
    state.timer -= 1.0f;
  }

  /* Update camera */
  camera_update(&state.camera, delta_time);

  /* Update uniforms */
  update_uniform_buffers(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* ---- Rendering ---- */
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* ============ Pass 1: Shadow cube map generation (6 faces) ============ */
  if (state.scene_loaded) {
    for (uint32_t face = 0; face < NUM_CUBE_FACES; face++) {
      WGPURenderPassColorAttachment offscreen_color_att = {
        .view       = state.shadow_cube_map.face_views[face],
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
      };

      WGPURenderPassDepthStencilAttachment offscreen_depth_att = {
        .view            = state.offscreen_depth.view,
        .depthLoadOp     = WGPULoadOp_Clear,
        .depthStoreOp    = WGPUStoreOp_Store,
        .depthClearValue = 1.0f,
      };

      WGPURenderPassDescriptor offscreen_pass_desc = {
        .colorAttachmentCount   = 1,
        .colorAttachments       = &offscreen_color_att,
        .depthStencilAttachment = &offscreen_depth_att,
      };

      WGPURenderPassEncoder pass
        = wgpuCommandEncoderBeginRenderPass(cmd_enc, &offscreen_pass_desc);

      wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)SHADOW_MAP_DIM,
                                       (float)SHADOW_MAP_DIM, 0.0f, 1.0f);
      wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, SHADOW_MAP_DIM,
                                          SHADOW_MAP_DIM);

      wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.offscreen);
      wgpuRenderPassEncoderSetBindGroup(
        pass, 0, state.bind_groups.offscreen[face], 0, NULL);

      draw_model(pass, &state.scene_model, state.scene_buffers.vertex_buffer,
                 state.scene_buffers.index_buffer);

      wgpuRenderPassEncoderEnd(pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass);
    }
  }

  /* ============ Pass 2: Scene rendering with shadows ============ */
  {
    state.color_attachment.view         = wgpu_context->swapchain_view;
    state.depth_stencil_attachment.view = state.depth.view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)wgpu_context->width,
                                     (float)wgpu_context->height, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0,
                                        (uint32_t)wgpu_context->width,
                                        (uint32_t)wgpu_context->height);

    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_groups.scene, 0,
                                      NULL);

    if (state.settings.display_cube_map) {
      /* Debug: display all 6 faces of the shadow cube map */
      wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.cubemap_display);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    }
    else if (state.scene_loaded) {
      /* Scene with shadow mapping */
      wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.scene);
      draw_model(pass, &state.scene_model, state.scene_buffers.vertex_buffer,
                 state.scene_buffers.index_buffer);
    }

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass);
  }

  /* Submit */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buffer);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc);

  /* ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Models */
  if (state.scene_loaded) {
    gltf_model_destroy(&state.scene_model);
  }
  if (state.cube_loaded) {
    gltf_model_destroy(&state.cube_model);
  }
  WGPU_RELEASE_RESOURCE(Buffer, state.scene_buffers.vertex_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.scene_buffers.index_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_buffers.vertex_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_buffers.index_buffer);

  /* Shadow cube map */
  for (uint32_t i = 0; i < NUM_CUBE_FACES; i++) {
    WGPU_RELEASE_RESOURCE(TextureView, state.shadow_cube_map.face_views[i]);
  }
  WGPU_RELEASE_RESOURCE(TextureView, state.shadow_cube_map.cube_view);
  WGPU_RELEASE_RESOURCE(Texture, state.shadow_cube_map.texture);
  WGPU_RELEASE_RESOURCE(Sampler, state.shadow_cube_map.sampler);

  /* Offscreen depth */
  WGPU_RELEASE_RESOURCE(TextureView, state.offscreen_depth.view);
  WGPU_RELEASE_RESOURCE(Texture, state.offscreen_depth.texture);

  /* Main depth */
  WGPU_RELEASE_RESOURCE(TextureView, state.depth.view);
  WGPU_RELEASE_RESOURCE(Texture, state.depth.texture);

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.scene);
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.offscreen);
  for (uint32_t i = 0; i < NUM_CUBE_FACES; i++) {
    WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.face_views[i]);
  }

  /* Bind groups */
  for (uint32_t i = 0; i < NUM_CUBE_FACES; i++) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.offscreen[i]);
  }
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.scene);

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.offscreen);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.scene);

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.offscreen);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.scene);

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.offscreen);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.scene);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.cubemap_display);
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  /* Request Float32Filterable so R32Float shadow map can be sampled */
  WGPUFeatureName required_features[] = {WGPUFeatureName_Float32Filterable};

  wgpu_start(&(wgpu_desc_t){
    .title                  = "Point light shadows (cubemap)",
    .init_cb                = init,
    .frame_cb               = frame,
    .shutdown_cb            = shutdown,
    .input_event_cb         = input_event_cb,
    .required_features      = required_features,
    .required_feature_count = ARRAY_SIZE(required_features),
  });
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off

/*
 * Offscreen shader: renders linear distance from light to the R32Float
 * cube map face. The UBO model matrix translates world so the light is at
 * origin. The face_view uniform provides the per-face look direction.
 */
static const char* offscreen_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
    lightPos   : vec4f,
  }

  struct FaceView {
    viewMatrix : mat4x4f,
  }

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var<uniform> faceView : FaceView;

  struct VSOutput {
    @builtin(position) position : vec4f,
    @location(0) worldPos  : vec3f,
    @location(1) lightPos  : vec3f,
  }

  @vertex
  fn vs_main(@location(0) inPos : vec3f) -> VSOutput {
    var out : VSOutput;
    out.position = ubo.projection * faceView.viewMatrix * ubo.model * vec4f(inPos, 1.0);
    out.worldPos = inPos;
    out.lightPos = ubo.lightPos.xyz;
    return out;
  }

  @fragment
  fn fs_main(in : VSOutput) -> @location(0) f32 {
    let lightVec = in.worldPos - in.lightPos;
    return length(lightVec);
  }
);

/*
 * Scene shader: renders the scene with ambient + diffuse lighting and
 * point-light shadow mapping via cube map distance comparison.
 */
static const char* scene_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
    lightPos   : vec4f,
  }

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var shadowCubeMap : texture_cube<f32>;
  @group(0) @binding(2) var shadowSampler : sampler;

  const EPSILON : f32 = 0.15;
  const SHADOW_OPACITY : f32 = 0.5;

  struct VSOutput {
    @builtin(position) position : vec4f,
    @location(0) normal    : vec3f,
    @location(1) color     : vec4f,
    @location(2) eyePos    : vec3f,
    @location(3) lightVec  : vec3f,
    @location(4) worldPos  : vec3f,
    @location(5) lightPos  : vec3f,
  }

  @vertex
  fn vs_main(
    @location(0) inPos    : vec3f,
    @location(1) inColor  : vec4f,
    @location(2) inNormal : vec3f,
  ) -> VSOutput {
    var out : VSOutput;
    out.color = inColor;
    out.normal = inNormal;

    out.position = ubo.projection * ubo.view * ubo.model * vec4f(inPos, 1.0);
    out.eyePos = (ubo.model * vec4f(inPos, 1.0)).xyz;
    out.lightVec = normalize(ubo.lightPos.xyz - inPos);
    out.worldPos = inPos;
    out.lightPos = ubo.lightPos.xyz;
    return out;
  }

  @fragment
  fn fs_main(in : VSOutput) -> @location(0) vec4f {
    let N = normalize(in.normal);

    let IAmbient = vec4f(vec3f(0.05), 1.0);
    let IDiffuse = vec4f(1.0) * max(dot(in.normal, in.lightVec), 0.0);

    var outColor = IAmbient + IDiffuse * in.color;

    /* Shadow test: sample cube map with direction from light to fragment */
    let lightVec = in.worldPos - in.lightPos;
    let sampledDist = textureSampleLevel(shadowCubeMap, shadowSampler, lightVec, 0.0).r;
    let dist = length(lightVec);

    var shadow : f32 = 1.0;
    if (dist > sampledDist + EPSILON) {
      shadow = SHADOW_OPACITY;
    }

    return vec4f(outColor.rgb * shadow, 1.0);
  }
);

/*
 * Cubemap display shader: visualizes all 6 faces of the shadow cube map
 * in a cross layout using a procedural fullscreen triangle.
 * Ported from Vulkan GLSL cubemapdisplay shader.
 */
static const char* cubemap_display_shader_wgsl = CODE(
  @group(0) @binding(0) var<uniform> ubo : vec4f; /* unused, for layout compat */
  @group(0) @binding(1) var shadowCubeMap : texture_cube<f32>;
  @group(0) @binding(2) var shadowSampler : sampler;

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
    var outColor = vec3f(0.05);

    var samplePos = vec3f(0.0);

    let x = i32(floor(in.uv.x / 0.25));
    let y = i32(floor(in.uv.y / (1.0 / 3.0)));

    if (y == 1) {
      let uv_local = vec2f(
        in.uv.x * 4.0,
        (in.uv.y - 1.0 / 3.0) * 3.0
      );
      let uv_mapped = 2.0 * vec2f(uv_local.x - f32(x) * 1.0, uv_local.y) - 1.0;

      switch (x) {
        case 0: { samplePos = vec3f(-1.0, uv_mapped.y, uv_mapped.x); }  /* -X */
        case 1: { samplePos = vec3f(uv_mapped.x, uv_mapped.y, 1.0); }   /* +Z */
        case 2: { samplePos = vec3f(1.0, uv_mapped.y, -uv_mapped.x); }  /* +X */
        case 3: { samplePos = vec3f(-uv_mapped.x, uv_mapped.y, -1.0); } /* -Z */
        default: { }
      }
    } else {
      if (x == 1) {
        let uv_local = vec2f(
          (in.uv.x - 0.25) * 4.0,
          (in.uv.y - f32(y) / 3.0) * 3.0
        );
        let uv_mapped = 2.0 * uv_local - 1.0;

        switch (y) {
          case 0: { samplePos = vec3f(uv_mapped.x, -1.0, uv_mapped.y); } /* -Y */
          case 2: { samplePos = vec3f(uv_mapped.x, 1.0, -uv_mapped.y); } /* +Y */
          default: { }
        }
      }
    }

    if (samplePos.x != 0.0 || samplePos.y != 0.0 || samplePos.z != 0.0) {
      let dist = length(textureSampleLevel(shadowCubeMap, shadowSampler, samplePos, 0.0).xyz) * 0.005;
      outColor = vec3f(dist);
    }

    return vec4f(outColor, 1.0);
  }
);

// clang-format on
