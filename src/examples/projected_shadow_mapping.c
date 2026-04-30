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
 * WebGPU Example - Projected Shadow Mapping
 *
 * Shadow mapping for directional light sources. The shadow map is generated in
 * a first pass from the light's point of view. The scene is then rendered with
 * shadow coord lookup in the shadow map. Includes PCF (Percentage Closer
 * Filtering) for soft shadow edges, plus a debug mode to visualize the shadow
 * map.
 *
 * Ported from the Vulkan example:
 *   src/examples/Vulkan/examples/shadowmapping/shadowmapping.cpp
 *
 * Ref:
 *   https://github.com/SaschaWillems/Vulkan/tree/master/examples/shadowmapping
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */
static const char* offscreen_shader_wgsl;
static const char* scene_shader_wgsl;
static const char* debug_quad_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define SHADOW_MAP_DIM 2048u
#define NUM_SCENES 2

/* Scene model file paths */
static const char* scene_paths[NUM_SCENES] = {
  "assets/models/vulkanscene_shadow.gltf",
  "assets/models/samplescene.gltf",
};

static const char* scene_names[NUM_SCENES] = {
  "Vulkan scene",
  "Teapots and pillars",
};

/* -------------------------------------------------------------------------- *
 * Uniform data
 * -------------------------------------------------------------------------- */

/* Offscreen pass: light projection MVP (depth only) */
typedef struct {
  mat4 depth_mvp;
} uniform_data_offscreen_t;

/* Scene pass: camera matrices + shadow coord + lighting */
typedef struct {
  mat4 projection;
  mat4 view;
  mat4 model;
  mat4 light_space;
  vec4 light_pos;
  float z_near;
  float z_far;
  float _pad[2]; /* 16-byte alignment */
} uniform_data_scene_t;

/* -------------------------------------------------------------------------- *
 * Global state struct
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Animation timer */
  float timer;
  float animation_speed;
  uint64_t last_frame_time;

  /* Models */
  gltf_model_t scenes[NUM_SCENES];
  bool scenes_loaded[NUM_SCENES];

  /* GPU vertex/index buffers per scene */
  struct {
    WGPUBuffer vertex_buffer;
    WGPUBuffer index_buffer;
  } scene_buffers[NUM_SCENES];

  /* Shadow map */
  struct {
    WGPUTexture depth_texture;
    WGPUTextureView depth_view;
  } shadow_map;

  /* Depth texture for main render pass */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth;

  /* Uniform buffers */
  struct {
    WGPUBuffer offscreen;
    WGPUBuffer scene;
  } uniform_buffers;

  /* Uniform data */
  uniform_data_offscreen_t ubo_offscreen;
  uniform_data_scene_t ubo_scene;

  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout offscreen;
    WGPUBindGroupLayout scene;
  } bind_group_layouts;

  /* Pipeline layouts */
  struct {
    WGPUPipelineLayout offscreen;
    WGPUPipelineLayout scene; /* also used for debug quad */
  } pipeline_layouts;

  /* Bind groups */
  struct {
    WGPUBindGroup offscreen;
    WGPUBindGroup scene;
    WGPUBindGroup debug;
  } bind_groups;

  /* Render pipelines */
  struct {
    WGPURenderPipeline offscreen;        /* Shadow pass: depth only */
    WGPURenderPipeline scene_shadow;     /* Scene with basic shadow */
    WGPURenderPipeline scene_shadow_pcf; /* Scene with PCF shadow */
    WGPURenderPipeline debug;            /* Shadow map visualization */
  } pipelines;

  /* Shadow pass render descriptors */
  struct {
    WGPURenderPassDepthStencilAttachment depth_att;
    WGPURenderPassDescriptor descriptor;
  } shadow_pass;

  /* Main pass render descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Settings (GUI) */
  struct {
    int32_t scene_index;
    bool display_shadow_map;
    bool filter_pcf;
  } settings;

  /* Light parameters */
  float z_near;
  float z_far;
  float light_fov;
  vec3 light_pos;

  WGPUBool initialized;
} state = {
  /* Default render pass */
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  /* Shadow pass (depth only) */
  .shadow_pass = {
    .depth_att = {
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
      .stencilLoadOp   = WGPULoadOp_Undefined,
      .stencilStoreOp  = WGPUStoreOp_Undefined,
    },
    .descriptor = {
      .colorAttachmentCount   = 0,
      .colorAttachments       = NULL,
      .depthStencilAttachment = &state.shadow_pass.depth_att,
    },
  },
  /* Settings defaults */
  .settings = {
    .scene_index        = 0,
    .display_shadow_map = false,
    .filter_pcf         = true,
  },
  /* Light defaults */
  .z_near    = 1.0f,
  .z_far     = 96.0f,
  .light_fov = 45.0f,
  .animation_speed = 0.125f, /* Vulkan base timerSpeed (0.25) * 0.5 */
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

  for (int i = 0; i < NUM_SCENES; i++) {
    state.scenes_loaded[i] = gltf_model_load_from_file_ext(
      &state.scenes[i], scene_paths[i], 1.0f, &desc);
    if (!state.scenes_loaded[i]) {
      printf("Failed to load scene: %s\n", scene_paths[i]);
    }
  }
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  for (int i = 0; i < NUM_SCENES; i++) {
    if (!state.scenes_loaded[i]) {
      continue;
    }

    gltf_model_t* m = &state.scenes[i];
    size_t vb_size  = m->vertex_count * sizeof(gltf_vertex_t);
    size_t ib_size  = m->index_count * sizeof(uint32_t);

    /* Vertex buffer */
    state.scene_buffers[i].vertex_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Scene - Vertex Buffer"),
                .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                .size  = vb_size,
                .mappedAtCreation = true,
              });
    void* vdata = wgpuBufferGetMappedRange(state.scene_buffers[i].vertex_buffer,
                                           0, vb_size);
    memcpy(vdata, m->vertices, vb_size);
    wgpuBufferUnmap(state.scene_buffers[i].vertex_buffer);

    /* Index buffer */
    if (m->index_count > 0) {
      state.scene_buffers[i].index_buffer = wgpuDeviceCreateBuffer(
        device, &(WGPUBufferDescriptor){
                  .label = STRVIEW("Scene - Index Buffer"),
                  .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                  .size  = ib_size,
                  .mappedAtCreation = true,
                });
      void* idata = wgpuBufferGetMappedRange(
        state.scene_buffers[i].index_buffer, 0, ib_size);
      memcpy(idata, m->indices, ib_size);
      wgpuBufferUnmap(state.scene_buffers[i].index_buffer);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Draw model helper
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass, int scene_index)
{
  if (!state.scenes_loaded[scene_index]) {
    return;
  }

  gltf_model_t* model = &state.scenes[scene_index];
  WGPUBuffer vb       = state.scene_buffers[scene_index].vertex_buffer;
  WGPUBuffer ib       = state.scene_buffers[scene_index].index_buffer;

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
 * Shadow map texture + sampler
 * -------------------------------------------------------------------------- */

static void init_shadow_map(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Depth texture for shadow map */
  state.shadow_map.depth_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("Shadow Map Depth"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {SHADOW_MAP_DIM, SHADOW_MAP_DIM, 1},
              .format        = WGPUTextureFormat_Depth32Float,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });

  /* Single 2D view for shadow pass render target */
  state.shadow_map.depth_view = wgpuTextureCreateView(
    state.shadow_map.depth_texture, &(WGPUTextureViewDescriptor){
                                      .label = STRVIEW("Shadow Map Depth View"),
                                      .format = WGPUTextureFormat_Depth32Float,
                                      .dimension = WGPUTextureViewDimension_2D,
                                      .baseMipLevel    = 0,
                                      .mipLevelCount   = 1,
                                      .baseArrayLayer  = 0,
                                      .arrayLayerCount = 1,
                                      .aspect = WGPUTextureAspect_DepthOnly,
                                    });

  /* Set the shadow pass depth view */
  state.shadow_pass.depth_att.view = state.shadow_map.depth_view;
}

/* -------------------------------------------------------------------------- *
 * Main pass depth texture
 * -------------------------------------------------------------------------- */

static void init_depth_texture(struct wgpu_context_t* wgpu_context)
{
  /* Release old */
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

  state.uniform_buffers.offscreen = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Offscreen UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(uniform_data_offscreen_t),
            });

  state.uniform_buffers.scene = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Scene UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(uniform_data_scene_t),
            });
}

/* -------------------------------------------------------------------------- *
 * Light animation
 * -------------------------------------------------------------------------- */

static void update_light(void)
{
  float angle = state.timer * 360.0f;
  float rad   = glm_rad(angle);

  /* Vulkan Y is down, WebGPU Y is up: negate Y */
  state.light_pos[0] = cosf(rad) * 40.0f;
  state.light_pos[1] = -(-50.0f + sinf(rad) * 20.0f); /* negated for WebGPU */
  state.light_pos[2] = 25.0f + sinf(rad) * 5.0f;
}

/* -------------------------------------------------------------------------- *
 * Update uniform buffers
 * -------------------------------------------------------------------------- */

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUQueue queue = wgpu_context->queue;

  /* === Light's point of view (shadow map generation) === */
  {
    mat4 depth_projection, depth_view, depth_mvp;

    glm_perspective(glm_rad(state.light_fov), 1.0f, state.z_near, state.z_far,
                    depth_projection);

    /* lookAt from light position to scene center */
    vec3 center = {0.0f, 0.0f, 0.0f};
    vec3 up     = {0.0f, 1.0f, 0.0f};
    glm_lookat(state.light_pos, center, up, depth_view);

    glm_mat4_mul(depth_projection, depth_view, depth_mvp);
    glm_mat4_copy(depth_mvp, state.ubo_offscreen.depth_mvp);

    wgpuQueueWriteBuffer(queue, state.uniform_buffers.offscreen, 0,
                         &state.ubo_offscreen, sizeof(state.ubo_offscreen));
  }

  /* === Camera point of view (scene rendering) === */
  {
    glm_mat4_copy(state.camera.matrices.perspective,
                  state.ubo_scene.projection);
    glm_mat4_copy(state.camera.matrices.view, state.ubo_scene.view);
    glm_mat4_identity(state.ubo_scene.model);

    /* depthBiasMVP = depthMVP (same as offscreen) */
    glm_mat4_copy(state.ubo_offscreen.depth_mvp, state.ubo_scene.light_space);

    glm_vec4_copy(
      (vec4){state.light_pos[0], state.light_pos[1], state.light_pos[2], 1.0f},
      state.ubo_scene.light_pos);

    state.ubo_scene.z_near = state.z_near;
    state.ubo_scene.z_far  = state.z_far;

    wgpuQueueWriteBuffer(queue, state.uniform_buffers.scene, 0,
                         &state.ubo_scene, sizeof(state.ubo_scene));
  }
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Offscreen: binding 0 = UBO (vertex only) */
  {
    WGPUBindGroupLayoutEntry entry = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uniform_data_offscreen_t),
      },
    };

    state.bind_group_layouts.offscreen = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Offscreen BGL"),
                .entryCount = 1,
                .entries    = &entry,
              });
  }

  /* Scene/Debug: binding 0 = UBO, binding 1 = shadow map depth texture */
  {
    WGPUBindGroupLayoutEntry entries[2] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniform_data_scene_t),
        },
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = {
          .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
    };

    state.bind_group_layouts.scene = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Scene BGL"),
                .entryCount = 2,
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

  /* Offscreen */
  {
    WGPUBindGroupEntry entry = {
      .binding = 0,
      .buffer  = state.uniform_buffers.offscreen,
      .offset  = 0,
      .size    = sizeof(uniform_data_offscreen_t),
    };

    state.bind_groups.offscreen = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Offscreen Bind Group"),
                .layout     = state.bind_group_layouts.offscreen,
                .entryCount = 1,
                .entries    = &entry,
              });
  }

  /* Scene + Debug (share same layout and resources) */
  {
    WGPUBindGroupEntry entries[2] = {
      {
        .binding = 0,
        .buffer  = state.uniform_buffers.scene,
        .offset  = 0,
        .size    = sizeof(uniform_data_scene_t),
      },
      {
        .binding     = 1,
        .textureView = state.shadow_map.depth_view,
      },
    };

    state.bind_groups.scene = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Scene Bind Group"),
                .layout     = state.bind_group_layouts.scene,
                .entryCount = 2,
                .entries    = entries,
              });

    /* Debug uses same resources */
    state.bind_groups.debug = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Debug Bind Group"),
                .layout     = state.bind_group_layouts.scene,
                .entryCount = 2,
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

  /* ===== Offscreen pipeline (shadow map, depth-only) ===== */
  {
    /* Only position attribute needed */
    WGPUVertexAttribute shadow_attr = {
      .shaderLocation = 0,
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, position),
    };

    WGPUVertexBufferLayout shadow_vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &shadow_attr,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, offscreen_shader_wgsl);

    WGPUDepthStencilState shadow_depth_stencil = {
      .format              = WGPUTextureFormat_Depth32Float,
      .depthWriteEnabled   = WGPUOptionalBool_True,
      .depthCompare        = WGPUCompareFunction_LessEqual,
      .depthBias           = 2,
      .depthBiasSlopeScale = 1.75f,
      .depthBiasClamp      = 0.0f,
    };

    state.pipelines.offscreen = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Offscreen Pipeline"),
        .layout = state.pipeline_layouts.offscreen,
        .vertex = (WGPUVertexState){
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &shadow_vb_layout,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_None,
        },
        .depthStencil = &shadow_depth_stencil,
        .multisample  = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = NULL, /* Depth-only pass */
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
  }

  /* ===== Scene pipeline (shadow, no PCF) ===== */
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

    WGPUVertexBufferLayout scene_vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(scene_attrs),
      .attributes     = scene_attrs,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, scene_shader_wgsl);

    WGPUBlendState blend_state        = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = WGPUTextureFormat_Depth24PlusStencil8,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    /* No PCF variant */
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
          .buffers     = &scene_vb_layout,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Back,
        },
        .depthStencil = &depth_stencil,
        .multisample = (WGPUMultisampleState){
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

    /* PCF variant */
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
          .buffers     = &scene_vb_layout,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Back,
        },
        .depthStencil = &depth_stencil,
        .multisample = (WGPUMultisampleState){
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

  /* ===== Debug pipeline (shadow map visualization) ===== */
  {
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, debug_quad_shader_wgsl);

    WGPUBlendState blend_state        = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = WGPUTextureFormat_Depth24PlusStencil8,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    state.pipelines.debug = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Debug Quad Pipeline"),
        .layout = state.pipeline_layouts.scene,
        .vertex = (WGPUVertexState){
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 0, /* Procedural quad */
          .buffers     = NULL,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_None,
        },
        .depthStencil = &depth_stencil,
        .multisample = (WGPUMultisampleState){
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
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_combo_box("Scenes", &state.settings.scene_index, scene_names,
                            NUM_SCENES);
    igCheckbox("Display shadow render target",
               &state.settings.display_shadow_map);
    igCheckbox("PCF filtering", &state.settings.filter_pcf);
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

  /* Camera setup */
  camera_init(&state.camera);
  state.camera.type = CameraType_LookAt;
  /* Vulkan: pos(0, 0, -12.5), rot(-25, -390, 0) */
  camera_set_position(&state.camera,
                      (vec3)VKY_TO_WGPU_VEC3(0.0f, 0.0f, -12.5f));
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-25.0f, -390.0f, 0.0f));
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 1.0f, 256.0f);

  /* Load models synchronously (they're small) */
  load_models();
  create_model_buffers(wgpu_context);

  /* Shadow map texture */
  init_shadow_map(wgpu_context);

  /* Main pass depth */
  init_depth_texture(wgpu_context);

  /* Uniform buffers */
  init_uniform_buffers(wgpu_context);

  /* Bind group layouts + pipeline layouts */
  init_bind_group_layouts(wgpu_context);
  init_pipeline_layouts(wgpu_context);

  /* Bind groups */
  init_bind_groups(wgpu_context);

  /* Render pipelines */
  init_pipelines(wgpu_context);

  /* ImGui overlay */
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

  /* Update animation timer */
  state.timer += delta_time * state.animation_speed;
  if (state.timer > 1.0f) {
    state.timer -= 1.0f;
  }

  /* Update camera */
  camera_update(&state.camera, delta_time);

  /* Update light + uniforms */
  update_light();
  update_uniform_buffers(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* ---- Rendering ---- */
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;
  int32_t si        = state.settings.scene_index;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* ============ Pass 1: Shadow map generation ============ */
  if (state.scenes_loaded[si]) {
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.shadow_pass.descriptor);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)SHADOW_MAP_DIM,
                                     (float)SHADOW_MAP_DIM, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, SHADOW_MAP_DIM,
                                        SHADOW_MAP_DIM);
    wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.offscreen);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_groups.offscreen, 0,
                                      NULL);
    draw_model(pass, si);

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass);
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

    if (state.settings.display_shadow_map) {
      /* Debug: show shadow map as fullscreen quad */
      wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.debug);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_groups.debug, 0,
                                        NULL);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    }
    else if (state.scenes_loaded[si]) {
      /* Scene with shadow mapping */
      WGPURenderPipeline pipeline = state.settings.filter_pcf ?
                                      state.pipelines.scene_shadow_pcf :
                                      state.pipelines.scene_shadow;
      wgpuRenderPassEncoderSetPipeline(pass, pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_groups.scene, 0,
                                        NULL);
      draw_model(pass, si);
    }

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass);
  }

  /* Submit */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
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
  for (int i = 0; i < NUM_SCENES; i++) {
    if (state.scenes_loaded[i]) {
      gltf_model_destroy(&state.scenes[i]);
    }
    WGPU_RELEASE_RESOURCE(Buffer, state.scene_buffers[i].vertex_buffer);
    WGPU_RELEASE_RESOURCE(Buffer, state.scene_buffers[i].index_buffer);
  }

  /* Shadow map */
  WGPU_RELEASE_RESOURCE(TextureView, state.shadow_map.depth_view);
  WGPU_RELEASE_RESOURCE(Texture, state.shadow_map.depth_texture);

  /* Depth */
  WGPU_RELEASE_RESOURCE(TextureView, state.depth.view);
  WGPU_RELEASE_RESOURCE(Texture, state.depth.texture);

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.offscreen);
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.scene);

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.offscreen);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.scene);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.debug);

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.offscreen);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.scene);

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.offscreen);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.scene);

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.offscreen);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.scene_shadow);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.scene_shadow_pcf);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.debug);
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Projected shadow mapping",
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

/* Offscreen shader: shadow map depth-only pass */
static const char* offscreen_shader_wgsl = CODE(
  struct OffscreenUBO {
    depthMVP : mat4x4f,
  }

  @group(0) @binding(0) var<uniform> ubo : OffscreenUBO;

  @vertex
  fn vs_main(@location(0) inPos : vec3f) -> @builtin(position) vec4f {
    return ubo.depthMVP * vec4f(inPos, 1.0);
  }
);

/* Scene shader: shadow mapping with lighting */
static const char* scene_shader_wgsl = CODE(
  struct SceneUBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
    lightSpace : mat4x4f,
    lightPos   : vec4f,
    zNear      : f32,
    zFar       : f32,
  }

  @group(0) @binding(0) var<uniform> ubo : SceneUBO;
  @group(0) @binding(1) var shadowMap : texture_2d<f32>;

  override enablePCF : u32 = 0u;

  struct VSOutput {
    @builtin(position) position : vec4f,
    @location(0) normal    : vec3f,
    @location(1) color     : vec4f,
    @location(2) viewVec   : vec3f,
    @location(3) lightVec  : vec3f,
    @location(4) shadowCoord : vec4f,
  }

  /* WebGPU NDC is [0,1] in Z and Y-up, so the bias matrix only maps x,y from
     [-1,1] to [0,1]. Z is already in [0,1]. */
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

    out.position = ubo.projection * ubo.view * ubo.model * vec4f(inPos, 1.0);

    let pos = ubo.model * vec4f(inPos, 1.0);
    out.normal   = (ubo.model * vec4f(inNormal, 0.0)).xyz;
    out.lightVec = normalize(ubo.lightPos.xyz - inPos);
    out.viewVec  = -pos.xyz;

    out.shadowCoord = biasMat * ubo.lightSpace * ubo.model * vec4f(inPos, 1.0);
    return out;
  }

  const ambient : f32 = 0.1;
  const SHADOW_MAP_SIZE : i32 = 2048;

  fn textureProjLoad(shadowCoord : vec4f, off : vec2i) -> f32 {
    var shadow : f32 = 1.0;
    let sc = shadowCoord;

    if (sc.z > 0.0 && sc.z < 1.0) {
      let texCoord = vec2i(vec2f(f32(SHADOW_MAP_SIZE)) * sc.xy) + off;
      /* Clamp to valid range */
      let tc = clamp(texCoord, vec2i(0), vec2i(SHADOW_MAP_SIZE - 1));
      let dist = textureLoad(shadowMap, tc, 0).r;
      if (sc.w > 0.0 && dist < sc.z) {
        shadow = ambient;
      }
    }
    return shadow;
  }

  fn filterPCF(sc : vec4f) -> f32 {
    var shadowFactor : f32 = 0.0;
    var count : i32 = 0;

    for (var x : i32 = -1; x <= 1; x++) {
      for (var y : i32 = -1; y <= 1; y++) {
        shadowFactor += textureProjLoad(sc, vec2i(x, y));
        count++;
      }
    }
    return shadowFactor / f32(count);
  }

  @fragment
  fn fs_main(in : VSOutput) -> @location(0) vec4f {
    let sc = in.shadowCoord / in.shadowCoord.w;

    var shadow : f32;
    if (enablePCF == 1u) {
      shadow = filterPCF(sc);
    } else {
      shadow = textureProjLoad(sc, vec2i(0, 0));
    }

    let N = normalize(in.normal);
    let L = normalize(in.lightVec);
    let diffuse = max(dot(N, L), ambient) * in.color.rgb;

    return vec4f(diffuse * shadow, 1.0);
  }
);

/* Debug quad shader: shadow map visualization */
static const char* debug_quad_shader_wgsl = CODE(
  struct SceneUBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
    lightSpace : mat4x4f,
    lightPos   : vec4f,
    zNear      : f32,
    zFar       : f32,
  }

  @group(0) @binding(0) var<uniform> ubo : SceneUBO;
  @group(0) @binding(1) var depthTexture : texture_2d<f32>;

  const SHADOW_MAP_SIZE : i32 = 2048;

  struct VSOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  }

  @vertex
  fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VSOutput {
    var out : VSOutput;
    let u = f32((vertexIndex << 1u) & 2u);
    let v = f32(vertexIndex & 2u);
    out.uv = vec2f(u, v);
    out.position = vec4f(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  fn LinearizeDepth(depth : f32) -> f32 {
    let n : f32 = ubo.zNear;
    let f : f32 = ubo.zFar;
    return (2.0 * n) / (f + n - depth * (f - n));
  }

  @fragment
  fn fs_main(in : VSOutput) -> @location(0) vec4f {
    /* Flip V for WebGPU UV convention */
    let uv = vec2f(in.uv.x, 1.0 - in.uv.y);
    let tc = clamp(vec2i(uv * vec2f(f32(SHADOW_MAP_SIZE))), vec2i(0), vec2i(SHADOW_MAP_SIZE - 1));
    let depth = textureLoad(depthTexture, tc, 0).r;
    let linearDepth = 1.0 - LinearizeDepth(depth);
    return vec4f(linearDepth, linearDepth, linearDepth, 1.0);
  }
);

// clang-format on
