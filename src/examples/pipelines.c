/**
 * @brief Using different pipelines in a single renderpass.
 *
 * Ported from the Vulkan pipelines example. Demonstrates how to use different
 * graphics pipelines within a single render pass. The scene (treasure_smooth
 * glTF model) is rendered three times side-by-side with different shading:
 * phong, toon, and wireframe.
 *
 * @ref
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/pipelines
 */

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

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

#include "core/camera.h"
#include "core/gltf_model.h"

#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations — defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* phong_shader_wgsl;
static const char* toon_shader_wgsl;
static const char* wireframe_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Model */
  gltf_model_t model;
  bool model_loaded;

  /* GPU vertex/index buffers */
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;

  /* Wireframe line-list index buffer */
  WGPUBuffer wireframe_index_buffer;
  uint32_t wireframe_index_count;

  /* Uniform buffer */
  WGPUBuffer uniform_buffer;
  struct {
    mat4 projection;
    mat4 model_view;
    vec4 light_pos;
  } ubo_data;

  /* Bind group layout + bind group */
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  /* Pipeline layout */
  WGPUPipelineLayout pipeline_layout;

  /* Three render pipelines */
  struct {
    WGPURenderPipeline phong;
    WGPURenderPipeline toon;
    WGPURenderPipeline wireframe;
  } pipelines;

  /* Render pass descriptor */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  .ubo_data = {
    /* Vulkan original: lightPos = (0.0, 2.0, 1.0, 0.0)
     * Negate Y for WebGPU: (0.0, -2.0, 1.0, 0.0) */
    .light_pos = {0.0f, -2.0f, 1.0f, 0.0f},
  },
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
};

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_model(void)
{
  gltf_model_load_from_file(&state.model, "assets/models/treasure_smooth.gltf",
                            1.0f);
  state.model_loaded = true;
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  if (!state.model_loaded) {
    return;
  }

  WGPUDevice device = wgpu_context->device;
  gltf_model_t* m   = &state.model;
  size_t vb_size    = m->vertex_count * sizeof(gltf_vertex_t);

  /* Bake node transforms (PreTransformVertices + PreMultiplyVertexColors).
   * Do NOT use FlipY — WebGPU is Y-up. */
  gltf_model_desc_t load_desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
  };

  gltf_vertex_t* xformed = (gltf_vertex_t*)malloc(vb_size);
  memcpy(xformed, m->vertices, vb_size);
  gltf_model_bake_node_transforms(m, xformed, &load_desc);

  /* Upload vertex buffer */
  state.vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label            = STRVIEW("Pipelines Vertex Buffer"),
              .usage            = WGPUBufferUsage_Vertex,
              .size             = vb_size,
              .mappedAtCreation = true,
            });
  void* vdata = wgpuBufferGetMappedRange(state.vertex_buffer, 0, vb_size);
  memcpy(vdata, xformed, vb_size);
  wgpuBufferUnmap(state.vertex_buffer);
  free(xformed);

  /* Upload index buffer */
  if (m->index_count > 0) {
    size_t ib_size     = m->index_count * sizeof(uint32_t);
    state.index_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label            = STRVIEW("Pipelines Index Buffer"),
                .usage            = WGPUBufferUsage_Index,
                .size             = ib_size,
                .mappedAtCreation = true,
              });
    void* idata = wgpuBufferGetMappedRange(state.index_buffer, 0, ib_size);
    memcpy(idata, m->indices, ib_size);
    wgpuBufferUnmap(state.index_buffer);
  }

  /* Build wireframe (line-list) index buffer from triangle indices.
   * For each triangle (i0, i1, i2), emit edges: (i0,i1), (i1,i2), (i2,i0). */
  if (m->index_count > 0) {
    uint32_t tri_count          = m->index_count / 3;
    state.wireframe_index_count = tri_count * 6;
    size_t wf_ib_size = state.wireframe_index_count * sizeof(uint32_t);

    uint32_t* wf_indices = (uint32_t*)malloc(wf_ib_size);
    for (uint32_t t = 0; t < tri_count; ++t) {
      uint32_t i0           = m->indices[t * 3 + 0];
      uint32_t i1           = m->indices[t * 3 + 1];
      uint32_t i2           = m->indices[t * 3 + 2];
      wf_indices[t * 6 + 0] = i0;
      wf_indices[t * 6 + 1] = i1;
      wf_indices[t * 6 + 2] = i1;
      wf_indices[t * 6 + 3] = i2;
      wf_indices[t * 6 + 4] = i2;
      wf_indices[t * 6 + 5] = i0;
    }

    state.wireframe_index_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label            = STRVIEW("Wireframe Index Buffer"),
                .usage            = WGPUBufferUsage_Index,
                .size             = wf_ib_size,
                .mappedAtCreation = true,
              });
    void* wf_data
      = wgpuBufferGetMappedRange(state.wireframe_index_buffer, 0, wf_ib_size);
    memcpy(wf_data, wf_indices, wf_ib_size);
    wgpuBufferUnmap(state.wireframe_index_buffer);
    free(wf_indices);
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("UBO"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.ubo_data),
    });
}

static void update_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  /* Override camera aspect for three side-by-side viewports */
  camera_update_aspect_ratio(&state.camera, ((float)wgpu_context->width / 3.0f)
                                              / (float)wgpu_context->height);

  glm_mat4_copy(state.camera.matrices.perspective, state.ubo_data.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo_data.model_view);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       &state.ubo_data, sizeof(state.ubo_data));
}

/* -------------------------------------------------------------------------- *
 * Bind group layout + bind group
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry entry = {
    .binding    = 0,
    .visibility = WGPUShaderStage_Vertex,
    .buffer = {
      .type           = WGPUBufferBindingType_Uniform,
      .minBindingSize = sizeof(state.ubo_data),
    },
  };
  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("UBO Bind Group Layout"),
                            .entryCount = 1,
                            .entries    = &entry,
                          });
}

static void init_bind_group(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry entry = {
    .binding = 0,
    .buffer  = state.uniform_buffer,
    .size    = sizeof(state.ubo_data),
  };
  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("UBO Bind Group"),
                            .layout     = state.bind_group_layout,
                            .entryCount = 1,
                            .entries    = &entry,
                          });
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static WGPURenderPipeline create_pipeline(struct wgpu_context_t* wgpu_context,
                                          const char* shader_source,
                                          const char* label,
                                          WGPUPrimitiveTopology topology)
{
  WGPUDevice device = wgpu_context->device;

  WGPUShaderModule shader_module
    = wgpu_create_shader_module(device, shader_source);

  /* Vertex buffer layout matching gltf_vertex_t:
   * Position (vec3f), Normal (vec3f), Color (vec4f) */
  WGPUVertexAttribute vertex_attrs[] = {
    {.shaderLocation = 0,
     .offset         = offsetof(gltf_vertex_t, position),
     .format         = WGPUVertexFormat_Float32x3},
    {.shaderLocation = 1,
     .offset         = offsetof(gltf_vertex_t, normal),
     .format         = WGPUVertexFormat_Float32x3},
    {.shaderLocation = 2,
     .offset         = offsetof(gltf_vertex_t, color),
     .format         = WGPUVertexFormat_Float32x4},
  };
  WGPUVertexBufferLayout vb_layout = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = (uint32_t)ARRAY_SIZE(vertex_attrs),
    .attributes     = vertex_attrs,
  };

  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  WGPUDepthStencilState depth_stencil_state = {
    .format               = wgpu_context->depth_stencil_format,
    .depthWriteEnabled    = WGPUOptionalBool_True,
    .depthCompare         = WGPUCompareFunction_LessEqual,
    .stencilFront.compare = WGPUCompareFunction_Always,
    .stencilBack.compare  = WGPUCompareFunction_Always,
  };

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW(label),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &vb_layout,
    },
    .fragment = &(WGPUFragmentState){
      .module      = shader_module,
      .entryPoint  = STRVIEW("fs_main"),
      .targetCount = 1,
      .targets     = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = topology,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    },
    .depthStencil = &depth_stencil_state,
    .multisample  = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  WGPURenderPipeline pipeline
    = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
  ASSERT(pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);
  return pipeline;
}

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  /* Pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = STRVIEW("Pipeline Layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });

  /* Phong shading pipeline */
  state.pipelines.phong
    = create_pipeline(wgpu_context, phong_shader_wgsl, "Phong Pipeline",
                      WGPUPrimitiveTopology_TriangleList);

  /* Toon shading pipeline */
  state.pipelines.toon
    = create_pipeline(wgpu_context, toon_shader_wgsl, "Toon Pipeline",
                      WGPUPrimitiveTopology_TriangleList);

  /* Wireframe pipeline (uses line-list topology) */
  state.pipelines.wireframe
    = create_pipeline(wgpu_context, wireframe_shader_wgsl, "Wireframe Pipeline",
                      WGPUPrimitiveTopology_LineList);
}

/* -------------------------------------------------------------------------- *
 * Draw model
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass)
{
  gltf_model_t* m = &state.model;

  for (uint32_t n = 0; n < m->linear_node_count; ++n) {
    gltf_node_t* node = m->linear_nodes[n];
    if (node->mesh == NULL) {
      continue;
    }
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; ++p) {
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

static void draw_model_wireframe(WGPURenderPassEncoder pass)
{
  /* Wireframe draws the entire model as line-list using the converted
   * index buffer. The wireframe index buffer covers all triangles. */
  if (state.wireframe_index_count > 0) {
    wgpuRenderPassEncoderDrawIndexed(pass, state.wireframe_index_count, 1, 0, 0,
                                     0);
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
  igSetNextWindowSize((ImVec2){200.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Pipelines", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Info", NULL, ImGuiTreeNodeFlags_DefaultOpen)) {
    igText("Left: Phong shading");
    igText("Center: Toon shading");
    igText("Right: Wireframe");
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
    camera_update_aspect_ratio(&state.camera,
                               ((float)wgpu_context->width / 3.0f)
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

  /* Camera: lookat type
   * Vulkan original: position(0, 0, -10.5), rotation(-25, 15, 0)
   * WebGPU: negate pitch → rotation(25, 15, 0) */
  camera_init(&state.camera);
  state.camera.type      = CameraType_LookAt;
  state.camera.invert_dx = true;
  state.camera.invert_dy = true;
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -10.5f});
  camera_set_rotation(&state.camera, (vec3){25.0f, 15.0f, 0.0f});
  camera_set_rotation_speed(&state.camera, 0.5f);
  camera_set_perspective(&state.camera, 60.0f,
                         ((float)wgpu_context->width / 3.0f)
                           / (float)wgpu_context->height,
                         0.1f, 256.0f);

  /* Load model synchronously */
  load_model();

  /* Create GPU buffers */
  create_model_buffers(wgpu_context);

  /* Create uniform buffer */
  init_uniform_buffer(wgpu_context);

  /* Create bind group layout + bind group */
  init_bind_group_layout(wgpu_context);
  init_bind_group(wgpu_context);

  /* Create render pipelines */
  init_pipelines(wgpu_context);

  /* ImGui */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized || !state.model_loaded) {
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

  /* Update camera */
  camera_update(&state.camera, delta_time);

  /* Update uniforms */
  update_uniform_buffer(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* ---- Render ---- */
  WGPUDevice device = wgpu_context->device;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  uint32_t w = (uint32_t)wgpu_context->width;
  uint32_t h = (uint32_t)wgpu_context->height;
  float fw   = (float)w;
  float fh   = (float)h;
  float vp_w = fw / 3.0f;

  /* Bind shared UBO and vertex buffer */
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_group, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);

  /* Left: Phong shading */
  wgpuRenderPassEncoderSetViewport(pass, 0.0f, 0.0f, vp_w, fh, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, (uint32_t)vp_w, h);
  wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.phong);
  if (state.index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  }
  draw_model(pass);

  /* Center: Toon shading */
  wgpuRenderPassEncoderSetViewport(pass, vp_w, 0.0f, vp_w, fh, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(pass, (uint32_t)vp_w, 0, (uint32_t)vp_w,
                                      h);
  wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.toon);
  draw_model(pass);

  /* Right: Wireframe */
  wgpuRenderPassEncoderSetViewport(pass, vp_w * 2.0f, 0.0f, vp_w, fh, 0.0f,
                                   1.0f);
  wgpuRenderPassEncoderSetScissorRect(pass, (uint32_t)(vp_w * 2.0f), 0,
                                      (uint32_t)vp_w, h);
  wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.wireframe);
  if (state.wireframe_index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(pass, state.wireframe_index_buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
  }
  draw_model_wireframe(pass);

  wgpuRenderPassEncoderEnd(pass);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  wgpuRenderPassEncoderRelease(pass);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* ImGui overlay render */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Destroy model CPU data */
  gltf_model_destroy(&state.model);

  /* Release GPU buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.wireframe_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.phong)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.toon)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.wireframe)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Pipeline State Objects",
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

/* Phong shading shader */
static const char* phong_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    modelView  : mat4x4f,
    lightPos   : vec4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position  : vec4f,
    @location(0)       normal    : vec3f,
    @location(1)       color     : vec3f,
    @location(2)       viewVec   : vec3f,
    @location(3)       lightVec  : vec3f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.color    = in.color.rgb;
    out.position = ubo.projection * ubo.modelView * vec4f(in.position, 1.0);

    let pos      = ubo.modelView * vec4f(in.position, 1.0);
    out.normal   = (ubo.modelView * vec4f(in.normal, 0.0)).xyz;
    let lPos     = (ubo.modelView * vec4f(ubo.lightPos.xyz, 0.0)).xyz;
    out.lightVec = lPos - pos.xyz;
    out.viewVec  = -pos.xyz;
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    // Desaturate color
    let color = mix(in.color, vec3f(dot(vec3f(0.2126, 0.7152, 0.0722), in.color)), 0.65);

    // High ambient colors because mesh materials are pretty dark
    let ambient = color * vec3f(1.0);
    let N = normalize(in.normal);
    let L = normalize(in.lightVec);
    let V = normalize(in.viewVec);
    let R = reflect(-L, N);
    let diffuse  = max(dot(N, L), 0.0) * color;
    let specular = pow(max(dot(R, V), 0.0), 32.0) * vec3f(0.35);
    return vec4f(ambient + diffuse * 1.75 + specular, 1.0);
  }
);

/* Toon shading shader */
static const char* toon_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    modelView  : mat4x4f,
    lightPos   : vec4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position  : vec4f,
    @location(0)       normal    : vec3f,
    @location(1)       color     : vec3f,
    @location(2)       viewVec   : vec3f,
    @location(3)       lightVec  : vec3f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.color    = in.color.rgb;
    out.position = ubo.projection * ubo.modelView * vec4f(in.position, 1.0);

    let pos      = ubo.modelView * vec4f(in.position, 1.0);
    out.normal   = (ubo.modelView * vec4f(in.normal, 0.0)).xyz;
    let lPos     = (ubo.modelView * vec4f(ubo.lightPos.xyz, 0.0)).xyz;
    out.lightVec = lPos - pos.xyz;
    out.viewVec  = -pos.xyz;
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let N = normalize(in.normal);
    let L = normalize(in.lightVec);

    let intensity = dot(N, L);
    var shade : f32 = 1.0;
    if (intensity < 0.5) { shade = 0.75; }
    if (intensity < 0.35) { shade = 0.6; }
    if (intensity < 0.25) { shade = 0.5; }
    if (intensity < 0.1) { shade = 0.25; }

    return vec4f(in.color * 3.0 * shade, 1.0);
  }
);

/* Wireframe shader */
static const char* wireframe_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    modelView  : mat4x4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       color    : vec3f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.color    = in.color.rgb;
    out.position = ubo.projection * ubo.modelView * vec4f(in.position, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color * 1.5, 1.0);
  }
);

// clang-format on
