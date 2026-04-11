#include "webgpu/text_overlay.h"
#include "webgpu/wgpu_common.h"

#include "core/camera.h"
#include "core/gltf_model.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <stdbool.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Text Overlay
 *
 * Renders a torus knot glTF model with basic lighting, with a custom text
 * overlay on top using a bitmap font (STB font consolas 24).
 *
 * Ported from Sascha Willems' Vulkan example "textoverlay"
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/textoverlay
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader (declared as variable, defined at end of file)
 * -------------------------------------------------------------------------- */

static const char* mesh_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

static const char* model_path = "assets/models/torusknot.gltf";

/* -------------------------------------------------------------------------- *
 * Uniform data (must match WGSL layout)
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 projection;
  mat4 model_view;
  vec4 light_pos;
} uniform_data_t;

/* -------------------------------------------------------------------------- *
 * Global state
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Model */
  gltf_model_t model;
  bool model_loaded;

  struct {
    WGPUBuffer vertex;
    WGPUBuffer index;
  } model_buffers;

  /* Uniform buffer */
  WGPUBuffer uniform_buffer;
  uniform_data_t ubo;

  /* Depth texture */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth;

  /* Bind group & layout */
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  /* Pipeline */
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Text overlay */
  text_overlay_t* text_overlay;
  bool text_visible;

  /* Timing */
  float frame_timer;
  float fps;
  uint64_t last_time;
  uint32_t frame_count;
  float fps_timer;

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.2f, 1.0f},
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
  .text_visible = true,
};

/* -------------------------------------------------------------------------- *
 * Depth texture management
 * -------------------------------------------------------------------------- */

static void init_depth_texture(struct wgpu_context_t* wgpu_context)
{
  /* Release previous depth texture */
  if (state.depth.view) {
    wgpuTextureViewRelease(state.depth.view);
    state.depth.view = NULL;
  }
  if (state.depth.texture) {
    wgpuTextureDestroy(state.depth.texture);
    wgpuTextureRelease(state.depth.texture);
    state.depth.texture = NULL;
  }

  state.depth.texture = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = STRVIEW("Depth texture"),
                            .usage         = WGPUTextureUsage_RenderAttachment,
                            .dimension     = WGPUTextureDimension_2D,
                            .size          = {(uint32_t)wgpu_context->width,
                                              (uint32_t)wgpu_context->height, 1},
                            .format        = WGPUTextureFormat_Depth24Plus,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                          });
  state.depth.view = wgpuTextureCreateView(state.depth.texture, NULL);
}

/* -------------------------------------------------------------------------- *
 * Model loading and GPU buffer creation
 * -------------------------------------------------------------------------- */

static void load_model(void)
{
  gltf_model_desc_t desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
  };

  state.model_loaded
    = gltf_model_load_from_file_ext(&state.model, model_path, 1.0f, &desc);
  if (!state.model_loaded) {
    printf("Failed to load model: %s\n", model_path);
  }
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  if (!state.model_loaded) {
    return;
  }

  WGPUDevice device = wgpu_context->device;

  /* Vertex buffer */
  {
    uint32_t vb_size
      = state.model.vertex_count * (uint32_t)sizeof(gltf_vertex_t);
    state.model_buffers.vertex = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Model vertex buffer"),
                .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                .size  = vb_size,
                .mappedAtCreation = false,
              });
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model_buffers.vertex, 0,
                         state.model.vertices, vb_size);
  }

  /* Index buffer */
  {
    uint32_t ib_size = state.model.index_count * (uint32_t)sizeof(uint32_t);
    state.model_buffers.index = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Model index buffer"),
                .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                .size  = ib_size,
                .mappedAtCreation = false,
              });
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model_buffers.index, 0,
                         state.model.indices, ib_size);
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
      .label            = STRVIEW("Uniform buffer"),
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = sizeof(uniform_data_t),
      .mappedAtCreation = false,
    });
}

static void update_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  camera_update(&state.camera, state.frame_timer);

  glm_mat4_copy(state.camera.matrices.perspective, state.ubo.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo.model_view);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0, &state.ubo,
                       sizeof(uniform_data_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group & pipeline
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry entry = {
    .binding    = 0,
    .visibility = WGPUShaderStage_Vertex,
    .buffer     = (WGPUBufferBindingLayout){
      .type           = WGPUBufferBindingType_Uniform,
      .minBindingSize = sizeof(uniform_data_t),
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Mesh bind group layout"),
                            .entryCount = 1,
                            .entries    = &entry,
                          });
}

static void init_bind_group(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry entry = {
    .binding = 0,
    .buffer  = state.uniform_buffer,
    .offset  = 0,
    .size    = sizeof(uniform_data_t),
  };

  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Mesh bind group"),
                            .layout     = state.bind_group_layout,
                            .entryCount = 1,
                            .entries    = &entry,
                          });
}

static void init_pipeline(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Mesh pipeline layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.bind_group_layout,
            });

  /* Shader module */
  WGPUShaderModule shader = wgpu_create_shader_module(device, mesh_shader_wgsl);

  /* Vertex attributes: position, normal, uv0 */
  WGPUVertexAttribute attrs[3] = {
    [0] = {
      .shaderLocation = 0,
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, position),
    },
    [1] = {
      .shaderLocation = 1,
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, normal),
    },
    [2] = {
      .shaderLocation = 2,
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = offsetof(gltf_vertex_t, uv0),
    },
  };

  WGPUVertexBufferLayout vb_layout = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = ARRAY_SIZE(attrs),
    .attributes     = attrs,
  };

  WGPUBlendState blend        = wgpu_create_blend_state(false);
  WGPUColorTargetState target = {
    .format    = wgpu_context->render_format,
    .blend     = &blend,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUDepthStencilState depth_stencil = {
    .format            = WGPUTextureFormat_Depth24Plus,
    .depthWriteEnabled = WGPUOptionalBool_True,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
  };

  state.pipeline = wgpuDeviceCreateRenderPipeline(
    device, &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Mesh pipeline"),
      .layout = state.pipeline_layout,
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

/* -------------------------------------------------------------------------- *
 * Draw model helper
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass, gltf_model_t* model,
                       WGPUBuffer vb, WGPUBuffer ib)
{
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vb, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);

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
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Text overlay update
 * -------------------------------------------------------------------------- */

static void update_text_overlay(struct wgpu_context_t* wgpu_context)
{
  if (!state.text_overlay) {
    return;
  }

  text_overlay_begin_text_update(state.text_overlay);

  /* Title */
  text_overlay_add_text(state.text_overlay, "Text overlay", 5.0f, 5.0f,
                        TextOverlay_Text_AlignLeft);

  /* Frame timing */
  text_overlay_add_formatted_text(
    state.text_overlay, 5.0f, 25.0f, TextOverlay_Text_AlignLeft,
    "%.2f ms (%.0f fps)", state.frame_timer * 1000.0f, state.fps);

  /* Device name */
  text_overlay_add_text(state.text_overlay, wgpu_context->platform_info.device,
                        5.0f, 45.0f, TextOverlay_Text_AlignLeft);

  /* Model-view matrix display (right-aligned) */
  float w = (float)wgpu_context->width;
  text_overlay_add_text(state.text_overlay, "model view matrix", w - 5.0f, 5.0f,
                        TextOverlay_Text_AlignRight);

  for (uint32_t i = 0; i < 4; i++) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%+.2f %+.2f %+.2f %+.2f",
             state.ubo.model_view[0][i], state.ubo.model_view[1][i],
             state.ubo.model_view[2][i], state.ubo.model_view[3][i]);
    text_overlay_add_text(state.text_overlay, buf, w - 5.0f,
                          25.0f + (float)i * 20.0f,
                          TextOverlay_Text_AlignRight);
  }

  /* Project world origin to screen: "A torus knot" label */
  {
    mat4 mvp;
    glm_mat4_mul(state.ubo.projection, state.ubo.model_view, mvp);
    float h = (float)wgpu_context->height;
    vec4 vp = {0.0f, 0.0f, w, h};
    vec3 projected;
    glm_project((vec3){0.0f, 0.0f, 0.0f}, mvp, vp, projected);
    /* glm_project returns OpenGL screen-space (Y=0 at bottom).
     * text_overlay_add_text expects pixel-space (Y=0 at top), so flip Y. */
    text_overlay_add_text(state.text_overlay, "A torus knot", projected[0],
                          h - projected[1], TextOverlay_Text_AlignCenter);
  }

  text_overlay_add_text(state.text_overlay,
                        "Press \"space\" to toggle text overlay", 5.0f, 65.0f,
                        TextOverlay_Text_AlignLeft);
  text_overlay_add_text(state.text_overlay,
                        "Hold middle mouse button and drag to move", 5.0f,
                        85.0f, TextOverlay_Text_AlignLeft);

  text_overlay_end_text_update(state.text_overlay);
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
    camera_set_perspective(
      &state.camera, 60.0f,
      (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_CHAR
           && input_event->char_code == (uint32_t)' ') {
    state.text_visible = !state.text_visible;
  }
  else {
    /* Camera input: negate mouse_dy to match Vulkan mouse behavior */
    if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE
        && input_event->mouse_btn_pressed
        && input_event->mouse_button == BUTTON_LEFT) {
      camera_rotate(&state.camera,
                    (vec3){-input_event->mouse_dy * state.camera.rotation_speed,
                           input_event->mouse_dx * state.camera.rotation_speed,
                           0.0f});
      return;
    }
    camera_on_input_event(&state.camera, input_event);
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  stm_setup();

  /* Camera: Vulkan values were pos(0, 0, -2.5), rot(-25, 0, 0), lookat */
  camera_init(&state.camera);
  state.camera.type      = CameraType_LookAt;
  state.camera.invert_dx = true;
  state.camera.invert_dy = true;
  camera_set_position(&state.camera, (vec3)VKY_TO_WGPU_VEC3(0.0f, 0.0f, -2.5f));
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-25.0f, 0.0f, 0.0f));
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Light position */
  state.ubo.light_pos[0] = 0.0f;
  state.ubo.light_pos[1] = 0.0f;
  state.ubo.light_pos[2] = 0.0f;
  state.ubo.light_pos[3] = 1.0f;

  /* Load models synchronously */
  load_model();
  create_model_buffers(wgpu_context);

  /* Init GPU resources */
  init_depth_texture(wgpu_context);
  init_uniform_buffer(wgpu_context);
  init_bind_group_layout(wgpu_context);
  init_bind_group(wgpu_context);
  init_pipeline(wgpu_context);

  /* Create text overlay */
  state.text_overlay = text_overlay_create(wgpu_context);

  state.last_time   = stm_now();
  state.initialized = true;

  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized || !state.model_loaded) {
    return EXIT_SUCCESS;
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;
  UNUSED_VAR(queue);

  /* Timing */
  uint64_t now      = stm_now();
  state.frame_timer = (float)stm_sec(stm_diff(now, state.last_time));
  state.last_time   = now;

  /* FPS counter */
  state.frame_count++;
  state.fps_timer += state.frame_timer;
  if (state.fps_timer >= 1.0f) {
    state.fps         = (float)state.frame_count / state.fps_timer;
    state.frame_count = 0;
    state.fps_timer   = 0.0f;
  }

  /* Update uniforms */
  update_uniform_buffer(wgpu_context);

  /* Update text overlay */
  if (state.text_visible) {
    update_text_overlay(wgpu_context);
  }

  /* Begin main render pass */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth.view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Draw the mesh */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group, 0, 0);
  draw_model(rpass_enc, &state.model, state.model_buffers.vertex,
             state.model_buffers.index);

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buffer);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc);

  /* Draw text overlay on top (separate render pass, loads existing content) */
  if (state.text_visible) {
    text_overlay_draw_frame(state.text_overlay, wgpu_context->swapchain_view);
  }

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  if (state.text_overlay) {
    text_overlay_release(state.text_overlay);
    state.text_overlay = NULL;
  }

  /* Depth texture */
  if (state.depth.view) {
    wgpuTextureViewRelease(state.depth.view);
  }
  if (state.depth.texture) {
    wgpuTextureDestroy(state.depth.texture);
    wgpuTextureRelease(state.depth.texture);
  }

  /* GPU resources */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout);
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.model_buffers.vertex);
  WGPU_RELEASE_RESOURCE(Buffer, state.model_buffers.index);

  gltf_model_destroy(&state.model);
}

/* -------------------------------------------------------------------------- *
 * Main entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Text overlay",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* mesh_shader_wgsl = CODE(
  struct Uniforms {
    projection: mat4x4f,
    modelView: mat4x4f,
    lightPos: vec4f,
  }

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) color: vec3f,
    @location(2) eyePos: vec3f,
    @location(3) lightVec: vec3f,
  }

  @vertex
  fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.projection * uniforms.modelView * vec4f(in.position, 1.0);
    let eyePos = (uniforms.modelView * vec4f(in.position, 1.0)).xyz;
    out.eyePos = eyePos;
    out.normal = (uniforms.modelView * vec4f(in.normal, 0.0)).xyz;
    out.lightVec = normalize(uniforms.lightPos.xyz - eyePos);
    out.color = vec3f(1.0, 0.7, 0.1);
    return out;
  }

  @fragment
  fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let N = normalize(in.normal);
    let L = normalize(in.lightVec);
    let V = normalize(-in.eyePos);
    let R = reflect(-L, N);
    let diffuse = max(dot(N, L), 0.15);
    let specular = pow(max(dot(R, V), 0.0), 16.0) * 0.75;
    let color = in.color * diffuse + vec3f(specular);
    return vec4f(color, 1.0);
  }
);
// clang-format on
