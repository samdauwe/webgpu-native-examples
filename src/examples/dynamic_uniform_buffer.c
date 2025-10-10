#include "webgpu/wgpu_common.h"

#include "core/camera.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <stdbool.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Dynamic Uniform Buffers
 *
 * Dynamic buffer offset can largely improve performance of the application.
 * Comparing to creating many binding goups and set every group for each object,
 * only one binding group will be created and buffer offset is dynamically set.
 *
 * Ref:
 * https://github.com/gpuweb/gpuweb/issues/116
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/dynamicuniformbuffer
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Dynamic Uniform Buffers example
 * -------------------------------------------------------------------------- */

#define OBJECT_INSTANCES (125u)
#define ALIGNMENT (256u) /* 256-byte alignment */

/* Vertex layout for this example */
typedef struct {
  vec3 pos;
  vec3 color;
} vertex_t;

/* State struct */
static struct {
  wgpu_buffer_t vertices;
  wgpu_buffer_t indices;
  struct {
    struct wgpu_buffer_t view;
    struct {
      WGPUBuffer buffer;
      uint64_t buffer_size;
      uint64_t model_size;
    } dynamic;
  } uniform_buffers;
  struct {
    mat4 projection_matrix;
    mat4 view_matrix;
  } ubo_vs;
  vec3 rotations[OBJECT_INSTANCES];
  vec3 rotation_speeds[OBJECT_INSTANCES];
  struct {
    mat4 model;
    uint8_t padding[192];
  } ubo_data_dynamic[OBJECT_INSTANCES];
  camera_t camera;
  WGPUBool view_updated;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPURenderBundle render_bundle;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  bool render_bundles;
  float animation_timer;
  float prev_time;
  bool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1, 0.2, 0.3, 1.0},
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
  .render_bundles  = true,
  .animation_timer = 0.0f,
};

static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type = CameraType_LookAt;
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -30.0f});
  camera_set_rotation(&state.camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_rotation_speed(&state.camera, 0.25f);
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);
}

static void generate_cube(wgpu_context_t* wgpu_context)
{
  /* Setup vertices for a colored cube */
  vertex_t vertex_buffer[8] = {
    // clang-format off
    {.pos = {-1.0f, -1.0f,  1.0f}, .color = {1.0f, 0.0f, 0.0f}},
    {.pos = { 1.0f, -1.0f,  1.0f}, .color = {0.0f, 1.0f, 0.0f}},
    {.pos = { 1.0f,  1.0f,  1.0f}, .color = {0.0f, 0.0f, 1.0f}},
    {.pos = {-1.0f,  1.0f,  1.0f}, .color = {0.0f, 0.0f, 0.0f}},
    {.pos = {-1.0f, -1.0f, -1.0f}, .color = {1.0f, 0.0f, 0.0f}},
    {.pos = { 1.0f, -1.0f, -1.0f}, .color = {0.0f, 1.0f, 0.0f}},
    {.pos = { 1.0f,  1.0f, -1.0f}, .color = {0.0f, 0.0f, 1.0f}},
    {.pos = {-1.0f,  1.0f, -1.0f}, .color = {0.0f, 0.0f, 0.0f}},
    // clang-format on
  };

  /* Create vertex buffer */
  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_buffer),
                    .count = (uint32_t)ARRAY_SIZE(vertex_buffer),
                    .initial.data = vertex_buffer,
                  });

  /* Setup indices for a colored cube */
  uint32_t index_buffer[36] = {
    0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7,
    4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3,
  };

  /* Create index buffer */
  state.indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(index_buffer),
                    .count = (uint32_t)ARRAY_SIZE(index_buffer),
                    .initial.data = index_buffer,
                  });
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : Projection/View matrix uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = state.uniform_buffers.view.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1 : Instance matrix as dynamic uniform buffer */
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = true,
        .minBindingSize   = (uint64_t)state.uniform_buffers.dynamic.model_size,
      },
      .sampler = {0},
    }
  };
  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Render - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.bind_group_layout != NULL);

  /* Create the pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Render - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Bind Group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Projection/View matrix uniform buffer */
      .binding = 0,
      .buffer  = state.uniform_buffers.view.buffer,
      .offset  = 0,
      .size    = state.uniform_buffers.view.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1 : Instance matrix as dynamic uniform buffer */
      .binding = 1,
      .buffer  = state.uniform_buffers.dynamic.buffer,
      .offset  = 0,
      .size    = state.uniform_buffers.dynamic.model_size,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Render - Bind group"),
    .layout     = state.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.bind_group != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    dyn_ubo, sizeof(vertex_t),
    /* Attribute location 0 : Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    /* Attribute location 1: Color */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, color)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Dynamic uniform buffer - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &dyn_ubo_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

#define RECORD_RENDER_PASS(Type, rpass_enc)                                    \
  if (rpass_enc) {                                                             \
    wgpu##Type##SetPipeline(rpass_enc, state.pipeline);                        \
    wgpu##Type##SetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,        \
                                WGPU_WHOLE_SIZE);                              \
    wgpu##Type##SetIndexBuffer(rpass_enc, state.indices.buffer,                \
                               WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);    \
    /* Render multiple objects using different model matrices by dynamically   \
     * offsetting into one uniform buffer */                                   \
    for (uint32_t i = 0; i < OBJECT_INSTANCES; ++i) {                          \
      /* One dynamic offset per dynamic bind group to offset into the ubo      \
       * containing all model matrices*/                                       \
      uint32_t dynamic_offset = i * ALIGNMENT;                                 \
      /* Bind the bind group for rendering a mesh using the dynamic offset */  \
      wgpu##Type##SetBindGroup(rpass_enc, 0, state.bind_group, 1,              \
                               &dynamic_offset);                               \
      wgpu##Type##DrawIndexed(rpass_enc, state.indices.count, 1, 0, 0, 0);     \
    }                                                                          \
  }

static void init_render_bundle_encoder(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat color_formats[1] = {wgpu_context->render_format};
  WGPURenderBundleEncoder render_bundle_encoder
    = wgpuDeviceCreateRenderBundleEncoder(
      wgpu_context->device,
      &(WGPURenderBundleEncoderDescriptor){
        .label = STRVIEW("Dynamic uniform buffer - Render bundle encoder"),
        .colorFormatCount   = (uint32_t)ARRAY_SIZE(color_formats),
        .colorFormats       = color_formats,
        .depthStencilFormat = wgpu_context->depth_stencil_format,
        .sampleCount        = 1,
      });
  RECORD_RENDER_PASS(RenderBundleEncoder, render_bundle_encoder)
  state.render_bundle
    = wgpuRenderBundleEncoderFinish(render_bundle_encoder, NULL);
  ASSERT(state.render_bundle != NULL);

  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, render_bundle_encoder)
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Fixed ubo with projection and view matrices */
  camera_t* camera = &state.camera;
  glm_mat4_copy(camera->matrices.perspective, state.ubo_vs.projection_matrix);
  glm_mat4_copy(camera->matrices.view, state.ubo_vs.view_matrix);

  /* Map uniform buffer and update it */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.view.buffer,
                       0, &state.ubo_vs, state.uniform_buffers.view.size);
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void update_dynamic_uniform_buffer(wgpu_context_t* wgpu_context,
                                          bool force)
{
  // Update at max. 60 fps
  const float now        = stm_sec(stm_now());
  const float frame_time = now - state.prev_time;
  state.prev_time        = now;
  state.animation_timer += frame_time;
  if ((state.animation_timer <= 1.0f / 60.0f) && (!force)) {
    return;
  }

  // Dynamic ubo with per-object model matrices indexed by offsets in the
  // command buffer
  const uint32_t dim = (uint32_t)(pow(OBJECT_INSTANCES, (1.0f / 3.0f)));
  const vec3 offset  = {5.0f, 5.0f, 5.0f};

  vec3 rotation_speed_scaled = GLM_VEC3_ZERO_INIT;
  for (uint32_t x = 0; x < dim; ++x) {
    for (uint32_t y = 0; y < dim; ++y) {
      for (uint32_t z = 0; z < dim; ++z) {
        uint32_t index = x * dim * dim + y * dim + z;

        // Model
        mat4* modelMat = &state.ubo_data_dynamic[index].model;

        // Update rotations
        glm_vec3_scale(state.rotation_speeds[index], state.animation_timer,
                       rotation_speed_scaled);
        glm_vec3_add(state.rotations[index], rotation_speed_scaled,
                     state.rotations[index]);

        // Update matrices
        vec3 pos = {
          -((dim * offset[0]) / 2.0f) + offset[0] / 2.0f + x * offset[0],
          -((dim * offset[1]) / 2.0f) + offset[1] / 2.0f + y * offset[1],
          -((dim * offset[2]) / 2.0f) + offset[2] / 2.0f + z * offset[2],
        };
        glm_mat4_identity(*modelMat);
        glm_translate(*modelMat, pos);
        glm_rotate(*modelMat, state.rotations[index][0],
                   (vec3){1.0f, 1.0f, 0.0f});
        glm_rotate(*modelMat, state.rotations[index][1],
                   (vec3){0.0f, 1.0f, 0.0f});
        glm_rotate(*modelMat, state.rotations[index][2],
                   (vec3){0.0f, 0.0f, 1.0f});
      }
    }
  }

  state.animation_timer = 0.0f;

  /* Update buffer */
  wgpuQueueWriteBuffer(
    wgpu_context->queue, state.uniform_buffers.dynamic.buffer, 0,
    &state.ubo_data_dynamic, state.uniform_buffers.dynamic.buffer_size);
}

static uint64_t calc_constant_buffer_byte_size(uint64_t byte_size)
{
  return (byte_size + 255) & ~255;
}

/* Initialize uniform buffer containing shader uniforms */
static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Vertex shader uniform buffer block */

  /* Static shared uniform buffer object with projection and view matrix */
  state.uniform_buffers.view = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(state.ubo_vs),
                  });

  /* Uniform buffer object with per-object matrices */
  state.uniform_buffers.dynamic.model_size = sizeof(mat4);
  state.uniform_buffers.dynamic.buffer_size
    = calc_constant_buffer_byte_size(sizeof(state.ubo_data_dynamic));
  state.uniform_buffers.dynamic.buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = state.uniform_buffers.dynamic.buffer_size,
    });

  /* Prepare per-object matrices with offsets and random rotations */
  for (uint32_t i = 0; i < OBJECT_INSTANCES; ++i) {
    glm_vec3_copy(
      (vec3){
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
      },
      state.rotations[i]);
    glm_vec3_scale(state.rotations[i], PI2, state.rotations[i]);
    glm_vec3_copy(
      (vec3){
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
      },
      state.rotation_speeds[i]);
  }

  update_uniform_buffers(wgpu_context);
  update_dynamic_uniform_buffer(wgpu_context, true);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_camera(wgpu_context);
    generate_cube(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_pipeline(wgpu_context);
    init_bind_groups(wgpu_context);
    init_render_bundle_encoder(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  UNUSED_VAR(wgpu_context);

  camera_on_input_event(&state.camera, input_event);
  state.view_updated = true;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update uniform buffers */
  update_dynamic_uniform_buffer(wgpu_context, false);
  if (state.view_updated) {
    update_uniform_buffers(wgpu_context);
    state.view_updated = false;
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  if (state.render_bundles) {
    wgpuRenderPassEncoderExecuteBundles(rpass_enc, 1, &state.render_bundle);
  }
  else {
    RECORD_RENDER_PASS(RenderPassEncoder, rpass_enc)
  }

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.view.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.dynamic.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
  WGPU_RELEASE_RESOURCE(RenderBundle, state.render_bundle)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Dynamic Uniform Buffers",
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
static const char* vertex_shader_wgsl = CODE(
  struct UboView {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
  };

  struct UboInstance {
    model : mat4x4<f32>,
  };

  @group(0) @binding(0) var<uniform> uboView : UboView;
  @group(0) @binding(1) var<uniform> uboInstance : UboInstance;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>,
  };

  @vertex
  fn main(
    @location(0) inPos : vec3<f32>,
    @location(1) inColor : vec3<f32>
  ) -> Output {
    var output : Output;
    output.color = inColor;
    let modelView : mat4x4<f32> = uboView.view * uboInstance.model;
    let worldPos : vec3<f32> = (modelView * vec4<f32>(inPos, 1.0)).xyz;
    output.position = uboView.projection * modelView * vec4(inPos.xyz, 1.0);
    return output;
  }
);

static const char* fragment_shader_wgsl = CODE(
  @fragment
  fn main(
    @location(0) inColor : vec3<f32>
  ) -> @location(0) vec4<f32> {
    return vec4(inColor, 1.0);
  }
);
// clang-format on
