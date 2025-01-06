#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

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

#define OBJECT_INSTANCES 125u
#define ALIGNMENT 256u // 256-byte alignment

// Vertex layout for this example
typedef struct {
  vec3 pos;
  vec3 color;
} vertex_t;

// Vertex buffer and attributes
static wgpu_buffer_t vertices = {0};

// Index buffer
static wgpu_buffer_t indices = {0};

static struct {
  struct wgpu_buffer_t view;
  struct {
    WGPUBuffer buffer;
    uint64_t buffer_size;
    uint64_t model_size;
  } dynamic;
} uniform_buffers = {0};

static struct {
  mat4 projection_matrix;
  mat4 view_matrix;
} ubo_vs = {0};

// Store random per-object rotations
static vec3 rotations[OBJECT_INSTANCES]       = {0};
static vec3 rotation_speeds[OBJECT_INSTANCES] = {0};

// One big uniform buffer that contains all matrices
static struct {
  mat4 model;
  uint8_t padding[192];
} ubo_data_dynamic[OBJECT_INSTANCES] = {0};

// Pipeline
static WGPUPipelineLayout pipeline_layout = NULL;
static WGPURenderPipeline pipeline        = NULL;

// Bindings
static WGPUBindGroupLayout bind_group_layout = NULL;
static WGPUBindGroup bind_group              = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Render bundle
static WGPURenderBundle render_bundle = NULL;

// Render bundle setting & animation timer
static bool render_bundles   = true;
static float animation_timer = 0.0f;

// Other variables
static const char* example_title = "Dynamic Uniform Buffers";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -30.0f});
  camera_set_rotation(context->camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_rotation_speed(context->camera, 0.25f);
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
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
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube vertex buffer",
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
  indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(index_buffer),
                    .count = (uint32_t)ARRAY_SIZE(index_buffer),
                    .initial.data = index_buffer,
                  });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
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
        .minBindingSize   = uniform_buffers.view.size,
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
        .minBindingSize   = (uint64_t)uniform_buffers.dynamic.model_size,
      },
      .sampler = {0},
    }
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL);

  /* Create the pipeline layout */
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Render - Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Bind Group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Projection/View matrix uniform buffer */
      .binding = 0,
      .buffer  = uniform_buffers.view.buffer,
      .offset  = 0,
      .size    = uniform_buffers.view.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1 : Instance matrix as dynamic uniform buffer */
      .binding = 1,
      .buffer  = uniform_buffers.dynamic.buffer,
      .offset  = 0,
      .size    = uniform_buffers.dynamic.model_size,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Bind group",
    .layout     = bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(bind_group != NULL);
}

static void prepare_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    dyn_ubo, sizeof(vertex_t),
    // Attribute location 0 : Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    // Attribute location 1: Color
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, color)))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label            = "Vertex shader WGSL",
                  .wgsl_code.source = vertex_shader_wgsl,
                  .entry            = "main",
                },
                .buffer_count = 1,
                .buffers      = &dyn_ubo_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .label            = "Fragment shader WGSL",
                  .wgsl_code.source = fragment_shader_wgsl,
                  .entry            = "main",
                },
                .target_count = 1,
                .targets      = &color_target_state,
              });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label = "Dynamic uniform buffer - Render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Attachment is acquired in render loop */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.1f,
        .g = 0.2f,
        .b = 0.3f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

#define RECORD_RENDER_PASS(Type, rpass_enc)                                    \
  if (rpass_enc) {                                                             \
    wgpu##Type##SetPipeline(rpass_enc, pipeline);                              \
    wgpu##Type##SetVertexBuffer(rpass_enc, 0, vertices.buffer, 0,              \
                                WGPU_WHOLE_SIZE);                              \
    wgpu##Type##SetIndexBuffer(rpass_enc, indices.buffer,                      \
                               WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);    \
    /* Render multiple objects using different model matrices by dynamically   \
     * offsetting into one uniform buffer */                                   \
    for (uint32_t i = 0; i < OBJECT_INSTANCES; ++i) {                          \
      /* One dynamic offset per dynamic bind group to offset into the ubo      \
       * containing all model matrices*/                                       \
      uint32_t dynamic_offset = i * ALIGNMENT;                                 \
      /* Bind the bind group for rendering a mesh using the dynamic offset */  \
      wgpu##Type##SetBindGroup(rpass_enc, 0, bind_group, 1, &dynamic_offset);  \
      wgpu##Type##DrawIndexed(rpass_enc, indices.count, 1, 0, 0, 0);           \
    }                                                                          \
  }

static void prepare_render_bundle_encoder(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat color_formats[1] = {wgpu_context->swap_chain.format};
  WGPURenderBundleEncoder render_bundle_encoder
    = wgpuDeviceCreateRenderBundleEncoder(
      wgpu_context->device,
      &(WGPURenderBundleEncoderDescriptor){
        .label              = "Dynamic uniform buffer - Render bundle encoder",
        .colorFormatCount   = (uint32_t)ARRAY_SIZE(color_formats),
        .colorFormats       = color_formats,
        .depthStencilFormat = WGPUTextureFormat_Depth24PlusStencil8,
        .sampleCount        = 1,
      });
  RECORD_RENDER_PASS(RenderBundleEncoder, render_bundle_encoder)
  render_bundle = wgpuRenderBundleEncoderFinish(render_bundle_encoder, NULL);

  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, render_bundle_encoder)
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Fixed ubo with projection and view matrices
  camera_t* camera = context->camera;
  glm_mat4_copy(camera->matrices.perspective, ubo_vs.projection_matrix);
  glm_mat4_copy(camera->matrices.view, ubo_vs.view_matrix);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.view.buffer, 0,
                          &ubo_vs, uniform_buffers.view.size);
}

// Prepare and initialize uniform buffer containing shader uniforms
static void update_dynamic_uniform_buffer(wgpu_example_context_t* context,
                                          bool force)
{
  // Update at max. 60 fps
  animation_timer += context->frame_timer;
  if ((animation_timer <= 1.0f / 60.0f) && (!force)) {
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
        mat4* modelMat = &ubo_data_dynamic[index].model;

        // Update rotations
        glm_vec3_scale(rotation_speeds[index], animation_timer,
                       rotation_speed_scaled);
        glm_vec3_add(rotations[index], rotation_speed_scaled, rotations[index]);

        // Update matrices
        vec3 pos = {
          -((dim * offset[0]) / 2.0f) + offset[0] / 2.0f + x * offset[0],
          -((dim * offset[1]) / 2.0f) + offset[1] / 2.0f + y * offset[1],
          -((dim * offset[2]) / 2.0f) + offset[2] / 2.0f + z * offset[2],
        };
        glm_mat4_identity(*modelMat);
        glm_translate(*modelMat, pos);
        glm_rotate(*modelMat, rotations[index][0], (vec3){1.0f, 1.0f, 0.0f});
        glm_rotate(*modelMat, rotations[index][1], (vec3){0.0f, 1.0f, 0.0f});
        glm_rotate(*modelMat, rotations[index][2], (vec3){0.0f, 0.0f, 1.0f});
      }
    }
  }

  animation_timer = 0.0f;

  // Update buffer
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.dynamic.buffer,
                          0, &ubo_data_dynamic,
                          uniform_buffers.dynamic.buffer_size);
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Vertex shader uniform buffer block

  // Static shared uniform buffer object with projection and view matrix
  uniform_buffers.view = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(ubo_vs),
    });

  // Uniform buffer object with per-object matrices
  uniform_buffers.dynamic.model_size = sizeof(mat4);
  uniform_buffers.dynamic.buffer_size
    = calc_constant_buffer_byte_size(sizeof(ubo_data_dynamic));
  uniform_buffers.dynamic.buffer = wgpuDeviceCreateBuffer(
    context->wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = uniform_buffers.dynamic.buffer_size,
    });

  // Prepare per-object matrices with offsets and random rotations
  for (uint32_t i = 0; i < OBJECT_INSTANCES; ++i) {
    glm_vec3_copy(
      (vec3){
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
      },
      rotations[i]);
    glm_vec3_scale(rotations[i], PI2, rotations[i]);
    glm_vec3_copy(
      (vec3){
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
      },
      rotation_speeds[i]);
  }

  update_uniform_buffers(context);
  update_dynamic_uniform_buffer(context, true);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    generate_cube(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipeline(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepare_render_bundle_encoder(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    imgui_overlay_checkBox(context->imgui_overlay, "Render bundles",
                           &render_bundles);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  if (render_bundles) {
    wgpuRenderPassEncoderExecuteBundles(wgpu_context->rpass_enc, 1,
                                        &render_bundle);
  }
  else {
    RECORD_RENDER_PASS(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_dynamic_uniform_buffer(context, false);
  }
  return draw_result;
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.view.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.dynamic.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(RenderBundle, render_bundle)
}

void example_dynamic_uniform_buffer(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
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
