#include "common_shaders.h"
#include "example_base.h"
#include "meshes.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Two Cubes
 *
 * This example shows some of the alignment requirements involved when updating
 * and binding multiple slices of a uniform buffer. It renders two rotating
 * cubes which have transform matrices at different offsets in a uniform buffer.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/blob/main/src/sample/twoCubes
 * -------------------------------------------------------------------------- */

#define NUMBER_OF_CUBES 2ull

// Settings
static struct {
  uint64_t number_of_cubes;
  bool render_bundles;
} settings = {
  .number_of_cubes = NUMBER_OF_CUBES,
  .render_bundles  = true,
};

// Cube mesh
static cube_mesh_t cube_mesh = {0};

// Cube struct
typedef struct cube_t {
  WGPUBindGroup uniform_buffer_bind_group;
  struct {
    mat4 model;
    mat4 model_view_projection;
    mat4 tmp;
  } view_mtx;
} cube_t;
static cube_t cubes[NUMBER_OF_CUBES] = {0};

// Vertex buffer
static wgpu_buffer_t vertices = {0};

// Uniform buffer object
static struct {
  WGPUBuffer buffer;
  uint64_t offset;
  uint64_t size;
  uint64_t size_with_offset;
} uniform_buffer = {0};

static struct {
  mat4 projection;
  mat4 view;
} view_matrices = {0};

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render bundle
static WGPURenderBundle render_bundle = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Other variables
static const char* example_title = "Two Cubes";
static bool prepared             = false;

// Prepare the cube geometry
static void prepare_cube_mesh(void)
{
  cube_mesh_init(&cube_mesh);
}

// Create a vertex buffer from the cube data.
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Vertices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(cube_mesh.vertex_array),
                    .initial.data = cube_mesh.vertex_array,
                  });
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
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
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void update_transformation_matrix(wgpu_example_context_t* context)
{
  const float now     = context->frame.timestamp_millis / 1000.0f;
  const float sin_now = sin(now), cos_now = cos(now);

  cube_t* cube = NULL;
  for (uint64_t i = 0; i < settings.number_of_cubes; ++i) {
    cube = &cubes[i];
    glm_mat4_copy(cube->view_mtx.model, cube->view_mtx.tmp);
    if (i % 2 == 0) {
      glm_rotate(cube->view_mtx.tmp, 1.0f, (vec3){sin_now, cos_now, 0.0f});
    }
    else if (i % 2 == 1) {
      glm_rotate(cube->view_mtx.tmp, 1.0f, (vec3){cos_now, sin_now, 0.0f});
    }
    glm_mat4_mul(view_matrices.view, cube->view_mtx.tmp,
                 cube->view_mtx.model_view_projection);
    glm_mat4_mul(view_matrices.projection, cube->view_mtx.model_view_projection,
                 cube->view_mtx.model_view_projection);
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  update_transformation_matrix(context);

  for (uint64_t i = 0; i < settings.number_of_cubes; ++i) {
    wgpu_queue_write_buffer(
      context->wgpu_context, uniform_buffer.buffer, i * uniform_buffer.offset,
      &cubes[i].view_mtx.model_view_projection, sizeof(mat4));
  }
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  /* Projection matrix */
  glm_mat4_identity(view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  view_matrices.projection);

  /* View matrix */
  glm_mat4_identity(view_matrices.view);
  glm_translate(view_matrices.view, (vec3){0.0f, 0.0f, -7.0f});

  const float start_x = -2.0f, increment_x = 4.0f;
  cube_t* cube = NULL;
  float x      = 0.0f;
  for (uint64_t i = 0; i < settings.number_of_cubes; ++i) {
    cube = &cubes[i];
    x    = start_x + i * increment_x;

    /* Model matrices */
    glm_mat4_identity(cube->view_mtx.model);
    glm_translate(cube->view_mtx.model, (vec3){x, 0.0f, 0.0f});

    /* Model view matrices */
    glm_mat4_identity(cube->view_mtx.model_view_projection);

    /* Temporary matrices */
    glm_mat4_identity(cube->view_mtx.tmp);
  }
}

static void prepare_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Setup the view matrices for the camera */
  prepare_view_matrices(wgpu_context);

  /* Unform buffer */
  uniform_buffer.size = sizeof(mat4); /* 4x4 matrix */
  uniform_buffer.offset
    = 256; /* uniformBindGroup offset must be 256-byte aligned */
  uniform_buffer.size_with_offset
    = ((settings.number_of_cubes - 1) * uniform_buffer.offset)
      + uniform_buffer.size;

  uniform_buffer.buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = "Cube - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = uniform_buffer.size_with_offset,
    });
  ASSERT(uniform_buffer.buffer != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  for (uint64_t i = 0; i < settings.number_of_cubes; ++i) {
    cubes[i].uniform_buffer_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
        .label      = "Uniform buffer - Bind group",
        .layout     = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
        .entryCount = 1,
        .entries    = &(WGPUBindGroupEntry) {
          .binding = 0,
          .buffer  = uniform_buffer.buffer,
          .offset  = i * uniform_buffer.offset,
          .size    = uniform_buffer.size,
        },
      }
    );
    ASSERT(cubes[i].uniform_buffer_bind_group != NULL);
  }
}

static void prepare_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    // Backface culling since the cube is solid piece of geometry.
    // Faces pointing away from the camera will be occluded by faces pointing
    // toward the camera.
    .cullMode = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  // Enable depth testing so that the fragment closest to the camera is rendered
  // in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    two_cubes, cube_mesh.vertex_size,
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                       cube_mesh.position_offset),
    /* Attribute location 1: Color */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4, cube_mesh.color_offset))

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
        wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
           /* Vertex shader WGSL */
           .label            = "Basic - Vertex shader WGSL",
           .wgsl_code.source = basic_vertex_shader_wgsl,
        },
        .buffer_count = 1,
        .buffers = &two_cubes_vertex_buffer_layout,
      });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          /* Fragment shader WGSL */
          .label            = "Vertex position color - Fragment shader WGSL",
          .wgsl_code.source = vertex_position_color_fragment_shader_wgsl,
        },
        .target_count = 1,
        .targets = &color_target_state,
      });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Create rendering pipeline using the specified states */
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Two cubes - Render pipeline",
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

#define RECORD_RENDER_PASS(Type, rpass_enc)                                    \
  if (rpass_enc) {                                                             \
    wgpu##Type##SetPipeline(rpass_enc, pipeline);                              \
    wgpu##Type##SetVertexBuffer(rpass_enc, 0, vertices.buffer, 0,              \
                                WGPU_WHOLE_SIZE);                              \
    for (uint64_t i = 0; i < settings.number_of_cubes; ++i) {                  \
      wgpu##Type##SetBindGroup(rpass_enc, 0,                                   \
                               cubes[i].uniform_buffer_bind_group, 0, 0);      \
      wgpu##Type##Draw(rpass_enc, cube_mesh.vertex_count, 1, 0, 0);            \
    }                                                                          \
  }

static void prepare_render_bundle_encoder(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat color_formats[1] = {wgpu_context->swap_chain.format};
  WGPURenderBundleEncoder render_bundle_encoder
    = wgpuDeviceCreateRenderBundleEncoder(
      wgpu_context->device,
      &(WGPURenderBundleEncoderDescriptor){
        .label              = "Two cubes - Render bundle encoder",
        .colorFormatCount   = (uint32_t)ARRAY_SIZE(color_formats),
        .colorFormats       = color_formats,
        .depthStencilFormat = WGPUTextureFormat_Depth24PlusStencil8,
        .sampleCount        = 1,
      });
  RECORD_RENDER_PASS(RenderBundleEncoder, render_bundle_encoder)
  render_bundle = wgpuRenderBundleEncoderFinish(render_bundle_encoder, NULL);
  ASSERT(render_bundle != NULL);

  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, render_bundle_encoder)
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_cube_mesh();
    prepare_vertex_buffer(context->wgpu_context);
    prepare_pipeline(context->wgpu_context);
    prepare_uniform_buffer(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepare_render_bundle_encoder(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    imgui_overlay_checkBox(context->imgui_overlay, "Render Bundles",
                           &settings.render_bundles);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  if (settings.render_bundles) {
    wgpuRenderPassEncoderExecuteBundles(wgpu_context->rpass_enc, 1,
                                        &render_bundle);
  }
  else {
    RECORD_RENDER_PASS(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  /* Submit command buffers to queue */
  submit_command_buffers(context);

  /* Submit frame */
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
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer.buffer)
  for (uint64_t i = 0; i < settings.number_of_cubes; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, cubes[i].uniform_buffer_bind_group)
  }
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(RenderBundle, render_bundle)
}

void example_two_cubes(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy
  });
  // clang-format on
}
