#include "common_shaders.h"

#include "example_base.h"
#include "meshes.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Textured Cube
 *
 * This example shows how to bind and sample textures.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/blob/main/src/sample/texturedCube
 * https://github.com/gfx-rs/wgpu-rs/tree/master/examples/cube
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* sampled_texture_mix_color_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Textured Cube example
 * -------------------------------------------------------------------------- */

// Cube mesh
static cube_mesh_t cube_mesh = {0};

// Cube struct
static struct {
  WGPUBindGroup uniform_buffer_bind_group;
  WGPUBindGroupLayout bind_group_layout;
  struct {
    mat4 model_view_projection;
  } view_mtx;
} cube = {0};

// Vertex buffer
static wgpu_buffer_t vertices = {0};

// Uniform buffer block object
static wgpu_buffer_t uniform_buffer_vs = {0};

static struct {
  mat4 projection;
  mat4 view;
} view_matrices = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Texture and sampler
static texture_t texture = {0};

// Other variables
static const char* example_title = "Textured Cube";
static bool prepared             = false;

// Prepare the cube geometry
static void prepare_cube_mesh(void)
{
  cube_mesh_init(&cube_mesh);
}

// Prepare vertex buffer
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(cube_mesh.vertex_array),
                    .initial.data = cube_mesh.vertex_array,
                  });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Transform */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(mat4), /* 4x4 matrix */
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Texture view */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    }
  };
  cube.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Cube - Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(cube.bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Render pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &cube.bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void prepare_texture(wgpu_context_t* wgpu_context)
{
  const char* file = "textures/Di-3d.png";
  texture          = wgpu_create_texture_from_file(wgpu_context, file, NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
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

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  // Projection matrix
  glm_mat4_identity(view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  view_matrices.projection);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Setup the view matrices for the camera */
  prepare_view_matrices(context->wgpu_context);

  /* Uniform buffer */
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Camera view matrices - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(mat4), // 4x4 matrix
    });
  ASSERT(uniform_buffer_vs.buffer != NULL);
}

static void update_transformation_matrix(wgpu_example_context_t* context)
{
  const float now = context->frame.timestamp_millis / 1000.0f;

  /* View matrix */
  glm_mat4_identity(view_matrices.view);
  glm_translate(view_matrices.view, (vec3){0.0f, 0.0f, -4.0f});
  glm_rotate(view_matrices.view, 1.0f, (vec3){sin(now), cos(now), 0.0f});

  /* Model view projection matrix */
  glm_mat4_identity(cube.view_mtx.model_view_projection);
  glm_mat4_mul(view_matrices.projection, view_matrices.view,
               cube.view_mtx.model_view_projection);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  /* Update the model-view-projection matrix */
  update_transformation_matrix(context);

  /* Map uniform buffer and update it */
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &cube.view_mtx.model_view_projection,
                          uniform_buffer_vs.size);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .sampler = texture.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = texture.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Cube uniform - Buffer bind group",
    .layout     = cube.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  cube.uniform_buffer_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(cube.uniform_buffer_bind_group != NULL);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
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

  // Depth stencil state
  // Enable depth testing so that the fragment closest to the camera
  // is rendered in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    textured_cube, cube_mesh.vertex_size,
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                       cube_mesh.position_offset),
    /* Attribute location 1: UV */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, cube_mesh.uv_offset))

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
       wgpu_context, &(wgpu_vertex_state_t){
       .shader_desc = (wgpu_shader_desc_t){
          /* Vertex shader WGSL */
          .label            = "Basic - Vertex shader WGSL",
          .wgsl_code.source = basic_vertex_shader_wgsl,
       },
       .buffer_count = 1,
       .buffers = &textured_cube_vertex_buffer_layout,
     });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
       wgpu_context, &(wgpu_fragment_state_t){
       .shader_desc = (wgpu_shader_desc_t){
          /* Fragment shader WGSL */
          .label            = "Sampled texture mix color - Fragment shader WGSL",
          .wgsl_code.source = sampled_texture_mix_color_fragment_shader_wgsl,
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
                            .label        = "Textured cube - Render pipeline",
                            .layout       = pipeline_layout,
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

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_cube_mesh();
    prepare_vertex_buffer(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_texture(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_bind_groups(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  /* Bind cube vertex buffer (contains position and colors) */
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);

  //* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    cube.uniform_buffer_bind_group, 0, 0);

  /* Draw textured cube */
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, cube_mesh.vertex_count, 1,
                            0, 0);

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

  /* Submit to queue */
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

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  wgpu_destroy_texture(&texture);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, cube.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, cube.uniform_buffer_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_textured_cube(int argc, char* argv[])
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
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* sampled_texture_mix_color_fragment_shader_wgsl = CODE(
  @group(0) @binding(1) var mySampler: sampler;
  @group(0) @binding(2) var myTexture: texture_2d<f32>;

  @fragment
  fn main(
    @location(0) fragUV: vec2<f32>,
    @location(1) fragPosition: vec4<f32>
  ) -> @location(0) vec4<f32> {
    return textureSample(myTexture, mySampler, fragUV) * fragPosition;
  }
);
// clang-format on
