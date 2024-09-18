#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Stencil Buffer Outlines
 *
 * Uses the stencil buffer and its compare functionality for rendering a 3D
 * model with dynamic outlines.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/stencilbuffer
 * -------------------------------------------------------------------------- */

static struct gltf_model_t* model = NULL;

static struct ubo_vs_t {
  mat4 projection;
  mat4 model;
  vec4 light_pos;
  /* Vertex shader extrudes model by this value along normals for outlining */
  float outline_width;
} ubo_vs = {
  .projection    = GLM_MAT4_ZERO_INIT,
  .model         = GLM_MAT4_ZERO_INIT,
  .light_pos     = {0.0f, 0.0f, 0.0f, 1.0f},
  .outline_width = 0.025f,
};

static wgpu_buffer_t uniform_buffer_vs = {0};

static struct {
  WGPURenderPipeline stencil;
  WGPURenderPipeline outline;
} pipelines = {0};

static WGPUPipelineLayout pipeline_layout    = {0};
static WGPUBindGroup bind_group              = {0};
static WGPUBindGroupLayout bind_group_layout = {0};

/* Render pass descriptor for frame buffer writes */
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

/* Other variables */
static const char* example_title = "Stencil Buffer Outlines";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera = camera_create();
  context->timer_speed *= 0.25f;
  context->camera->type = CameraType_LookAt;
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 512.0f);
  camera_set_rotation(context->camera, (vec3){2.5f, -35.0f, 0.0f});
  camera_set_translation(context->camera, (vec3){0.0f, 0.15f, -2.0f});
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices;
  model = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/venus.gltf",
    .file_loading_flags = gltf_loading_flags,
  });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0: Vertex shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(ubo_vs),
      },
      .sampler = {0},
    },
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
                            .label                = "Render pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  /* Bind Group */
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0: Uniform buffer (Vertex shader) */
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
  };

  bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "Bind group",
                            .layout     = bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(bind_group != NULL);
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
        .r = 0.025f,
        .g = 0.025f,
        .b = 0.025f,
        .a = 1.0f,
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Set clear sample for this example */
  wgpu_context->depth_stencil.att_desc.depthClearValue   = 1.0f;
  wgpu_context->depth_stencil.att_desc.stencilClearValue = 1;

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Construct the different states making up the pipeline

  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
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
  WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
    tunnel_cylinder,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex color
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Color),
    // Location 2: Vertex Normal
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_Normal));

  // Toon render and stencil fill pass
  {
    depth_stencil_state.depthWriteEnabled       = true;
    depth_stencil_state.stencilBack.compare     = WGPUCompareFunction_Always;
    depth_stencil_state.stencilBack.failOp      = WGPUStencilOperation_Replace;
    depth_stencil_state.stencilBack.depthFailOp = WGPUStencilOperation_Replace;
    depth_stencil_state.stencilBack.passOp      = WGPUStencilOperation_Replace;
    depth_stencil_state.stencilReadMask         = 0xff;
    depth_stencil_state.stencilWriteMask        = 0xff;
    depth_stencil_state.stencilFront = depth_stencil_state.stencilBack;

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
          wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader SPIR-V
              .label = "Toon - Vertex shader SPIR-V",
              .file  = "shaders/stencil_buffer/toon.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &tunnel_cylinder_vertex_buffer_layout,
          });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
          wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader SPIR-V
              .label = "Toon - Fragment shader SPIR-V",
              .file  = "shaders/stencil_buffer/toon.frag.spv",
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
    pipelines.stencil = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Stencil - Render pipeline",
                              .layout       = pipeline_layout,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.stencil != NULL);

    // Shader modules are no longer needed once the graphics pipeline has been
    // created
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  // Outline pass
  {
    depth_stencil_state.depthWriteEnabled       = false;
    depth_stencil_state.stencilBack.compare     = WGPUCompareFunction_NotEqual;
    depth_stencil_state.stencilBack.failOp      = WGPUStencilOperation_Keep;
    depth_stencil_state.stencilBack.depthFailOp = WGPUStencilOperation_Keep;
    depth_stencil_state.stencilBack.passOp      = WGPUStencilOperation_Replace;
    depth_stencil_state.stencilFront = depth_stencil_state.stencilBack;

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
          wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader SPIR-V
              .label = "Outline - Vertex shader SPIR-V",
              .file  = "shaders/stencil_buffer/outline.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &tunnel_cylinder_vertex_buffer_layout,
          });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
          wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader SPIR-V
              .label = "Outline - Fragment shader SPIR-V",
              .file  = "shaders/stencil_buffer/outline.frag.spv",
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
    pipelines.outline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Outline - Render pipeline",
                              .layout       = pipeline_layout,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.outline != NULL);

    // Shader modules are no longer needed once the graphics pipeline has been
    // created
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  glm_mat4_copy(context->camera->matrices.perspective, ubo_vs.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_vs.model);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &ubo_vs, sizeof(ubo_vs));
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Mesh vertex shader uniform buffer block */
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Mesh vertex shader uniform buffer block",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_vs),
    });
  ASSERT(uniform_buffer_vs.buffer != NULL);

  update_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    if (imgui_overlay_input_float(context->imgui_overlay, "Outline Width",
                                  &ubo_vs.outline_width, 0.05f, "%.2f")) {
      update_uniform_buffers(context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  /* First pass renders object (toon shaded) and fills stencil buffer */
  {
    /* Bind the rendering pipeline */
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.stencil);

    /* Draw model */
    wgpu_gltf_model_draw(model, (wgpu_gltf_model_render_options_t){0});
  }

  // Second pass renders scaled object only where stencil was not set by first
  // pass
  {
    /* Bind the rendering pipeline */
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.outline);

    /* Draw model */
    wgpu_gltf_model_draw(model, (wgpu_gltf_model_render_options_t){0});
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
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0] = build_command_buffer(context);

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
  return example_draw(context);
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_uniform_buffers(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  wgpu_gltf_model_destroy(model);
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.stencil)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.outline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
}

void example_stencil_buffer(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
