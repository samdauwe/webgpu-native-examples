#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/text_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Text Overlay
 *
 * Load and render a 2D text overlay created from the bitmap glyph data of a stb
 * font file. This data is uploaded as a texture and used for displaying text on
 * top of a 3D scene in a second pass.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/textoverlay
 * -------------------------------------------------------------------------- */

static struct gltf_model_t* model = NULL;

static struct {
  text_overlay_t* handle;
  bool redraw_needed;
  bool visible;
} text_overlay = {
  .handle        = NULL,
  .redraw_needed = true,
  .visible       = true,
};

static wgpu_buffer_t uniform_buffer_vs = {0};

static struct ubo_vs_t {
  mat4 projection;
  mat4 model_view;
  vec4 light_pos;
} ubo_vs = {
  .projection = GLM_MAT4_ZERO_INIT,
  .model_view = GLM_MAT4_ZERO_INIT,
  .light_pos  = {0.0f, 0.0f, 0.0f, 1.0f},
};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

static WGPURenderPipeline pipeline           = NULL;
static WGPUPipelineLayout pipeline_layout    = NULL;
static WGPUBindGroup bind_group              = NULL;
static WGPUBindGroupLayout bind_group_layout = NULL;

// Other variables
static const char* example_title = "Text Overlay";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -2.5f});
  camera_set_rotation(context->camera, (vec3){25.0f, -0.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors
      | WGPU_GLTF_FileLoadingFlags_DontLoadImages;
  model = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/cube.gltf",
    .file_loading_flags = gltf_loading_flags,
  });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Vertex shader uniform buffer
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

  // Create the pipeline layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Pipeline layout",
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
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.2f,
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
    // Location 1: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal),
    // Location 2: Texture coordinates
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_UV));

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader SPIR-V
              .label = "Mesh - Vertex shader SPIR-V",
              .file  = "shaders/text_overlay/mesh.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &tunnel_cylinder_vertex_buffer_layout,
          });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader SPIR-V
              .label = "Mesh - Fragment shader SPIR-V",
              .file  = "shaders/text_overlay/mesh.frag.spv",
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
                            .label        = "Cube mesh - Render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  glm_mat4_copy(context->camera->matrices.perspective, ubo_vs.projection);
  mat4 model_view_scale = GLM_MAT4_IDENTITY_INIT;
  glm_scale(model_view_scale, (vec3){0.1f, 0.1f, 0.1f});
  glm_mat4_mul(context->camera->matrices.view, model_view_scale,
               ubo_vs.model_view);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &ubo_vs, uniform_buffer_vs.size);
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Vertex shader uniform buffer block
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Vertex shader - Uniform buffer block",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_vs),
    });
  ASSERT(uniform_buffer_vs.buffer != NULL);

  update_uniform_buffers(context);
}

// Update the text buffer displayed by the text overlay
void update_text_overlay(wgpu_example_context_t* context)
{
  // Window height
  const float width  = (float)context->wgpu_context->surface.width;
  const float height = (float)context->wgpu_context->surface.height;

  text_overlay_begin_text_update(text_overlay.handle);

  // Display example info
  text_overlay_add_formatted_text(
    text_overlay.handle, 5.0f, 5.0f, TextOverlay_Text_AlignLeft,
    "WebGPU Example - %s", context->example_title);
  text_overlay_add_formatted_text(
    text_overlay.handle, 5.0f, 25.0f, TextOverlay_Text_AlignLeft,
    "%.2f ms/frame (%.1d fps)", (1000.0f / context->last_fps),
    context->last_fps);
  text_overlay_add_text(text_overlay.handle, context->adapter_info[0], 5.0f,
                        45.0f, TextOverlay_Text_AlignLeft);

  // Display current model view matrix
  text_overlay_add_text(text_overlay.handle, "Model View Matrix", (float)width,
                        5.0f, TextOverlay_Text_AlignRight);

  for (uint32_t i = 0; i < 4; i++) {
    text_overlay_add_formatted_text(
      text_overlay.handle, (float)width, 25.0f + (float)i * 20.0f,
      TextOverlay_Text_AlignRight, "%+.2f %+.2f %+.2f %+.2f",
      ubo_vs.model_view[0][i], ubo_vs.model_view[1][i], ubo_vs.model_view[2][i],
      ubo_vs.model_view[3][i]);
  }

  // Display text overlay visibility toggle info
  text_overlay_add_text(text_overlay.handle,
                        "Press \"space\" to toggle text overlay", 5.0f, 65.0f,
                        TextOverlay_Text_AlignLeft);

  // Display cube dragging related text
  text_overlay_add_text(text_overlay.handle,
                        "Hold middle mouse button and drag to move", 5.0f,
                        85.0f, TextOverlay_Text_AlignLeft);
  mat4 model_view_projection = GLM_MAT4_ZERO_INIT;
  glm_mat4_mul(ubo_vs.projection, ubo_vs.model_view, model_view_projection);
  vec3 projected = GLM_VEC3_ZERO_INIT;
  glm_project((vec3){0.0f, 0.0f, 0.0}, model_view_projection,
              (vec4){0.f, 0.f, width, height}, projected);
  text_overlay_add_text(text_overlay.handle, "A cube", projected[0],
                        height - projected[1], TextOverlay_Text_AlignCenter);

  text_overlay_end_text_update(text_overlay.handle);
}

static void prepare_text_overlay(wgpu_example_context_t* context)
{
  text_overlay.handle        = text_overlay_create(context->wgpu_context);
  text_overlay.redraw_needed = true;
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
    prepare_text_overlay(context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
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

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  /* Draw model */
  wgpu_gltf_model_draw(model, (wgpu_gltf_model_render_options_t){0});

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw text overlay on frame */
  if (text_overlay.visible) {
    if (text_overlay.redraw_needed) {
      text_overlay.redraw_needed = false;
      update_text_overlay(context);
    }
    text_overlay_draw_frame(text_overlay.handle,
                            render_pass.color_attachments[0].view);
  }

  // Get command buffer */
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
  bool result = example_draw(context);
  if (context->frame_counter == 0) {
    text_overlay.redraw_needed = true;
  }
  return result;
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_uniform_buffers(context);
  text_overlay.redraw_needed = true;
}

static void example_on_key_pressed(keycode_t key)
{
  if (key == KEY_SPACE) {
    // Toggle text overlay visibility
    text_overlay.visible = !text_overlay.visible;
  }
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  text_overlay_release(text_overlay.handle);
  camera_release(context->camera);
  wgpu_gltf_model_destroy(model);
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
}

void example_text_overlay(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title          = example_title,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
    .example_on_key_pressed_func  = &example_on_key_pressed,
  });
  // clang-format on
}
