#include "example_base.h"

#include <string.h>

#include "../core/video_decode.h"

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Immersive Video
 *
 * This example shows how to display a 360-degree video where the viewer has
 * control of the viewing direction.
 *
 * Ref:
 * https://gist.github.com/fieldOfView/5106319
 * https://yanwsh.github.io/videojs-panorama/index_v4.html
 * https://github.com/muimota/p5video360
 * -------------------------------------------------------------------------- */

// Uniform buffer block object
static wgpu_buffer_t uniform_buffer_vs = {0};

// Uniform block data - inputs of the shader
static bool shader_inputs_ubo_update_needed = false;
static struct {
  vec2 iResolution; // Viewport resolution (in pixels)
  vec4 iMouse;      // Mouse pixel coords. xy: current (if MLB down), zw: click
  float iHFovDegrees;   // Horizontal field of view in degrees
  float iVFovDegrees;   // Vertical field of view in degrees
  bool iVisualizeInput; // Show the unprocessed input image
  vec4 padding;         // Padding to reach the minimum binding size of 64 bytes
} shader_inputs_ubo = {
  .iHFovDegrees = 120.0f,
  .iVFovDegrees = 80.0f,
};

// Used for mouse pixel coordinates calculation
static struct {
  vec2 initial_mouse_position;
  vec2 prev_mouse_position;
  vec2 mouse_drag_distance;
  bool dragging;
} mouse_state = {
  .initial_mouse_position = GLM_VEC2_ZERO_INIT,
  .prev_mouse_position    = GLM_VEC2_ZERO_INIT,
  .mouse_drag_distance    = GLM_VEC2_ZERO_INIT,
  .dragging               = false,
};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// The bind group layout
static WGPUBindGroupLayout bind_group_layout = NULL;

// The bind group
static WGPUBindGroup bind_group = NULL;

/* Texture and sampler */
static struct video_texture_t {
  WGPUSampler sampler;
  WGPUTexture texture;
  WGPUTextureView view;
} video_texture = {0};

static struct video_info_t {
  struct {
    int32_t width;
    int32_t height;
  } frame_size;
} video_info = {0};

static const char* video_file_location
  = "videos/immersive_video/underwater_diving_360degrees.mp4";

/* Other variables */
static const char* example_title = "Immersive Video";
static bool prepared             = false;

static void prepare_video_texture(wgpu_context_t* wgpu_context)
{
  /* Create the texture */
  video_texture.texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = "Video - Texture",
      .size          = (WGPUExtent3D){
        .width               = video_info.frame_size.width,
        .height              = video_info.frame_size.height,
        .depthOrArrayLayers  = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
  });
  ASSERT(video_texture.texture != NULL);

  /* Create the texture view */
  video_texture.view = wgpuTextureCreateView(
    video_texture.texture, &(WGPUTextureViewDescriptor){
                             .label           = "Video - Texture view",
                             .format          = WGPUTextureFormat_RGBA8Unorm,
                             .dimension       = WGPUTextureViewDimension_2D,
                             .baseMipLevel    = 0,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = 0,
                             .arrayLayerCount = 1,
                           });
  ASSERT(video_texture.view != NULL);

  /* Create the sampler */
  video_texture.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Video - Texture sampler",
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                            .maxAnisotropy = 1,
                          });
  ASSERT(video_texture.sampler != NULL);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /*  Binding 1: Fragment shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(mat4), /* 4x4 matrix */
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Fragment shader texture view */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2: Fragment shader texture sampler */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type  = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Immersive video - Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL);

  /* Create the pipeline layout */
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = "Immersive video - Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /*  Binding 1: Fragment shader uniform buffer */
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1: Fragment shader texture view */
      .binding     = 1,
      .textureView = video_texture.view,
    },
    [2] = (WGPUBindGroupEntry) {
      /* Binding 2: Fragment shader texture sampler */
      .binding = 2,
      .sampler = video_texture.sampler,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Immersive video - Bind group",
    .layout     = bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(bind_group != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Color attachment */
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                = "Render pass descriptor",
    .colorAttachmentCount = 1,
    .colorAttachments     = render_pass.color_attachments,
  };
}

static bool window_resized(wgpu_context_t* wgpu_context)
{
  return ((uint32_t)shader_inputs_ubo.iResolution[0]
          != (uint32_t)wgpu_context->surface.width)
         || ((uint32_t)shader_inputs_ubo.iResolution[1]
             != (uint32_t)wgpu_context->surface.height);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  /* iResolution: viewport resolution (in pixels) */
  if (window_resized(context->wgpu_context)) {
    shader_inputs_ubo.iResolution[0]
      = (float)context->wgpu_context->surface.width;
    shader_inputs_ubo.iResolution[1]
      = (float)context->wgpu_context->surface.height;
    shader_inputs_ubo_update_needed = true;
  }

  /* iMouse: mouse pixel coords. xy: current (if MLB down), zw: click */
  if (!mouse_state.dragging && context->mouse_buttons.left) {
    glm_vec2_copy(context->mouse_position, mouse_state.prev_mouse_position);
    mouse_state.dragging = true;
  }
  else if (mouse_state.dragging && context->mouse_buttons.left) {
    glm_vec2_sub(context->mouse_position, mouse_state.prev_mouse_position,
                 mouse_state.mouse_drag_distance);
    glm_vec2_add(shader_inputs_ubo.iMouse, mouse_state.mouse_drag_distance,
                 shader_inputs_ubo.iMouse);
    glm_vec2_copy(context->mouse_position, mouse_state.prev_mouse_position);
    shader_inputs_ubo_update_needed
      = shader_inputs_ubo_update_needed
        || ((fabs(mouse_state.mouse_drag_distance[0]) > 1.0f)
            || (fabs(mouse_state.mouse_drag_distance[1]) > 1.0f));
  }
  else if (mouse_state.dragging && !context->mouse_buttons.left) {
    mouse_state.dragging = false;
  }

  /* Map uniform buffer and update when needed */
  if (shader_inputs_ubo_update_needed) {
    wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                            &shader_inputs_ubo, uniform_buffer_vs.size);
    shader_inputs_ubo_update_needed = false;
  }
}

static void prepare_mouse_state(wgpu_context_t* wgpu_context)
{
  glm_vec2_copy(
    (vec2){wgpu_context->surface.width - (wgpu_context->surface.width / 4.0f),
           wgpu_context->surface.height / 2.0f},
    mouse_state.initial_mouse_position);
  glm_vec2_copy((vec4){mouse_state.initial_mouse_position[0],
                       mouse_state.initial_mouse_position[1], 0.0f, 0.0f},
                shader_inputs_ubo.iMouse);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Uniform buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size         = sizeof(shader_inputs_ubo),
      .initial.data = &shader_inputs_ubo,
    });
  ASSERT(uniform_buffer_vs.buffer != NULL);

  update_uniform_buffers(context);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader SPIR-V
                      .label = "Main - Vertex shader SPIR-V",
                      .file  = "shaders/immersive_video/main.vert.spv",
                    },
                    .buffer_count = 0,
                    .buffers      = NULL,
                  });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader SPIR-V
                      .label = "Main - Fragment shader SPIR-V",
                      .file  = "shaders/immersive_video/main.frag.spv",
                    },
                    .target_count = 1,
                    .targets      = &color_target_state,
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
                            .label       = "Immersive video - Render pipeline",
                            .layout      = pipeline_layout,
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int prepare_video(const char* fname)
{
  init_video_decode();
  open_video_file(fname);

  get_video_dimension(&video_info.frame_size.width,
                      &video_info.frame_size.height);

  start_video_decode();

  return EXIT_SUCCESS;
}

static int update_capture_texture(wgpu_context_t* wgpu_context)
{
  int video_w = 0, video_h = 0;
  void* video_buf = NULL;

  get_video_dimension(&video_w, &video_h);
  get_video_buffer(&video_buf);

  if (video_buf) {
    wgpu_image_to_texure(wgpu_context, video_texture.texture,
                         (uint8_t*)video_buf,
                         (WGPUExtent3D){
                           .width              = video_info.frame_size.width,
                           .height             = video_info.frame_size.height,
                           .depthOrArrayLayers = 1,
                         },
                         4u);
  }

  return EXIT_SUCCESS;
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_video(video_file_location);
    prepare_video_texture(context->wgpu_context);
    prepare_mouse_state(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
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
    if (imgui_overlay_input_float(
          context->imgui_overlay, "Horizontal FOV (degrees)",
          &shader_inputs_ubo.iHFovDegrees, 1.0f, "%.0f")) {
      shader_inputs_ubo.iHFovDegrees
        = clamp_float(shader_inputs_ubo.iHFovDegrees, 10, 1000);
      shader_inputs_ubo_update_needed = true;
    }
    if (imgui_overlay_input_float(
          context->imgui_overlay, "Vertical FOV (degrees)",
          &shader_inputs_ubo.iVFovDegrees, 1.0f, "%.0f")) {
      shader_inputs_ubo.iVFovDegrees
        = clamp_float(shader_inputs_ubo.iVFovDegrees, 10, 1000);
      shader_inputs_ubo_update_needed = true;
    }
    if (imgui_overlay_checkBox(context->imgui_overlay, "Show Input",
                               &shader_inputs_ubo.iVisualizeInput)) {
      shader_inputs_ubo_update_needed = true;
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Draw quad */
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);

  // End render pass */
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
  wgpu_context_t* wgpu_context = context->wgpu_context;

  // Update video texture
  update_capture_texture(wgpu_context);

  // Update the uniform buffers
  update_uniform_buffers(context);

  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit command buffer to queue
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

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  WGPU_RELEASE_RESOURCE(Texture, video_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, video_texture.view)
  WGPU_RELEASE_RESOURCE(Sampler, video_texture.sampler)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_immersive_video(int argc, char* argv[])
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
