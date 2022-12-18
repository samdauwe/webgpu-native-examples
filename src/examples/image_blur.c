#include "common_shaders.h"
#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Image Blur
 *
 * This example shows how to blur an image using a WebGPU compute shader.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/tree/main/src/sample/imageBlur
 * -------------------------------------------------------------------------- */

// Contants from the blur.wgsl shader.
static const uint32_t tile_dim = 128;
static const uint32_t batch[2] = {4, 4};

// Uniform buffers
static wgpu_buffer_t uniform_buffers[2];
static uint32_t uniform_buffer_data[2] = {0, 1};
static wgpu_buffer_t blur_params_buffer;

// Pipelines
static WGPUComputePipeline blur_pipeline           = NULL;
static WGPURenderPipeline fullscreen_quad_pipeline = NULL;

// Bind groups
static WGPUBindGroup compute_constants_bind_group = NULL;
static WGPUBindGroup compute_bind_groups[3]       = {0};
static WGPUBindGroup show_result_bind_group       = NULL;

// Texture and sampler
static texture_t texture          = {0};
static texture_t blur_textures[2] = {0};

// Settings
static struct {
  int32_t filter_size;
  int32_t iterations;
} settings = {
  .filter_size = 15,
  .iterations  = 2,
};
static uint32_t block_dim    = 1;
static uint32_t image_width  = 0;
static uint32_t image_height = 0;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Other variables
static const char* example_title = "Image Blur";
static bool prepared             = false;

static void prepare_texture(wgpu_context_t* wgpu_context)
{
  const char* file = "textures/Di-3d.png";
  texture          = wgpu_create_texture_from_file(wgpu_context, file, NULL);

  for (uint32_t i = 0; i < ARRAY_SIZE(blur_textures); ++i) {
    blur_textures[i].texture = wgpuDeviceCreateTexture(
      wgpu_context->device,
      &(WGPUTextureDescriptor){
        .label = "Blur texture",
        .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding
                 | WGPUTextureUsage_TextureBinding,
        .dimension     = WGPUTextureDimension_2D,
        .size          = (WGPUExtent3D) {
          .width               = texture.size.width,
          .height              = texture.size.height,
          .depthOrArrayLayers  = texture.size.depth,
        },
        .format        = texture.format,
        .mipLevelCount = 1,
        .sampleCount   = 1,
      });
    ASSERT(blur_textures[i].texture != NULL);

    blur_textures[i].view = wgpuTextureCreateView(
      blur_textures[i].texture, &(WGPUTextureViewDescriptor){
                                  .label          = "Blur texture view",
                                  .format         = texture.format,
                                  .dimension      = WGPUTextureViewDimension_2D,
                                  .baseMipLevel   = 0,
                                  .mipLevelCount  = 1,
                                  .baseArrayLayer = 0,
                                  .arrayLayerCount = 1,
                                });
    ASSERT(blur_textures[i].view != NULL);
  }

  image_width  = texture.size.width;
  image_height = texture.size.height;
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount = 1,
    .colorAttachments     = render_pass.color_attachments,
  };
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // buffer 0 and buffer 1
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(uniform_buffers); ++i) {
    uniform_buffers[i] = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
        .size         = 4,
        .initial.data = &uniform_buffer_data[i],
      });
  }

  // Compute shader blur parameters
  blur_params_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute shader blur parameters",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = 8,
                  });

  // Compute constants bind group
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0 : texture sampler
        .binding = 0,
        .sampler = texture.sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1 : blur parameters
        .binding = 1,
        .buffer  = blur_params_buffer.buffer,
        .offset  = 0,
        .size    = blur_params_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Compute constants bind group",
      .layout     = wgpuComputePipelineGetBindGroupLayout(blur_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    compute_constants_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(compute_constants_bind_group != NULL);
  }

  // Compute bind group 0
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0 : texture
        .binding     = 1,
        .textureView = texture.view,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1 : blur texture
        .binding     = 2,
        .textureView = blur_textures[0].view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2 : uniform buffer
        .binding = 3,
        .buffer  = uniform_buffers[0].buffer,
        .offset  = 0,
        .size    = uniform_buffers[0].size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Compute bind group 0",
      .layout     = wgpuComputePipelineGetBindGroupLayout(blur_pipeline, 1),
      .entryCount = 3,
      .entries    = bg_entries,
    };
    compute_bind_groups[0]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(compute_bind_groups[0] != NULL);
  }

  // Compute bind group 1
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 1 : texture
        .binding     = 1,
        .textureView = blur_textures[0].view,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 2 : blur texture
        .binding     = 2,
        .textureView = blur_textures[1].view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 3 : uniform buffer
        .binding = 3,
        .buffer  = uniform_buffers[1].buffer,
        .offset  = 0,
        .size    = uniform_buffers[1].size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Compute bind group 1",
      .layout     = wgpuComputePipelineGetBindGroupLayout(blur_pipeline, 1),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    compute_bind_groups[1]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(compute_bind_groups[1] != NULL);
  }

  // Compute bind group 2
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 1 : texture
        .binding     = 1,
        .textureView = blur_textures[1].view,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 2 : blur texture
        .binding     = 2,
        .textureView = blur_textures[0].view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 3 : uniform buffer
        .binding = 3,
        .buffer  = uniform_buffers[0].buffer,
        .offset  = 0,
        .size    = uniform_buffers[0].size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Compute bind group 2",
      .layout     = wgpuComputePipelineGetBindGroupLayout(blur_pipeline, 1),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    compute_bind_groups[2]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(compute_bind_groups[2] != NULL);
  }

  // Uniform bind group
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0 : texture sampler
        .binding = 0,
        .sampler = texture.sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1 : blur texture
        .binding     = 1,
        .textureView = blur_textures[1].view,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label = "Uniform bind group",
      .layout
      = wgpuRenderPipelineGetBindGroupLayout(fullscreen_quad_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    show_result_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(show_result_bind_group != NULL);
  }
}

// Create the compute & graphics pipelines
static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Blur compute pipeline
  {
    // Compute shader
    wgpu_shader_t blur_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      // Compute shader WGSL
                      .label = "Blur compute shader WGSL",
                      .file  = "shaders/image_blur/blur.wgsl",
                    });

    // Compute pipeline
    blur_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "Image blur render pipeline",
        .compute = blur_comp_shader.programmable_stage_descriptor,
      });
    ASSERT(blur_pipeline != NULL);

    // Partial clean-up
    wgpu_shader_release(&blur_comp_shader);
  }

  // Fullscreen quad render pipeline
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CW,
      .cullMode  = WGPUCullMode_Back,
    };

    // Color target state
    WGPUBlendState blend_state              = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
                  wgpu_context, &(wgpu_vertex_state_t){
                  .shader_desc = (wgpu_shader_desc_t){
                    // Vertex shader WGSL
                    .label            = "Fullscreen texturedquad wgsl",
                    .wgsl_code.source = fullscreen_textured_quad_wgsl,
                    .entry            = "vert_main"
                  },
                });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                  wgpu_context, &(wgpu_fragment_state_t){
                  .shader_desc = (wgpu_shader_desc_t){
                    // Fragment shader WGSL
                    .label            = "Fullscreen textured quad wgsl",
                    .wgsl_code.source = fullscreen_textured_quad_wgsl,
                    .entry            = "frag_main"
                  },
                  .target_count = 1,
                  .targets = &color_target_state,
                });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    fullscreen_quad_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label       = "Fullscreen quad pipeline",
                              .primitive   = primitive_state,
                              .vertex      = vertex_state,
                              .fragment    = &fragment_state,
                              .multisample = multisample_state,
                            });
    ASSERT(fullscreen_quad_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void update_settings(wgpu_context_t* wgpu_context)
{
  block_dim = tile_dim - (settings.filter_size - 1);

  uniform_buffer_data[0] = settings.filter_size;
  uniform_buffer_data[1] = block_dim;

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(wgpu_context, blur_params_buffer.buffer, 0,
                          &uniform_buffer_data, sizeof(uniform_buffer_data));
}

static int round_up_to_odd(int value, int min, int max)
{
  return MIN(MAX(min, (value % 2 == 0) ? value + 1 : value), max);
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    if (imgui_overlay_slider_int(context->imgui_overlay, "Filter Size",
                                 &settings.filter_size, 1, 33)) {
      settings.filter_size = round_up_to_odd(settings.filter_size, 1, 33);
      update_settings(context->wgpu_context);
    }
    imgui_overlay_slider_int(context->imgui_overlay, "Iterations",
                             &settings.iterations, 1, 10);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Compute pass */
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc, blur_pipeline);
    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                       compute_constants_bind_group, 0, NULL);

    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                       compute_bind_groups[0], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      wgpu_context->cpass_enc, ceil((float)image_width / block_dim),
      ceil((float)image_height / batch[1]), 1);

    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                       compute_bind_groups[1], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      wgpu_context->cpass_enc, ceil((float)image_height / block_dim),
      ceil((float)image_width / batch[1]), 1);

    for (uint32_t i = 0; i < (uint32_t)settings.iterations - 1; ++i) {
      wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                         compute_bind_groups[2], 0, NULL);
      wgpuComputePassEncoderDispatchWorkgroups(
        wgpu_context->cpass_enc, ceil((float)image_width / block_dim),
        ceil((float)image_height / batch[1]), 1);

      wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                         compute_bind_groups[1], 0, NULL);
      wgpuComputePassEncoderDispatchWorkgroups(
        wgpu_context->cpass_enc, ceil((float)image_height / block_dim),
        ceil((float)image_width / batch[1]), 1);
    }

    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  /* Fullscreen quad pipeline */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     fullscreen_quad_pipeline);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      show_result_bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, 1, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

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
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return 0;
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_texture(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    prepare_uniform_buffers(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    update_settings(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(TextureView, blur_textures[0].view)
  WGPU_RELEASE_RESOURCE(TextureView, blur_textures[1].view)
  WGPU_RELEASE_RESOURCE(Texture, blur_textures[0].texture)
  WGPU_RELEASE_RESOURCE(Texture, blur_textures[1].texture)
  wgpu_destroy_texture(&texture);

  WGPU_RELEASE_RESOURCE(BindGroup, compute_constants_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, compute_bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, compute_bind_groups[1])
  WGPU_RELEASE_RESOURCE(BindGroup, compute_bind_groups[2])
  WGPU_RELEASE_RESOURCE(BindGroup, show_result_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers[0].buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers[1].buffer)
  WGPU_RELEASE_RESOURCE(Buffer, blur_params_buffer.buffer)
  WGPU_RELEASE_RESOURCE(ComputePipeline, blur_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, fullscreen_quad_pipeline)
}

void example_image_blur(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
