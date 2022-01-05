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

static const uint32_t tile_dim = 128;
static const uint32_t batch[2] = {4, 4};

// Vertex buffer
static struct {
  WGPUBuffer buffer;
  uint32_t count;
} vertices = {0};

// Uniform buffers
static WGPUBuffer uniform_buffers[2];
static uint32_t uniform_buffer_data[2] = {0, 1};
static WGPUBuffer blur_params_buffer;

// Pipelines
static WGPUComputePipeline blur_pipeline;
static WGPURenderPipeline render_pipeline;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

// Bind groups
static WGPUBindGroup compute_constants_bind_group;
static WGPUBindGroup compute_bind_groups[3];
static WGPUBindGroup uniform_bind_group;

// Texture and sampler
static texture_t texture;
static WGPUTexture textures[2];
static WGPUTextureView texture_views[2];

// Settings
static struct {
  int32_t filter_size;
  int32_t iterations;
} settings = {
  .filter_size = 15,
  .iterations  = 2,
};
static uint32_t block_dim = 1;
static uint32_t image_width;
static uint32_t image_height;

// Other variables
static const char* example_title = "Image Blur";
static bool prepared             = false;

// Prepare vertex buffers
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  static const float vertices_data[(3 + 2) * 6] = {
    // position data  /**/ uv data
    1.0f,  1.0f,  0.0f, /**/ 1.0f, 0.0f, //
    1.0f,  -1.0f, 0.0f, /**/ 1.0f, 1.0f, //
    -1.0f, -1.0f, 0.0f, /**/ 0.0f, 1.0f, //
    1.0f,  1.0f,  0.0f, /**/ 1.0f, 0.0f, //
    -1.0f, -1.0f, 0.0f, /**/ 0.0f, 1.0f, //
    -1.0f, 1.0f,  0.0f, /**/ 0.0f, 0.0f, //
  };

  vertices.count              = 6u;
  uint64_t vertex_buffer_size = (uint64_t)sizeof(vertices_data);

  // Create a host-visible staging buffer that contains the raw image data
  WGPUBufferDescriptor vertices_buffer_desc = {
    .usage            = WGPUBufferUsage_Vertex,
    .size             = vertex_buffer_size,
    .mappedAtCreation = true,
  };
  WGPUBuffer vertices_buffer
    = wgpuDeviceCreateBuffer(wgpu_context->device, &vertices_buffer_desc);
  ASSERT(vertices_buffer)

  // Copy vertices data into staging buffer
  void* mapping
    = wgpuBufferGetMappedRange(vertices_buffer, 0, vertex_buffer_size);
  ASSERT(mapping)
  memcpy(mapping, vertices_data, vertex_buffer_size);
  wgpuBufferUnmap(vertices_buffer);

  vertices.buffer = vertices_buffer;
}

static void prepare_texture(wgpu_context_t* wgpu_context)
{
  const char* file = "textures/Di-3d.png";
  texture          = wgpu_create_texture_from_file(wgpu_context, file, NULL);

  for (uint32_t i = 0; i < ARRAY_SIZE(textures); ++i) {
    textures[i] = wgpuDeviceCreateTexture(
      wgpu_context->device,
      &(WGPUTextureDescriptor){
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
    texture_views[i] = wgpuTextureCreateView(
      textures[i], &(WGPUTextureViewDescriptor){
                     .format          = texture.format,
                     .dimension       = WGPUTextureViewDimension_2D,
                     .baseMipLevel    = 0,
                     .mipLevelCount   = 1,
                     .baseArrayLayer  = 0,
                     .arrayLayerCount = 1,
                   });
  }

  image_width  = texture.size.width;
  image_height = texture.size.height;
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount = 1,
    .colorAttachments     = rp_color_att_descriptors,
  };
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // buffer 0 and buffer 1
  for (uint32_t i = 0; i < ARRAY_SIZE(uniform_buffers); ++i) {
    const WGPUBufferDescriptor buffer_desc = {
      .usage            = WGPUBufferUsage_Uniform,
      .size             = 4,
      .mappedAtCreation = true,
    };
    uniform_buffers[i]
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(uniform_buffers[i])

    void* mapping
      = wgpuBufferGetMappedRange(uniform_buffers[i], 0, buffer_desc.size);
    ASSERT(mapping)
    memcpy(mapping, &uniform_buffer_data[i], buffer_desc.size);
    wgpuBufferUnmap(uniform_buffers[i]);
  }

  // Compute shader blur parameters
  WGPUBufferDescriptor blur_params_buffer_desc = {
    .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
    .size             = 8,
    .mappedAtCreation = false,
  };
  blur_params_buffer
    = wgpuDeviceCreateBuffer(wgpu_context->device, &blur_params_buffer_desc);

  // Compute constants bind group
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .sampler = texture.sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer = blur_params_buffer,
        .offset = 0,
        .size = blur_params_buffer_desc.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .layout     = wgpuComputePipelineGetBindGroupLayout(blur_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    compute_constants_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(compute_constants_bind_group != NULL)
  }

  // Compute bind group 0
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 1,
        .textureView = texture.view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 2,
        .textureView = texture_views[0],
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 3,
        .buffer = uniform_buffers[0],
        .offset = 0,
        .size = 4,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .layout     = wgpuComputePipelineGetBindGroupLayout(blur_pipeline, 1),
      .entryCount = 3,
      .entries    = bg_entries,
    };
    compute_bind_groups[0]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(compute_bind_groups[0] != NULL)
  }

  // Compute bind group 1
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 1,
        .textureView = texture_views[0],
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 2,
        .textureView = texture_views[1],
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 3,
        .buffer = uniform_buffers[1],
        .offset = 0,
        .size = 4,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .layout     = wgpuComputePipelineGetBindGroupLayout(blur_pipeline, 1),
      .entryCount = 3,
      .entries    = bg_entries,
    };
    compute_bind_groups[1]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(compute_bind_groups[1] != NULL)
  }

  // Compute bind group 2
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 1,
        .textureView = texture_views[1],
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 2,
        .textureView = texture_views[0],
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 3,
        .buffer = uniform_buffers[0],
        .offset = 0,
        .size = 4,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .layout     = wgpuComputePipelineGetBindGroupLayout(blur_pipeline, 1),
      .entryCount = 3,
      .entries    = bg_entries,
    };
    compute_bind_groups[2]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(compute_bind_groups[2] != NULL)
  }

  // Uniform bindgroup
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .sampler = texture.sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .textureView = texture_views[1],
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .layout     = wgpuRenderPipelineGetBindGroupLayout(render_pipeline, 0),
      .entryCount = 2,
      .entries    = bg_entries,
    };
    uniform_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(uniform_bind_group != NULL)
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
                      // Compute shader SPIR-V
                      .file = "shaders/image_blur/blur.comp.spv",
                    });

    // Compute pipeline
    blur_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .compute = blur_comp_shader.programmable_stage_descriptor,
      });

    // Partial clean-up
    wgpu_shader_release(&blur_comp_shader);
  }

  // Fullscreen quad render pipeline
  {
    // Primitive state
    WGPUPrimitiveState primitive_state_desc = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CW,
      .cullMode  = WGPUCullMode_Back,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex buffer layout
    WGPU_VERTEX_BUFFER_LAYOUT(
      image_blur, 20,
      // Attribute location 0: Position
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
      // Attribute location 1: UV
      WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 12))

    // Vertex state
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
                  wgpu_context, &(wgpu_vertex_state_t){
                  .shader_desc = (wgpu_shader_desc_t){
                    // Vertex shader SPIR-V
                    .file = "shaders/image_blur/shader.vert.spv",
                  },
                  .buffer_count = 1,
                  .buffers = &image_blur_vertex_buffer_layout,
                });

    // Fragment state
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
                  wgpu_context, &(wgpu_fragment_state_t){
                  .shader_desc = (wgpu_shader_desc_t){
                    // Fragment shader SPIR-V
                    .file = "shaders/image_blur/shader.frag.spv",
                  },
                  .target_count = 1,
                  .targets = &color_target_state_desc,
                });

    // Multisample state
    WGPUMultisampleState multisample_state_desc
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    render_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label       = "image_blur_render_pipeline",
                              .primitive   = primitive_state_desc,
                              .vertex      = vertex_state_desc,
                              .fragment    = &fragment_state_desc,
                              .multisample = multisample_state_desc,
                            });

    // Shader modules are no longer needed once the graphics pipeline has been
    // created
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }
}

static void update_settings(wgpu_context_t* wgpu_context)
{
  block_dim = tile_dim - (settings.filter_size - 1);

  uniform_buffer_data[0] = settings.filter_size;
  uniform_buffer_data[1] = block_dim;

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(wgpu_context, blur_params_buffer, 0,
                          &uniform_buffer_data, sizeof(uniform_buffer_data));
}

static int round_up_to_odd(int value, int min, int max)
{
  return MIN(MAX(min, (value % 2 == 0) ? value + 1 : value), max);
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    if (imgui_overlay_slider_int(context->imgui_overlay, "Filter size",
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
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Compute pass
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc, blur_pipeline);
    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                       compute_constants_bind_group, 0, NULL);

    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                       compute_bind_groups[0], 0, NULL);
    wgpuComputePassEncoderDispatch(wgpu_context->cpass_enc,
                                   ceil((float)image_width / block_dim),
                                   ceil((float)image_height / batch[1]), 1);

    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                       compute_bind_groups[1], 0, NULL);
    wgpuComputePassEncoderDispatch(wgpu_context->cpass_enc,
                                   ceil((float)image_height / block_dim),
                                   ceil((float)image_width / batch[1]), 1);

    for (uint32_t i = 0; i < (uint32_t)settings.iterations - 1; ++i) {
      wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                         compute_bind_groups[2], 0, NULL);
      wgpuComputePassEncoderDispatch(wgpu_context->cpass_enc,
                                     ceil((float)image_width / block_dim),
                                     ceil((float)image_height / batch[1]), 1);

      wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                         compute_bind_groups[1], 0, NULL);
      wgpuComputePassEncoderDispatch(wgpu_context->cpass_enc,
                                     ceil((float)image_height / block_dim),
                                     ceil((float)image_width / batch[1]), 1);
    }

    wgpuComputePassEncoderEndPass(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  // Render pass
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, render_pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         vertices.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      uniform_bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, vertices.count, 1, 0, 0);
    wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

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

  return 0;
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertex_buffer(context->wgpu_context);
    prepare_texture(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    prepare_uniform_buffers(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    update_settings(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(TextureView, texture_views[0])
  WGPU_RELEASE_RESOURCE(TextureView, texture_views[1])
  WGPU_RELEASE_RESOURCE(Texture, textures[0])
  WGPU_RELEASE_RESOURCE(Texture, textures[1])
  wgpu_destroy_texture(&texture);

  WGPU_RELEASE_RESOURCE(BindGroup, compute_constants_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, compute_bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, compute_bind_groups[1])
  WGPU_RELEASE_RESOURCE(BindGroup, compute_bind_groups[2])
  WGPU_RELEASE_RESOURCE(BindGroup, uniform_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers[0])
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers[1])
  WGPU_RELEASE_RESOURCE(Buffer, blur_params_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(ComputePipeline, blur_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipeline)
}

void example_image_blur(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
  });
  // clang-format on
}
