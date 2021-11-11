#include "example_base.h"
#include "examples.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Conservative-raster
 *
 * This example shows how to render with conservative rasterization (native
 * extension with limited support).
 *
 * When enabled, any pixel touched by a triangle primitive is rasterized.
 * This is useful for various advanced techniques, most prominently for
 * implementing realtime voxelization.
 *
 * The demonstration here is implemented by rendering a triangle to a
 * low-resolution target and then upscaling it with nearest-neighbor filtering.
 * The outlines of the triangle are then rendered in the original solution,
 * using the same vertex shader as the triangle. Pixels only drawn with
 * conservative rasterization enabled are colored red.
 *
 * Note: Conservative rasterization not supported in Google Dawn.
 *
 * Ref:
 * https://github.com/gpuweb/gpuweb/issues/137
 * https://github.com/gfx-rs/wgpu/tree/master/wgpu/examples/conservative-raster
 * -------------------------------------------------------------------------- */

// Textures
static texture_t low_res_target_texture;

// Bind group layouts
static WGPUBindGroupLayout bind_group_layout_upscale;

// Bind group
static WGPUBindGroup bind_group_upscale;

// Pipeline layout
static WGPUPipelineLayout pipeline_layout;

// Render pipelines
static WGPURenderPipeline pipeline_triangle_conservative;
static WGPURenderPipeline pipeline_triangle_regular;
static WGPURenderPipeline pipeline_lines;
static WGPURenderPipeline pipeline_upscale;

// Other variables
static const char* example_title = "Conservative-raster";
static bool prepared             = false;

static void create_low_res_target(wgpu_context_t* wgpu_context)
{
  WGPUTextureDescriptor texture_desc = {
     .label         = "Low Resolution Target",
     .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment,
     .dimension     = WGPUTextureDimension_2D,
     .size          = (WGPUExtent3D) {
       .width              = wgpu_context->surface.width / 16,
       .height             = wgpu_context->surface.height / 16,
       .depthOrArrayLayers = 1,
     },
     .format        = wgpu_context->swap_chain.format,
     .mipLevelCount = 1,
     .sampleCount   = 1,
   };
  low_res_target_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(low_res_target_texture.texture != NULL)

  WGPUTextureViewDescriptor texture_view_dec = {
    .format          = wgpu_context->swap_chain.format,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  low_res_target_texture.view
    = wgpuTextureCreateView(low_res_target_texture.texture, &texture_view_dec);
  ASSERT(low_res_target_texture.view != NULL)

  WGPUSamplerDescriptor sampler_desc = {
    .label         = "Nearest Neighbor Sampler",
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .minFilter     = WGPUFilterMode_Nearest,
    .magFilter     = WGPUFilterMode_Nearest,
    .mipmapFilter  = WGPUFilterMode_Nearest,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  low_res_target_texture.sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(low_res_target_texture.sampler != NULL)

  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0 : Fragment shader texture view
      .binding = 0,
      .textureView = low_res_target_texture.view,
    },
    [1] = (WGPUBindGroupEntry) {
      // Binding 1: Fragment shader image sampler
      .binding = 1,
      .sampler = low_res_target_texture.sampler,
    },
  };
  bind_group_upscale = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "upscale bind group",
                            .layout     = bind_group_layout_upscale,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(bind_group_upscale != NULL)
}

static void prepare_pipeline_triangle_conservative(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
          wgpu_context, &(wgpu_vertex_state_t){
          .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader SPIR-V
            .file = "shaders/conservative_raster/triangle_and_lines.vert.spv",
          },
          .buffer_count = 0,
          .buffers = NULL,
        });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
        wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Fragment shader SPIR-V
          .file = "shaders/conservative_raster/triangle_and_lines_red.frag.spv",
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
  pipeline_triangle_conservative = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Conservative Rasterization",
                            .primitive   = primitive_state_desc,
                            .vertex      = vertex_state_desc,
                            .fragment    = &fragment_state_desc,
                            .multisample = multisample_state_desc,
                          });
  ASSERT(pipeline_triangle_conservative != NULL)

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static void prepare_pipeline_triangle_regular(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
        wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Vertex shader SPIR-V
          .file = "shaders/conservative_raster/triangle_and_lines.vert.spv",
        },
        .buffer_count = 0,
        .buffers = NULL,
      });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        // Fragment shader SPIR-V
        .file = "shaders/conservative_raster/triangle_and_lines_blue.frag.spv",
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
  pipeline_triangle_regular = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Regular Rasterization",
                            .primitive   = primitive_state_desc,
                            .vertex      = vertex_state_desc,
                            .fragment    = &fragment_state_desc,
                            .multisample = multisample_state_desc,
                          });
  ASSERT(pipeline_triangle_regular != NULL)

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static void prepare_pipeline_lines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology         = WGPUPrimitiveTopology_LineStrip,
    .stripIndexFormat = WGPUIndexFormat_Uint32,
    .frontFace        = WGPUFrontFace_CCW,
    .cullMode         = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
        wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Vertex shader SPIR-V
          .file = "shaders/conservative_raster/triangle_and_lines.vert.spv",
        },
        .buffer_count = 0,
        .buffers = NULL,
      });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        // Fragment shader SPIR-V
        .file = "shaders/conservative_raster/triangle_and_lines_white.frag.spv",
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
  pipeline_lines = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Lines",
                            .primitive   = primitive_state_desc,
                            .vertex      = vertex_state_desc,
                            .fragment    = &fragment_state_desc,
                            .multisample = multisample_state_desc,
                          });
  ASSERT(pipeline_lines != NULL)

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Texture view (Fragment shader)
      .binding = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Sampler (Fragment shader)
      .binding = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type=WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    }
  };
  bind_group_layout_upscale = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "upscale bindgroup",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout_upscale != NULL)

  // Create the pipeline layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout_upscale,
                          });
  ASSERT(pipeline_layout != NULL)
}

static void prepare_pipeline_upscale(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
        wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Vertex shader SPIR-V
          .file = "shaders/conservative_raster/upscale.vert.spv",
        },
        .buffer_count = 0,
        .buffers = NULL,
      });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        // Fragment shader SPIR-V
        .file = "shaders/conservative_raster/upscale.frag.spv",
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
  pipeline_upscale = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Upscale",
                            .layout      = pipeline_layout,
                            .primitive   = primitive_state_desc,
                            .vertex      = vertex_state_desc,
                            .fragment    = &fragment_state_desc,
                            .multisample = multisample_state_desc,
                          });
  ASSERT(pipeline_upscale != NULL)

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_pipeline_triangle_conservative(context->wgpu_context);
    prepare_pipeline_triangle_regular(context->wgpu_context);
    prepare_pipeline_lines(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipeline_upscale(context->wgpu_context);
    create_low_res_target(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  WGPUTextureView view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  {
    // Render pass descriptor
    WGPURenderPassDescriptor render_pass_desc = (WGPURenderPassDescriptor){
      .label                  = "low resolution",
      .colorAttachmentCount   = 1,
      .colorAttachments       = &(WGPURenderPassColorAttachment) {
        .view       = low_res_target_texture.view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearColor = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      },
      .depthStencilAttachment = NULL,
    };

    // Render pass
    WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);

    wgpuRenderPassEncoderSetPipeline(rpass, pipeline_triangle_conservative);
    wgpuRenderPassEncoderDraw(rpass, 3, 1, 0, 0);

    wgpuRenderPassEncoderSetPipeline(rpass, pipeline_triangle_regular);
    wgpuRenderPassEncoderDraw(rpass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEndPass(rpass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass)
  }

  {
    // Render pass descriptor
    WGPURenderPassDescriptor render_pass_desc = (WGPURenderPassDescriptor){
      .label                  = "full resolution",
      .colorAttachmentCount   = 1,
      .colorAttachments       = &(WGPURenderPassColorAttachment) {
        .view       = view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearColor = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      },
      .depthStencilAttachment = NULL,
    };

    // Render pass
    WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);

    wgpuRenderPassEncoderSetPipeline(rpass, pipeline_upscale);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, bind_group_upscale, 0, 0);
    wgpuRenderPassEncoderDraw(rpass, 3, 1, 0, 0);

    wgpuRenderPassEncoderSetPipeline(rpass, pipeline_lines);
    wgpuRenderPassEncoderDraw(rpass, 4, 1, 0, 0);

    wgpuRenderPassEncoderEndPass(rpass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass)
  }

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

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  wgpu_destroy_texture(&low_res_target_texture);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout_upscale)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group_upscale)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline_triangle_conservative)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline_triangle_regular)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline_lines)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline_upscale)
}

void example_conservative_raster(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
  });
  // clang-format on
}
