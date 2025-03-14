#include "example_base.h"

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
 * Note: Conservative rasterization is not yet supported in Google Dawn.
 *
 * Ref:
 * https://github.com/gpuweb/gpuweb/issues/137
 * https://github.com/gfx-rs/wgpu/tree/master/wgpu/examples/conservative-raster
 * -------------------------------------------------------------------------- */

// Texture
static texture_t low_res_target_texture = {0};

// Bind group layout
static WGPUBindGroupLayout bind_group_layout_upscale = NULL;

// Bind group
static WGPUBindGroup bind_group_upscale = NULL;

// Pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Render pipelines
static struct {
  WGPURenderPipeline triangle_conservative;
  WGPURenderPipeline triangle_regular;
  WGPURenderPipeline lines;
  WGPURenderPipeline upscale;
} render_pipelines = {0};

// Other variables
static const char* example_title = "Conservative-raster";
static bool prepared             = false;

static void create_low_res_target(wgpu_context_t* wgpu_context)
{
  WGPUTextureDescriptor texture_desc = {
     .label         = "Low Resolution Target - Texture",
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
  ASSERT(low_res_target_texture.texture != NULL);

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
  ASSERT(low_res_target_texture.view != NULL);

  WGPUSamplerDescriptor sampler_desc = {
    .label         = "Nearest Neighbor - Texture Sampler",
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .minFilter     = WGPUFilterMode_Nearest,
    .magFilter     = WGPUFilterMode_Nearest,
    .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  low_res_target_texture.sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(low_res_target_texture.sampler != NULL);

  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0 : Fragment shader texture view
      .binding     = 0,
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
                            .label      = "Upscale - Bind group",
                            .layout     = bind_group_layout_upscale,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(bind_group_upscale != NULL);
}

static void prepare_pipeline_triangle_conservative(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
          wgpu_context, &(wgpu_vertex_state_t){
          .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader SPIR-V
            .label = "Triangle and lines - Vertex shader SPIR-V",
            .file  = "shaders/conservative_raster/triangle_and_lines.vert.spv",
          },
          .buffer_count = 0,
          .buffers      = NULL,
        });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Fragment shader SPIR-V
          .label = "Triangle and lines red colored - Fragment shader SPIR-V",
          .file = "shaders/conservative_raster/triangle_and_lines_red.frag.spv",
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
  render_pipelines.triangle_conservative = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label       = "Conservative Rasterization - Render pipeline",
      .primitive   = primitive_state,
      .vertex      = vertex_state,
      .fragment    = &fragment_state,
      .multisample = multisample_state,
    });
  ASSERT(render_pipelines.triangle_conservative != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void prepare_pipeline_triangle_regular(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
        wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Vertex shader SPIR-V
          .label = "Triangle and lines - Vertex shader SPIR-V",
          .file  = "shaders/conservative_raster/triangle_and_lines.vert.spv",
        },
        .buffer_count = 0,
        .buffers      = NULL,
      });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        // Fragment shader SPIR-V
        .label = "Triangle and lines blue colored - Fragment shader SPIR-V",
        .file = "shaders/conservative_raster/triangle_and_lines_blue.frag.spv",
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
  render_pipelines.triangle_regular = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label = "Regular Rasterization - Render pipeline",
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(render_pipelines.triangle_regular != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void prepare_pipeline_lines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology         = WGPUPrimitiveTopology_LineStrip,
    .stripIndexFormat = WGPUIndexFormat_Uint32,
    .frontFace        = WGPUFrontFace_CCW,
    .cullMode         = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
        wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Vertex shader SPIR-V
          .label = "Triangle and lines - Vertex shader SPIR-V",
          .file  = "shaders/conservative_raster/triangle_and_lines.vert.spv",
        },
        .buffer_count = 0,
        .buffers      = NULL,
      });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        // Fragment shader SPIR-V
        .label = "Triangle and lines white colored - Fragment shader SPIR-V",
        .file = "shaders/conservative_raster/triangle_and_lines_white.frag.spv",
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
  render_pipelines.lines = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Lines - Render pipeline",
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(render_pipelines.lines != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Texture view (Fragment shader)
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Sampler (Fragment shader)
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    }
  };
  bind_group_layout_upscale = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Upscale - Bind group",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout_upscale != NULL);

  // Create the pipeline layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Render - Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout_upscale,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void prepare_pipeline_upscale(wgpu_context_t* wgpu_context)
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
          /* Vertex shader SPIR-V */
          .label = "Upscale - Vertex shader SPIR-V",
          .file  = "shaders/conservative_raster/upscale.vert.spv",
        },
        .buffer_count = 0,
        .buffers      = NULL,
      });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        /* Fragment shader SPIR-V */
        .label = "Upscale - Fragment shader SPIR-V",
        .file  = "shaders/conservative_raster/upscale.frag.spv",
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
  render_pipelines.upscale = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Upscale - Rendering pipeline",
                            .layout      = pipeline_layout,
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(render_pipelines.upscale != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
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
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  WGPUTextureView view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  {
    /* Render pass descriptor */
    WGPURenderPassDescriptor render_pass_desc = (WGPURenderPassDescriptor){
      .label                  = "Low resolution - Render pass",
      .colorAttachmentCount   = 1,
      .colorAttachments       = &(WGPURenderPassColorAttachment) {
        .view       = low_res_target_texture.view,
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      },
      .depthStencilAttachment = NULL,
    };

    /* Render pass */
    WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);

    wgpuRenderPassEncoderSetPipeline(rpass,
                                     render_pipelines.triangle_conservative);
    wgpuRenderPassEncoderDraw(rpass, 3, 1, 0, 0);

    wgpuRenderPassEncoderSetPipeline(rpass, render_pipelines.triangle_regular);
    wgpuRenderPassEncoderDraw(rpass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(rpass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass)
  }

  {
    /* Render pass descriptor */
    WGPURenderPassDescriptor render_pass_desc = (WGPURenderPassDescriptor){
      .label                  = "Full resolution - Render pass",
      .colorAttachmentCount   = 1,
      .colorAttachments       = &(WGPURenderPassColorAttachment) {
        .view       = view,
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      },
      .depthStencilAttachment = NULL,
    };

    /* Render pass */
    WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);

    wgpuRenderPassEncoderSetPipeline(rpass, render_pipelines.upscale);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, bind_group_upscale, 0, 0);
    wgpuRenderPassEncoderDraw(rpass, 3, 1, 0, 0);

    wgpuRenderPassEncoderSetPipeline(rpass, render_pipelines.lines);
    wgpuRenderPassEncoderDraw(rpass, 4, 1, 0, 0);

    wgpuRenderPassEncoderEnd(rpass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass)
  }

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

  /* Submit command buffer to queue */
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
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  wgpu_destroy_texture(&low_res_target_texture);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout_upscale)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group_upscale)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.triangle_conservative)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.triangle_regular)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.lines)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.upscale)
}

void example_conservative_raster(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .vsync = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
