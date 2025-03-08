#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - A Conway Game Of Life With Paletted Blurring Over Time
 *
 * A binary Conway game of life.
 *
 * Ref:
 * https://github.com/Palats/webgpu/blob/main/src/demos/conway2.ts
 * https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* compute_shader_wgsl;
static const char* graphics_vertex_shader_wgsl;
static const char* graphics_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * A Conway Game Of Life With Paletted Blurring Over Time example
 * -------------------------------------------------------------------------- */

static struct {
  struct wgpu_buffer_t buffer;
  struct {
    uint32_t compute_width;
    uint32_t compute_height;
  } desc;
} uniforms = {
  .buffer              = {0},
  .desc.compute_width  = 0u,
  .desc.compute_height = 1u,
};

// Textures
static WGPUTextureFormat COMPUTE_TEX_FORMAT = WGPUTextureFormat_RGBA8Unorm;
static uint32_t COMPUTE_TEX_BYTES           = 4u; // Bytes per pixel in compute.
static struct {
  // Swapchain for the cellular automata progression
  texture_t cells[2];
  // Swap chain for the intermediate compute effect on top of the cellular
  // automata
  texture_t trails[2];
} textures = {0};

// Resources for the graphics part of the example
static struct {
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_groups[2];
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
} graphics = {0};

// Resources for the compute part of the example
static struct {
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_groups[2];
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline pipeline;
} compute = {0};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

static bool is_forward = true;

// Other variables
static const char* example_title
  = "A Conway Game Of Life With Paletted Blurring Over Time";
static bool prepared = false;

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Update unfirms data
  uniforms.desc.compute_width  = wgpu_context->surface.width;
  uniforms.desc.compute_height = wgpu_context->surface.height;

  // Uplaad buffer to the GPU
  wgpu_queue_write_buffer(wgpu_context, uniforms.buffer.buffer, 0,
                          &uniforms.desc, uniforms.buffer.size);
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Create uniforms buffer */
  uniforms.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(uniforms.desc),
                  });

  /* Update unifroms buffer data */
  update_uniform_buffers(wgpu_context);
}

/* Textures, used for compute part */
static void prepare_textures(wgpu_context_t* wgpu_context)
{
  const uint32_t compute_width  = wgpu_context->surface.width;
  const uint32_t compute_height = wgpu_context->surface.height;

  WGPUExtent3D texture_extent = {
    .width              = compute_width,
    .height             = compute_height,
    .depthOrArrayLayers = 1,
  };

  texture_t* textureArray[4] = {
    &textures.cells[0],  //
    &textures.cells[1],  //
    &textures.trails[0], //
    &textures.trails[1], //
  };

  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(textureArray); ++i) {
    texture_t* tex = textureArray[i];

    /* Create the texture */
    WGPUTextureDescriptor texture_desc = {
      .label         = "Compute - Texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = COMPUTE_TEX_FORMAT,
      .usage
      = ((i % 2) == 0) ?
          (WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding
           | WGPUTextureUsage_CopyDst) :
          (WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding),
    };
    tex->texture = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(tex->texture);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Compute - Texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    tex->view = wgpuTextureCreateView(tex->texture, &texture_view_dec);
    ASSERT(tex->view);

    // Create sampler to sample to pick from the texture and write to the screen
    tex->sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = "Compute - Texture sampler",
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(tex->sampler);
  }

  /* Setup the initial cellular automata */
  {
    uint8_t* b = malloc(compute_width * compute_height * COMPUTE_TEX_BYTES
                        * sizeof(uint8_t));
    ASSERT(b);
    bool has_life = false;
    uint8_t v     = 0;
    for (uint32_t y = 0; y < compute_height; ++y) {
      for (uint32_t x = 0; x < compute_width; ++x) {
        has_life = random_float() > 0.8f;
        v        = has_life ? 255 : 0;
        b[COMPUTE_TEX_BYTES * (x + y * compute_width) + 0] = v;
        b[COMPUTE_TEX_BYTES * (x + y * compute_width) + 1] = v;
        b[COMPUTE_TEX_BYTES * (x + y * compute_width) + 2] = v;
        b[COMPUTE_TEX_BYTES * (x + y * compute_width) + 3] = 255;
      }
    }
    wgpu_image_to_texure(wgpu_context, textures.cells[0].texture, b,
                         texture_extent, COMPUTE_TEX_BYTES);
    free(b);
  }
}

static void setup_pipeline_layouts(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Input automata texture */
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: Output automata texture */
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = (WGPUStorageTextureBindingLayout) {
          .access        = WGPUStorageTextureAccess_WriteOnly,
          .format        = COMPUTE_TEX_FORMAT,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2: Input trail texture */
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        /* Binding 3: Output trail texture */
        .binding    = 3,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = (WGPUStorageTextureBindingLayout) {
          .access        = WGPUStorageTextureAccess_WriteOnly,
          .format        = COMPUTE_TEX_FORMAT,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Compute pipeline main - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    compute.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(compute.bind_group_layout != NULL);

    // Compute pipeline layout
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "Compute - Pipeline layout",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &compute.bind_group_layout,
    };
    compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(compute.pipeline_layout != NULL);
  }

  /* Graphics pipeline layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Texture from compute */
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
        /* Sampler for  the texture */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      }
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Rendering pipeline main - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    graphics.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(graphics.bind_group_layout != NULL);

    // Render pipeline layout
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "Rendering - Pipeline layout",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &graphics.bind_group_layout,
    };
    graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(graphics.pipeline_layout != NULL);
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
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

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                = "Render pass descriptor",
    .colorAttachmentCount = 1,
    .colorAttachments     = render_pass.color_attachments,
  };
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Compute binding group for rendering:
  //    1 -> 2
  //    2 -> 1
  for (uint32_t i = 0; i < 2; ++i) {
    WGPUBindGroupEntry bg_entries[4] = {
        [0] = (WGPUBindGroupEntry) {
          .binding     = 0,
          .textureView = (i == 0) ? textures.cells[0].view : textures.cells[1].view,
        },
        [1] = (WGPUBindGroupEntry) {
          .binding     = 1,
          .textureView = (i == 0) ? textures.cells[1].view : textures.cells[0].view,
        },
        [2] = (WGPUBindGroupEntry) {
          .binding     = 2,
          .textureView = (i == 0) ? textures.trails[0].view : textures.trails[1].view,
        },
        [3] = (WGPUBindGroupEntry) {
          .binding     = 3,
          .textureView = (i == 0) ? textures.trails[1].view : textures.trails[0].view,
        },
      };

    compute.bind_groups[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label  = "Compute - Bind group",
        .layout = wgpuComputePipelineGetBindGroupLayout(compute.pipeline, 0),
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(compute.bind_groups[i] != NULL);
  }

  // Create 2 bind group for the render pipeline:
  //    1 -> 2
  //    2 -> 1
  texture_t* trailTextureArray[2] = {
    &textures.trails[0], /* 1 -> 2 */
    &textures.trails[1], /* 2 -> 1 */
  };
  const uint32_t nbTextures = (uint32_t)ARRAY_SIZE(trailTextureArray);
  for (uint32_t i = 0; i < nbTextures; ++i) {
    texture_t* tex = trailTextureArray[(i + 1) % nbTextures];
    WGPUBindGroupEntry bg_entries[2] = {
        [0] = (WGPUBindGroupEntry) {
          .binding     = 0,
          .textureView = tex->view,
        },
        [1] = (WGPUBindGroupEntry) {
          .binding = 1,
          .sampler = tex->sampler,
        },
      };

    graphics.bind_groups[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label  = "Graphics - Bind group",
        .layout = wgpuRenderPipelineGetBindGroupLayout(graphics.pipeline, 0),
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(graphics.bind_groups[i] != NULL);
  }
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline */
  {
    // Compute shader
    wgpu_shader_t conway_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      // Compute shader WGSL
                      .label            = "Compute shader WGSL",
                      .wgsl_code.source = compute_shader_wgsl,
                      .entry            = "main",
                    });

    // Create compute pipeline
    compute.pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "Effect - Compute pipeline",
        .layout  = compute.pipeline_layout,
        .compute = conway_comp_shader.programmable_stage_descriptor,
      });
    ASSERT(compute.pipeline != NULL);

    // Partial cleanup
    wgpu_shader_release(&conway_comp_shader);
  }

  /* Graphics pipeline */
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

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
          wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader WGSL
              .label            = "Graphics - Vertex shader WGSL",
              .wgsl_code.source = graphics_vertex_shader_wgsl,
              .entry            = "main",
            },
            .buffer_count = 0,
            .buffers = NULL,
          });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
          wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader WGSL
              .label            = "Graphics - Fragment shader WGSL",
              .wgsl_code.source = graphics_fragment_shader_wgsl,
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
    graphics.pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label       = "Conway - Graphics pipeline",
                              .layout      = graphics.pipeline_layout,
                              .primitive   = primitive_state,
                              .vertex      = vertex_state,
                              .fragment    = &fragment_state,
                              .multisample = multisample_state,
                            });
    ASSERT(graphics.pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_uniform_buffers(context->wgpu_context);
    prepare_textures(context->wgpu_context);
    setup_pipeline_layouts(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context          = context->wgpu_context;
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* -- Frame compute -- */
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                      compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(
      wgpu_context->cpass_enc, 0,
      is_forward ? compute.bind_groups[0] : compute.bind_groups[1], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      wgpu_context->cpass_enc,
      (uint32_t)ceil(uniforms.desc.compute_width / 8.0f),
      (uint32_t)ceil(uniforms.desc.compute_height / 8.0f), 1);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  /* -- Frame rendering -- */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(
      wgpu_context->rpass_enc, 0,
      is_forward ? graphics.bind_groups[0] : graphics.bind_groups[1], 0, NULL);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, 1, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL);
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

  // Submit command buffers to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  // Switch for next frame
  is_forward = !is_forward;

  return EXIT_SUCCESS;
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

  // Textures
  wgpu_destroy_texture(&textures.cells[0]);
  wgpu_destroy_texture(&textures.cells[1]);
  wgpu_destroy_texture(&textures.trails[0]);
  wgpu_destroy_texture(&textures.trails[1]);

  // Graphics pipeline
  WGPU_RELEASE_RESOURCE(BindGroupLayout, graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, graphics.bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, graphics.bind_groups[1])
  WGPU_RELEASE_RESOURCE(PipelineLayout, graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, graphics.pipeline)

  // Compute pipeline
  WGPU_RELEASE_RESOURCE(Buffer, uniforms.buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, compute.bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, compute.bind_groups[1])
  WGPU_RELEASE_RESOURCE(PipelineLayout, compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, compute.pipeline)
}

void example_conway_paletted_blurring(int argc, char* argv[])
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

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* compute_shader_wgsl = CODE(
  @group(0) @binding(0) var cellsSrc : texture_2d<f32>;
  @group(0) @binding(1) var cellsDst : texture_storage_2d<rgba8unorm, write>;
  @group(0) @binding(2) var trailSrc : texture_2d<f32>;
  @group(0) @binding(3) var trailDst : texture_storage_2d<rgba8unorm, write>;

  fn cellAt(x: i32, y: i32) -> i32 {
    let v = textureLoad(cellsSrc, vec2<i32>(x, y), 0);
    if (v.r < 0.5) {
      return 0;
    }
    return 1;
  }

  fn trailAt(x: i32, y: i32) -> vec4<f32> {
    return textureLoad(trailSrc, vec2<i32>(x, y), 0);
  }

  @compute @workgroup_size(8, 8)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let pos = vec2<i32>(x, y);

    // Prepare trailing.
    var trail =
        trailAt(x - 1, y - 1)
        + trailAt(x, y - 1)
        + trailAt(x + 1, y - 1)
        + trailAt(x - 1, y)
        + trailAt(x + 1, y)
        + trailAt(x - 1, y + 1)
        + trailAt(x, y + 1)
        + trailAt(x + 1, y + 1);
    trail = trail / 9.5;
    trail.a = 1.0;

    // Update cellular automata.
    let current = cellAt(x, y);
    let neighbors =
        cellAt(x - 1, y - 1)
        + cellAt(x, y - 1)
        + cellAt(x + 1, y - 1)
        + cellAt(x - 1, y)
        + cellAt(x + 1, y)
        + cellAt(x - 1, y + 1)
        + cellAt(x, y + 1)
        + cellAt(x + 1, y + 1);

    var s = 0.0;
    if (current != 0 && (neighbors == 2 || neighbors == 3)) {
        s = 1.0;
        trail = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    } else if (current == 0 && neighbors == 3) {
        s = 1.0;
        trail = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    } else {
    }

    textureStore(cellsDst, pos, vec4<f32>(s, s, s, 1.0));
    textureStore(trailDst, pos, trail);
  }
);

// Create triangles to cover the screen
static const char* graphics_vertex_shader_wgsl = CODE(
  struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) coord: vec2<f32>
  }

  @vertex
  fn main(@builtin(vertex_index) idx : u32) -> VSOut {
    var data = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0),
      vec2<f32>(1.0, -1.0),
      vec2<f32>(1.0, 1.0),

      vec2<f32>(-1.0, -1.0),
      vec2<f32>(-1.0, 1.0),
      vec2<f32>(1.0, 1.0),
    );

    let pos = data[idx];

    var out : VSOut;
    out.pos = vec4<f32>(pos, 0.0, 1.0);
    out.coord.x = (pos.x + 1.0) / 2.0;
    out.coord.y = (1.0 - pos.y) / 2.0;

    return out;
  }
);

// Just write some color on each pixel
static const char* graphics_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var tex : texture_2d<f32>;
  @group(0) @binding(1) var smplr : sampler;

  fn palette(v: f32) -> vec4<f32> {
    let key = v * 8.0;
    let c = (v * 256.0) % 32.0;
    if (key < 1.0) { return vec4<f32>(0.0, 0.0, c * 2.0 / 256.0, 1.0); }
    if (key < 2.0) { return vec4<f32>(c * 8.0 / 256.0, 0.0, (64.0 - c * 2.0) / 256.0, 1.0); }
    if (key < 3.0) { return vec4<f32>(1.0, c * 8.0 / 256.0, 0.0, 1.0); }
    if (key < 4.0) { return vec4<f32>(1.0, 1.0, c * 4.0 / 256.0, 1.0); }
    if (key < 5.0) { return vec4<f32>(1.0, 1.0, (64.0 + c * 4.0) / 256.0, 1.0); }
    if (key < 6.0) { return vec4<f32>(1.0, 1.0, (128.0 + c * 4.0) / 256.0, 1.0); }
    if (key < 7.0) { return vec4<f32>(1.0, 1.0, (192.0 + c * 4.0) / 256.0, 1.0); }
    return vec4<f32>(1.0, 1.0, (224.0 + c * 4.0) / 256.0, 1.0);
  }

  @fragment
  fn main(@location(0) coord: vec2<f32>) -> @location(0) vec4<f32> {
    return palette(textureSample(tex, smplr, coord).r);
  }
);
// clang-format on
