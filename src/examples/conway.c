#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - A Conway Game Of Life
 *
 * A binary Conway game of life.
 *
 * Ref:
 * https://github.com/Palats/webgpu/blob/main/src/demos/conway.ts
 * https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* compute_shader_wgsl;
static const char* graphics_vertex_shader_wgsl;
static const char* graphics_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * A Conway Game Of Life example
 * -------------------------------------------------------------------------- */

static struct {
  struct wgpu_buffer_t buffer;
  struct {
    uint32_t compute_width;
    uint32_t compute_height;
  } desc;
} uniforms = {
  .desc.compute_width  = 0u,
  .desc.compute_height = 1u,
};

// Textures
static WGPUTextureFormat COMPUTE_TEX_FORMAT = WGPUTextureFormat_RGBA8Unorm;
static texture_t textures[2]                = {0};

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
static const char* example_title = "A Conway Game Of Life";
static bool prepared             = false;

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Update unfirms data
  uniforms.desc.compute_width  = wgpu_context->surface.width;
  uniforms.desc.compute_height = wgpu_context->surface.height;

  // Upload buffer to the GPU
  wgpu_queue_write_buffer(wgpu_context, uniforms.buffer.buffer, 0,
                          &uniforms.desc, uniforms.buffer.size);
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Create uniforms buffer
  uniforms.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniforms buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(uniforms.desc),
                  });

  // Update unifroms buffer data
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

  for (uint32_t i = 0; i < 2; ++i) {
    texture_t* tex = &textures[i];

    /* Create the texture */
    WGPUTextureDescriptor texture_desc = {
      .label         = "Compute part - Texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = COMPUTE_TEX_FORMAT,
      .usage
      = (i == 0) ?
          (WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding
           | WGPUTextureUsage_CopyDst) :
          (WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding),
    };
    tex->texture = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(tex->texture);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Compute part - Texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    tex->view = wgpuTextureCreateView(tex->texture, &texture_view_dec);
    ASSERT(tex->view);

    /* Create sampler to sample to pick from the texture and write to the screen
     */
    tex->sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = "Compute part - Texture sampler",
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

  /* Setup the initial texture1, with some initial data. */
  uint8_t* b = malloc(compute_width * compute_height * 4 * sizeof(uint8_t));
  ASSERT(b);
  bool has_life = false;
  uint8_t v     = 0;
  for (uint32_t y = 0; y < compute_height; ++y) {
    for (uint32_t x = 0; x < compute_width; ++x) {
      has_life                           = random_float() > 0.8f;
      v                                  = has_life ? 255 : 0;
      b[4 * (x + y * compute_width) + 0] = v;
      b[4 * (x + y * compute_width) + 1] = v;
      b[4 * (x + y * compute_width) + 2] = v;
      b[4 * (x + y * compute_width) + 3] = 255;
    }
  }
  wgpu_image_to_texure(wgpu_context, textures[0].texture, b, texture_extent, 4);
  free(b);
}

static void setup_pipeline_layouts(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0 : Uniforms */
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniforms.desc),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1 : Input compute buffer as texture */
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
         /* Binding 2 : Output compute buffer as texture */
        .binding    = 2,
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
      .label      = "Compute - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    compute.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(compute.bind_group_layout != NULL);

    /* Compute pipeline layout */
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
        /* Binding 0 : Current compute texture updated by the compute shader */
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
        /* Binding 1 : Sampler for the texture */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      }
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Graphics - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    graphics.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(graphics.bind_group_layout != NULL);

    /* Render pipeline layout */
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

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Create 2 bind group for the compute pipeline, depending on what is the
  // current src & dst texture.
  for (uint32_t i = 0; i < 2; ++i) {
    WGPUBindGroupEntry bg_entries[3] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .buffer  = uniforms.buffer.buffer,
          .offset  = 0,
          .size    = uniforms.buffer.size,
        },
        [1] = (WGPUBindGroupEntry) {
          .binding     = 1,
          .textureView = (i == 0) ? textures[0].view : textures[1].view,
        },
        [2] = (WGPUBindGroupEntry) {
          .binding     = 2,
          .textureView = (i == 0) ? textures[1].view : textures[0].view,
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

  // Create 2 bind group for the render pipeline, depending on what is the
  // current src & dst texture.
  const uint32_t nbTextures = (uint32_t)ARRAY_SIZE(textures);
  for (uint32_t i = 0; i < nbTextures; ++i) {
    texture_t* tex = &textures[(i + 1) % nbTextures];
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
    /* Compute shader */
    wgpu_shader_t conway_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      /* Compute shader WGSL */
                      .label            = "Compute - Shader WGSL",
                      .wgsl_code.source = compute_shader_wgsl,
                      .entry            = "main",
                    });

    /* Create compute pipeline */
    compute.pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "Effect - Compute pipeline",
        .layout  = compute.pipeline_layout,
        .compute = conway_comp_shader.programmable_stage_descriptor,
      });

    /* Partial cleanup */
    wgpu_shader_release(&conway_comp_shader);
  }

  /* Graphics pipeline */
  {
    /* Primitive state */
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    /* Color target state */
    WGPUBlendState blend_state              = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* Vertex state */
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
          wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader WGSL */
              .label            = "Graphics - Vertex shader WGSL",
              .wgsl_code.source = graphics_vertex_shader_wgsl,
              .entry            = "main",
            },
            .buffer_count = 0,
            .buffers = NULL,
          });

    /* Fragment state */
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
          wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader WGSL */
              .label            = "Graphics - Fragment shader WGSL",
              .wgsl_code.source = graphics_fragment_shader_wgsl,
              .entry            = "main",
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
    graphics.pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label       = "Conway - Graphics pipeline",
                              .layout      = graphics.pipeline_layout,
                              .primitive   = primitive_state,
                              .vertex      = vertex_state,
                              .fragment    = &fragment_state,
                              .multisample = multisample_state,
                            });

    /* Partial cleanup */
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

  /* -- Do compute pass, where the actual effect is -- */
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

  /* -- And do the frame rendering -- */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(
      wgpu_context->rpass_enc, 0,
      is_forward ? graphics.bind_groups[0] : graphics.bind_groups[1], 0, NULL);
    /* Double-triangle for fullscreen has 6 vertices */
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

  // Submit command buffer to queue
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
  wgpu_destroy_texture(&textures[0]);
  wgpu_destroy_texture(&textures[1]);

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

void example_conway(int argc, char* argv[])
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
  struct UniformsDesc {
    computeWidth : u32,
    computeHeight : u32
  };

  @group(0) @binding(0) var<uniform> uniforms : UniformsDesc;
  @group(0) @binding(1) var srcTexture : texture_2d<f32>;
  @group(0) @binding(2) var dstTexture : texture_storage_2d<rgba8unorm, write>;

  fn isOn(x: i32, y: i32) -> i32 {
    let v = textureLoad(srcTexture, vec2<i32>(x, y), 0);
    if (v.r < 0.5) {
      return 0;
    }
    return 1;
  }

  @compute @workgroup_size(8, 8)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    // Guard against out-of-bounds work group sizes
    if (global_id.x >= uniforms.computeWidth || global_id.y >= uniforms.computeHeight) {
        return;
    }

    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let current = isOn(x, y);
    let neighbors =
        isOn(x - 1, y - 1)
      + isOn(x, y - 1)
      + isOn(x + 1, y - 1)
      + isOn(x - 1, y)
      + isOn(x + 1, y)
      + isOn(x - 1, y + 1)
      + isOn(x, y + 1)
      + isOn(x + 1, y + 1);

    var s = 0.0;
    if (current != 0 && (neighbors == 2 || neighbors == 3)) {
      s = 1.0;
  }
   if (current == 0 && neighbors == 3) {
      s = 1.0;
    }
    textureStore(dstTexture, vec2<i32>(x, y), vec4<f32>(s, s, s, 1.0));
  }
);

static const char* graphics_vertex_shader_wgsl = CODE(
  struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) coord: vec2<f32>
  };

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

static const char* graphics_fragment_shader_wgsl = CODE(
  struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) coord: vec2<f32>
  };

  @group(0) @binding(0) var computeTexture : texture_2d<f32>;
  @group(0) @binding(1) var dstSampler : sampler;

  @fragment
  fn main(inp: VSOut) -> @location(0) vec4<f32> {
    return textureSample(computeTexture, dstSampler, inp.coord);
  }
);
// clang-format on
