#include "webgpu/wgpu_common.h"

#include <stdio.h>

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

#define COMPUTE_TEX_FORMAT (WGPUTextureFormat_RGBA8Unorm)
#define COMPUTE_TEX_BYTES (4u) /* Bytes per pixel in compute. */

static struct {
  struct {
    struct wgpu_buffer_t buffer;
    struct {
      uint32_t compute_width;
      uint32_t compute_height;
    } desc;
  } uniforms;
  struct {
    /* Swapchain for the cellular automata progression */
    wgpu_texture_t cells[2];
    // Swap chain for the intermediate compute effect on top of the cellular
    // automata
    wgpu_texture_t trails[2];
  } textures;
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_groups[2];
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
  } graphics;
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_groups[2];
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
  } compute;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  WGPUBool is_forward;
  WGPUBool initialized;
} state = {
  .uniforms.desc.compute_width  = 0u,
  .uniforms.desc.compute_height = 1u,
  .color_attachment = {
  .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_dscriptor = {
   .colorAttachmentCount = 1,
   .colorAttachments     = &state.color_attachment,
  },
  .is_forward = 1,
};

static void update_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Update unfirms data */
  state.uniforms.desc.compute_width  = wgpu_context->width;
  state.uniforms.desc.compute_height = wgpu_context->height;

  // Upload buffer to the GPU
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniforms.buffer.buffer, 0,
                       &state.uniforms.desc, state.uniforms.buffer.size);
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Create uniforms buffer */
  state.uniforms.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.uniforms.desc),
                  });

  /* Update unifroms buffer data */
  update_uniform_buffer(wgpu_context);
}

/* Textures, used for compute part */
static void init_textures(wgpu_context_t* wgpu_context)
{
  const uint32_t compute_width  = wgpu_context->width;
  const uint32_t compute_height = wgpu_context->height;

  WGPUExtent3D texture_extent = {
    .width              = compute_width,
    .height             = compute_height,
    .depthOrArrayLayers = 1,
  };

  wgpu_texture_t* textureArray[4] = {
    &state.textures.cells[0],  //
    &state.textures.cells[1],  //
    &state.textures.trails[0], //
    &state.textures.trails[1], //
  };

  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(textureArray); ++i) {
    wgpu_texture_t* tex = textureArray[i];

    /* Create the texture */
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Compute - Texture"),
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
    tex->handle = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(tex->handle);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = STRVIEW("Compute - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    tex->view = wgpuTextureCreateView(tex->handle, &texture_view_dec);
    ASSERT(tex->view);

    // Create sampler to sample to pick from the texture and write to the screen
    tex->sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label = STRVIEW("Compute - Texture sampler"),
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
    WGPUBool has_life = 0;
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
    wgpu_image_to_texure(wgpu_context, state.textures.cells[0].handle, b,
                         texture_extent, COMPUTE_TEX_BYTES);
    free(b);
  }
}

static void init_pipeline_layouts(wgpu_context_t* wgpu_context)
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
          .multisampled  = 0,
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
          .multisampled  = 0,
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
      .label      = STRVIEW("Compute pipeline main - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    state.compute.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(state.compute.bind_group_layout != NULL);

    // Compute pipeline layout
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = STRVIEW("Compute - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.compute.bind_group_layout,
    };
    state.compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(state.compute.pipeline_layout != NULL);
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
          .multisampled  = 0,
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
      .label      = STRVIEW("Rendering pipeline main - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    state.graphics.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(state.graphics.bind_group_layout != NULL);

    // Render pipeline layout
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = STRVIEW("Rendering - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.graphics.bind_group_layout,
    };
    state.graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(state.graphics.pipeline_layout != NULL);
  }
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Compute binding group for rendering: */
  /*    1 -> 2                            */
  /*    2 -> 1                            */
  for (uint32_t i = 0; i < 2; ++i) {
    WGPUBindGroupEntry bg_entries[4] = {
        [0] = (WGPUBindGroupEntry) {
          .binding     = 0,
          .textureView = (i == 0) ? state.textures.cells[0].view : state.textures.cells[1].view,
        },
        [1] = (WGPUBindGroupEntry) {
          .binding     = 1,
          .textureView = (i == 0) ? state.textures.cells[1].view : state.textures.cells[0].view,
        },
        [2] = (WGPUBindGroupEntry) {
          .binding     = 2,
          .textureView = (i == 0) ? state.textures.trails[0].view : state.textures.trails[1].view,
        },
        [3] = (WGPUBindGroupEntry) {
          .binding     = 3,
          .textureView = (i == 0) ? state.textures.trails[1].view : state.textures.trails[0].view,
        },
      };

    state.compute.bind_groups[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Compute - Bind group"),
                              .layout = wgpuComputePipelineGetBindGroupLayout(
                                state.compute.pipeline, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.compute.bind_groups[i] != NULL);
  }

  /* Create 2 bind group for the render pipeline: */
  /*    1 -> 2                                    */
  /*    2 -> 1                                    */
  wgpu_texture_t* trailTextureArray[2] = {
    &state.textures.trails[0], /* 1 -> 2 */
    &state.textures.trails[1], /* 2 -> 1 */
  };
  const uint32_t nbTextures = (uint32_t)ARRAY_SIZE(trailTextureArray);
  for (uint32_t i = 0; i < nbTextures; ++i) {
    wgpu_texture_t* tex = trailTextureArray[(i + 1) % nbTextures];
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

    state.graphics.bind_groups[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Graphics - Bind group"),
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                state.graphics.pipeline, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.graphics.bind_groups[i] != NULL);
  }
}

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline */
  {
    /* Compute shader */
    WGPUShaderModule comp_shader_module
      = wgpu_create_shader_module(wgpu_context->device, compute_shader_wgsl);

    /* Create compute pipeline */
    state.compute.pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Effect - Compute pipeline"),
        .layout  = state.compute.pipeline_layout,
        .compute = {
          .module     = comp_shader_module,
          .entryPoint = STRVIEW("main"),
        },
      });
    ASSERT(state.compute.pipeline != NULL);

    /* Partial cleanup */
    wgpuShaderModuleRelease(comp_shader_module);
  }

  /* Graphics pipeline */
  {
    WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
      wgpu_context->device, graphics_vertex_shader_wgsl);
    WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
      wgpu_context->device, graphics_fragment_shader_wgsl);

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Conway - Render pipeline"),
      .layout = state.graphics.pipeline_layout,
      .vertex = {
        .module      = vert_shader_module,
        .entryPoint  = STRVIEW("main"),
      },
      .fragment = &(WGPUFragmentState) {
        .entryPoint  = STRVIEW("main"),
        .module      = frag_shader_module,
        .targetCount = 1,
        .targets = &(WGPUColorTargetState) {
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff
      },
    };

    state.graphics.pipeline
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.graphics.pipeline != NULL);

    wgpuShaderModuleRelease(vert_shader_module);
    wgpuShaderModuleRelease(frag_shader_module);
  }
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    init_uniform_buffers(wgpu_context);
    init_textures(wgpu_context);
    init_pipeline_layouts(wgpu_context);
    init_pipelines(wgpu_context);
    init_bind_groups(wgpu_context);
    state.initialized = 1;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* -- Frame compute -- */
  {
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass_enc, state.compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(cpass_enc, 0,
                                       state.is_forward ?
                                         state.compute.bind_groups[0] :
                                         state.compute.bind_groups[1],
                                       0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      cpass_enc, (uint32_t)ceil(state.uniforms.desc.compute_width / 8.0f),
      (uint32_t)ceil(state.uniforms.desc.compute_height / 8.0f), 1);
    wgpuComputePassEncoderEnd(cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, cpass_enc)
  }

  /* -- Frame rendering -- */
  {
    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_dscriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                      state.is_forward ?
                                        state.graphics.bind_groups[0] :
                                        state.graphics.bind_groups[1],
                                      0, NULL);
    wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
  }

  /* Get command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Switch for next frame */
  state.is_forward = !state.is_forward;

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Textures */
  wgpu_destroy_texture(&state.textures.cells[0]);
  wgpu_destroy_texture(&state.textures.cells[1]);
  wgpu_destroy_texture(&state.textures.trails[0]);
  wgpu_destroy_texture(&state.textures.trails[1]);

  /* Graphics pipeline */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_groups[1])
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)

  /* Compute pipeline */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute.bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute.bind_groups[1])
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "A Conway Game Of Life With Paletted Blurring Over Time",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
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
