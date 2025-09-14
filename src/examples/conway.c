#include "webgpu/wgpu_common.h"

#include <stdio.h>

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

#define COMPUTE_TEX_FORMAT (WGPUTextureFormat_RGBA8Unorm)

static struct {
  struct {
    struct wgpu_buffer_t buffer;
    struct {
      uint32_t compute_width;
      uint32_t compute_height;
    } desc;
  } uniforms;
  wgpu_texture_t textures[2];
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
  .is_forward = true,
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

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Create uniforms buffer */
  state.uniforms.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniforms buffer",
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

  for (uint32_t i = 0; i < 2; ++i) {
    wgpu_texture_t* tex = &state.textures[i];

    /* Create the texture */
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Compute part - Texture"),
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
    tex->handle = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(tex->handle);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = STRVIEW("Compute part - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    tex->view = wgpuTextureCreateView(tex->handle, &texture_view_dec);
    ASSERT(tex->view);

    /* Create sampler to sample to pick from the texture and write to the screen
     */
    tex->sampler = wgpuDeviceCreateSampler(
      wgpu_context->device,
      &(WGPUSamplerDescriptor){
        .label         = STRVIEW("Compute part - Texture sampler"),
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
  wgpu_image_to_texure(wgpu_context, state.textures[0].handle, b,
                       texture_extent, 4);
  free(b);
}

static void init_pipeline_layouts(wgpu_context_t* wgpu_context)
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
          .minBindingSize = sizeof(state.uniforms.desc),
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
      .label      = STRVIEW("Compute - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    state.compute.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(state.compute.bind_group_layout != NULL);

    /* Compute pipeline layout */
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
      .label      = STRVIEW("Graphics - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    state.graphics.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(state.graphics.bind_group_layout != NULL);

    /* Render pipeline layout */
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
  // Create 2 bind group for the compute pipeline, depending on what is the
  // current src & dst texture.
  for (uint32_t i = 0; i < 2; ++i) {
    WGPUBindGroupEntry bg_entries[3] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .buffer  = state.uniforms.buffer.buffer,
          .offset  = 0,
          .size    = state.uniforms.buffer.size,
        },
        [1] = (WGPUBindGroupEntry) {
          .binding     = 1,
          .textureView = (i == 0) ? state.textures[0].view : state.textures[1].view,
        },
        [2] = (WGPUBindGroupEntry) {
          .binding     = 2,
          .textureView = (i == 0) ? state.textures[1].view : state.textures[0].view,
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

  // Create 2 bind group for the render pipeline, depending on what is the
  // current src & dst texture.
  const uint32_t nbTextures = (uint32_t)ARRAY_SIZE(state.textures);
  for (uint32_t i = 0; i < nbTextures; ++i) {
    wgpu_texture_t* tex = &state.textures[(i + 1) % nbTextures];
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
    init_uniform_buffer(wgpu_context);
    init_textures(wgpu_context);
    init_pipeline_layouts(wgpu_context);
    init_pipelines(wgpu_context);
    init_bind_groups(wgpu_context);
    state.initialized = true;
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

  /* -- Do compute pass, where the actual effect is -- */
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
    wgpuComputePassEncoderRelease(cpass_enc);
  }

  /* -- And do the frame rendering -- */
  {
    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_dscriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                      state.is_forward ?
                                        state.graphics.bind_groups[0] :
                                        state.graphics.bind_groups[1],
                                      0, NULL);
    /* Double-triangle for fullscreen has 6 vertices */
    wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    wgpuRenderPassEncoderRelease(rpass_enc);
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

  // Textures
  wgpu_destroy_texture(&state.textures[0]);
  wgpu_destroy_texture(&state.textures[1]);

  // Graphics pipeline
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_groups[1])
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)

  // Compute pipeline
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
    .title       = "A Conway Game Of Life",
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
