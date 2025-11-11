#include "webgpu/wgpu_common.h"

#include "common_shaders.h"

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

#include <stdbool.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Image Blur
 *
 * This example shows how to blur an image using a WebGPU compute shader.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/tree/main/src/sample/imageBlur
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* blur_compute_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Image Blur example
 * -------------------------------------------------------------------------- */

/* State struct */
static struct {
  WGPUComputePipeline blur_pipeline;
  WGPURenderPipeline fullscreen_quad_pipeline;
  wgpu_texture_t texture;
  WGPUSampler texture_sampler;
  wgpu_texture_t blur_textures[2];
  wgpu_buffer_t uniform_buffers[2];
  uint8_t file_buffer[512 * 512 * 4];
  uint32_t uniform_buffer_data[2];
  wgpu_buffer_t blur_params_buffer;
  WGPUBindGroup compute_constants_bind_group;
  WGPUBindGroup compute_bind_groups[3];
  WGPUBindGroup show_result_bind_group;
  /* Contants from the blur.wgsl shader. */
  const uint32_t tile_dim;
  const uint32_t batch[2];
  // Settings
  struct {
    int32_t filter_size;
    int32_t iterations;
  } settings;
  uint32_t block_dim;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  bool initialized;
} state = {
  .uniform_buffer_data = {0, 1},
  .tile_dim = 128,
  .batch    = {4, 4},
  .settings = {
    .filter_size = 15,
    .iterations  = 2,
  },
  .block_dim    = 1,
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
  },
};

/**
 * @brief The fetch-callback is called by sokol_fetch.h when the data is loaded,
 * or when an error has occurred.
 */
static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("File fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* The file data has been fetched, since we provided a big-enough buffer we
   * can be sure that all data has been loaded here */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D) {
        .width              = img_width,
        .height             = img_height,
        .depthOrArrayLayers = 4,
      },
      .format = texture->desc.format,
      .usage  = texture->desc.usage,
      .pixels = {
        .ptr  = pixels,
        .size = img_width * img_height * 4,
      },
    };
    texture->desc.is_dirty = true;
  }
}

static void init_texture(wgpu_context_t* wgpu_context)
{
  state.texture = wgpu_create_color_bars_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
               | WGPUTextureUsage_RenderAttachment,
    });
  wgpu_texture_t* texture = &state.texture;
  sfetch_send(&(sfetch_request_t){
    .path      = "assets/textures/Di-3d.png",
    .callback  = fetch_callback,
    .buffer    = SFETCH_RANGE(state.file_buffer),
    .user_data = {
      .ptr  = &texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });
}

static void init_texture_sampler(wgpu_context_t* wgpu_context)
{
  WGPUSamplerDescriptor sampler_desc = {
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
    .addressModeW  = WGPUAddressMode_Repeat,
    .magFilter     = WGPUFilterMode_Linear,
    .minFilter     = WGPUFilterMode_Nearest,
    .mipmapFilter  = WGPUMipmapFilterMode_Linear,
    .lodMinClamp   = 0,
    .lodMaxClamp   = 1,
    .compare       = WGPUCompareFunction_Undefined,
    .maxAnisotropy = 1,
  };
  state.texture_sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(state.texture_sampler != NULL);
}

static void init_blur_textures(wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < ARRAY_SIZE(state.blur_textures); ++i) {
    /* Blur texture */
    WGPU_RELEASE_RESOURCE(Texture, state.blur_textures[i].handle)
    state.blur_textures[i].handle = wgpuDeviceCreateTexture(
      wgpu_context->device,
      &(WGPUTextureDescriptor){
        .label = STRVIEW("Image Blur - Texture"),
        .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding
                 | WGPUTextureUsage_TextureBinding,
        .dimension     = WGPUTextureDimension_2D,
        .size          = (WGPUExtent3D) {
          .width               = state.texture.desc.extent.width,
          .height              = state.texture.desc.extent.height,
          .depthOrArrayLayers  = state.texture.desc.extent.depthOrArrayLayers,
        },
        .format        = state.texture.desc.format,
        .mipLevelCount = 1,
        .sampleCount   = 1,
      });
    ASSERT(state.blur_textures[i].handle != NULL);
    /* Blur texture view */
    WGPU_RELEASE_RESOURCE(TextureView, state.blur_textures[i].view)
    state.blur_textures[i].view
      = wgpuTextureCreateView(state.blur_textures[i].handle,
                              &(WGPUTextureViewDescriptor){
                                .label  = STRVIEW("Image Blur - Texture view"),
                                .format = state.texture.desc.format,
                                .dimension       = WGPUTextureViewDimension_2D,
                                .baseMipLevel    = 0,
                                .mipLevelCount   = 1,
                                .baseArrayLayer  = 0,
                                .arrayLayerCount = 1,
                              });
    ASSERT(state.blur_textures[i].view != NULL);
  }
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* buffer 0 and buffer 1 */
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(state.uniform_buffers); ++i) {
    state.uniform_buffer_data[i] = i;
    state.uniform_buffers[i]     = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
            .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
            .size         = 4,
            .initial.data = &state.uniform_buffer_data[i],
      });
  }

  /* Compute shader blur parameters */
  state.blur_params_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute shader blur parameters - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = 8,
                  });
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Cleanup */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.compute_constants_bind_group)
    WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_groups[0])
    WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_groups[1])
    WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_groups[2])
    WGPU_RELEASE_RESOURCE(BindGroup, state.show_result_bind_group)
  }

  /* Compute constants bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0 : texture sampler */
        .binding = 0,
        .sampler = state.texture_sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1 : blur parameters */
        .binding = 1,
        .buffer  = state.blur_params_buffer.buffer,
        .offset  = 0,
        .size    = state.blur_params_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label  = STRVIEW("Compute constants - Bind group"),
      .layout = wgpuComputePipelineGetBindGroupLayout(state.blur_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.compute_constants_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.compute_constants_bind_group != NULL);
  }

  /* Compute bind group 0 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0 : texture */
        .binding     = 1,
        .textureView = state.texture.view,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1 : blur texture */
        .binding     = 2,
        .textureView = state.blur_textures[0].view,
      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 2 : uniform buffer */
        .binding = 3,
        .buffer  = state.uniform_buffers[0].buffer,
        .offset  = 0,
        .size    = state.uniform_buffers[0].size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label  = STRVIEW("Compute - Bind group 0"),
      .layout = wgpuComputePipelineGetBindGroupLayout(state.blur_pipeline, 1),
      .entryCount = 3,
      .entries    = bg_entries,
    };
    state.compute_bind_groups[0]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.compute_bind_groups[0] != NULL);
  }

  /* Compute bind group 1 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 1 : texture */
        .binding     = 1,
        .textureView = state.blur_textures[0].view,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 2 : blur texture */
        .binding     = 2,
        .textureView = state.blur_textures[1].view,
      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 3 : uniform buffer */
        .binding = 3,
        .buffer  = state.uniform_buffers[1].buffer,
        .offset  = 0,
        .size    = state.uniform_buffers[1].size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label  = STRVIEW("Compute - Bind group 1"),
      .layout = wgpuComputePipelineGetBindGroupLayout(state.blur_pipeline, 1),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.compute_bind_groups[1]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.compute_bind_groups[1] != NULL);
  }

  /* Compute bind group 2 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 1 : texture */
        .binding     = 1,
        .textureView = state.blur_textures[1].view,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 2 : blur texture */
        .binding     = 2,
        .textureView = state.blur_textures[0].view,
      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 3 : uniform buffer */
        .binding = 3,
        .buffer  = state.uniform_buffers[0].buffer,
        .offset  = 0,
        .size    = state.uniform_buffers[0].size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label  = STRVIEW("Compute - Bind group 2"),
      .layout = wgpuComputePipelineGetBindGroupLayout(state.blur_pipeline, 1),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.compute_bind_groups[2]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.compute_bind_groups[2] != NULL);
  }

  /* Uniform bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0 : texture sampler */
        .binding = 0,
        .sampler = state.texture_sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1 : blur texture */
        .binding     = 1,
        .textureView = state.blur_textures[1].view,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label = STRVIEW("Uniform - Bind group"),
      .layout
      = wgpuRenderPipelineGetBindGroupLayout(state.fullscreen_quad_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.show_result_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.show_result_bind_group != NULL);
  }
}

/* Create the compute & graphics pipelines */
static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Blur compute pipeline */
  {
    /* Compute shader */
    WGPUShaderModule blur_comp_shader_module = wgpu_create_shader_module(
      wgpu_context->device, blur_compute_shader_wgsl);

    /* Compute pipeline */
    state.blur_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Image Blur - Compute pipeline"),
        .compute = {
          .module     = blur_comp_shader_module,
          .entryPoint = STRVIEW("main"),
        },
      });
    ASSERT(state.blur_pipeline != NULL);

    /* Partial clean-up */
    wgpuShaderModuleRelease(blur_comp_shader_module);
  }

  /* Fullscreen quad render pipeline */
  {
    WGPUShaderModule fullscreen_textured_quad_shader_module
      = wgpu_create_shader_module(wgpu_context->device,
                                  fullscreen_textured_quad_wgsl);

    /* Color target state */
    WGPUBlendState blend_state = wgpu_create_blend_state(true);

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Fullscreen - Quad pipeline"),
      .vertex = {
        .module      = fullscreen_textured_quad_shader_module,
        .entryPoint  = STRVIEW("vert_main"),
      },
      .fragment = &(WGPUFragmentState) {
        .module      = fullscreen_textured_quad_shader_module,
        .entryPoint  = STRVIEW("frag_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState) {
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CW,
        .cullMode  = WGPUCullMode_Back,
      },
      .multisample = {
         .count = 1,
         .mask  = 0xffffffff
      },
    };

    state.fullscreen_quad_pipeline
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.fullscreen_quad_pipeline != NULL);

    wgpuShaderModuleRelease(fullscreen_textured_quad_shader_module);
  }
}

static int round_up_to_odd(int value, int min, int max)
{
  return MIN(MAX(min, (value % 2 == 0) ? value + 1 : value), max);
}

static void update_settings(wgpu_context_t* wgpu_context)
{
  state.settings.filter_size
    = round_up_to_odd(state.settings.filter_size, 1, 33);
  state.block_dim = state.tile_dim - state.settings.filter_size;

  state.uniform_buffer_data[0] = state.settings.filter_size + 1;
  state.uniform_buffer_data[1] = state.block_dim;

  /* Map uniform buffer and update the blur parameters */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.blur_params_buffer.buffer, 0,
                       &state.uniform_buffer_data,
                       sizeof(state.uniform_buffer_data));
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Recreate texture when pixel data loaded */
  if (state.texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.texture);
    FREE_TEXTURE_PIXELS(state.texture);
    init_blur_textures(wgpu_context);
    init_bind_groups(wgpu_context);
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass */
  {
    uint32_t image_width  = state.texture.desc.extent.width;
    uint32_t image_height = state.texture.desc.extent.height;

    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass_enc, state.blur_pipeline);
    wgpuComputePassEncoderSetBindGroup(
      cpass_enc, 0, state.compute_constants_bind_group, 0, NULL);

    wgpuComputePassEncoderSetBindGroup(cpass_enc, 1,
                                       state.compute_bind_groups[0], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      cpass_enc, ceil((float)image_width / state.block_dim),
      ceil((float)image_height / state.batch[1]), 1);

    wgpuComputePassEncoderSetBindGroup(cpass_enc, 1,
                                       state.compute_bind_groups[1], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      cpass_enc, ceil((float)image_height / state.block_dim),
      ceil((float)image_width / state.batch[1]), 1);

    for (uint32_t i = 0; i < (uint32_t)state.settings.iterations - 1; ++i) {
      wgpuComputePassEncoderSetBindGroup(cpass_enc, 1,
                                         state.compute_bind_groups[2], 0, NULL);
      wgpuComputePassEncoderDispatchWorkgroups(
        cpass_enc, ceil((float)image_width / state.block_dim),
        ceil((float)image_height / state.batch[1]), 1);

      wgpuComputePassEncoderSetBindGroup(cpass_enc, 1,
                                         state.compute_bind_groups[1], 0, NULL);
      wgpuComputePassEncoderDispatchWorkgroups(
        cpass_enc, ceil((float)image_height / state.block_dim),
        ceil((float)image_width / state.batch[1]), 1);
    }

    wgpuComputePassEncoderEnd(cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, cpass_enc)
  }

  /* Fullscreen quad pipeline */
  {
    state.color_attachment.view = wgpu_context->swapchain_view;

    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.fullscreen_quad_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                      state.show_result_bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}
static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
    });
    init_texture(wgpu_context);
    init_texture_sampler(wgpu_context);
    init_blur_textures(wgpu_context);
    init_pipelines(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_bind_groups(wgpu_context);
    update_settings(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  WGPU_RELEASE_RESOURCE(TextureView, state.blur_textures[0].view)
  WGPU_RELEASE_RESOURCE(TextureView, state.blur_textures[1].view)
  WGPU_RELEASE_RESOURCE(Texture, state.blur_textures[0].handle)
  WGPU_RELEASE_RESOURCE(Texture, state.blur_textures[1].handle)
  wgpu_destroy_texture(&state.texture);
  WGPU_RELEASE_RESOURCE(Sampler, state.texture_sampler)

  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_constants_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_groups[1])
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_groups[2])
  WGPU_RELEASE_RESOURCE(BindGroup, state.show_result_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers[0].buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers[1].buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.blur_params_buffer.buffer)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.blur_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.fullscreen_quad_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Image Blur",
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
static const char* blur_compute_shader_wgsl = CODE(
  struct Params {
    filterDim : i32,
    blockDim : u32,
  }

  @group(0) @binding(0) var samp : sampler;
  @group(0) @binding(1) var<uniform> params : Params;
  @group(1) @binding(1) var inputTex : texture_2d<f32>;
  @group(1) @binding(2) var outputTex : texture_storage_2d<rgba8unorm, write>;

  struct Flip {
    value : u32,
  }
  @group(1) @binding(3) var<uniform> flip : Flip;

  // This shader blurs the input texture in one direction, depending on whether
  // |flip.value| is 0 or 1.
  // It does so by running (128 / 4) threads per workgroup to load 128
  // texels into 4 rows of shared memory. Each thread loads a
  // 4 x 4 block of texels to take advantage of the texture sampling
  // hardware.
  // Then, each thread computes the blur result by averaging the adjacent texel values
  // in shared memory.
  // Because we're operating on a subset of the texture, we cannot compute all of the
  // results since not all of the neighbors are available in shared memory.
  // Specifically, with 128 x 128 tiles, we can only compute and write out
  // square blocks of size 128 - (filterSize - 1). We compute the number of blocks
  // needed in Javascript and dispatch that amount.

  var<workgroup> tile : array<array<vec3f, 128>, 4>;

  @compute @workgroup_size(32, 1, 1)
  fn main(
    @builtin(workgroup_id) WorkGroupID : vec3u,
    @builtin(local_invocation_id) LocalInvocationID : vec3u
  ) {
    let filterOffset = (params.filterDim - 1) / 2;
    let dims = vec2i(textureDimensions(inputTex, 0));
    let baseIndex = vec2i(WorkGroupID.xy * vec2(params.blockDim, 4) +
                              LocalInvocationID.xy * vec2(4, 1))
                    - vec2(filterOffset, 0);

    for (var r = 0; r < 4; r++) {
      for (var c = 0; c < 4; c++) {
        var loadIndex = baseIndex + vec2(c, r);
        if (flip.value != 0u) {
          loadIndex = loadIndex.yx;
        }

        tile[r][4 * LocalInvocationID.x + u32(c)] = textureSampleLevel(
          inputTex,
          samp,
          (vec2f(loadIndex) + vec2f(0.5, 0.5)) / vec2f(dims),
          0.0
        ).rgb;
      }
    }

    workgroupBarrier();

    for (var r = 0; r < 4; r++) {
      for (var c = 0; c < 4; c++) {
        var writeIndex = baseIndex + vec2(c, r);
        if (flip.value != 0) {
          writeIndex = writeIndex.yx;
        }

        let center = i32(4 * LocalInvocationID.x) + c;
        if (center >= filterOffset &&
            center < 128 - filterOffset &&
            all(writeIndex < dims)) {
          var acc = vec3(0.0, 0.0, 0.0);
          for (var f = 0; f < params.filterDim; f++) {
            var i = center + f - filterOffset;
            acc = acc + (1.0 / f32(params.filterDim)) * tile[r][i];
          }
          textureStore(outputTex, writeIndex, vec4(acc, 1.0));
        }
      }
    }
  }
);
// clang-format on
