#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - A Conway Game Of Life
 *
 * A binary Conway game of life.
 *
 * Ref:
 * https://github.com/Palats/webgpu/blob/main/src/demos/conway.ts
 * https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
 * -------------------------------------------------------------------------- */

static struct {
  struct {
    WGPUBuffer handle;
    uint64_t size;
  } buffer;
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
static texture_t textures[2];

// Bind group layouts
static struct {
  WGPUBindGroupLayout compute;
  WGPUBindGroupLayout render;
} bind_group_layouts;

// Bind groups
static struct {
  WGPUBindGroup compute[2];
  WGPUBindGroup render[2];
} bind_groups;

// Pipeline layouts
static struct {
  WGPUPipelineLayout compute;
  WGPUPipelineLayout render;
} pipeline_layouts;

// Pipelines
static struct {
  WGPUComputePipeline compute;
  WGPURenderPipeline render;
} pipelines;

// Other variables
static const char* example_title = "A Conway Game Of Life";
static bool prepared             = false;

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  uniforms.buffer.size                     = sizeof(uniforms.desc);
  WGPUBufferDescriptor texture_buffer_desc = {
    .label            = "Compute uniforms buffer",
    .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    .size             = uniforms.buffer.size,
    .mappedAtCreation = false,
  };
  uniforms.buffer.handle
    = wgpuDeviceCreateBuffer(wgpu_context->device, &texture_buffer_desc);
  ASSERT(uniforms.buffer.handle)
}

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

    // Create the texture
    WGPUTextureDescriptor texture_desc = {
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
    ASSERT(tex->texture)

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    tex->view = wgpuTextureCreateView(tex->texture, &texture_view_dec);
    ASSERT(tex->view)

    // Create sampler to sample from
    tex->sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUFilterMode_Linear,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(tex->sampler)
  }

  // Setup the initial texture1, with some initial data.
  uint8_t* b = malloc(compute_width * compute_height * 4 * sizeof(uint8_t));
  ASSERT(b)
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
        // Uniforms
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniforms.desc),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Input compute buffer as texture
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
         // Output compute buffer as texture
        .binding = 2,
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
      .label      = "compute pipeline main layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    bind_group_layouts.compute
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.compute != NULL)

    // Compute pipeline layout
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "compute pipeline layouts",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &bind_group_layouts.compute,
    };
    pipeline_layouts.compute = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(pipeline_layouts.compute != NULL)
  }

  /* Rendering pipeline layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Current compute texture updated by the compute shader
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
        // Sampler for  the texture
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      }
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "rendering pipeline main layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    bind_group_layouts.render
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.render != NULL)

    // Render pipeline layout
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "rendering pipeline layouts",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &bind_group_layouts.render,
    };
    pipeline_layouts.render = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(pipeline_layouts.render != NULL)
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Create 2 bind group for the compute pipeline, depending on what is the
  // current src & dst texture.
  for (uint32_t i = 0; i < 2; ++i) {
    WGPUBindGroupEntry bg_entries[3] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .buffer = uniforms.buffer.handle,
          .offset = 0,
          .size = uniforms.buffer.size,
        },
        [1] = (WGPUBindGroupEntry) {
          .binding = 1,
          .textureView = (i == 0) ? textures[1].view : textures[2].view,
        },
        [2] = (WGPUBindGroupEntry) {
          .binding = 2,
          .textureView = (i == 0) ? textures[2].view : textures[1].view,
        },
      };

    bind_groups.compute[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .layout = wgpuComputePipelineGetBindGroupLayout(pipelines.compute, 0),
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(bind_groups.compute[i] != NULL)
  }

  // Create 2 bind group for the render pipeline, depending on what is the
  // current src & dst texture.
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(textures); ++i) {
    texture_t* tex = &textures[(i + 1) % (uint32_t)ARRAY_SIZE(textures)];
    WGPUBindGroupEntry bg_entries[2] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .textureView = tex->view,
        },
        [1] = (WGPUBindGroupEntry) {
          .binding = 1,
          .sampler = tex->sampler,
        },
      };

    bind_groups.render[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .layout     = wgpuRenderPipelineGetBindGroupLayout(pipelines.render, 0),
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(bind_groups.render[i] != NULL)
  }
}
