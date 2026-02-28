#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

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

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <cglm/cglm.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader Image Load/Store
 *
 * Uses a compute shader to apply different convolution kernels (and effects) on
 * an input image in realtime.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/computeshader
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* texture_vertex_shader_wgsl;
static const char* texture_fragment_shader_wgsl;
static const char* emboss_compute_shader_wgsl;
static const char* edgedetect_compute_shader_wgsl;
static const char* sharpen_compute_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Compute Shader example
 * -------------------------------------------------------------------------- */

/* Vertex layout */
typedef struct vertex_t {
  float pos[3];
  float uv[2];
} vertex_t;

/* State struct */
static struct {
  /* Textures */
  wgpu_texture_t color_map;
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
    WGPUSampler sampler;
    WGPUExtent3D size;
  } compute_target;

  /* Vertex/Index buffers */
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  uint32_t index_count;

  /* Uniform buffer */
  WGPUBuffer uniform_buffer;
  struct {
    mat4 projection;
    mat4 model_view;
  } ubo_vs;

  /* Graphics resources */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group_pre_compute;
    WGPUBindGroup bind_group_post_compute;
    WGPURenderPipeline pipeline;
    WGPUPipelineLayout pipeline_layout;
  } graphics;

  /* Compute resources */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipelines[3];
    int32_t pipeline_index;
  } compute;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* File loading */
  uint8_t file_buffer[2048 * 2048 * 4];

  /* ImGui */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .compute = {
    .pipeline_index = 0,
  },
};

static const char* shader_names[3] = {"Emboss", "Edge Detect", "Sharpen"};

/* -- Fetch callback for texture loading ----------------------------------- */

static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Texture fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_width, img_height, num_channels;
  stbi_uc* pixels
    = stbi_load_from_memory(response->data.ptr, (int)response->data.size,
                            &img_width, &img_height, &num_channels, 4);
  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = (uint32_t)img_width,
        .height             = (uint32_t)img_height,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding
               | WGPUTextureUsage_StorageBinding,
      .pixels = {
        .ptr  = pixels,
        .size = (size_t)(img_width * img_height * 4),
      },
    };
    texture->desc.is_dirty = true;
  }
}

/* -- Texture loading ------------------------------------------------------ */

static void init_texture(wgpu_context_t* wgpu_context)
{
  state.color_map = wgpu_create_color_bars_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding
               | WGPUTextureUsage_StorageBinding,
    });

  wgpu_texture_t* texture = &state.color_map;
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

/* -- Compute target texture ----------------------------------------------- */

static void init_compute_target(wgpu_context_t* wgpu_context, uint32_t width,
                                uint32_t height)
{
  WGPU_RELEASE_RESOURCE(Texture, state.compute_target.handle)
  WGPU_RELEASE_RESOURCE(TextureView, state.compute_target.view)
  WGPU_RELEASE_RESOURCE(Sampler, state.compute_target.sampler)

  state.compute_target.size = (WGPUExtent3D){
    .width              = width,
    .height             = height,
    .depthOrArrayLayers = 1,
  };

  state.compute_target.handle = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label = STRVIEW("Compute target - Texture"),
                            .usage = WGPUTextureUsage_TextureBinding
                                     | WGPUTextureUsage_StorageBinding,
                            .dimension     = WGPUTextureDimension_2D,
                            .size          = state.compute_target.size,
                            .format        = WGPUTextureFormat_RGBA8Unorm,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                          });
  ASSERT(state.compute_target.handle != NULL)

  state.compute_target.view
    = wgpuTextureCreateView(state.compute_target.handle,
                            &(WGPUTextureViewDescriptor){
                              .label = STRVIEW("Compute target - Texture view"),
                              .format          = WGPUTextureFormat_RGBA8Unorm,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                            });
  ASSERT(state.compute_target.view != NULL)

  state.compute_target.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label        = STRVIEW("Compute target - Sampler"),
                            .addressModeU = WGPUAddressMode_ClampToEdge,
                            .addressModeV = WGPUAddressMode_ClampToEdge,
                            .addressModeW = WGPUAddressMode_ClampToEdge,
                            .minFilter    = WGPUFilterMode_Linear,
                            .magFilter    = WGPUFilterMode_Linear,
                            .mipmapFilter = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp  = 0.0f,
                            .lodMaxClamp  = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.compute_target.sampler != NULL)
}

/* -- Vertex / Index buffers ----------------------------------------------- */

static void init_buffers(wgpu_context_t* wgpu_context)
{
  // clang-format off
  static const vertex_t vertices[4] = {
    {.pos = { 1.0f, -1.0f, 0.0f}, .uv = {1.0f, 1.0f}},
    {.pos = {-1.0f, -1.0f, 0.0f}, .uv = {0.0f, 1.0f}},
    {.pos = {-1.0f,  1.0f, 0.0f}, .uv = {0.0f, 0.0f}},
    {.pos = { 1.0f,  1.0f, 0.0f}, .uv = {1.0f, 0.0f}},
  };
  // clang-format on

  static const uint32_t indices[6] = {0, 1, 2, 2, 3, 0};

  state.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertices),
                    .initial.data = vertices,
                  });

  state.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(indices),
                    .initial.data = indices,
                  });
  state.index_count = (uint32_t)ARRAY_SIZE(indices);
}

/* -- Uniform buffer ------------------------------------------------------- */

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpu_create_buffer_from_data(
    wgpu_context, &state.ubo_vs, sizeof(state.ubo_vs),
    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width * 0.5f / (float)wgpu_context->height;

  glm_perspective(glm_rad(60.0f), aspect_ratio, 0.1f, 256.0f,
                  state.ubo_vs.projection);

  vec3 eye    = {0.0f, 0.0f, -2.0f};
  vec3 center = {0.0f, 0.0f, 0.0f};
  vec3 up     = {0.0f, 1.0f, 0.0f};
  glm_lookat(eye, center, up, state.ubo_vs.model_view);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       &state.ubo_vs, sizeof(state.ubo_vs));
}

/* -- Graphics pipeline layout --------------------------------------------- */

static void init_graphics_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.ubo_vs),
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [2] = (WGPUBindGroupLayoutEntry){
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };
  state.graphics.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Graphics - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.graphics.bind_group_layout != NULL);

  state.graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Graphics - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.graphics.bind_group_layout,
    });
  ASSERT(state.graphics.pipeline_layout != NULL);
}

/* -- Graphics bind groups ------------------------------------------------- */

static void init_graphics_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group_pre_compute)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group_post_compute)

  /* Pre-compute bind group (original image) */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniform_buffer,
        .offset  = 0,
        .size    = sizeof(state.ubo_vs),
      },
      [1] = (WGPUBindGroupEntry){
        .binding     = 1,
        .textureView = state.color_map.view,
      },
      [2] = (WGPUBindGroupEntry){
        .binding = 2,
        .sampler = state.color_map.sampler,
      },
    };
    state.graphics.bind_group_pre_compute = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Pre-compute - Bind group"),
                              .layout     = state.graphics.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.graphics.bind_group_pre_compute != NULL);
  }

  /* Post-compute bind group (processed image) */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniform_buffer,
        .offset  = 0,
        .size    = sizeof(state.ubo_vs),
      },
      [1] = (WGPUBindGroupEntry){
        .binding     = 1,
        .textureView = state.compute_target.view,
      },
      [2] = (WGPUBindGroupEntry){
        .binding = 2,
        .sampler = state.compute_target.sampler,
      },
    };
    state.graphics.bind_group_post_compute = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Post-compute - Bind group"),
                              .layout = state.graphics.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.graphics.bind_group_post_compute != NULL);
  }
}

/* -- Graphics render pipeline --------------------------------------------- */

static void init_graphics_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  WGPU_VERTEX_BUFFER_LAYOUT(
    texture_quad, sizeof(vertex_t),
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, offsetof(vertex_t, uv)))

  WGPUShaderModule vert_module = wgpu_create_shader_module(
    wgpu_context->device, texture_vertex_shader_wgsl);
  WGPUShaderModule frag_module = wgpu_create_shader_module(
    wgpu_context->device, texture_fragment_shader_wgsl);

  state.graphics.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Graphics - Render pipeline"),
      .layout = state.graphics.pipeline_layout,
      .vertex = (WGPUVertexState){
        .module      = vert_module,
        .entryPoint  = STRVIEW("vert_main"),
        .bufferCount = 1,
        .buffers     = &texture_quad_vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = frag_module,
        .entryPoint  = STRVIEW("frag_main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format            = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = false,
        .depthCompare      = WGPUCompareFunction_Always,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });
  ASSERT(state.graphics.pipeline != NULL);

  wgpuShaderModuleRelease(vert_module);
  wgpuShaderModuleRelease(frag_module);
}

/* -- Compute pipeline ----------------------------------------------------- */

static void init_compute_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .storageTexture = (WGPUStorageTextureBindingLayout){
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA8Unorm,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
  };

  state.compute.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Compute - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.compute.bind_group_layout != NULL);

  state.compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Compute - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.compute.bind_group_layout,
    });
  ASSERT(state.compute.pipeline_layout != NULL);
}

static void init_compute_bind_group(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute.bind_group)

  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.color_map.view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding     = 1,
      .textureView = state.compute_target.view,
    },
  };
  state.compute.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Compute - Bind group"),
                            .layout     = state.compute.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.compute.bind_group != NULL);
}

static void init_compute_pipelines(wgpu_context_t* wgpu_context)
{
  const char* compute_shaders[3] = {
    emboss_compute_shader_wgsl,
    edgedetect_compute_shader_wgsl,
    sharpen_compute_shader_wgsl,
  };

  for (uint32_t i = 0; i < 3; ++i) {
    WGPUShaderModule cs_module
      = wgpu_create_shader_module(wgpu_context->device, compute_shaders[i]);
    state.compute.pipelines[i] = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Compute pipeline"),
        .layout  = state.compute.pipeline_layout,
        .compute = (WGPUComputeState){
          .module     = cs_module,
          .entryPoint = STRVIEW("main"),
        },
      });
    ASSERT(state.compute.pipelines[i] != NULL);
    wgpuShaderModuleRelease(cs_module);
  }
}

/* -- GUI ------------------------------------------------------------------ */

static void render_gui(wgpu_context_t* wgpu_context)
{
  const uint64_t now    = stm_now();
  const float dt_sec    = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  imgui_overlay_new_frame(wgpu_context, dt_sec);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  igText("Compute Shader Filter");
  igSeparator();

  igCombo("Shader", &state.compute.pipeline_index, shader_names, 3, 3);

  igEnd();
}

/* -- Init / Frame / Shutdown ---------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
    });

    init_texture(wgpu_context);
    init_compute_target(wgpu_context, state.color_map.desc.extent.width,
                        state.color_map.desc.extent.height);
    init_buffers(wgpu_context);
    init_uniform_buffer(wgpu_context);
    update_uniform_buffers(wgpu_context);
    init_graphics_pipeline_layout(wgpu_context);
    init_graphics_bind_groups(wgpu_context);
    init_graphics_pipeline(wgpu_context);
    init_compute_pipeline_layout(wgpu_context);
    init_compute_bind_group(wgpu_context);
    init_compute_pipelines(wgpu_context);
    imgui_overlay_init(wgpu_context);

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

  sfetch_dowork();

  /* Recreate texture when pixel data loaded */
  if (state.color_map.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.color_map);
    FREE_TEXTURE_PIXELS(state.color_map);
    init_compute_target(wgpu_context, state.color_map.desc.extent.width,
                        state.color_map.desc.extent.height);
    init_graphics_bind_groups(wgpu_context);
    init_compute_bind_group(wgpu_context);
  }

  /* Update uniforms */
  update_uniform_buffers(wgpu_context);

  /* Render GUI */
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass */
  {
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(
      cpass_enc, state.compute.pipelines[state.compute.pipeline_index]);
    wgpuComputePassEncoderSetBindGroup(cpass_enc, 0, state.compute.bind_group,
                                       0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      cpass_enc, (state.compute_target.size.width + 15) / 16,
      (state.compute_target.size.height + 15) / 16, 1);
    wgpuComputePassEncoderEnd(cpass_enc);
    wgpuComputePassEncoderRelease(cpass_enc);
  }

  /* Render pass */
  {
    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);

    wgpuRenderPassEncoderSetVertexBuffer(
      rpass_enc, 0, state.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.index_buffer.buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);

    /* Left half: original image (pre-compute) */
    wgpuRenderPassEncoderSetViewport(rpass_enc, 0.0f, 0.0f,
                                     (float)wgpu_context->width * 0.5f,
                                     (float)wgpu_context->height, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(rpass_enc, 0, 0,
                                        (uint32_t)wgpu_context->width,
                                        (uint32_t)wgpu_context->height);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(
      rpass_enc, 0, state.graphics.bind_group_pre_compute, 0, NULL);
    wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.index_count, 1, 0, 0, 0);

    /* Right half: processed image (post-compute) */
    wgpuRenderPassEncoderSetViewport(rpass_enc,
                                     (float)wgpu_context->width * 0.5f, 0.0f,
                                     (float)wgpu_context->width * 0.5f,
                                     (float)wgpu_context->height, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(
      rpass_enc, 0, state.graphics.bind_group_post_compute, 0, NULL);
    wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.index_count, 1, 0, 0, 0);

    wgpuRenderPassEncoderEnd(rpass_enc);
    wgpuRenderPassEncoderRelease(rpass_enc);
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Render imgui overlay */
  imgui_overlay_render(wgpu_context);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  /* Textures */
  wgpu_destroy_texture(&state.color_map);
  WGPU_RELEASE_RESOURCE(Texture, state.compute_target.handle)
  WGPU_RELEASE_RESOURCE(TextureView, state.compute_target.view)
  WGPU_RELEASE_RESOURCE(Sampler, state.compute_target.sampler)

  /* Buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)

  /* Graphics */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group_pre_compute)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group_post_compute)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)

  /* Compute */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute.pipeline_layout)
  for (uint32_t i = 0; i < 3; ++i) {
    WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipelines[i])
  }

  imgui_overlay_shutdown();
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Compute Shader Image Load/Store",
    .init_cb        = init,
    .frame_cb       = frame,
    .input_event_cb = input_event_cb,
    .shutdown_cb    = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* texture_vertex_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4<f32>,
    modelView : mat4x4<f32>,
  }

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
  }

  @vertex
  fn vert_main(
    @location(0) pos : vec3<f32>,
    @location(1) uv : vec2<f32>
  ) -> VertexOutput {
    var output : VertexOutput;
    output.uv = uv;
    output.position = ubo.projection * ubo.modelView * vec4<f32>(pos, 1.0);
    return output;
  }
);

static const char* texture_fragment_shader_wgsl = CODE(
  @group(0) @binding(1) var texColor : texture_2d<f32>;
  @group(0) @binding(2) var texSampler : sampler;

  @fragment
  fn frag_main(
    @location(0) uv : vec2<f32>
  ) -> @location(0) vec4<f32> {
    return textureSample(texColor, texSampler, uv);
  }
);

static const char* emboss_compute_shader_wgsl = CODE(
  @group(0) @binding(0) var inputImage : texture_2d<f32>;
  @group(0) @binding(1) var resultImage : texture_storage_2d<rgba8unorm, write>;

  fn conv(kernel : array<f32, 9>, data : array<f32, 9>, denom : f32, offset : f32) -> f32 {
    var res : f32 = 0.0;
    for (var i = 0; i < 9; i++) {
      res += kernel[i] * data[i];
    }
    return clamp(res / denom + offset, 0.0, 1.0);
  }

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let coords = vec2<i32>(gid.xy);
    let dims = textureDimensions(inputImage);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
      return;
    }

    var avg : array<f32, 9>;
    var n = 0;
    for (var i = -1; i < 2; i++) {
      for (var j = -1; j < 2; j++) {
        let sc = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
        let rgb = textureLoad(inputImage, sc, 0).rgb;
        avg[n] = (rgb.r + rgb.g + rgb.b) / 3.0;
        n++;
      }
    }

    var kernel : array<f32, 9>;
    kernel[0] = -1.0; kernel[1] =  0.0; kernel[2] =  0.0;
    kernel[3] =  0.0; kernel[4] = -1.0; kernel[5] =  0.0;
    kernel[6] =  0.0; kernel[7] =  0.0; kernel[8] =  2.0;

    let value = conv(kernel, avg, 1.0, 0.5);
    textureStore(resultImage, coords, vec4<f32>(value, value, value, 1.0));
  }
);

static const char* edgedetect_compute_shader_wgsl = CODE(
  @group(0) @binding(0) var inputImage : texture_2d<f32>;
  @group(0) @binding(1) var resultImage : texture_storage_2d<rgba8unorm, write>;

  fn conv(kernel : array<f32, 9>, data : array<f32, 9>, denom : f32, offset : f32) -> f32 {
    var res : f32 = 0.0;
    for (var i = 0; i < 9; i++) {
      res += kernel[i] * data[i];
    }
    return clamp(res / denom + offset, 0.0, 1.0);
  }

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let coords = vec2<i32>(gid.xy);
    let dims = textureDimensions(inputImage);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
      return;
    }

    var avg : array<f32, 9>;
    var n = 0;
    for (var i = -1; i < 2; i++) {
      for (var j = -1; j < 2; j++) {
        let sc = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
        let rgb = textureLoad(inputImage, sc, 0).rgb;
        avg[n] = (rgb.r + rgb.g + rgb.b) / 3.0;
        n++;
      }
    }

    var kernel : array<f32, 9>;
    kernel[0] = -0.125; kernel[1] = -0.125; kernel[2] = -0.125;
    kernel[3] = -0.125; kernel[4] =  1.0;   kernel[5] = -0.125;
    kernel[6] = -0.125; kernel[7] = -0.125; kernel[8] = -0.125;

    let value = conv(kernel, avg, 0.1, 0.0);
    textureStore(resultImage, coords, vec4<f32>(value, value, value, 1.0));
  }
);

static const char* sharpen_compute_shader_wgsl = CODE(
  @group(0) @binding(0) var inputImage : texture_2d<f32>;
  @group(0) @binding(1) var resultImage : texture_storage_2d<rgba8unorm, write>;

  fn conv(kernel : array<f32, 9>, data : array<f32, 9>, denom : f32, offset : f32) -> f32 {
    var res : f32 = 0.0;
    for (var i = 0; i < 9; i++) {
      res += kernel[i] * data[i];
    }
    return clamp(res / denom + offset, 0.0, 1.0);
  }

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let coords = vec2<i32>(gid.xy);
    let dims = textureDimensions(inputImage);

    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) {
      return;
    }

    var r_data : array<f32, 9>;
    var g_data : array<f32, 9>;
    var b_data : array<f32, 9>;
    var n = 0;
    for (var i = -1; i < 2; i++) {
      for (var j = -1; j < 2; j++) {
        let sc = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
        let rgb = textureLoad(inputImage, sc, 0).rgb;
        r_data[n] = rgb.r;
        g_data[n] = rgb.g;
        b_data[n] = rgb.b;
        n++;
      }
    }

    var kernel : array<f32, 9>;
    kernel[0] = -1.0; kernel[1] = -1.0; kernel[2] = -1.0;
    kernel[3] = -1.0; kernel[4] =  9.0; kernel[5] = -1.0;
    kernel[6] = -1.0; kernel[7] = -1.0; kernel[8] = -1.0;

    let r = conv(kernel, r_data, 1.0, 0.0);
    let g = conv(kernel, g_data, 1.0, 0.0);
    let b = conv(kernel, b_data, 1.0, 0.0);
    textureStore(resultImage, coords, vec4<f32>(r, g, b, 1.0));
  }
);
// clang-format on
