#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Bloom (Offscreen Rendering)
 *
 * Advanced fullscreen effect example adding a bloom effect to a scene. Glowing
 * scene parts are rendered to a low res offscreen framebuffer that is applied
 * atop the scene using a two pass separated gaussian blur.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/bloom
 * -------------------------------------------------------------------------- */

// Offscreen frame buffer properties
#define FB_DIM 256u
#define FB_COLOR_FORMAT WGPUTextureFormat_RGBA8Unorm
#define FB_DEPTH_FORMAT WGPUTextureFormat_Depth24PlusStencil8

static bool bloom = true;

static texture_t cubemap_texture = {0};

static struct {
  struct gltf_model_t* ufo;
  struct gltf_model_t* ufo_glow;
  struct gltf_model_t* skybox;
} models = {0};

static struct {
  wgpu_buffer_t scene;
  wgpu_buffer_t skybox;
  wgpu_buffer_t blur_params;
} uniform_buffers = {0};

typedef struct {
  mat4 projection;
  mat4 view;
  mat4 model;
} ubo_scene_t;

typedef struct {
  float blur_scale;
  float blur_strength;
} ubo_blur_params_t;

static struct {
  ubo_scene_t scene, skybox;
  ubo_blur_params_t blur_params;
} ubos = {
  .blur_params = (ubo_blur_params_t){
    .blur_scale    = 1.0f,
    .blur_strength = 1.5f,
  },
};

static struct {
  WGPURenderPipeline blur_vert;
  WGPURenderPipeline blur_horz;
  WGPURenderPipeline glow_pass;
  WGPURenderPipeline phong_pass;
  WGPURenderPipeline skybox;
} pipelines = {0};

static struct {
  WGPUPipelineLayout blur;
  WGPUPipelineLayout scene;
  WGPUPipelineLayout skybox;
} pipeline_layouts = {0};

static struct {
  WGPUBindGroup blur_vert;
  WGPUBindGroup blur_horz;
  WGPUBindGroup scene;
  WGPUBindGroup skybox;
} bind_groups = {0};

static struct {
  WGPUBindGroupLayout blur;
  WGPUBindGroupLayout scene;
  WGPUBindGroupLayout skybox;
} bind_group_layouts = {0};

// Framebuffer for offscreen rendering
typedef struct {
  WGPUTexture texture;
  WGPUTextureView texture_view;
} frame_buffer_attachment_t;

typedef struct {
  frame_buffer_attachment_t color;
  frame_buffer_attachment_t depth;
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass_desc;
} frame_buffer_t;

static struct {
  uint32_t width, height;
  WGPUSampler sampler;
  frame_buffer_t frame_buffers[2];
} offscreen_pass = {0};

static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

static const char* example_title = "Bloom (Offscreen Rendering)";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->timer_speed *= 0.5f;
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, -1.25f, -10.25f});
  camera_set_rotation(context->camera, (vec3){-7.5f, -343.0f, 0.0f});
  camera_set_perspective(context->camera, 45.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors;
  models.ufo = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/retroufo.gltf",
    .file_loading_flags = gltf_loading_flags,
  });
  models.ufo_glow
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/retroufo_glow.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
  models.skybox
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/cube.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
  // Cube map
  static const char* cubemap[6] = {
    "textures/cubemaps/cubemap_space_px.png", /* Right  */
    "textures/cubemaps/cubemap_space_nx.png", /* Left   */
    "textures/cubemaps/cubemap_space_py.png", /* Top    */
    "textures/cubemaps/cubemap_space_ny.png", /* Bottom */
    "textures/cubemaps/cubemap_space_pz.png", /* Back   */
    "textures/cubemaps/cubemap_space_nz.png", /* Front  */
  };
  cubemap_texture = wgpu_create_texture_cubemap_from_files(
    wgpu_context, cubemap,
    &(struct wgpu_texture_load_options_t){
      .flip_y = true, /* Flip y to match gcanyon.ktx hdr cubemap */
    });
  ASSERT(cubemap_texture.texture)
}

// Setup the offscreen framebuffer for rendering the mirrored scene
// The color attachment of this framebuffer will then be sampled from
static void prepare_offscreen_frame_buffer(wgpu_context_t* wgpu_context,
                                           frame_buffer_t* frame_buf,
                                           WGPUTextureFormat color_format,
                                           WGPUTextureFormat depth_format)
{
  // Create the texture extent
  WGPUExtent3D texture_extent = {
    .width              = FB_DIM,
    .height             = FB_DIM,
    .depthOrArrayLayers = 1,
  };

  // Color attachment
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Offscreen frame buffer color texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = color_format,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    frame_buf->color.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(frame_buf->color.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Offscreen frame buffer color texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    frame_buf->color.texture_view
      = wgpuTextureCreateView(frame_buf->color.texture, &texture_view_dec);
    ASSERT(frame_buf->color.texture_view != NULL);
  }

  // Depth stencil attachment
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Offscreen frame buffer depth stencil texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = depth_format,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
    };
    frame_buf->depth.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(frame_buf->depth.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Offscreen frame buffer depth stencil texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    frame_buf->depth.texture_view
      = wgpuTextureCreateView(frame_buf->depth.texture, &texture_view_dec);
    ASSERT(frame_buf->depth.texture_view != NULL);
  }

  // Create a separate render pass for the offscreen rendering as it may differ
  // from the one used for scene rendering
  // Color attachment
  frame_buf->render_pass_desc.color_attachment[0]
    = (WGPURenderPassColorAttachment) {
      .view       = frame_buf->color.texture_view,
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 0.0f,
      },
  };

  // Depth stencil attachment
  frame_buf->render_pass_desc.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view              = frame_buf->depth.texture_view,
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    };

  // Render pass descriptor
  frame_buf->render_pass_desc.render_pass_descriptor
    = (WGPURenderPassDescriptor){
      .label                = "Render pass descriptor",
      .colorAttachmentCount = 1,
      .colorAttachments     = frame_buf->render_pass_desc.color_attachment,
      .depthStencilAttachment
      = &frame_buf->render_pass_desc.depth_stencil_attachment,
    };
}

// Prepare the offscreen framebuffers used for the vertical- and horizontal blur
static void prepare_offscreen(wgpu_context_t* wgpu_context)
{
  offscreen_pass.width  = (uint32_t)FB_DIM;
  offscreen_pass.height = (uint32_t)FB_DIM;

  // Create sampler to sample from the color attachments
  offscreen_pass.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Offscreen pass sampler",
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
  ASSERT(offscreen_pass.sampler != NULL);

  // Create two frame buffers
  prepare_offscreen_frame_buffer(wgpu_context, &offscreen_pass.frame_buffers[0],
                                 FB_COLOR_FORMAT, FB_DEPTH_FORMAT);
  prepare_offscreen_frame_buffer(wgpu_context, &offscreen_pass.frame_buffers[1],
                                 FB_COLOR_FORMAT, FB_DEPTH_FORMAT);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout for fullscreen blur
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Fragment shader uniform buffer
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(ubo_blur_params_t),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Fragment shader image view
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Fragment shader image sampler
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      }
    };

    // Create the bind group layout
    bind_group_layouts.blur = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Blur bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.blur != NULL);

    // Create the pipeline layout
    pipeline_layouts.blur = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label                = "Blur pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &bind_group_layouts.blur,
                            });
    ASSERT(pipeline_layouts.blur != NULL);
  }

  // Bind group layout for scene rendering
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Vertex shader uniform buffer
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(ubo_scene_t),
        },
        .sampler = {0},
      },
    };

    // Create the bind group layout
    bind_group_layouts.scene = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Scene bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.scene != NULL);

    // Create the pipeline layout
    pipeline_layouts.scene = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label                = "Scene pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &bind_group_layouts.scene,
                            });
    ASSERT(pipeline_layouts.scene != NULL);
  }

  // Bind group layout for skybox rendering
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Vertex shader uniform buffer
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(ubo_scene_t),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Fragment shader image view
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Fragment shader image sampler
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      }
    };

    // Create the bind group layout
    bind_group_layouts.skybox = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Skybox bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.skybox != NULL);

    // Create the pipeline layout
    pipeline_layouts.skybox = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label                = "Skybox pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &bind_group_layouts.skybox,
                            });
    ASSERT(pipeline_layouts.skybox != NULL);
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Bind group for full screen blur
  /* Vertical blur */
  {
    WGPUBindGroupEntry bg_entries[3] = {
        [0] = (WGPUBindGroupEntry) {
          // Binding 0: Fragment shader uniform buffer
          .binding = 0,
          .buffer  = uniform_buffers.blur_params.buffer,
          .offset  = 0,
          .size    = uniform_buffers.blur_params.size,
        },
        [1] = (WGPUBindGroupEntry) {
         // Binding 1: Fragment shader image sampler
          .binding     = 1,
          .textureView = offscreen_pass.frame_buffers[0].color.texture_view,
        },
        [2] = (WGPUBindGroupEntry) {
          // Binding 2: Fragment shader image sampler
          .binding = 2,
          .sampler = offscreen_pass.sampler,
        },
      };

    bind_groups.blur_vert = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "Full screen vertical blur bind group",
                              .layout = bind_group_layouts.blur,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.blur_vert != NULL);
  }

  /* Horizontal blur */
  {
    WGPUBindGroupEntry bg_entries[3] = {
        [0] = (WGPUBindGroupEntry) {
          // Binding 0: Fragment shader uniform buffer
          .binding = 0,
          .buffer  = uniform_buffers.blur_params.buffer,
          .offset  = 0,
          .size    = uniform_buffers.blur_params.size,
        },
        [1] = (WGPUBindGroupEntry) {
         // Binding 1: Fragment shader image sampler
          .binding     = 1,
          .textureView = offscreen_pass.frame_buffers[1].color.texture_view,
        },
        [2] = (WGPUBindGroupEntry) {
          // Binding 2: Fragment shader image sampler
          .binding = 2,
          .sampler = offscreen_pass.sampler,
        },
      };

    bind_groups.blur_horz = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label = "Full screen horizontal blur bind group",
                              .layout     = bind_group_layouts.blur,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.blur_horz != NULL);
  }

  /* Bind group for scene rendering */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform buffer
        .binding = 0,
        .buffer  = uniform_buffers.scene.buffer,
        .offset  = 0,
        .size    = uniform_buffers.scene.size,
      },
    };

    bind_groups.scene = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Scene rendering bind group",
                              .layout     = bind_group_layouts.scene,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.scene != NULL);
  }

  /* Bind group for skybox rendering */
  {
    WGPUBindGroupEntry bg_entries[3] = {
        [0] = (WGPUBindGroupEntry) {
          // Binding 0: Vertex shader uniform buffer
          .binding = 0,
          .buffer  = uniform_buffers.skybox.buffer,
          .offset  = 0,
          .size    = uniform_buffers.skybox.size,
        },
        [1] = (WGPUBindGroupEntry) {
         // Binding 1: Fragment shader image sampler
          .binding     = 1,
          .textureView = cubemap_texture.view,
        },
        [2] = (WGPUBindGroupEntry) {
          // Binding 2: Fragment shader image sampler
          .binding = 2,
          .sampler = cubemap_texture.sampler,
        },
      };

    bind_groups.skybox = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Skybox rendering bind group",
                              .layout     = bind_group_layouts.skybox,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.skybox != NULL);
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.025f,
        .g = 0.025f,
        .b = 0.025f,
        .a = 1.000f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = FB_DEPTH_FORMAT,
      .depth_write_enabled = true,
    });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Blur pipelines
  {
    // Additive blending
    WGPUBlendState blend_state_radial_blur = {
      .color.operation = WGPUBlendOperation_Add,
      .color.srcFactor = WGPUBlendFactor_One,
      .color.dstFactor = WGPUBlendFactor_One,
      .alpha.operation = WGPUBlendOperation_Add,
      .alpha.srcFactor = WGPUBlendFactor_SrcAlpha,
      .alpha.dstFactor = WGPUBlendFactor_DstAlpha,
    };
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = FB_COLOR_FORMAT,
      .blend     = &blend_state_radial_blur,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertical blur pipeline
    {
      // Vertex state
      WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader SPIR-V
                  .label = "Vertical gaussian blur vertex shader SPIR-V",
                  .file  = "shaders/bloom/gaussblur.vert.spv",
                },
                // Empty vertex input state
                .buffer_count = 0,
                .buffers      = NULL,
              });

      // Fragment state
      WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader SPIR-V
                  .label = "Vertical gaussian blur fragment shader SPIR-V",
                  .file  = "shaders/bloom/gaussblur_vert.frag.spv",
                },
                .target_count = 1,
                .targets      = &color_target_state,
              });

      // Create rendering pipeline using the specified states
      pipelines.blur_vert = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                .label        = "Vertical blur render pipeline",
                                .layout       = pipeline_layouts.blur,
                                .primitive    = primitive_state,
                                .vertex       = vertex_state,
                                .fragment     = &fragment_state,
                                .depthStencil = &depth_stencil_state,
                                .multisample  = multisample_state,
                              });
      ASSERT(pipelines.blur_vert);

      // Partial cleanup
      WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
      WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
    }

    // Horizontal blur pipeline
    {
      // Using swapchain format in this pipeline
      color_target_state.format = wgpu_context->swap_chain.format;

      // Vertex state
      WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader SPIR-V
                  .label = "Horizontal gaussian blur vertex shader SPIR-V",
                  .file  = "shaders/bloom/gaussblur.vert.spv",
                },
                // Empty vertex input state
                .buffer_count = 0,
                .buffers      = NULL,
              });

      // Fragment state
      WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader SPIR-V
                  .label = "Horizontal gaussian blur fragment shader SPIR-V",
                  .file  = "shaders/bloom/gaussblur_horz.frag.spv",
                },
                .target_count = 1,
                .targets      = &color_target_state,
              });

      // Create rendering pipeline using the specified states
      pipelines.blur_horz = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                .label     = "Horizontal blur render pipeline",
                                .layout    = pipeline_layouts.blur,
                                .primitive = primitive_state,
                                .vertex    = vertex_state,
                                .fragment  = &fragment_state,
                                .depthStencil = &depth_stencil_state,
                                .multisample  = multisample_state,
                              });
      ASSERT(pipelines.blur_horz != NULL);

      // Partial cleanup
      WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
      WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
    }
  }

  // Vertex buffer layout
  WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
    gltf_model,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex uv
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_UV),
    // Location 2: Vertex color
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_Color),
    // Location 3: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(3, WGPU_GLTF_VertexComponent_Normal));

  // Phong pass (3D model)
  {
    // Change back face culling mode
    primitive_state.cullMode = WGPUCullMode_Back;

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
                .label = "Phong pass vertex shader SPIR-V",
                .file  = "shaders/bloom/phongpass.vert.spv",
              },
              .buffer_count = 1,
              .buffers      = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Phong pass fragment shader SPIR-V",
                .file  = "shaders/bloom/phongpass.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create rendering pipeline using the specified states
    pipelines.phong_pass = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Phong render pipeline",
                              .layout       = pipeline_layouts.scene,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.phong_pass != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  // Color only pass (offscreen blur base)
  {
    // Color target state
    WGPUBlendState blend_state              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = FB_COLOR_FORMAT,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .label = "Color pass vertex shader SPIR-V",
                .file  = "shaders/bloom/colorpass.vert.spv",
              },
              .buffer_count = 1,
              .buffers      = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Color pass fragment shader SPIR-V",
                .file  = "shaders/bloom/colorpass.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create rendering pipeline using the specified states
    pipelines.glow_pass = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Color pass render pipeline",
                              .layout       = pipeline_layouts.scene,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.glow_pass != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  // Skybox (cubemap)
  {
    primitive_state.cullMode              = WGPUCullMode_Front;
    depth_stencil_state.depthWriteEnabled = false;

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
                .label = "Skybox vertex shader SPIR-V",
                .file  = "shaders/bloom/skybox.vert.spv",
              },
              .buffer_count = 1,
              .buffers      = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Skybox fragment shader SPIR-V",
                .file  = "shaders/bloom/skybox.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create rendering pipeline using the specified states
    pipelines.skybox = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Skybox render pipeline",
                              .layout       = pipeline_layouts.skybox,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.skybox != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

// Update uniform buffers for rendering the 3D scene
static void update_uniform_buffers_scene(wgpu_example_context_t* context)
{
  camera_t* camera = context->camera;
  float timer      = context->timer;

  // UBO
  glm_mat4_copy(camera->matrices.perspective, ubos.scene.projection);
  glm_mat4_copy(camera->matrices.view, ubos.scene.view);

  glm_mat4_identity(ubos.scene.model);
  glm_translate(ubos.scene.model,
                (vec3){sin(glm_rad(timer * 360.0f)) * 0.25f, -1.0f,
                       cos(glm_rad(timer * 360.0f)) * 0.25f});
  glm_rotate(ubos.scene.model, -sinf(glm_rad(timer * 360.0f)) * 0.15f,
             (vec3){1.0f, 0.0f, 0.0f});
  glm_rotate(ubos.scene.model, glm_rad(timer * 360.0f),
             (vec3){0.0f, 1.0f, 0.0f});

  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.scene.buffer,
                          0, &ubos.scene, uniform_buffers.scene.size);

  // Skybox
  glm_perspective(glm_rad(45.0f), context->window_size.aspect_ratio, 0.1f,
                  256.0f, ubos.skybox.projection);
  mat3 mat3_tmp = GLM_MAT3_ZERO_INIT;
  glm_mat4_pick3(camera->matrices.view, mat3_tmp);
  glm_mat4_ins3(mat3_tmp, ubos.skybox.view);
  ubos.skybox.view[3][3] = 1.0f;
  glm_mat4_identity(ubos.skybox.model);

  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.skybox.buffer,
                          0, &ubos.skybox, uniform_buffers.skybox.size);
}

// Update blur pass parameter uniform buffer
void update_uniform_buffers_blur(wgpu_example_context_t* context)
{
  wgpu_queue_write_buffer(context->wgpu_context,
                          uniform_buffers.blur_params.buffer, 0,
                          &ubos.blur_params, uniform_buffers.blur_params.size);
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Phong and color pass vertex shader uniform buffer
  uniform_buffers.scene = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Phong and color pass vertex shader uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_scene_t),
    });

  // Blur parameters uniform buffer
  uniform_buffers.blur_params = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Blur parameters uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_blur_params_t),
    });

  // Skybox
  uniform_buffers.skybox = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Skybox parameters uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_scene_t),
    });

  // Initialize uniform buffers
  update_uniform_buffers_scene(context);
  update_uniform_buffers_blur(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_offscreen(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    imgui_overlay_checkBox(context->imgui_overlay, "Bloom", &bloom);
    if (imgui_overlay_input_float(context->imgui_overlay, "Scale",
                                  &ubos.blur_params.blur_scale, 0.1f, "%.1f")) {
      update_uniform_buffers_blur(context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /*
   * The blur method used in this example is multi pass and renders the vertical
   * blur first and then the horizontal one.
   * While it's possible to blur in one pass, this method is widely used as it
   * requires far less samples to generate the blur.
   */

  if (bloom) {
    /*
     * First render pass: Render glow parts of the model (separate mesh) to an
     * offscreen frame buffer.
     */

    // Create render pass encoder for encoding drawing commands
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc,
      &offscreen_pass.frame_buffers[0].render_pass_desc.render_pass_descriptor);

    // Set viewport
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                     (float)offscreen_pass.width,
                                     (float)offscreen_pass.height, 0.0f, 1.0f);

    // Set scissor rectangle
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        offscreen_pass.width,
                                        offscreen_pass.height);

    // 3D Scene
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.glow_pass);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.scene, 0, 0);
    wgpu_gltf_model_draw(models.ufo_glow,
                         (wgpu_gltf_model_render_options_t){0});

    // End render pass
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

    /*
     * Second render pass: Vertical blur.
     *
     * Render contents of the first pass into a second framebuffer and apply a
     * vertical blur.
     * This is the first blur pass, the horizontal blur is applied
     * when rendering on top of the scene.
     */

    // Create render pass encoder for encoding drawing commands
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc,
      &offscreen_pass.frame_buffers[1].render_pass_desc.render_pass_descriptor);

    // Set viewport
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                     (float)offscreen_pass.width,
                                     (float)offscreen_pass.height, 0.0f, 1.0f);

    // Set scissor rectangle
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        offscreen_pass.width,
                                        offscreen_pass.height);

    // Render
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.blur_vert);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.blur_vert, 0, 0);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);

    // End render pass
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /*
   * Third render pass: Scene rendering with applied vertical blur
   *
   * Renders the scene and the (vertically blurred) contents of the second
   * framebuffer and apply a horizontal blur.
   */
  {
    // Set target frame buffer
    rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

    // Create render pass encoder for encoding drawing commands
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);

    // Set viewport
    wgpuRenderPassEncoderSetViewport(
      wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
      (float)wgpu_context->surface.height, 0.0f, 1.0f);

    // Set scissor rectangle
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        wgpu_context->surface.width,
                                        wgpu_context->surface.height);

    // Skybox
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.skybox);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.skybox, 0, 0);
    wgpu_gltf_model_draw(models.skybox, (wgpu_gltf_model_render_options_t){0});

    // 3D scene
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.phong_pass);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.scene, 0, 0);
    wgpu_gltf_model_draw(models.ufo, (wgpu_gltf_model_render_options_t){0});

    // Fullscreen triangle (clipped to a quad) with horizontal blur
    if (bloom) {
      wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                       pipelines.blur_horz);
      wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                        bind_groups.blur_horz, 0, 0);
      wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);
    }

    // End render pass
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
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

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  int result = example_draw(context);
  if (!context->paused || context->camera->updated) {
    update_uniform_buffers_scene(context);
  }
  return result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);

  wgpu_destroy_texture(&cubemap_texture);

  wgpu_gltf_model_destroy(models.ufo);
  wgpu_gltf_model_destroy(models.ufo_glow);
  wgpu_gltf_model_destroy(models.skybox);

  for (uint32_t i = 0; i < 2; ++i) {
    WGPU_RELEASE_RESOURCE(Texture,
                          offscreen_pass.frame_buffers[i].color.texture)
    WGPU_RELEASE_RESOURCE(Texture,
                          offscreen_pass.frame_buffers[i].depth.texture)

    WGPU_RELEASE_RESOURCE(TextureView,
                          offscreen_pass.frame_buffers[i].color.texture_view)
    WGPU_RELEASE_RESOURCE(TextureView,
                          offscreen_pass.frame_buffers[i].depth.texture_view)
  }

  WGPU_RELEASE_RESOURCE(Sampler, offscreen_pass.sampler)

  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.scene.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.skybox.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.blur_params.buffer)

  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.blur_vert)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.blur_horz)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.glow_pass)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.phong_pass)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.skybox)

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.blur)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.scene)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.skybox)

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.blur_vert)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.blur_horz)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.scene)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.skybox)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.blur)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.scene)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.skybox)
}

void example_bloom(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
