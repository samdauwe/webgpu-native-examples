#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Full Screen Radial Blur Effect
 *
 * Demonstrates the basics of fullscreen shader effects. The scene is rendered
 * into an offscreen framebuffer at lower resolution and rendered as a
 * fullscreen quad atop the scene using a radial blur fragment shader.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/radialblur
 * http://halisavakis.com/my-take-on-shaders-radial-blur/
 * -------------------------------------------------------------------------- */

// Offscreen frame buffer properties
#define FB_DIM 512u
#define FB_COLOR_FORMAT WGPUTextureFormat_RGBA8Unorm
#define FB_DEPTH_STENCIL_FORMAT WGPUTextureFormat_Depth24PlusStencil8

static bool blur            = true;
static bool display_texture = false;

static struct {
  texture_t gradient;
} textures = {0};

static struct gltf_model_t* scene;

static struct {
  wgpu_buffer_t scene;
  wgpu_buffer_t blur_params;
} ubo = {0};

static struct {
  mat4 projection;
  mat4 model_view;
  float gradient_pos;
} ubo_scene = {
  .gradient_pos = 0.0f,
};

static struct {
  float radial_blur_scale;
  float radial_blur_strength;
  vec2 radial_origin;
} ubo_blur_params = {
  .radial_blur_scale    = 0.35f,
  .radial_blur_strength = 0.75f,
  .radial_origin        = {0.5f, 0.5f},
};

static struct {
  WGPURenderPipeline radial_blur;
  WGPURenderPipeline color_pass;
  WGPURenderPipeline phong_pass;
  WGPURenderPipeline offscreen_display;
} pipelines = {0};

static struct {
  WGPUPipelineLayout radial_blur;
  WGPUPipelineLayout scene;
} pipeline_layouts = {0};

static struct {
  WGPUBindGroup scene;
  WGPUBindGroup radial_blur;
} bind_groups = {0};

static struct {
  WGPUBindGroupLayout scene;
  WGPUBindGroupLayout radial_blur;
} bind_group_layouts = {0};

static struct {
  uint32_t width, height;
  // Framebuffer for offscreen rendering
  struct {
    WGPUTexture texture;
    WGPUTextureView texture_view;
  } color, depth_stencil;
  WGPUSampler sampler;
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass;
} offscreen_pass = {0};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

static const char* example_title = "Full Screen Radial Blur Effect";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->timer_speed *= 0.5f;
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -17.5f});
  camera_set_rotation(context->camera, (vec3){-16.25f, -28.75f, 0.0f});
  camera_set_perspective(context->camera, 45.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors;
  scene = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/glowsphere.gltf",
    .file_loading_flags = gltf_loading_flags,
  });
  textures.gradient = wgpu_create_texture_from_file(
    wgpu_context, "textures/particle_gradient_rgba.ktx", NULL);
}

// Setup the offscreen framebuffer for rendering the blurred scene
// The color attachment of this framebuffer will then be used to sample frame in
// the fragment shader of the final pass
static void prepare_offscreen(wgpu_context_t* wgpu_context)
{
  offscreen_pass.width  = FB_DIM;
  offscreen_pass.height = FB_DIM;

  // Create the texture
  WGPUExtent3D texture_extent = {
    .width              = offscreen_pass.width,
    .height             = offscreen_pass.height,
    .depthOrArrayLayers = 1,
  };

  // Color attachment
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Offscreen texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = FB_COLOR_FORMAT,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    offscreen_pass.color.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(offscreen_pass.color.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_desc = {
      .label           = "Offscreen texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    offscreen_pass.color.texture_view
      = wgpuTextureCreateView(offscreen_pass.color.texture, &texture_view_desc);
    ASSERT(offscreen_pass.color.texture_view != NULL);
  }

  // Create sampler to sample from the attachment in the fragment shader
  offscreen_pass.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Offscreen texture sampler",
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

  // Depth stencil attachment
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Offscreen depth stencil texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = FB_DEPTH_STENCIL_FORMAT,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
    };
    offscreen_pass.depth_stencil.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(offscreen_pass.depth_stencil.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_desc = {
      .label           = "Offscreen depth stencil texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    offscreen_pass.depth_stencil.texture_view = wgpuTextureCreateView(
      offscreen_pass.depth_stencil.texture, &texture_view_desc);
    ASSERT(offscreen_pass.depth_stencil.texture_view != NULL);
  }

  // Create a separate render pass for the offscreen rendering as it may differ
  // from the one used for scene rendering
  // Color attachment
  offscreen_pass.render_pass.color_attachment[0]
    = (WGPURenderPassColorAttachment) {
      .view       = offscreen_pass.color.texture_view,
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

  // Depth attachment
  offscreen_pass.render_pass.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view              = offscreen_pass.depth_stencil.texture_view,
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    };

  // Render pass descriptor
  offscreen_pass.render_pass.render_pass_descriptor
    = (WGPURenderPassDescriptor){
      .label                = "Render pass descriptor",
      .colorAttachmentCount = 1,
      .colorAttachments     = offscreen_pass.render_pass.color_attachment,
      .depthStencilAttachment
      = &offscreen_pass.render_pass.depth_stencil_attachment,
    };
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout for scene rendering */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Vertex shader uniform buffer
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(ubo_scene),
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
    bind_group_layouts.scene = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Scene rendering bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.scene != NULL);

    // Create the pipeline layout
    pipeline_layouts.scene = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = "Scene rendering pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &bind_group_layouts.scene,
                            });
    ASSERT(pipeline_layouts.scene != NULL);
  }

  /* Bind group layout for fullscreen radial blur */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Fragment shader uniform buffer
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(ubo_blur_params),
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
    bind_group_layouts.radial_blur = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = "Full-screen radial blur rendering bind group layout",
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(bind_group_layouts.radial_blur != NULL);

    // Create the pipeline layout
    pipeline_layouts.radial_blur = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label = "Full-screen radial blur rendering pipeline layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &bind_group_layouts.radial_blur,
      });
    ASSERT(pipeline_layouts.radial_blur != NULL);
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Bind group for scene rendering
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform buffer
        .binding = 0,
        .buffer  = ubo.scene.buffer,
        .offset  = 0,
        .size    = ubo.scene.size,
      },
      [1] = (WGPUBindGroupEntry) {
       // Binding 1: Fragment shader image sampler
        .binding     = 1,
        .textureView = textures.gradient.view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader image sampler
        .binding = 2,
        .sampler = textures.gradient.sampler,
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

  // Bind group for fullscreen radial blur
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Fragment shader uniform buffer
        .binding = 0,
        .buffer  = ubo.blur_params.buffer,
        .offset  = 0,
        .size    = ubo.blur_params.size,
      },
      [1] = (WGPUBindGroupEntry) {
       // Binding 1: Fragment shader image sampler
        .binding     = 1,
        .textureView = offscreen_pass.color.texture_view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader image sampler
        .binding = 2,
        .sampler = offscreen_pass.sampler,
      },
    };

    bind_groups.radial_blur = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = "Fullscreen radial blur rendering bind group",
        .layout     = bind_group_layouts.radial_blur,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(bind_groups.radial_blur != NULL);
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.025f,
        .g = 0.025f,
        .b = 0.025f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
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
      .format              = FB_DEPTH_STENCIL_FORMAT,
      .depth_write_enabled = true,
    });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Radial blur pipeline
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
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state_radial_blur,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .label = "Radial blur vertex shader SPIR-V",
                .file  = "shaders/radial_blur/radialblur.vert.spv",
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
                .label = "Radial blur fragment shader SPIR-V",
                .file  = "shaders/radial_blur/radialblur.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create rendering pipeline using the specified states
    pipelines.radial_blur = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Radial blur render pipeline",
                              .layout       = pipeline_layouts.radial_blur,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.radial_blur != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  // No blending (for debug display)
  {
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
                .label = "Radial blur vertex shader SPIR-V",
                .file  = "shaders/radial_blur/radialblur.vert.spv",
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
                .label = "Radial blur fragment shader SPIR-V",
                .file  = "shaders/radial_blur/radialblur.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create rendering pipeline using the specified states
    pipelines.offscreen_display = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Debug render pipeline",
                              .layout       = pipeline_layouts.radial_blur,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.offscreen_display != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
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

  // Phong pass
  {
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
                .label = "Phongpass vertex shader SPIR-V",
                .file  = "shaders/radial_blur/phongpass.vert.spv",
              },
              .buffer_count = 1,
              .buffers      = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Phongpass fragment shader",
                .file  = "shaders/radial_blur/phongpass.frag.spv",
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
                .label = "Color pass vertex shader",
                .file  = "shaders/radial_blur/colorpass.vert.spv",
              },
              .buffer_count = 1,
              .buffers      = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Color pass fragment shader",
                .file  = "shaders/radial_blur/colorpass.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create rendering pipeline using the specified states
    pipelines.color_pass = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Color pass render pipeline",
                              .layout       = pipeline_layouts.scene,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.color_pass != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

/* Update uniform buffers for rendering the 3D scene */
static void update_uniform_buffers_scene(wgpu_example_context_t* context)
{
  camera_t* camera = context->camera;

  if (!context->paused) {
    vec3 new_camera_rotation = GLM_VEC3_ZERO_INIT;
    glm_vec3_add(camera->rotation,
                 (vec3){0.0f, context->frame_timer * 10.0f, 0.0f},
                 new_camera_rotation);
    camera_set_rotation(camera, new_camera_rotation);
  }
  glm_mat4_copy(camera->matrices.perspective, ubo_scene.projection);
  glm_mat4_copy(camera->matrices.view, ubo_scene.model_view);
  if (!context->paused) {
    ubo_scene.gradient_pos += context->frame_timer * 0.1f;
  }

  // Update GPU buffer
  wgpu_queue_write_buffer(context->wgpu_context, ubo.scene.buffer, 0,
                          &ubo_scene, ubo.scene.size);
}

/* Update radial blur uniform buffer */
void update_uniform_buffer_radial_blur(wgpu_example_context_t* context)
{
  wgpu_queue_write_buffer(context->wgpu_context, ubo.blur_params.buffer, 0,
                          &ubo_blur_params, ubo.blur_params.size);
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Phong and color pass vertex shader uniform buffer
  ubo.scene = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Phong and color pass vertex shader uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_scene),
    });

  // Fullscreen radial blur parameters
  ubo.blur_params = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_blur_params),
    });

  update_uniform_buffers_scene(context);
  update_uniform_buffer_radial_blur(context);
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
    imgui_overlay_checkBox(context->imgui_overlay, "Radial blur", &blur);
    imgui_overlay_checkBox(context->imgui_overlay, "Display render target",
                           &display_texture);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /*
   * First render pass: Offscreen rendering
   */
  {
    // Create render pass encoder for encoding drawing commands
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc,
      &offscreen_pass.render_pass.render_pass_descriptor);

    // Set viewport
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                     (float)offscreen_pass.width,
                                     (float)offscreen_pass.height, 0.0f, 1.0f);

    // Set scissor rectangle
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        offscreen_pass.width,
                                        offscreen_pass.height);

    // 3D scene
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.color_pass);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.scene, 0, 0);
    wgpu_gltf_model_draw(scene, (wgpu_gltf_model_render_options_t){0});

    // End render pass
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /*
   * Second render pass: Scene rendering with applied radial blur
   */
  {
    // Set target frame buffer
    render_pass.color_attachments[0].view
      = wgpu_context->swap_chain.frame_buffer;

    // Create render pass encoder for encoding drawing commands
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);

    // Set viewport
    wgpuRenderPassEncoderSetViewport(
      wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
      (float)wgpu_context->surface.height, 0.0f, 1.0f);

    // Set scissor rectangle
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        wgpu_context->surface.width,
                                        wgpu_context->surface.height);

    // 3D scene
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.phong_pass);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.scene, 0, 0);
    wgpu_gltf_model_draw(scene, (wgpu_gltf_model_render_options_t){0});

    // Fullscreen triangle (clipped to a quad) with radial blur
    if (blur) {
      wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                       (display_texture) ?
                                         pipelines.offscreen_display :
                                         pipelines.radial_blur);
      wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                        bind_groups.radial_blur, 0, 0);
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

  // Submit command buffers to queue
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
  wgpu_destroy_texture(&textures.gradient);
  wgpu_gltf_model_destroy(scene);

  WGPU_RELEASE_RESOURCE(Texture, offscreen_pass.color.texture)
  WGPU_RELEASE_RESOURCE(Texture, offscreen_pass.depth_stencil.texture)

  WGPU_RELEASE_RESOURCE(TextureView, offscreen_pass.color.texture_view)
  WGPU_RELEASE_RESOURCE(TextureView, offscreen_pass.depth_stencil.texture_view)

  WGPU_RELEASE_RESOURCE(Sampler, offscreen_pass.sampler)

  WGPU_RELEASE_RESOURCE(Buffer, ubo.scene.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, ubo.blur_params.buffer)

  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.radial_blur)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.color_pass)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.phong_pass)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.offscreen_display)

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.radial_blur)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.scene)

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.scene)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.radial_blur)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.scene)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.radial_blur)
}

void example_radial_blur(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
