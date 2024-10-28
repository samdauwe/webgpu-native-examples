#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Offscreen Rendering
 *
 * Basic offscreen rendering in two passes. First pass renders the mirrored
 * scene to a separate framebuffer with color and depth attachments, second pass
 * samples from that color attachment for rendering a mirror surface.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/offscreen
 * -------------------------------------------------------------------------- */

// Offscreen frame buffer properties
#define FB_DIM 512u
#define FB_COLOR_FORMAT WGPUTextureFormat_RGBA8Unorm

static bool debug_display = false;

static struct {
  struct gltf_model_t* dragon;
  struct gltf_model_t* plane;
} models = {0};

static struct {
  wgpu_buffer_t shared;
  wgpu_buffer_t mirror;
  wgpu_buffer_t offScreen;
} uniform_buffers_vs = {0};

static struct ubo_vs_t {
  mat4 projection;
  mat4 view;
  mat4 model;
  vec4 light_pos;
} ubo_shared_vs = {
  .light_pos = {0.0f, 0.0f, 0.0f, 1.0f},
};

static struct model_t {
  vec3 position;
  vec3 rotation;
} model = {
  .position = {0.0f, -1.0f, 0.0f},
  .rotation = {0.0f, 0.0f, 0.0f},
};

static struct {
  WGPURenderPipeline debug;
  WGPURenderPipeline shaded;
  WGPURenderPipeline shaded_offscreen;
  WGPURenderPipeline mirror;
} pipelines = {0};

static struct {
  WGPUPipelineLayout shaded;
  WGPUPipelineLayout textured;
} pipeline_layouts = {0};

static struct {
  WGPUBindGroup offscreen;
  WGPUBindGroup mirror;
  WGPUBindGroup model;
} bind_groups = {0};

static struct {
  WGPUBindGroupLayout shaded;
  WGPUBindGroupLayout textured;
} bind_group_layouts = {0};

static struct offscreen_pass_t {
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

static WGPURenderPassColorAttachment rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

static const char* example_title = "Offscreen Rendering";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->timer_speed *= 0.25f;
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 1.0f, -6.0f});
  camera_set_rotation(context->camera, (vec3){-2.5f, 0.0f, 0.0f});
  camera_set_rotation_speed(context->camera, 0.5f);
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors;
  models.plane
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/plane.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
  models.dragon
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/chinesedragon.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
}

static void prepare_offscreen(wgpu_context_t* wgpu_context)
{
  offscreen_pass.width  = (uint32_t)FB_DIM;
  offscreen_pass.height = (uint32_t)FB_DIM;

  // Create the texture
  WGPUExtent3D texture_extent = {
    .width              = offscreen_pass.width,
    .height             = offscreen_pass.height,
    .depthOrArrayLayers = 1,
  };

  // Color attachment
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Texture",
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
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    offscreen_pass.color.texture_view
      = wgpuTextureCreateView(offscreen_pass.color.texture, &texture_view_dec);
    ASSERT(offscreen_pass.color.texture_view != NULL);
  }

  // Create sampler to sample from the attachment in the fragment shader
  offscreen_pass.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Texture sampler",
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
      .label         = "Depth stencil - Texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
    };
    offscreen_pass.depth_stencil.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(offscreen_pass.depth_stencil.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Depth stencil - Texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    offscreen_pass.depth_stencil.texture_view = wgpuTextureCreateView(
      offscreen_pass.depth_stencil.texture, &texture_view_dec);
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
      .view           = offscreen_pass.depth_stencil.texture_view,
      .depthLoadOp    = WGPULoadOp_Clear,
      .depthStoreOp   = WGPUStoreOp_Store,
      .stencilLoadOp  = WGPULoadOp_Clear,
      .stencilStoreOp = WGPUStoreOp_Store,
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
  /* Bind group layout entries */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0: Vertex shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(ubo_shared_vs),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Fragment shader image view */
      .binding = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2: Fragment shader image sampler */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout) {
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    }
  };

  /* Shaded layouts (only use first layout binding) */
  {
    bind_group_layouts.shaded = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Shaded - Bind group layout",
                              .entryCount = 1,
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.shaded != NULL);

    /* Create the pipeline layout */
    pipeline_layouts.shaded = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = "Shaded - Pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &bind_group_layouts.shaded,
                            });
    ASSERT(pipeline_layouts.shaded != NULL);
  }

  /* Textured layouts (use all layout bindings) */
  {
    bind_group_layouts.textured = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Textured - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.textured != NULL);

    /* Create the pipeline layout */
    pipeline_layouts.textured = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = "Textured - Pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &bind_group_layouts.textured,
                            });
    ASSERT(pipeline_layouts.textured != NULL);
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Bind group for Mirror */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0: Vertex shader uniform buffer */
        .binding = 0,
        .buffer  = uniform_buffers_vs.mirror.buffer,
        .offset  = 0,
        .size    =  uniform_buffers_vs.mirror.size,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: Fragment shader image sampler */
        .binding     = 1,
        .textureView = offscreen_pass.color.texture_view,
      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 2: Fragment shader image sampler */
        .binding = 2,
        .sampler = offscreen_pass.sampler,
      },
    };

    bind_groups.mirror = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Mirror - Bind group",
                              .layout     = bind_group_layouts.textured,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.mirror != NULL);
  }

  /* Bind group for Model */
  {
    WGPUBindGroupEntry bg_entry = {
      /* Binding 0 : Vertex shader uniform buffer */
      .binding = 0,
      .buffer  = uniform_buffers_vs.shared.buffer,
      .offset  = 0,
      .size    = uniform_buffers_vs.shared.size,
    };
    bind_groups.model = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Model - Bind group",
                              .layout     = bind_group_layouts.shaded,
                              .entryCount = 1,
                              .entries    = &bg_entry,
                            });
    ASSERT(bind_groups.model != NULL);
  }

  /* Bind group for Offscreen */
  {
    WGPUBindGroupEntry bg_entry = {
      /* Binding 0 : Vertex shader uniform buffer */
      .binding = 0,
      .buffer  = uniform_buffers_vs.offScreen.buffer,
      .offset  = 0,
      .size    = uniform_buffers_vs.offScreen.size,
    };
    bind_groups.offscreen = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Offscreen - Bind group",
                              .layout     = bind_group_layouts.shaded,
                              .entryCount = 1,
                              .entries    = &bg_entry,
                            });
    ASSERT(bind_groups.offscreen != NULL);
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
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

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
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

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  // Vertex buffer layout
  WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
    gltf_model,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex color
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Color),
    // Location 2: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_Normal));

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Render pipeline description
  WGPURenderPipelineDescriptor pipeline_desc = {
    .layout       = pipeline_layouts.textured,
    .primitive    = primitive_state,
    .depthStencil = &depth_stencil_state,
    .multisample  = multisample_state,
  };

  // Render-target debug display
  {
    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .label = "Quad - Vertex shader",
                .file  = "shaders/offscreen_rendering/quad.vert.spv",
              },
              .buffer_count = 0,
              .buffers      = NULL,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Quad - Fragment shader",
                .file  = "shaders/offscreen_rendering/quad.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create debug pipeline
    pipeline_desc.vertex   = vertex_state;
    pipeline_desc.fragment = &fragment_state;
    pipelines.debug
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
    ASSERT(pipelines.debug != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  // Mirror render pipeline
  {
    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .label = "Mirror - Vertex shader",
                .file  = "shaders/offscreen_rendering/mirror.vert.spv",
              },
              .buffer_count = 1,
              .buffers      = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Mirror - Fragment shader",
                .file  = "shaders/offscreen_rendering/mirror.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create mirror pipeline
    pipeline_desc.vertex   = vertex_state;
    pipeline_desc.fragment = &fragment_state;
    pipelines.mirror
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
    ASSERT(pipelines.mirror != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  // Phong shading render pipelines
  {
    primitive_state.cullMode = WGPUCullMode_Back;
    pipeline_desc.layout     = pipeline_layouts.shaded;

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .label = "Phong - Vertex shader",
                .file  = "shaders/offscreen_rendering/phong.vert.spv",
              },
              .buffer_count = 1,
              .buffers      = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Phong - Fragment shader",
                .file  = "shaders/offscreen_rendering/phong.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Scene
    pipeline_desc.vertex   = vertex_state;
    pipeline_desc.fragment = &fragment_state;
    pipelines.shaded
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
    ASSERT(pipelines.shaded != NULL);

    // Offscreen
    // Flip cull mode
    primitive_state.cullMode  = WGPUCullMode_Front;
    color_target_state.format = FB_COLOR_FORMAT;
    pipelines.shaded_offscreen
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
    ASSERT(pipelines.shaded_offscreen != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  glm_mat4_copy(context->camera->matrices.perspective,
                ubo_shared_vs.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_shared_vs.view);

  /* Model */
  glm_mat4_identity(ubo_shared_vs.model);
  glm_rotate(ubo_shared_vs.model, glm_rad(model.rotation[1]),
             (vec3){0.0f, 1.0f, 0.0f});
  glm_translate(ubo_shared_vs.model, model.position);
  wgpu_queue_write_buffer(context->wgpu_context,
                          uniform_buffers_vs.shared.buffer, 0, &ubo_shared_vs,
                          uniform_buffers_vs.shared.size);

  /* Mirror */
  glm_mat4_identity(ubo_shared_vs.model);
  wgpu_queue_write_buffer(context->wgpu_context,
                          uniform_buffers_vs.mirror.buffer, 0, &ubo_shared_vs,
                          uniform_buffers_vs.mirror.size);
}

static void update_uniform_buffer_offscreen(wgpu_example_context_t* context)
{
  glm_mat4_copy(context->camera->matrices.perspective,
                ubo_shared_vs.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_shared_vs.view);
  glm_mat4_identity(ubo_shared_vs.model);
  glm_rotate(ubo_shared_vs.model, glm_rad(model.rotation[1]),
             (vec3){0.0f, 1.0f, 0.0f});
  glm_scale(ubo_shared_vs.model, (vec3){1.0f, -1.0f, 1.0f});
  glm_translate(ubo_shared_vs.model, model.position);
  wgpu_queue_write_buffer(context->wgpu_context,
                          uniform_buffers_vs.offScreen.buffer, 0,
                          &ubo_shared_vs, uniform_buffers_vs.offScreen.size);
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Mesh vertex shader uniform buffer block */
  uniform_buffers_vs.shared = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Mesh vertex shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_shared_vs),
    });

  /* Mirror plane vertex shader uniform buffer block */
  uniform_buffers_vs.mirror = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Mirror plane vertex shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_shared_vs),
    });

  /* Offscreen vertex shader uniform buffer block */
  uniform_buffers_vs.offScreen = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Offscreen vertex shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_shared_vs),
    });

  update_uniform_buffers(context);
  update_uniform_buffer_offscreen(context);
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
    imgui_overlay_checkBox(context->imgui_overlay, "Display render target",
                           &debug_display);
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

    // Mirrored scene
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.shaded_offscreen);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.offscreen, 0, 0);
    wgpu_gltf_model_draw(models.dragon, (wgpu_gltf_model_render_options_t){0});

    // End render pass
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /*
   * Second render pass: Scene rendering with applied radial blur
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

    if (debug_display) {
      // Display the offscreen render target
      wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                       pipelines.debug);
      wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                        bind_groups.mirror, 0, 0);
      wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);
    }
    {
      // Render the scene
      // Reflection plane
      wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                       pipelines.mirror);
      wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                        bind_groups.mirror, 0, 0);
      wgpu_gltf_model_draw(models.plane, (wgpu_gltf_model_render_options_t){0});
      // Model
      wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                       pipelines.shaded);
      wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                        bind_groups.model, 0, 0);
      wgpu_gltf_model_draw(models.dragon,
                           (wgpu_gltf_model_render_options_t){0});
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
    if (!context->paused) {
      model.rotation[1] += context->frame_timer * 10.0f;
    }
    update_uniform_buffers(context);
    update_uniform_buffer_offscreen(context);
  }
  return result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);

  wgpu_gltf_model_destroy(models.dragon);
  wgpu_gltf_model_destroy(models.plane);

  WGPU_RELEASE_RESOURCE(Texture, offscreen_pass.color.texture)
  WGPU_RELEASE_RESOURCE(Texture, offscreen_pass.depth_stencil.texture)

  WGPU_RELEASE_RESOURCE(TextureView, offscreen_pass.color.texture_view)
  WGPU_RELEASE_RESOURCE(TextureView, offscreen_pass.depth_stencil.texture_view)

  WGPU_RELEASE_RESOURCE(Sampler, offscreen_pass.sampler)

  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers_vs.shared.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers_vs.mirror.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers_vs.offScreen.buffer)

  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.debug)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.shaded)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.shaded_offscreen)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.mirror)

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.shaded)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.textured)

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.offscreen)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.mirror)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.model)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.shaded)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.textured)
}

void example_offscreen_rendering(int argc, char* argv[])
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
