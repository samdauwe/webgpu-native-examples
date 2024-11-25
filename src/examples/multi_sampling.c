#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Multisampling
 *
 * Implements multisample anti-aliasing (MSAA) using a renderpass with
 * multisampled attachment that get resolved into the visible frame buffer.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/multisampling/multisampling.cpp
 * -------------------------------------------------------------------------- */

static const uint32_t msaa_sample_count = 4u;
static bool use_msaa                    = false;

static struct {
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } color;
} multi_sample_target = {0};

static struct gltf_model_t* gltf_model = NULL;

static wgpu_buffer_t uniform_buffer = {0};

static struct {
  mat4 projection;
  mat4 model;
  vec4 light_pos;
} ubo_vs = {
  .light_pos = {5.0f, -5.0f, 5.0f, 1.0f},
};

static struct {
  WGPUBindGroupLayout ubo_vs;
  WGPUBindGroupLayout textures;
} bind_group_layouts = {0};

static struct {
  WGPURenderPipeline normal;
  WGPURenderPipeline msaa;
} pipelines = {0};

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

static WGPUPipelineLayout pipeline_layout    = NULL;
static WGPUBindGroup bind_group              = NULL;
static WGPUBindGroupLayout bind_group_layout = NULL;

// Other variables
static const char* example_title = "Multisampling";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
  camera_set_rotation(context->camera, (vec3){0.0f, -90.0f, 0.0f});
  camera_set_translation(context->camera, (vec3){2.5f, -1.5f, -7.5f});
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices;
  gltf_model = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/voyager.gltf",
    .file_loading_flags = gltf_loading_flags,
  });
}

// Creates a multi sample render target (texture and view) that is used to
// resolve into the visible frame buffer target in the render pass
static void setup_multisample_target(wgpu_context_t* wgpu_context)
{
  /* Color target */
  {
    /* Create the multi-sampled texture */
    WGPUTextureDescriptor multisampled_frame_desc = {
      .label         = "Multi-sampled texture",
      .size          = (WGPUExtent3D){
        .width              = wgpu_context->surface.width,
        .height             = wgpu_context->surface.height,
        .depthOrArrayLayers = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = msaa_sample_count,
      .dimension     = WGPUTextureDimension_2D,
      .format        = wgpu_context->swap_chain.format,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    multi_sample_target.color.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &multisampled_frame_desc);
    ASSERT(multi_sample_target.color.texture != NULL);

    /* Create the multi-sampled texture view */
    multi_sample_target.color.view
      = wgpuTextureCreateView(multi_sample_target.color.texture,
                              &(WGPUTextureViewDescriptor){
                                .label        = "Multi-sampled texture view",
                                .format       = wgpu_context->swap_chain.format,
                                .dimension    = WGPUTextureViewDimension_2D,
                                .baseMipLevel = 0,
                                .mipLevelCount   = 1,
                                .baseArrayLayer  = 0,
                                .arrayLayerCount = 1,
                              });
    ASSERT(multi_sample_target.color.view != NULL);
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view          = NULL, /* Assigned later */
      .resolveTarget = NULL,
      .depthSlice    = ~0,
      .loadOp        = WGPULoadOp_Clear,
      .storeOp       = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{

  /* Set 0 for passing vertex shader ubo */
  {
    WGPUBindGroupLayoutEntry bgl_entry = {
        /* Binding 0: Uniform buffer (Vertex shader) => UBOScene */
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(ubo_vs),
        },
        .sampler = {0},
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Vertex shader - ubo bind group layout",
      .entryCount = 1,
      .entries    = &bgl_entry,
    };
    bind_group_layouts.ubo_vs
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.ubo_vs != NULL);
  }

  /* Set 1 for fragment shader images (taken from glTF model) */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: texture2D (Fragment shader) => Color map */
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
        /* Binding 1: sampler (Fragment shader) => Color map */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type  = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Textures - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    bind_group_layouts.textures
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.textures != NULL);
  }

  /* Pipeline layout using the bind group layouts */
  {
    WGPUBindGroupLayout bind_group_layout_sets[2] = {
      bind_group_layouts.ubo_vs,   /* Set 0 */
      bind_group_layouts.textures, /* Set 1 */
    };
    // Pipeline layout
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "Render - Pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layout_sets),
      .bindGroupLayouts     = bind_group_layout_sets,
    };
    pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                     &pipeline_layout_desc);
    ASSERT(pipeline_layout != NULL);
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Bind group for scene matrices */
  {
    WGPUBindGroupEntry bg_entry = {
      /* Binding 0: Uniform buffer (Vertex shader) => UBOScene */
      .binding = 0,
      .buffer  = uniform_buffer.buffer,
      .offset  = 0,
      .size    = uniform_buffer.size,
    };
    bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Scene matrices - Bind group",
                              .layout     = bind_group_layouts.ubo_vs,
                              .entryCount = 1,
                              .entries    = &bg_entry,
                            });
    ASSERT(bind_group != NULL);
  }

  /* Bind group for materials */
  {
    wgpu_gltf_materials_t materials = wgpu_gltf_model_get_materials(gltf_model);
    for (uint32_t i = 0; i < materials.material_count; ++i) {
      wgpu_gltf_material_t* material = &materials.materials[i];
      if (material->base_color_texture) {
        WGPUBindGroupEntry bg_entries[2] = {
            [0] = (WGPUBindGroupEntry) {
              /* Binding 0: texture2D (Fragment shader) => Color map */
              .binding     = 0,
              .textureView = material->base_color_texture->wgpu_texture.view,
            },
            [1] = (WGPUBindGroupEntry) {
              /* Binding 1: sampler (Fragment shader) => Color map */
              .binding = 1,
              .sampler =  material->base_color_texture->wgpu_texture.sampler,
            }
          };
        material->bind_group = wgpuDeviceCreateBindGroup(
          wgpu_context->device,
          &(WGPUBindGroupDescriptor){
            .label      = "Materials - Bind group",
            .layout     = bind_group_layouts.textures,
            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
            .entries    = bg_entries,
          });
        ASSERT(material->bind_group != NULL);
      }
    }
  }
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
    multi_sampling,
    // Location 0: Vertex position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal),
    // Location 2: Texture coordinates
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_UV),
    // Location 3: Vertex color
    WGPU_GLTF_VERTATTR_DESC(3, WGPU_GLTF_VertexComponent_Color));

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader SPIR-V
                      .label = "Mesh - Vertex shader SPIR-V",
                      .file  = "shaders/multi_sampling/mesh.vert.spv",
                    },
                    .buffer_count = 1,
                    .buffers      = &multi_sampling_vertex_buffer_layout,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader SPIR-V
                      .label = "Mesh - Fragment shader SPIR-V",
                      .file  = "shaders/multi_sampling/mesh.frag.spv",
                    },
                    .target_count = 1,
                    .targets      = &color_target_state,
                  });

  /* Normal non-MSAA rendering pipeline */
  {
    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    pipelines.normal = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Normal - Render pipeline",
                              .layout       = pipeline_layout,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.normal != NULL);
  }

  /* MSAA rendering pipeline */
  {
    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = msaa_sample_count,
        });

    // Create rendering pipeline using the specified states
    pipelines.msaa = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "MSAA - Render pipeline",
                              .layout       = pipeline_layout,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.msaa != NULL);
  }

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  camera_t* camera = context->camera;
  glm_mat4_copy(camera->matrices.perspective, ubo_vs.projection);
  glm_mat4_copy(camera->matrices.view, ubo_vs.model);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer.buffer, 0,
                          &ubo_vs, uniform_buffer.size);
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Vertex shader uniform buffer block */
  uniform_buffer = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Vertex shader - Uniform buffer block",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_vs),
    });
  ASSERT(uniform_buffer.buffer != NULL);

  update_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    setup_multisample_target(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    prepare_uniform_buffers(context);
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
    imgui_overlay_checkBox(context->imgui_overlay, "MSAA", &use_msaa);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  if (use_msaa) {
    rp_color_att_descriptors[0].view = multi_sample_target.color.view;
    rp_color_att_descriptors[0].resolveTarget
      = wgpu_context->swap_chain.frame_buffer;
  }
  else {
    rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;
    rp_color_att_descriptors[0].resolveTarget = NULL;
  }

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  /* Bind the rendering pipeline */
  if (use_msaa) {
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.msaa);
  }
  else {
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.normal);
  }

  /* Bind scene matrices descriptor to set 0 */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  /* Draw model */
  static wgpu_gltf_render_flags_enum_t render_flags
    = WGPU_GLTF_RenderFlags_BindImages;
  wgpu_gltf_model_draw(gltf_model, (wgpu_gltf_model_render_options_t){
                                     .render_flags   = render_flags,
                                     .bind_image_set = 1,
                                   });

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  /* Submit command buffer to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context);
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  wgpu_gltf_model_destroy(gltf_model);
  WGPU_RELEASE_RESOURCE(Texture, multi_sample_target.color.texture)
  WGPU_RELEASE_RESOURCE(TextureView, multi_sample_target.color.view)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.ubo_vs)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.textures)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.normal)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.msaa)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
}

void example_multi_sampling(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title   = example_title,
     .overlay = true,
     .vsync   = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
