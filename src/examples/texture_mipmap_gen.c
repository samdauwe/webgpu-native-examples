#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
#include <ktx.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Runtime Mip Map Generation
 *
 * This example shows how to load and sample textures (including mip maps).
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/texturemipmapgen/texturemipmapgen.cpp
 * -------------------------------------------------------------------------- */

static struct texture {
  WGPUTexture texture;
  WGPUTextureView view;
  uint32_t width, height;
  uint32_t mip_levels;
} texture;

// To demonstrate mip mapping and filtering this example uses separate samplers
static const char* sampler_names[3]
  = {"No mip maps", "Mip maps (bilinear)", "Mip maps (anisotropic)"};
static WGPUSampler samplers[3];

static struct gltf_model_t* model;
static wgpu_buffer_t uniform_buffer_vs;

static struct ubo_vs_t {
  mat4 projection;
  mat4 view;
  mat4 model;
  vec4 view_pos;
  float lod_bias;
  int32_t sampler_index;
} ubo_vs = {
  .projection    = GLM_MAT4_ZERO_INIT,
  .view          = GLM_MAT4_ZERO_INIT,
  .model         = GLM_MAT4_ZERO_INIT,
  .view_pos      = GLM_VEC4_ZERO_INIT,
  .lod_bias      = 0.0f,
  .sampler_index = 2,
};

static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass;

static WGPURenderPipeline pipeline;
static WGPUPipelineLayout pipeline_layout;
static WGPUBindGroup bind_group;
static WGPUBindGroupLayout bind_group_layout;

// Other variables
static const char* example_title = "Runtime Mip Map Generation";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_FirstPerson;
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 1024.0f);
  camera_set_rotation(context->camera, (vec3){0.0f, 90.0f, 0.0f});
  camera_set_translation(context->camera, (vec3){40.75f, 0.0f, 0.0f});
  camera_set_movement_speed(context->camera, 2.5f);
  camera_set_rotation_speed(context->camera, 0.5f);
  context->timer_speed *= 0.05f;
}

static void load_texture(wgpu_context_t* wgpu_context, const char* filename,
                         WGPUTextureFormat format)
{
  ktxResult result = KTX_NOT_FOUND;
  ktxTexture* ktx_texture;

  if (!file_exists(filename)) {
    log_fatal("Could not load texture from %s", filename);
  }
  result = ktxTexture_CreateFromNamedFile(
    filename, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktx_texture);

  ASSERT(result == KTX_SUCCESS);

  texture.width                 = ktx_texture->baseWidth;
  texture.height                = ktx_texture->baseHeight;
  ktx_uint8_t* ktx_texture_data = ktxTexture_GetData(ktx_texture);
  ktx_size_t ktx_texture_size   = ktxTexture_GetImageSize(ktx_texture, 0);

  // calculate num of mip maps
  // numLevels = 1 + floor(log2(max(w, h, d)))
  // Calculated as log2(max(width, height, depth))c + 1 (see specs)
  texture.mip_levels = floor(log2(MAX(texture.width, texture.height))) + 1;

  // Create a host-visible staging buffer that contains the raw image data
  WGPUBufferDescriptor staging_buffer_desc = {
    .label            = "Host-visible staging buffer",
    .usage            = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
    .size             = ktx_texture_size,
    .mappedAtCreation = true,
  };
  WGPUBuffer staging_buffer
    = wgpuDeviceCreateBuffer(wgpu_context->device, &staging_buffer_desc);
  ASSERT(staging_buffer);

  // Copy texture data into staging buffer
  void* mapping = wgpuBufferGetMappedRange(staging_buffer, 0, ktx_texture_size);
  ASSERT(mapping)
  memcpy(mapping, ktx_texture_data, ktx_texture_size);
  wgpuBufferUnmap(staging_buffer);

  // Create texture
  WGPUTextureDescriptor texture_desc = {
    .label         = "Mip map texture",
    .size          = (WGPUExtent3D) {
      .width               = texture.width,
      .height              = texture.height,
      .depthOrArrayLayers  = 1,
     },
    .mipLevelCount = texture.mip_levels,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = format,
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding
                     | WGPUTextureUsage_RenderAttachment,
  };
  texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(texture.texture != NULL);

  // Copy the first mip of the chain, remaining mips will be generated
  // Upload staging buffer to texture
  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  wgpuCommandEncoderCopyBufferToTexture(cmd_encoder,
    // Source
    &(WGPUImageCopyBuffer) {
      .buffer = staging_buffer,
      .layout = (WGPUTextureDataLayout) {
        .offset       = 0,
        .bytesPerRow  = texture.width * 4,
        .rowsPerImage = texture.height,
      },
    },
    // Destination
    &(WGPUImageCopyTexture){
      .texture = texture.texture,
      .mipLevel = 0,
      .origin = (WGPUOrigin3D) {
        .x = 0,
        .y = 0,
        .z = 0,
      },
      .aspect = WGPUTextureAspect_All,
    },
    // Copy size
    &(WGPUExtent3D){
      .width               = texture.width,
      .height              = texture.height,
      .depthOrArrayLayers  = 1,
    });

  WGPUCommandBuffer command_buffer
    = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  // Sumbit commmand buffer and cleanup
  ASSERT(command_buffer != NULL)

  // Submit to the queue
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  // Release command buffer
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

  // Clean up staging resources
  WGPU_RELEASE_RESOURCE(Buffer, staging_buffer)
  ktxTexture_Destroy(ktx_texture);

  // Generate the mip chain
  // ---------------------------------------------------------------------------
  wgpu_mipmap_generator_t* mipmap_generator
    = wgpu_mipmap_generator_create(wgpu_context);
  texture.texture = wgpu_mipmap_generator_generate_mipmap(
    mipmap_generator, texture.texture, &texture_desc);
  wgpu_mipmap_generator_destroy(mipmap_generator);
  // ---------------------------------------------------------------------------

  // Create samplers
  WGPUSamplerDescriptor sampler_desc = {
    .label         = "Mip map texture sampler",
    .addressModeU  = WGPUAddressMode_MirrorRepeat,
    .addressModeV  = WGPUAddressMode_MirrorRepeat,
    .addressModeW  = WGPUAddressMode_MirrorRepeat,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUMipmapFilterMode_Linear,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 0.0f,
    .maxAnisotropy = 1,
  };
  // Without mip mapping
  samplers[0] = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(samplers[0] != NULL);

  // With mip mapping
  sampler_desc.lodMaxClamp = (float)texture.mip_levels - 1;
  samplers[1] = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(samplers[1] != NULL);

  // With mip mapping and anisotropic filtering
  sampler_desc.maxAnisotropy = 16;
  samplers[2] = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(samplers[2] != NULL);

  // Create texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "Mip map texture view",
    .format          = texture_desc.format,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = texture.mip_levels,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  texture.view = wgpuTextureCreateView(texture.texture, &texture_view_dec);
  ASSERT(texture.view != NULL);
}

// Free all WebGPU resources used a texture object
static void destroy_texture(void)
{
  WGPU_RELEASE_RESOURCE(Texture, texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, texture.view)
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  model = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/tunnel_cylinder.gltf",
    .file_loading_flags = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
                          | WGPU_GLTF_FileLoadingFlags_DontLoadImages,
  });
  load_texture(wgpu_context, "textures/metalplate_nomips_rgba.ktx",
               WGPUTextureFormat_RGBA8Unorm);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[5] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0: Uniform buffer (Vertex shader) */
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(ubo_vs),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Fragment shader texture image view */
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
      /* Binding 2: Fragment shader texture image sampler 1 */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      /* Binding 3: Fragment shader texture image sampler 2 */
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [4] = (WGPUBindGroupLayoutEntry) {
      /* Binding 4: Fragment shader texture image sampler 3 */
      .binding    = 4,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this descriptor set layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  /* Bind Group */
  WGPUBindGroupEntry bg_entries[5] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0: Uniform buffer (Vertex shader) */
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1: Fragment shader texture image view */
      .binding     = 1,
      .textureView = texture.view,
    },
    [2] = (WGPUBindGroupEntry) {
      /* Binding 2: Fragment shader texture image sampler 1 */
      .binding = 2,
      .sampler = samplers[0],
    },
    [3] = (WGPUBindGroupEntry) {
      /* Binding 3: Fragment shader texture image sampler 2 */
      .binding = 3,
      .sampler = samplers[1],
    },
    [4] = (WGPUBindGroupEntry) {
     /* Binding 4: Fragment shader texture image sampler 3 */
      .binding = 4,
      .sampler = samplers[2],
    },
  };

  bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "Bind group",
                            .layout     = bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(bind_group != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
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
        .a = 0.0f,
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Construct the different states making up the pipeline */

  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  /* Vertex buffer layout */
  WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
    tunnel_cylinder,
    /* Location 0: Position */
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    /* Location 1: Texture coordinates */
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_UV),
    /* Location 2: Vertex normal */
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_Normal));

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader SPIR-V */
              .label = "Texture - Vertex shader SPIR-V",
              .file  = "shaders/texture_mipmap_gen/texture.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &tunnel_cylinder_vertex_buffer_layout,
          });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader SPIR-V */
              .label = "Texture - Fragment shader SPIR-V",
              .file  = "shaders/texture_mipmap_gen/texture.frag.spv",
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
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label     = "Texture mipmap gen - Render pipeline",
                            .layout    = pipeline_layout,
                            .primitive = primitive_state,
                            .vertex    = vertex_state,
                            .fragment  = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  glm_mat4_copy(context->camera->matrices.perspective, ubo_vs.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_vs.view);
  glm_mat4_identity(ubo_vs.model);
  glm_rotate(ubo_vs.model, glm_rad(context->timer * 360.0f),
             (vec3){1.0f, 0.0f, 0.0f});
  glm_vec4(context->camera->position, 0.0f, ubo_vs.view_pos);
  glm_vec4_mul(ubo_vs.view_pos, (vec4){-1.0f, 0.0f, 0.0f, 0.0f},
               ubo_vs.view_pos);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &ubo_vs, uniform_buffer_vs.size);
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Vertex shader uniform buffer block */
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Vertex shader - Uniform buffer block",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_vs),
    });
  ASSERT(uniform_buffer_vs.buffer != NULL);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
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
    if (imgui_overlay_slider_float(context->imgui_overlay, "LOD Bias",
                                   &ubo_vs.lod_bias, 0.0f,
                                   (float)texture.mip_levels, "%.1f")) {
      update_uniform_buffers(context);
    }
    if (imgui_overlay_combo_box(context->imgui_overlay, "Sampler Type",
                                &ubo_vs.sampler_index, sampler_names, 3)) {
      update_uniform_buffers(context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  /* Draw glTF model */
  wgpu_gltf_model_draw(model, (wgpu_gltf_model_render_options_t){0});

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

  /* Submit to queue */
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
  const int draw_result = example_draw(context);
  if (!context->paused || context->camera->updated) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  wgpu_gltf_model_destroy(model);
  destroy_texture();
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(samplers); ++i) {
    WGPU_RELEASE_RESOURCE(Sampler, samplers[i])
  }
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
}

void example_texture_mipmap_gen(int argc, char* argv[])
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
