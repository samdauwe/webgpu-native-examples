#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader Image Load/Store
 *
 * Uses a compute shader to apply different convolution kernels (and effects) on
 * an input image in realtime.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/computeshader
 * -------------------------------------------------------------------------- */

// Vertex layout used in this example
typedef struct vertex_t {
  vec3 pos;
  vec2 uv;
} vertex_t;

static struct {
  texture_t color_map;
  texture_t compute_target;
} textures = {0};

// Resources for the graphics part of the example
static struct {
  WGPUBindGroupLayout bind_group_layout; // Image display shader binding layout
  WGPUBindGroup bind_group_pre_compute;  // Image display shader bindings before
                                         // compute shader image manipulation
  WGPUBindGroup bind_group_post_compute; // Image display shader bindings after
                                         // compute shader image manipulation
  WGPURenderPipeline pipeline;           // Image display pipeline
  WGPUPipelineLayout pipeline_layout;    // Layout of the graphics pipeline
} graphics = {0};

// Resources for the compute part of the example
static struct Compute {
  WGPUBindGroupLayout bind_group_layout; // Compute shader binding layout
  WGPUBindGroup bind_group;              // Compute shader bindings
  WGPUPipelineLayout pipeline_layout;    // Layout of the compute pipeline
  WGPUComputePipeline pipelines[3];      // Compute pipelines for image filters
  int32_t pipeline_index; // Current image filtering compute pipeline index
} compute = {0};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

static wgpu_buffer_t vertex_buffer     = {0};
static wgpu_buffer_t index_buffer      = {0};
static wgpu_buffer_t uniform_buffer_vs = {0};

// Compute shaders containing convolution kernels (and effects)
static struct {
  const char* name;
  const char* location;
} compute_shaders[3] = {
  [0] = {
    .name     = "emboss",
    .location = "shaders/compute_shader/emboss.comp.spv",
  },
  [1] = {
    .name     = "edgedetect",
    .location = "shaders/compute_shader/edgedetect.comp.spv",
  },
  [2] = {
    .name     = "sharpen",
    .location = "shaders/compute_shader/sharpen.comp.spv",
  },
};
static const char* shader_names[3] = {"emboss", "edgedetect", "sharpen"};

static struct {
  mat4 projection;
  mat4 model_view;
} ubo_vs = {0};

// Other variables
static const char* example_title = "Compute Shader Image Load/Store";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  const float aspect_ratio = (float)context->wgpu_context->surface.width * 0.5f
                             / (float)context->wgpu_context->surface.height;

  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -2.0f});
  camera_set_rotation(context->camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f, aspect_ratio, 0.0f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const char* file   = "textures/Di-3d.png";
  textures.color_map = wgpu_create_texture_from_file(
    wgpu_context, file,
    &(struct wgpu_texture_load_options_t){
      .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding
               | WGPUTextureUsage_StorageBinding,
    });
}

// Prepare a texture target that is used to store compute shader calculations
static void prepare_texture_target(wgpu_context_t* wgpu_context, texture_t* tex,
                                   uint32_t width, uint32_t height,
                                   WGPUTextureFormat format)
{
  // Prepare blit target texture
  tex->size.width              = width;
  tex->size.height             = height;
  tex->size.depthOrArrayLayers = 1;
  tex->mip_level_count         = 1;
  tex->format                  = format;

  tex->texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label         = "Blit target - Texture",
      .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = (WGPUExtent3D){
        .width               = tex->size.width,
        .height              = tex->size.height,
        .depthOrArrayLayers  = tex->size.depthOrArrayLayers,
      },
      .format        = tex->format,
      .mipLevelCount = tex->mip_level_count,
      .sampleCount   = 1,
    });
  ASSERT(tex->texture != NULL)

  // Create the texture view
  tex->view = wgpuTextureCreateView(tex->texture,
                                    &(WGPUTextureViewDescriptor){
                                      .label     = "Blit target - Texture view",
                                      .format    = tex->format,
                                      .dimension = WGPUTextureViewDimension_2D,
                                      .baseMipLevel    = 0,
                                      .mipLevelCount   = tex->mip_level_count,
                                      .baseArrayLayer  = 0,
                                      .arrayLayerCount = 1,
                                    });
  ASSERT(tex->view != NULL)

  // Create sampler
  tex->sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Blit target - Texture sampler",
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
  ASSERT(tex->sampler != NULL)
}

// Setup vertices for a single uv-mapped quad
static void generate_quad(wgpu_context_t* wgpu_context)
{
  // Setup vertices for a single uv-mapped quad made from two triangles
  struct vertex_t vertices[4] = {
    // clang-format off
    {.pos = { 1.0f, -1.0f, 0.0f}, .uv = {1.0f, 1.0f}},
    {.pos = {-1.0f, -1.0f, 0.0f}, .uv = {0.0f, 1.0f}},
    {.pos = {-1.0f,  1.0f, 0.0f}, .uv = {0.0f, 0.0f}},
    {.pos = { 1.0f,  1.0f, 0.0f}, .uv = {1.0f, 0.0f}},
    // clang-format on
  };

  // Setup indices
  static uint32_t indices[6] = {
    0, 1, 2, //
    2, 3, 0, //
  };

  // Create buffers
  // Vertex buffer
  vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_t) * 4,
                    .initial.data = vertices,
                  });
  // Index buffer
  index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(uint32_t) * (uint32_t)ARRAY_SIZE(indices),
                    .count = (uint32_t)ARRAY_SIZE(indices),
                    .initial.data = indices,
                  });
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Input image (before compute post processing) */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0 : Vertex shader uniform buffer
        .binding = 0,
        .buffer  = uniform_buffer_vs.buffer,
        .offset  = 0,
        .size    = uniform_buffer_vs.size,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1 : Fragment shader texture view
        .binding     = 1,
        .textureView = textures.color_map.view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader image sampler
        .binding = 2,
        .sampler = textures.color_map.sampler,
      },
    };
    graphics.bind_group_pre_compute = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Input image - Bind group",
                              .layout     = graphics.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(graphics.bind_group_pre_compute != NULL);
  }

  /* Final image (after compute shader processing) */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0 : Vertex shader uniform buffer
        .binding = 0,
        .buffer  = uniform_buffer_vs.buffer,
        .offset  = 0,
        .size    = uniform_buffer_vs.size,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1 : Fragment shader texture view
        .binding     = 1,
        .textureView = textures.compute_target.view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader image sampler
        .binding = 2,
        .sampler = textures.compute_target.sampler,
      },
    };
    graphics.bind_group_post_compute = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Final image - Bind group",
                              .layout     = graphics.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(graphics.bind_group_post_compute != NULL);
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
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
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

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Graphics bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Vertex shader)
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(ubo_vs),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Texture view (Fragment shader)
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
        // Binding 2: Sampler (Fragment shader)
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      }
    };
    graphics.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Graphics - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(graphics.bind_group_layout != NULL);
  }

  /* Graphics pipeline layout */
  {
    graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = "Graphics - Pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &graphics.bind_group_layout,
                            });
    ASSERT(graphics.pipeline_layout != NULL);
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
      .depth_write_enabled = false,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    texture, sizeof(vertex_t),
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    // Attribute location 1: Texture coordinates
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, offsetof(vertex_t, uv)))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader SPIR-V
                      .label = "Vertex shader SPIR-V",
                      .file  = "shaders/compute_shader/texture.vert.spv",
                    },
                    .buffer_count = 1,
                    .buffers      = &texture_vertex_buffer_layout,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader SPIR-V
                      .label = "Fragment shader SPIR-V",
                      .file  = "shaders/compute_shader/texture.frag.spv",
                    },
                    .target_count = 1,
                    .targets      = &color_target_state,
                  });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  graphics.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Graphics - Render pipeline",
                            .layout       = graphics.pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(graphics.pipeline != NULL)

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void prepare_compute(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Input image (read-only)
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Output image (write)
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = (WGPUStorageTextureBindingLayout) {
          .access        = WGPUStorageTextureAccess_WriteOnly,
          .format        = WGPUTextureFormat_RGBA8Unorm,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Compute pipeline - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    compute.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(compute.bind_group_layout != NULL);
  }

  /* Compute pipeline layout */
  {
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "Compute - Pipeline layout",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &compute.bind_group_layout,
    };
    compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(compute.pipeline_layout != NULL);
  }

  /* Compute pipeline bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Input image (read-only)
        .binding     = 0,
        .textureView = textures.color_map.view,
      },
      [1] = (WGPUBindGroupEntry) {
       // Binding 1: Output image (write)
        .binding     = 1,
        .textureView = textures.compute_target.view,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Compute pipeline - Bind group",
      .layout     = compute.bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    compute.bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(compute.bind_group != NULL);
  }

  /* One pipeline for each effect */
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(compute_shaders); ++i) {
    // Compute shader
    wgpu_shader_t comp_shader
      = wgpu_shader_create(wgpu_context, &(wgpu_shader_desc_t){
                                           // Compute shader SPIR-V
                                           .file = compute_shaders[i].location,
                                         });
    // Compute pipeline
    compute.pipelines[i] = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "Compute pipeline",
        .layout  = compute.pipeline_layout,
        .compute = comp_shader.programmable_stage_descriptor,
      });
    // Clean-up
    wgpu_shader_release(&comp_shader);
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  /* Updated view matrices */
  glm_mat4_copy(context->camera->matrices.perspective, ubo_vs.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_vs.model_view);

  /* Map uniform buffer and update it */
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
      .label = "Vertex shader uniform buffer block",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_vs),
    });

  update_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    generate_quad(context->wgpu_context);
    prepare_uniform_buffers(context);
    prepare_texture_target(context->wgpu_context, &textures.compute_target,
                           textures.color_map.size.width,
                           textures.color_map.size.height,
                           WGPUTextureFormat_RGBA8Unorm);
    setup_pipeline_layout(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    prepare_compute(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_combo_box(context->imgui_overlay, "Shader",
                            &compute.pipeline_index, shader_names, 3);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(
      wgpu_context->cpass_enc, compute.pipelines[compute.pipeline_index]);
    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                       compute.bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      wgpu_context->cpass_enc, textures.compute_target.size.width / 16,
      textures.compute_target.size.height / 16, 1);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  /* Render pass */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);

    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                     (float)wgpu_context->surface.width / 2.0f,
                                     (float)wgpu_context->surface.height, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        wgpu_context->surface.width,
                                        wgpu_context->surface.height);
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 0, vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, index_buffer.buffer, WGPUIndexFormat_Uint32, 0,
      WGPU_WHOLE_SIZE);

    /* Left (pre compute) */
    {
      wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                       graphics.pipeline);
      wgpuRenderPassEncoderSetBindGroup(
        wgpu_context->rpass_enc, 0, graphics.bind_group_pre_compute, 0, NULL);

      wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                       index_buffer.count, 1, 0, 0, 0);
    }

    /* Right (post compute) */
    {
      wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                       graphics.pipeline);
      wgpuRenderPassEncoderSetBindGroup(
        wgpu_context->rpass_enc, 0, graphics.bind_group_post_compute, 0, NULL);
      wgpuRenderPassEncoderSetViewport(
        wgpu_context->rpass_enc, (float)wgpu_context->surface.width / 2, 0.0f,
        (float)wgpu_context->surface.width / 2,
        (float)wgpu_context->surface.height, 0.0f, 1.0f);

      wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                       index_buffer.count, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

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
  return example_draw(context);
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  // Update the uniform buffer when the view is changed by user input
  update_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);

  wgpu_destroy_texture(&textures.color_map);
  wgpu_destroy_texture(&textures.compute_target);

  // Graphics
  WGPU_RELEASE_RESOURCE(BindGroupLayout, graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, graphics.bind_group_pre_compute)
  WGPU_RELEASE_RESOURCE(BindGroup, graphics.bind_group_post_compute)
  WGPU_RELEASE_RESOURCE(RenderPipeline, graphics.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, graphics.pipeline_layout)

  // Compute
  WGPU_RELEASE_RESOURCE(BindGroupLayout, compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, compute.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, compute.pipeline_layout)
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(compute.pipelines); ++i) {
    WGPU_RELEASE_RESOURCE(ComputePipeline, compute.pipelines[i])
  }

  WGPU_RELEASE_RESOURCE(Buffer, vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, index_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
}

void example_compute_shader(int argc, char* argv[])
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
