#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - High Dynamic Range Rendering
 *
 * Implements a high dynamic range rendering pipeline using 16/32 bit floating
 * point precision for all internal formats, textures and calculations,
 * including a bloom pass, manual exposure and tone mapping.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/hdr
 * -------------------------------------------------------------------------- */

#define NUMBER_OF_CONSTANTS 3
#define ALIGNMENT 256 // 256-byte alignment

static bool display_skybox = true;

static struct {
  texture_t envmap;
} textures;

static struct {
  struct gltf_model_t* skybox;
} models;

static struct {
  WGPUBuffer matrices;
  WGPUBuffer params;
  struct {
    WGPUBuffer buffer;
    uint64_t buffer_size;
    uint64_t model_size;
  } dynamic;
} uniform_buffers;

static struct {
  mat4 projection;
  mat4 model_view;
  mat4 inverse_modelview;
} ubo_matrices;

static struct {
  float exposure;
} ubo_params = {
  .exposure = 1.0f,
};

static struct {
  int value;
  uint8_t padding[252];
} ubo_constants[NUMBER_OF_CONSTANTS] = {0};

static struct {
  WGPURenderPipeline skybox;
} pipelines;

static struct {
  WGPUPipelineLayout models;
} pipeline_layouts;

static struct {
  WGPUBindGroup skybox;
} bind_groups;

static struct {
  WGPUBindGroupLayout models;
} bind_group_layouts;

static WGPURenderPassColorAttachment rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

static const char* example_title = "High Dynamic Range Rendering";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -6.0f});
  camera_set_rotation(context->camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  // Load glTF models
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices;
  models.skybox
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/cube.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
  // Load cube map
  static const char* cubemap[6] = {
    "textures/cubemaps/uffizi_cube_px.jpg", // Right
    "textures/cubemaps/uffizi_cube_nx.jpg", // Left
    "textures/cubemaps/uffizi_cube_py.jpg", // Top
    "textures/cubemaps/uffizi_cube_ny.jpg", // Bottom
    "textures/cubemaps/uffizi_cube_pz.jpg", // Back
    "textures/cubemaps/uffizi_cube_nz.jpg", // Front
  };
  textures.envmap = wgpu_create_texture_cubemap_from_files(
    wgpu_context, cubemap,
    &(struct wgpu_texture_load_options_t){
      .flip_y = false,
    });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout for models
  {
    WGPUBindGroupLayoutEntry bgl_entries[5] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Vertex / fragment shader uniform buffer
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = sizeof(ubo_matrices),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Fragment shader image view
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Fragment shader image sampler
        .binding = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        // Binding 3: Fragment shader uniform buffer
        .binding = 3,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = sizeof(ubo_params),
        },
        .sampler = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        // Binding 4:  Vertex / fragment shader dynamic uniform buffer
        .binding = 4,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize = sizeof(ubo_constants[0].value),
        },
        .sampler = {0},
      },
    };

    // Create the bind group layout
    bind_group_layouts.models = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.models != NULL)

    // Create the pipeline layout
    pipeline_layouts.models = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &bind_group_layouts.models,
                            });
    ASSERT(pipeline_layouts.models != NULL)
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Model bind groups
  {
    WGPUBindGroupEntry bg_entries[5] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex / fragment shader uniform buffer
        .binding = 0,
        .buffer = uniform_buffers.matrices,
        .offset = 0,
        .size = sizeof(ubo_matrices),
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment shader image sampler
        .binding = 1,
        .textureView = textures.envmap.view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader image sampler
        .binding = 2,
        .sampler = textures.envmap.sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        // Binding 3: Fragment shader uniform buffer
        .binding = 3,
        .buffer = uniform_buffers.params,
        .offset = 0,
        .size = sizeof(ubo_params),
      },
      [4] = (WGPUBindGroupEntry) {
        // Binding 4: Vertex / fragment shader dynamic uniform buffer
        .binding = 4,
        .buffer = uniform_buffers.dynamic.buffer,
        .offset = 0,
        .size = sizeof(ubo_constants[0].value),
      },
    };

    // Sky box bind group
    {
      WGPUBindGroupDescriptor bg_desc = {
        .layout     = bind_group_layouts.models,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      };
      bind_groups.skybox
        = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
      ASSERT(bind_groups.skybox != NULL)
    }
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.025f,
        .g = 0.025f,
        .b = 0.025f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  // Multisample state
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Vertex buffer layout
  WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
    gltf_model,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal));

  // Skybox pipeline (background cube)
  {
    primitive_state_desc.cullMode              = WGPUCullMode_Front;
    depth_stencil_state_desc.depthWriteEnabled = false;

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex state
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .file = "shaders/hdr/gbuffer.vert.spv",
              },
              .buffer_count = 1,
              .buffers = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .file = "shaders/hdr/gbuffer.frag.spv",
              },
              .target_count = 1,
              .targets = &color_target_state_desc,
            });

    // Create rendering pipeline using the specified states
    pipelines.skybox = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "skybox_render_pipeline",
                              .layout       = pipeline_layouts.models,
                              .primitive    = primitive_state_desc,
                              .vertex       = vertex_state_desc,
                              .fragment     = &fragment_state_desc,
                              .depthStencil = &depth_stencil_state_desc,
                              .multisample  = multisample_state_desc,
                            });
    ASSERT(pipelines.skybox);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  camera_t* camera = context->camera;

  glm_mat4_copy(camera->matrices.perspective, ubo_matrices.projection);
  glm_mat4_copy(camera->matrices.view, ubo_matrices.model_view);
  glm_mat4_inv(camera->matrices.view, ubo_matrices.inverse_modelview);

  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.matrices, 0,
                          &ubo_matrices, sizeof(ubo_matrices));
}

static void update_params(wgpu_example_context_t* context)
{
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.params, 0,
                          &ubo_params, sizeof(ubo_params));
}

static void update_dynamic_uniform_buffers(wgpu_example_context_t* context)
{
  // Set constant values
  for (uint32_t i = 0; i < (uint32_t)NUMBER_OF_CONSTANTS; ++i) {
    ubo_constants[i].value = (int32_t)i;
  }

  // Update buffer
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.dynamic.buffer,
                          0, &ubo_constants,
                          uniform_buffers.dynamic.buffer_size);
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Matrices vertex shader uniform buffer
  uniform_buffers.matrices = wgpuDeviceCreateBuffer(
    context->wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = sizeof(ubo_matrices),
      .mappedAtCreation = false,
    });

  // Params
  uniform_buffers.params = wgpuDeviceCreateBuffer(
    context->wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = sizeof(ubo_params),
      .mappedAtCreation = false,
    });

  // Uniform buffer object with constants
  uniform_buffers.dynamic.model_size  = sizeof(int);
  uniform_buffers.dynamic.buffer_size = sizeof(ubo_constants);
  uniform_buffers.dynamic.buffer      = wgpuDeviceCreateBuffer(
    context->wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = uniform_buffers.dynamic.buffer_size,
      .mappedAtCreation = false,
    });

  // Initialize uniform buffers
  update_uniform_buffers(context);
  update_params(context);
  update_dynamic_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Skybox", &display_skybox);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

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
    uint32_t dynamic_offset = 0 * (uint32_t)ALIGNMENT;
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.skybox, 1, &dynamic_offset);
    wgpu_gltf_model_draw(models.skybox, (wgpu_gltf_model_render_options_t){0});

    // End render pass
    wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
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

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  int result = example_draw(context);
  if (context->camera->updated) {
    update_uniform_buffers(context);
  }
  return result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);

  wgpu_destroy_texture(&textures.envmap);

  wgpu_gltf_model_destroy(models.skybox);

  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.matrices)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.params)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.dynamic.buffer)

  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.skybox)

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.models)

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.skybox)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.models)
}

void example_hdr(int argc, char* argv[])
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
  });
  // clang-format on
}
