#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Parallax Mapping
 *
 * Implements multiple texture mapping methods to simulate depth based on
 * texture information: Normal mapping, parallax mapping, steep parallax mapping
 * and parallax occlusion mapping (best quality, worst performance).
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/parallaxmapping/parallaxmapping.cpp
 * -------------------------------------------------------------------------- */

static struct {
  texture_t color_map;
  // Normals and height are combined into one texture (height = alpha channel)
  texture_t normal_height_map;
} textures = {0};

struct gltf_model_t* plane = NULL;

static struct {
  wgpu_buffer_t vertex_shader;
  wgpu_buffer_t fragment_shader;
} uniform_buffers = {0};

static struct {
  struct {
    mat4 projection;
    mat4 view;
    mat4 model;
    vec4 light_pos;
    vec4 camera_pos;
  } vertex_shader;
  struct {
    float height_scale;
    // Basic parallax mapping needs a bias to look any good (and is hard to
    // tweak)
    float parallax_bias;
    // Number of layers for steep parallax and parallax occlusion (more layer =
    // better result for less performance)
    float num_layers;
    // (Parallax) mapping mode to use
    int32_t mapping_mode;
  } fragment_shader;
} ubos = {
  .vertex_shader = {
    .light_pos = {0.0f, -2.0f * -1.0f, 0.0f, 1.0f},
  },
  .fragment_shader = {
    .height_scale = 0.1f,
    .parallax_bias = -0.02f,
    .num_layers = 48.0f,
    .mapping_mode = 4,
  },
};

static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

static WGPUPipelineLayout pipeline_layout    = NULL;
static WGPURenderPipeline pipeline           = NULL;
static WGPUBindGroupLayout bind_group_layout = NULL;
static WGPUBindGroup bind_group              = NULL;

static const char* mapping_modes[5] = {
  "Color only",                 //
  "Normal mapping",             //
  "Parallax mapping",           //
  "Steep parallax mapping",     //
  "Parallax occlusion mapping", //
};

// Other variables
static const char* example_title = "Parallax Mapping";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->timer_speed *= 0.5f;
  context->camera       = camera_create();
  context->camera->type = CameraType_FirstPerson;
  camera_set_position(context->camera, (vec3){0.0f, 1.25f, -1.5f});
  camera_set_rotation(context->camera, (vec3){45.0f, 0.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors
      | WGPU_GLTF_FileLoadingFlags_DontLoadImages;
  plane = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/plane.gltf",
    .file_loading_flags = gltf_loading_flags,
  });
  textures.normal_height_map = wgpu_create_texture_from_file(
    wgpu_context, "textures/rocks_normal_height_rgba.ktx", NULL);
  textures.color_map = wgpu_create_texture_from_file(
    wgpu_context, "textures/rocks_color_rgba.ktx", NULL);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[6] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (Vertex shader)
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .minBindingSize   = sizeof(ubos.vertex_shader),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Fragment shader color map image view
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
      .storageTexture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      // Binding 2: Fragment shader color map image sampler
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      // Binding 3: Fragment combined normal and heightmap view
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [4] = (WGPUBindGroupLayoutEntry) {
      // Binding 4: Fragment combined normal and heightmap sampler
      .binding    = 4,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout) {
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [5] = (WGPUBindGroupLayoutEntry) {
      // Binding 5: Fragment shader uniform buffer
      .binding    = 5,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .minBindingSize   = sizeof(ubos.fragment_shader),
      },
      .sampler = {0},
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
  // Bind Group
  WGPUBindGroupEntry bg_entries[6] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0: Uniform buffer (Vertex shader)
      .binding = 0,
      .buffer  = uniform_buffers.vertex_shader.buffer,
      .offset  = 0,
      .size    = uniform_buffers.vertex_shader.size,
    },
    [1] = (WGPUBindGroupEntry) {
      // Binding 1: Fragment shader color map image view
      .binding     = 1,
      .textureView = textures.color_map.view,
    },
    [2] = (WGPUBindGroupEntry) {
      // Binding 2: Fragment shader color map image sampler
      .binding = 2,
      .sampler = textures.color_map.sampler,
    },
    [3] = (WGPUBindGroupEntry) {
      // Binding 3: Fragment combined normal and heightmap view
      .binding     = 3,
      .textureView = textures.normal_height_map.view,
    },
    [4] = (WGPUBindGroupEntry) {
      // Binding 4: Fragment combined normal and heightmap sampler
      .binding = 4,
      .sampler = textures.normal_height_map.sampler,
    },
    [5] = (WGPUBindGroupEntry) {
      // Binding 5: Fragment shader uniform buffer
      .binding = 5,
      .buffer  = uniform_buffers.fragment_shader.buffer,
      .offset  = 0,
      .size    = uniform_buffers.fragment_shader.size,
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
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
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

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
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
    quad,
    /* Location 0: Vertex positions */
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    /* Location 1: Texture coordinates */
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_UV),
    /* Location 2: Vertex normals */
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_Normal),
    /* Location 3: Vertex tangents */
    WGPU_GLTF_VERTATTR_DESC(3, WGPU_GLTF_VertexComponent_Tangent));

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader SPIR-V */
              .label = "Parallax mapping - Vertex shader SPIR-V",
              .file  = "shaders/parallax_mapping/parallax.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &quad_vertex_buffer_layout,
          });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader SPIR-V */
              .label = "Parallax mapping - Fragment shader SPIR-V",
              .file  = "shaders/parallax_mapping/parallax.frag.spv",
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
                            .label     = "Parallax mapping - Render pipeline",
                            .layout    = pipeline_layout,
                            .primitive = primitive_state,
                            .vertex    = vertex_state,
                            .fragment  = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  /* Vertex shader */
  glm_mat4_copy(context->camera->matrices.perspective,
                ubos.vertex_shader.projection);
  glm_mat4_copy(context->camera->matrices.view, ubos.vertex_shader.view);
  glm_mat4_identity(ubos.vertex_shader.model);
  glm_scale(ubos.vertex_shader.model, (vec3){0.2f, 0.2f, 0.2f});
  glm_rotate(ubos.vertex_shader.model, glm_rad(-90), (vec3){0.0f, 1.0f, 0.0f});

  if (!context->paused) {
    ubos.vertex_shader.light_pos[0]
      = sin(glm_rad(context->timer * 360.0f)) * 1.5f;
    ubos.vertex_shader.light_pos[2]
      = cos(glm_rad(context->timer * 360.0f)) * 1.5f;
  }

  glm_vec4(context->camera->position, -1.0f, ubos.vertex_shader.camera_pos);
  glm_vec4_scale(ubos.vertex_shader.camera_pos, -1.0f,
                 ubos.vertex_shader.camera_pos);
  wgpu_queue_write_buffer(
    context->wgpu_context, uniform_buffers.vertex_shader.buffer, 0,
    &ubos.vertex_shader, uniform_buffers.vertex_shader.size);

  /* Fragment shader */
  wgpu_queue_write_buffer(
    context->wgpu_context, uniform_buffers.fragment_shader.buffer, 0,
    &ubos.fragment_shader, uniform_buffers.fragment_shader.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Vertex shader uniform buffer */
  uniform_buffers.vertex_shader = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Vertex shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubos.vertex_shader),
    });

  /* Fragment shader uniform buffer */
  uniform_buffers.fragment_shader = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Fragment shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubos.fragment_shader),
    });

  update_uniform_buffers(context);
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
    if (imgui_overlay_combo_box(context->imgui_overlay, "Mode",
                                &ubos.fragment_shader.mapping_mode,
                                mapping_modes, 5)) {
      update_uniform_buffers(context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Draw plane */
  wgpu_gltf_model_draw(plane, (wgpu_gltf_model_render_options_t){0});

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
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  wgpu_gltf_model_destroy(plane);
  wgpu_destroy_texture(&textures.color_map);
  wgpu_destroy_texture(&textures.normal_height_map);
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.vertex_shader.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.fragment_shader.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
}

void example_parallax_mapping(int argc, char* argv[])
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
