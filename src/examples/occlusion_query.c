#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Occlusion Queries
 *
 * Demonstrated how to use occlusion queries to get the number of fragment
 * samples that pass all the per-fragment tests for a set of drawing commands.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/occlusionquery/occlusionquery.cpp
 * -------------------------------------------------------------------------- */

#define MAX_DEST_BUFFERS 1

static struct {
  struct gltf_model_t* teapot;
  struct gltf_model_t* plane;
  struct gltf_model_t* sphere;
} models;

static struct {
  WGPUBuffer teapot;
  WGPUBuffer occluder;
  WGPUBuffer sphere;
} uniform_buffers;

static struct ubo_vs_t {
  mat4 projection;
  mat4 view;
  mat4 model;
  vec4 color;
  vec4 light_pos;
  float visible;
} ubo_vs = {
  .color     = GLM_VEC4_ZERO_INIT,
  .light_pos = {10.0f, -10.0f, 10.0f, 1.0f},
};

static struct {
  WGPURenderPipeline solid;
  WGPURenderPipeline occluder;
  // Pipeline with basic shaders used for occlusion pass
  WGPURenderPipeline simple;
} pipelines;

static struct {
  WGPUBindGroup teapot;
  WGPUBindGroup sphere;
} bind_groups;

static WGPURenderPassColorAttachment rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

static WGPUPipelineLayout pipeline_layout;
static WGPUBindGroup bind_group;
static WGPUBindGroupLayout bind_group_layout;

static WGPUQuerySet occlusion_query_set;
static WGPUBuffer occlusion_query_set_src_buffer;
static WGPUBuffer occlusion_query_set_dst_buffer[MAX_DEST_BUFFERS];
static bool dest_buffer_mapped[MAX_DEST_BUFFERS] = {0};

// Passed query samples
static uint64_t passed_samples[2] = {1, 1};

// Other variables
static const char* example_title = "Occlusion Queries";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -7.5f});
  camera_set_rotation(context->camera, (vec3){0.0f, -123.75f, 0.0f});
  camera_set_rotation_speed(context->camera, 0.5f);
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 1.0f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors
      | WGPU_GLTF_FileLoadingFlags_FlipY
      | WGPU_GLTF_FileLoadingFlags_DontLoadImages;
  models.plane
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/plane_z.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
  models.teapot
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/teapot.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
  models.sphere
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/sphere.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Vertex shader uniform buffer
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize = sizeof(ubo_vs),
      },
      .sampler = {0},
    },
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL)

  // Create the pipeline layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL)
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  // Occluder (plane)
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform buffer
        .binding = 0,
        .buffer = uniform_buffers.occluder,
        .offset = 0,
        .size = sizeof(ubo_vs),
      },
    };
    bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_group != NULL)
  }

  // Teapot
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform buffer
        .binding = 0,
        .buffer = uniform_buffers.teapot,
        .offset = 0,
        .size = sizeof(ubo_vs),
      },
    };
    bind_groups.teapot = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.teapot != NULL)
  }

  // Sphere
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform buffer
        .binding = 0,
        .buffer = uniform_buffers.sphere,
        .offset = 0,
        .size = sizeof(ubo_vs),
      },
    };
    bind_groups.sphere = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.sphere != NULL)
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
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 0.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Occlusion query set
  occlusion_query_set = wgpuDeviceCreateQuerySet(
    wgpu_context->device, &(WGPUQuerySetDescriptor){
                            .type  = WGPUQueryType_Occlusion,
                            .count = 2,
                          });

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
    .occlusionQuerySet      = occlusion_query_set,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  // Vertex buffer layout
  WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
    gltf_model,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal),
    // Location 3: Vertex color
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_Color));

  // Multisample state
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Render pipeline description
  WGPURenderPipelineDescriptor pipeline_desc = {
    .layout       = pipeline_layout,
    .primitive    = primitive_state_desc,
    .depthStencil = &depth_stencil_state_desc,
    .multisample  = multisample_state_desc,
  };

  // Solid rendering pipeline
  {
    // Vertex state
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .file = "shaders/occlusion_query/mesh.vert.spv",
              },
              .buffer_count = 1,
              .buffers = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .file = "shaders/occlusion_query/mesh.frag.spv",
              },
              .target_count = 1,
              .targets = &color_target_state_desc,
            });

    // Create solid pipeline
    pipeline_desc.vertex   = vertex_state_desc;
    pipeline_desc.fragment = &fragment_state_desc;
    pipelines.solid
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
    ASSERT(pipelines.solid);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }

  // Basic pipeline for coloring occluded objects
  {
    // Vertex state
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .file = "shaders/occlusion_query/simple.vert.spv",
              },
              .buffer_count = 1,
              .buffers = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .file = "shaders/occlusion_query/simple.frag.spv",
              },
              .target_count = 1,
              .targets = &color_target_state_desc,
            });

    // Create solid pipeline
    pipeline_desc.primitive.cullMode = WGPUCullMode_None;
    pipeline_desc.vertex             = vertex_state_desc;
    pipeline_desc.fragment           = &fragment_state_desc;
    pipelines.simple
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
    ASSERT(pipelines.simple);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }

  // Visual pipeline for the occluder
  {
    // Color target state
    blend_state             = wgpu_create_blend_state(true);
    color_target_state_desc = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex state
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .file = "shaders/occlusion_query/occluder.vert.spv",
              },
              .buffer_count = 1,
              .buffers = &gltf_model_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .file = "shaders/occlusion_query/occluder.frag.spv",
              },
              .target_count = 1,
              .targets = &color_target_state_desc,
            });

    // Create solid pipeline
    pipeline_desc.vertex   = vertex_state_desc;
    pipeline_desc.fragment = &fragment_state_desc;
    pipelines.occluder
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
    ASSERT(pipelines.occluder);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  glm_mat4_copy(context->camera->matrices.perspective, ubo_vs.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_vs.view);

  // Occluder
  ubo_vs.visible     = 1.0f;
  mat4 identity_mtx  = GLM_MAT4_IDENTITY_INIT;
  const float scale  = 6.0f;
  identity_mtx[0][0] = scale;
  identity_mtx[1][1] = scale;
  identity_mtx[2][2] = scale;
  glm_mat4_copy(identity_mtx, ubo_vs.model);
  glm_rotate(ubo_vs.model, glm_rad(90), (vec3){1.0f, 0.0f, 0.0f});
  glm_vec4_copy((vec4){0.0f, 0.0f, 1.0f, 0.5f}, ubo_vs.color);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.occluder, 0,
                          &ubo_vs, sizeof(ubo_vs));

  // Teapot
  // Toggle color depending on visibility
  ubo_vs.visible = (passed_samples[0] > 0) ? 1.0f : 0.0f;
  glm_mat4_identity(identity_mtx);
  glm_translate(identity_mtx, (vec3){0.0f, 0.0f, -3.0f});
  glm_mat4_copy(identity_mtx, ubo_vs.model);
  glm_vec4_copy((vec4){1.0f, 0.0f, 0.0f, 1.0f}, ubo_vs.color);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.teapot, 0,
                          &ubo_vs, sizeof(ubo_vs));

  // Sphere
  // Toggle color depending on visibility
  ubo_vs.visible = (passed_samples[1] > 0) ? 1.0f : 0.0f;
  glm_mat4_identity(identity_mtx);
  glm_translate(identity_mtx, (vec3){0.0f, 0.0f, 3.0f});
  glm_mat4_copy(identity_mtx, ubo_vs.model);
  glm_vec4_copy((vec4){0.0f, 1.0f, 0.0f, 1.0f}, ubo_vs.color);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.sphere, 0,
                          &ubo_vs, sizeof(ubo_vs));
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Vertex shader uniform buffer block
  uniform_buffers.occluder = wgpuDeviceCreateBuffer(
    context->wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = sizeof(ubo_vs),
      .mappedAtCreation = false,
    });

  // Teapot
  uniform_buffers.teapot = wgpuDeviceCreateBuffer(
    context->wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = sizeof(ubo_vs),
      .mappedAtCreation = false,
    });

  // Sphere
  uniform_buffers.sphere = wgpuDeviceCreateBuffer(
    context->wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = sizeof(ubo_vs),
      .mappedAtCreation = false,
    });

  update_uniform_buffers(context);
}

// Create a buffers for storing the occlusion query result
static void prepare_occlusion_query_set_buffers(wgpu_context_t* wgpu_context)
{
  occlusion_query_set_src_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc,
      .size  = sizeof(passed_samples),
      .mappedAtCreation = false,
    });

  for (uint32_t i = 0; i < (uint32_t)MAX_DEST_BUFFERS; ++i) {
    occlusion_query_set_dst_buffer[i] = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .usage            = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
        .size             = sizeof(passed_samples),
        .mappedAtCreation = false,
      });
  }
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_uniform_buffers(context);
    prepare_occlusion_query_set_buffers(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static int32_t get_unmapped_dest_buffer_index()
{
  int32_t index = -1;
  for (uint32_t i = 0; i < (uint32_t)MAX_DEST_BUFFERS; ++i) {
    if (!dest_buffer_mapped[i]) {
      index = i;
      break;
    }
  }
  return index;
}

static int32_t get_mapped_dest_buffer_index()
{
  int32_t index = -1;
  for (uint32_t i = 0; i < (uint32_t)MAX_DEST_BUFFERS; ++i) {
    if (dest_buffer_mapped[i]) {
      index = i;
      break;
    }
  }
  return index;
}

static void read_buffer_map_cb(WGPUBufferMapAsyncStatus status, void* user_data)
{
  UNUSED_VAR(user_data);

  if (status == WGPUBufferMapAsyncStatus_Success) {
    int32_t mapped_dest_buffer_index = get_mapped_dest_buffer_index();
    uint64_t const* mapping          = (uint64_t*)wgpuBufferGetConstMappedRange(
      occlusion_query_set_dst_buffer[mapped_dest_buffer_index], 0,
      sizeof(passed_samples));
    ASSERT(mapping)
    memcpy(passed_samples, mapping, sizeof(passed_samples));
    wgpuBufferUnmap(occlusion_query_set_dst_buffer[mapped_dest_buffer_index]);
    dest_buffer_mapped[(uint64_t)mapped_dest_buffer_index] = false;
  }
}

// Retrieves the results of the occlusion queries submitted to the command
// buffer
static void get_occlusion_query_results(void)
{
  int32_t unmapped_dest_buffer_index = get_unmapped_dest_buffer_index();
  if (unmapped_dest_buffer_index != -1) {
    dest_buffer_mapped[(uint64_t)unmapped_dest_buffer_index] = true;
    wgpuBufferMapAsync(
      occlusion_query_set_dst_buffer[(uint64_t)unmapped_dest_buffer_index],
      WGPUMapMode_Read, 0, sizeof(passed_samples), read_buffer_map_cb, NULL);
  }
}

static WGPUCommandBuffer resolve_query_set(wgpu_context_t* wgpu_context)
{
  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Resolve occlusion queries
  wgpuCommandEncoderResolveQuerySet(wgpu_context->cmd_enc, occlusion_query_set,
                                    0, (uint32_t)ARRAY_SIZE(passed_samples),
                                    occlusion_query_set_src_buffer, 0);

  int32_t unmapped_dest_buffer_index = get_unmapped_dest_buffer_index();
  if (unmapped_dest_buffer_index != -1) {
    // Copy occlusion query result to destination buffer
    wgpuCommandEncoderCopyBufferToBuffer(
      wgpu_context->cmd_enc, occlusion_query_set_src_buffer, 0,
      occlusion_query_set_dst_buffer[(uint64_t)unmapped_dest_buffer_index], 0,
      sizeof(passed_samples));
  }

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  if (imgui_overlay_header("Occlusion query results")) {
    imgui_overlay_text("Teapot: %d samples passed", passed_samples[0]);
    imgui_overlay_text("Sphere: %d samples passed", passed_samples[1]);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

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

  // Occlusion pass
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.simple);

  // Occluder first
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);
  wgpu_gltf_model_draw(models.plane, (wgpu_gltf_model_render_options_t){0});

  // Teapot
  wgpuRenderPassEncoderBeginOcclusionQuery(wgpu_context->rpass_enc, 0);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.teapot, 0, 0);
  wgpu_gltf_model_draw(models.teapot, (wgpu_gltf_model_render_options_t){0});
  wgpuRenderPassEncoderEndOcclusionQuery(wgpu_context->rpass_enc);

  // Sphere
  wgpuRenderPassEncoderBeginOcclusionQuery(wgpu_context->rpass_enc, 1);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.sphere, 0, 0);
  wgpu_gltf_model_draw(models.sphere, (wgpu_gltf_model_render_options_t){0});
  wgpuRenderPassEncoderEndOcclusionQuery(wgpu_context->rpass_enc);

  // Visible pass
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.solid);

  // Teapot
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.teapot, 0, 0);
  wgpu_gltf_model_draw(models.teapot, (wgpu_gltf_model_render_options_t){0});

  // Sphere
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.sphere, 0, 0);
  wgpu_gltf_model_draw(models.sphere, (wgpu_gltf_model_render_options_t){0});

  // Occluder
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.occluder);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);
  wgpu_gltf_model_draw(models.plane, (wgpu_gltf_model_render_options_t){0});

  // End render pass
  wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

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
  wgpu_context->submit_info.command_buffer_count = 2;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);
  wgpu_context->submit_info.command_buffers[1]
    = resolve_query_set(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Read query results for displaying in next frame
  get_occlusion_query_results();

  // Submit frame
  submit_frame(context);

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
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

  wgpu_gltf_model_destroy(models.teapot);
  wgpu_gltf_model_destroy(models.plane);
  wgpu_gltf_model_destroy(models.sphere);

  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.teapot)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.occluder)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.sphere)

  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.solid);
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.occluder)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.simple)

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.teapot)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.sphere)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)

  WGPU_RELEASE_RESOURCE(QuerySet, occlusion_query_set)
  WGPU_RELEASE_RESOURCE(Buffer, occlusion_query_set_src_buffer)
  for (uint32_t i = 0; i < (uint32_t)MAX_DEST_BUFFERS; ++i) {
    if (dest_buffer_mapped[i]) {
      wgpuBufferUnmap(occlusion_query_set_dst_buffer[i]);
      dest_buffer_mapped[i] = false;
    }
    WGPU_RELEASE_RESOURCE(Buffer, occlusion_query_set_dst_buffer[i])
  }
}

void example_occlusion_query(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
