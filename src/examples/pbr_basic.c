#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Physical Based Shading Basics
 *
 * Demonstrates a basic specular BRDF implementation with solid materials and
 * fixed light sources on a grid of objects with varying material parameters,
 * demonstrating how metallic reflectance and surface roughness affect the
 * appearance of pbr lit objects.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/pbrbasic/pbrbasic.cpp
 * -------------------------------------------------------------------------- */

#define GRID_DIM 7
#define ALIGNMENT 256 // 256-byte alignment

static struct {
  const char* name;
  // Parameter block used as uniforms block
  struct {
    float roughness;
    float metallic;
    vec3 color;
  } params;
} materials[11] = {
  // clang-format off
  // Setup some default materials (source:
  // https://seblagarde.wordpress.com/2011/08/17/feeding-a-physical-based-lighting-mode/)
  { .name = "Gold", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 1.0f, 0.765557f, 0.336057f } } },
  { .name = "Copper", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 0.955008f, 0.637427f, 0.538163f } } },
  { .name = "Chromium", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 0.549585f, 0.556114f, 0.554256f } } },
  { .name = "Nickel", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 0.659777f, 0.608679f, 0.525649f } } },
  { .name = "Titanium", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 0.541931f, 0.496791f, 0.449419f } } },
  { .name = "Cobalt", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 0.662124f, 0.654864f, 0.633732f } } },
  { .name = "Platinum", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 0.672411f, 0.637331f, 0.585456f } } },
  // Testing materials
  { .name = "White", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 1.0f, 1.0f, 1.0f } } },
  { .name = "Red", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 1.0f, 0.0f, 0.0f } } },
  { .name = "Blue", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 0.0f, 0.0f, 1.0f } } },
  { .name = "Black", .params = { .roughness = 0.1f, .metallic = 1.0f, .color = { 0.0f, 0.0f, 0.0f } } },
  // clang-format on
};

static struct {
  const char* name;
  const char* filelocation;
  struct gltf_model_t* object;
} models[4] = {
  // clang-format off
  { .name = "Sphere", .filelocation = "models/sphere.gltf" },
  { .name = "Teapot", .filelocation = "models/teapot.gltf" },
  { .name = "Torusknot", .filelocation = "models/torusknot.gltf" },
  { .name = "Venus", .filelocation = "models/venus.gltf" },
  // clang-format on
};

// Arrays used for GUI
static const char* material_names[11] = {
  // Default materials
  "Gold", "Copper", "Chromium", "Nickel", "Titanium", "Cobalt", "Platinum", //
  // Testing materials
  "White", "Red", "Blue", "Black", //
};
static const char* object_names[4] = {"Sphere", "Teapot", "Torusknot", "Venus"};

static int32_t current_material_index = 0;
static int32_t current_object_index   = 0;

static struct {
  // Object vertex shader uniform buffer
  wgpu_buffer_t ubo_matrices;
  // Shared parameter uniform buffer
  wgpu_buffer_t ubo_params;
  // Material parameter uniform buffer
  struct {
    WGPUBuffer buffer;
    uint64_t buffer_size;
    uint64_t model_size;
  } material_params;
  // Object parameter uniform buffer
  struct {
    WGPUBuffer buffer;
    uint64_t buffer_size;
    uint64_t model_size;
  } object_params;
} uniform_buffers;

static struct {
  mat4 projection;
  mat4 model;
  mat4 view;
  vec3 cam_pos;
} ubo_matrices;

static struct {
  vec4 lights[4];
} ubo_params;

static struct matrial_params_dynamic_t {
  float roughness;
  float metallic;
  vec3 color;
  uint8_t padding[236];
} material_params_dynamic[GRID_DIM * GRID_DIM] = {0};

static struct object_params_dynamic_t {
  vec3 position;
  uint8_t padding[244];
} object_params_dynamic[GRID_DIM * GRID_DIM] = {0};

static WGPURenderPassColorAttachment rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

static WGPUPipelineLayout pipeline_layout;
static WGPURenderPipeline pipeline;
static WGPUBindGroupLayout bind_group_layout;
static WGPUBindGroup bind_group;

// Other variables
static const char* example_title = "Physical Based Shading Basics";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->timer_speed *= 0.5f;
  context->camera       = camera_create();
  context->camera->type = CameraType_FirstPerson;
  camera_set_position(context->camera, (vec3){10.0f, 13.0f, 1.8f});
  camera_set_rotation(context->camera, (vec3){62.5f, 90.0f, 0.0f});
  camera_set_movement_speed(context->camera, 4.0f);
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
  camera_set_rotation_speed(context->camera, 0.25f);
  context->paused = true;
  context->timer_speed *= 0.25f;
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_DontLoadImages;
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(models); ++i) {
    models[i].object
      = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
        .wgpu_context       = wgpu_context,
        .filename           = models[i].filelocation,
        .file_loading_flags = gltf_loading_flags,
      });
  }
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (Vertex shader & Fragment shader)
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = uniform_buffers.ubo_matrices.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Uniform buffer (Fragment shader)
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = uniform_buffers.ubo_params.size,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      // Binding 2: Dynamic uniform buffer (Fragment shader)
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = true,
        .minBindingSize   = uniform_buffers.material_params.model_size,
      },
      .sampler = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      // Binding 3: Dynamic uniform buffer (Vertex shader)
      .binding    = 3,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = true,
        .minBindingSize   = uniform_buffers.object_params.model_size,
      },
      .sampler = {0},
    },
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL);

  // Create the pipeline layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  // Bind Group
  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0: Uniform buffer (Vertex shader & Fragment shader)
      .binding = 0,
      .buffer  = uniform_buffers.ubo_matrices.buffer,
      .offset  = 0,
      .size    = uniform_buffers.ubo_matrices.size,
    },
    [1] = (WGPUBindGroupEntry) {
      // Binding 1: Uniform buffer (Fragment shader)
      .binding = 1,
      .buffer  = uniform_buffers.ubo_params.buffer,
      .offset  = 0,
      .size    = uniform_buffers.ubo_params.size,
    },
    [2] = (WGPUBindGroupEntry) {
      // Binding 2: Dynamic uniform buffer (Fragment shader)
      .binding = 2,
      .buffer  = uniform_buffers.material_params.buffer,
      .offset  = 0,
      .size    = uniform_buffers.material_params.model_size,
    },
    [3] = (WGPUBindGroupEntry) {
      // Binding 3: Dynamic uniform buffer (Vertex shader)
      .binding = 3,
      .buffer  = uniform_buffers.object_params.buffer,
      .offset  = 0,
      .size    = uniform_buffers.object_params.model_size,
    },
  };

  bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .layout     = bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(bind_group != NULL);
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
  // Construct the different states making up the pipeline

  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
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
    sphere,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal));

  // Shaders - PBR pipeline
  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader SPIR-V
              .file = "shaders/pbr_basic/pbr.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &sphere_vertex_buffer_layout,
          });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader SPIR-V
              .file = "shaders/pbr_basic/pbr.frag.spv",
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
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "pbr_render_pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // 3D object
  camera_t* camera = context->camera;
  glm_mat4_copy(camera->matrices.perspective, ubo_matrices.projection);
  glm_mat4_copy(camera->matrices.view, ubo_matrices.view);
  glm_mat4_identity(ubo_matrices.model);
  glm_rotate(ubo_matrices.model,
             glm_rad(-90.0f + (current_object_index == 1 ? 45.0f : 0.0f)),
             (vec3){0.0f, 1.0f, 0.0f});
  glm_vec3_scale(camera->position, -1.0f, ubo_matrices.cam_pos);
  wgpu_queue_write_buffer(context->wgpu_context,
                          uniform_buffers.ubo_matrices.buffer, 0, &ubo_matrices,
                          uniform_buffers.ubo_matrices.size);
}

static void update_dynamic_uniform_buffer(wgpu_context_t* wgpu_context)
{
  // Set objects positions and material properties
  uint32_t index = 0;
  for (uint32_t y = 0; y < GRID_DIM; y++) {
    for (uint32_t x = 0; x < GRID_DIM; x++) {
      // Set object position
      vec3* pos = &object_params_dynamic[index].position;
      glm_vec3_copy((vec3){(float)(x - (GRID_DIM / 2.0f)) * 2.5f, 0.0f,
                           (float)(y - (GRID_DIM / 2.0f)) * 2.5f},
                    *pos);
      // Set material metallic and roughness properties
      struct matrial_params_dynamic_t* mat_params
        = &material_params_dynamic[index];
      mat_params->metallic
        = glm_clamp((float)x / (float)(GRID_DIM - 1), 0.1f, 1.0f);
      mat_params->roughness
        = glm_clamp((float)y / (float)(GRID_DIM - 1), 0.05f, 1.0f);
      glm_vec3_copy(materials[current_material_index].params.color,
                    (*mat_params).color);
      index++;
    }
  }

  // Update buffers
  wgpu_queue_write_buffer(wgpu_context, uniform_buffers.object_params.buffer, 0,
                          &object_params_dynamic,
                          uniform_buffers.object_params.buffer_size);
  wgpu_queue_write_buffer(wgpu_context, uniform_buffers.material_params.buffer,
                          0, &material_params_dynamic,
                          uniform_buffers.material_params.buffer_size);
}

static void update_lights(wgpu_example_context_t* context)
{
  const float p = 15.0f;
  glm_vec4_copy((vec4){-p, -p * 0.5f, -p, 1.0f}, ubo_params.lights[0]);
  glm_vec4_copy((vec4){-p, -p * 0.5f, p, 1.0f}, ubo_params.lights[1]);
  glm_vec4_copy((vec4){p, -p * 0.5f, p, 1.0f}, ubo_params.lights[2]);
  glm_vec4_copy((vec4){p, -p * 0.5f, -p, 1.0f}, ubo_params.lights[3]);

  if (!context->paused) {
    float timer             = context->timer;
    ubo_params.lights[0][0] = sin(glm_rad(timer * 360.0f)) * 20.0f;
    ubo_params.lights[0][2] = cos(glm_rad(timer * 360.0f)) * 20.0f;
    ubo_params.lights[1][0] = cos(glm_rad(timer * 360.0f)) * 20.0f;
    ubo_params.lights[1][1] = sin(glm_rad(timer * 360.0f)) * 20.0f;
  }

  wgpu_queue_write_buffer(context->wgpu_context,
                          uniform_buffers.ubo_params.buffer, 0, &ubo_params,
                          uniform_buffers.ubo_params.size);
}

static uint64_t calc_constant_buffer_byte_size(uint64_t byte_size)
{
  return (byte_size + 255) & ~255;
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Object vertex shader uniform buffer
  uniform_buffers.ubo_matrices = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_matrices),
    });

  // Shared parameter uniform buffer
  uniform_buffers.ubo_params = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_params),
    });

  // Material parameter uniform buffer
  {
    uniform_buffers.material_params.model_size = sizeof(vec2) + sizeof(vec3);
    uniform_buffers.material_params.buffer_size
      = calc_constant_buffer_byte_size(sizeof(material_params_dynamic));
    WGPUBufferDescriptor ubo_desc = {
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = uniform_buffers.material_params.buffer_size,
      .mappedAtCreation = false,
    };
    uniform_buffers.material_params.buffer
      = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_desc);
    ASSERT(uniform_buffers.material_params.buffer != NULL);
  }

  // Object parameter uniform buffer
  {
    uniform_buffers.object_params.model_size = sizeof(vec3);
    uniform_buffers.object_params.buffer_size
      = calc_constant_buffer_byte_size(sizeof(object_params_dynamic));
    WGPUBufferDescriptor ubo_desc = {
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = uniform_buffers.object_params.buffer_size,
      .mappedAtCreation = false,
    };
    uniform_buffers.object_params.buffer
      = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_desc);
    ASSERT(uniform_buffers.object_params.buffer != NULL);
  }

  update_uniform_buffers(context);
  update_dynamic_uniform_buffer(context->wgpu_context);
  update_lights(context);
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
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    if (imgui_overlay_combo_box(context->imgui_overlay, "Material",
                                &current_material_index, material_names, 11)) {
      update_dynamic_uniform_buffer(context->wgpu_context);
    }
    if (imgui_overlay_combo_box(context->imgui_overlay, "Object type",
                                &current_object_index, object_names, 4)) {
      update_dynamic_uniform_buffer(context->wgpu_context);
    }
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

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  for (uint32_t i = 0; i < GRID_DIM * GRID_DIM; ++i) {
    uint32_t dynamic_offset     = i * (uint32_t)ALIGNMENT;
    uint32_t dynamic_offsets[2] = {dynamic_offset, dynamic_offset};
    // Bind the bind group for rendering a mesh using the dynamic offset
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 2,
                                      dynamic_offsets);
    // Draw object
    wgpu_gltf_model_draw(models[current_object_index].object,
                         (wgpu_gltf_model_render_options_t){0});
  }

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
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
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_lights(context);
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
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(models); ++i) {
    wgpu_gltf_model_destroy(models[i].object);
  }
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.ubo_matrices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.ubo_params.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.material_params.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.object_params.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
}

void example_pbr_basic(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
