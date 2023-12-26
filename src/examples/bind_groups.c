#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Using Bind Groups
 *
 * Bind groups are used to pass data to shader binding points. This example sets
 * up bind groups & layouts, creates a single render pipeline based on the bind
 * group layout and renders multiple objects with different bind groups.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/descriptorsets/descriptorsets.cpp
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* bind_groups_vertex_shader_wgsl;
static const char* bind_groups_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Bind Groups example
 * -------------------------------------------------------------------------- */

static bool animate = true;

struct view_matrices_t {
  mat4 projection;
  mat4 view;
  mat4 model;
} view_matrices_t = {0};

typedef struct cube_t {
  struct view_matrices_t matrices;
  WGPUBindGroup bind_group;
  texture_t texture;
  WGPUBuffer uniform_buffer;
  vec3 rotation;
} cube_t;
static cube_t cubes[2] = {0};

static struct gltf_model_t* model = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

static WGPURenderPipeline pipeline        = NULL;
static WGPUPipelineLayout pipeline_layout = NULL;

static WGPUBindGroupLayout bind_group_layout = NULL;

// Other variables
static const char* example_title = "Using Bind Groups";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 512.0f);
  camera_set_rotation(context->camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_translation(context->camera, (vec3){0.0f, 0.0f, -5.0f});
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors
      | WGPU_GLTF_FileLoadingFlags_DontLoadImages;
  model = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/cube.gltf",
    .file_loading_flags = gltf_loading_flags,
  });
  cubes[0].texture = wgpu_create_texture_from_file(
    wgpu_context, "textures/crate01_color_height_rgba.ktx", NULL);
  cubes[1].texture = wgpu_create_texture_from_file(
    wgpu_context, "textures/crate02_color_height_rgba.ktx", NULL);
}

/*
 * Set up bind groups and set layout
 */
static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /*
   * Bind group layout
   *
   * The layout describes the shader bindings and types used for a certain
   * descriptor layout and as such must match the shader bindings
   *
   * Shader bindings used in this example:
   *
   * VS:
   *    layout (set = 0, binding = 0) uniform UBOMatrices
   *
   * FS:
   *    layout (set = 0, binding = 1) uniform texture2D ...;
   *    layout (set = 0, binding = 2) uniform sampler ...;
   */
  WGPUBindGroupLayoutEntry bind_group_layout_entries[3] = {0};

  /*
   * Binding 0: Uniform buffers (used to pass matrices)
   */
  bind_group_layout_entries[0] = (WGPUBindGroupLayoutEntry) {
    // Shader binding point
    .binding = 0,
    // Accessible from the vertex shader only
    .visibility = WGPUShaderStage_Vertex,
    .buffer = (WGPUBufferBindingLayout) {
      .type             = WGPUBufferBindingType_Uniform,
      .hasDynamicOffset = false,
      .minBindingSize   = sizeof(view_matrices_t),
    },
    .sampler = {0},
  };

  /*
   * Binding 1: Image view (used to pass per object texture information)
   */
  bind_group_layout_entries[1] = (WGPUBindGroupLayoutEntry) {
    .binding = 1,
    // Accessible from the fragment shader only
    .visibility = WGPUShaderStage_Fragment,
    .texture = (WGPUTextureBindingLayout) {
      .sampleType    = WGPUTextureSampleType_Float,
      .viewDimension = WGPUTextureViewDimension_2D,
      .multisampled  = false,
    },
    .storageTexture = {0},
  };

  /*
   * Binding 2: Image sampler (used to pass per object texture information)
   */
  bind_group_layout_entries[2] = (WGPUBindGroupLayoutEntry) {
    .binding    = 2,
    .visibility = WGPUShaderStage_Fragment,
    .sampler = (WGPUSamplerBindingLayout){
      .type = WGPUSamplerBindingType_Filtering,
    },
    .texture = {0},
  };

  /* Create the bind group layout */
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = "Cube bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bind_group_layout_entries),
      .entries    = bind_group_layout_entries,
    });
  ASSERT(bind_group_layout != NULL);

  /*
   * Bind groups
   *
   * Using the shared bind group layout we will now allocate the bind groups.
   *
   * Bind groups contain the actual descriptor for the objects (buffers, images)
   * used at render time.
   */
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(cubes); ++i) {
    cube_t* cube = &cubes[i];

    WGPUBindGroupEntry bind_group_entries[3] = {0};

    /*
     * Binding 0: Object matrices Uniform buffer
     */
    bind_group_entries[0] = (WGPUBindGroupEntry){
      // Binding 0: Uniform buffer (Vertex shader)
      .binding = 0,
      .buffer  = cube->uniform_buffer,
      .offset  = 0,
      .size    = sizeof(view_matrices_t),
    };

    /*
     * Binding 1: Object texture view
     */
    bind_group_entries[1] = (WGPUBindGroupEntry){
      // Binding 1: Fragment shader color map image view
      .binding     = 1,
      .textureView = cube->texture.view,
    };

    /*
     * Binding 2: Object texture sampler
     */
    bind_group_entries[2] = (WGPUBindGroupEntry){
      // Binding 2: Fragment shader color map image sampler
      .binding = 2,
      .sampler = cube->texture.sampler,
    };

    /* Create the bind group */
    cube->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = "Cube bind group",
        .layout     = bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bind_group_entries),
        .entries    = bind_group_entries,
      });
    ASSERT(cube->bind_group != NULL);
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

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Create a pipeline layout used for our graphics pipeline
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = "Graphics pipeline layout",
      .bindGroupLayoutCount = 1,
      // The pipeline layout is based on the bind group layout we created above
      .bindGroupLayouts = &bind_group_layout,
    });
  ASSERT(pipeline_layout != NULL);

  // Construct the different states making up the pipeline

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
    cube,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal),
    // Location 2: Texture coordinates
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_UV),
    // Location 3: Vertex color
    WGPU_GLTF_VERTATTR_DESC(3, WGPU_GLTF_VertexComponent_Color));

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader WGSL
              .label            = "Cube vertex shader WGSL",
              .wgsl_code.source = bind_groups_vertex_shader_wgsl,
              .entry            = "main",
            },
            .buffer_count = 1,
            .buffers      = &cube_vertex_buffer_layout,
          });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader WGSL
              .label            = "Cube fragment shader WGSL",
              .wgsl_code.source = bind_groups_fragment_shader_wgsl,
              .entry            = "main",
            },
            .target_count = 1,
            .targets      = &color_target_state_desc,
          });

  // Multisample state
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Cube render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state_desc,
                            .vertex       = vertex_state_desc,
                            .fragment     = &fragment_state_desc,
                            .depthStencil = &depth_stencil_state_desc,
                            .multisample  = multisample_state_desc,
                          });
  ASSERT(pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // clang-format off
  static vec3 translations[2] = {
    {-2.0f, 0.0f, 0.0f}, /* Cube 1 */
    { 1.5f, 0.5f, 0.0f}, /* Cube 2 */
  };
  // clang-format on

  camera_t* camera = context->camera;
  cube_t* cube     = NULL;
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(cubes); ++i) {
    cube = &cubes[i];
    glm_mat4_identity(cube->matrices.model);
    glm_translate(cube->matrices.model, translations[i]);
    glm_mat4_copy(camera->matrices.perspective, cube->matrices.projection);
    glm_mat4_copy(camera->matrices.view, cube->matrices.view);
    glm_rotate(cube->matrices.model, glm_rad(cube->rotation[0]),
               (vec3){1.0f, 0.0f, 0.0f});
    glm_rotate(cube->matrices.model, glm_rad(cube->rotation[1]),
               (vec3){0.0f, 1.0f, 0.0f});
    glm_rotate(cube->matrices.model, glm_rad(cube->rotation[2]),
               (vec3){0.0f, 0.0f, 1.0f});
    glm_scale(cube->matrices.model, (vec3){0.25f, 0.25f, 0.25f});
    wgpu_queue_write_buffer(context->wgpu_context, cube->uniform_buffer, 0,
                            &cube->matrices, sizeof(view_matrices_t));
  }
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Vertex shader matrix uniform buffer block
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(cubes); ++i) {
    cube_t* cube = &cubes[i];

    WGPUBufferDescriptor uniform_buffer_desc = {
      .label            = "Cube uniform buffer - View matrices",
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = sizeof(view_matrices_t),
      .mappedAtCreation = false,
    };
    cube->uniform_buffer = wgpuDeviceCreateBuffer(context->wgpu_context->device,
                                                  &uniform_buffer_desc);
    ASSERT(cube->uniform_buffer != NULL);
  }

  update_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_bind_groups(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Animate", &animate);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  // Set viewport
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Render cubes with separate bind groups
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(cubes); ++i) {
    // Bind the cube's bind group. This tells the command buffer to use the
    // uniform buffer and image set for this cube
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      cubes[i].bind_group, 0, 0);
    wgpu_gltf_model_draw(model, (wgpu_gltf_model_render_options_t){0});
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

  // Submit command buffer to queue
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
  const int draw_result = example_draw(context);
  if (animate) {
    cubes[0].rotation[0] += 2.5f * context->frame_timer;
    if (cubes[0].rotation[0] > 360.0f) {
      cubes[0].rotation[0] -= 360.0f;
    }
    cubes[1].rotation[1] += 2.0f * context->frame_timer;
    if (cubes[1].rotation[1] > 360.0f) {
      cubes[1].rotation[1] -= 360.0f;
    }
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
  wgpu_gltf_model_destroy(model);
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(cubes); ++i) {
    wgpu_destroy_texture(&cubes[i].texture);
    WGPU_RELEASE_RESOURCE(Buffer, cubes[i].uniform_buffer)
    WGPU_RELEASE_RESOURCE(BindGroup, cubes[i].bind_group)
  }
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
}

void example_bind_groups(int argc, char* argv[])
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

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* bind_groups_vertex_shader_wgsl = CODE(
  struct UBOMatrices {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
    model : mat4x4<f32>,
  };

  @group(0) @binding(0) var<uniform> uboMatrices : UBOMatrices;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) normal : vec3<f32>,
    @location(1) color : vec3<f32>,
    @location(2) uv : vec2<f32>,
  };

  @vertex
  fn main(
    @location(0) inPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) inUV: vec2<f32>,
    @location(3) inColor: vec3<f32>
  ) -> Output {
    var output: Output;
    output.normal = inNormal;
    output.color = inColor;
    output.uv = inUV;
    output.position = uboMatrices.projection * uboMatrices.view * uboMatrices.model * vec4<f32>(inPos.xyz, 1.0);
    return output;
  }
);

static const char* bind_groups_fragment_shader_wgsl = CODE(
  @group(0) @binding(1) var textureColorMap: texture_2d<f32>;
  @group(0) @binding(2) var samplerColorMap: sampler;

  @fragment
  fn main(
    @location(0) inNormal : vec3<f32>,
    @location(1) inColor : vec3<f32>,
    @location(2) inUV : vec2<f32>
   ) -> @location(0) vec4<f32> {
    return textureSample(textureColorMap, samplerColorMap, inUV) * vec4<f32>(inColor, 1.0);
  }
);
// clang-format on
