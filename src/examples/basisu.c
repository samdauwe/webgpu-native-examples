#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Basis Universal Texture Loading
 *
 * This example shows how to how to load Basis universal supercompressed GPU
 * textures in a WebGPU application.
 *
 * Ref:
 * https://github.com/BinomialLLC/basis_universal
 * https://github.com/KhronosGroup/Vulkan-Samples/tree/master/samples/performance/texture_compression_basisu
 * https://github.com/floooh/sokol-samples/blob/master/sapp/basisu-sapp.c
 *
 * Also see:
 * https://www.khronos.org/blog/google-and-binomial-contribute-basis-universal-texture-format-to-khronos-gltf-3d-transmission-open-standard
 * https://www.khronos.org/ktx/
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* basisu_vertex_shader_wgsl;
static const char* basisu_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Basis Universal Texture Loading example
 * -------------------------------------------------------------------------- */

// Vertex layout used in this example
typedef struct vertex_t {
  vec3 pos;
  vec2 uv;
} vertex_t;

// Vertex buffer
static struct {
  WGPUBuffer buffer;
  uint32_t count;
} vertices = {0};

// Index buffer
static struct {
  WGPUBuffer buffer;
  uint32_t count;
} indices = {0};

// Uniform buffer block object
static struct {
  WGPUBuffer buffer;
  uint64_t size;
} uniform_buffer_vs = {0};

static struct {
  mat4 projection;
  mat4 model_view;
} ubo_vs = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL; // solid

// Pipeline
static WGPURenderPipeline pipeline = NULL; // solid

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Bind groups stores the resources bound to the binding points in a shader
static struct {
  WGPUBindGroup opaque;
  WGPUBindGroup alpha;
} bind_groups = {0};

static WGPUBindGroupLayout bind_group_layout = NULL;

// Basis Universal textures
static struct {
  texture_t opaque;
  texture_t alpha;
} textures = {0};

// GUI - current texture type
static int32_t current_texture_type = 0;

// Other variables
static const char* example_title = "Basis Universal Texture Loading";
static bool prepared             = false;

// Setup a default look-at camera
static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -2.0f});
  camera_set_rotation(context->camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.0f, 256.0f);
}

// Upload texture image data to the GPU
static void load_texture(wgpu_context_t* wgpu_context)
{
  textures.opaque = wgpu_create_texture_from_file(
    wgpu_context, "textures/basisu/testcard.basis", NULL);
  textures.alpha = wgpu_create_texture_from_file(
    wgpu_context, "textures/basisu/testcard_rgba.basis", NULL);
}

// Setup vertices for a single uv-mapped quad
static void generate_quad(wgpu_context_t* wgpu_context)
{
  // Setup vertices for a single uv-mapped quad made from two triangles
  struct vertex_t vertex_data[4] = {
    // clang-format off
    {.pos = { 1.0f, -1.0f, 0.0f}, .uv = {1.0f, 1.0f}}, //
    {.pos = {-1.0f, -1.0f, 0.0f}, .uv = {0.0f, 1.0f}}, //
    {.pos = {-1.0f,  1.0f, 0.0f}, .uv = {0.0f, 0.0f}}, //
    {.pos = { 1.0f,  1.0f, 0.0f}, .uv = {1.0f, 0.0f}}, //
    // clang-format on
  };
  vertices.count              = (uint32_t)ARRAY_SIZE(vertex_data);
  uint32_t vertex_buffer_size = vertices.count * sizeof(vertex_t);

  // Setup indices
  static uint32_t index_data[6] = {
    0, 1, 2, //
    2, 3, 0, //
  };
  indices.count              = (uint32_t)ARRAY_SIZE(index_data);
  uint32_t index_buffer_size = indices.count * sizeof(uint32_t);

  // Create vertex buffer
  vertices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, vertex_data, vertex_buffer_size, WGPUBufferUsage_Vertex);

  // Create index buffer
  indices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, index_data, index_buffer_size, WGPUBufferUsage_Index);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Update view matrices
  glm_mat4_copy(context->camera->matrices.perspective, ubo_vs.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_vs.model_view);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &ubo_vs, sizeof(ubo_vs));
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Vertex shader uniform buffer block
  uniform_buffer_vs.buffer = wgpu_create_buffer_from_data(
    context->wgpu_context, &ubo_vs, sizeof(ubo_vs), WGPUBufferUsage_Uniform);
  uniform_buffer_vs.size = sizeof(ubo_vs);

  update_uniform_buffers(context);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (Vertex shader)
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = uniform_buffer_vs.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Texture view (Fragment shader)
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture         = (WGPUTextureBindingLayout) {
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
                            .label                = "Render pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  texture_t* texture_array[2]       = {&textures.opaque, &textures.alpha};
  WGPUBindGroup* bindgroup_array[2] = {&bind_groups.opaque, &bind_groups.alpha};

  // Bind Group for opaque and alpha texture
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(texture_array); ++i) {
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
        .textureView = texture_array[i]->view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader image sampler
        .binding = 2,
        .sampler = texture_array[i]->sampler,
      },
    };

    (*bindgroup_array)[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "Opaque and alpha texture bind group",
                              .layout = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT((*bindgroup_array)[i] != NULL);
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

// Create the graphics pipeline
static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = false,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    quad, sizeof(vertex_t),
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    // Attribute location 1: Texture coordinates
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, offsetof(vertex_t, uv)))

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label            = "Basisu vertex shader WGSL",
                  .wgsl_code.source = basisu_vertex_shader_wgsl,
                  .entry            = "main"
                },
                .buffer_count = 1,
                .buffers      = &quad_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .label            = "Basisu fragment shader WGSL",
                  .wgsl_code.source = basisu_fragment_shader_wgsl,
                  .entry            = "main"
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
                            .label        = "Textured quad render pipeline",
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

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_texture(context->wgpu_context);
    generate_quad(context->wgpu_context);
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
    static const char* texture_types[2] = {"Opaque", "Alpha"};
    imgui_overlay_combo_box(context->imgui_overlay, "Texture Type",
                            &current_texture_type, texture_types, 2);
  }
}

/* Build separate command buffer for the framebuffer image */
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

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(
    wgpu_context->rpass_enc, 0,
    (current_texture_type == 0) ? bind_groups.opaque : bind_groups.alpha, 0, 0);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Bind triangle vertex buffer (contains position and colors) */
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);

  /* Bind triangle index buffer */
  wgpuRenderPassEncoderSetIndexBuffer(wgpu_context->rpass_enc, indices.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);

  /* Draw indexed triangle */
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, indices.count, 1, 0,
                                   0, 0);

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
  wgpu_destroy_texture(&textures.opaque);
  wgpu_destroy_texture(&textures.alpha);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.opaque)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.alpha)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_basisu(int argc, char* argv[])
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
static const char* basisu_vertex_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4<f32>,
    model : mat4x4<f32>,
  };
  @binding(0) @group(0) var<uniform> ubo : UBO;

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
  };

  @vertex
  fn main(
    @location(0) position : vec3<f32>,
    @location(1) uv : vec2<f32>
  ) -> VertexOutput {
    var output : VertexOutput;
    output.uv = uv;
    output.position = ubo.projection * ubo.model * vec4(position.xyz, 1.0);
    return output;
  }
);

// clang-format off
static const char* basisu_fragment_shader_wgsl = CODE(
  @group(0) @binding(1) var textureColor: texture_2d<f32>;
  @group(0) @binding(2) var samplerColor: sampler;

  @fragment
  fn main(
    @location(0) uv: vec2<f32>,
  ) -> @location(0) vec4<f32> {
    return textureSample(textureColor, samplerColor, uv);
  }
);
// clang-format on
