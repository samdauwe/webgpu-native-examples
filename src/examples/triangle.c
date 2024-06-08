#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Basic Indexed Triangle
 *
 * This is a "pedal to the metal" example to show off how to get WebGPU up an
 * displaying something.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/blob/main/src/pages/samples/helloTriangle.ts
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/triangle/triangle.cpp
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* triangle_vertex_shader_wgsl;
static const char* triangle_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Basic Indexed Triangle example
 * -------------------------------------------------------------------------- */

// Vertex layout used in this example
typedef struct {
  vec3 position;
  vec3 color;
} vertex_t;

// Vertex buffer and attributes
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
  uint32_t count;
} uniform_buffer_vs = {0};

// Uniform block vertex shader
static struct {
  mat4 projection_matrix;
  mat4 model_matrix;
  mat4 view_matrix;
} ubo_vs = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// The bind group layout describes the shader binding layout (without actually
// referencing descriptor)
// Like the pipeline layout it's pretty much a blueprint and can be used with
// different descriptor sets as long as their layout matches
static WGPUBindGroupLayout bind_group_layout = NULL;

// The bind group stores the resources bound to the binding points in a shader
// It connects the binding points of the different shaders with the buffers and
// images used for those bindings
static WGPUBindGroup bind_group = NULL;

// Other variables
static const char* example_title = "Basic Indexed Triangle";
static bool prepared             = false;

// Setup a default look-at camera
static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -2.5f});
  camera_set_rotation(context->camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.0f, 256.0f);
}

// Prepare vertex and index buffers for an indexed triangle
static void prepare_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  // Setup vertices (x, y, z, r, g, b)
  static const vertex_t vertex_buffer[3] = {
    {
      .position = {1.0f, -1.0f, 0.0f},
      .color    = {1.0f, 0.0f, 0.0f},
    },
    {
      .position = {-1.0f, -1.0f, 0.0f},
      .color    = {0.0f, 1.0f, 0.0f},
    },
    {
      .position = {0.0f, 1.0f, 0.0f},
      .color    = {0.0f, 0.0f, 1.0f},
    },
  };
  vertices.count              = (uint32_t)ARRAY_SIZE(vertex_buffer);
  uint32_t vertex_buffer_size = vertices.count * sizeof(vertex_t);

  // Setup indices
  static const uint16_t index_buffer[4] = {
    0, 1, 2,
    0 // padding
  };
  indices.count              = (uint32_t)ARRAY_SIZE(index_buffer);
  uint32_t index_buffer_size = indices.count * sizeof(uint32_t);

  // Create vertex buffer
  vertices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, vertex_buffer, vertex_buffer_size, WGPUBufferUsage_Vertex);

  // Create index buffer
  indices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, index_buffer, index_buffer_size, WGPUBufferUsage_Index);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Setup layout of descriptors used in this example
  // Basically connects the different shader stages to descriptors for binding
  // uniform buffers, image samplers, etc. So every shader binding should map to
  // one descriptor set layout binding

  // Bind group layout
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor) {
      .label      = "Bind group layout",
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Vertex shader)
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = sizeof(ubo_vs),
        },
        .sampler = {0},
      }
    }
  );
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

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Bind Group
  bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor) {
     .label      = "Bind group",
     .layout     = bind_group_layout,
     .entryCount = 1,
     .entries    = &(WGPUBindGroupEntry) {
       // Binding 0 : Uniform buffer
       .binding = 0,
       .buffer  = uniform_buffer_vs.buffer,
       .offset  = 0,
       .size    = sizeof(ubo_vs),
     },
   }
  );
  ASSERT(bind_group != NULL);
}

// Describe the attachments used during rendering. This allows the driver to
// know up-front what the rendering will look like and is a good opportunity to
// optimize.
static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.1f,
        .g = 0.2f,
        .b = 0.3f,
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

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Pass matrices to the shaders
  camera_t* camera = context->camera;
  glm_mat4_copy(camera->matrices.perspective, ubo_vs.projection_matrix);
  glm_mat4_copy(camera->matrices.view, ubo_vs.view_matrix);
  glm_mat4_identity(ubo_vs.model_matrix);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &ubo_vs, uniform_buffer_vs.count);
}

// Prepare and initialize a uniform buffer block containing shader uniforms
// All Shader uniforms are passed via uniform buffer blocks
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Create the uniform bind group (note 'rotDeg' is copied here, not bound in
  // any way)
  uniform_buffer_vs.buffer = wgpu_create_buffer_from_data(
    context->wgpu_context, &ubo_vs, sizeof(ubo_vs), WGPUBufferUsage_Uniform);
  uniform_buffer_vs.count = sizeof(ubo_vs);

  update_uniform_buffers(context);
}

/* Create the graphics pipeline */
static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Construct the different states making up the pipeline */

  /* Primitive state */
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state                   = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(triangle, sizeof(float) * 6,
                            /* Attribute location 0: Position */
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                                               offsetof(vertex_t, position)),
                            /* Attribute location 1: Color */
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                                               offsetof(vertex_t, color)))

  /* Vertex state */
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      /* Vertex shader WGSL */
      .label             = "Triangle - Vertex shader WGSL",
      .wgsl_code.source  = triangle_vertex_shader_wgsl,
      .entry             = "main",
    },
    .buffer_count = 1,
    .buffers = &triangle_vertex_buffer_layout,
  });

  /* Fragment state */
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      /* Fragment shader WGSL */
      .label             = "Triangle - Fragment shader WGSL",
      .wgsl_code.source  = triangle_fragment_shader_wgsl,
      .entry             = "main",
    },
    .target_count = 1,
    .targets = &color_target_state_desc,
  });

  /* Multisample state */
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Create rendering pipeline using the specified states */
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Triangle - Render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state_desc,
                            .vertex       = vertex_state_desc,
                            .fragment     = &fragment_state_desc,
                            .depthStencil = &depth_stencil_state_desc,
                            .multisample  = multisample_state_desc,
                          });
  ASSERT(pipeline != NULL);

  /* Shader modules are no longer needed once the graphics pipeline has been
     created */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    /* Setup a default look-at camera */
    setup_camera(context);
    /* Initialize vertex and index buffers */
    prepare_vertex_and_index_buffers(context->wgpu_context);
    /* Prepare and initialize a uniform buffer block containing shader uniforms
     */
    prepare_uniform_buffers(context);
    /* Create the pipeline layout that is used to generate the rendering
     * pipelines */
    setup_pipeline_layout(context->wgpu_context);
    /* Setup bind groups */
    setup_bind_groups(context->wgpu_context);
    /* Create the graphics pipeline */
    prepare_pipelines(context->wgpu_context);
    /* Setup render pass */
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

// Build separate command buffer for the framebuffer image
// Unlike in OpenGL all rendering commands are recorded once into command
// buffers that are then resubmitted to the queue This allows to generate work
// upfront and from multiple threads
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

  // Set the bind group
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  // Set viewport
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Bind triangle vertex buffer (contains position and colors)
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);

  // Bind triangle index buffer
  wgpuRenderPassEncoderSetIndexBuffer(wgpu_context->rpass_enc, indices.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);

  // Draw indexed triangle
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, indices.count, 1, 0,
                                   0, 0);

  // Create command buffer and cleanup
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_context_t* wgpu_context)
{
  /* Get next image in the swap chain (back/front buffer) */
  wgpu_swap_chain_get_current_image(wgpu_context);

  /* Create command buffer */
  WGPUCommandBuffer command_buffer = build_command_buffer(wgpu_context);
  ASSERT(command_buffer != NULL);

  /* Submit command buffer to the queue */
  wgpu_flush_command_buffers(wgpu_context, &command_buffer, 1);

  /* Present the current buffer to the swap chain */
  wgpu_swap_chain_present(wgpu_context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context->wgpu_context);
}

// This function is called by the base example class each time the view is
// changed by user input
static void example_on_view_changed(wgpu_example_context_t* context)
{
  // Update the uniform buffer when the view is changed by user input
  update_uniform_buffers(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_triangle(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .vsync = true,
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
static const char* triangle_vertex_shader_wgsl = CODE(
  struct UBO {
    projectionMatrix : mat4x4<f32>,
    modelMatrix      : mat4x4<f32>,
    viewMatrix       : mat4x4<f32>,
  }

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) color : vec3<f32>,
  };

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>,
  }

  @vertex
  fn main(vertex : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix
                      * vec4<f32>(vertex.position.xyz, 1.0);
    output.color = vertex.color;
    return output;
  }
);

static const char* triangle_fragment_shader_wgsl = CODE(
  struct FragmentInput {
    @location(0) color : vec3<f32>,
  }

  struct FragmentOutput {
    @location(0) color : vec4<f32>,
  }

  @fragment
  fn main(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = vec4<f32>(fragment.color, 1.0);
    return output;
  }
);
// clang-format on
