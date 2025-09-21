#include "webgpu/wgpu_common.h"

#include "core/camera.h"

#include <cglm/cglm.h>

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

/* Vertex layout used in this example */
typedef struct {
  vec3 position;
  vec3 color;
} vertex_t;

/* State struct */
static struct {
  /* Vertex buffer and attributes */
  struct {
    WGPUBuffer buffer;
    uint32_t count;
  } vertices;
  /* Index buffer */
  struct {
    WGPUBuffer buffer;
    uint32_t count;
  } indices;
  /* Uniform buffer block object */
  struct {
    WGPUBuffer buffer;
    uint32_t count;
  } uniform_buffer_vs;
  /* Uniform block vertex shader */
  struct {
    mat4 projection_matrix;
    mat4 model_matrix;
    mat4 view_matrix;
  } ubo_vs;
  /* Camera object */
  camera_t camera;
  WGPUBool view_updated;
  /* The pipeline layout */
  WGPUPipelineLayout pipeline_layout;
  /* Pipeline */
  WGPURenderPipeline pipeline;
  // The bind group layout describes the shader binding layout (without actually
  // referencing descriptor)
  // Like the pipeline layout it's pretty much a blueprint and can be used with
  // different descriptor sets as long as their layout matches
  WGPUBindGroupLayout bind_group_layout;
  // The bind group stores the resources bound to the binding points in a shader
  // It connects the binding points of the different shaders with the buffers
  // and images used for those bindings
  WGPUBindGroup bind_group;
  /* Render pass descriptor for frame buffer writes */
  // Describe the attachments used during rendering. This allows the driver to
  // know up-front what the rendering will look like and is a good opportunity
  // to optimize.
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } render_pass;
  WGPUBool initialized;
} state = {
  .render_pass = {
    .color_attachment = {
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.1, 0.2, 0.3, 1.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    },
    .depth_stencil_attachment = {
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    },
    .descriptor = {
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.render_pass.color_attachment,
      .depthStencilAttachment = &state.render_pass.depth_stencil_attachment,
    },
  }
};

// Initialize a default look-at camera
static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type = CameraType_LookAt;
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -2.5f});
  camera_set_rotation(&state.camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.0f, 256.0f);
}

/* Initialize vertex and index buffers for an indexed triangle */
static void init_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  /* Setup vertices (x, y, z, r, g, b) */
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
  state.vertices.count        = (uint32_t)ARRAY_SIZE(vertex_buffer);
  uint32_t vertex_buffer_size = state.vertices.count * sizeof(vertex_t);

  /* Setup indices */
  static const uint16_t index_buffer[4] = {
    0, 1, 2, 0 /* padding */
  };
  state.indices.count        = (uint32_t)ARRAY_SIZE(index_buffer);
  uint32_t index_buffer_size = state.indices.count * sizeof(uint32_t);

  /* Create vertex buffer */
  state.vertices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, vertex_buffer, vertex_buffer_size, WGPUBufferUsage_Vertex);

  /* Create index buffer */
  state.indices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, index_buffer, index_buffer_size, WGPUBufferUsage_Index);
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Setup layout of descriptors used in this example
  // Basically connects the different shader stages to descriptors for binding
  // uniform buffers, image samplers, etc. So every shader binding should map to
  // one descriptor set layout binding

  // Bind group layout
  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor) {
      .label      = STRVIEW("Triangle - Bind group layout"),
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry) {
        /* Binding 0: Uniform buffer (Vertex shader) */
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = sizeof(state.ubo_vs),
        },
        .sampler = {0},
      }
    }
  );
  ASSERT(state.bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this descriptor set layout
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Triangle - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Bind Group */
  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor) {
     .label      = STRVIEW("Triangle - Bind group"),
     .layout     = state.bind_group_layout,
     .entryCount = 1,
     .entries    = &(WGPUBindGroupEntry) {
       /* Binding 0 : Uniform buffer */
       .binding = 0,
       .buffer  = state.uniform_buffer_vs.buffer,
       .offset  = 0,
       .size    = sizeof(state.ubo_vs),
     },
   }
  );
  ASSERT(state.bind_group != NULL);
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Pass matrices to the shaders */
  glm_mat4_copy(state.camera.matrices.perspective,
                state.ubo_vs.projection_matrix);
  glm_mat4_copy(state.camera.matrices.view, state.ubo_vs.view_matrix);
  glm_mat4_identity(state.ubo_vs.model_matrix);

  /* Map uniform buffer and update it */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_vs.buffer, 0,
                       &state.ubo_vs, state.uniform_buffer_vs.count);
}

// Prepare and initialize a uniform buffer block containing shader uniforms
// All Shader uniforms are passed via uniform buffer blocks
static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Create the uniform bind group (note 'rotDeg' is copied here, not bound in
  // any way)
  state.uniform_buffer_vs.buffer = wgpu_create_buffer_from_data(
    wgpu_context, &state.ubo_vs, sizeof(state.ubo_vs), WGPUBufferUsage_Uniform);
  state.uniform_buffer_vs.count = sizeof(state.ubo_vs);

  update_uniform_buffers(wgpu_context);
}

/* Create the graphics pipeline */
static void init_pipeline(wgpu_context_t* wgpu_context)
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
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
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
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, triangle_vertex_shader_wgsl);
  WGPUVertexState vertex_state_desc = {
    .module      = vert_shader_module,
    .entryPoint  = STRVIEW("main"),
    .bufferCount = 1,
    .buffers     = &triangle_vertex_buffer_layout,
  };

  /* Fragment state */
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, triangle_fragment_shader_wgsl);
  WGPUFragmentState fragment_state_desc = {
    .entryPoint  = STRVIEW("main"),
    .module      = frag_shader_module,
    .targetCount = 1,
    .targets     = &color_target_state_desc,
  };

  /* Multisample state */
  WGPUMultisampleState multisample_state_desc = {
    .count = 1,
    .mask  = 0xffffffff,
  };

  /* Create rendering pipeline using the specified states */
  state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label     = STRVIEW("Triangle - Render pipeline"),
                            .layout    = state.pipeline_layout,
                            .primitive = primitive_state_desc,
                            .vertex    = vertex_state_desc,
                            .fragment  = &fragment_state_desc,
                            .depthStencil = &depth_stencil_state_desc,
                            .multisample  = multisample_state_desc,
                          });
  ASSERT(state.pipeline != NULL);

  /* Shader modules are no longer needed once the graphics pipeline has been
     created */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    /* Initialize a default look-at camera */
    init_camera(wgpu_context);
    /* Initialize vertex and index buffers */
    init_vertex_and_index_buffers(wgpu_context);
    /* Initialize a uniform buffer block containing shader uniforms */
    init_uniform_buffers(wgpu_context);
    /* Create the pipeline layout that is used to generate the rendering
     * pipelines */
    init_pipeline_layout(wgpu_context);
    /* Setup bind groups */
    init_bind_groups(wgpu_context);
    /* Create the graphics pipeline */
    init_pipeline(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  UNUSED_VAR(wgpu_context);

  camera_on_input_event(&state.camera, input_event);
  state.view_updated = 1;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  if (state.view_updated) {
    update_uniform_buffers(wgpu_context);
    state.view_updated = 0;
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* Set target frame buffer */
  state.render_pass.color_attachment.view = wgpu_context->swapchain_view;
  state.render_pass.depth_stencil_attachment.view
    = wgpu_context->depth_stencil_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass.descriptor);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group, 0, 0);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(rpass_enc, 0.0f, 0.0f,
                                   (float)wgpu_context->width,
                                   (float)wgpu_context->height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(rpass_enc, 0u, 0u, wgpu_context->width,
                                      wgpu_context->height);

  /* Bind triangle vertex buffer (contains position and colors) */
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);

  /* Bind triangle index buffer */
  wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.indices.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);

  /* Draw indexed triangle */
  wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.indices.count, 1, 0, 0, 0);

  /* Create command buffer */
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

/* Clean up used resources */
static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Basic Indexed Triangle",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
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
