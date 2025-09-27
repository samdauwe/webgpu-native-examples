#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#include <stdio.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Cube Reflection
 *
 * This example shows how to create a basic reflection pipeline.
 *
 * Ref:
 * https://dawn.googlesource.com/dawn/+/refs/heads/main/examples/CubeReflection.cpp
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* fragment_shader_wgsl;
static const char* reflection_fragment_shader_wgsl;
static const char* vertex_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Cube Reflection Example
 * -------------------------------------------------------------------------- */

/* State struct */
static struct {
  wgpu_buffer_t indices;
  wgpu_buffer_t cube_vertices;
  wgpu_buffer_t plane_vertices;
  struct {
    mat4 view;
    mat4 proj;
  } camera_data;
  wgpu_buffer_t camera_buffer;
  wgpu_buffer_t transform_buffer[2];
  WGPUBindGroup bind_group[2];
  WGPUBindGroupLayout bind_group_layout;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
  WGPURenderPipeline plane_pipeline;
  WGPURenderPipeline reflection_pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  struct {
    uint32_t a;
    float b;
  } render_state;
  WGPUBool initialized;
} state = {
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
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
};

static void init_buffers(wgpu_context_t* wgpu_context)
{
  /* Index buffer */
  {
    static const uint32_t index_data[6 * 6] = {
      0,  1,  2,  0,  2,  3, //

      4,  5,  6,  4,  6,  7, //

      8,  9,  10, 8,  10, 11, //

      12, 13, 14, 12, 14, 15, //

      16, 17, 18, 16, 18, 19, //

      20, 21, 22, 20, 22, 23, //
    };
    state.indices = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Cube - Indices buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                      .size  = sizeof(index_data),
                      .count = (uint32_t)ARRAY_SIZE(index_data),
                      .initial.data = index_data,
                    });
  }

  /* Cube vertices data */
  {
    static const float vertex_data[6 * 4 * 6] = {
      -1.f, -1.f, 1.f,  1.f, 0.f, 0.f, 1.f,  -1.f, 1.f,  1.f, 0.f, 0.f, //
      1.f,  1.f,  1.f,  1.f, 0.f, 0.f, -1.f, 1.f,  1.f,  1.f, 0.f, 0.f, //

      -1.f, -1.f, -1.f, 1.f, 1.f, 0.f, -1.f, 1.f,  -1.f, 1.f, 1.f, 0.f, //
      1.f,  1.f,  -1.f, 1.f, 1.f, 0.f, 1.f,  -1.f, -1.f, 1.f, 1.f, 0.f, //

      -1.f, 1.f,  -1.f, 1.f, 0.f, 1.f, -1.f, 1.f,  1.f,  1.f, 0.f, 1.f, //
      1.f,  1.f,  1.f,  1.f, 0.f, 1.f, 1.f,  1.f,  -1.f, 1.f, 0.f, 1.f, //

      -1.f, -1.f, -1.f, 0.f, 1.f, 0.f, 1.f,  -1.f, -1.f, 0.f, 1.f, 0.f, //
      1.f,  -1.f, 1.f,  0.f, 1.f, 0.f, -1.f, -1.f, 1.f,  0.f, 1.f, 0.f, //

      1.f,  -1.f, -1.f, 0.f, 1.f, 1.f, 1.f,  1.f,  -1.f, 0.f, 1.f, 1.f, //
      1.f,  1.f,  1.f,  0.f, 1.f, 1.f, 1.f,  -1.f, 1.f,  0.f, 1.f, 1.f, //

      -1.f, -1.f, -1.f, 1.f, 1.f, 1.f, -1.f, -1.f, 1.f,  1.f, 1.f, 1.f, //
      -1.f, 1.f,  1.f,  1.f, 1.f, 1.f, -1.f, 1.f,  -1.f, 1.f, 1.f, 1.f, //
    };
    state.cube_vertices = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Cube - Vertices buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = sizeof(vertex_data),
                      .count = (uint32_t)ARRAY_SIZE(vertex_data),
                      .initial.data = vertex_data,
                    });
  }

  /* Plane vertice data */
  {
    static const float plane_data[6 * 4] = {
      -2.f, -1.f, -2.f, 0.5f, 0.5f, 0.5f, 2.f,  -1.f, -2.f, 0.5f, 0.5f, 0.5f, //
      2.f,  -1.f, 2.f,  0.5f, 0.5f, 0.5f, -2.f, -1.f, 2.f,  0.5f, 0.5f, 0.5f, //
    };
    state.plane_vertices = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Plane - Vertices buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = sizeof(plane_data),
                      .count = (uint32_t)ARRAY_SIZE(plane_data),
                      .initial.data = plane_data,
                    });
  }
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Camera data buffer */
  {
    state.camera_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Camera - Data buffer",
                                           .usage = WGPUBufferUsage_CopyDst
                                                    | WGPUBufferUsage_Uniform,
                                           .size = sizeof(state.camera_data),
                                         });
  }

  /* Camera projection matrix */
  glm_perspective(glm_rad(45.0f), 1.f, 1.0f, 100.0f, state.camera_data.proj);

  /* Transform buffers */
  {
    mat4 transform = GLM_MAT4_IDENTITY_INIT;
    state.transform_buffer[0]
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Transform buffer 0",
                                           .usage = WGPUBufferUsage_CopyDst
                                                    | WGPUBufferUsage_Uniform,
                                           .size         = sizeof(mat4),
                                           .initial.data = &transform,
                                         });

    glm_translate(transform, (vec3){0.f, -2.f, 0.f});
    state.transform_buffer[1]
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Transform buffer 1",
                                           .usage = WGPUBufferUsage_CopyDst
                                                    | WGPUBufferUsage_Uniform,
                                           .size         = sizeof(mat4),
                                           .initial.data = &transform,
                                         });
  }
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0: Uniform buffer (Vertex shader) => cameraData */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(state.camera_data),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Uniform buffer (Vertex shader) => modelData */
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(mat4),
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = STRVIEW("Bind group layout"),
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  state.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.bind_group_layout != NULL);

  /* Pipeline layout */
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .label                = STRVIEW("Pipeline layout"),
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &state.bind_group_layout,
  };
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                         &pipeline_layout_desc);
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Bind groups */
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(state.bind_group); ++i) {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = {
        /* Binding 0: Uniform buffer (Vertex shader) => cameraData */
        .binding = 0,
        .buffer  = state.camera_buffer.buffer,
        .offset  = 0,
        .size    = state.camera_buffer.size,
      },
      [1] = {
        /* Binding 1: Uniform buffer (Vertex shader) => modelData */
        .binding = 1,
        .buffer  = state.transform_buffer[i].buffer,
        .offset  = 0,
        .size    = state.transform_buffer[i].size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Uniform - Bind group"),
      .layout     = state.bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.bind_group[i]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.bind_group[i] != NULL);
  }
}

/* Create the graphics pipeline */
static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Vertex and fragment shaders */
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);
  WGPUShaderModule reflection_frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, reflection_fragment_shader_wgsl);

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    cube_reflection, 6 * sizeof(float),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    /* Attribute location 1: Color */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, 3 * sizeof(float)));

  /* Vertex state */
  WGPUVertexState vertex_state = {
    .module      = vert_shader_module,
    .entryPoint  = STRVIEW("main"),
    .bufferCount = 1,
    .buffers     = &cube_reflection_vertex_buffer_layout,
  };

  /* Fragment state */
  WGPUFragmentState fragment_state = {
    .entryPoint  = STRVIEW("main"),
    .module      = frag_shader_module,
    .targetCount = 1,
    .targets     = &color_target_state,
  };

  /* Reflection fragment state */
  WGPUFragmentState reflection_fragment_state = {
    .entryPoint  = STRVIEW("main"),
    .module      = reflection_frag_shader_module,
    .targetCount = 1,
    .targets     = &color_target_state,
  };

  /* Multisample state */
  WGPUMultisampleState multisample_state = {
    .count = 1,
    .mask  = 0xffffffff,
  };

  /* Cube rendering pipeline */
  {
    /* Depth stencil state */
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = wgpu_context->depth_stencil_format,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    WGPURenderPipelineDescriptor rp_desc = {
      .label        = STRVIEW("Cube - Render pipeline"),
      .layout       = state.pipeline_layout,
      .vertex       = vertex_state,
      .fragment     = &fragment_state,
      .primitive    = primitive_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

    state.pipeline
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.pipeline != NULL);
  }

  /* Plane rendering pipeline */
  {
    /* Depth stencil state */
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = wgpu_context->depth_stencil_format,
        .depth_write_enabled = false,
      });
    depth_stencil_state.stencilFront.passOp = WGPUStencilOperation_Replace;
    depth_stencil_state.stencilBack.passOp  = WGPUStencilOperation_Replace;
    depth_stencil_state.depthCompare        = WGPUCompareFunction_Less;

    WGPURenderPipelineDescriptor rp_desc = {
      .label        = STRVIEW("Plane - Render pipeline"),
      .layout       = state.pipeline_layout,
      .vertex       = vertex_state,
      .fragment     = &fragment_state,
      .primitive    = primitive_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

    state.plane_pipeline
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.plane_pipeline != NULL);
  }

  /* Cube reflection rendering pipeline */
  {
    /* Depth stencil state */
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = wgpu_context->depth_stencil_format,
        .depth_write_enabled = true,
      });
    depth_stencil_state.stencilFront.compare = WGPUCompareFunction_Equal;
    depth_stencil_state.stencilBack.compare  = WGPUCompareFunction_Equal;
    depth_stencil_state.stencilFront.passOp  = WGPUStencilOperation_Replace;
    depth_stencil_state.stencilBack.passOp   = WGPUStencilOperation_Replace;
    depth_stencil_state.depthCompare         = WGPUCompareFunction_Less;

    WGPURenderPipelineDescriptor rp_desc = {
      .label        = STRVIEW("Cube reflection - Render pipeline"),
      .layout       = state.pipeline_layout,
      .vertex       = vertex_state,
      .fragment     = &reflection_fragment_state,
      .primitive    = primitive_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

    state.reflection_pipeline
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.reflection_pipeline != NULL);
  }

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module);
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module);
  WGPU_RELEASE_RESOURCE(ShaderModule, reflection_frag_shader_module);
}

static void update_camera_view(wgpu_context_t* wgpu_context)
{
  /* Update render state */
  state.render_state.a = (state.render_state.a + 1) % 256;
  state.render_state.b += 0.002f;
  if (state.render_state.b >= 1.0f) {
    state.render_state.b = 0.0f;
  }

  /* Update camera view */
  glm_lookat(
    (vec3){8.0f * sin(glm_rad(state.render_state.b * 360.0f)),  /* x (eye) */
           2.0f,                                                /* y (eye) */
           8.0f * cos(glm_rad(state.render_state.b * 360.0f))}, /* z (eye) */
    (vec3){0.0f, 0.0f, 0.0f},                                   /* center  */
    (vec3){0.0f, 1.0f, 0.0f},                                   /* up      */
    state.camera_data.view);

  /* Update uniform buffer */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.camera_buffer.buffer, 0,
                       &state.camera_data, state.camera_buffer.size);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    init_buffers(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_bind_groups(wgpu_context);
    init_pipelines(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update camera */
  update_camera_view(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  {
    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);

    /* Render cube */
    {
      wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
      wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group[0], 0,
                                        0);
      wgpuRenderPassEncoderSetVertexBuffer(
        rpass_enc, 0, state.cube_vertices.buffer, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.indices.buffer,
                                          WGPUIndexFormat_Uint32, 0,
                                          WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDrawIndexed(rpass_enc, 36, 1, 0, 0, 0);
    }

    /* Render plane */
    {
      wgpuRenderPassEncoderSetStencilReference(rpass_enc, 0x1);
      wgpuRenderPassEncoderSetPipeline(rpass_enc, state.plane_pipeline);
      wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group[0], 0,
                                        0);
      wgpuRenderPassEncoderSetVertexBuffer(
        rpass_enc, 0, state.plane_vertices.buffer, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDrawIndexed(rpass_enc, 6, 1, 0, 0, 0);
    }

    /* Render cube reflection */
    {
      wgpuRenderPassEncoderSetPipeline(rpass_enc, state.reflection_pipeline);
      wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group[1], 0,
                                        0);
      wgpuRenderPassEncoderSetVertexBuffer(
        rpass_enc, 0, state.cube_vertices.buffer, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDrawIndexed(rpass_enc, 36, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(rpass_enc);
    wgpuRenderPassEncoderRelease(rpass_enc);
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  WGPU_RELEASE_RESOURCE(Buffer, state.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.plane_vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.camera_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.transform_buffer[0].buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.transform_buffer[1].buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group[0])
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group[1])
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.plane_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.reflection_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Cube Reflection",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* fragment_shader_wgsl = CODE(
  struct FragmentInput {
    @location(2) f_col : vec3<f32>
  };

  struct FragmentOutput {
    @location(0) fragColor : vec4<f32>
  };

  @fragment
  fn main(input : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.fragColor = vec4<f32>(input.f_col, 1.0);
    return output;
  }
);

static const char* reflection_fragment_shader_wgsl = CODE(
  struct FragmentInput {
    @location(2) f_col : vec3<f32>
  };

  struct FragmentOutput {
    @location(0) fragColor : vec4<f32>
  };

  @fragment
  fn main(input : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.fragColor = vec4(mix(input.f_col, vec3<f32>(0.5, 0.5, 0.5), 0.5), 1.0);
    return output;
  }
);

static const char* vertex_shader_wgsl = CODE(
  struct CameraData {
      view: mat4x4<f32>,
      proj: mat4x4<f32>,
  };
  @group(0) @binding(0) var<uniform> camera: CameraData;

  struct ModelData {
      modelMatrix: mat4x4<f32>,
  };
  @group(0) @binding(1) var<uniform> model: ModelData;

  struct VertexInput {
      @location(0) pos: vec3<f32>,
      @location(1) col: vec3<f32>,
  };

  struct VertexOutput {
      @builtin(position) position: vec4<f32>,
      @location(2) f_col: vec3<f32>,
  };

  @vertex
  fn main(input: VertexInput) -> VertexOutput {
      var output: VertexOutput;
      output.f_col = input.col;
      output.position = camera.proj * camera.view * model.modelMatrix
                        * vec4<f32>(input.pos, 1.0);
      return output;
  }
);
// clang-format on
