#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Wireframe and Thick-Line Rendering in WebGPU
 *
 * This example shows how to render a single indexed triangle model as mesh,
 * wireframe, or wireframe with thick lines, without the need to generate
 * additional buffers for line rendering.
 *
 * Uses vertex pulling to let the vertex shader decide which vertices to load,
 * which allows us to render indexed triangle meshes as wireframes or even
 * thick-wireframes.
 *  ** A normal wireframe is obtained by drawing 3 lines (6 vertices) per
 *     triangle. The vertex shader then uses the index buffer to load the
 *     triangle vertices in the order in which we need them to draw lines.
 *  ** A thick wireframe is obtained by rendering each of the 3 lines of a
 *     triangle as a quad (comprising 2 triangles). For each triangle of the
 *     indexed model, we are drawing a total of 3 lines/quads = 6 triangles = 18
 *     vertices. Each of these 18 vertices belongs to one of three lines, and
 *     each vertex shader invocation loads the start and end of the
 *     corresponding line. The line is then projected to screen space, and the
 *     orthoginal of the screen-space line direction is used to shift the
 *     vertices of each quad into the appropriate directions to obtain a thick
 *     line.
 *
 * Ref:
 * https://github.com/m-schuetz/webgpu_wireframe_thicklines
 * https://potree.org/permanent/wireframe_rendering/ (requires Chrome 96)
 * https://xeolabs.com/pdfs/OpenGLInsights.pdf
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* render_points_wgsl;
static const char* render_solid_mesh_wgsl;
static const char* render_wireframe_thick_wgsl;
static const char* render_wireframe_wgsl;

/* -------------------------------------------------------------------------- *
 * Wireframe and Thick-Line Rendering example
 * -------------------------------------------------------------------------- */

#define SWITCH_RENDER_TIME_INTERVAL (5.0f) /* seconds unit */

/* Cube mesh */
static indexed_cube_mesh_t indexed_cube_mesh = {0};

/* Cube struct */
typedef struct cube_t {
  WGPUBindGroup uniform_buffer_bind_group;
  WGPUBindGroupLayout bind_group_layout;
  /* Vertex buffer */
  wgpu_buffer_t positions;
  /* Colors */
  wgpu_buffer_t colors;
  /* Index buffer */
  wgpu_buffer_t indices;
  /* Uniform buffer block object */
  wgpu_buffer_t uniform_buffer_vs;
  /* View matrices */
  struct view_matrices_t {
    mat4 world;
    mat4 view;
    mat4 proj;
    float screen_width;
    float screen_height;
    float padding[12];
  } view_matrices;
} cube_t;

/* Render modes */
typedef enum render_mode_enum_t {
  RenderMode_Solid_Mesh      = 0,
  RenderMode_Points          = 1,
  RenderMode_Wireframe       = 2,
  RenderMode_Wireframe_Thick = 3,
  RenderMode_Count           = 4,
} render_mode_enum_t;

/* Render mode descriptions */
static const char* render_modes_desc[4]
  = {"Solid", "Points", "Wireframe", "Wireframe Thick"};

/* State struct */
static struct {
  cube_t cube;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipelines[4];
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  struct {
    render_mode_enum_t current_render_mode;
  } settings;
  float prev_render_mode_switch_time;
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
  .render_pass_dscriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .settings = {
    .current_render_mode = RenderMode_Solid_Mesh,
  },
  .prev_render_mode_switch_time = -1.0f,
};

static void init_cube_mesh(void)
{
  indexed_cube_mesh_init(&indexed_cube_mesh);
}

static void init_storage_buffers(wgpu_context_t* wgpu_context)
{
  /* Create position buffer */
  state.cube.positions = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube position - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage,
                    .size  = sizeof(indexed_cube_mesh.vertex_array),
                    .initial.data = indexed_cube_mesh.vertex_array,
                  });

  /* Create color buffer */
  state.cube.colors = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube color - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage,
                    .size  = sizeof(indexed_cube_mesh.color_array),
                    .initial.data = indexed_cube_mesh.color_array,
                  });
}

static void init_index_buffer(wgpu_context_t* wgpu_context)
{
  /* Create index buffer */
  state.cube.indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Index buffer",
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size         = sizeof(indexed_cube_mesh.index_array),
                    .count        = indexed_cube_mesh.index_count,
                    .initial.data = indexed_cube_mesh.index_array,
                  });
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /*  Binding 0: uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = state.cube.uniform_buffer_vs.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: positions */
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .hasDynamicOffset = false,
        .minBindingSize   = state.cube.positions.size,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2: colors */
      .binding    = 2,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .hasDynamicOffset = false,
        .minBindingSize   = state.cube.colors.size,
      },
      .sampler = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      /* Binding 3: indices */
      .binding    = 3,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .hasDynamicOffset = false,
        .minBindingSize   = state.cube.indices.size,
      },
      .sampler = {0},
    }
  };
  state.cube.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Cube - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.cube.bind_group_layout != NULL)

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Cube - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts = &state.cube.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL)
}

static void update_view_matrices(wgpu_context_t* wgpu_context)
{
  glm_mat4_identity(state.cube.view_matrices.view);
  glm_translate(state.cube.view_matrices.view, (vec3){0.0f, 0.0f, -4.0f});
  const float now = stm_sec(stm_now());
  glm_rotate(state.cube.view_matrices.view, 1.0f,
             (vec3){sin(now), cos(now), 0.0f});

  glm_mat4_identity(state.cube.view_matrices.world);
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;
  glm_mat4_identity(state.cube.view_matrices.proj);
  glm_perspective((2 * PI) / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  state.cube.view_matrices.proj);

  state.cube.view_matrices.screen_width  = (float)wgpu_context->width;
  state.cube.view_matrices.screen_height = (float)wgpu_context->height;
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update the view matrices */
  update_view_matrices(wgpu_context);

  /* Map uniform buffer and update it */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.cube.uniform_buffer_vs.buffer,
                       0, &state.cube.view_matrices,
                       state.cube.uniform_buffer_vs.size);
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Create uniform buffer */
  state.cube.uniform_buffer_vs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.cube.view_matrices),
                  });
  ASSERT(state.cube.uniform_buffer_vs.buffer != NULL)

  /* Update uniform buffer */
  update_uniform_buffers(wgpu_context);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0: uniform buffer */
      .binding = 0,
      .buffer  = state.cube.uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = state.cube.uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
       /* Binding 1: positions */
      .binding = 1,
      .buffer  = state.cube.positions.buffer,
      .offset  = 0,
      .size    = state.cube.positions.size,
    },
    [2] = (WGPUBindGroupEntry) {
      /* Binding 2: colors */
      .binding = 2,
      .buffer  = state.cube.colors.buffer,
      .offset  = 0,
      .size    = state.cube.colors.size,
    },
    [3] = (WGPUBindGroupEntry) {
      /* Binding 3: indices */
      .binding = 3,
      .buffer  = state.cube.indices.buffer,
      .offset  = 0,
      .size    = state.cube.indices.size,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Cube - Bind group layout"),
    .layout     = state.cube.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.cube.uniform_buffer_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.cube.uniform_buffer_bind_group != NULL)
}

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Multisample state */
  WGPUMultisampleState multisample_state = {
    .count = 1,
    .mask  = 0xffffffff,
  };

  /* Render pipeline: Solid mesh */
  {
    WGPUShaderModule vert_shader_module
      = wgpu_create_shader_module(wgpu_context->device, render_solid_mesh_wgsl);
    WGPUShaderModule frag_shader_module
      = wgpu_create_shader_module(wgpu_context->device, render_solid_mesh_wgsl);

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Solid mesh - Render pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module      = vert_shader_module,
        .entryPoint  = STRVIEW("main_vertex"),
      },
      .fragment = &(WGPUFragmentState) {
        .entryPoint  = STRVIEW("main_fragment"),
        .module      = frag_shader_module,
        .targetCount = 1,
        .targets     = &color_target_state,
      },
      .primitive    = primitive_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

    state.render_pipelines[(uint32_t)RenderMode_Solid_Mesh]
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.render_pipelines[(uint32_t)RenderMode_Solid_Mesh] != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module);
    WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module);
  }

  /* Render pipeline: Points */
  {
    primitive_state.topology = WGPUPrimitiveTopology_PointList;

    WGPUShaderModule shader_module
      = wgpu_create_shader_module(wgpu_context->device, render_points_wgsl);

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Points - Render pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("main_vertex"),
      },
      .fragment = &(WGPUFragmentState) {
        .entryPoint  = STRVIEW("main_fragment"),
        .module      = shader_module,
        .targetCount = 1,
        .targets     = &color_target_state,
      },
      .primitive    = primitive_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

    state.render_pipelines[(uint32_t)RenderMode_Points]
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.render_pipelines[(uint32_t)RenderMode_Points] != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, shader_module);
  }

  /* Render pipeline: Wireframe */
  {
    primitive_state.topology = WGPUPrimitiveTopology_LineList;

    WGPUShaderModule shader_module
      = wgpu_create_shader_module(wgpu_context->device, render_wireframe_wgsl);

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Wireframe - Render pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("main_vertex"),
      },
      .fragment = &(WGPUFragmentState) {
        .entryPoint  = STRVIEW("main_fragment"),
        .module      = shader_module,
        .targetCount = 1,
        .targets     = &color_target_state,
      },
      .primitive    = primitive_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

    state.render_pipelines[(uint32_t)RenderMode_Wireframe]
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.render_pipelines[(uint32_t)RenderMode_Wireframe] != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, shader_module);
  }

  /* Render pipeline: Wireframe Thick */
  {
    primitive_state.topology = WGPUPrimitiveTopology_TriangleList;

    WGPUShaderModule shader_module = wgpu_create_shader_module(
      wgpu_context->device, render_wireframe_thick_wgsl);

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Wireframe thick - Render pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("main_vertex"),
      },
      .fragment = &(WGPUFragmentState) {
        .entryPoint  = STRVIEW("main_fragment"),
        .module      = shader_module,
        .targetCount = 1,
        .targets     = &color_target_state,
      },
      .primitive    = primitive_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

    state.render_pipelines[(uint32_t)RenderMode_Wireframe_Thick]
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.render_pipelines[(uint32_t)RenderMode_Wireframe_Thick]
           != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, shader_module);
  }
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_cube_mesh();
    init_storage_buffers(wgpu_context);
    init_index_buffer(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_bind_groups(wgpu_context);
    init_pipelines(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void update_render_mode(void)
{
  if (state.prev_render_mode_switch_time < 0.0f) {
    state.prev_render_mode_switch_time = stm_sec(stm_now());
    return;
  }

  const float now = stm_sec(stm_now());
  if ((now - state.prev_render_mode_switch_time)
      > SWITCH_RENDER_TIME_INTERVAL) {
    state.settings.current_render_mode
      = (state.settings.current_render_mode + 1) % RenderMode_Count;
    state.prev_render_mode_switch_time = now;
    printf("Switched to new render mode: %s\n",
           render_modes_desc[state.settings.current_render_mode]);
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update matrix data */
  update_uniform_buffers(wgpu_context);

  /* Update render mode */
  update_render_mode();

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_dscriptor);

  wgpuRenderPassEncoderSetPipeline(
    rpass_enc,
    state.render_pipelines[(uint32_t)state.settings.current_render_mode]);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                    state.cube.uniform_buffer_bind_group, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 1, state.cube.positions.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 2, state.cube.colors.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 3, state.cube.indices.buffer,
                                       0, WGPU_WHOLE_SIZE);

  if (state.settings.current_render_mode == RenderMode_Solid_Mesh) {
    /* Bind index buffer */
    wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.cube.indices.buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        state.cube.indices.size);
    /* Draw indexed cube */
    wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.cube.indices.count, 1, 0,
                                     0, 0);
  }
  else if (state.settings.current_render_mode == RenderMode_Points) {
    wgpuRenderPassEncoderDraw(rpass_enc, indexed_cube_mesh.vertex_count, 1, 0,
                              0);
  }
  else if (state.settings.current_render_mode == RenderMode_Wireframe) {
    const uint32_t num_triangles = indexed_cube_mesh.index_count / 3;
    wgpuRenderPassEncoderDraw(rpass_enc, 6 * num_triangles, 1, 0, 0);
  }
  else if (state.settings.current_render_mode == RenderMode_Wireframe_Thick) {
    const uint32_t num_triangles = indexed_cube_mesh.index_count / 3;
    wgpuRenderPassEncoderDraw(rpass_enc, 3 * 6 * num_triangles, 1, 0, 0);
  }

  /* End render pass */
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

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.cube.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.cube.uniform_buffer_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube.positions.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube.colors.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube.uniform_buffer_vs.buffer)
  for (uint32_t i = 0; i < ARRAY_SIZE(state.render_pipelines); ++i) {
    WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipelines[i])
  }
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Wireframe and Thick-Line Rendering in WebGPU",
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
static const char* render_points_wgsl = CODE(
  struct Uniforms {
    world         : mat4x4<f32>,
    view          : mat4x4<f32>,
    proj          : mat4x4<f32>,
    screen_width  : f32,
    screen_height : f32
  }

  struct U32s {
      values : array<u32>
  }

  struct F32s {
    values : array<f32>
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<storage, read> positions : F32s;
  @binding(2) @group(0) var<storage, read> colors : U32s;
  @binding(3) @group(0) var<storage, read> indices : U32s;

  struct VertexInput {
    @builtin(instance_index) instanceID : u32,
    @builtin(vertex_index) vertexID : u32
  }

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>
  }

  @vertex
  fn main_vertex(vertex : VertexInput) -> VertexOutput {
    var position = vec4<f32>(
        positions.values[3u * vertex.vertexID + 0u],
        positions.values[3u * vertex.vertexID + 1u],
        positions.values[3u * vertex.vertexID + 2u],
        1.0
    );

    position = uniforms.proj * uniforms.view * uniforms.world * position;

    var color_u32 = colors.values[vertex.vertexID];
    var color = vec4<f32>(
        f32((color_u32 >>  0u) & 0xFFu) / 255.0,
        f32((color_u32 >>  8u) & 0xFFu) / 255.0,
        f32((color_u32 >> 16u) & 0xFFu) / 255.0,
        f32((color_u32 >> 24u) & 0xFFu) / 255.0,
    );

    var output : VertexOutput;
    output.position = position;
    output.color = color;

    return output;
  }

  struct FragmentInput {
    @location(0) color : vec4<f32>
  }

  struct FragmentOutput {
    @location(0) color : vec4<f32>
  }

  @fragment
  fn main_fragment(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = fragment.color;

    return output;
  }
);

static const char* render_solid_mesh_wgsl = CODE(
  struct Uniforms {
    world         : mat4x4<f32>,
    view          : mat4x4<f32>,
    proj          : mat4x4<f32>,
    screen_width  : f32,
    screen_height : f32
  }

  struct U32s {
    values : array<u32>
  }

  struct F32s {
    values : array<f32>
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<storage, read> positions : F32s;
  @binding(2) @group(0) var<storage, read> colors : U32s;
  @binding(3) @group(0) var<storage, read> indices : U32s;

  struct VertexInput {
    @builtin(instance_index) instanceID : u32,
    @builtin(vertex_index) vertexID : u32
  }

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>
  }

  @vertex
  fn main_vertex(vertex : VertexInput) -> VertexOutput {
    var position = vec4<f32>(
        positions.values[3u * vertex.vertexID + 0u],
        positions.values[3u * vertex.vertexID + 1u],
        positions.values[3u * vertex.vertexID + 2u],
        1.0
    );

    position = uniforms.proj * uniforms.view * uniforms.world * position;

    var color_u32 = colors.values[vertex.vertexID];
    var color = vec4<f32>(
        f32((color_u32 >>  0u) & 0xFFu) / 255.0,
        f32((color_u32 >>  8u) & 0xFFu) / 255.0,
        f32((color_u32 >> 16u) & 0xFFu) / 255.0,
        f32((color_u32 >> 24u) & 0xFFu) / 255.0,
    );

    var output : VertexOutput;
    output.position = position;
    output.color = color;

    return output;
  }

  struct FragmentInput {
    @location(0) color : vec4<f32>
  }

  struct FragmentOutput {
    @location(0) color : vec4<f32>
  }

  @fragment
  fn main_fragment(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = fragment.color;

    return output;
  }
);

static const char* render_wireframe_thick_wgsl = CODE(
  struct Uniforms {
    world         : mat4x4<f32>,
    view          : mat4x4<f32>,
    proj          : mat4x4<f32>,
    screen_width  : f32,
    screen_height : f32
  }

  struct U32s {
    values : array<u32>
  }

  struct F32s {
    values : array<f32>
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<storage, read> positions : F32s;
  @binding(2) @group(0) var<storage, read> colors : U32s;
  @binding(3) @group(0) var<storage, read> indices : U32s;

  struct VertexInput {
    @builtin(instance_index) instanceID : u32,
    @builtin(vertex_index) vertexID : u32
  }

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>
  }

  @vertex
  fn main_vertex(vertex : VertexInput) -> VertexOutput {
    var lineWidth = 5.0;

    var localToElement = array<u32, 6>(0u, 1u, 1u, 2u, 2u, 0u);

    var triangleIndex = vertex.vertexID / 18u;        // 18 vertices per triangle
    var localVertexIndex = vertex.vertexID % 18u;     // 18 vertices
    var localLineIndex = localVertexIndex / 6u;       // 3 lines, 6 vertices per line, 2 triangles per line

    var startElementIndex = indices.values[3u * triangleIndex + localLineIndex + 0u];
    var endElementIndex = indices.values[3u * triangleIndex + (localLineIndex + 1u) % 3u];

    var start = vec4<f32>(
      positions.values[3u * startElementIndex + 0u],
      positions.values[3u * startElementIndex + 1u],
      positions.values[3u * startElementIndex + 2u],
      1.0
    );

    var end = vec4<f32>(
      positions.values[3u * endElementIndex + 0u],
      positions.values[3u * endElementIndex + 1u],
      positions.values[3u * endElementIndex + 2u],
      1.0
    );

    var localIndex = vertex.vertexID % 6u;

    var position = start;
    var currElementIndex = startElementIndex;
    if (localIndex == 0u || localIndex == 3u|| localIndex == 5u){
      position = start;
      currElementIndex = startElementIndex;
    } else{
      position = end;
      currElementIndex = endElementIndex;
    }

    var worldPos = uniforms.world * position;
    var viewPos = uniforms.view * worldPos;
    var projPos = uniforms.proj * viewPos;

    var dirScreen : vec2<f32>;
    {
      var projStart = uniforms.proj * uniforms.view * uniforms.world * start;
      var projEnd = uniforms.proj * uniforms.view * uniforms.world * end;

      var screenStart = projStart.xy / projStart.w;
      var screenEnd = projEnd.xy / projEnd.w;

      dirScreen = normalize(screenEnd - screenStart);
    }

    { // apply pixel offsets to the 6 vertices of the quad
      var pxOffset = vec2<f32>(1.0, 0.0);

      // move vertices of quad sidewards
      if (localIndex == 0u || localIndex == 1u || localIndex == 3u){
        pxOffset = vec2<f32>(dirScreen.y, -dirScreen.x);
      } else{
        pxOffset = vec2<f32>(-dirScreen.y, dirScreen.x);
      }

      // move vertices of quad outwards
      if (localIndex == 0u || localIndex == 3u || localIndex == 5u){
         pxOffset = pxOffset - dirScreen;
      } else{
        pxOffset = pxOffset + dirScreen;
      }

      var screenDimensions = vec2<f32>(uniforms.screen_width, uniforms.screen_height);
      var adjusted = projPos.xy / projPos.w + lineWidth * pxOffset / screenDimensions;
      projPos = vec4<f32>(adjusted * projPos.w, projPos.zw);
    }

    var color_u32 = colors.values[currElementIndex];
    var color = vec4<f32>(
      f32((color_u32 >>  0u) & 0xFFu) / 255.0,
      f32((color_u32 >>  8u) & 0xFFu) / 255.0,
      f32((color_u32 >> 16u) & 0xFFu) / 255.0,
      f32((color_u32 >> 24u) & 0xFFu) / 255.0,
    );
    // var color = vec4<f32>(0.0, 1.0, 0.0, 1.0);

    var output : VertexOutput;
    output.position = projPos;
    output.color = color;

    return output;
  }

  struct FragmentInput {
    @location(0) color : vec4<f32>
  }

  struct FragmentOutput {
    @location(0) color : vec4<f32>
  }

  @fragment
  fn main_fragment(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = fragment.color;

    return output;
  }
);

static const char* render_wireframe_wgsl = CODE(
  struct Uniforms {
    world         : mat4x4<f32>,
    view          : mat4x4<f32>,
    proj          : mat4x4<f32>,
    screen_width  : f32,
    screen_height : f32
  }

  struct U32s {
    values : array<u32>
  }

  struct F32s {
    values : array<f32>
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<storage, read> positions : F32s;
  @binding(2) @group(0) var<storage, read> colors : U32s;
  @binding(3) @group(0) var<storage, read> indices : U32s;

  struct VertexInput {
    @builtin(instance_index) instanceID : u32,
    @builtin(vertex_index) vertexID : u32
  }

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>
  }

  @vertex
  fn main_vertex(vertex : VertexInput) -> VertexOutput {
    var localToElement = array<u32, 6>(0u, 1u, 1u, 2u, 2u, 0u);

    var triangleIndex = vertex.vertexID / 6u;
    var localVertexIndex = vertex.vertexID % 6u;

    var elementIndexIndex = 3u * triangleIndex + localToElement[localVertexIndex];
    var elementIndex = indices.values[elementIndexIndex];

    var position = vec4<f32>(
        positions.values[3u * elementIndex + 0u],
        positions.values[3u * elementIndex + 1u],
        positions.values[3u * elementIndex + 2u],
        1.0
    );

    position = uniforms.proj * uniforms.view * uniforms.world * position;

    var color_u32 = colors.values[elementIndex];
    var color = vec4<f32>(
        f32((color_u32 >>  0u) & 0xFFu) / 255.0,
        f32((color_u32 >>  8u) & 0xFFu) / 255.0,
        f32((color_u32 >> 16u) & 0xFFu) / 255.0,
        f32((color_u32 >> 24u) & 0xFFu) / 255.0,
    );

    var output : VertexOutput;
    output.position = position;
    output.color = color;

    return output;
  }

  struct FragmentInput {
    @location(0) color : vec4<f32>
  }

  struct FragmentOutput {
    @location(0) color : vec4<f32>
  }

  @fragment
  fn main_fragment(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = fragment.color;

    return output;
  }
);
// clang-format on
