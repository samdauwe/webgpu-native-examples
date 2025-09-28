#include "webgpu/wgpu_common.h"

#include <stdio.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Conway's Game of Life
 *
 * This example shows how to make Conway's game of life. First, use compute
 * shader to calculate how cells grow or die. Then use render pipeline to draw
 * cells by using instance mesh.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/gameOfLife
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* compute_shader_wgsl;
static const char* fragment_shader_wgsl;
static const char* vertex_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Conway's Game of Life example
 * -------------------------------------------------------------------------- */

/* State struct */
static struct {
  struct {
    struct {
      wgpu_buffer_t handle;
    } square;
    struct {
      wgpu_buffer_t handle;
      uint32_t data[2];
    } size;
    struct {
      wgpu_buffer_t handle;
      uint32_t* data;
    } buffer0;
    struct {
      wgpu_buffer_t handle;
    } buffer1;
  } buffers;
  struct {
    WGPUBindGroup uniform;
    WGPUBindGroup bind_group0;
    WGPUBindGroup bind_group1;
  } bind_groups;
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
  } compute;
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
  } graphics;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  uint32_t whole_time;
  uint32_t loop_times;
  struct {
    uint32_t width;
    uint32_t height;
    uint32_t timestep;
    uint32_t workgroup_size;
  } game_options;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  .game_options = {
    .width          = 128,
    .height         = 128,
    .timestep       = 4,
    .workgroup_size = 8,
  },
};

static uint32_t get_cell_count(void)
{
  return state.game_options.width * state.game_options.height;
}

static void update_size_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Update the buffer data */
  state.buffers.size.data[0] = state.game_options.width;
  state.buffers.size.data[1] = state.game_options.height;

  // Map uniform buffer and update it
  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.size.handle.buffer, 0,
                       state.buffers.size.data, state.buffers.size.handle.size);
}

static void init_static_buffers(wgpu_context_t* wgpu_context)
{
  /* Square vertex buffer */
  {
    /* Setup vertices of the square */
    uint32_t square_vertices[8] = {0, 0, 0, 1, 1, 0, 1, 1};

    /* Create the vertex buffer */
    state.buffers.square.handle = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Square - Vertex buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = sizeof(square_vertices),
                      .initial.data = square_vertices,
                    });
  }

  /* Size uniform buffer */
  {
    state.buffers.size.handle = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Size - Uniform buffer",
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Uniform
                 | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
        .size = sizeof(state.buffers.size.data),
      });

    update_size_uniform_buffer(wgpu_context);
  }
}

static void init_storage_buffers(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_buffer(&state.buffers.buffer0.handle);
  wgpu_destroy_buffer(&state.buffers.buffer1.handle);

  if (state.buffers.buffer0.data) {
    free(state.buffers.buffer0.data);
    state.buffers.buffer0.data = NULL;
  }

  /* Update the buffer data */
  const uint32_t cell_count  = get_cell_count();
  const uint32_t length      = cell_count * sizeof(uint32_t);
  state.buffers.buffer0.data = (uint32_t*)malloc(length);
  for (uint32_t i = 0; i < cell_count; ++i) {
    state.buffers.buffer0.data[i]
      = random_float_min_max(0.0f, 1.0f) < 0.25f ? 1 : 0;
  }

  /* Storage buffer 0 */
  {
    state.buffers.buffer0.handle = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Storage buffer 0",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage
                               | WGPUBufferUsage_Vertex,
                      .size         = length,
                      .initial.data = state.buffers.buffer0.data,
                    });
    ASSERT(state.buffers.buffer0.handle.buffer != NULL);
  }

  /* Storage buffer 1 */
  {
    state.buffers.buffer1.handle = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Storage buffer 1",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage
                               | WGPUBufferUsage_Vertex,
                      .size = length,
                    });
    ASSERT(state.buffers.buffer1.handle.buffer != NULL);
  }
}

static void init_bind_group_layout_compute(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0: Storage buffer (size) */
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize   = state.buffers.size.handle.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Storage buffer (current) */
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize   = state.buffers.buffer0.handle.size,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2: Storage buffer (next) */
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Storage,
        .minBindingSize   = state.buffers.buffer1.handle.size,
      },
      .sampler = {0},
    }
  };
  state.compute.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Compute - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.compute.bind_group_layout != NULL);
}

static void init_bind_groups_compute(wgpu_context_t* wgpu_context)
{
  /* Bind group 0 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.buffers.size.handle.buffer,
        .size    = state.buffers.size.handle.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = state.buffers.buffer0.handle.buffer,
        .size    = state.buffers.buffer0.handle.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = state.buffers.buffer1.handle.buffer,
        .size    = state.buffers.buffer1.handle.size,
      },
    };
    state.bind_groups.bind_group0 = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Bind group - Compute 0"),
                              .layout     = state.compute.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.bind_group0 != NULL);
  }

  /* Bind group 1 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.buffers.size.handle.buffer,
        .size    = state.buffers.size.handle.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = state.buffers.buffer1.handle.buffer,
        .size    = state.buffers.buffer1.handle.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = state.buffers.buffer0.handle.buffer,
        .size    = state.buffers.buffer0.handle.size,
      },
    };
    state.bind_groups.bind_group1 = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Bind group - Compute 1"),
                              .layout     = state.compute.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.bind_group1 != NULL);
  }
}

static void init_pipeline_layout_compute(wgpu_context_t* wgpu_context)
{
  state.compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Pipeline layout - Compute"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.compute.bind_group_layout,
    });
  ASSERT(state.compute.pipeline_layout != NULL);
}

static void init_pipeline_compute(wgpu_context_t* wgpu_context)
{
  /* Compute shader constants */
  WGPUConstantEntry constant_entries[1] = {
    [0] = (WGPUConstantEntry){
      .key   = STRVIEW("blockSize"),
      .value = state.game_options.workgroup_size,
    },
  };

  /* Compute shader */
  WGPUShaderModule comp_shader_module
    = wgpu_create_shader_module(wgpu_context->device, compute_shader_wgsl);

  /* Create compute pipeline */
  state.compute.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Effect - Compute pipeline"),
      .layout  = state.compute.pipeline_layout,
      .compute = {
        .module        = comp_shader_module,
        .entryPoint    = STRVIEW("main"),
        .constantCount = (uint32_t)ARRAY_SIZE(constant_entries),
        .constants     = constant_entries,
      },
    });
  ASSERT(state.compute.pipeline != NULL);

  /* Partial cleanup */
  wgpuShaderModuleRelease(comp_shader_module);
}

static void init_bind_group_layout_graphics(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (size)
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .minBindingSize   = state.buffers.size.handle.size,
      },
      .sampler = {0},
    },
  };
  state.graphics.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Graphics - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.graphics.bind_group_layout != NULL);
}

static void init_bind_group_graphics(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = state.buffers.size.handle.buffer,
      .size    = state.buffers.size.handle.size,
    },
  };
  state.bind_groups.uniform = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Uniform - Bind group"),
                            .layout     = state.graphics.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.bind_groups.uniform != NULL);
}

static void init_pipeline_layout_graphics(wgpu_context_t* wgpu_context)
{
  state.graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Graphics - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.graphics.bind_group_layout,
    });
  ASSERT(state.graphics.pipeline_layout != NULL);
}

static void init_pipeline_graphics(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  // Vertex buffer layouts
  WGPU_VERTEX_BUFFER_LAYOUT(cell_stride, sizeof(uint32_t),
                            // Attribute location 0: Cell
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Uint32, 0))
  cell_stride_vertex_buffer_layout.stepMode = WGPUVertexStepMode_Instance;
  WGPU_VERTEX_BUFFER_LAYOUT(square_stride, 2 * sizeof(uint32_t),
                            // Attribute location 1: Position
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Uint32x2, 0))
  WGPUVertexBufferLayout vertex_state_buffers[2]
    = {cell_stride_vertex_buffer_layout, square_stride_vertex_buffer_layout};

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Graphics - Render pipeline"),
    .layout = state.graphics.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = (uint32_t)ARRAY_SIZE(vertex_state_buffers),
      .buffers     = vertex_state_buffers,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    },
    .multisample = {
      .count = 1,
      .mask  = 0xffffffff
    },
  };

  state.graphics.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.graphics.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    init_static_buffers(wgpu_context);
    init_storage_buffers(wgpu_context);
    /* Compute */
    init_bind_group_layout_compute(wgpu_context);
    init_bind_groups_compute(wgpu_context);
    init_pipeline_layout_compute(wgpu_context);
    init_pipeline_compute(wgpu_context);
    /* Graphics */
    init_bind_group_layout_graphics(wgpu_context);
    init_bind_group_graphics(wgpu_context);
    init_pipeline_layout_graphics(wgpu_context);
    init_pipeline_graphics(wgpu_context);
    state.initialized = 1;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  if (state.game_options.timestep) {
    state.whole_time++;
    if (state.whole_time >= state.game_options.timestep) {
      state.whole_time -= state.game_options.timestep;
      state.loop_times = 1 - state.loop_times;
    }
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass */
  {
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass_enc, state.compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(cpass_enc, 0,
                                       state.loop_times ?
                                         state.bind_groups.bind_group1 :
                                         state.bind_groups.bind_group0,
                                       0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      cpass_enc,
      ceil((float)state.game_options.width
           / (float)state.game_options.workgroup_size),
      ceil((float)state.game_options.height
           / (float)state.game_options.workgroup_size),
      1);
    wgpuComputePassEncoderEnd(cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, cpass_enc)
  }

  /* Graphics render pipeline */
  {
    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.graphics.pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0,
                                         state.loop_times ?
                                           state.buffers.buffer1.handle.buffer :
                                           state.buffers.buffer0.handle.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass_enc, 1, state.buffers.square.handle.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_groups.uniform,
                                      0, NULL);
    wgpuRenderPassEncoderDraw(rpass_enc, 4, get_cell_count(), 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
  }

  /* Get command buffer */
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

  if (state.buffers.buffer0.data) {
    free(state.buffers.buffer0.data);
    state.buffers.buffer0.data = NULL;
  }

  wgpu_destroy_buffer(&state.buffers.square.handle);
  wgpu_destroy_buffer(&state.buffers.size.handle);
  wgpu_destroy_buffer(&state.buffers.buffer0.handle);
  wgpu_destroy_buffer(&state.buffers.buffer1.handle);

  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.uniform)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.bind_group0)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.bind_group1)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipeline)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Conway's Game of Life",
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
static const char* compute_shader_wgsl = CODE(
  @binding(0) @group(0) var<storage, read> size: vec2<u32>;
  @binding(1) @group(0) var<storage, read> current: array<u32>;
  @binding(2) @group(0) var<storage, read_write> next: array<u32>;

  override blockSize = 8;

  fn getIndex(x: u32, y: u32) -> u32 {
    let h = size.y;
    let w = size.x;

    return (y % h) * w + (x % w);
  }

  fn getCell(x: u32, y: u32) -> u32 {
    return current[getIndex(x, y)];
  }

  fn countNeighbors(x: u32, y: u32) -> u32 {
    return getCell(x - 1, y - 1) + getCell(x, y - 1) + getCell(x + 1, y - 1) +
           getCell(x - 1, y) +                         getCell(x + 1, y) +
           getCell(x - 1, y + 1) + getCell(x, y + 1) + getCell(x + 1, y + 1);
  }

  @compute @workgroup_size(blockSize, blockSize)
  fn main(@builtin(global_invocation_id) grid: vec3<u32>) {
    let x = grid.x;
    let y = grid.y;
    let n = countNeighbors(x, y);
    next[getIndex(x, y)] = select(u32(n == 3u), u32(n == 2u || n == 3u), getCell(x, y) == 1u);
 }
);

static const char* fragment_shader_wgsl = CODE(
  @fragment
  fn main(@location(0) cell: f32) -> @location(0) vec4<f32> {
    return vec4<f32>(cell, cell, cell, 1.);
  }
);

static const char* vertex_shader_wgsl = CODE(
  struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) cell: f32,
  }

  @binding(0) @group(0) var<uniform> size: vec2<u32>;

  @vertex
  fn main(@builtin(instance_index) i: u32, @location(0) cell: u32, @location(1) pos: vec2<u32>) -> Out {
    let w = size.x;
    let h = size.y;
    let x = (f32(i % w + pos.x) / f32(w) - 0.5) * 2. * f32(w) / f32(max(w, h));
    let y = (f32((i - (i % w)) / w + pos.y) / f32(h) - 0.5) * 2. * f32(h) / f32(max(w, h));

    return Out(vec4<f32>(x, y, 0., 1.), f32(cell));
  }
);
// clang-format on
