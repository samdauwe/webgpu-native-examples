#include "example_base.h"
#include "meshes.h"

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

static struct {
  uint32_t width;
  uint32_t height;
  uint32_t timestep;
  uint32_t workgroup_size;
} game_options = {
  .width          = 128,
  .height         = 128,
  .timestep       = 4,
  .workgroup_size = 8,
};

static struct {
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
} buffers = {0};

static struct {
  WGPUBindGroup uniform;
  WGPUBindGroup bind_group0;
  WGPUBindGroup bind_group1;
} bind_groups = {0};

// Resources for the compute part of the example
static struct {
  WGPUBindGroupLayout bind_group_layout;
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline pipeline;
} compute = {0};

// Resources for the graphics part of the example
static struct {
  WGPUBindGroupLayout bind_group_layout;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
} graphics = {0};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Used during render step
static uint32_t whole_time = 0, loop_times = 0;

// Other variables
static const char* example_title = "Conway's Game of Life";
static bool prepared             = false;

static uint32_t get_cell_count(void)
{
  return game_options.width * game_options.height;
}

static void update_size_uniform_buffer(wgpu_context_t* wgpu_context)
{
  // Update the buffer data
  buffers.size.data[0] = game_options.width;
  buffers.size.data[1] = game_options.height;

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(wgpu_context, buffers.size.handle.buffer, 0,
                          buffers.size.data, buffers.size.handle.size);
}

static void prepare_static_buffers(wgpu_context_t* wgpu_context)
{
  /* Square vertex buffer */
  {
    // Setup vertices of the square
    uint32_t square_vertices[8] = {0, 0, 0, 1, 1, 0, 1, 1};

    // Create the vertex buffer
    buffers.square.handle = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Square - Vertex buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = sizeof(square_vertices),
                      .initial.data = square_vertices,
                    });
  }

  /* Size uniform buffer */
  {
    buffers.size.handle = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Size - Uniform buffer",
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Uniform
                 | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
        .size = sizeof(buffers.size.data),
      });

    update_size_uniform_buffer(wgpu_context);
  }
}

static float float_random(float min, float max)
{
  const float scale = rand() / (float)RAND_MAX; /* [0, 1.0]   */
  return min + scale * (max - min);             /* [min, max] */
}

static void prepare_storage_buffers(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_buffer(&buffers.buffer0.handle);
  wgpu_destroy_buffer(&buffers.buffer1.handle);

  if (buffers.buffer0.data) {
    free(buffers.buffer0.data);
    buffers.buffer0.data = NULL;
  }

  /* Update the buffer data */
  const uint32_t cell_count = get_cell_count();
  const uint32_t length     = cell_count * sizeof(uint32_t);
  buffers.buffer0.data      = (uint32_t*)malloc(length);
  for (uint32_t i = 0; i < cell_count; ++i) {
    buffers.buffer0.data[i] = float_random(0.0f, 1.0f) < 0.25f ? 1 : 0;
  }

  /* Storage buffer 0 */
  {
    buffers.buffer0.handle = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Storage buffer 0",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage
                               | WGPUBufferUsage_Vertex,
                      .size         = length,
                      .initial.data = buffers.buffer0.data,
                    });
    ASSERT(buffers.buffer0.handle.buffer != NULL);
  }

  /* Storage buffer 1 */
  {
    buffers.buffer1.handle = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Storage buffer 1",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage
                               | WGPUBufferUsage_Vertex,
                      .size = length,
                    });
    ASSERT(buffers.buffer1.handle.buffer != NULL);
  }
}

static void setup_bind_group_layout_compute(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0: Storage buffer (size) */
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize   = buffers.size.handle.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Storage buffer (current) */
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize   = buffers.buffer0.handle.size,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2: Storage buffer (next) */
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Storage,
        .minBindingSize   = buffers.buffer1.handle.size,
      },
      .sampler = {0},
    }
  };
  compute.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Compute - Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(compute.bind_group_layout != NULL);
}

static void setup_bind_groups_compute(wgpu_context_t* wgpu_context)
{
  /* Bind group 0 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = buffers.size.handle.buffer,
        .size    = buffers.size.handle.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = buffers.buffer0.handle.buffer,
        .size    = buffers.buffer0.handle.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = buffers.buffer1.handle.buffer,
        .size    = buffers.buffer1.handle.size,
      },
    };
    bind_groups.bind_group0 = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Bind group - Compute 0",
                              .layout     = compute.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.bind_group0 != NULL);
  }

  /* Bind group 1 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = buffers.size.handle.buffer,
        .size    = buffers.size.handle.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = buffers.buffer1.handle.buffer,
        .size    = buffers.buffer1.handle.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = buffers.buffer0.handle.buffer,
        .size    = buffers.buffer0.handle.size,
      },
    };
    bind_groups.bind_group1 = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Bind group - Compute 1",
                              .layout     = compute.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.bind_group1 != NULL);
  }
}

static void prepare_pipeline_layout_compute(wgpu_context_t* wgpu_context)
{
  compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Pipeline layout - Compute",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &compute.bind_group_layout,
                          });
  ASSERT(compute.pipeline_layout != NULL);
}

static void prepare_pipeline_compute(wgpu_context_t* wgpu_context)
{
  /* Compute shader constants */
  WGPUConstantEntry constant_entries[1] = {
    [0] = (WGPUConstantEntry){
      .key   = "blockSize",
      .value = game_options.workgroup_size,
    },
  };

  /* Compute shader */
  wgpu_shader_t compute_shader = wgpu_shader_create(
    wgpu_context,
    &(wgpu_shader_desc_t){
      /* Compute shader WGSL */
      .label            = "Compute shader WGSL",
      .wgsl_code.source = compute_shader_wgsl,
      .entry            = "main",
      .constants        = {
        .count   = (uint32_t)ARRAY_SIZE(constant_entries),
        .entries = constant_entries,
      },
    });

  /* Compute pipeline */
  compute.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = "Compute pipeline",
      .layout  = compute.pipeline_layout,
      .compute = compute_shader.programmable_stage_descriptor,
    });
  ASSERT(compute.pipeline != NULL);

  /* Partial clean-up */
  wgpu_shader_release(&compute_shader);
}

static void prepare_bind_group_layout_graphics(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (size)
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .minBindingSize   = buffers.size.handle.size,
      },
      .sampler = {0},
    },
  };
  graphics.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Graphics - Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(graphics.bind_group_layout != NULL);
}

static void setup_bind_group_graphics(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = buffers.size.handle.buffer,
      .size    = buffers.size.handle.size,
    },
  };
  bind_groups.uniform = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "Uniform - Bind group",
                            .layout     = graphics.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(bind_groups.uniform != NULL);
}

static void prepare_pipeline_layout_graphics(wgpu_context_t* wgpu_context)
{
  graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = "Graphics - Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &graphics.bind_group_layout,
                          });
  ASSERT(graphics.pipeline_layout != NULL);
}

static void prepare_pipeline_graphics(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleStrip,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

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

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "Vertex shader WGSL",
                      .wgsl_code.source = vertex_shader_wgsl,
                      .entry            = "main",
                    },
                    .buffer_count = (uint32_t)ARRAY_SIZE(vertex_state_buffers),
                    .buffers      = vertex_state_buffers,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "Fragment shader WGSL",
                      .wgsl_code.source = fragment_shader_wgsl,
                      .entry            = "main",
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
  graphics.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Graphics - Render pipeline",
                            .layout      = graphics.pipeline_layout,
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(graphics.pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

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

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                = "Render pass descriptor",
    .colorAttachmentCount = (uint32_t)ARRAY_SIZE(render_pass.color_attachments),
    .colorAttachments     = render_pass.color_attachments,
  };
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Compute pass */
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                      compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(
      wgpu_context->cpass_enc, 0,
      loop_times ? bind_groups.bind_group1 : bind_groups.bind_group0, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      wgpu_context->cpass_enc,
      ceil((float)game_options.width / game_options.workgroup_size),
      ceil((float)game_options.height / game_options.workgroup_size), 1);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  /* Graphics render pipeline */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     graphics.pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         loop_times ?
                                           buffers.buffer1.handle.buffer :
                                           buffers.buffer0.handle.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                         buffers.square.handle.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.uniform, 0, NULL);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 4, get_cell_count(), 0,
                              0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

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

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_static_buffers(context->wgpu_context);
    prepare_storage_buffers(context->wgpu_context);
    /* Compute */
    setup_bind_group_layout_compute(context->wgpu_context);
    setup_bind_groups_compute(context->wgpu_context);
    prepare_pipeline_layout_compute(context->wgpu_context);
    prepare_pipeline_compute(context->wgpu_context);
    /* Graphics */
    prepare_bind_group_layout_graphics(context->wgpu_context);
    setup_bind_group_graphics(context->wgpu_context);
    prepare_pipeline_layout_graphics(context->wgpu_context);
    prepare_pipeline_graphics(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }

  if (game_options.timestep) {
    whole_time++;
    if (whole_time >= game_options.timestep) {
      example_draw(context);
      whole_time -= game_options.timestep;
      loop_times = 1 - loop_times;
    }
  }

  return EXIT_SUCCESS;
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  if (buffers.buffer0.data) {
    free(buffers.buffer0.data);
    buffers.buffer0.data = NULL;
  }

  wgpu_destroy_buffer(&buffers.square.handle);
  wgpu_destroy_buffer(&buffers.size.handle);
  wgpu_destroy_buffer(&buffers.buffer0.handle);
  wgpu_destroy_buffer(&buffers.buffer1.handle);

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.uniform)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.bind_group0)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.bind_group1)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, compute.pipeline)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, graphics.pipeline)
}

void example_game_of_life(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
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
