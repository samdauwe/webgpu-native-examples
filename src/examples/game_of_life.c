#include "example_base.h"
#include "meshes.h"

#include "../webgpu/imgui_overlay.h"

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
  wgpu_buffer_t vertex_buffer;
} square = {0};

// Resources for the compute part of the example
static struct {
  WGPUBindGroupLayout bind_group_layout;
  WGPUComputePipeline pipelines;
} compute;

// Resources for the graphics part of the example
static struct {
  WGPUBindGroupLayout bind_group_layout;
  WGPUComputePipeline pipelines;
} graphics;

// Other variables
static const char* example_title = "Conway's Game of Life";
static bool prepared             = false;

static void generate_square(wgpu_context_t* wgpu_context)
{
  // Setup vertices of the square
  uint32_t square_vertices[8] = {0, 0, 0, 1, 1, 0, 1, 1};

  // Create the Vertex buffer
  square.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Square vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(square_vertices),
                    .initial.data = square_vertices,
                  });
}

static void setup_compute_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Storage buffer (size)
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize   = sizeof(ubo_vs),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Storage buffer (current)
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize   = sizeof(ubo_vs),
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      // Binding 2: Storage buffer (next)
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Storage,
        .minBindingSize   = sizeof(ubo_vs),
      },
      .sampler = {0},
    }
  };
  compute.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Compute bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(compute.bind_group_layout != NULL);
}

static void prepare_graphics_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (size)
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .minBindingSize   = sizeof(ubo_vs),
      },
      .sampler = {0},
    },
  };
  graphics.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Graphics bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(graphics.bind_group_layout != NULL);
}

static void prepare_graphics_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleStrip,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Vertex buffer layouts
  WGPU_VERTEX_BUFFER_LAYOUT(cell_stride, sizeof(uint32_t),
                            // Attribute location 0: Cell
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Uint32, 0))
  WGPU_VERTEX_BUFFER_LAYOUT(
    square_stride, 2 * sizeof(uint32_t),
    // Attribute location 1: Position
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 0))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "Vertex shader WGSL",
                      .wgsl_code.source = vertex_shader_wgsl,
                      .entry            = "main",
                    },
                    .buffer_count = 1,
                    .buffers      = &square_stride_vertex_buffer_layout,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "Fragment shader WGSL",
                      .wgsl_code.source = fragment_shader_wgsl,
                      .entry            = "main",
                    },
                    .target_count = 1,
                    .targets      = &color_target_state,
                  });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
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
