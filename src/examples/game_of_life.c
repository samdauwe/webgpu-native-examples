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
