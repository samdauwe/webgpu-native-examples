#ifndef COMMON_SHADERS_H
#define COMMON_SHADERS_H

#include "../core/macro.h"

// clang-format off
static const char* basic_vertex_shader_wgsl = CODE(
  struct Uniforms {
    modelViewProjectionMatrix : mat4x4<f32>,
  }
  @binding(0) @group(0) var<uniform> uniforms : Uniforms;

  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragUV : vec2<f32>,
    @location(1) fragPosition: vec4<f32>,
  }

  @vertex
  fn main(
    @location(0) position : vec4<f32>,
    @location(1) uv : vec2<f32>
  ) -> VertexOutput {
    var output : VertexOutput;
    output.Position = uniforms.modelViewProjectionMatrix * position;
    output.fragUV = uv;
    output.fragPosition = 0.5 * (position + vec4<f32>(1.0, 1.0, 1.0, 1.0));
    return output;
  }
);

static const char* vertex_position_color_fragment_shader_wgsl =
  "@stage(fragment)\n"
  "fn main(@location(0) fragUV: vec2<f32>,\n"
  "        @location(1) fragPosition: vec4<f32>) -> @location(0) vec4<f32> {\n"
  "  return fragPosition;\n"
  "}";

static const char* fullscreen_textured_quad_wgsl =
  "@group(0) @binding(0) var mySampler : sampler;\n"
  "@group(0) @binding(1) var myTexture : texture_2d<f32>;\n"
  "\n"
  "struct VertexOutput {\n"
  "  @builtin(position) Position : vec4<f32>,\n"
  "  @location(0) fragUV : vec2<f32>\n"
  "};\n"
  "\n"
  "@stage(vertex)\n"
  "fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {\n"
  "  var pos = array<vec2<f32>, 6>(\n"
  "      vec2<f32>( 1.0,  1.0),\n"
  "      vec2<f32>( 1.0, -1.0),\n"
  "      vec2<f32>(-1.0, -1.0),\n"
  "      vec2<f32>( 1.0,  1.0),\n"
  "      vec2<f32>(-1.0, -1.0),\n"
  "      vec2<f32>(-1.0,  1.0));\n"
  "\n"
  "  var uv = array<vec2<f32>, 6>(\n"
  "      vec2<f32>(1.0, 0.0),\n"
  "      vec2<f32>(1.0, 1.0),\n"
  "      vec2<f32>(0.0, 1.0),\n"
  "      vec2<f32>(1.0, 0.0),\n"
  "      vec2<f32>(0.0, 1.0),\n"
  "      vec2<f32>(0.0, 0.0));\n"
  "\n"
  "  var output : VertexOutput;\n"
  "  output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);\n"
  "  output.fragUV = uv[VertexIndex];\n"
  "  return output;\n"
  "}\n"
  "\n"
  "@stage(fragment)\n"
  "fn frag_main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {\n"
  "  return textureSample(myTexture, mySampler, fragUV);\n"
  "}";
// clang-format on

#endif /* COMMON_SHADERS_H */
