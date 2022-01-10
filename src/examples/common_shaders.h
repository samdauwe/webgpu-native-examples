#ifndef COMMON_SHADERS_H
#define COMMON_SHADERS_H

// clang-format off
static const char* basic_vertex_shader_wgsl =
  "[[block]] struct Uniforms {\n"
  "  modelViewProjectionMatrix : mat4x4<f32>;\n"
  "};\n"
  "[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;\n"
  "\n"
  "struct VertexOutput {\n"
  "  [[builtin(position)]] Position : vec4<f32>;\n"
  "  [[location(0)]] fragUV : vec2<f32>;\n"
  "  [[location(1)]] fragPosition: vec4<f32>;\n"
  "};\n"
  "\n"
  "[[stage(vertex)]]\n"
  "fn main([[location(0)]] position : vec4<f32>,\n"
  "        [[location(1)]] uv : vec2<f32>) -> VertexOutput {\n"
  "  var output : VertexOutput;\n"
  "  output.Position = uniforms.modelViewProjectionMatrix * position;\n"
  "  output.fragUV = uv;\n"
  "  output.fragPosition = 0.5 * (position + vec4<f32>(1.0, 1.0, 1.0, 1.0));\n"
  "  return output;\n"
  "}";

static const char* vertex_position_color_fragment_shader_wgsl =
  "[[stage(fragment)]]\n"
  "fn main([[location(0)]] fragUV: vec2<f32>,\n"
  "        [[location(1)]] fragPosition: vec4<f32>) -> [[location(0)]] vec4<f32> {\n"
  "  return fragPosition;\n"
  "}";
// clang-format on

#endif /* COMMON_SHADERS_H */
