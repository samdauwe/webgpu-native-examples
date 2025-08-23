#ifndef COMMON_SHADERS_H
#define COMMON_SHADERS_H

#include "webgpu/wgpu_common.h"

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
    output.fragPosition = 0.5 * (position + vec4(1.0, 1.0, 1.0, 1.0));
    return output;
  }
);

static const char* vertex_position_color_fragment_shader_wgsl = CODE(
  @fragment
  fn main(
    @location(0) fragUV: vec2<f32>,
    @location(1) fragPosition: vec4<f32>
  ) -> @location(0) vec4<f32> {
    return fragPosition;
  }
);

static const char* fullscreen_textured_quad_wgsl = CODE(
  @group(0) @binding(0) var mySampler : sampler;
  @group(0) @binding(1) var myTexture : texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragUV : vec2<f32>,
  }

  @vertex
  fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
    const pos = array(
      vec2( 1.0,  1.0),
      vec2( 1.0, -1.0),
      vec2(-1.0, -1.0),
      vec2( 1.0,  1.0),
      vec2(-1.0, -1.0),
      vec2(-1.0,  1.0),
    );

    const uv = array(
      vec2(1.0, 0.0),
      vec2(1.0, 1.0),
      vec2(0.0, 1.0),
      vec2(1.0, 0.0),
      vec2(0.0, 1.0),
      vec2(0.0, 0.0),
    );

    var output : VertexOutput;
    output.Position = vec4(pos[VertexIndex], 0.0, 1.0);
    output.fragUV = uv[VertexIndex];
    return output;
  }

  @fragment
  fn frag_main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(myTexture, mySampler, fragUV);
  }
);
// clang-format on

#endif /* COMMON_SHADERS_H */
