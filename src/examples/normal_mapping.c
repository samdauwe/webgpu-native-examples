#include "example_base.h"

#include <cJSON.h>
#include <string.h>

#include "../core/log.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Normal Mapping example
 *
 * This example demonstrates how to achieve normal mapping in WebGPU. A normal
 * map uses RGB information that corresponds directly with the X, Y and Z axis
 * in 3D space. This RGB information tells the 3D application the exact
 * direction of the surface normals are oriented in for each and every polygon.
 *
 * Ref:
 * https://github.com/Konstantin84UKR/webgpu_examples/tree/master/normalMap
 *
 * Note:
 * http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */
static const char* normal_map_shadow_vertex_shader_wgsl;
static const char* normal_map_vertex_shader_wgsl;
static const char* normal_map_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* normal_map_shadow_vertex_shader_wgsl = CODE(
  struct Uniform {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };

  @group(0) @binding(0) var<uniform> uniforms : Uniform;

  @vertex
  fn main(@location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>
  ) -> @builtin(position) vec4<f32> {
    return uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
  }
);

static const char* normal_map_vertex_shader_wgsl = CODE(
  struct Uniform {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @group(0) @binding(0) var<uniform> uniforms : Uniform;

  struct UniformLight {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @group(0) @binding(4) var<uniform> uniformsLight : UniformLight;

  struct Output {
     @builtin(position) Position : vec4<f32>,
     @location(0) fragPosition : vec3<f32>,
     @location(1) fragUV : vec2<f32>,
     // @location(2) fragNormal : vec3<f32>,
     @location(3) shadowPos : vec3<f32>,
     @location(4) fragNor : vec3<f32>,
     @location(5) fragTangent : vec3<f32>,
     @location(6) fragBitangent : vec3<f32>
  };


  @vertex
  fn main(@location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>
  ) -> Output {
    var output: Output;
    output.Position = uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
    output.fragPosition = (uniforms.mMatrix * pos).xyz;
    output.fragUV = uv;
    //output.fragNormal  = (uniforms.mMatrix * vec4<f32>(normal,1.0)).xyz;

    // -----NORMAL --------------------------------

    var nMatrix : mat4x4<f32> = uniforms.mMatrix;
    nMatrix[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    let norm : vec3<f32>  = normalize((nMatrix * vec4<f32>(normal,1.0)).xyz);
    let tang : vec3<f32> = normalize((nMatrix * vec4<f32>(tangent,1.0)).xyz);
    let binormal : vec3<f32> = normalize((nMatrix * vec4<f32>(bitangent,1.0)).xyz);

    output.fragNor  = norm;
    output.fragTangent  = tang;
    output.fragBitangent  = binormal;


    let posFromLight: vec4<f32> = uniformsLight.pMatrix * uniformsLight.vMatrix * uniformsLight.mMatrix * pos;
    // Convert shadowPos XY to (0, 1) to fit texture UV
    output.shadowPos = vec3<f32>(posFromLight.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5), posFromLight.z);

    return output;
   }
);

static const char* normal_map_fragment_shader_wgsl = CODE(
  @binding(1) @group(0) var textureSampler : sampler;
  @binding(2) @group(0) var textureData : texture_2d<f32>;
  @binding(5) @group(0) var textureDataNormal : texture_2d<f32>;
  @binding(6) @group(0) var textureDataSpecular : texture_2d<f32>;

  struct Uniforms {
    eyePosition : vec4<f32>,
    lightPosition : vec4<f32>,
  };
  @binding(3) @group(0) var<uniform> uniforms : Uniforms;

  @binding(0) @group(1) var shadowMap : texture_depth_2d;
  @binding(1) @group(1) var shadowSampler : sampler_comparison;
  @binding(2) @group(1) var<uniform> test : vec3<f32>;


  @fragment
  fn main(@location(0) fragPosition: vec3<f32>,
    @location(1) fragUV: vec2<f32>,
    //@location(2) fragNormal: vec3<f32>,
    @location(3) shadowPos: vec3<f32>,
    @location(4) fragNor: vec3<f32>,
    @location(5) fragTangent: vec3<f32>,
    @location(6) fragBitangent: vec3<f32>
  ) -> @location(0) vec4<f32> {
    let specularColor: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    let i = 4.0f;
    let textureColor: vec3<f32> = (textureSample(textureData, textureSampler, fragUV * i)).rgb;
    let texturSpecular: vec3<f32> = (textureSample(textureDataSpecular, textureSampler, fragUV * i)).rgb;

    var textureNormal: vec3<f32> = normalize(2.0 * (textureSample(textureDataNormal, textureSampler, fragUV * i)).rgb - 1.0);
    var colorNormal = normalize(vec3<f32>(textureNormal.x, textureNormal.y, textureNormal.z));
    colorNormal.y *= -1;

    var tbnMatrix : mat3x3<f32> = mat3x3<f32>(
      normalize(fragTangent),
      normalize(fragBitangent),
      normalize(fragNor)
    );

    colorNormal = normalize(tbnMatrix * colorNormal);

    var shadow : f32 = 0.0;
    // apply Percentage-closer filtering (PCF)
    // sample nearest 9 texels to smooth result
    let size = f32(textureDimensions(shadowMap).x);
    for (var y : i32 = -1 ; y <= 1 ; y = y + 1) {
        for (var x : i32 = -1 ; x <= 1 ; x = x + 1) {
            let offset = vec2<f32>(f32(x) / size, f32(y) / size);
            shadow = shadow + textureSampleCompare(
                shadowMap,
                shadowSampler,
                shadowPos.xy + offset,
                shadowPos.z - 0.005  // apply a small bias to avoid acne
            );
        }
    }
    shadow = shadow / 9.0;

    let N: vec3<f32> = normalize(colorNormal.xyz);
    let L: vec3<f32> = normalize((uniforms.lightPosition).xyz - fragPosition.xyz);
    let V: vec3<f32> = normalize((uniforms.eyePosition).xyz - fragPosition.xyz);
    let H: vec3<f32> = normalize(L + V);

    let diffuse: f32 = 0.8 * max(dot(N, L), 0.0);
    let specular = pow(max(dot(N, H),0.0),100.0);
    let ambient: vec3<f32> = vec3<f32>(test.x + 0.2, 0.4, 0.5);

    let finalColor: vec3<f32> =  textureColor * ( shadow * diffuse + ambient) + (texturSpecular * specular * shadow);
    //let finalColor:vec3<f32> =  colorNormal * 0.5 + 0.5;  //let color = N * 0.5 + 0.5;
    //let finalColor:vec3<f32> =  texturSpecular ;  //let color = N * 0.5 + 0.5;

    return vec4<f32>(finalColor, 1.0);
);
// clang-format on
