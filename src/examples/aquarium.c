#include "example_base.h"

#include <limits.h>
#include <string.h>

#include <cJSON.h>
#include <sc_array.h>
#include <sc_queue.h>

#ifdef __linux__
#include <unistd.h>
static const char* slash = "/";
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Aquarium
 *
 * Aquarium is a native implementation of WebGL Aquarium.
 *
 * Ref:
 * https://github.com/webatintel/aquarium
 * https://webglsamples.org/aquarium/aquarium.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Aquarium Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* diffuse_vertex_shader_wgsl = CODE(
  struct LightWorldPositionUniform {
    lightWorldPos : vec3<f32>,
    viewProjection : mat4x4<f32>,
    viewInverse : mat4x4<f32>,
  }

  struct WorldUniform {
    world : mat4x4<f32>,
    worldInverseTranspose : mat4x4<f32>,
    worldViewProjection : mat4x4<f32>,
  }

  struct WorldUniforms {
    worlds : array<WorldUniform, 20>,
  }

  @group(1) @binding(0) var<uniform> lightWorldPositionUniform : LightWorldPositionUniform;
  @group(3) @binding(0) var<uniform> worldUniforms : WorldUniforms;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @builtin(instance_index) instanceIndex: u32,
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_normal : vec3<f32>,
    @location(3) v_surfaceToLight : vec3<f32>,
    @location(4) v_surfaceToView : vec3<f32>,
  }

  @vertex
  fn main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>
  ) -> Output {
    var output: Output;
    output.v_texCoord = texCoord;
    output.v_position = (worldUniforms.worlds[instanceIndex].worldViewProjection * position);
    output.v_normal = (worldUniforms.worlds[instanceIndex].worldInverseTranspose * vec4<f32>(normal, 0)).xyz;
    output.v_surfaceToLight = lightWorldPositionUniform.lightWorldPos - (worldUniforms.worlds[instanceIndex].world * position).xyz;
    output.v_surfaceToView = (lightWorldPositionUniform.viewInverse[3] - (worldUniforms.worlds[instanceIndex].world * position)).xyz;
    output.position = output.v_position;
    return output;
  }
);

static const char* diffuse_fragment_shader_wgsl = CODE(
  struct LightUniforms {
    lightColor : vec4<f32>,
    specular : vec4<f32>,
    ambient : vec4<f32>,
  }

  struct Fogs {
    fogPower : f32,
    fogMult : f32,
    fogOffset : f32,
    fogColor : vec4<f32>,
  }

  struct LightFactorUniforms {
    shininess : f32,
    specularFactor : f32,
  }

  @group(0) @binding(0) var<uniform> lightUniforms : LightUniforms;
  @group(0) @binding(1) var<uniform> fogs : Fogs;
  @group(2) @binding(0) var<uniform> lightFactorUniforms : LightFactorUniforms;
  @group(2) @binding(1) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(2) var diffuseTextureSampler: sampler;

  fn lit(l : f32 , h : f32, m : f32) -> vec4f {
    return vec4<f32>(1.0,
                     max(l, 0.0),
                     select(0.0, pow(0.0, max(0.0, h), m), l > 0.0),
                     1.0);
  }

  @fragment
  fn main(
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_normal : vec3<f32>,
    @location(3) v_surfaceToLight : vec3<f32>,
    @location(4) v_surfaceToView : vec3<f32>
  ) -> @location(0) vec4<f32> {
    let diffuseColor : vec4<f32> = textureSample(diffuseTexture, diffuseTextureSampler, v_texCoord);
    let normal : vec3<f32> = normalize(v_normal);
    let surfaceToLight : vec3<f32> = normalize(v_surfaceToLight);
    let surfaceToView : vec3<f32> = normalize(v_surfaceToView);
    let halfVector : vec3<f32> = normalize(surfaceToLight + surfaceToView);
    let litR : vec4<f32> = lit(dot(normal, surfaceToLight),
                               dot(normal, halfVector), lightFactorUniforms.shininess);
    var outColor : vec4<f32> = vec4<f32>((
      lightUniforms.lightColor * (diffuseColor * litR.y + diffuseColor * lightUniforms.ambient +
                    lightUniforms.specular * litR.z * lightFactorUniforms.specularFactor)).rgb,
            diffuseColor.a);
    outColor = mix(outColor, vec4(fogs.fogColor.rgb, diffuseColor.a),
      clamp(pow((v_position.z / v_position.w), 0) * 0 - 0,0.0,1.0));
    return outColor;
  }
);

static const char* fish_vertex_shader_wgsl = CODE(
  struct LightWorldPositionUniform {
    lightWorldPos : vec3<f32>,
    viewProjection : mat4x4<f32>,
    viewInverse : mat4x4<f32>,
  }

  struct FishVertexUniforms {
    fishLength : f32,
    fishWaveLength : f32,
    fishBendAmount : f32,
  }

  struct FishPer {
    worldPosition : vec3<f32>,
    scale : f32,
    nextPosition : vec3<f32>,
    time : f32,
  }

  @group(1) @binding(0) var<uniform> lightWorldPositionUniform : LightWorldPositionUniform;
  @group(2) @binding(0) var<uniform> fishVertexUnifoms : FishVertexUniforms;
  @group(3) @binding(0) var<uniform> fishPer : FishPer;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>, // #normalMap
    @location(3) v_binormal : vec3<f32>, // #normalMap
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>,
  }

  @vertex
  fn main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>
  ) -> Output {
    var output: Output;
    let vz: vec3<f32> = normalize(fishPer.worldPosition - fishPer.nextPosition);
    let vx: vec3<f32> = normalize(cross(vec3<f32>(0,1,0), vz));
    let vy: vec3<f32> = cross(vz, vx);
    let orientMat : mat4x4<f32> = mat4x4<f32>(
      vec4<f32>(vx, 0),
      vec4<f32>(vy, 0),
      vec4<f32>(vz, 0),
      vec4<f32>(fishPer.worldPosition, 1));
    let scaleMat : mat4x4<f32> = mat4x4<f32>(
      vec4<f32>(fishPer.scale, 0, 0, 0),
      vec4<f32>(0, fishPer.scale, 0, 0),
      vec4<f32>(0, 0, fishPer.scale, 0),
      vec4<f32>(0, 0, 0, 1));
    let world : mat4x4<f32> = orientMat * scaleMat;
    let worldViewProjection : mat4x4<f32> = lightWorldPositionUniform.viewProjection * world;
    let worldInverseTranspose : mat4x4<f32> = world;

    output.v_texCoord = texCoord;
    // NOTE:If you change this you need to change the laser code to match!
    let mult : f32 = select(
      (-position.z / fishVertexUnifoms.fishLength * 2.0),
      (position.z / fishVertexUnifoms.fishLength),
      position.z > 0.0
    );
    let s : f32 = sin(fishPer.time + mult * fishVertexUnifoms.fishWaveLength);
    let offset : f32 = pow(mult, 2.0) * s * fishVertexUnifoms.fishBendAmount;
    output.v_position = (worldViewProjection * (position + vec4<f32>(offset, 0, 0, 0)));
    output.v_normal = (worldInverseTranspose * vec4<f32>(normal, 0)).xyz;
    output.v_surfaceToLight = lightWorldPositionUniform.lightWorldPos - (world * position).xyz;
    output.v_surfaceToView = (lightWorldPositionUniform.viewInverse[3] - (world * position)).xyz;
    output.v_binormal = (worldInverseTranspose * vec4<f32>(binormal, 0)).xyz;  // #normalMap
    output.v_tangent = (worldInverseTranspose * vec4<f32>(tangent, 0)).xyz;  // #normalMap
    output.position = output.v_position;
    return output;
  }
);

static const char* fish_instanced_draws_vertex_shader_wgsl = CODE(
  struct LightWorldPositionUniform {
    lightWorldPos : vec3<f32>,
    viewProjection : mat4x4<f32>,
    viewInverse : mat4x4<f32>,
  }

  struct FishVertexUniforms {
    fishLength : f32,
    fishWaveLength : f32,
    fishBendAmount : f32,
  }

  @group(1) @binding(0) var<uniform> lightWorldPositionUniform : LightWorldPositionUniform;
  @group(2) @binding(0) var<uniform> fishVertexUnifoms : FishVertexUniforms;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>, // #normalMap
    @location(3) v_binormal : vec3<f32>, // #normalMap
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>,
  }

  @vertex
  fn main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>,
    @location(5) worldPosition: vec3<f32>,
    @location(6) scale: f32,
    @location(7) nextPosition: vec3<f32>,
    @location(8) time: f32,
  ) -> Output {
    var output: Output;
    let vz: vec3<f32> = normalize(worldPosition - nextPosition);
    let vx: vec3<f32> = normalize(cross(vec3<f32>(0,1,0), vz));
    let vy: vec3<f32> = cross(vz, vx);
    let orientMat : mat4x4<f32> = mat4x4<f32>(
      vec4<f32>(vx, 0),
      vec4<f32>(vy, 0),
      vec4<f32>(vz, 0),
      vec4<f32>(worldPosition, 1));
    let scaleMat : mat4x4<f32> = mat4x4<f32>(
      vec4<f32>(scale, 0, 0, 0),
      vec4<f32>(0, scale, 0, 0),
      vec4<f32>(0, 0, scale, 0),
      vec4<f32>(0, 0, 0, 1));
    let world : mat4x4<f32> = orientMat * scaleMat;
    let worldViewProjection : mat4x4<f32> = lightWorldPositionUniform.viewProjection * world;
    let worldInverseTranspose : mat4x4<f32> = world;

    output.v_texCoord = texCoord;
    // NOTE:If you change this you need to change the laser code to match!
    let mult : f32 = select(
      (-position.z / fishVertexUnifoms.fishLength * 2.0),
      (position.z / fishVertexUnifoms.fishLength),
      position.z > 0.0
    );
    let s : f32 = sin(time + mult * fishVertexUnifoms.fishWaveLength);
    let offset : f32 = pow(mult, 2.0) * s * fishVertexUnifoms.fishBendAmount;
    output.v_position = (worldViewProjection * (position + vec4<f32>(offset, 0, 0, 0)));
    output.v_normal = (worldInverseTranspose * vec4<f32>(normal, 0)).xyz;
    output.v_surfaceToLight = lightWorldPositionUniform.lightWorldPos - (world * position).xyz;
    output.v_surfaceToView = (lightWorldPositionUniform.viewInverse[3] - (world * position)).xyz;
    output.v_binormal = (worldInverseTranspose * vec4<f32>(binormal, 0)).xyz;  // #normalMap
    output.v_tangent = (worldInverseTranspose * vec4<f32>(tangent, 0)).xyz;  // #normalMap
    output.v_position.y = -output.v_position.y;
    output.position = output.v_position;
    return output;
  }
);

static const char* fish_normal_map_fragment_shader_wgsl = CODE(
  struct LightUniforms {
    lightColor : vec4<f32>,
    specular : vec4<f32>,
    ambient : vec4<f32>,
  }

  struct LightFactorUniforms {
    shininess : f32,
    specularFactor : f32,
  }

  struct Fogs {
    fogPower : f32,
    fogMult : f32,
    fogOffset : f32,
    fogColor : vec4<f32>,
  }

  @group(0) @binding(0) var<uniform> lightUniforms : LightUniforms;
  @group(0) @binding(1) var<uniform> fogs : Fogs;
  @group(2) @binding(1) var<uniform> lightFactorUniforms : LightFactorUniforms;
  @group(2) @binding(2) var samplerTex2D: sampler;
  @group(2) @binding(3) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(4) var normalMapTexture: texture_2d<f32>; // #normalMap

  fn lit(l : f32 , h : f32, m : f32) -> vec4f {
    return vec4<f32>(1.0,
                     max(l, 0.0),
                     select(0.0, pow(0.0, max(0.0, h), m), l > 0.0),
                     1.0);
  }

  @fragment
  fn main(
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>, // #normalMap
    @location(3) v_binormal : vec3<f32>, // #normalMap
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>
  ) -> @location(0) vec4<f32> {
    let diffuseColor : vec4<f32> = textureSample(diffuseTexture, samplerTex2D, v_texCoord);
    let tangentToWorld : mat3x3<f32> = mat3x3<f32>(v_tangent,  // #normalMap
                                                   v_binormal, // #normalMap
                                                   v_normal);  // #normalMap
    let normalSpec : vec4<f32> = textureSample(normalMapTexture, samplerTex2D, v_texCoord); // #normalMap
    let tangentNormal : vec3<f32> = normalSpec.xyz - vec3<f32>(0.5, 0.5, 0.5); // #normalMap
    tangentNormal = normalize(tangentNormal + vec3<f32>(0, 0, 2)); // #normalMap
    var normal : vec3<f32> = (tangentToWorld * tangentNormal); // #normalMap
    normal = normalize(normal); // #normalMap
    let surfaceToLight : vec3<f32> = normalize(v_surfaceToLight);
    let surfaceToView : vec3<f32> = normalize(v_surfaceToView);
    let halfVector : vec3<f32> = normalize(surfaceToLight + surfaceToView);
    let litR : vec4<f32> = lit(dot(normal, surfaceToLight),
                               dot(normal, halfVector), lightFactorUniforms.shininess);
    var outColor : vec4<f32> = vec4<f32>(
      (lightUniforms.lightColor * (diffuseColor * litR.y + diffuseColor * lightUniforms.ambient +
                     lightUniforms.specular * litR.z * lightFactorUniforms.specularFactor * normalSpec.a)).rgb,
            diffuseColor.a);
    outColor = mix(outColor, vec4(fogs.fogColor.rgb, diffuseColor.a),
      clamp(pow((v_position.z / v_position.w), fogs.fogPower) * fogs.fogMult - fogs.fogOffset,0.0,1.0));
    return outColor;
  }
);

static const char* fish_reflection_fragment_shader_wgsl = CODE(
  struct LightUniforms {
    lightColor : vec4<f32>,
    specular : vec4<f32>,
    ambient : vec4<f32>,
  }

  struct LightFactorUniforms {
    shininess : f32,
    specularFactor : f32,
  }

  struct Fogs {
    fogPower : f32,
    fogMult : f32,
    fogOffset : f32,
    fogColor : vec4<f32>,
  }

  @group(0) @binding(0) var<uniform> lightUniforms : LightUniforms;
  @group(0) @binding(1) var<uniform> fogs : Fogs;
  @group(2) @binding(1) var<uniform> lightFactorUniforms : LightFactorUniforms;
  @group(2) @binding(2) var samplerTex2D: sampler;
  @group(2) @binding(3) var samplerSkybox: sampler;
  @group(2) @binding(4) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(5) var normalMapTexture: texture_2d<f32>;
  @group(2) @binding(6) var reflectionMapTexture: texture_2d<f32>; // #reflection
  @group(2) @binding(7) var skyboxTexture: texture_cube<f32>; // #reflecton

  fn lit(l : f32 , h : f32, m : f32) -> vec4f {
    return vec4<f32>(1.0,
                     max(l, 0.0),
                     select(0.0, pow(0.0, max(0.0, h), m), l > 0.0),
                     1.0);
  }

  @fragment
  fn main(
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>, // #normalMap
    @location(3) v_binormal : vec3<f32>, // #normalMap
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>
  ) -> @location(0) vec4<f32> {
    let diffuseColor : vec4<f32> = textureSample(diffuseTexture, samplerTex2D, v_texCoord);
    let tangentToWorld : mat3x3<f32> = mat3x3<f32>(v_tangent,  // #normalMap
                                                   v_binormal, // #normalMap
                                                   v_normal);  // #normalMap
    let normalSpec : vec4<f32> = textureSample(normalMapTexture, samplerTex2D, v_texCoord); // #normalMap
    let reflection : vec4<f32> = textureSample(reflectionMapTexture, samplerTex2D, v_texCoord); // #reflection
    let tangentNormal : vec3<f32> = normalSpec.xyz - vec3<f32>(0.5, 0.5, 0.5); // #normalMap
    var normal : vec3<f32> = (tangentToWorld * tangentNormal); // #normalMap
    normal = normalize(normal); // #normalMap
    let surfaceToLight : vec3<f32> = normalize(v_surfaceToLight);
    let surfaceToView : vec3<f32> = normalize(v_surfaceToView);
    let skyColor : vec4<f32> = textureSample(skyboxTexture, samplerSkybox, -reflect(surfaceToView, normal)); // #reflection

    let halfVector : vec3<f32> = normalize(surfaceToLight + surfaceToView);
    let litR : vec4<f32> = lit(dot(normal, surfaceToLight),
                               dot(normal, halfVector), lightFactorUniforms.shininess);

    var outColor : vec4<f32> = vec4<f32>(mix(
      skyColor,
      lightUniforms.lightColor * (diffuseColor * litR.y + diffuseColor * lightUniforms.ambient +
                    lightUniforms.specular * litR.z * lightFactorUniforms.specularFactor * normalSpec.a),
      1.0 - reflection.r).rgb,
      diffuseColor.a);
    outColor = mix(outColor, vec4<f32>(fogs.fogColor.rgb, diffuseColor.a),
      clamp(pow((v_position.z / v_position.w), fogs.fogPower) * fogs.fogMult - fogs.fogOffset,0.0,1.0));
    return outColor;
  }
);

static const char* inner_refraction_map_vertex_shader_wgsl = CODE(
  struct LightWorldPositionUniform {
    lightWorldPos : vec3<f32>,
    viewProjection : mat4x4<f32>,
    viewInverse : mat4x4<f32>,
  }

  struct WorldUniforms {
    world : mat4x4<f32>,
    worldInverseTranspose : mat4x4<f32>,
    worldViewProjection : mat4x4<f32>,
  }

  @group(1) @binding(0) var<uniform> lightWorldPositionUniform : LightWorldPositionUniform;
  @group(3) @binding(0) var<uniform> worldUniforms : WorldUniforms;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>, // #normalMap
    @location(3) v_binormal : vec3<f32>, // #normalMap
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>,
  }

  @vertex
  fn main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>
  ) -> Output {
    var output: Output;
    output.v_texCoord = texCoord;
    output.v_position = (worldUniforms.worldViewProjection * position);
    output.v_normal = (worldUniforms.worldInverseTranspose * vec4<f32>(normal, 0)).xyz;
    output.v_surfaceToLight = lightWorldPositionUniform.lightWorldPos - (worldUniforms.world * position).xyz;
    output.v_surfaceToView = (lightWorldPositionUniform.viewInverse[3] - (worldUniforms.world * position)).xyz;
    output.v_binormal = (worldUniforms.worldInverseTranspose * vec4<f32>(binormal, 0)).xyz;  // #normalMap
    output.v_tangent = (worldUniforms.worldInverseTranspose * vec4<f32>(tangent, 0)).xyz;  // #normalMap
    output.position = output.v_position;
    return output;
  }
);

static const char* inner_refraction_map_fragment_shader_wgsl = CODE(
  struct LightUniforms {
    lightColor : vec4<f32>,
    specular : vec4<f32>,
    ambient : vec4<f32>,
  }

  struct Fogs {
    fogPower : f32,
    fogMult : f32,
    fogOffset : f32,
    padding : f32,
    fogColor : vec4<f32>,
  }

  struct InnerUniforms {
    eta : f32,
    tankColorFudge : f32,
    refractionFudge : f32,
    padding : f32,
  }

  @group(0) @binding(0) var<uniform> lightUniforms : LightUniforms;
  @group(0) @binding(1) var<uniform> fogs : Fogs;
  @group(2) @binding(0) var<uniform> innerUniforms : InnerUniforms;
  @group(2) @binding(1) var samplerTex2D: sampler;
  @group(2) @binding(2) var samplerSkybox: sampler;
  @group(2) @binding(3) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(4) var normalMapTexture: texture_2d<f32>;
  @group(2) @binding(5) var reflectionMapTexture: texture_2d<f32>; // #reflection
  @group(2) @binding(6) var skyboxTexture: texture_cube<f32>; // #reflecton

  fn lit(l : f32 , h : f32, m : f32) -> vec4f {
    return vec4<f32>(1.0,
                     max(l, 0.0),
                     select(0.0, pow(0.0, max(0.0, h), m), l > 0.0),
                     1.0);
  }

  @fragment
  fn main(
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>, // #normalMap
    @location(3) v_binormal : vec3<f32>, // #normalMap
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>
  ) -> @location(0) vec4<f32> {
    let diffuseColor : vec4<f32> = textureSample(diffuseTexture, samplerTex2D, v_texCoord) +
        vec4<f32>(innerUniforms.tankColorFudge, innerUniforms.tankColorFudge, innerUniforms.tankColorFudge, 1);
    let tangentToWorld : mat3x3<f32> = mat3x3<f32>(v_tangent,  // #normalMap
                                                   v_binormal, // #normalMap
                                                   v_normal);  // #normalMap
    let normalSpec : vec4<f32> = textureSample(normalMapTexture, samplerTex2D, v_texCoord); // #normalMap
    let refraction : vec4<f32> = textureSample(reflectionMapTexture, samplerTex2D, v_texCoord); // #reflection
    var tangentNormal : vec3<f32> = normalSpec.xyz - vec3<f32>(0.5, 0.5, 0.5); // #normalMap
    tangentNormal = normalize(tangentNormal + vec3<f32>(0,0,innerUniforms.refractionFudge));  // #normalMap
    var normal : vec3<f32> = (tangentToWorld * tangentNormal); // #normalMap
    normal = normalize(normal); // #normalMap

    let surfaceToView : vec3<f32> = normalize(v_surfaceToView);

    let refractionVec : vec3<f32> = refract(surfaceToView, normal, innerUniforms.eta);

    let skyColor : vec4<f32> = textureSample(skyboxTexture, samplerSkybox, refractionVec);

    var outColor : vec4<f32> = vec4<f32>(
      mix(skyColor * diffuseColor, diffuseColor, refraction.r).rgb,
      diffuseColor.a);
    outColor = mix(outColor, vec4<f32>(fogs.fogColor.rgb, diffuseColor.a),
      clamp(pow((v_position.z / v_position.w), fogs.fogPower) * fogs.fogMult - fogs.fogOffset,0.0,1.0));
    return outColor;
  }
);

static const char* normal_map_vertex_shader_wgsl = CODE(
  struct LightWorldPositionUniform {
    lightWorldPos : vec3<f32>,
    viewProjection : mat4x4<f32>,
    viewInverse : mat4x4<f32>,
  }

  struct WorldUniform {
    world : mat4x4<f32>,
    worldInverseTranspose : mat4x4<f32>,
    worldViewProjection : mat4x4<f32>,
  }

  struct WorldUniforms {
    worlds : array<WorldUniform, 20>,
  }

  @group(1) @binding(0) var<uniform> lightWorldPositionUniform : LightWorldPositionUniform;
  @group(3) @binding(0) var<uniform> worldUniforms : WorldUniforms;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @builtin(instance_index) instanceIndex: u32,
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>, // #normalMap
    @location(3) v_binormal : vec3<f32>, // #normalMap
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>,
  }

  @vertex
  fn main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>
  ) -> Output {
    var output: Output;
    v_texCoord = texCoord;
    output.v_position = (worldUniforms.worlds[instanceIndex].worldViewProjection * position);
    output.v_normal = (worldUniforms.worlds[instanceIndex].worldInverseTranspose * vec4<f32>(normal, 0)).xyz;
    output.v_surfaceToLight = lightWorldPositionUniform.lightWorldPos - (worldUniforms.worlds[instanceIndex].world * position).xyz;
    output.v_surfaceToView = (lightWorldPositionUniform.viewInverse[3] - (worldUniforms.worlds[instanceIndex].world * position)).xyz;
    output.v_binormal = (worldUniforms.worlds[instanceIndex].worldInverseTranspose * vec4<f32>(binormal, 0)).xyz;  // #normalMap
    output.v_tangent = (worldUniforms.worlds[instanceIndex].worldInverseTranspose * vec4<f32>(tangent, 0)).xyz;  // #normalMap
    output.position = output.v_position;
    return output;
  }
);

static const char* normal_map_fragment_shader_wgsl = CODE(
  struct LightUniforms {
    lightColor : vec4<f32>,
    specular : vec4<f32>,
    ambient : vec4<f32>,
  }

  struct Fogs {
    fogPower : f32,
    fogMult : f32,
    fogOffset : f32,
    fogColor : vec4<f32>,
  }

  struct LightFactorUniforms {
    shininess : f32,
    specularFactor : f32,
  }

  @group(0) @binding(0) var<uniform> lightUniforms : LightUniforms;
  @group(0) @binding(1) var<uniform> fogs : Fogs;
  @group(2) @binding(0) var<uniform> lightFactorUniforms : LightFactorUniforms;
  @group(2) @binding(1) var samplerTex2D: sampler;
  @group(2) @binding(2) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(3) var normalMapTexture: texture_2d<f32>; // #normalMap

  fn lit(l : f32 , h : f32, m : f32) -> vec4f {
    return vec4<f32>(1.0,
                     max(l, 0.0),
                     select(0.0, pow(0.0, max(0.0, h), m), l > 0.0),
                     1.0);
  }

  fn main(
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>, // #normalMap
    @location(3) v_binormal : vec3<f32>, // #normalMap
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>
  ) -> @location(0) vec4<f32> {
    let diffuseColor : vec4<f32> = textureSample(diffuseTexture, diffuseTextureSampler, v_texCoord);
    let tangentToWorld : mat3x3<f32> = mat3x3<f32>(v_tangent,  // #normalMap
                                                   v_binormal, // #normalMap
                                                   v_normal);  // #normalMap
    let normalSpec : vec4<f32> = textureSample(normalMapTexture, samplerTex2D, v_texCoord); // #normalMap
    let tangentNormal : vec3<f32> = normalSpec.xyz - vec3<f32>(0.5, 0.5, 0.5); // #normalMap
    var normal : vec3<f32> = (tangentToWorld * tangentNormal); // #normalMap
    normal = normalize(normal); // #normalMap
    let surfaceToLight : vec3<f32> = normalize(v_surfaceToLight);
    let surfaceToView : vec3<f32> = normalize(v_surfaceToView);
    let halfVector : vec3<f32> = normalize(surfaceToLight + surfaceToView);
    let litR : vec4<f32> = lit(dot(normal, surfaceToLight),
                               dot(normal, halfVector), lightFactorUniforms.shininess);
    var outColor : vec4<f32> = vec4<f32>(
      (lightUniforms.lightColor * (diffuseColor * litR.y + diffuseColor * lightUniforms.ambient +
                     lightUniforms.specular * litR.z * lightFactorUniforms.specularFactor * normalSpec.a)).rgb,
      diffuseColor.a);
    outColor = mix(outColor, vec4(fogs.fogColor.rgb, diffuseColor.a),
      clamp(pow((v_position.z / v_position.w), fogs.fogPower) * fogs.fogMult - fogs.fogOffset,0.0,1.0));
    return outColor;
  }
);

static const char* reflection_map_vertex_shader_wgsl = CODE(
  struct LightWorldPositionUniform {
    lightWorldPos : vec3<f32>,
    viewProjection : mat4x4<f32>,
    viewInverse : mat4x4<f32>,
  }

  struct WorldUniforms {
    world : mat4x4<f32>,
    worldInverseTranspose : mat4x4<f32>,
    worldViewProjection : mat4x4<f32>,
  }

  @group(1) @binding(0) var<uniform> lightWorldPositionUniform : LightWorldPositionUniform;
  @group(3) @binding(0) var<uniform> worldUniforms : WorldUniforms;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>,
    @location(3) v_binormal : vec3<f32>,
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>,
  }

  @vertex
  fn main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>
  ) -> Output {
    var output: Output;
    output.v_texCoord = texCoord;
    output.v_position = (worldUniforms.worldViewProjection * position);
    output.v_normal = (worldUniforms.worldInverseTranspose * vec4<f32>(normal, 0)).xyz;
    output.v_surfaceToLight = lightWorldPositionUniform.lightWorldPos - (worldUniforms.world * position).xyz;
    output.v_surfaceToView = (lightWorldPositionUniform.viewInverse[3] - (worldUniforms.world * position)).xyz;
    output.v_binormal = (worldUniforms.worldInverseTranspose * vec4<f32>(binormal, 0)).xyz;
    output.v_tangent = (worldUniforms.worldInverseTranspose * vec4<f32>(tangent, 0)).xyz;
    output.position = output.v_position;
    return output;
  }
);

static const char* reflection_map_fragment_shader_wgsl = CODE(
  struct LightUniforms {
    lightColor : vec4<f32>,
    specular : vec4<f32>,
    ambient : vec4<f32>,
  }

  struct Fogs {
    fogPower : f32,
    fogMult : f32,
    fogOffset : f32,
    padding : f32,
    fogColor : vec4<f32>,
  }

  struct LightFactorUniforms {
    shininess : f32,
    specularFactor : f32,
  }

  @group(0) @binding(0) var<uniform> lightUniforms : LightUniforms;
  @group(0) @binding(1) var<uniform> fogs : Fogs;
  @group(2) @binding(0) var<uniform> lightFactorUniforms : LightFactorUniforms;
  @group(2) @binding(1) var samplerTex2D: sampler;
  @group(2) @binding(2) var samplerSkybox: sampler;
  @group(2) @binding(3) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(4) var normalMapTexture: texture_2d<f32>;
  @group(2) @binding(5) var reflectionMapTexture: texture_2d<f32>; // #reflection
  @group(2) @binding(6) var skyboxTexture: texture_cube<f32>; // #reflecton

  fn lit(l : f32 , h : f32, m : f32) -> vec4f {
    return vec4<f32>(1.0,
                     max(l, 0.0),
                     select(0.0, pow(0.0, max(0.0, h), m), l > 0.0),
                     1.0);
  }

  @fragment
  fn main(
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_tangent : vec3<f32>,
    @location(3) v_binormal : vec3<f32>,
    @location(4) v_normal : vec3<f32>,
    @location(5) v_surfaceToLight : vec3<f32>,
    @location(6) v_surfaceToView : vec3<f32>
  ) -> @location(0) vec4<f32> {
    let diffuseColor : vec4<f32> = textureSample(diffuseTexture, samplerTex2D, v_texCoord);
    let tangentToWorld : mat3x3<f32> = mat3x3<f32>(v_tangent,
                                                   v_binormal,
                                                   v_normal);
    let normalSpec : vec4<f32> = texture(normalMapTexture, v_texCoord);
    let reflection : vec4<f32> = texture(reflectionMapTexture, v_texCoord);
    var tangentNormal : vec3<f32> = normalSpec.xyz - vec3<f32>(0.5, 0.5, 0.5);
    var normal : vec3<f32> = (tangentToWorld * tangentNormal);
    normal = normalize(normal);
    let surfaceToLight : vec3<f32> = normalize(v_surfaceToLight);
    let surfaceToView : vec3<f32> = normalize(v_surfaceToView);
    let skyColor : vec4<f32> = textureSample(skyboxTexture, samplerSkybox, -reflect(surfaceToView, normal));
    let halfVector : vec3<f32> = normalize(surfaceToLight + surfaceToView);
    let litR : vec4<f32> = lit(dot(normal, surfaceToLight),
                               dot(normal, halfVector), lightFactorUniforms.shininess);
    var outColor : vec4<f32> = vec4<f32>(mix(
      skyColor,
      lightUniforms.lightColor * (diffuseColor * litR.y + diffuseColor * lightUniforms.ambient +
                    lightUniforms.specular * litR.z * lightFactorUniforms.specularFactor * normalSpec.a),
      1.0 - reflection.r).rgb,
      diffuseColor.a);
    outColor = mix(outColor, vec4<f32>(fogs.fogColor.rgb, diffuseColor.a),
      clamp(pow((v_position.z / v_position.w), fogs.fogPower) * fogs.fogMult - fogs.fogOffset,0.0,1.0));
    return outColor;
  }
);

static const char* seaweed_vertex_shader_wgsl = CODE(
  struct LightWorldPositionUniform {
    lightWorldPos : vec3<f32>,
    viewProjection : mat4x4<f32>,
    viewInverse : mat4x4<f32>,
  }

  struct WorldUniform {
    world : mat4x4<f32>,
    worldInverseTranspose : mat4x4<f32>,
    worldViewProjection : mat4x4<f32>,
  }

  struct WorldUniforms {
    worlds : array<WorldUniform, 20>,
  }

  struct SeaweedPer {
    time : array<f32, 20>,
  }

  @group(1) @binding(0) var<uniform> lightWorldPositionUniform : LightWorldPositionUniform;
  @group(3) @binding(0) var<uniform> worldUniforms : WorldUniforms;
  @group(3) @binding(1) var<uniform> seaweedPer : SeaweedPer;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @builtin(instance_index) instanceIndex: u32,
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_normal : vec3<f32>,
    @location(3) v_surfaceToLight : vec3<f32>,
    @location(4) v_surfaceToView : vec3<f32>,
  }

  @vertex
  fn main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>
  ) -> Output {
    var output: Output;
    let toCamera: vec3<f32> = normalize(lightWorldPositionUniform.viewInverse[3].xyz - worldUniforms.worlds[instanceIndex].world[3].xyz);
    let yAxis: vec3<f32> = vec3<f32>(0, 1, 0);
    let xAxis: vec3<f32> = cross(yAxis, toCamera);

    let newWorld : mat4x4<f32> = mat4x4<f32>(
        vec4<f32>(xAxis, 0),
        vec4<f32>(yAxis, 0),
        vec4<f32>(xAxis, 0),
        worldUniforms.worlds[instanceIndex].world[3]
    );

    output.v_texCoord = texCoord;
    output.v_position = position + vec4<f32>(
        sin(seaweedPer.time[instanceIndex] * 0.5) * pow(position.y * 0.07, 2.0) * 1.0,
        -4,  // TODO(gman): remove this hack
        0,
        0
    );
    output.v_position = (lightWorldPositionUniform.viewProjection * newWorld) * v_position;
    output.v_normal = (newWorld * vec4<f32>(normal, 0)).xyz;
    output.v_surfaceToLight = lightWorldPositionUniform.lightWorldPos - (worldUniforms.worlds[instanceIndex].world * position).xyz;
    output.v_surfaceToView = (lightWorldPositionUniform.viewInverse[3] - (worldUniforms.worlds[instanceIndex].world * position)).xyz;
    output.position = output.v_position;
    return output;
  }
);

static const char* seaweed_fragment_shader_wgsl = CODE(
  struct LightUniforms {
    lightColor : vec4<f32>,
    specular : vec4<f32>,
    ambient : vec4<f32>,
  }

  struct Fogs {
    fogPower : f32,
    fogMult : f32,
    fogOffset : f32,
    fogColor : vec4<f32>,
  }

  struct LightFactorUniforms {
    shininess : f32,
    specularFactor : f32,
  }

  @group(0) @binding(0) var<uniform> lightUniforms : LightUniforms;
  @group(0) @binding(1) var<uniform> fogs : Fogs;
  @group(2) @binding(0) var<uniform> lightFactorUniforms : LightFactorUniforms;
  @group(2) @binding(1) var diffuseTextureSampler: sampler;
  @group(2) @binding(2) var diffuseTexture: texture_2d<f32>;

  fn lit(l : f32 , h : f32, m : f32) -> vec4f {
    return vec4<f32>(1.0,
                     max(l, 0.0),
                     select(0.0, pow(0.0, max(0.0, h), m), l > 0.0),
                     1.0);
  }

  @fragment
  fn main(
    @location(0) v_position : vec4<f32>,
    @location(1) v_texCoord : vec2<f32>,
    @location(2) v_normal : vec3<f32>,
    @location(3) v_surfaceToLight : vec3<f32>,
    @location(4) v_surfaceToView : vec3<f32>
  ) -> @location(0) vec4<f32> {
    let diffuseColor : vec4<f32> = textureSample(diffuseTexture, diffuseTextureSampler, v_texCoord);
    if (diffuseColor.a < 0.3) {
      discard;
    }
    let normal : vec3<f32> = normalize(v_normal);
    let surfaceToLight : vec3<f32> = normalize(v_surfaceToLight);
    let surfaceToView : vec3<f32> = normalize(v_surfaceToView);
    let halfVector : vec3<f32> = normalize(surfaceToLight + surfaceToView);
    let litR : vec4<f32> = lit(dot(normal, surfaceToLight),
                               dot(normal, halfVector), lightFactorUniforms.shininess);
    var outColor : vec4<f32> = vec4<f32>((
      lightUniforms.lightColor * (diffuseColor * litR.y + diffuseColor * lightUniforms.ambient +
                    lightUniforms.specular * litR.z * lightFactorUniforms.specularFactor)).rgb,
            diffuseColor.a);
    outColor = mix(outColor, vec4(fogs.fogColor.rgb, diffuseColor.a),
      clamp(pow((v_position.z / v_position.w), fogs.fogPower) * fogs.fogMult - fogs.fogOffset,0.0,1.0));
    return outColor;
  }

);
// clang-format on

/* -------------------------------------------------------------------------- *
 * Aquarium Assert
 * -------------------------------------------------------------------------- */

#ifndef NDEBUG
#define AQUARIUM_ASSERT(expression)                                            \
  {                                                                            \
    if (!(expression)) {                                                       \
      printf("Assertion(%s) failed: file \"%s\", line %d\n", #expression,      \
             __FILE__, __LINE__);                                              \
      abort();                                                                 \
    }                                                                          \
  }
#else
#define AQUARIUM_ASSERT(expression) NULL;
#endif

#ifndef NDEBUG
#define SWALLOW_ERROR(expression)                                              \
  {                                                                            \
    if (!(expression)) {                                                       \
      printf("Assertion(%s) failed: file \"%s\", line %d\n", #expression,      \
             __FILE__, __LINE__);                                              \
    }                                                                          \
  }
#else
#define SWALLOW_ERROR(expression) expression
#endif

/* -------------------------------------------------------------------------- *
 * Matrix: Do matrix calculations including multiply, addition, substraction,
 * transpose, inverse, translation, etc.
 * -------------------------------------------------------------------------- */

static const long long MATRIX_RANDOM_RANGE_ = 4294967296;

static void matrix_mul_matrix_matrix4(float* dst, const float* a,
                                      const float* b)
{
  float a00 = a[0];
  float a01 = a[1];
  float a02 = a[2];
  float a03 = a[3];
  float a10 = a[4 + 0];
  float a11 = a[4 + 1];
  float a12 = a[4 + 2];
  float a13 = a[4 + 3];
  float a20 = a[8 + 0];
  float a21 = a[8 + 1];
  float a22 = a[8 + 2];
  float a23 = a[8 + 3];
  float a30 = a[12 + 0];
  float a31 = a[12 + 1];
  float a32 = a[12 + 2];
  float a33 = a[12 + 3];
  float b00 = b[0];
  float b01 = b[1];
  float b02 = b[2];
  float b03 = b[3];
  float b10 = b[4 + 0];
  float b11 = b[4 + 1];
  float b12 = b[4 + 2];
  float b13 = b[4 + 3];
  float b20 = b[8 + 0];
  float b21 = b[8 + 1];
  float b22 = b[8 + 2];
  float b23 = b[8 + 3];
  float b30 = b[12 + 0];
  float b31 = b[12 + 1];
  float b32 = b[12 + 2];
  float b33 = b[12 + 3];
  dst[0]    = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
  dst[1]    = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
  dst[2]    = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
  dst[3]    = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;
  dst[4]    = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
  dst[5]    = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
  dst[6]    = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
  dst[7]    = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;
  dst[8]    = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
  dst[9]    = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
  dst[10]   = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
  dst[11]   = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;
  dst[12]   = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;
  dst[13]   = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;
  dst[14]   = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;
  dst[15]   = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
}

static void matrix_inverse4(float* dst, const float* m)
{
  float m00    = m[0 * 4 + 0];
  float m01    = m[0 * 4 + 1];
  float m02    = m[0 * 4 + 2];
  float m03    = m[0 * 4 + 3];
  float m10    = m[1 * 4 + 0];
  float m11    = m[1 * 4 + 1];
  float m12    = m[1 * 4 + 2];
  float m13    = m[1 * 4 + 3];
  float m20    = m[2 * 4 + 0];
  float m21    = m[2 * 4 + 1];
  float m22    = m[2 * 4 + 2];
  float m23    = m[2 * 4 + 3];
  float m30    = m[3 * 4 + 0];
  float m31    = m[3 * 4 + 1];
  float m32    = m[3 * 4 + 2];
  float m33    = m[3 * 4 + 3];
  float tmp_0  = m22 * m33;
  float tmp_1  = m32 * m23;
  float tmp_2  = m12 * m33;
  float tmp_3  = m32 * m13;
  float tmp_4  = m12 * m23;
  float tmp_5  = m22 * m13;
  float tmp_6  = m02 * m33;
  float tmp_7  = m32 * m03;
  float tmp_8  = m02 * m23;
  float tmp_9  = m22 * m03;
  float tmp_10 = m02 * m13;
  float tmp_11 = m12 * m03;
  float tmp_12 = m20 * m31;
  float tmp_13 = m30 * m21;
  float tmp_14 = m10 * m31;
  float tmp_15 = m30 * m11;
  float tmp_16 = m10 * m21;
  float tmp_17 = m20 * m11;
  float tmp_18 = m00 * m31;
  float tmp_19 = m30 * m01;
  float tmp_20 = m00 * m21;
  float tmp_21 = m20 * m01;
  float tmp_22 = m00 * m11;
  float tmp_23 = m10 * m01;

  float t0 = (tmp_0 * m11 + tmp_3 * m21 + tmp_4 * m31)
             - (tmp_1 * m11 + tmp_2 * m21 + tmp_5 * m31);
  float t1 = (tmp_1 * m01 + tmp_6 * m21 + tmp_9 * m31)
             - (tmp_0 * m01 + tmp_7 * m21 + tmp_8 * m31);
  float t2 = (tmp_2 * m01 + tmp_7 * m11 + tmp_10 * m31)
             - (tmp_3 * m01 + tmp_6 * m11 + tmp_11 * m31);
  float t3 = (tmp_5 * m01 + tmp_8 * m11 + tmp_11 * m21)
             - (tmp_4 * m01 + tmp_9 * m11 + tmp_10 * m21);

  float d = 1.0f / (m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3);

  dst[0] = d * t0;
  dst[1] = d * t1;
  dst[2] = d * t2;
  dst[3] = d * t3;
  dst[4] = d
           * ((tmp_1 * m10 + tmp_2 * m20 + tmp_5 * m30)
              - (tmp_0 * m10 + tmp_3 * m20 + tmp_4 * m30));
  dst[5] = d
           * ((tmp_0 * m00 + tmp_7 * m20 + tmp_8 * m30)
              - (tmp_1 * m00 + tmp_6 * m20 + tmp_9 * m30));
  dst[6] = d
           * ((tmp_3 * m00 + tmp_6 * m10 + tmp_11 * m30)
              - (tmp_2 * m00 + tmp_7 * m10 + tmp_10 * m30));
  dst[7] = d
           * ((tmp_4 * m00 + tmp_9 * m10 + tmp_10 * m20)
              - (tmp_5 * m00 + tmp_8 * m10 + tmp_11 * m20));
  dst[8] = d
           * ((tmp_12 * m13 + tmp_15 * m23 + tmp_16 * m33)
              - (tmp_13 * m13 + tmp_14 * m23 + tmp_17 * m33));
  dst[9] = d
           * ((tmp_13 * m03 + tmp_18 * m23 + tmp_21 * m33)
              - (tmp_12 * m03 + tmp_19 * m23 + tmp_20 * m33));
  dst[10] = d
            * ((tmp_14 * m03 + tmp_19 * m13 + tmp_22 * m33)
               - (tmp_15 * m03 + tmp_18 * m13 + tmp_23 * m33));
  dst[11] = d
            * ((tmp_17 * m03 + tmp_20 * m13 + tmp_23 * m23)
               - (tmp_16 * m03 + tmp_21 * m13 + tmp_22 * m23));
  dst[12] = d
            * ((tmp_14 * m22 + tmp_17 * m32 + tmp_13 * m12)
               - (tmp_16 * m32 + tmp_12 * m12 + tmp_15 * m22));
  dst[13] = d
            * ((tmp_20 * m32 + tmp_12 * m02 + tmp_19 * m22)
               - (tmp_18 * m22 + tmp_21 * m32 + tmp_13 * m02));
  dst[14] = d
            * ((tmp_18 * m12 + tmp_23 * m32 + tmp_15 * m02)
               - (tmp_22 * m32 + tmp_14 * m02 + tmp_19 * m12));
  dst[15] = d
            * ((tmp_22 * m22 + tmp_16 * m02 + tmp_21 * m12)
               - (tmp_20 * m12 + tmp_23 * m22 + tmp_17 * m02));
}

static void matrix_transpose4(float* dst, const float* m)
{
  float m00 = m[0 * 4 + 0];
  float m01 = m[0 * 4 + 1];
  float m02 = m[0 * 4 + 2];
  float m03 = m[0 * 4 + 3];
  float m10 = m[1 * 4 + 0];
  float m11 = m[1 * 4 + 1];
  float m12 = m[1 * 4 + 2];
  float m13 = m[1 * 4 + 3];
  float m20 = m[2 * 4 + 0];
  float m21 = m[2 * 4 + 1];
  float m22 = m[2 * 4 + 2];
  float m23 = m[2 * 4 + 3];
  float m30 = m[3 * 4 + 0];
  float m31 = m[3 * 4 + 1];
  float m32 = m[3 * 4 + 2];
  float m33 = m[3 * 4 + 3];

  dst[0]  = m00;
  dst[1]  = m10;
  dst[2]  = m20;
  dst[3]  = m30;
  dst[4]  = m01;
  dst[5]  = m11;
  dst[6]  = m21;
  dst[7]  = m31;
  dst[8]  = m02;
  dst[9]  = m12;
  dst[10] = m22;
  dst[11] = m32;
  dst[12] = m03;
  dst[13] = m13;
  dst[14] = m23;
  dst[15] = m33;
}

static void matrix_frustum(float* dst, float left, float right, float bottom,
                           float top, float near_, float far_)
{
  const float dx = right - left;
  const float dy = top - bottom;
  const float dz = near_ - far_;

  dst[0]  = 2 * near_ / dx;
  dst[1]  = 0;
  dst[2]  = 0;
  dst[3]  = 0;
  dst[4]  = 0;
  dst[5]  = 2 * near_ / dy;
  dst[6]  = 0;
  dst[7]  = 0;
  dst[8]  = (left + right) / dx;
  dst[9]  = (top + bottom) / dy;
  dst[10] = far_ / dz;
  dst[11] = -1;
  dst[12] = 0;
  dst[13] = 0;
  dst[14] = near_ * far_ / dz;
  dst[15] = 0;
}

static void matrix_get_axis(float* dst, const float* m, int axis)
{
  const int off = axis * 4;
  dst[0]        = m[off + 0];
  dst[1]        = m[off + 1];
  dst[2]        = m[off + 2];
}

static void matrix_mul_scalar_vector(float k, float* v, size_t length)
{
  for (size_t i = 0; i < length; ++i) {
    v[i] = v[i] * k;
  }
}

static void matrix_add_vector(float* dst, const float* a, const float* b,
                              size_t length)
{
  for (size_t i = 0; i < length; ++i) {
    dst[i] = a[i] + b[i];
  }
}

static void matrix_normalize(float* dst, const float* a, size_t length)
{
  float n = 0.0f;

  for (size_t i = 0; i < length; ++i) {
    n += a[i] * a[i];
  }
  n = sqrt(n);
  if (n > 0.00001f) {
    for (size_t i = 0; i < length; ++i) {
      dst[i] = a[i] / n;
    }
  }
  else {
    for (size_t i = 0; i < length; ++i) {
      dst[i] = 0;
    }
  }
}

static void matrix_sub_vector(float* dst, const float* a, const float* b,
                              size_t length)
{
  for (size_t i = 0; i < length; ++i) {
    dst[i] = a[i] - b[i];
  }
}

static void matrix_cross(float* dst, const float* a, const float* b)
{
  dst[0] = a[1] * b[2] - a[2] * b[1];
  dst[1] = a[2] * b[0] - a[0] * b[2];
  dst[2] = a[0] * b[1] - a[1] * b[0];
}

static float matrix_dot(float* a, float* b)
{
  return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

static void matrix_camera_look_at(float* dst, const float* eye,
                                  const float* target, const float* up)
{
  float t0[3];
  float t1[3];
  float t2[3];
  matrix_sub_vector(t0, eye, target, 3);
  matrix_normalize(t0, t0, 3);
  matrix_cross(t1, up, t0);
  matrix_normalize(t1, t1, 3);
  matrix_cross(t2, t0, t1);

  dst[0]  = t1[0];
  dst[1]  = t1[1];
  dst[2]  = t1[2];
  dst[3]  = 0;
  dst[4]  = t2[0];
  dst[5]  = t2[1];
  dst[6]  = t2[2];
  dst[7]  = 0;
  dst[8]  = t0[0];
  dst[9]  = t0[1];
  dst[10] = t0[2];
  dst[11] = 0;
  dst[12] = eye[0];
  dst[13] = eye[1];
  dst[14] = eye[2];
  dst[15] = 1;
}

static long long matrix_random_seed_ = 0;

static void matrix_reset_pseudoRandom(void)
{
  matrix_random_seed_ = 0;
}

static double matrix_pseudo_random(void)
{
  matrix_random_seed_
    = (134775813 * matrix_random_seed_ + 1) % MATRIX_RANDOM_RANGE_;
  return ((double)matrix_random_seed_) / ((double)MATRIX_RANDOM_RANGE_);
}

static void matrix_translation(float* dst, const float* v)
{
  dst[0]  = 1;
  dst[1]  = 0;
  dst[2]  = 0;
  dst[3]  = 0;
  dst[4]  = 0;
  dst[5]  = 1;
  dst[6]  = 0;
  dst[7]  = 0;
  dst[8]  = 0;
  dst[9]  = 0;
  dst[10] = 1;
  dst[11] = 0;
  dst[12] = v[0];
  dst[13] = v[1];
  dst[14] = v[2];
  dst[15] = 1;
}

static void matrix_translate(float* m, const float* v)
{
  float v0  = v[0];
  float v1  = v[1];
  float v2  = v[2];
  float m00 = m[0];
  float m01 = m[1];
  float m02 = m[2];
  float m03 = m[3];
  float m10 = m[1 * 4 + 0];
  float m11 = m[1 * 4 + 1];
  float m12 = m[1 * 4 + 2];
  float m13 = m[1 * 4 + 3];
  float m20 = m[2 * 4 + 0];
  float m21 = m[2 * 4 + 1];
  float m22 = m[2 * 4 + 2];
  float m23 = m[2 * 4 + 3];
  float m30 = m[3 * 4 + 0];
  float m31 = m[3 * 4 + 1];
  float m32 = m[3 * 4 + 2];
  float m33 = m[3 * 4 + 3];

  m[12] = m00 * v0 + m10 * v1 + m20 * v2 + m30;
  m[13] = m01 * v0 + m11 * v1 + m21 * v2 + m31;
  m[14] = m02 * v0 + m12 * v1 + m22 * v2 + m32;
  m[15] = m03 * v0 + m13 * v1 + m23 * v2 + m33;
}

static float deg_to_rad(float degrees)
{
  return degrees * PI / 180.0f;
}

/* -------------------------------------------------------------------------- *
 * Aquarium - Global enums
 * -------------------------------------------------------------------------- */

typedef enum {
  /* Begin of background */
  MODELNAME_MODELRUINCOLUMN,
  MODELNAME_MODELARCH,
  MODELNAME_MODELROCKA,
  MODELNAME_MODELROCKB,
  MODELNAME_MODELROCKC,
  MODELNAME_MODELSUNKNSHIPBOXES,
  MODELNAME_MODELSUNKNSHIPDECK,
  MODELNAME_MODELSUNKNSHIPHULL,
  MODELNAME_MODELFLOORBASE_BAKED,
  MODELNAME_MODELSUNKNSUB,
  MODELNAME_MODELCORAL,
  MODELNAME_MODELSTONE,
  MODELNAME_MODELCORALSTONEA,
  MODELNAME_MODELCORALSTONEB,
  MODELNAME_MODELGLOBEBASE,
  MODELNAME_MODELTREASURECHEST,
  MODELNAME_MODELENVIRONMENTBOX,
  MODELNAME_MODELSUPPORTBEAMS,
  MODELNAME_MODELSKYBOX,
  MODELNAME_MODELGLOBEINNER,
  MODELNAME_MODELSEAWEEDA,
  MODELNAME_MODELSEAWEEDB,

  /* Begin of fish */
  MODELNAME_MODELSMALLFISHA,
  MODELNAME_MODELMEDIUMFISHA,
  MODELNAME_MODELMEDIUMFISHB,
  MODELNAME_MODELBIGFISHA,
  MODELNAME_MODELBIGFISHB,
  MODELNAME_MODELSMALLFISHAINSTANCEDDRAWS,
  MODELNAME_MODELMEDIUMFISHAINSTANCEDDRAWS,
  MODELNAME_MODELMEDIUMFISHBINSTANCEDDRAWS,
  MODELNAME_MODELBIGFISHAINSTANCEDDRAWS,
  MODELNAME_MODELBIGFISHBINSTANCEDDRAWS,
  MODELNAME_MODELMAX,
} model_name_t;

typedef enum {
  MODELGROUP_FISH,
  MODELGROUP_FISHINSTANCEDDRAW,
  MODELGROUP_INNER,
  MODELGROUP_SEAWEED,
  MODELGROUP_GENERIC,
  MODELGROUP_OUTSIDE,
  MODELGROUP_GROUPMAX,
} model_group_t;

typedef enum {
  FISHENUM_BIG,
  FISHENUM_MEDIUM,
  FISHENUM_SMALL,
  FISHENUM_MAX,
} fish_num_t;

typedef enum {
  BUFFERTYPE_POSITION,
  BUFFERTYPE_NORMAL,
  BUFFERTYPE_TEX_COORD,
  BUFFERTYPE_TANGENT,
  BUFFERTYPE_BI_NORMAL,
  BUFFERTYPE_INDICES,
  BUFFERTYPE_MAX,
} buffer_type_t;

typedef enum {
  TEXTURETYPE_DIFFUSE,
  TEXTURETYPE_NORMAL_MAP,
  TEXTURETYPE_REFLECTION,
  TEXTURETYPE_REFLECTION_MAP,
  TEXTURETYPE_SKYBOX,
  TEXTURETYPE_MAX,
} texture_type_t;

typedef enum {
  VERTEX_SHADER_DIFFUSE,
  VERTEX_SHADER_FISH,
  VERTEX_SHADER_FISH_INSTANCED_DRAWS,
  VERTEX_SHADER_INNER_REFRACTION_MAP,
  VERTEX_SHADER_NORMAL_MAP,
  VERTEX_SHADER_REFLECTION_MAP,
  VERTEX_SHADER_SEAWEED,
  VERTEX_SHADER_MAX,
} vertex_shader_t;

typedef enum {
  FRAGMENT_SHADER_DIFFUSE,
  FRAGMENT_SHADER_FISH_NORMAL_MAP,
  FRAGMENT_SHADER_FISH_REFLECTION,
  FRAGMENT_SHADER_INNER_REFRACTION_MAP,
  FRAGMENT_SHADER_NORMAL_MAP,
  FRAGMENT_SHADER_REFLECTION_MAP,
  FRAGMENT_SHADER_SEAWEED,
  FRAGMENT_SHADER_MAX,
} fragment_shader_t;

typedef enum {
  /* Enable alpha blending */
  ENABLEALPHABLENDING,
  /* Go through instanced draw */
  ENABLEINSTANCEDDRAWS,
  // The toggle is only supported on Dawn backend
  // By default, the app will enable dynamic buffer offset
  // The toggle is to disable dbo feature
  ENABLEDYNAMICBUFFEROFFSET,
  /* Turn off render pass on dawn_d3d12 */
  DISABLED3D12RENDERPASS,
  /* Turn off dawn validation */
  DISABLEDAWNVALIDATION,
  /* Disable control panel */
  DISABLECONTROLPANEL,
  /* Select integrated gpu if available */
  INTEGRATEDGPU,
  /* Select discrete gpu if available */
  DISCRETEGPU,
  /* Draw per instance or model */
  DRAWPERMODEL,
  /* Support Full Screen mode */
  ENABLEFULLSCREENMODE,
  /* Print logs such as avg fps */
  PRINTLOG,
  /* Use async buffer mapping to upload data */
  BUFFERMAPPINGASYNC,
  /* Simulate fish come and go for Dawn backend */
  SIMULATINGFISHCOMEANDGO,
  /* Turn off vsync, donot limit fps to 60 */
  TURNOFFVSYNC,
  TOGGLEMAX,
} toggle_t;

/* -------------------------------------------------------------------------- *
 * Aquarium - Global enums convertion functions
 * -------------------------------------------------------------------------- */

static texture_type_t string_to_texture_type(const char* texture_type_str)
{
  texture_type_t texture_type = TEXTURETYPE_MAX;
  if (strcmp(texture_type_str, "diffuse") == 0) {
    texture_type = TEXTURETYPE_DIFFUSE;
  }
  else if (strcmp(texture_type_str, "normalMap") == 0) {
    texture_type = TEXTURETYPE_NORMAL_MAP;
  }
  else if (strcmp(texture_type_str, "reflectionMap") == 0) {
    texture_type = TEXTURETYPE_REFLECTION_MAP;
  }
  else if (strcmp(texture_type_str, "skybox") == 0) {
    texture_type = TEXTURETYPE_SKYBOX;
  }
  return texture_type;
}

static buffer_type_t string_to_buffer_type(const char* buffer_type_str)
{
  buffer_type_t buffer_type = BUFFERTYPE_MAX;
  if (strcmp(buffer_type_str, "position") == 0) {
    buffer_type = BUFFERTYPE_POSITION;
  }
  else if (strcmp(buffer_type_str, "normal") == 0) {
    buffer_type = BUFFERTYPE_NORMAL;
  }
  else if (strcmp(buffer_type_str, "texCoord") == 0) {
    buffer_type = BUFFERTYPE_TEX_COORD;
  }
  else if (strcmp(buffer_type_str, "tangent") == 0) {
    buffer_type = BUFFERTYPE_TANGENT;
  }
  else if (strcmp(buffer_type_str, "binormal") == 0) {
    buffer_type = BUFFERTYPE_BI_NORMAL;
  }
  else if (strcmp(buffer_type_str, "indices") == 0) {
    buffer_type = BUFFERTYPE_INDICES;
  }
  return buffer_type;
}

/* -------------------------------------------------------------------------- *
 * Aquarium - Global classes
 * -------------------------------------------------------------------------- */

typedef struct {
  const char* name_str;
  model_name_t name;
  model_group_t type;
  struct {
    char vertex[STRMAX];
    char fragment[STRMAX];
  } program;
  bool fog;
  bool blend;
} g_scene_info_t;

#define FISH_BEHAVIOR_COUNT (3u)

typedef struct {
  uint32_t frame;
  char op;
  uint32_t count;
} g_fish_behavior_t;

typedef struct {
  const char* name;
  model_name_t model_name;
  fish_num_t type;
  float speed;
  float speed_range;
  float radius;
  float radius_range;
  float tail_speed;
  float height_offset;
  float height_range;

  float fish_length;
  float fish_wave_length;
  float fish_bend_amount;

  bool lasers;
  float laser_rot;
  vec3 laser_off;
  vec3 laser_scale;
} fish_t;

typedef struct {
  float tail_offset_mult;
  float end_of_dome;
  float tank_radius;
  float tank_height;
  float stand_height;
  float shark_speed;
  float shark_clock_offset;
  float shark_xclock;
  float shark_yclock;
  float shark_zclock;
  int32_t numBubble_sets;
  float laser_eta;
  float laser_len_fudge;
  int32_t num_light_rays;
  int32_t light_ray_y;
  int32_t light_ray_duration_min;
  int32_t light_ray_duration_range;
  int32_t light_ray_speed;
  int32_t light_ray_spread;
  int32_t light_ray_pos_range;
  float light_ray_rot_range;
  float light_ray_rot_lerp;
  float light_ray_offset;
  float bubble_timer;
  int32_t bubble_index;

  int32_t num_fish_small;
  int32_t num_fish_medium;
  int32_t num_fish_big;
  int32_t num_fish_left_small;
  int32_t num_fish_left_big;
  float sand_shininess;
  float sand_specular_factor;
  float generic_shininess;
  float generic_specular_factor;
  float outside_shininess;
  float outside_specular_factor;
  float seaweed_shininess;
  float seaweed_specular_factor;
  float inner_shininess;
  float inner_specular_factor;
  float fish_shininess;
  float fish_specular_factor;

  float speed;
  float target_height;
  float target_radius;
  float eye_height;
  float eye_speed;
  float filed_of_view;
  float ambient_red;
  float ambient_green;
  float ambient_blue;
  float fog_power;
  float fog_mult;
  float fog_offset;
  float fog_red;
  float fog_green;
  float fog_blue;
  float fish_height_range;
  float fish_height;
  float fish_speed;
  float fish_offset;
  float fish_xclock;
  float fish_yclock;
  float fish_zclock;
  float fish_tail_speed;
  float refraction_fudge;
  float eta;
  float tank_colorfudge;
  float fov_fudge;
  vec3 net_offset;
  float net_offset_mult;
  float eye_radius;
  float field_of_view;
} g_settings_t;

typedef struct {
  float projection[16];
  float view[16];
  float world_inverse[16];
  float view_projection_inverse[16];
  float sky_view[16];
  float sky_view_projection[16];
  float sky_view_projection_inverse[16];
  float eye_position[3];
  float target[3];
  float up[3];
  float v3t0[3];
  float v3t1[3];
  float m4t0[16];
  float m4t1[16];
  float m4t2[16];
  float m4t3[16];
  float color_mult[4];
  float start;
  float then;
  float mclock;
  float eye_clock;
  float alpha;
} global_t;

typedef struct {
  float light_world_pos[3];
  float padding;
  float view_projection[16];
  float view_inverse[16];
} light_world_position_uniform_t;

typedef struct {
  float world[16];
  float world_inverse_transpose[16];
  float world_view_projection[16];
} world_uniforms_t;

typedef struct {
  vec4 light_color;
  vec4 specular;
  vec4 ambient;
} light_uniforms_t;

typedef struct {
  float fog_power;
  float fog_mult;
  float fog_offset;
  float padding;
  vec4 fog_color;
} fog_uniforms_t;

typedef struct {
  vec3 world_position;
  float scale;
  vec3 next_position;
  float time;
  float padding[56]; // Padding to align with 256 byte offset.
} fish_per_t;

/* -------------------------------------------------------------------------- *
 * Aquarium - Global (constant) variables
 * -------------------------------------------------------------------------- */

enum {
  /* Begin of background */
  MODELRUINCOLUMN_COUNT      = 2,
  MODELARCH_COUNT            = 2,
  MODELROCKA_COUNT           = 5,
  MODELROCKB_COUNT           = 4,
  MODELROCKC_COUNT           = 4,
  MODELSUNKNSHIPBOXES_COUNT  = 3,
  MODELSUNKNSHIPDECK_COUNT   = 3,
  MODELSUNKNSHIPHULL_COUNT   = 3,
  MODELFLOORBASE_BAKED_COUNT = 1,
  MODELSUNKNSUB_COUNT        = 2,
  MODELCORAL_COUNT           = 16,
  MODELSTONE_COUNT           = 11,
  MODELCORALSTONEA_COUNT     = 4,
  MODELCORALSTONEB_COUNT     = 3,
  MODELGLOBEBASE_COUNT       = 1,
  MODELTREASURECHEST_COUNT   = 6,
  MODELENVIRONMENTBOX_COUNT  = 1,
  MODELSUPPORTBEAMS_COUNT    = 1,
  MODELSKYBOX_COUNT          = 1,
  MODELGLOBEINNER_COUNT      = 1,
  MODELSEAWEEDA_COUNT        = 11,
  MODELSEAWEEDB_COUNT        = 11,
};

static struct {
  bool enable_alpha_blending;        /* Enable alpha blending */
  bool enable_instanced_draw;        /* Go through instanced draw */
  bool enable_dynamic_buffer_offset; /* Enable dynamic buffer offset */
  bool draw_per_model;               /*  Draw per instance or model */
  bool print_log;                    /* Print logs such as avg fps */
  bool buffer_mapping_async;      /* Use async buffer mapping to upload data */
  bool simulate_fish_come_and_go; /* Simulate fish come and go */
  bool turn_off_vsync;            /* Turn off vsync, donot limit fps to 60 */
  uint32_t msaa_sample_count;     /* MSAA sample count */
} aquarium_settings;

static const g_scene_info_t g_scene_info[MODELNAME_MODELMAX] = {
  {
    .name_str         = "SmallFishA",
    .name             = MODELNAME_MODELSMALLFISHA,
    .type             = MODELGROUP_FISH,
    .program.vertex   = "fishVertexShader",
    .program.fragment = "fishReflectionFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "MediumFishA",
    .name             = MODELNAME_MODELMEDIUMFISHA,
    .type             = MODELGROUP_FISH,
    .program.vertex   = "fishVertexShader",
    .program.fragment = "fishNormalMapFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "MediumFishB",
    .name             = MODELNAME_MODELMEDIUMFISHB,
    .type             = MODELGROUP_FISH,
    .program.vertex   = "fishVertexShader",
    .program.fragment = "fishReflectionFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "BigFishA",
    .name             = MODELNAME_MODELBIGFISHA,
    .type             = MODELGROUP_FISH,
    .program.vertex   = "fishVertexShader",
    .program.fragment = "fishNormalMapFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "BigFishB",
    .name             = MODELNAME_MODELBIGFISHB,
    .type             = MODELGROUP_FISH,
    .program.vertex   = "fishVertexShader",
    .program.fragment = "fishNormalMapFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "SmallFishA",
    .name             = MODELNAME_MODELSMALLFISHAINSTANCEDDRAWS,
    .type             = MODELGROUP_FISHINSTANCEDDRAW,
    .program.vertex   = "fishVertexShaderInstancedDraws",
    .program.fragment = "fishReflectionFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "MediumFishA",
    .name             = MODELNAME_MODELMEDIUMFISHAINSTANCEDDRAWS,
    .type             = MODELGROUP_FISHINSTANCEDDRAW,
    .program.vertex   = "fishVertexShaderInstancedDraws",
    .program.fragment = "fishNormalMapFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "MediumFishB",
    .name             = MODELNAME_MODELMEDIUMFISHBINSTANCEDDRAWS,
    .type             = MODELGROUP_FISHINSTANCEDDRAW,
    .program.vertex   = "fishVertexShaderInstancedDraws",
    .program.fragment = "fishReflectionFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "BigFishA",
    .name             = MODELNAME_MODELBIGFISHAINSTANCEDDRAWS,
    .type             = MODELGROUP_FISHINSTANCEDDRAW,
    .program.vertex   = "fishVertexShaderInstancedDraws",
    .program.fragment = "fishNormalMapFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "BigFishB",
    .name             = MODELNAME_MODELBIGFISHBINSTANCEDDRAWS,
    .type             = MODELGROUP_FISHINSTANCEDDRAW,
    .program.vertex   = "fishVertexShaderInstancedDraws",
    .program.fragment = "fishNormalMapFragmentShader",
    .fog              = true,
  },
  {
    .name_str         = "Arch",
    .name             = MODELNAME_MODELARCH,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "Coral",
    .name             = MODELNAME_MODELCORAL,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "CoralStoneA",
    .name             = MODELNAME_MODELCORALSTONEA,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "CoralStoneB",
    .name             = MODELNAME_MODELCORALSTONEB,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "EnvironmentBox",
    .name             = MODELNAME_MODELENVIRONMENTBOX,
    .type             = MODELGROUP_OUTSIDE,
    .program.vertex   = "diffuseVertexShader",
    .program.fragment = "diffuseFragmentShader",
    .fog              = false,
  },
  {
    .name_str         = "FloorBase_Baked",
    .name             = MODELNAME_MODELFLOORBASE_BAKED,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "GlobeBase",
    .name             = MODELNAME_MODELGLOBEBASE,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "diffuseVertexShader",
    .program.fragment = "diffuseFragmentShader",
    .fog              = false,
  },
  {
    .name_str         = "GlobeInner",
    .name             = MODELNAME_MODELGLOBEINNER,
    .type             = MODELGROUP_INNER,
    .program.vertex   = "innerRefractionMapVertexShader",
    .program.fragment = "innerRefractionMapFragmentShader",
    .fog              = false,
  },
  {
    .name_str         = "RockA",
    .name             = MODELNAME_MODELROCKA,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "RockB",
    .name             = MODELNAME_MODELROCKB,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "RockC",
    .name             = MODELNAME_MODELROCKC,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "RuinColumn",
    .name             = MODELNAME_MODELRUINCOLUMN,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "Stone",
    .name             = MODELNAME_MODELSTONE,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "SunknShipBoxes",
    .name             = MODELNAME_MODELSUNKNSHIPBOXES,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "SunknShipDeck",
    .name             = MODELNAME_MODELSUNKNSHIPDECK,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "SunknShipHull",
    .name             = MODELNAME_MODELSUNKNSHIPHULL,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "SunknSub",
    .name             = MODELNAME_MODELSUNKNSUB,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
  {
    .name_str         = "SeaweedA",
    .name             = MODELNAME_MODELSEAWEEDA,
    .type             = MODELGROUP_SEAWEED,
    .program.vertex   = "seaweedVertexShader",
    .program.fragment = "seaweedFragmentShader",
    .fog              = false,
  },
  {
    .name_str         = "SeaweedB",
    .name             = MODELNAME_MODELSEAWEEDB,
    .type             = MODELGROUP_SEAWEED,
    .program.vertex   = "seaweedVertexShader",
    .program.fragment = "seaweedFragmentShader",
    .fog              = false,
  },
  {
    .name_str         = "Skybox",
    .name             = MODELNAME_MODELSKYBOX,
    .type             = MODELGROUP_OUTSIDE,
    .program.vertex   = "diffuseVertexShader",
    .program.fragment = "diffuseFragmentShader",
    .fog              = false,
  },
  {
    .name_str         = "SupportBeams",
    .name             = MODELNAME_MODELSUPPORTBEAMS,
    .type             = MODELGROUP_OUTSIDE,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = false,
  },
  {
    .name_str         = "TreasureChest",
    .name             = MODELNAME_MODELTREASURECHEST,
    .type             = MODELGROUP_GENERIC,
    .program.vertex   = "",
    .program.fragment = "",
    .fog              = true,
  },
};

static const g_fish_behavior_t g_fish_behaviors[FISH_BEHAVIOR_COUNT] = {
  {
    .frame = 200,
    .op    = '+',
    .count = 5000,
  },
  {
    .frame = 200,
    .op    = '+',
    .count = 15000,
  },
  {
    .frame = 200,
    .op    = '-',
    .count = 5000,
  },
};

static const fish_t fish_table[5] = {
  {
    .name             = "SmallFishA",
    .model_name       = MODELNAME_MODELSMALLFISHA,
    .type             = FISHENUM_SMALL,
    .speed            = 1.0f,
    .speed_range      = 1.5f,
    .radius           = 30.0f,
    .radius_range     = 25.0f,
    .tail_speed       = 10.0f,
    .height_offset    = 0.0f,
    .height_range     = 16.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = 1.0f,
    .fish_bend_amount = 2.0f,
  },
  {
    .name             = "MediumFishA",
    .model_name       = MODELNAME_MODELMEDIUMFISHA,
    .type             = FISHENUM_MEDIUM,
    .speed            = 1.0f,
    .speed_range      = 2.0f,
    .radius           = 10.0f,
    .radius_range     = 20.0f,
    .tail_speed       = 1.0f,
    .height_offset    = 0.0f,
    .height_range     = 16.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = -2.0f,
    .fish_bend_amount = 2.0f,
  },
  {
    .name             = "MediumFishB",
    .model_name       = MODELNAME_MODELMEDIUMFISHB,
    .type             = FISHENUM_MEDIUM,
    .speed            = 0.5f,
    .speed_range      = 4.0f,
    .radius           = 10.0f,
    .radius_range     = 20.0f,
    .tail_speed       = 3.0f,
    .height_offset    = -8.0f,
    .height_range     = 5.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = -2.0f,
    .fish_bend_amount = 2.0f,
  },
  {
    .name             = "BigFishA",
    .model_name       = MODELNAME_MODELBIGFISHA,
    .type             = FISHENUM_BIG,
    .speed            = 0.5f,
    .speed_range      = 0.5f,
    .radius           = 50.0f,
    .radius_range     = 3.0f,
    .tail_speed       = 1.5f,
    .height_offset    = 0.0f,
    .height_range     = 16.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = -1.0f,
    .fish_bend_amount = 0.5f,
    .lasers           = true,
    .laser_rot        = 0.04f,
    .laser_off        = {0.0f, 0.1f, 9.0f},
    .laser_scale      = {0.3f, 0.3f, 1000.0f},
  },
  {
    .name             = "BigFishB",
    .model_name       = MODELNAME_MODELBIGFISHA,
    .type             = FISHENUM_BIG,
    .speed            = 0.5f,
    .speed_range      = 0.5f,
    .radius           = 45.0f,
    .radius_range     = 3.0f,
    .tail_speed       = 1.0f,
    .height_offset    = 0.0f,
    .height_range     = 16.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = -0.7f,
    .fish_bend_amount = 0.3f,
    .lasers           = true,
    .laser_rot        = 0.04f,
    .laser_off        = {0.0f, -0.3f, 9.0f},
    .laser_scale      = {0.3f, 0.3f, 1000.0f},
  },
};

static const int g_num_light_rays = 5;

static g_settings_t g_settings = {
  .tail_offset_mult         = 1.0f,
  .end_of_dome              = PI / 8.f,
  .tank_radius              = 74.0f,
  .tank_height              = 36.0f,
  .stand_height             = 25.0f,
  .shark_speed              = 0.3f,
  .shark_clock_offset       = 17.0f,
  .shark_xclock             = 1.0f,
  .shark_yclock             = 0.17f,
  .shark_zclock             = 1.0f,
  .numBubble_sets           = 10,
  .laser_eta                = 1.2f,
  .laser_len_fudge          = 1.0f,
  .num_light_rays           = g_num_light_rays,
  .light_ray_y              = 50,
  .light_ray_duration_min   = 1,
  .light_ray_duration_range = 1,
  .light_ray_speed          = 4,
  .light_ray_spread         = 7,
  .light_ray_pos_range      = 20,
  .light_ray_rot_range      = 1.0f,
  .light_ray_rot_lerp       = 0.2f,
  .light_ray_offset         = PI2 / (float)g_num_light_rays,
  .bubble_timer             = 0.0f,
  .bubble_index             = 0,

  .num_fish_small          = 100,
  .num_fish_medium         = 1000,
  .num_fish_big            = 10000,
  .num_fish_left_small     = 80,
  .num_fish_left_big       = 160,
  .sand_shininess          = 5.0f,
  .sand_specular_factor    = 0.3f,
  .generic_shininess       = 50.0f,
  .generic_specular_factor = 1.0f,
  .outside_shininess       = 50.0f,
  .outside_specular_factor = 0.0f,
  .seaweed_shininess       = 50.0f,
  .seaweed_specular_factor = 1.0f,
  .inner_shininess         = 50.0f,
  .inner_specular_factor   = 1.0f,
  .fish_shininess          = 5.0f,
  .fish_specular_factor    = 0.3f,

  .speed             = 1.0f,
  .target_height     = 63.3f,
  .target_radius     = 91.6f,
  .eye_height        = 7.5f,
  .eye_speed         = 0.0258f,
  .filed_of_view     = 82.699f,
  .ambient_red       = 0.218f,
  .ambient_green     = 0.502f,
  .ambient_blue      = 0.706f,
  .fog_power         = 16.5f,
  .fog_mult          = 1.5f,
  .fog_offset        = 0.738f,
  .fog_red           = 0.338f,
  .fog_green         = 0.81f,
  .fog_blue          = 1.0f,
  .fish_height_range = 1.0f,
  .fish_height       = 25.0f,
  .fish_speed        = 0.124f,
  .fish_offset       = 0.52f,
  .fish_xclock       = 1.0f,
  .fish_yclock       = 0.556f,
  .fish_zclock       = 1.0f,
  .fish_tail_speed   = 1.0f,
  .refraction_fudge  = 3.0f,
  .eta               = 1.0f,
  .tank_colorfudge   = 0.796f,
  .fov_fudge         = 1.0f,
  .net_offset        = {0.0f, 0.0f, 0.0f},
  .net_offset_mult   = 1.21f,
  .eye_radius        = 13.2f,
  .field_of_view     = 82.699f,
};

/* -------------------------------------------------------------------------- *
 * FPSTimer - Defines fps timer, uses millseconds time unit.
 * -------------------------------------------------------------------------- */

#define NUM_HISTORY_DATA 100u
#define NUM_FRAMES_TO_AVERAGE 128u
#define FPS_VALID_THRESHOLD 5u

sc_array_def(float, float);

typedef struct {
  float total_time;
  struct sc_array_float time_table;
  int32_t time_table_cursor;
  struct sc_array_float history_fps;
  struct sc_array_float history_frame_time;
  struct sc_array_float log_fps;
  float average_fps;
} fps_timer_t;

static void fps_timer_init_defaults(fps_timer_t* this)
{
  memset(this, 0, sizeof(*this));

  this->total_time = NUM_FRAMES_TO_AVERAGE * 1000.0f;

  for (uint32_t i = 0; i < NUM_FRAMES_TO_AVERAGE; ++i) {
    sc_array_add(&this->time_table, 1000.0f);
  }

  this->time_table_cursor = 0;

  for (uint32_t i = 0; i < NUM_HISTORY_DATA; ++i) {
    sc_array_add(&this->history_fps, 1.0f);
    sc_array_add(&this->history_frame_time, 100.0f);
  }

  this->average_fps = 0.0f;
}

static void fps_timer_create(fps_timer_t* this)
{
  fps_timer_init_defaults(this);
}

static void fps_timer_update(fps_timer_t* this, float elapsed_time,
                             float rendering_time, float test_time)
{
  this->total_time
    += elapsed_time - this->time_table.elems[this->time_table_cursor];
  this->time_table.elems[this->time_table_cursor] = elapsed_time;

  ++this->time_table_cursor;
  if (this->time_table_cursor == NUM_FRAMES_TO_AVERAGE) {
    this->time_table_cursor = 0;
  }

  float frame_time  = this->total_time / NUM_FRAMES_TO_AVERAGE;
  this->average_fps = floor(1000.0f / frame_time + 0.5f);

  for (uint32_t i = 0; i < NUM_HISTORY_DATA - 1; ++i) {
    this->history_fps.elems[i]        = this->history_fps.elems[i + 1];
    this->history_frame_time.elems[i] = this->history_frame_time.elems[i + 1];
  }

  this->history_fps.elems[NUM_HISTORY_DATA - 1] = this->average_fps;
  this->history_frame_time.elems[NUM_HISTORY_DATA - 1]
    = 1000.0 / this->average_fps;

  if (test_time - rendering_time > 5000.0f
      && test_time - rendering_time < 25000.0f) {
    sc_array_add(&this->log_fps, this->average_fps);
  }
}

static float fps_timer_get_average_fps(fps_timer_t* this)
{
  return this->average_fps;
}

static int32_t fps_timer_variance(fps_timer_t* this)
{
  float avg = 0.0f;

  for (size_t i = 0; i < sc_array_size(&this->log_fps); ++i) {
    avg += this->log_fps.elems[i];
  }
  avg /= sc_array_size(&this->log_fps);

  float var = 0.0f;
  for (size_t i = 0; i < sc_array_size(&this->log_fps); ++i) {
    var += pow(this->log_fps.elems[i] - avg, 2);
  }
  var /= sc_array_size(&this->log_fps);

  if (var < FPS_VALID_THRESHOLD) {
    return (int32_t)ceil(avg);
  }

  return 0;
}

/* -------------------------------------------------------------------------- *
 * Behavior - Base class for behavior.
 * -------------------------------------------------------------------------- */

typedef enum {
  OPERATION_PLUS,
  OPERATION_MINUS,
} behavior_op_t;

typedef struct behavior_t {
  int32_t _frame;
  behavior_op_t _op;
  int32_t _count;
} behavior_t;

static void behavior_create(behavior_t* this, int32_t frame, char op,
                            int32_t count)
{
  this->_frame = frame;
  this->_op    = (op == '+') ? OPERATION_PLUS : OPERATION_MINUS;
  this->_count = count;
}

static int32_t behavior_get_frame(behavior_t* this)
{
  return this->_frame;
}

static behavior_op_t behavior_get_op(behavior_t* this)
{
  return this->_op;
}

static int32_t behavior_get_count(behavior_t* this)
{
  return this->_count;
}

static void behavior_set_frame(behavior_t* this, int32_t frame)
{
  this->_frame = frame;
}

/* -------------------------------------------------------------------------- *
 * Program - Load programs from shaders.
 * -------------------------------------------------------------------------- */

typedef struct {
  void* context;
  const char* vertex_shader_code;
  const char* fragment_shader_code;
  wgpu_shader_t vs_module;
  wgpu_shader_t fs_module;
  struct {
    bool enable_alpha_blending;
    float alpha;
  } options;
} program_t;

/* Forward declarations */
static wgpu_shader_t context_create_shader_module(void* context,
                                                  WGPUShaderStage shader_stage,
                                                  const char* shader_path);

static void program_init_defaults(program_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void program_create(program_t* this, void* context,
                           const char* vertex_shader_code,
                           const char* fragment_shader_code)
{
  program_init_defaults(this);
  this->context = context;

  this->vertex_shader_code   = vertex_shader_code;
  this->fragment_shader_code = fragment_shader_code;
}

static void program_destroy(program_t* this)
{
  WGPU_RELEASE_RESOURCE(ShaderModule, this->vs_module.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, this->fs_module.module);
}

static WGPUShaderModule program_get_vs_module(program_t* this)
{
  return this->vs_module.module;
}

static WGPUShaderModule program_get_fs_module(program_t* this)
{
  return this->fs_module.module;
}

static void program_set_options(program_t* this, bool enable_alpha_blending,
                                float alpha)
{
  this->options.enable_alpha_blending = enable_alpha_blending;
  this->options.alpha                 = alpha;
}

static void program_compile_program(program_t* this)
{
  this->vs_module = context_create_shader_module(
    this->context, WGPUShaderStage_Vertex, this->vertex_shader_code);
  this->fs_module = context_create_shader_module(
    this->context, WGPUShaderStage_Fragment, this->fragment_shader_code);
}

/* -------------------------------------------------------------------------- *
 * Dawn Buffer - Defines the buffer wrapper of dawn, abstracting the vetex and
 * index buffer binding.
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUBuffer buffer;
  WGPUBufferUsage usage;
  int32_t total_components;
  uint32_t stride;
  void* offset;
  int32_t size;
  bool valid;
} buffer_dawn_t;

/* Forward declarations */
static WGPUBuffer context_create_buffer(void* context,
                                        WGPUBufferDescriptor const* descriptor);
static void context_set_buffer_data(void* context, WGPUBuffer buffer,
                                    uint32_t buffer_size, const void* data,
                                    uint32_t data_size);
static void context_update_buffer_data(void* this, WGPUBuffer buffer,
                                       size_t buffer_size, void* data,
                                       size_t data_size);

/* Copy size must be a multiple of 4 bytes on dawn mac backend. */
static void buffer_dawn_create_f32(buffer_dawn_t* this, void* context,
                                   int32_t total_components,
                                   int32_t num_components, float* buffer,
                                   bool is_index)
{
  this->usage = is_index ? WGPUBufferUsage_Index : WGPUBufferUsage_Vertex;
  this->total_components = total_components;
  this->stride           = 0;
  this->offset           = NULL;

  this->size = num_components * sizeof(float);
  // Create buffer for vertex buffer. Because float is multiple of 4 bytes,
  // dummy padding isnt' needed.
  uint64_t buffer_size             = sizeof(float) * num_components;
  WGPUBufferDescriptor buffer_desc = {
    .usage            = this->usage | WGPUBufferUsage_CopyDst,
    .size             = buffer_size,
    .mappedAtCreation = false,
  };
  this->buffer = context_create_buffer(context, &buffer_desc);

  context_set_buffer_data(context, this->buffer, buffer_size, buffer,
                          buffer_size);
}

static void buffer_dawn_create_uint16(buffer_dawn_t* this, void* context,
                                      int32_t total_components,
                                      int32_t num_components, uint16_t* buffer,
                                      uint64_t buffer_count, bool is_index)
{
  this->usage = is_index ? WGPUBufferUsage_Index : WGPUBufferUsage_Vertex;
  this->total_components = total_components;
  this->stride           = 0;
  this->offset           = NULL;

  this->size = num_components * sizeof(uint16_t);
  // Create buffer for index buffer. Because unsigned short is multiple of 2
  // bytes, in order to align with 4 bytes of dawn metal, dummy padding need to
  // be added.
  if (total_components % 2 != 0) {
    ASSERT((uint64_t)num_components <= buffer_count);
    buffer[num_components] = 0;
    num_components++;
  }

  uint64_t buffer_size             = sizeof(uint16_t) * num_components;
  WGPUBufferDescriptor buffer_desc = {
    .usage            = this->usage | WGPUBufferUsage_CopyDst,
    .size             = buffer_size,
    .mappedAtCreation = false,
  };
  this->buffer = context_create_buffer(context, &buffer_desc);

  context_set_buffer_data(context, this->buffer, buffer_size, buffer,
                          buffer_size);
}

static void buffer_dawn_destroy(buffer_dawn_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->buffer)
  memset(this, 0, sizeof(*this));
}

static WGPUBuffer buffer_dawn_get_buffer(buffer_dawn_t* this)
{
  return this->buffer;
}

static int32_t buffer_dawn_get_total_components(buffer_dawn_t* this)
{
  return this->total_components;
}

static uint32_t buffer_dawn_get_stride(buffer_dawn_t* this)
{
  return this->stride;
}

static void* buffer_dawn_get_offset(buffer_dawn_t* this)
{
  return this->offset;
}

static WGPUBufferUsage buffer_dawn_get_usage_bit(buffer_dawn_t* this)
{
  return this->usage;
}

static int32_t buffer_dawn_get_data_size(buffer_dawn_t* this)
{
  return this->size;
}

/* -------------------------------------------------------------------------- *
 * Buffer Manager - Implements buffer pool to manage buffer allocation and
 * recycle.
 * -------------------------------------------------------------------------- */

#define BUFFER_POOL_MAX_SIZE 409600000ull
#define BUFFER_MAX_COUNT 10ull
#define BUFFER_PER_ALLOCATE_SIZE (BUFFER_POOL_MAX_SIZE / BUFFER_MAX_COUNT)

typedef struct {
  size_t head;
  size_t tail;
  size_t size;

  void* buffer_manager;
  void* context;
  WGPUBuffer buf;
  void* mapped_data;
  void* pixels;
} ring_buffer_t;

sc_array_def(ring_buffer_t*, ring_buffer);
sc_queue_def(ring_buffer_t*, ring_buffer);

typedef struct {
  wgpu_context_t* wgpu_context;
  struct sc_queue_ring_buffer mapped_buffer_list;
  struct sc_array_ring_buffer enqueued_buffer_list;
  size_t buffer_pool_size;
  size_t used_size;
  size_t count;
  WGPUCommandEncoder encoder;
  void* context;
  bool sync;
} buffer_manager_t;

static void context_wait_a_bit(void* this);

static size_t ring_buffer_get_size(ring_buffer_t* this)
{
  return this->size;
}

static size_t ring_buffer_get_available_size(ring_buffer_t* this)
{
  return this->size - this->tail;
}

static bool ring_buffer_push(ring_buffer_t* this, WGPUCommandEncoder encoder,
                             WGPUBuffer dest_buffer, size_t src_offset,
                             size_t dest_offset, void* pixels, size_t size)
{
  memcpy(((unsigned char*)this->pixels) + src_offset, pixels, size);
  wgpuCommandEncoderCopyBufferToBuffer(encoder, this->buf, src_offset,
                                       dest_buffer, dest_offset, size);
  return true;
}

/* Reset current buffer and reuse the buffer. */
static bool ring_buffer_reset(ring_buffer_t* this, size_t size)
{
  if (size > this->size) {
    return false;
  }

  this->head = 0;
  this->tail = 0;

  WGPUBufferDescriptor buffer_desc = {
    .label            = "Ring buffer",
    .usage            = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopyDst,
    .size             = this->size,
    .mappedAtCreation = true,
  };
  this->buf = context_create_buffer(this->context, &buffer_desc);
  ASSERT(this->buf);
  this->pixels = wgpuBufferGetMappedRange(this->buf, 0, this->size);

  return true;
}

static void ring_buffer_create(ring_buffer_t* this,
                               buffer_manager_t* buffer_manager, size_t size)
{
  this->head = 0;
  this->tail = size;
  this->size = size;

  this->buffer_manager = buffer_manager;
  this->context        = buffer_manager->context;
  this->mapped_data    = NULL;
  this->pixels         = NULL;

  ring_buffer_reset(this, size);
}

static void ring_buffer_map_callback(WGPUBufferMapAsyncStatus status,
                                     void* user_data)
{
  if (status == WGPUBufferMapAsyncStatus_Success) {
    ring_buffer_t* ring_buffer = (ring_buffer_t*)user_data;
    ring_buffer->mapped_data   = (uint64_t*)wgpuBufferGetMappedRange(
      ring_buffer->buf, 0, ring_buffer->size);
    ASSERT(ring_buffer->mapped_data);

    sc_queue_add_last(
      &((buffer_manager_t*)ring_buffer->buffer_manager)->mapped_buffer_list,
      ring_buffer);
  }
}

static void ring_buffer_flush(ring_buffer_t* this)
{
  this->head = 0;
  this->tail = 0;

  wgpuBufferUnmap(this->buf);
}

static void ring_buffer_destroy(ring_buffer_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->buf);
}

static void ring_buffer_re_map(ring_buffer_t* this)
{
  wgpuBufferMapAsync(this->buf, WGPUMapMode_Write, 0, 0,
                     ring_buffer_map_callback, this);
}

/* Allocate size in a ring_buffer_t, return offset of the buffer */
static size_t ring_buffer_allocate(ring_buffer_t* this, size_t size)
{
  this->tail += size;
  ASSERT(this->tail < this->size);

  return this->tail - size;
}

static size_t buffer_manager_find(buffer_manager_t* this,
                                  ring_buffer_t* ring_buffer);

static void buffer_manager_init_defaults(buffer_manager_t* this)
{
  memset(this, 0, sizeof(*this));

  this->buffer_pool_size = BUFFER_POOL_MAX_SIZE;
  this->used_size        = 0;
  this->count            = 0;
}

static void buffer_manager_create(buffer_manager_t* this,
                                  wgpu_context_t* wgpu_context, bool sync)
{
  buffer_manager_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->sync         = sync;

  this->encoder
    = wgpuDeviceCreateCommandEncoder(this->wgpu_context->device, NULL);

  sc_queue_init(&this->mapped_buffer_list);
  sc_array_init(&this->enqueued_buffer_list);
}

static void buffer_manager_destroy_buffer_pool(buffer_manager_t* this)
{
  if (!this->sync) {
    return;
  }

  for (size_t i = 0; i < sc_array_size(&this->enqueued_buffer_list); i++) {
    ring_buffer_destroy(this->enqueued_buffer_list.elems[i]);
  }
  sc_array_clear(&this->enqueued_buffer_list);
}

static void buffer_manager_destroy(buffer_manager_t* this)
{
  buffer_manager_destroy_buffer_pool(this);
  WGPU_RELEASE_RESOURCE(CommandEncoder, this->encoder)

  sc_queue_term(&this->mapped_buffer_list);
  sc_array_term(&this->enqueued_buffer_list);
}

static size_t buffer_manager_get_size(buffer_manager_t* this)
{
  return this->buffer_pool_size;
}

static bool buffer_manager_reset_buffer(buffer_manager_t* this,
                                        ring_buffer_t* ring_buffer, size_t size)
{
  const size_t index = buffer_manager_find(this, ring_buffer);

  if (index >= sc_array_size(&this->enqueued_buffer_list)) {
    return false;
  }

  const size_t old_size = ring_buffer_get_size(ring_buffer);

  const bool result = ring_buffer_reset(ring_buffer, size);
  // If the size is larger than the ring buffer size, reset fails and the ring
  // buffer retains.
  // If the size is equal or smaller than the ring buffer size, reset success
  // and the used size need to be updated.
  if (!result) {
    return false;
  }
  else {
    this->used_size = this->used_size - old_size + size;
  }

  return true;
}

static bool buffer_manager_destroy_buffer(buffer_manager_t* this,
                                          ring_buffer_t* ring_buffer)
{
  const size_t index = buffer_manager_find(this, ring_buffer);

  if (index >= sc_array_size(&this->enqueued_buffer_list)) {
    return false;
  }

  this->used_size -= ring_buffer_get_size(ring_buffer);
  ring_buffer_destroy(ring_buffer);
  sc_array_del(&this->enqueued_buffer_list, index);

  return true;
}

static size_t buffer_manager_find(buffer_manager_t* this,
                                  ring_buffer_t* ring_buffer)
{
  size_t index = 0;
  for (index = 0; index < sc_array_size(&this->enqueued_buffer_list); index++) {
    if (this->enqueued_buffer_list.elems[index] == ring_buffer) {
      break;
    }
  }
  return index;
}

/* Flush copy commands in buffer pool */
static void buffer_manager_flush(buffer_manager_t* this)
{
  // The front buffer in MappedBufferList will be remap after submit, pop the
  // buffer from MappedBufferList.
  if (sc_array_size(&this->enqueued_buffer_list) == 0
      && (sc_array_last(&this->enqueued_buffer_list))
           == (sc_queue_peek_first(&this->mapped_buffer_list))) {
    sc_queue_del_first(&this->mapped_buffer_list);
  }

  ring_buffer_t* buffer = NULL;
  sc_array_foreach(&this->enqueued_buffer_list, buffer)
  {
    ring_buffer_flush(buffer);
  }

  WGPUCommandBuffer copy = wgpuCommandEncoderFinish(this->encoder, NULL);
  ASSERT(copy != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, this->encoder)
  wgpuQueueSubmit(this->wgpu_context->queue, 1, &copy);

  /* Async function */
  if (!this->sync) {
    sc_array_foreach(&this->enqueued_buffer_list, buffer)
    {
      ring_buffer_re_map(buffer);
    }
  }
  else {
    /* All buffers are used once in buffer sync mode. */
    for (size_t i = 0; i < sc_array_size(&this->enqueued_buffer_list); i++) {
      free(this->enqueued_buffer_list.elems[i]);
    }
    this->used_size = 0;
  }

  sc_array_clear(&this->enqueued_buffer_list);
  this->encoder
    = wgpuDeviceCreateCommandEncoder(this->wgpu_context->device, NULL);
}

/* Allocate new buffer from buffer pool. */
static ring_buffer_t* buffer_manager_allocate(buffer_manager_t* this,
                                              size_t size, size_t* offset)
{
  // If update data by sync method, create new buffer to upload every frame.
  // If updaye data by async method, get new buffer from pool if available. If
  // no available buffer and size is enough in the buffer pool, create a new
  // buffer. If size reach the limit of the buffer pool, force wait for the
  // buffer on mapping. Get the last one and check if the ring buffer is full.
  // If the buffer can hold extra size space, use the last one directly.
  // TODO(yizhou): Return nullptr if size reach the limit or no available
  // buffer, this means small bubbles in some of the ring buffers and we haven't
  // deal with the problem now.

  ring_buffer_t* ring_buffer = NULL;
  size_t cur_offset          = 0ll;
  if (this->sync) {
    /* Upper limit */
    if (this->used_size + size > this->buffer_pool_size) {
      return NULL;
    }

    ring_buffer = malloc(sizeof(ring_buffer_t));
    ring_buffer_create(ring_buffer, this, size);
    sc_array_add(&this->enqueued_buffer_list, ring_buffer);
  }
  else {
    /* Buffer mapping async */
    while (!sc_queue_empty(&this->mapped_buffer_list)) {
      ring_buffer = sc_queue_peek_first(&this->mapped_buffer_list);
      if (ring_buffer_get_available_size(ring_buffer) < size) {
        sc_queue_del_first(&this->mapped_buffer_list);
        ring_buffer = NULL;
      }
      else {
        break;
      }
    }

    if (ring_buffer == NULL) {
      if (this->count < BUFFER_MAX_COUNT) {
        this->used_size += size;
        ring_buffer = malloc(sizeof(ring_buffer_t));
        ring_buffer_create(ring_buffer, this, BUFFER_PER_ALLOCATE_SIZE);
        sc_queue_add_last(&this->mapped_buffer_list, ring_buffer);
        this->count++;
      }
      else if (sc_queue_size(&this->mapped_buffer_list)
                 + sc_array_size(&this->enqueued_buffer_list)
               < this->count) {
        /* Force wait for the buffer remapping */
        while (sc_queue_empty(&this->mapped_buffer_list)) {
          context_wait_a_bit(this->context);
        }

        ring_buffer = sc_queue_peek_first(&this->mapped_buffer_list);
        if (ring_buffer_get_available_size(ring_buffer) < size) {
          sc_queue_del_first(&this->mapped_buffer_list);
          ring_buffer = NULL;
        }
      }
      else { /* Upper limit */
        return NULL;
      }
    }

    if (sc_array_size(&this->enqueued_buffer_list) == 0
        && (sc_array_last(&this->enqueued_buffer_list)) != ring_buffer) {
      sc_array_add(&this->enqueued_buffer_list, ring_buffer);
    }

    /* allocate size in the ring buffer */
    cur_offset = ring_buffer_allocate(ring_buffer, size);
    *offset    = cur_offset;
  }

  return ring_buffer;
}

/* -------------------------------------------------------------------------- *
 * Resource Helper
 * -------------------------------------------------------------------------- */

static const char* aquarium_folder = "Aquarium";
static const char* models_folder   = "models";
static const char* shaders_folder  = "shaders";
static const char* textures_folder = "textures";

static const char* const sky_box_urls[6] = {
  "GlobeOuter_EM_positive_x.jpg", "GlobeOuter_EM_negative_x.jpg",
  "GlobeOuter_EM_positive_y.jpg", "GlobeOuter_EM_negative_y.jpg",
  "GlobeOuter_EM_positive_z.jpg", "GlobeOuter_EM_negative_z.jpg",
};

typedef struct {
  char image_path[STRMAX];
  char sky_box_paths[6][STRMAX];
  char program_path[STRMAX];
  char prop_placement_path[STRMAX];
  char model_path[STRMAX / 2];
} resource_helper_t;

static void resource_helper_init_defaults(resource_helper_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void resource_helper_create(resource_helper_t* this)
{
  resource_helper_init_defaults(this);

  /* Model path */
  snprintf(this->model_path, sizeof(this->model_path), "%s%s%s%s",
           models_folder, slash, aquarium_folder, slash);

  /* Placement path */
  snprintf(this->prop_placement_path, sizeof(this->prop_placement_path), "%s%s",
           this->model_path, "PropPlacement.js");

  /* Image path */
  snprintf(this->image_path, sizeof(this->image_path), "%s%s%s%s",
           textures_folder, slash, aquarium_folder, slash);

  /* Program path */
  snprintf(this->program_path, sizeof(this->program_path), "%s%s%s%s",
           shaders_folder, slash, aquarium_folder, slash);

  /* Skybox urls */
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(sky_box_urls); ++i) {
    snprintf(this->sky_box_paths[i], sizeof(this->sky_box_paths[i]), "%s%s",
             this->image_path, sky_box_urls[i]);
  }
}

static void resource_helper_get_sky_box_urls(const resource_helper_t* this,
                                             const char (*dst)[6][STRMAX])
{
  for (uint8_t i = 0; i < 6; ++i) {
    snprintf((char*)dst[i], sizeof(*dst[i]), "%s", this->sky_box_paths[i]);
  }
}

static const char*
resource_helper_get_prop_placement_path(const resource_helper_t* this)
{
  return this->prop_placement_path;
}

static const char* resource_helper_get_image_path(const resource_helper_t* this)
{
  return this->image_path;
}

static void resource_helper_get_model_path(const resource_helper_t* this,
                                           const char* model_name,
                                           char (*dst)[STRMAX])
{
  snprintf((char*)dst, sizeof(*dst), "%s%s.js", this->model_path, model_name);
}

static const char*
resource_helper_get_program_path(const resource_helper_t* this)
{
  return this->program_path;
}

/* -------------------------------------------------------------------------- *
 * Aquarium context - Defines the render context.
 * -------------------------------------------------------------------------- */

#define MAX_TEXTURE_COUNT 64u

sc_array_def(WGPUCommandBuffer, command_buffer);

typedef struct context_t {
  wgpu_context_t* wgpu_context;
  WGPUDevice device;
  uint32_t client_width;
  uint32_t client_height;
  uint32_t pre_total_instance;
  uint32_t cur_total_instance;
  resource_helper_t resource_helper;
  uint32_t msaa_sample_count;
  struct sc_array_command_buffer command_buffers;
  struct {
    WGPUBindGroupLayout general;
    WGPUBindGroupLayout world;
    WGPUBindGroupLayout fish_per;
  } bind_group_layouts;
  struct {
    WGPUBindGroup general;
    WGPUBindGroup world;
    WGPUBindGroup fish_per;
  } bind_groups;
  WGPUBuffer fish_pers_buffer;
  WGPUBindGroup* bind_group_fish_pers;
  fish_per_t* fish_pers;
  WGPUTextureUsage swapchain_back_buffer_usage;
  bool is_swapchain_out_of_date;
  WGPUCommandEncoder command_encoder;
  WGPURenderPassEncoder render_pass;
  WGPURenderPassDescriptor render_pass_descriptor;
  struct {
    WGPUTextureView backbuffer;
    texture_t scene_render_target;
    texture_t scene_depth_stencil;
  } texture_views;
  WGPURenderPipeline pipeline;
  WGPUBindGroup bind_group;
  WGPUTextureFormat preferred_swap_chain_format;
  struct {
    WGPUBuffer light_world_position;
    WGPUBuffer light;
    WGPUBuffer fog;
  } uniform_buffers;
  bool enable_dynamic_buffer_offset;
  buffer_manager_t* buffer_manager;
} context_t;

sc_queue_def(behavior_t*, behavior);

typedef struct {
  wgpu_example_context_t* wgpu_example_context;
  wgpu_context_t* wgpu_context;
  context_t context;
  light_world_position_uniform_t light_world_position_uniform;
  world_uniforms_t world_uniforms;
  light_uniforms_t light_uniforms;
  fog_uniforms_t fog_uniforms;
  global_t g;
  int32_t fish_count[5];
  struct {
    char key[STRMAX];
    model_name_t value;
  } model_enum_map[MODELNAME_MODELMAX];
  struct {
    char key[STRMAX];
    texture_t value;
  } texture_map[MAX_TEXTURE_COUNT];
  uint32_t texture_count;
  WGPUShaderModule vertex_shaders[VERTEX_SHADER_MAX];
  WGPUShaderModule fragment_shaders[FRAGMENT_SHADER_MAX];
  struct {
    char key[STRMAX];
    program_t value;
  } program_map[FRAGMENT_SHADER_MAX];
  uint32_t program_count;
  void* aquarium_models[MODELNAME_MODELMAX];
  int32_t cur_fish_count;
  int32_t pre_fish_count;
  int32_t test_time;
  behavior_t fish_behaviors[FISH_BEHAVIOR_COUNT];
  struct sc_queue_behavior fish_behavior;
} aquarium_t;

/* Forward declarations context */
static void context_realloc_resource(context_t* this,
                                     uint32_t pre_total_instance,
                                     uint32_t cur_total_instance,
                                     bool enable_dynamic_buffer_offset);
static void context_destroy_fish_resource(context_t* this);

/* Forward declarations aquarium */
static int32_t aquarium_get_cur_fish_count(aquarium_t* this);
static int32_t aquarium_get_pre_fish_count(aquarium_t* this);

static void context_update_world_uniforms(context_t* this,
                                          aquarium_t* aquarium);

static void context_init_defaults(context_t* this)
{
  memset(this, 0, sizeof(*this));

  this->preferred_swap_chain_format = WGPUTextureFormat_RGBA8Unorm;
  this->msaa_sample_count           = 1;
}

static void context_create(context_t* this)
{
  context_init_defaults(this);

  resource_helper_create(&this->resource_helper);
}

static void context_detroy(context_t* this)
{
}

static bool context_initialize(context_t* this)
{
  return true;
}

static void context_set_window_size(context_t* this, uint32_t window_width,
                                    uint32_t window_height)
{
  if (window_width != 0) {
    this->client_width = window_width;
  }
  if (window_height != 0) {
    this->client_height = window_height;
  }
}

static uint32_t context_get_client_width(context_t* this)
{
  return this->client_width;
}

static uint32_t context_get_client_height(context_t* this)
{
  return this->client_height;
}

static void set_msaa_sample_count(context_t* this, uint32_t msaa_sample_count)
{
  this->msaa_sample_count = msaa_sample_count;
}

static texture_t context_create_texture(context_t* this, const char* name,
                                        const char* url)
{
  UNUSED_VAR(name);

  return wgpu_create_texture_from_file(this->wgpu_context, url, NULL);
}

static texture_t context_create_cubemap_texture(context_t* this,
                                                const char* name,
                                                const char (*urls)[6][STRMAX])
{
  UNUSED_VAR(name);

  return wgpu_create_texture_cubemap_from_files(
    this->wgpu_context, (const char**)urls,
    &(struct wgpu_texture_load_options_t){
      .flip_y = false,
    });
}

static WGPUSampler
context_create_sampler(context_t* this, WGPUSamplerDescriptor const* descriptor)
{
  return wgpuDeviceCreateSampler(this->device, descriptor);
}

static WGPUBuffer context_create_buffer_from_data(context_t* this,
                                                  const void* data,
                                                  uint32_t size,
                                                  uint32_t max_size,
                                                  WGPUBufferUsage usage)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPUBufferDescriptor buffer_desc = {
    .usage            = usage | WGPUBufferUsage_CopyDst,
    .size             = max_size,
    .mappedAtCreation = false,
  };
  WGPUBuffer buffer = context_create_buffer(wgpu_context, &buffer_desc);

  context_set_buffer_data(this, buffer, max_size, data, size);
  ASSERT(buffer != NULL);
  return buffer;
}

static program_t* context_create_program(context_t* this, const char* vs_id,
                                         const char* fs_id)
{
  program_t* program = (program_t*)malloc(sizeof(program_t));
  program_create(program, this, vs_id, fs_id);

  return program;
}

static WGPUImageCopyBuffer
context_create_image_copy_buffer(context_t* this, WGPUBuffer buffer,
                                 uint32_t offset, uint32_t bytes_per_row,
                                 uint32_t rows_per_image)
{
  UNUSED_VAR(this);

  WGPUImageCopyBuffer image_copy_buffer_desc = {
    .layout.offset       = offset,
    .layout.bytesPerRow  = bytes_per_row,
    .layout.rowsPerImage = rows_per_image,
    .buffer              = buffer,
  };

  return image_copy_buffer_desc;
}

static WGPUImageCopyTexture
context_create_image_copy_texture(context_t* this, WGPUTexture texture,
                                  uint32_t level, WGPUOrigin3D origin)
{
  UNUSED_VAR(this);

  WGPUImageCopyTexture image_copy_texture_desc = {
    .texture  = texture,
    .mipLevel = level,
    .origin   = origin,
  };

  return image_copy_texture_desc;
}

static WGPUCommandBuffer context_copy_buffer_to_texture(
  context_t* this, WGPUImageCopyBuffer const* image_copy_buffer,
  WGPUImageCopyTexture const* image_copy_texture, WGPUExtent3D const* ext_3d)
{
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(this->device, NULL);
  wgpuCommandEncoderCopyBufferToTexture(encoder, image_copy_buffer,
                                        image_copy_texture, ext_3d);
  WGPUCommandBuffer copy = wgpuCommandEncoderFinish(encoder, NULL);
  ASSERT(copy != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)

  return copy;
}

static WGPUCommandBuffer
context_copy_buffer_to_buffer(context_t* this, WGPUBuffer src_buffer,
                              uint64_t src_offset, WGPUBuffer dest_buffer,
                              uint64_t dest_offset, uint64_t size)
{
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(this->device, NULL);
  wgpuCommandEncoderCopyBufferToBuffer(encoder, src_buffer, src_offset,
                                       dest_buffer, dest_offset, size);
  WGPUCommandBuffer copy = wgpuCommandEncoderFinish(encoder, NULL);
  ASSERT(copy != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)

  return copy;
}

static wgpu_shader_t context_create_shader_module(void* context,
                                                  WGPUShaderStage shader_stage,
                                                  const char* shader_code_wgsl)
{
  UNUSED_VAR(shader_stage);

  wgpu_shader_t shader = wgpu_shader_create(
    ((context_t*)context)->wgpu_context, &(wgpu_shader_desc_t){
                                           /* Shader WGSL */
                                           .wgsl_code.source = shader_code_wgsl,
                                           .entry            = "main",
                                         });
  ASSERT(shader.module);

  return shader;
}

static WGPUBindGroupLayout context_make_bind_group_layout(
  context_t* this, WGPUBindGroupLayoutEntry const* bind_group_layout_entries,
  uint32_t bind_group_layout_entry_count)
{
  WGPUBindGroupLayout bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    this->device, &(WGPUBindGroupLayoutDescriptor){
                    .label      = "Bind group layout",
                    .entryCount = bind_group_layout_entry_count,
                    .entries    = bind_group_layout_entries,
                  });
  ASSERT(bind_group_layout != NULL);
  return bind_group_layout;
}

static WGPUPipelineLayout context_make_basic_pipeline_layout(
  context_t* this, WGPUBindGroupLayout const* bind_group_layouts,
  uint32_t bind_group_layout_count)
{
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    this->device, &(WGPUPipelineLayoutDescriptor){
                    .label                = "Basic pipeline layout",
                    .bindGroupLayoutCount = bind_group_layout_count,
                    .bindGroupLayouts     = bind_group_layouts,
                  });
  ASSERT(pipeline_layout != NULL);
  return pipeline_layout;
}

static WGPURenderPipeline context_create_render_pipeline(
  context_t* this, WGPUPipelineLayout pipeline_layout, program_t* program,
  WGPUVertexState const* vertex_state, bool enable_blend)
{
  WGPUShaderModule* fs_module = &program->fs_module.module;

  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  WGPUStencilFaceState stencil_face_state = {
    .compare     = WGPUCompareFunction_Always,
    .failOp      = WGPUStencilOperation_Keep,
    .depthFailOp = WGPUStencilOperation_Keep,
    .passOp      = WGPUStencilOperation_Keep,
  };

  WGPUDepthStencilState depth_stencil_state = {
    .format              = WGPUTextureFormat_Depth24PlusStencil8,
    .depthWriteEnabled   = true,
    .depthCompare        = WGPUCompareFunction_Less,
    .stencilFront        = stencil_face_state,
    .stencilBack         = stencil_face_state,
    .stencilReadMask     = 0xffffffff,
    .stencilWriteMask    = 0xffffffff,
    .depthBias           = 0,
    .depthBiasSlopeScale = 0.0f,
    .depthBiasClamp      = 0.0f,
  };

  WGPUMultisampleState multisample_state = {
    .count                  = aquarium_settings.msaa_sample_count,
    .mask                   = 0xffffffff,
    .alphaToCoverageEnabled = false,
  };

  WGPUBlendComponent blend_component = {
    .operation = WGPUBlendOperation_Add,
  };
  if (enable_blend) {
    blend_component.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend_component.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
  }
  else {
    blend_component.srcFactor = WGPUBlendFactor_One;
    blend_component.dstFactor = WGPUBlendFactor_Zero;
  }

  WGPUBlendState blend_state = {
    .color = blend_component,
    .alpha = blend_component,
  };

  WGPUColorTargetState color_target_state = {
    .format    = this->preferred_swap_chain_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUFragmentState fragment_state = {
    .module      = *fs_module,
    .entryPoint  = "main",
    .targetCount = 1,
    .targets     = &color_target_state,
  };

  WGPURenderPipelineDescriptor pipeline_descriptor = {
    .label        = "Render pipeline",
    .layout       = pipeline_layout,
    .vertex       = *vertex_state,
    .primitive    = primitive_state,
    .depthStencil = &depth_stencil_state,
    .multisample  = multisample_state,
    .fragment     = &fragment_state,
  };

  WGPURenderPipeline pipeline
    = wgpuDeviceCreateRenderPipeline(this->device, &pipeline_descriptor);
  ASSERT(pipeline != NULL);

  return pipeline;
}

static texture_t context_create_multisampled_render_target_view(context_t* this)
{
  texture_t texture = {0};

  WGPUTextureDescriptor texture_desc = {
    .dimension               = WGPUTextureDimension_2D,
    .size.width              = this->client_width,
    .size.height             = this->client_height,
    .size.depthOrArrayLayers = 1,
    .sampleCount             = this->msaa_sample_count,
    .format                  = this->preferred_swap_chain_format,
    .mipLevelCount           = 1,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  texture.texture = wgpuDeviceCreateTexture(this->device, &texture_desc);
  ASSERT(texture.texture != NULL);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  texture.view = wgpuTextureCreateView(texture.texture, &texture_view_dec);
  ASSERT(texture.view != NULL);

  return texture;
}

static texture_t context_create_depth_stencil_view(context_t* this)
{
  texture_t texture = {0};

  WGPUTextureDescriptor texture_desc = {
    .dimension               = WGPUTextureDimension_2D,
    .size.width              = this->client_width,
    .size.height             = this->client_height,
    .size.depthOrArrayLayers = 1,
    .sampleCount             = this->msaa_sample_count,
    .format                  = WGPUTextureFormat_Depth24PlusStencil8,
    .mipLevelCount           = 1,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  texture.texture = wgpuDeviceCreateTexture(this->device, &texture_desc);
  ASSERT(texture.texture != NULL);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  texture.view = wgpuTextureCreateView(texture.texture, &texture_view_dec);
  ASSERT(texture.view != NULL);

  return texture;
}

static WGPUBuffer context_create_buffer(void* this,
                                        WGPUBufferDescriptor const* descriptor)
{
  context_t* _this = (context_t*)this;

  return wgpuDeviceCreateBuffer(_this->device, descriptor);
}

static void context_set_buffer_data(void* this, WGPUBuffer buffer,
                                    uint32_t buffer_size, const void* data,
                                    uint32_t data_size)
{
  context_t* _this = (context_t*)this;

  wgpu_context_t* wgpu_context = _this->wgpu_context;

  WGPUBufferDescriptor buffer_desc = {
    .usage            = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc,
    .size             = buffer_size,
    .mappedAtCreation = true,
  };
  WGPUBuffer staging = context_create_buffer(wgpu_context, &buffer_desc);
  ASSERT(staging);
  void* mapping = wgpuBufferGetMappedRange(staging, 0, buffer_size);
  ASSERT(mapping);
  memcpy(mapping, data, data_size);
  wgpuBufferUnmap(staging);

  WGPUCommandBuffer command
    = context_copy_buffer_to_buffer(this, staging, 0, buffer, 0, buffer_size);
  ASSERT(command != NULL);
  WGPU_RELEASE_RESOURCE(Buffer, staging);
  sc_array_add(&_this->command_buffers, command);
}

static WGPUBindGroup
context_make_bind_group(context_t* this, WGPUBindGroupLayout layout,
                        WGPUBindGroupEntry const* bind_group_entries,
                        uint32_t bind_group_entry_count)
{
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    this->device, &(WGPUBindGroupDescriptor){
                    .layout     = layout,
                    .entryCount = bind_group_entry_count,
                    .entries    = bind_group_entries,
                  });
  ASSERT(bind_group != NULL);
  return bind_group;
}

static void context_init_general_resources(context_t* this,
                                           aquarium_t* aquarium)
{
  /* Initialize general uniform buffers */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    this->bind_group_layouts.general = context_make_bind_group_layout(
      this, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  this->uniform_buffers.light = context_create_buffer_from_data(
    this, &aquarium->light_uniforms, sizeof(aquarium->light_uniforms),
    sizeof(aquarium->light_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  this->uniform_buffers.fog = context_create_buffer_from_data(
    this, &aquarium->fog_uniforms, sizeof(aquarium->fog_uniforms),
    sizeof(aquarium->fog_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.light,
        .offset  = 0,
        .size    = sizeof(aquarium->light_uniforms),
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->uniform_buffers.fog,
        .offset  = 0,
        .size    = sizeof(aquarium->fog_uniforms),
      },
    };
    this->bind_groups.general
      = context_make_bind_group(this, this->bind_group_layouts.general,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  context_set_buffer_data(this, this->uniform_buffers.light,
                          sizeof(light_uniforms_t), &aquarium->light_uniforms,
                          sizeof(light_uniforms_t));
  context_set_buffer_data(this, this->uniform_buffers.fog,
                          sizeof(fog_uniforms_t), &aquarium->fog_uniforms,
                          sizeof(fog_uniforms_t));

  /* Initialize world uniform buffers */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    this->bind_group_layouts.world = context_make_bind_group_layout(
      this, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  this->uniform_buffers.light_world_position = context_create_buffer_from_data(
    this, &aquarium->light_world_position_uniform,
    sizeof(aquarium->light_world_position_uniform),
    calc_constant_buffer_byte_size(
      sizeof(aquarium->light_world_position_uniform)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.light_world_position,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(
          sizeof(aquarium->light_world_position_uniform)),
      },
    };
    this->bind_groups.world
      = context_make_bind_group(this, this->bind_group_layouts.world,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  bool enable_dynamic_buffer_offset
    = aquarium_settings.enable_dynamic_buffer_offset;

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {0};
    if (enable_dynamic_buffer_offset) {
      bgl_entries[0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      };
    }
    else {
      bgl_entries[0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      };
    }
    this->bind_group_layouts.fish_per = context_make_bind_group_layout(
      this, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  context_realloc_resource(this, aquarium_get_pre_fish_count(aquarium),
                           aquarium_get_cur_fish_count(aquarium),
                           enable_dynamic_buffer_offset);
}

static void context_update_world_uniforms(context_t* this, aquarium_t* aquarium)
{
  context_update_buffer_data(
    this->wgpu_context, this->uniform_buffers.light_world_position,
    calc_constant_buffer_byte_size(sizeof(light_world_position_uniform_t)),
    &aquarium->light_world_position_uniform,
    sizeof(light_world_position_uniform_t));
}

static resource_helper_t* context_get_resource_helper(context_t* this)
{
  return &this->resource_helper;
}

static buffer_dawn_t* context_create_buffer_f32(context_t* this,
                                                int32_t num_components,
                                                float* buf, size_t buf_count,
                                                bool is_index)
{
  buffer_dawn_t* buffer = malloc(sizeof(buffer_dawn_t));
  buffer_dawn_create_f32(buffer, this, (int)buf_count, num_components, buf,
                         is_index);

  return buffer;
}

static buffer_dawn_t*
context_create_buffer_uint16(context_t* this, int32_t num_components,
                             uint16_t* buf, size_t buf_count,
                             uint64_t buffer_count, bool is_index)
{
  buffer_dawn_t* buffer = malloc(sizeof(buffer_dawn_t));
  buffer_dawn_create_uint16(buffer, this, (int)buf_count, num_components, buf,
                            buffer_count, is_index);

  return buffer;
}

static void context_flush(context_t* this)
{
  /* Submit to the queue */
  wgpuQueueSubmit(this->wgpu_context->queue,
                  sc_array_size(&this->command_buffers),
                  this->command_buffers.elems);

  /* Release command buffer */
  for (size_t i = 0; i < sc_array_size(&this->command_buffers); ++i) {
    WGPU_RELEASE_RESOURCE(CommandBuffer, this->command_buffers.elems[i])
  }
  sc_array_clear(&this->command_buffers);
}

/* Submit commands of the frame */
static void context_do_flush(context_t* this)
{
  /* End render pass */
  wgpuRenderPassEncoderEnd(this->render_pass);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, this->render_pass)

  buffer_manager_flush(this->buffer_manager);

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(this->command_encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, this->command_encoder)
  sc_array_add(&this->command_buffers, cmd);

  context_flush(this);

  wgpuSwapChainPresent(this->wgpu_context->swap_chain.instance);
}

static void context_pre_frame(context_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  if (this->is_swapchain_out_of_date) {
    this->client_width  = wgpu_context->surface.width;
    this->client_height = wgpu_context->surface.height;
    if (this->msaa_sample_count > 1) {
      this->texture_views.scene_render_target
        = context_create_multisampled_render_target_view(this);
    }
    this->texture_views.scene_depth_stencil
      = context_create_depth_stencil_view(this);
    // mSwapchain.Configure(mPreferredSwapChainFormat,
    // kSwapchainBackBufferUsage,
    //                     mClientWidth, mClientHeight);
    this->is_swapchain_out_of_date = false;
  }

  this->command_encoder = wgpuDeviceCreateCommandEncoder(this->device, NULL);
  this->texture_views.backbuffer
    = wgpuSwapChainGetCurrentTextureView(wgpu_context->swap_chain.instance);

  WGPURenderPassColorAttachment color_attachment = {0};
  if (this->msaa_sample_count > 1) {
    // If MSAA is enabled, we render to a multisampled texture and then resolve
    // to the backbuffer
    color_attachment.view = this->texture_views.scene_render_target.view;
    color_attachment.resolveTarget = this->texture_views.backbuffer;
    color_attachment.loadOp        = WGPULoadOp_Clear;
    color_attachment.storeOp       = WGPUStoreOp_Store;
    color_attachment.clearValue    = (WGPUColor){0.f, 0.8f, 1.f, 0.f};
  }
  else {
    /* When MSAA is off, we render directly to the backbuffer */
    color_attachment.view       = this->texture_views.backbuffer;
    color_attachment.loadOp     = WGPULoadOp_Clear;
    color_attachment.storeOp    = WGPUStoreOp_Store;
    color_attachment.clearValue = (WGPUColor){0.f, 0.8f, 1.f, 0.f};
  }

  WGPURenderPassDepthStencilAttachment depth_stencil_attachment = {0};
  depth_stencil_attachment.view = this->texture_views.scene_depth_stencil.view;
  depth_stencil_attachment.depthLoadOp    = WGPULoadOp_Clear;
  depth_stencil_attachment.depthStoreOp   = WGPUStoreOp_Store;
  depth_stencil_attachment.stencilLoadOp  = WGPULoadOp_Clear;
  depth_stencil_attachment.stencilStoreOp = WGPUStoreOp_Store;

  this->render_pass_descriptor.colorAttachmentCount = 1;
  this->render_pass_descriptor.colorAttachments     = &color_attachment;
  this->render_pass_descriptor.depthStencilAttachment
    = &depth_stencil_attachment;

  this->render_pass = wgpuCommandEncoderBeginRenderPass(
    this->command_encoder, &this->render_pass_descriptor);
}

static void context_realloc_resource(context_t* this,
                                     uint32_t pre_total_instance,
                                     uint32_t cur_total_instance,
                                     bool enable_dynamic_buffer_offset)
{
  this->pre_total_instance           = pre_total_instance;
  this->cur_total_instance           = cur_total_instance;
  this->enable_dynamic_buffer_offset = enable_dynamic_buffer_offset;

  if (cur_total_instance == 0) {
    return;
  }

  /* If current fish number > pre fish number, allocate a new bigger buffer. */
  /* If current fish number <= prefish number, do not allocate a new one. */
  if (pre_total_instance >= cur_total_instance) {
    return;
  }

  context_destroy_fish_resource(this);

  this->fish_pers = malloc(sizeof(fish_per_t) * cur_total_instance);

  if (enable_dynamic_buffer_offset) {
    this->bind_group_fish_pers = malloc(sizeof(WGPUBindGroup) * 1);
  }
  else {
    this->bind_group_fish_pers
      = malloc(sizeof(WGPUBindGroup) * cur_total_instance);
  }

  WGPUBufferDescriptor buffer_desc = {
    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
    .size
    = calc_constant_buffer_byte_size(sizeof(fish_per_t) * cur_total_instance),
    .mappedAtCreation = false,
  };
  this->fish_pers_buffer
    = context_create_buffer(this->wgpu_context, &buffer_desc);

  if (this->enable_dynamic_buffer_offset) {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->fish_pers_buffer,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(fish_per_t)),
      },
    };
    this->bind_group_fish_pers[0]
      = context_make_bind_group(this, this->bind_group_layouts.fish_per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }
  else {
    for (uint32_t i = 0; i < cur_total_instance; ++i) {
      WGPUBindGroupEntry bg_entries[1] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .buffer  = this->fish_pers_buffer,
          .offset  = calc_constant_buffer_byte_size(sizeof(fish_per_t) * i),
          .size    = calc_constant_buffer_byte_size(sizeof(fish_per_t)),
        },
      };
      this->bind_group_fish_pers[i]
        = context_make_bind_group(this, this->bind_group_layouts.fish_per,
                                  bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
    }
  }
}

static void context_wait_a_bit(void* this)
{
  wgpuDeviceTick(((context_t*)this)->device);

  usleep(100);
}

static WGPUCommandEncoder context_create_command_encoder(context_t* this)
{
  return wgpuDeviceCreateCommandEncoder(this->device, NULL);
}

static void context_update_all_fish_data(context_t* this)
{
  size_t size = calc_constant_buffer_byte_size(sizeof(fish_per_t)
                                               * this->cur_total_instance);
  context_update_buffer_data(this, this->fish_pers_buffer, size,
                             this->fish_pers,
                             sizeof(fish_per_t) * this->cur_total_instance);
}

static void context_update_buffer_data(void* this, WGPUBuffer buffer,
                                       size_t buffer_size, void* data,
                                       size_t data_size)
{
  context_t* _this = (context_t*)this;

  size_t offset = 0;
  ring_buffer_t* ring_buffer
    = buffer_manager_allocate(_this->buffer_manager, buffer_size, &offset);

  if (ring_buffer == NULL) {
    log_error("Memory upper limit.");
    return;
  }

  ring_buffer_push(ring_buffer, _this->buffer_manager->encoder, buffer, offset,
                   0, data, data_size);
}

static void context_destroy_fish_resource(context_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->fish_pers_buffer);

  if (this->fish_pers != NULL) {
    free(this->fish_pers);
    this->fish_pers = NULL;
  }
  if (aquarium_settings.enable_dynamic_buffer_offset) {
    if (this->bind_group_fish_pers != NULL) {
      if (this->bind_group_fish_pers[0] != NULL) {
        WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group_fish_pers[0]);
      }
    }
  }
  else {
    if (this->bind_group_fish_pers != NULL) {
      for (uint32_t i = 0; i < this->pre_total_instance; ++i) {
        if (this->bind_group_fish_pers[i] != NULL) {
          WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group_fish_pers[i]);
        }
      }
    }
  }

  free(this->bind_group_fish_pers);
  this->bind_group_fish_pers = NULL;

  buffer_manager_destroy_buffer_pool(this->buffer_manager);
}

static void context_begin_render_pass(context_t* this)
{
  UNUSED_VAR(this);
}

static void context_destory_fish_resource(context_t* this)
{
  buffer_manager_destroy_buffer_pool(this->buffer_manager);
}

/* -------------------------------------------------------------------------- *
 * Aquarium - Main class functions.
 * -------------------------------------------------------------------------- */

static void aquarium_reset_fps_time(aquarium_t* this);
static void aquarium_load_resource(aquarium_t* this);
static float aquarium_get_elapsed_time(aquarium_t* this);
static void aquarium_setup_model_enum_map(aquarium_t* this);
static void aquarium_calculate_fish_count(aquarium_t* this);
static int32_t aquarium_load_placement(aquarium_t* this);
static int32_t aquarium_load_models(aquarium_t* this);
static int32_t aquarium_load_fish_scenario(aquarium_t* this);
static int32_t aquarium_load_model(aquarium_t* this,
                                   const g_scene_info_t* info);

static uint64_t
get_current_time_point_ms(wgpu_example_context_t* wgpu_example_context)
{
  return wgpu_example_context->frame.timestamp_millis;
}

static void aquarium_init_defaults(aquarium_t* this)
{
  memset(this, 0, sizeof(*this));

  this->cur_fish_count = 500;
  this->pre_fish_count = 0;
  this->test_time      = INT_MAX;

  this->g.then      = 0.0f;
  this->g.mclock    = 0.0f;
  this->g.eye_clock = 0.0f;
  this->g.alpha     = 1.0f;

  this->light_uniforms.light_color[0] = 1.0f;
  this->light_uniforms.light_color[1] = 1.0f;
  this->light_uniforms.light_color[2] = 1.0f;
  this->light_uniforms.light_color[3] = 1.0f;

  this->light_uniforms.specular[0] = 1.0f;
  this->light_uniforms.specular[1] = 1.0f;
  this->light_uniforms.specular[2] = 1.0f;
  this->light_uniforms.specular[3] = 1.0f;

  this->fog_uniforms.fog_color[0] = g_settings.fog_red;
  this->fog_uniforms.fog_color[1] = g_settings.fog_green;
  this->fog_uniforms.fog_color[2] = g_settings.fog_blue;
  this->fog_uniforms.fog_color[3] = 1.0f;

  this->fog_uniforms.fog_power  = g_settings.fog_power;
  this->fog_uniforms.fog_mult   = g_settings.fog_mult;
  this->fog_uniforms.fog_offset = g_settings.fog_offset;

  this->light_uniforms.ambient[0] = g_settings.ambient_red;
  this->light_uniforms.ambient[1] = g_settings.ambient_green;
  this->light_uniforms.ambient[2] = g_settings.ambient_blue;
  this->light_uniforms.ambient[3] = 0.0f;

  memset(this->fish_count, 0, sizeof(this->fish_count));
}

static void aquarium_create(aquarium_t* this)
{
  aquarium_init_defaults(this);

  this->g.then = get_current_time_point_ms(this->wgpu_example_context);
}

/**
 * @brief Looks up the index of a key in the texture map.
 * @returns The index of the key or -1 if the key does not  exists.
 */
static int32_t aquarium_texture_map_lookup_index(aquarium_t* this,
                                                 const char* key)
{
  int32_t index = -1;
  for (uint32_t i = 0; i < this->texture_count; ++i) {
    if (strcmp(this->texture_map[i].key, key) == 0) {
      index = i;
      break;
    }
  }
  return index;
}

/**
 * @brief Looks up the texture mapped to a key in the texture map.
 * @return Pointer to the texture of NULL if the key does not exists
 */
static texture_t* aquarium_texture_map_lookup_texture(aquarium_t* this,
                                                      const char* key)
{
  int32_t key_index = aquarium_texture_map_lookup_index(this, key);
  return (key_index != -1) ? &this->texture_map[key_index].value : NULL;
}

/**
 * @brief Extends the texture map by inserting a new texture, effectively
 * increasing the map size by the number of elements inserted.
 */
static void aquarium_texture_map_insert(aquarium_t* this, const char* key,
                                        texture_t* texture)
{
  int32_t key_index = aquarium_texture_map_lookup_index(this, key);
  uint32_t insert_index
    = (key_index == -1) ? this->texture_count : (uint32_t)key_index;
  uint32_t texture_count_inc = (key_index == -1) ? 1 : 0;

  snprintf(this->texture_map[insert_index].key,
           sizeof(this->texture_map[insert_index].key), "%s", key);
  memcpy(&this->texture_map[insert_index].value, texture, sizeof(texture_t));
  this->texture_count += texture_count_inc;
}

static int32_t aquarium_program_map_lookup_index(aquarium_t* this,
                                                 const char* key)
{
  int32_t index = -1;
  for (uint32_t i = 0; i < this->program_count; ++i) {
    if (strcmp(this->program_map[i].key, key) == 0) {
      index = i;
      break;
    }
  }
  return index;
}

static program_t* aquarium_program_map_lookup_program(aquarium_t* this,
                                                      const char* key)
{
  int32_t key_index = aquarium_program_map_lookup_index(this, key);
  return (key_index != -1) ? &this->program_map[key_index].value : NULL;
}

static void aquarium_program_map_insert(aquarium_t* this, const char* key,
                                        program_t* program)
{
  int32_t key_index = aquarium_program_map_lookup_index(this, key);
  uint32_t insert_index
    = (key_index == -1) ? this->program_count : (uint32_t)key_index;
  uint32_t program_count_inc = (key_index == -1) ? 1 : 0;

  snprintf(this->program_map[insert_index].key,
           sizeof(this->program_map[insert_index].key), "%s", key);
  memcpy(&this->program_map[insert_index].value, program, sizeof(program_t));
  this->program_count += program_count_inc;
}

static bool aquarium_init(aquarium_t* this)
{
  /* Create context */
  context_create(&this->context);

  if (!context_initialize(&this->context)) {
    return false;
  }

  aquarium_calculate_fish_count(this);

  printf("Init resources ...\n");
  aquarium_get_elapsed_time(this);

  const resource_helper_t* resource_helper
    = context_get_resource_helper(&this->context);
  const char sky_urls[6][STRMAX];
  resource_helper_get_sky_box_urls(resource_helper, &sky_urls);
  texture_t skybox
    = context_create_cubemap_texture(&this->context, "skybox", &sky_urls);
  aquarium_texture_map_insert(this, "skybox", &skybox);

  /* Init general buffer and binding groups for dawn backend. */
  context_init_general_resources(&this->context, this);
  /* Avoid resource allocation in the first render loop. */
  this->pre_fish_count = this->cur_fish_count;

  aquarium_setup_model_enum_map(this);
  aquarium_load_resource(this);

  printf("End loading.\nCost %0.3f ms totally.",
         aquarium_get_elapsed_time(this));

  aquarium_reset_fps_time(this);

  return true;
}

static int32_t aquarium_get_cur_fish_count(aquarium_t* this)
{
  return this->cur_fish_count;
}

static int32_t aquarium_get_pre_fish_count(aquarium_t* this)
{
  return this->pre_fish_count;
}

static void aquarium_reset_fps_time(aquarium_t* this)
{
  this->g.start = get_current_time_point_ms(this->wgpu_example_context);
  this->g.then  = this->g.start;
}

static void aquarium_load_resource(aquarium_t* this)
{
  aquarium_load_models(this);
  aquarium_load_placement(this);
  if (aquarium_settings.simulate_fish_come_and_go) {
    aquarium_load_fish_scenario(this);
  }
}

static model_name_t
aquarium_map_model_name_str_to_model_name(aquarium_t* this,
                                          const char* model_name_str)
{
  model_name_t model_name = MODELNAME_MODELMAX;
  for (uint32_t i = 0; i < MODELNAME_MODELMAX; ++i) {
    if (strcmp(this->model_enum_map[i].key, model_name_str) == 0) {
      model_name = this->model_enum_map[i].value;
      break;
    }
  }
  return model_name;
}

static void aquarium_setup_model_enum_map(aquarium_t* this)
{
  for (uint32_t i = 0; i < MODELNAME_MODELMAX; ++i) {
    snprintf(this->model_enum_map[i].key, strlen(g_scene_info[i].name_str) + 1,
             "%s", g_scene_info[i].name_str);
    this->model_enum_map[i].value = g_scene_info[i].name;
  }
}

static int32_t aquarium_load_models(aquarium_t* this)
{
  const bool enable_instanced_draw = aquarium_settings.enable_instanced_draw;
  for (uint32_t i = 0; i < MODELNAME_MODELMAX; ++i) {
    const g_scene_info_t* info = &g_scene_info[i];
    if ((enable_instanced_draw && info->type == MODELGROUP_FISH)
        || ((!enable_instanced_draw)
            && info->type == MODELGROUP_FISHINSTANCEDDRAW)) {
      continue;
    }
    aquarium_load_model(this, info);
    break;
  }

  return EXIT_SUCCESS;
}

static void aquarium_calculate_fish_count(aquarium_t* this)
{
  /* Calculate fish count for each type of fish */
  int32_t num_left = this->cur_fish_count;
  for (int i = 0; i < FISHENUM_MAX; ++i) {
    for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(fish_table); ++i) {
      const fish_t* fish_info = &fish_table[i];
      if (fish_info->type != i) {
        continue;
      }
      int32_t num_float = num_left;
      if (i == FISHENUM_BIG) {
        int32_t temp = this->cur_fish_count < g_settings.num_fish_small ? 1 : 2;
        num_float    = MIN(num_left, temp);
      }
      else if (i == FISHENUM_MEDIUM) {
        if (this->cur_fish_count < g_settings.num_fish_medium) {
          num_float = MIN(num_left, this->cur_fish_count / 10);
        }
        else if (this->cur_fish_count < g_settings.num_fish_big) {
          num_float = MIN(num_left, g_settings.num_fish_left_small);
        }
        else {
          num_float = MIN(num_left, g_settings.num_fish_left_big);
        }
      }
      num_left = num_left - num_float;
      this->fish_count[fish_info->model_name - MODELNAME_MODELSMALLFISHA]
        = num_float;
    }
  }
}

static float aquarium_get_elapsed_time(aquarium_t* this)
{
  /* Update our time */
  const float now          = this->wgpu_example_context->frame.timestamp_millis;
  const float elapsed_time = now - this->g.then;
  this->g.then             = now;

  return elapsed_time;
}

static void aquarium_update_world_uniforms(aquarium_t* this)
{
  context_update_world_uniforms(&this->context, this);
}

static void aquarium_update_global_uniforms(aquarium_t* this)
{
  global_t* g = &this->g;
  light_world_position_uniform_t* light_world_position_uniform
    = &this->light_world_position_uniform;

  float elapsed_time = aquarium_get_elapsed_time(this);
  g->mclock += elapsed_time * g_settings.speed;
  g->eye_clock += elapsed_time * g_settings.eye_speed;

  g->eye_position[0] = sin(g->eye_clock) * g_settings.eye_radius;
  g->eye_position[1] = g_settings.eye_height;
  g->eye_position[2] = cos(g->eye_clock) * g_settings.eye_radius;
  g->target[0] = (float)(sin(g->eye_clock + PI)) * g_settings.target_radius;
  g->target[1] = g_settings.target_height;
  g->target[2] = (float)(cos(g->eye_clock + PI)) * g_settings.target_radius;

  float near_plane   = 1.0f;
  float far_plane    = 25000.0f;
  const float aspect = (float)context_get_client_width(&this->context)
                       / (float)context_get_client_height(&this->context);
  float top
    = tan(deg_to_rad(g_settings.field_of_view * g_settings.fov_fudge) * 0.5f)
      * near_plane;
  float bottom = -top;
  float left   = aspect * bottom;
  float right  = aspect * top;
  float width  = fabs(right - left);
  float height = fabs(top - bottom);
  float xOff   = width * g_settings.net_offset[0] * g_settings.net_offset_mult;
  float yOff   = height * g_settings.net_offset[1] * g_settings.net_offset_mult;

  /* Set frustum and camera look at */
  matrix_frustum(g->projection, left + xOff, right + xOff, bottom + yOff,
                 top + yOff, near_plane, far_plane);
  matrix_camera_look_at(light_world_position_uniform->view_inverse,
                        g->eye_position, g->target, g->up);
  matrix_inverse4(g->view, light_world_position_uniform->view_inverse);
  matrix_mul_matrix_matrix4(light_world_position_uniform->view_projection,
                            g->view, g->projection);
  matrix_inverse4(g->view_projection_inverse,
                  light_world_position_uniform->view_projection);

  memcpy(g->sky_view, g->view, 16 * sizeof(float));
  g->sky_view[12] = 0.0f;
  g->sky_view[13] = 0.0f;
  g->sky_view[14] = 0.0f;
  matrix_mul_matrix_matrix4(g->sky_view_projection, g->sky_view, g->projection);
  matrix_inverse4(g->sky_view_projection_inverse, g->sky_view_projection);

  matrix_get_axis(g->v3t0, light_world_position_uniform->view_inverse, 0);
  matrix_get_axis(g->v3t1, light_world_position_uniform->view_inverse, 1);
  matrix_mul_scalar_vector(20.0f, g->v3t0, 3);
  matrix_mul_scalar_vector(30.0f, g->v3t1, 3);
  matrix_add_vector(light_world_position_uniform->light_world_pos,
                    g->eye_position, g->v3t0, 3);
  matrix_add_vector(light_world_position_uniform->light_world_pos,
                    light_world_position_uniform->light_world_pos, g->v3t1, 3);

  /* Update world uniforms  */
  context_update_world_uniforms(&this->context, this);
}

static void aquarium_update_and_draw(aquarium_t* this);

static void aquarium_render(aquarium_t* this)
{
  matrix_reset_pseudoRandom();

  context_pre_frame(&this->context);

  /* Global Uniforms should update after command reallocation. */
  aquarium_update_global_uniforms(this);

  if (aquarium_settings.simulate_fish_come_and_go) {
    if (!sc_queue_empty(&this->fish_behavior)) {
      behavior_t* behave = sc_queue_peek_first(&this->fish_behavior);
      int32_t frame      = behave->_frame;
      if (frame == 0) {
        sc_queue_del_first(&this->fish_behavior);
        if (behave->_op == OPERATION_PLUS) {
          this->cur_fish_count += behave->_count;
        }
        else {
          this->cur_fish_count -= behave->_count;
        }
      }
      else {
        behave->_frame = --frame;
      }
    }
  }

  if (!aquarium_settings.enable_instanced_draw) {
    if (this->cur_fish_count != this->pre_fish_count) {
      aquarium_calculate_fish_count(this);
      bool enable_dynamic_buffer_offset
        = aquarium_settings.enable_dynamic_buffer_offset;
      context_realloc_resource(&this->context, this->pre_fish_count,
                               this->cur_fish_count,
                               enable_dynamic_buffer_offset);
      this->pre_fish_count = this->cur_fish_count;

      aquarium_reset_fps_time(this);
    }
  }

  aquarium_update_and_draw(this);
}

/* -------------------------------------------------------------------------- *
 * Model - Defines generic model.
 * -------------------------------------------------------------------------- */

#define MAX_WORLD_MATRIX_COUNT (16u)

typedef float world_matrix_t[16];

struct model_t;

typedef struct model_vtbl_t {
  void (*destroy)(struct model_t* this);
  void (*prepare_for_draw)(struct model_t* this);
  void (*update_per_instance_uniforms)(struct model_t* this,
                                       const world_uniforms_t* world_uniforms);
  void (*draw)(struct model_t* this);
  void (*set_program)(struct model_t* this, program_t* prgm);
  void (*init)(struct model_t* this);
} model_vtbl_t;

typedef struct model_t {
  model_group_t _type;
  model_name_t _name;
  program_t* _program;
  bool _blend;

  world_matrix_t world_matrices[MAX_WORLD_MATRIX_COUNT];
  uint16_t world_matrix_count;
  texture_t* texture_map[TEXTURETYPE_MAX];
  buffer_dawn_t* buffer_map[BUFFERTYPE_MAX];
  /* Function pointers */
  model_vtbl_t _vtbl;
} model_t;

/* Function prototypes */
static void model_set_program(model_t* this, program_t* prgm);

static void model_init_defaults(model_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void model_init_virtual_method_table(model_t* this)
{
  this->_vtbl.set_program = model_set_program;
}

static void model_create(model_t* this, model_group_t type, model_name_t name,
                         bool blend)
{
  model_init_defaults(this);
  model_init_virtual_method_table(this);

  this->_type    = type;
  this->_name    = name;
  this->_program = NULL;
  this->_blend   = blend;
}

static void model_destroy(model_t* this)
{
  if (this->_vtbl.destroy != NULL) {
    this->_vtbl.destroy(this);
  }

  for (uint32_t i = 0; i < BUFFERTYPE_MAX; ++i) {
    buffer_dawn_t* buf = this->buffer_map[i];
    if (buf) {
      buffer_dawn_destroy(buf);
      buf = NULL;
    }
  }
}

static void model_prepare_for_draw(model_t* this)
{
  this->_vtbl.prepare_for_draw(this);
}

static void
model_update_per_instance_uniforms(model_t* this,
                                   const world_uniforms_t* world_uniforms)
{
  this->_vtbl.update_per_instance_uniforms(this, world_uniforms);
}

static void model_draw(model_t* this)
{
  this->_vtbl.draw(this);
}

static void model_set_program(model_t* this, program_t* prgm)
{
  this->_program = prgm;
}

static void model_init(model_t* this)
{
  this->_vtbl.init(this);
}

/* -------------------------------------------------------------------------- *
 * Fish model - Defines the fish model
 *  - Updates fish specific uniforms.
 *  - Implement common functions of fish models.
 * -------------------------------------------------------------------------- */

struct fish_model_t;

typedef struct fish_model_vtbl_t {
  void (*update_fish_per_uniforms)(struct fish_model_t* this, float x, float y,
                                   float z, float next_x, float next_y,
                                   float next_z, float scale, float time,
                                   int index);
} fish_model_vtbl_t;

typedef struct fish_model_t {
  fish_model_vtbl_t _vtbl;
  model_t _model;
  int32_t _pre_instance;
  int32_t _cur_instance;
  int32_t _fish_per_offset;
  aquarium_t* _aquarium;
} fish_model_t;

static void fish_model_prepare_for_draw(model_t* this);

static void fish_model_init_defaults(fish_model_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void fish_model_init_virtual_method_table(fish_model_t* this)
{
  this->_model._vtbl.prepare_for_draw = fish_model_prepare_for_draw;
}

static void fish_model_create(fish_model_t* this, model_group_t type,
                              model_name_t name, bool blend,
                              aquarium_t* aquarium)
{
  fish_model_init_defaults(this);

  /* Create model and set function pointers */
  model_create(&this->_model, type, name, blend);
  fish_model_init_virtual_method_table(this);

  this->_aquarium = aquarium;
}

static void fish_model_update_fish_per_uniforms(fish_model_t* this, float x,
                                                float y, float z, float next_x,
                                                float next_y, float next_z,
                                                float scale, float time,
                                                int index)
{
  this->_vtbl.update_fish_per_uniforms(this, x, y, z, next_x, next_y, next_z,
                                       scale, time, index);
}

static void fish_model_prepare_for_draw(model_t* this)
{
  fish_model_t* _this     = (fish_model_t*)this;
  _this->_fish_per_offset = 0;
  for (uint32_t i = 0; i < _this->_model._name - MODELNAME_MODELSMALLFISHA;
       ++i) {
    const fish_t* fish_info = &fish_table[i];
    _this->_fish_per_offset
      += _this->_aquarium
           ->fish_count[fish_info->model_name - MODELNAME_MODELSMALLFISHA];
  }

  const fish_t* fish_info
    = &fish_table[_this->_model._name - MODELNAME_MODELSMALLFISHA];
  _this->_cur_instance
    = _this->_aquarium
        ->fish_count[fish_info->model_name - MODELNAME_MODELSMALLFISHA];
}

/* -------------------------------------------------------------------------- *
 * Fish model - Defines the base fish model.
 * -------------------------------------------------------------------------- */

typedef struct {
  fish_model_t _fish_model;
  struct {
    float fish_length;
    float fish_wave_length;
    float fish_bend_amount;
  } fish_vertex_uniforms;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* tangent;
    buffer_dawn_t* bi_normal;
    buffer_dawn_t* indices;
  } buffers;
  WGPUVertexState _vertex_state;
  WGPURenderPipeline _pipeline;
  WGPUBindGroupLayout _bind_group_layout_model;
  WGPUPipelineLayout _pipeline_layout;
  WGPUBindGroup _bind_group_model;
  WGPUBuffer _fish_vertex_buffer;
  struct {
    WGPUBuffer light_factor;
  } _uniform_buffers;
  wgpu_context_t* _wgpu_context;
  context_t* _context;
  program_t* _program;
  aquarium_t* _aquarium;
  bool enable_dynamic_buffer_offset;
} fish_model_draw_t;

static void fish_model_draw_destroy(model_t* self);
static void fish_model_draw_init(model_t* self);
static void fish_model_draw_draw(model_t* self);
static void fish_model_draw_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms);
static void fish_model_draw_update_fish_per_uniforms(fish_model_t* this,
                                                     float x, float y, float z,
                                                     float next_x, float next_y,
                                                     float next_z, float scale,
                                                     float time, int index);

static void fish_model_draw_init_defaults(fish_model_draw_t* this)
{
  memset(this, 0, sizeof(*this));

  this->enable_dynamic_buffer_offset
    = aquarium_settings.enable_dynamic_buffer_offset;

  this->light_factor_uniforms.shininess       = 5.0f;
  this->light_factor_uniforms.specular_factor = 0.3f;
}

static void fish_model_draw_init_virtual_method_table(fish_model_draw_t* this)
{
  /* Override model functions */
  this->_fish_model._model._vtbl.destroy = fish_model_draw_destroy;
  this->_fish_model._model._vtbl.init    = fish_model_draw_init;
  this->_fish_model._model._vtbl.draw    = fish_model_draw_draw;
  this->_fish_model._model._vtbl.update_per_instance_uniforms
    = fish_model_draw_update_per_instance_uniforms;

  /* Override fish model functions */
  this->_fish_model._vtbl.update_fish_per_uniforms
    = fish_model_draw_update_fish_per_uniforms;
}

static void fish_model_draw_create(fish_model_draw_t* this, context_t* context,
                                   aquarium_t* aquarium, model_group_t type,
                                   model_name_t name, bool blend)
{
  fish_model_draw_init_defaults(this);

  /* Create model and set function pointers */
  fish_model_create(&this->_fish_model, type, name, blend, aquarium);
  fish_model_draw_init_virtual_method_table(this);

  this->_wgpu_context = context->wgpu_context;
  this->_context      = context;
  this->_aquarium     = aquarium;

  const fish_t* fish_info = &fish_table[name - MODELNAME_MODELSMALLFISHA];
  this->fish_vertex_uniforms.fish_length      = fish_info->fish_length;
  this->fish_vertex_uniforms.fish_bend_amount = fish_info->fish_bend_amount;
  this->fish_vertex_uniforms.fish_wave_length = fish_info->fish_wave_length;

  this->_fish_model._cur_instance
    = aquarium->fish_count[fish_info->model_name - MODELNAME_MODELSMALLFISHA];
  this->_fish_model._pre_instance = this->_fish_model._cur_instance;
}

static void fish_model_draw_destroy(model_t* this)
{
  fish_model_draw_t* _this = (fish_model_draw_t*)this;

  WGPU_RELEASE_RESOURCE(RenderPipeline, _this->_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layout_model)
  WGPU_RELEASE_RESOURCE(PipelineLayout, _this->_pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_group_model)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_fish_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.light_factor)
}

static void fish_model_draw_init(model_t* this)
{
  fish_model_draw_t* _this = (fish_model_draw_t*)this;
  model_t* model           = &_this->_fish_model._model;

  _this->_program            = model->_program;
  WGPUShaderModule vs_module = program_get_vs_module(_this->_program);

  texture_t** texture_map    = model->texture_map;
  _this->textures.diffuse    = texture_map[TEXTURETYPE_DIFFUSE];
  _this->textures.normal     = texture_map[TEXTURETYPE_NORMAL_MAP];
  _this->textures.reflection = texture_map[TEXTURETYPE_REFLECTION_MAP];
  _this->textures.skybox     = texture_map[TEXTURETYPE_SKYBOX];

  buffer_dawn_t** buffer_map = model->buffer_map;
  _this->buffers.position    = buffer_map[BUFFERTYPE_POSITION];
  _this->buffers.normal      = buffer_map[BUFFERTYPE_NORMAL];
  _this->buffers.tex_coord   = buffer_map[BUFFERTYPE_TEX_COORD];
  _this->buffers.tangent     = buffer_map[BUFFERTYPE_TANGENT];
  _this->buffers.bi_normal   = buffer_map[BUFFERTYPE_BI_NORMAL];
  _this->buffers.indices     = buffer_map[BUFFERTYPE_INDICES];

  WGPUVertexAttribute vertex_attributes[5] = {
    [0] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = 0,
      .shaderLocation = 2,
    },
    [3] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 3,
    },
    [4] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 4,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[5] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.position),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.tex_coord),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[2],
    },
    [3] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.tangent),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[3],
    },
    [4] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.bi_normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[4],
    },
  };

  _this->_vertex_state.module     = vs_module;
  _this->_vertex_state.entryPoint = "main";
  _this->_vertex_state.bufferCount
    = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  _this->_vertex_state.buffers = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[8] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [5] = (WGPUBindGroupLayoutEntry) {
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [6] = (WGPUBindGroupLayoutEntry) {
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [7] = (WGPUBindGroupLayoutEntry) {
        .binding    = 7,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    uint32_t bgl_entry_count = 0;
    if (_this->textures.skybox && _this->textures.reflection) {
      bgl_entry_count = 8;
    }
    else {
      bgl_entry_count = 5;
      bgl_entries[3]  = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      };
    }

    _this->_bind_group_layout_model = context_make_bind_group_layout(
      _this->_context, bgl_entries, bgl_entry_count);
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    _this->_context->bind_group_layouts.general,  /* Group 0 */
    _this->_context->bind_group_layouts.world,    /* Group 1 */
    _this->_bind_group_layout_model,              /* Group 2 */
    _this->_context->bind_group_layouts.fish_per, /* Group 3 */
  };
  _this->_pipeline_layout = context_make_basic_pipeline_layout(
    _this->_context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  _this->_pipeline = context_create_render_pipeline(
    _this->_context, _this->_pipeline_layout, model->_program,
    &_this->_vertex_state, model->_blend);

  _this->_fish_vertex_buffer = context_create_buffer_from_data(
    _this->_context, &_this->fish_vertex_uniforms,
    sizeof(_this->fish_vertex_uniforms), sizeof(_this->fish_vertex_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  _this->_uniform_buffers.light_factor = context_create_buffer_from_data(
    _this->_context, &_this->light_factor_uniforms,
    sizeof(_this->light_factor_uniforms), sizeof(_this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  // Fish models includes small, medium and big. Some of them contains
  // reflection and skybox texture, but some doesn't.
  {
    WGPUBindGroupEntry bg_entries[8] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = _this->_fish_vertex_buffer,
        .offset  = 0,
        .size    = sizeof(_this->fish_vertex_uniforms)
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = _this->_uniform_buffers.light_factor,
        .offset  = 0,
        .size    = sizeof(_this->light_factor_uniforms)
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = _this->textures.reflection->sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .sampler = _this->textures.skybox->sampler,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding     = 4,
        .textureView = _this->textures.diffuse->view,
      },
      [5] = (WGPUBindGroupEntry) {
        .binding     = 5,
        .textureView = _this->textures.normal->view,
      },
      [6] = (WGPUBindGroupEntry) {
        .binding     = 6,
        .textureView = _this->textures.reflection->view,
      },
      [7] = (WGPUBindGroupEntry) {
        .binding     = 7,
        .textureView = _this->textures.skybox->view,
      },
    };
    uint32_t bg_entry_count = 0;
    if (_this->textures.skybox && _this->textures.reflection) {
      bg_entry_count = 8;
    }
    else {
      bg_entry_count = 5;
      bg_entries[2]  = (WGPUBindGroupEntry){
         .binding = 2,
         .sampler = _this->textures.diffuse->sampler,
      };
      bg_entries[3] = (WGPUBindGroupEntry){
        .binding     = 3,
        .textureView = _this->textures.diffuse->view,
      };
      bg_entries[4] = (WGPUBindGroupEntry){
        .binding     = 4,
        .textureView = _this->textures.normal->view,
      };
    }
    _this->_bind_group_model = context_make_bind_group(
      _this->_context, _this->_bind_group_layout_model, bg_entries,
      bg_entry_count);
  }

  context_set_buffer_data(_this->_context, _this->_uniform_buffers.light_factor,
                          sizeof(_this->light_factor_uniforms),
                          &_this->light_factor_uniforms,
                          sizeof(_this->light_factor_uniforms));
  context_set_buffer_data(_this->_context, _this->_fish_vertex_buffer,
                          sizeof(_this->fish_vertex_uniforms),
                          &_this->fish_vertex_uniforms,
                          sizeof(_this->fish_vertex_uniforms));
}

static void fish_model_draw_draw(model_t* this)
{
  fish_model_draw_t* _this = (fish_model_draw_t*)this;

  if (_this->_fish_model._cur_instance == 0) {
    return;
  }

  WGPURenderPassEncoder render_pass = _this->_context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, _this->_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    _this->_context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    _this->_context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, _this->_bind_group_model, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, _this->buffers.position->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, _this->buffers.normal->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 2, _this->buffers.tex_coord->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 3, _this->buffers.tangent->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 4, _this->buffers.bi_normal->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, _this->buffers.indices->buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);

  if (_this->enable_dynamic_buffer_offset) {
    for (int32_t i = 0; i < _this->_fish_model._cur_instance; ++i) {
      const uint32_t offset = 256u * (i + _this->_fish_model._fish_per_offset);
      wgpuRenderPassEncoderSetBindGroup(
        render_pass, 3, _this->_context->bind_group_fish_pers[0], 1, &offset);
      wgpuRenderPassEncoderDrawIndexed(
        render_pass, _this->buffers.indices->total_components, 1, 0, 0, 0);
    }
  }
  else {
    for (int32_t i = 0; i < _this->_fish_model._cur_instance; ++i) {
      const uint32_t offset = i + _this->_fish_model._fish_per_offset;
      wgpuRenderPassEncoderSetBindGroup(
        render_pass, 3, _this->_context->bind_group_fish_pers[offset], 0, NULL);
      wgpuRenderPassEncoderDrawIndexed(
        render_pass, _this->buffers.indices->total_components, 1, 0, 0, 0);
    }
  }
}

static void fish_model_draw_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms)
{
  UNUSED_VAR(this);
  UNUSED_VAR(world_uniforms);
}

static void fish_model_draw_update_fish_per_uniforms(fish_model_t* this,
                                                     float x, float y, float z,
                                                     float next_x, float next_y,
                                                     float next_z, float scale,
                                                     float time, int index)
{
  fish_model_draw_t* _this = (fish_model_draw_t*)this;

  index += _this->_fish_model._fish_per_offset;
  fish_per_t* fish_pers        = &_this->_context->fish_pers[index];
  fish_pers->world_position[0] = x;
  fish_pers->world_position[1] = y;
  fish_pers->world_position[2] = z;
  fish_pers->next_position[0]  = next_x;
  fish_pers->next_position[1]  = next_y;
  fish_pers->next_position[2]  = next_z;
  fish_pers->scale             = scale;
  fish_pers->time              = time;
}

/* -------------------------------------------------------------------------- *
 * Fish model Instanced Draw - Defines instance fish model.
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 world_position;
  float scale;
  vec3 next_position;
  float time;
} fish_model_instanced_draw_fish_per;

typedef struct {
  fish_model_t _fish_model;
  struct {
    float fish_length;
    float fish_wave_length;
    float fish_bend_amount;
  } fish_vertex_uniforms;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  fish_model_instanced_draw_fish_per* fish_pers;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* tangent;
    buffer_dawn_t* bi_normal;
    buffer_dawn_t* indices;
  } buffers;
  WGPUVertexState _vertex_state;
  WGPURenderPipeline _pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } _bind_group_layouts;
  WGPUPipelineLayout _pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } _bind_groups;
  WGPUBuffer _fish_vertex_buffer;
  struct {
    WGPUBuffer light_factor;
  } _uniform_buffers;
  WGPUBuffer _fish_pers_buffer;
  int32_t _instance;
  wgpu_context_t* _wgpu_context;
  context_t* _context;
  program_t* _program;
  aquarium_t* _aquarium;
} fish_model_instanced_draw_t;

static void fish_model_instanced_draw_destroy(model_t* self);
static void fish_model_instanced_draw_init(model_t* self);
static void fish_model_instanced_draw_draw(model_t* self);
static void fish_model_instanced_draw_prepare_for_draw(model_t* this);
static void fish_model_instanced_draw_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms);
static void fish_model_instanced_draw_update_fish_per_uniforms(
  fish_model_t* this, float x, float y, float z, float next_x, float next_y,
  float next_z, float scale, float time, int index);

static void
fish_model_instanced_draw_init_defaults(fish_model_instanced_draw_t* this)
{
  memset(this, 0, sizeof(*this));

  this->light_factor_uniforms.shininess       = 5.0f;
  this->light_factor_uniforms.specular_factor = 0.3f;

  this->_instance = 0;
}

static void fish_model_instanced_draw_init_virtual_method_table(
  fish_model_instanced_draw_t* this)
{
  /* Override model functions */
  this->_fish_model._model._vtbl.destroy = fish_model_instanced_draw_destroy;
  this->_fish_model._model._vtbl.init    = fish_model_instanced_draw_init;
  this->_fish_model._model._vtbl.draw    = fish_model_instanced_draw_draw;
  this->_fish_model._model._vtbl.prepare_for_draw
    = fish_model_instanced_draw_prepare_for_draw;
  this->_fish_model._model._vtbl.update_per_instance_uniforms
    = fish_model_instanced_draw_update_per_instance_uniforms;

  /* Override fish model functions */
  this->_fish_model._vtbl.update_fish_per_uniforms
    = fish_model_instanced_draw_update_fish_per_uniforms;
}

static void fish_model_instanced_draw_create(fish_model_instanced_draw_t* this,
                                             context_t* context,
                                             aquarium_t* aquarium,
                                             model_group_t type,
                                             model_name_t name, bool blend)
{
  fish_model_instanced_draw_init_defaults(this);

  /* Create model and set function pointers */
  fish_model_create(&this->_fish_model, type, name, blend, aquarium);
  fish_model_instanced_draw_init_virtual_method_table(this);

  this->_wgpu_context = context->wgpu_context;
  this->_context      = context;
  this->_aquarium     = aquarium;

  const fish_t* fish_info
    = &fish_table[name - MODELNAME_MODELSMALLFISHAINSTANCEDDRAWS];
  this->fish_vertex_uniforms.fish_length      = fish_info->fish_length;
  this->fish_vertex_uniforms.fish_bend_amount = fish_info->fish_bend_amount;
  this->fish_vertex_uniforms.fish_wave_length = fish_info->fish_wave_length;

  this->_instance
    = aquarium->fish_count[fish_info->model_name - MODELNAME_MODELSMALLFISHA];
  this->fish_pers
    = malloc(this->_instance + sizeof(fish_model_instanced_draw_fish_per));
  memset(this->fish_pers, 0, sizeof(*this->fish_pers));
}

static void fish_model_instanced_draw_destroy(model_t* this)
{
  fish_model_instanced_draw_t* _this = (fish_model_instanced_draw_t*)this;

  WGPU_RELEASE_RESOURCE(RenderPipeline, _this->_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, _this->_pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_fish_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.light_factor)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_fish_pers_buffer)
  free(_this->fish_pers);
}

static void fish_model_instanced_draw_init(model_t* this)
{
  fish_model_instanced_draw_t* _this = (fish_model_instanced_draw_t*)this;

  if (_this->_instance == 0) {
    return;
  }

  model_t* model = &_this->_fish_model._model;

  _this->_program            = model->_program;
  WGPUShaderModule vs_module = program_get_vs_module(_this->_program);

  texture_t** texture_map    = model->texture_map;
  _this->textures.diffuse    = texture_map[TEXTURETYPE_DIFFUSE];
  _this->textures.normal     = texture_map[TEXTURETYPE_NORMAL_MAP];
  _this->textures.reflection = texture_map[TEXTURETYPE_REFLECTION_MAP];
  _this->textures.skybox     = texture_map[TEXTURETYPE_SKYBOX];

  buffer_dawn_t** buffer_map = model->buffer_map;
  _this->buffers.position    = buffer_map[BUFFERTYPE_POSITION];
  _this->buffers.normal      = buffer_map[BUFFERTYPE_NORMAL];
  _this->buffers.tex_coord   = buffer_map[BUFFERTYPE_TEX_COORD];
  _this->buffers.tangent     = buffer_map[BUFFERTYPE_TANGENT];
  _this->buffers.bi_normal   = buffer_map[BUFFERTYPE_BI_NORMAL];
  _this->buffers.indices     = buffer_map[BUFFERTYPE_INDICES];

  WGPUBufferDescriptor buffer_desc = {
    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
    .size  = sizeof(fish_model_instanced_draw_fish_per) * _this->_instance,
    .mappedAtCreation = false,
  };
  _this->_fish_pers_buffer
    = context_create_buffer(_this->_context, &buffer_desc);

  WGPUVertexAttribute vertex_attributes[9] = {
    [0] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = 0,
      .shaderLocation = 2,
    },
    [3] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 3,
    },
    [4] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = offsetof(fish_model_instanced_draw_fish_per, world_position),
      .shaderLocation = 4,
    },
    [5] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 5,
    },
    [6] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32,
      .offset = offsetof(fish_model_instanced_draw_fish_per, scale),
      .shaderLocation = 6,
    },
    [7] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = offsetof(fish_model_instanced_draw_fish_per, next_position),
      .shaderLocation = 7,
    },
    [8] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = offsetof(fish_model_instanced_draw_fish_per, time),
      .shaderLocation = 8,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[6] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.position),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.tex_coord),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[2],
    },
    [3] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.tangent),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[3],
    },
    [4] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.bi_normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[4],
    },
    [5] = (WGPUVertexBufferLayout) {
      .arrayStride    = sizeof(fish_model_instanced_draw_fish_per),
      .stepMode       = WGPUVertexStepMode_Instance,
      .attributeCount = 4,
      .attributes     = &vertex_attributes[5],
    },
  };

  _this->_vertex_state.module     = vs_module;
  _this->_vertex_state.entryPoint = "main";
  _this->_vertex_state.bufferCount
    = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  _this->_vertex_state.buffers = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[8] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [5] = (WGPUBindGroupLayoutEntry) {
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [6] = (WGPUBindGroupLayoutEntry) {
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [7] = (WGPUBindGroupLayoutEntry) {
        .binding    = 7,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    uint32_t bgl_entry_count = 0;
    if (_this->textures.skybox && _this->textures.reflection) {
      bgl_entry_count = 8;
    }
    else {
      bgl_entry_count = 5;
      bgl_entries[3]  = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      };
    }

    _this->_bind_group_layouts.model = context_make_bind_group_layout(
      _this->_context, bgl_entries, bgl_entry_count);
  }

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    _this->_bind_group_layouts.per = context_make_bind_group_layout(
      _this->_context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    _this->_context->bind_group_layouts.general, /* Group 0 */
    _this->_context->bind_group_layouts.world,   /* Group 1 */
    _this->_bind_group_layouts.model,            /* Group 2 */
    _this->_bind_group_layouts.per,              /* Group 3 */
  };
  _this->_pipeline_layout = context_make_basic_pipeline_layout(
    _this->_context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  _this->_pipeline = context_create_render_pipeline(
    _this->_context, _this->_pipeline_layout, model->_program,
    &_this->_vertex_state, _this->_fish_model._model._blend);

  _this->_fish_vertex_buffer = context_create_buffer_from_data(
    _this->_context, &_this->fish_vertex_uniforms,
    sizeof(_this->fish_vertex_uniforms), sizeof(_this->fish_vertex_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  _this->_uniform_buffers.light_factor = context_create_buffer_from_data(
    _this->_context, &_this->light_factor_uniforms,
    sizeof(_this->light_factor_uniforms), sizeof(_this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  /**
   * Fish models includes small, medium and big. Some of them contains
   * reflection and skybox texture, but some doesn't.
   */
  {
    WGPUBindGroupEntry bg_entries[8] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = _this->_fish_vertex_buffer,
        .offset  = 0,
        .size    = sizeof(_this->fish_vertex_uniforms)
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = _this->_uniform_buffers.light_factor,
        .offset  = 0,
        .size    = sizeof(_this->light_factor_uniforms)
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = _this->textures.reflection->sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .sampler = _this->textures.skybox->sampler,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding     = 4,
        .textureView = _this->textures.diffuse->view,
      },
      [5] = (WGPUBindGroupEntry) {
        .binding     = 5,
        .textureView = _this->textures.normal->view,
      },
      [6] = (WGPUBindGroupEntry) {
        .binding     = 6,
        .textureView = _this->textures.reflection->view,
      },
      [7] = (WGPUBindGroupEntry) {
        .binding     = 7,
        .textureView = _this->textures.skybox->view,
      },
    };
    uint32_t bg_entry_count = 0;
    if (_this->textures.skybox && _this->textures.reflection) {
      bg_entry_count = 8;
    }
    else {
      bg_entry_count = 5;
      bg_entries[2]  = (WGPUBindGroupEntry){
         .binding = 2,
         .sampler = _this->textures.diffuse->sampler,
      };
      bg_entries[3] = (WGPUBindGroupEntry){
        .binding     = 3,
        .textureView = _this->textures.diffuse->view,
      };
      bg_entries[4] = (WGPUBindGroupEntry){
        .binding     = 4,
        .textureView = _this->textures.normal->view,
      };
    }
    _this->_bind_groups.model = context_make_bind_group(
      _this->_context, _this->_bind_group_layouts.model, bg_entries,
      bg_entry_count);
  }

  context_set_buffer_data(_this->_context, _this->_uniform_buffers.light_factor,
                          sizeof(_this->light_factor_uniforms),
                          &_this->light_factor_uniforms,
                          sizeof(_this->light_factor_uniforms));
  context_set_buffer_data(_this->_context, _this->_fish_vertex_buffer,
                          sizeof(_this->fish_vertex_uniforms),
                          &_this->fish_vertex_uniforms,
                          sizeof(_this->fish_vertex_uniforms));
}

static void fish_model_instanced_draw_prepare_for_draw(model_t* this)
{
  UNUSED_VAR(this);
}

static void fish_model_instanced_draw_draw(model_t* this)
{
  fish_model_instanced_draw_t* _this = (fish_model_instanced_draw_t*)this;

  if (_this->_instance == 0) {
    return;
  }

  context_set_buffer_data(
    _this->_context, _this->_fish_pers_buffer,
    sizeof(fish_model_instanced_draw_fish_per) * _this->_instance,
    _this->fish_pers,
    sizeof(fish_model_instanced_draw_fish_per) * _this->_instance);

  WGPURenderPassEncoder render_pass = _this->_context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, _this->_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    _this->_context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    _this->_context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, _this->_bind_groups.model,
                                    0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, _this->buffers.position->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, _this->buffers.normal->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 2, _this->buffers.tex_coord->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 3, _this->buffers.tangent->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 4, _this->buffers.bi_normal->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(render_pass, 5, _this->_fish_pers_buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, _this->buffers.indices->buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(render_pass,
                                   _this->buffers.indices->total_components,
                                   _this->_instance, 0, 0, 0);
}

static void fish_model_instanced_draw_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms)
{
  UNUSED_VAR(this);
  UNUSED_VAR(world_uniforms);
}

static void fish_model_instanced_draw_update_fish_per_uniforms(
  fish_model_t* self, float x, float y, float z, float next_x, float next_y,
  float next_z, float scale, float time, int index)
{
  fish_model_instanced_draw_t* _this = (fish_model_instanced_draw_t*)self;
  fish_model_instanced_draw_fish_per* fish_pers = &_this->fish_pers[index];

  fish_pers->world_position[0] = x;
  fish_pers->world_position[1] = y;
  fish_pers->world_position[2] = z;
  fish_pers->next_position[0]  = next_x;
  fish_pers->next_position[1]  = next_y;
  fish_pers->next_position[2]  = next_z;
  fish_pers->scale             = scale;
  fish_pers->time              = time;
}

/* -------------------------------------------------------------------------- *
 * Generic model - Defines generic model.
 * -------------------------------------------------------------------------- */

typedef struct {
  model_t _model;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* tangent;
    buffer_dawn_t* bi_normal;
    buffer_dawn_t* indices;
  } buffers;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  struct {
    world_uniforms_t world_uniforms[20];
  } world_uniform_per;
  WGPUVertexState _vertex_state;
  WGPURenderPipeline _pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } _bind_group_layouts;
  WGPUPipelineLayout _pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } _bind_groups;
  struct {
    WGPUBuffer light_factor;
    WGPUBuffer world;
  } _uniform_buffers;
  wgpu_context_t* _wgpu_context;
  context_t* _context;
  program_t* _program;
  aquarium_t* _aquarium;
  int32_t _instance;
} generic_model_t;

static void generic_model_destroy(model_t* this);
static void generic_model_init(model_t* this);
static void generic_model_prepare_for_draw(model_t* this);
static void generic_model_draw(model_t* this);
static void generic_model_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms);

static void generic_model_init_defaults(generic_model_t* this)
{
  memset(this, 0, sizeof(*this));

  this->light_factor_uniforms.shininess       = 50.0f;
  this->light_factor_uniforms.specular_factor = 1.0f;
}

static void generic_model_init_virtual_method_table(generic_model_t* this)
{
  /* Override model functions */
  this->_model._vtbl.destroy          = generic_model_destroy;
  this->_model._vtbl.init             = generic_model_init;
  this->_model._vtbl.prepare_for_draw = generic_model_prepare_for_draw;
  this->_model._vtbl.draw             = generic_model_draw;
  this->_model._vtbl.update_per_instance_uniforms
    = generic_model_update_per_instance_uniforms;
}

static void generic_model_create(generic_model_t* this, context_t* context,
                                 aquarium_t* aquarium, model_group_t type,
                                 model_name_t name, bool blend)
{
  generic_model_init_defaults(this);

  /* Create model and set function pointers */
  model_create(&this->_model, type, name, blend);
  generic_model_init_virtual_method_table(this);

  this->_wgpu_context = context->wgpu_context;
  this->_context      = context;
  this->_aquarium     = aquarium;
}

static void generic_model_destroy(model_t* this)
{
  generic_model_t* _this = (generic_model_t*)this;

  WGPU_RELEASE_RESOURCE(RenderPipeline, _this->_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, _this->_pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.light_factor)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.world)
}

static void generic_model_init(model_t* self)
{
  generic_model_t* _this = (generic_model_t*)self;

  _this->_program            = _this->_model._program;
  WGPUShaderModule vs_module = program_get_vs_module(_this->_program);

  texture_t** texture_map    = _this->_model.texture_map;
  _this->textures.diffuse    = texture_map[TEXTURETYPE_DIFFUSE];
  _this->textures.normal     = texture_map[TEXTURETYPE_NORMAL_MAP];
  _this->textures.reflection = texture_map[TEXTURETYPE_REFLECTION_MAP];
  _this->textures.skybox     = texture_map[TEXTURETYPE_SKYBOX];

  buffer_dawn_t** buffer_map = _this->_model.buffer_map;
  _this->buffers.position    = buffer_map[BUFFERTYPE_POSITION];
  _this->buffers.normal      = buffer_map[BUFFERTYPE_NORMAL];
  _this->buffers.tex_coord   = buffer_map[BUFFERTYPE_TEX_COORD];
  _this->buffers.tangent     = buffer_map[BUFFERTYPE_TANGENT];
  _this->buffers.bi_normal   = buffer_map[BUFFERTYPE_BI_NORMAL];
  _this->buffers.indices     = buffer_map[BUFFERTYPE_INDICES];

  // Generic models use reflection, normal or diffuse shaders, of which
  // groupLayouts are diiferent in texture binding.  MODELGLOBEBASE use diffuse
  // shader though it contains normal and reflection textures.
  WGPUVertexAttribute vertex_attributes[5] = {0};
  {
    vertex_attributes[0].format         = WGPUVertexFormat_Float32x3;
    vertex_attributes[0].offset         = 0;
    vertex_attributes[0].shaderLocation = 0;
    vertex_attributes[1].format         = WGPUVertexFormat_Float32x3;
    vertex_attributes[1].offset         = 0;
    vertex_attributes[1].shaderLocation = 1;
    vertex_attributes[2].format         = WGPUVertexFormat_Float32x2;
    vertex_attributes[2].offset         = 0;
    vertex_attributes[2].shaderLocation = 2;
    vertex_attributes[3].format         = WGPUVertexFormat_Float32x3;
    vertex_attributes[3].offset         = 0;
    vertex_attributes[3].shaderLocation = 3;
    vertex_attributes[4].format         = WGPUVertexFormat_Float32x3;
    vertex_attributes[4].offset         = 0;
    vertex_attributes[4].shaderLocation = 4;
  }

  // Generic models use reflection, normal or diffuse shaders, of which
  // groupLayouts are diiferent in texture binding.  MODELGLOBEBASE use diffuse
  // shader though it contains normal and reflection textures.
  WGPUVertexBufferLayout vertex_buffer_layouts[5] = {0};
  uint32_t vertex_buffer_layout_count             = 0;
  {
    vertex_buffer_layouts[0].arrayStride    = _this->buffers.position->size,
    vertex_buffer_layouts[0].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[0].attributeCount = 1;
    vertex_buffer_layouts[0].attributes     = &vertex_attributes[0];
    vertex_buffer_layouts[1].arrayStride    = _this->buffers.normal->size,
    vertex_buffer_layouts[1].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[1].attributeCount = 1;
    vertex_buffer_layouts[1].attributes     = &vertex_attributes[1];
    vertex_buffer_layouts[2].arrayStride    = _this->buffers.tex_coord->size,
    vertex_buffer_layouts[2].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[2].attributeCount = 1;
    vertex_buffer_layouts[2].attributes     = &vertex_attributes[2];
    vertex_buffer_layouts[3].arrayStride    = _this->buffers.tangent->size,
    vertex_buffer_layouts[3].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[3].attributeCount = 1;
    vertex_buffer_layouts[3].attributes     = &vertex_attributes[3];
    vertex_buffer_layouts[4].arrayStride    = _this->buffers.normal->size,
    vertex_buffer_layouts[4].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[4].attributeCount = 1;
    vertex_buffer_layouts[4].attributes     = &vertex_attributes[4];
    if (_this->textures.normal
        && _this->_model._name != MODELNAME_MODELGLOBEBASE) {
      vertex_buffer_layout_count = 5;
    }
    else {
      vertex_buffer_layout_count = 3;
    }
  }

  _this->_vertex_state.module      = vs_module;
  _this->_vertex_state.entryPoint  = "main";
  _this->_vertex_state.bufferCount = vertex_buffer_layout_count;
  _this->_vertex_state.buffers     = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[7] = {0};
    uint32_t bgl_entry_count                = 0;
    bgl_entries[0].binding                  = 0;
    bgl_entries[0].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[0].buffer.type              = WGPUBufferBindingType_Uniform;
    bgl_entries[0].buffer.hasDynamicOffset  = false;
    bgl_entries[0].buffer.minBindingSize    = 0;
    bgl_entries[1].binding                  = 1;
    bgl_entries[1].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[1].sampler.type             = WGPUSamplerBindingType_Filtering;
    bgl_entries[2].binding                  = 2;
    bgl_entries[2].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[2].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[2].texture.viewDimension    = WGPUTextureViewDimension_2D;
    bgl_entries[2].texture.multisampled     = false;
    bgl_entries[3].binding                  = 3;
    bgl_entries[3].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[3].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[3].texture.viewDimension    = WGPUTextureViewDimension_2D;
    bgl_entries[3].texture.multisampled     = false;
    bgl_entries[4].binding                  = 4;
    bgl_entries[4].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[4].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[4].texture.viewDimension    = WGPUTextureViewDimension_2D;
    bgl_entries[4].texture.multisampled     = false;
    bgl_entries[5].binding                  = 5;
    bgl_entries[5].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[5].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[5].texture.viewDimension    = WGPUTextureViewDimension_2D;
    bgl_entries[5].texture.multisampled     = false;
    bgl_entries[6].binding                  = 6;
    bgl_entries[6].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[6].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[6].texture.viewDimension    = WGPUTextureViewDimension_Cube;
    bgl_entries[6].texture.multisampled     = false;
    if (_this->textures.skybox && _this->textures.reflection
        && _this->_model._name != MODELNAME_MODELGLOBEBASE) {
      bgl_entry_count = 7;
    }
    else if (_this->textures.normal
             && _this->_model._name != MODELNAME_MODELGLOBEBASE) {
      bgl_entry_count = 4;
    }
    else {
      bgl_entry_count = 3;
    }
    _this->_bind_group_layouts.model = context_make_bind_group_layout(
      _this->_context, bgl_entries, bgl_entry_count);
  }

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    _this->_bind_group_layouts.per = context_make_bind_group_layout(
      _this->_context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    _this->_context->bind_group_layouts.general, /* Group 0 */
    _this->_context->bind_group_layouts.world,   /* Group 1 */
    _this->_bind_group_layouts.model,            /* Group 2 */
    _this->_bind_group_layouts.per,              /* Group 3 */
  };
  _this->_pipeline_layout = context_make_basic_pipeline_layout(
    _this->_context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  _this->_pipeline = context_create_render_pipeline(
    _this->_context, _this->_pipeline_layout, _this->_model._program,
    &_this->_vertex_state, _this->_model._blend);

  _this->_uniform_buffers.light_factor = context_create_buffer_from_data(
    _this->_context, &_this->light_factor_uniforms,
    sizeof(_this->light_factor_uniforms), sizeof(_this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  _this->_uniform_buffers.world = context_create_buffer_from_data(
    _this->_context, &_this->world_uniform_per,
    sizeof(_this->world_uniform_per),
    calc_constant_buffer_byte_size(sizeof(_this->world_uniform_per)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  // Generic models use reflection, normal or diffuse shaders, of which
  // grouplayouts are diiferent in texture binding. MODELGLOBEBASE use diffuse
  // shader though it contains normal and reflection textures.
  {
    WGPUBindGroupEntry bg_entries[7] = {0};
    uint32_t bg_entry_count          = 0;
    bg_entries[0].binding            = 0;
    bg_entries[0].buffer             = _this->_uniform_buffers.light_factor;
    bg_entries[0].offset             = 0;
    bg_entries[0].size               = sizeof(light_uniforms_t);
    if (_this->textures.skybox && _this->textures.reflection
        && _this->_model._name != MODELNAME_MODELGLOBEBASE) {
      bg_entry_count            = 7;
      bg_entries[1].binding     = 1;
      bg_entries[1].sampler     = _this->textures.reflection->sampler,
      bg_entries[2].binding     = 2;
      bg_entries[2].sampler     = _this->textures.skybox->sampler,
      bg_entries[3].binding     = 3;
      bg_entries[3].textureView = _this->textures.diffuse->view,
      bg_entries[4].binding     = 4;
      bg_entries[4].textureView = _this->textures.normal->view,
      bg_entries[5].binding     = 5;
      bg_entries[5].textureView = _this->textures.reflection->view,
      bg_entries[6].binding     = 6;
      bg_entries[6].textureView = _this->textures.skybox->view;
    }
    else if (_this->textures.normal
             && _this->_model._name != MODELNAME_MODELGLOBEBASE) {
      bg_entry_count            = 4;
      bg_entries[1].binding     = 1;
      bg_entries[1].sampler     = _this->textures.diffuse->sampler;
      bg_entries[2].binding     = 2;
      bg_entries[2].textureView = _this->textures.diffuse->view;
      bg_entries[3].binding     = 3;
      bg_entries[3].textureView = _this->textures.normal->view;
    }
    else {
      bg_entry_count            = 3;
      bg_entries[1].binding     = 1;
      bg_entries[1].sampler     = _this->textures.diffuse->sampler;
      bg_entries[2].binding     = 2;
      bg_entries[2].textureView = _this->textures.diffuse->view;
    }
    _this->_bind_groups.model = context_make_bind_group(
      _this->_context, _this->_bind_group_layouts.model, bg_entries,
      bg_entry_count);
  }

  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = _this->_uniform_buffers.world,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(world_uniforms_t)),
      },
    };
    _this->_bind_groups.per
      = context_make_bind_group(_this->_context, _this->_bind_group_layouts.per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  context_set_buffer_data(_this->_context, _this->_uniform_buffers.light_factor,
                          sizeof(_this->light_factor_uniforms),
                          &_this->light_factor_uniforms,
                          sizeof(_this->light_factor_uniforms));
}

static void generic_model_prepare_for_draw(model_t* this)
{
  generic_model_t* _this = (generic_model_t*)this;
  context_update_buffer_data(_this->_context, _this->_uniform_buffers.world,
                             sizeof(_this->world_uniform_per),
                             &_this->world_uniform_per,
                             sizeof(_this->world_uniform_per));
}

static void generic_model_draw(model_t* this)
{
  generic_model_t* _this            = (generic_model_t*)this;
  WGPURenderPassEncoder render_pass = _this->_context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, _this->_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    _this->_context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    _this->_context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, _this->_bind_groups.model,
                                    0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 3, _this->_bind_groups.per, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, _this->buffers.position->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, _this->buffers.normal->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 2, _this->buffers.tex_coord->buffer, 0, WGPU_WHOLE_SIZE);
  /* diffuseShader doesn't have to input tangent buffer or binormal buffer. */
  if (_this->buffers.tangent->valid && _this->buffers.bi_normal->valid
      && _this->_model._name != MODELNAME_MODELGLOBEBASE) {
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 3, _this->buffers.tangent->buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 4, _this->buffers.bi_normal->buffer, 0, WGPU_WHOLE_SIZE);
  }
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, _this->buffers.indices->buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(render_pass,
                                   _this->buffers.indices->total_components,
                                   _this->_instance, 0, 0, 0);
  _this->_instance = 0;
}

static void generic_model_update_per_instance_uniforms(
  model_t* self, const world_uniforms_t* world_uniforms)
{
  generic_model_t* _this = (generic_model_t*)self;
  memcpy(&_this->world_uniform_per.world_uniforms[_this->_instance],
         world_uniforms, sizeof(world_uniforms_t));
  _this->_instance++;
}

/* -------------------------------------------------------------------------- *
 * Inner model - Defines the inner model.
 * -------------------------------------------------------------------------- */

typedef struct {
  model_t _model;
  struct {
    float eta;
    float tank_color_fudge;
    float refraction_fudge;
    float padding;
  } inner_uniforms;
  world_uniforms_t world_uniform_per;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* tangent;
    buffer_dawn_t* bi_normal;
    buffer_dawn_t* indices;
  } buffers;
  WGPUVertexState _vertex_state;
  WGPURenderPipeline _pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } _bind_group_layouts;
  WGPUPipelineLayout _pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } _bind_groups;
  struct {
    WGPUBuffer inner;
    WGPUBuffer view;
  } _uniform_buffers;
  wgpu_context_t* _wgpu_context;
  context_t* _context;
  program_t* _program;
  aquarium_t* _aquarium;
} inner_model_t;

static void inner_model_destroy(model_t* this);
static void inner_model_init(model_t* this);
static void inner_model_prepare_for_draw(model_t* this);
static void inner_model_draw(model_t* this);
static void inner_model_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms);

static void inner_model_init_defaults(inner_model_t* this)
{
  memset(this, 0, sizeof(*this));

  this->inner_uniforms.eta              = 1.0f;
  this->inner_uniforms.tank_color_fudge = 0.796f;
  this->inner_uniforms.refraction_fudge = 3.0f;
}

static void inner_model_init_virtual_method_table(inner_model_t* this)
{
  /* Override model functions */
  this->_model._vtbl.destroy          = inner_model_destroy;
  this->_model._vtbl.init             = inner_model_init;
  this->_model._vtbl.prepare_for_draw = inner_model_prepare_for_draw;
  this->_model._vtbl.draw             = inner_model_draw;
  this->_model._vtbl.update_per_instance_uniforms
    = inner_model_update_per_instance_uniforms;
}

static void inner_model_create(inner_model_t* this, context_t* context,
                               aquarium_t* aquarium, model_group_t type,
                               model_name_t name, bool blend)
{
  inner_model_init_defaults(this);

  /* Create model and set function pointers */
  model_create(&this->_model, type, name, blend);
  inner_model_init_virtual_method_table(this);

  this->_wgpu_context = context->wgpu_context;
  this->_context      = context;
  this->_aquarium     = aquarium;
}

static void inner_model_destroy(model_t* this)
{
  inner_model_t* _this = (inner_model_t*)this;

  WGPU_RELEASE_RESOURCE(RenderPipeline, _this->_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, _this->_pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.inner)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.view)
}

static void inner_model_init(model_t* this)
{
  inner_model_t* _this = (inner_model_t*)this;

  _this->_program            = _this->_model._program;
  WGPUShaderModule vs_module = program_get_vs_module(_this->_program);

  texture_t** texture_map    = _this->_model.texture_map;
  _this->textures.diffuse    = texture_map[TEXTURETYPE_DIFFUSE];
  _this->textures.normal     = texture_map[TEXTURETYPE_NORMAL_MAP];
  _this->textures.reflection = texture_map[TEXTURETYPE_REFLECTION_MAP];
  _this->textures.skybox     = texture_map[TEXTURETYPE_SKYBOX];

  buffer_dawn_t** buffer_map = _this->_model.buffer_map;
  _this->buffers.position    = buffer_map[BUFFERTYPE_POSITION];
  _this->buffers.normal      = buffer_map[BUFFERTYPE_NORMAL];
  _this->buffers.tex_coord   = buffer_map[BUFFERTYPE_TEX_COORD];
  _this->buffers.tangent     = buffer_map[BUFFERTYPE_TANGENT];
  _this->buffers.bi_normal   = buffer_map[BUFFERTYPE_BI_NORMAL];
  _this->buffers.indices     = buffer_map[BUFFERTYPE_INDICES];

  WGPUVertexAttribute vertex_attributes[5] = {
    [0] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = 0,
      .shaderLocation = 2,
    },
    [3] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 3,
    },
    [4] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 4,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[5] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.position),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride = buffer_dawn_get_data_size(_this->buffers.normal),
      .stepMode = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.tex_coord),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[2],
    },
    [3] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.tangent),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[3],
    },
    [4] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[4],
    },
  };

  _this->_vertex_state.module     = vs_module;
  _this->_vertex_state.entryPoint = "main";
  _this->_vertex_state.bufferCount
    = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  _this->_vertex_state.buffers = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[7] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [5] = (WGPUBindGroupLayoutEntry) {
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [6] = (WGPUBindGroupLayoutEntry) {
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    _this->_bind_group_layouts.model = context_make_bind_group_layout(
      _this->_context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    _this->_bind_group_layouts.per = context_make_bind_group_layout(
      _this->_context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    _this->_context->bind_group_layouts.general, /* Group 0 */
    _this->_context->bind_group_layouts.world,   /* Group 1 */
    _this->_bind_group_layouts.model,            /* Group 2 */
    _this->_bind_group_layouts.per,              /* Group 3 */
  };
  _this->_pipeline_layout = context_make_basic_pipeline_layout(
    _this->_context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  _this->_pipeline = context_create_render_pipeline(
    _this->_context, _this->_pipeline_layout, _this->_model._program,
    &_this->_vertex_state, _this->_model._blend);

  _this->_uniform_buffers.inner = context_create_buffer_from_data(
    _this->_context, &_this->inner_uniforms, sizeof(_this->inner_uniforms),
    sizeof(_this->inner_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  _this->_uniform_buffers.view = context_create_buffer_from_data(
    _this->_context, &_this->world_uniform_per, sizeof(world_uniforms_t),
    calc_constant_buffer_byte_size(sizeof(world_uniforms_t)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  {
    WGPUBindGroupEntry bg_entries[7] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = _this->_uniform_buffers.inner,
        .offset  = 0,
        .size    = sizeof(_this->inner_uniforms)
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .sampler = _this->textures.reflection->sampler,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = _this->textures.skybox->sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding     = 3,
        .textureView = _this->textures.diffuse->view,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding     = 4,
        .textureView = _this->textures.normal->view,
      },
      [5] = (WGPUBindGroupEntry) {
        .binding     = 5,
        .textureView = _this->textures.reflection->view,
      },
      [6] = (WGPUBindGroupEntry) {
        .binding     = 6,
        .textureView = _this->textures.skybox->view,
      },
    };
    _this->_bind_groups.model = context_make_bind_group(
      _this->_context, _this->_bind_group_layouts.model, bg_entries,
      (uint32_t)ARRAY_SIZE(bg_entries));
  }

  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = _this->_uniform_buffers.view,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(world_uniforms_t)),
      },
    };
    _this->_bind_groups.per
      = context_make_bind_group(_this->_context, _this->_bind_group_layouts.per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  context_set_buffer_data(_this->_context, _this->_uniform_buffers.inner,
                          sizeof(_this->inner_uniforms), &_this->inner_uniforms,
                          sizeof(_this->inner_uniforms));
}

static void inner_model_prepare_for_draw(model_t* this)
{
  UNUSED_VAR(this);
}

static void inner_model_draw(model_t* this)
{
  inner_model_t* _this              = (inner_model_t*)this;
  WGPURenderPassEncoder render_pass = _this->_context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, _this->_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    _this->_context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    _this->_context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, _this->_bind_groups.model,
                                    0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 3, _this->_bind_groups.per, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, _this->buffers.position->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, _this->buffers.normal->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 2, _this->buffers.tex_coord->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 3, _this->buffers.tangent->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 4, _this->buffers.bi_normal->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, _this->buffers.indices->buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(
    render_pass, _this->buffers.indices->total_components, 1, 0, 0, 0);
}

static void
inner_model_update_per_instance_uniforms(model_t* this,
                                         const world_uniforms_t* world_uniforms)
{
  inner_model_t* _this = (inner_model_t*)this;

  memcpy(&_this->world_uniform_per, world_uniforms, sizeof(world_uniforms_t));

  context_update_buffer_data(
    _this->_context, _this->_uniform_buffers.view,
    calc_constant_buffer_byte_size(sizeof(world_uniforms_t)),
    &_this->world_uniform_per, sizeof(world_uniforms_t));
}

/* -------------------------------------------------------------------------- *
 * Outside model - Defines outside model.
 * -------------------------------------------------------------------------- */

typedef struct {
  model_t _model;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* tangent;
    buffer_dawn_t* bi_normal;
    buffer_dawn_t* indices;
  } buffers;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  world_uniforms_t world_uniform_per[20];
  WGPUVertexState _vertex_state;
  WGPURenderPipeline _pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } _bind_group_layouts;
  WGPUPipelineLayout _pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } _bind_groups;
  struct {
    WGPUBuffer light_factor;
    WGPUBuffer view;
  } _uniform_buffers;
  wgpu_context_t* _wgpu_context;
  context_t* _context;
  program_t* _program;
  aquarium_t* _aquarium;
} outside_model_t;

static void outside_model_destroy(model_t* this);
static void outside_model_init(model_t* this);
static void outside_model_prepare_for_draw(model_t* this);
static void outside_model_draw(model_t* this);
static void outside_model_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms);

static void outside_model_init_defaults(outside_model_t* this)
{
  memset(this, 0, sizeof(*this));

  this->light_factor_uniforms.shininess       = 50.0f;
  this->light_factor_uniforms.specular_factor = 0.0f;
}

static void outside_model_init_virtual_method_table(outside_model_t* this)
{
  /* Override model functions */
  this->_model._vtbl.destroy          = outside_model_destroy;
  this->_model._vtbl.init             = outside_model_init;
  this->_model._vtbl.prepare_for_draw = outside_model_prepare_for_draw;
  this->_model._vtbl.draw             = outside_model_draw;
  this->_model._vtbl.update_per_instance_uniforms
    = outside_model_update_per_instance_uniforms;
}

static void outside_model_create(outside_model_t* this, context_t* context,
                                 aquarium_t* aquarium, model_group_t type,
                                 model_name_t name, bool blend)
{
  outside_model_init_defaults(this);

  /* Create model and set function pointers */
  model_create(&this->_model, type, name, blend);
  outside_model_init_virtual_method_table(this);

  this->_wgpu_context = context->wgpu_context;
  this->_context      = context;
  this->_aquarium     = aquarium;
}

static void outside_model_destroy(model_t* this)
{
  outside_model_t* _this = (outside_model_t*)this;

  WGPU_RELEASE_RESOURCE(RenderPipeline, _this->_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, _this->_pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.light_factor)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.view)
}

static void outside_model_init(model_t* this)
{
  outside_model_t* _this = (outside_model_t*)this;

  _this->_program            = _this->_model._program;
  WGPUShaderModule vs_module = program_get_vs_module(_this->_program);

  texture_t** texture_map    = _this->_model.texture_map;
  _this->textures.diffuse    = texture_map[TEXTURETYPE_DIFFUSE];
  _this->textures.normal     = texture_map[TEXTURETYPE_NORMAL_MAP];
  _this->textures.reflection = texture_map[TEXTURETYPE_REFLECTION_MAP];
  _this->textures.skybox     = texture_map[TEXTURETYPE_SKYBOX];

  buffer_dawn_t** buffer_map = _this->_model.buffer_map;
  _this->buffers.position    = buffer_map[BUFFERTYPE_POSITION];
  _this->buffers.normal      = buffer_map[BUFFERTYPE_NORMAL];
  _this->buffers.tex_coord   = buffer_map[BUFFERTYPE_TEX_COORD];
  _this->buffers.tangent     = buffer_map[BUFFERTYPE_TANGENT];
  _this->buffers.bi_normal   = buffer_map[BUFFERTYPE_BI_NORMAL];
  _this->buffers.indices     = buffer_map[BUFFERTYPE_INDICES];

  WGPUVertexAttribute vertex_attributes[5] = {
    [0] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = 0,
      .shaderLocation = 2,
    },
    [3] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 3,
    },
    [4] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 4,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[5] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.position),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.tex_coord),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[2],
    },
    [3] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.tangent),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[3],
    },
    [4] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(_this->buffers.normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[4],
    },
  };

  _this->_vertex_state.module     = vs_module;
  _this->_vertex_state.entryPoint = "main";
  _this->_vertex_state.bufferCount
    = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  _this->_vertex_state.buffers = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    _this->_bind_group_layouts.per = context_make_bind_group_layout(
      _this->_context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  /* Outside models use diffuse shaders. */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    _this->_bind_group_layouts.model = context_make_bind_group_layout(
      _this->_context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    _this->_context->bind_group_layouts.general, /* Group 0 */
    _this->_context->bind_group_layouts.world,   /* Group 1 */
    _this->_bind_group_layouts.model,            /* Group 2 */
    _this->_bind_group_layouts.per,              /* Group 3 */
  };
  _this->_pipeline_layout = context_make_basic_pipeline_layout(
    _this->_context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  _this->_pipeline = context_create_render_pipeline(
    _this->_context, _this->_pipeline_layout, _this->_model._program,
    &_this->_vertex_state, _this->_model._blend);

  _this->_uniform_buffers.light_factor = context_create_buffer_from_data(
    _this->_context, &_this->light_factor_uniforms,
    sizeof(_this->light_factor_uniforms), sizeof(_this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  _this->_uniform_buffers.view = context_create_buffer_from_data(
    _this->_context, &_this->world_uniform_per, sizeof(world_uniforms_t) * 20,
    calc_constant_buffer_byte_size(sizeof(world_uniforms_t) * 20),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = _this->_uniform_buffers.light_factor,
        .offset  = 0,
        .size    = sizeof(_this->light_factor_uniforms)
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .sampler = _this->textures.diffuse->sampler,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding     = 2,
        .textureView = _this->textures.diffuse->view,
      },
    };
    _this->_bind_groups.model = context_make_bind_group(
      _this->_context, _this->_bind_group_layouts.model, bg_entries,
      (uint32_t)ARRAY_SIZE(bg_entries));
  }

  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = _this->_uniform_buffers.view,
        .offset  = 0,
        .size  = calc_constant_buffer_byte_size(sizeof(world_uniforms_t) * 20),
      },
    };
    _this->_bind_groups.per
      = context_make_bind_group(_this->_context, _this->_bind_group_layouts.per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  context_set_buffer_data(_this->_context, _this->_uniform_buffers.light_factor,
                          sizeof(light_uniforms_t),
                          &_this->light_factor_uniforms,
                          sizeof(light_uniforms_t));
}

static void outside_model_prepare_for_draw(model_t* this)
{
  UNUSED_VAR(this);
}

static void outside_model_draw(model_t* this)
{
  outside_model_t* _this            = (outside_model_t*)this;
  WGPURenderPassEncoder render_pass = _this->_context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, _this->_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    _this->_context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    _this->_context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, _this->_bind_groups.model,
                                    0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 3, _this->_bind_groups.per, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, _this->buffers.position->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, _this->buffers.normal->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 2, _this->buffers.tex_coord->buffer, 0, WGPU_WHOLE_SIZE);
  /* diffuseShader doesn't have to input tangent buffer or binormal buffer. */
  if (_this->buffers.tangent->valid && _this->buffers.bi_normal->valid) {
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 3, _this->buffers.tangent->buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 4, _this->buffers.bi_normal->buffer, 0, WGPU_WHOLE_SIZE);
  }
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, _this->buffers.indices->buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(
    render_pass, _this->buffers.indices->total_components, 1, 0, 0, 0);
}

static void outside_model_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms)
{
  outside_model_t* _this = (outside_model_t*)this;

  memcpy(&_this->world_uniform_per, world_uniforms, sizeof(world_uniforms_t));

  context_update_buffer_data(
    _this->_context, _this->_uniform_buffers.view,
    calc_constant_buffer_byte_size(sizeof(world_uniforms_t) * 20),
    &_this->world_uniform_per, sizeof(world_uniforms_t));
}

/* -------------------------------------------------------------------------- *
 * Seaweed model - Defines the seaweed model.
 * -------------------------------------------------------------------------- */

typedef struct {
  float time;
  vec3 padding;
} seaweed_t;

struct seaweed_model_t;

typedef struct seaweed_model_vtbl_t {
  void (*update_seaweed_model_time)(struct seaweed_model_t* self, float time);
} seaweed_model_vtbl_t;

typedef struct seaweed_model_t {
  seaweed_model_vtbl_t _vtbl;
  model_t _model;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* indices;
  } buffers;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  struct {
    seaweed_t seaweed[20];
  } seaweed_per;
  struct {
    world_uniforms_t world_uniforms[20];
  } world_uniform_per;
  WGPUVertexState _vertex_state;
  WGPURenderPipeline _pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } _bind_group_layouts;
  WGPUPipelineLayout _pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } _bind_groups;
  struct {
    WGPUBuffer light_factor;
    WGPUBuffer time;
    WGPUBuffer view;
  } _uniform_buffers;
  wgpu_context_t* _wgpu_context;
  context_t* _context;
  program_t* _program;
  aquarium_t* _aquarium;
  int32_t _instance;
} seaweed_model_t;

static void seaweed_model_destroy(model_t* this);
static void seaweed_model_init(model_t* this);
static void seaweed_model_prepare_for_draw(model_t* this);
static void seaweed_model_draw(model_t* this);
static void seaweed_model_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms);
static void seaweed_model_update_seaweed_model_time(seaweed_model_t* this,
                                                    float time);

static void seaweed_model_init_defaults(seaweed_model_t* this)
{
  memset(this, 0, sizeof(*this));

  this->light_factor_uniforms.shininess       = 50.0f;
  this->light_factor_uniforms.specular_factor = 1.0f;

  this->_instance = 0;
}

static void seaweed_model_init_virtual_method_table(seaweed_model_t* this)
{
  /* Override model functions */
  this->_model._vtbl.destroy          = seaweed_model_destroy;
  this->_model._vtbl.init             = seaweed_model_init;
  this->_model._vtbl.prepare_for_draw = seaweed_model_prepare_for_draw;
  this->_model._vtbl.draw             = seaweed_model_draw;
  this->_model._vtbl.update_per_instance_uniforms
    = seaweed_model_update_per_instance_uniforms;

  /* Override seaweed model functions */
  this->_vtbl.update_seaweed_model_time
    = seaweed_model_update_seaweed_model_time;
}

static void seaweed_model_create(seaweed_model_t* this, context_t* context,
                                 aquarium_t* aquarium, model_group_t type,
                                 model_name_t name, bool blend)
{
  seaweed_model_init_defaults(this);

  /* Create model and set function pointers */
  model_create(&this->_model, type, name, blend);
  seaweed_model_init_virtual_method_table(this);

  this->_wgpu_context = context->wgpu_context;
  this->_context      = context;
  this->_aquarium     = aquarium;
}

static void seaweed_model_init(model_t* this)
{
  seaweed_model_t* _this       = (seaweed_model_t*)this;
  wgpu_context_t* wgpu_context = _this->_wgpu_context;

  _this->_program            = _this->_model._program;
  WGPUShaderModule vs_module = program_get_vs_module(_this->_program);

  texture_t** texture_map    = _this->_model.texture_map;
  _this->textures.diffuse    = texture_map[TEXTURETYPE_DIFFUSE];
  _this->textures.normal     = texture_map[TEXTURETYPE_NORMAL_MAP];
  _this->textures.reflection = texture_map[TEXTURETYPE_REFLECTION_MAP];
  _this->textures.skybox     = texture_map[TEXTURETYPE_SKYBOX];

  buffer_dawn_t** buffer_map = _this->_model.buffer_map;
  _this->buffers.position    = buffer_map[BUFFERTYPE_POSITION];
  _this->buffers.normal      = buffer_map[BUFFERTYPE_NORMAL];
  _this->buffers.tex_coord   = buffer_map[BUFFERTYPE_TEX_COORD];
  _this->buffers.indices     = buffer_map[BUFFERTYPE_INDICES];

  WGPUVertexAttribute vertex_attributes[3] = {
    [0] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x2,
      .offset = 0,
      .shaderLocation = 2,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[3] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride = buffer_dawn_get_data_size(_this->buffers.position),
      .stepMode = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride = buffer_dawn_get_data_size(_this->buffers.normal),
      .stepMode = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride = buffer_dawn_get_data_size(_this->buffers.tex_coord),
      .stepMode = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes = &vertex_attributes[2],
    },
  };

  _this->_vertex_state.module     = vs_module;
  _this->_vertex_state.entryPoint = "main";
  _this->_vertex_state.bufferCount
    = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  _this->_vertex_state.buffers = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    _this->_bind_group_layouts.model = context_make_bind_group_layout(
      _this->_context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    _this->_bind_group_layouts.per = context_make_bind_group_layout(
      _this->_context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    _this->_context->bind_group_layouts.general, /* Group 0 */
    _this->_context->bind_group_layouts.world,   /* Group 1 */
    _this->_bind_group_layouts.model,            /* Group 2 */
    _this->_bind_group_layouts.per,              /* Group 3 */
  };

  _this->_pipeline_layout = context_make_basic_pipeline_layout(
    _this->_context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  _this->_pipeline = context_create_render_pipeline(
    _this->_context, _this->_pipeline_layout, _this->_program,
    &_this->_vertex_state, _this->_model._blend);

  _this->_uniform_buffers.light_factor = context_create_buffer_from_data(
    _this->_context, &_this->light_factor_uniforms,
    sizeof(_this->light_factor_uniforms), sizeof(_this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  _this->_uniform_buffers.time = context_create_buffer_from_data(
    _this->_context, &_this->seaweed_per, sizeof(_this->seaweed_per),
    calc_constant_buffer_byte_size(sizeof(_this->seaweed_per)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  _this->_uniform_buffers.view = context_create_buffer_from_data(
    _this->_context, &_this->world_uniform_per,
    sizeof(_this->world_uniform_per),
    calc_constant_buffer_byte_size(sizeof(_this->world_uniform_per)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = _this->_uniform_buffers.light_factor,
        .offset  = 0,
        .size    = sizeof(_this->light_factor_uniforms)
      },
      [1] = (WGPUBindGroupEntry){
         .binding = 1,
         .sampler = _this->textures.diffuse->sampler,
      },
      [2] = (WGPUBindGroupEntry){
        .binding     = 2,
        .textureView = _this->textures.diffuse->view,
      },
      };
    _this->_bind_groups.model = context_make_bind_group(
      _this->_context, _this->_bind_group_layouts.model, bg_entries,
      (uint32_t)ARRAY_SIZE(bg_entries));
  }

  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = _this->_uniform_buffers.view,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(_this->world_uniform_per)),
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = _this->_uniform_buffers.time,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(_this->seaweed_per)),
      },
    };
    _this->_bind_groups.per
      = context_make_bind_group(_this->_context, _this->_bind_group_layouts.per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  context_set_buffer_data(wgpu_context, _this->_uniform_buffers.light_factor,
                          sizeof(_this->light_factor_uniforms),
                          &_this->light_factor_uniforms,
                          sizeof(_this->light_factor_uniforms));
}

static void seaweed_model_destroy(model_t* this)
{
  seaweed_model_t* _this = (seaweed_model_t*)this;

  WGPU_RELEASE_RESOURCE(RenderPipeline, _this->_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, _this->_bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, _this->_pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, _this->_bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.light_factor)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.time)
  WGPU_RELEASE_RESOURCE(Buffer, _this->_uniform_buffers.view)
}

static void seaweed_model_prepare_for_draw(model_t* this)
{
  seaweed_model_t* _this = (seaweed_model_t*)this;

  context_update_buffer_data(
    _this->_wgpu_context, _this->_uniform_buffers.view,
    calc_constant_buffer_byte_size(sizeof(_this->world_uniform_per)),
    &_this->world_uniform_per, sizeof(_this->world_uniform_per));
  context_update_buffer_data(
    _this->_wgpu_context, _this->_uniform_buffers.time,
    calc_constant_buffer_byte_size(sizeof(_this->seaweed_per)),
    &_this->seaweed_per, sizeof(_this->seaweed_per));
}

static void seaweed_model_draw(model_t* this)
{
  seaweed_model_t* _this            = (seaweed_model_t*)this;
  WGPURenderPassEncoder render_pass = _this->_context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, _this->_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    _this->_context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    _this->_context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, _this->_bind_groups.model,
                                    0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 3, _this->_bind_groups.per, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, _this->buffers.position->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, _this->buffers.normal->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 2, _this->buffers.tex_coord->buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, _this->buffers.indices->buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(
    render_pass, _this->buffers.indices->total_components, 1, 0, 0, 0);
  _this->_instance = 0;
}

static void seaweed_model_update_per_instance_uniforms(
  model_t* this, const world_uniforms_t* world_uniforms)
{
  seaweed_model_t* _this = (seaweed_model_t*)this;

  memcpy(&_this->world_uniform_per.world_uniforms[_this->_instance],
         world_uniforms, sizeof(world_uniforms_t));
  _this->seaweed_per.seaweed[_this->_instance].time
    = _this->_aquarium->g.mclock + _this->_instance;

  _this->_instance++;
}

static void seaweed_model_update_seaweed_model_time(seaweed_model_t* this,
                                                    float time)
{
  this->_vtbl.update_seaweed_model_time(this, time);
}

/* -------------------------------------------------------------------------- *
 * Helper functions.
 * -------------------------------------------------------------------------- */

/* Load world matrices of models from json file. */
static int32_t aquarium_load_placement(aquarium_t* this)
{
  int32_t status = EXIT_FAILURE;
  const resource_helper_t* resource_helper
    = context_get_resource_helper(&this->context);
  const char* prop_path
    = resource_helper_get_prop_placement_path(resource_helper);
  if (!file_exists(prop_path)) {
    log_fatal("Could not load placements file %s", prop_path);
    return status;
  }
  file_read_result_t file_read_result = {0};
  read_file(prop_path, &file_read_result, true);
  const char* const placement = (const char* const)file_read_result.data;

  const cJSON* objects           = NULL;
  const cJSON* object            = NULL;
  const cJSON* name              = NULL;
  const cJSON* world_matrix      = NULL;
  const cJSON* world_matrix_item = NULL;
  cJSON* placement_json          = cJSON_Parse(placement);
  if (placement_json == NULL) {
    const char* error_ptr = cJSON_GetErrorPtr();
    if (error_ptr != NULL) {
      fprintf(stderr, "Error before: %s\n", error_ptr);
    }
    goto load_placement_end;
  }

  if (!cJSON_IsObject(placement_json)
      || !cJSON_HasObjectItem(placement_json, "objects")) {
    fprintf(stderr, "Invalid placements file\n");
    goto load_placement_end;
  }

  objects = cJSON_GetObjectItemCaseSensitive(placement_json, "objects");
  if (!cJSON_IsArray(objects)) {
    fprintf(stderr, "Objects item is not an array\n");
    goto load_placement_end;
  }

  model_name_t model_name         = MODELNAME_MODELMAX;
  uint16_t model_world_matrix_ctr = 0;

  cJSON_ArrayForEach(object, objects)
  {
    name       = cJSON_GetObjectItemCaseSensitive(object, "name");
    model_name = MODELNAME_MODELMAX;
    if (cJSON_IsString(name) && (name->valuestring != NULL)) {
      model_name
        = aquarium_map_model_name_str_to_model_name(this, name->valuestring);
    }

    world_matrix = cJSON_GetObjectItemCaseSensitive(object, "worldMatrix");
    if (model_name != MODELNAME_MODELMAX && cJSON_IsArray(world_matrix)
        && cJSON_GetArraySize(world_matrix) == 16) {
      model_t* model = (model_t*)this->aquarium_models[model_name];
      world_matrix_t* model_world_matrix
        = &model->world_matrices[model->world_matrix_count++];
      model_world_matrix_ctr = 0;

      cJSON_ArrayForEach(world_matrix_item, world_matrix)
      {
        if (!cJSON_IsNumber(world_matrix_item)) {
          goto load_placement_end;
        }
        (*model_world_matrix)[model_world_matrix_ctr++]
          = (float)world_matrix_item->valuedouble;
      }
    }
  }

  status = EXIT_SUCCESS;

load_placement_end:
  cJSON_Delete(placement_json);
  free(file_read_result.data);
  return status;
}

static int32_t aquarium_load_fish_scenario(aquarium_t* this)
{
  for (uint32_t i = 0; i < FISH_BEHAVIOR_COUNT; ++i) {
    behavior_create(&this->fish_behaviors[i], g_fish_behaviors[i].frame,
                    g_fish_behaviors[i].op, g_fish_behaviors[i].count);
    sc_queue_add_last(&this->fish_behavior, &this->fish_behaviors[i]);
  }

  return EXIT_SUCCESS;
}

static model_t* context_create_model(context_t* this, aquarium_t* aquarium,
                                     model_group_t type, model_name_t name,
                                     bool blend)
{
  model_t* model = NULL;
  switch (type) {
    case MODELGROUP_FISH: {
      fish_model_draw_t* _model = malloc(sizeof(fish_model_draw_t));
      fish_model_draw_create(_model, this, aquarium, type, name, blend);
      model = (model_t*)_model;
    } break;
    case MODELGROUP_FISHINSTANCEDDRAW: {
      fish_model_instanced_draw_t* _model
        = malloc(sizeof(fish_model_instanced_draw_t));
      fish_model_instanced_draw_create(_model, this, aquarium, type, name,
                                       blend);
      model = (model_t*)_model;
    } break;
    case MODELGROUP_GENERIC: {
      generic_model_t* _model = malloc(sizeof(generic_model_t));
      generic_model_create(_model, this, aquarium, type, name, blend);
      model = (model_t*)_model;
    } break;
    case MODELGROUP_INNER: {
      inner_model_t* _model = malloc(sizeof(inner_model_t));
      inner_model_create(_model, this, aquarium, type, name, blend);
      model = (model_t*)_model;
    } break;
    case MODELGROUP_SEAWEED: {
      seaweed_model_t* _model = malloc(sizeof(seaweed_model_t));
      seaweed_model_create(_model, this, aquarium, type, name, blend);
      model = (model_t*)_model;
    } break;
    case MODELGROUP_OUTSIDE: {
      outside_model_t* _model = malloc(sizeof(outside_model_t));
      outside_model_create(_model, this, aquarium, type, name, blend);
      model = (model_t*)_model;
    } break;
    default: {
      log_error("Can not create model type");
    } break;
  }

  return model;
}

/* Load vertex and index buffers, textures and program for each model. */
static int32_t aquarium_load_model(aquarium_t* this, const g_scene_info_t* info)
{
  int32_t status = EXIT_FAILURE;
  const resource_helper_t* resource_helper
    = context_get_resource_helper(&this->context);
  const char* image_path   = resource_helper_get_image_path(resource_helper);
  const char* program_path = resource_helper_get_program_path(resource_helper);
  char model_path[STRMAX]  = {0};
  resource_helper_get_model_path(resource_helper, info->name_str, &model_path);
  if (!file_exists(model_path)) {
    log_fatal("Could not load model file %s", model_path);
    return status;
  }

  file_read_result_t file_read_result = {0};
  read_file(model_path, &file_read_result, true);
  const char* const model_data = (const char* const)file_read_result.data;

  const cJSON* models        = NULL;
  const cJSON* model_item    = NULL;
  const cJSON* texture_array = NULL;
  const cJSON* texture_item  = NULL;
  const cJSON* arrays        = NULL;
  const cJSON* array_item    = NULL;
  cJSON* model_json          = cJSON_Parse(model_data);
  if (model_json == NULL) {
    const char* error_ptr = cJSON_GetErrorPtr();
    if (error_ptr != NULL) {
      fprintf(stderr, "Error before: %s\n", error_ptr);
    }
    goto load_model_end;
  }

  if (!cJSON_IsObject(model_json)
      || !cJSON_HasObjectItem(model_json, "models")) {
    fprintf(stderr, "Invalid models file\n");
    goto load_model_end;
  }

  models = cJSON_GetObjectItemCaseSensitive(model_json, "models");
  if (!cJSON_IsArray(models)) {
    fprintf(stderr, "Models item is not an array\n");
    goto load_model_end;
  }

  model_t* model = NULL;
  if (aquarium_settings.enable_alpha_blending && info->type != MODELGROUP_INNER
      && info->type != MODELGROUP_OUTSIDE) {
    model = context_create_model(&this->context, this, info->type, info->name,
                                 true);
  }
  else {
    model = context_create_model(&this->context, this, info->type, info->name,
                                 info->blend);
  }

  model_item = cJSON_GetArrayItem(models, cJSON_GetArraySize(models) - 1);
  {
    /* Set up textures */
    texture_array = cJSON_GetObjectItemCaseSensitive(model_item, "textures");
    cJSON_ArrayForEach(texture_item, texture_array)
    {
      const char* name  = texture_item->string;
      const char* image = texture_item->valuestring;

      if (aquarium_texture_map_lookup_index(this, image) == -1) {
        char image_url[STRMAX] = {0};
        snprintf(image_url, sizeof(image_url), "%s%s", image_path, image);
        texture_t texture
          = context_create_texture(&this->context, name, image_url);
        aquarium_texture_map_insert(this, image, &texture);
      }

      texture_type_t texture_type = string_to_texture_type(name);
      ASSERT(texture_type < TEXTURETYPE_MAX);
      model->texture_map[texture_type]
        = aquarium_texture_map_lookup_texture(this, image);
    }

    /* Set up vertices */
    arrays = cJSON_GetObjectItemCaseSensitive(model_item, "fields");
    cJSON_ArrayForEach(array_item, arrays)
    {
      const char* name = array_item->string;
      int32_t num_components
        = cJSON_GetObjectItemCaseSensitive(array_item, "numComponents")
            ->valueint;
      const char* type
        = cJSON_GetObjectItemCaseSensitive(array_item, "type")->valuestring;
      buffer_dawn_t* buffer = NULL;
      if (strcmp(name, "indices") == 0) {
        ASSERT(strcmp(type, "Uint16Array") == 0);
        const cJSON* data
          = cJSON_GetObjectItemCaseSensitive(array_item, "data");
        if (data != NULL && cJSON_IsArray(data)) {
          uint16_t* vec
            = (uint16_t*)malloc(cJSON_GetArraySize(data) * sizeof(uint16_t));
          uint64_t vec_count     = 0;
          const cJSON* data_item = NULL;
          cJSON_ArrayForEach(data_item, data)
          {
            if (cJSON_IsNumber(data_item)) {
              vec[vec_count++] = (float)data_item->valuedouble;
            }
          }
          buffer = context_create_buffer_uint16(
            &this->context, num_components, vec, vec_count, vec_count, false);
          free(vec);
        }
      }
      else {
        ASSERT(strcmp(type, "Float32Array") == 0);
        const cJSON* data
          = cJSON_GetObjectItemCaseSensitive(array_item, "data");
        if (data != NULL && cJSON_IsArray(data)) {
          float* vec = (float*)malloc(cJSON_GetArraySize(data) * sizeof(float));
          uint64_t vec_count     = 0;
          const cJSON* data_item = NULL;
          cJSON_ArrayForEach(data_item, data)
          {
            if (cJSON_IsNumber(data_item)) {
              vec[vec_count++] = (float)data_item->valuedouble;
            }
          }
          buffer = context_create_buffer_f32(&this->context, num_components,
                                             vec, vec_count, false);
          free(vec);
        }
      }

      if (buffer != NULL) {
        buffer_type_t buffer_type = string_to_buffer_type(name);
        ASSERT(buffer_type < BUFFERTYPE_MAX);
        model->buffer_map[buffer_type] = buffer;
      }
    }

    /**
     * setup program
     * There are 3 programs
     * DM
     * DM+NM
     * DM+NM+RM
     */
    char vs_id[STRMAX];
    char fs_id[STRMAX];
    char concat_id[STRMAX * 2];

    snprintf(vs_id, sizeof(vs_id), "%s", info->program.vertex);
    snprintf(fs_id, sizeof(fs_id), "%s", info->program.fragment);

    if ((strcmp(vs_id, "") == 0) && (strcmp(fs_id, "") == 0)) {
      model->texture_map[TEXTURETYPE_SKYBOX]
        = &this->texture_map[TEXTURETYPE_SKYBOX].value;
    }
    else if (model->texture_map[TEXTURETYPE_REFLECTION] != NULL) {
      snprintf(vs_id, sizeof(vs_id), "%s", "reflectionMapVertexShader");
      snprintf(fs_id, sizeof(fs_id), "%s", "reflectionMapFragmentShader");

      model->texture_map[TEXTURETYPE_SKYBOX]
        = &this->texture_map[TEXTURETYPE_SKYBOX].value;
    }
    else if (model->texture_map[TEXTURETYPE_NORMAL_MAP] != NULL) {
      snprintf(vs_id, sizeof(vs_id), "%s", "normalMapVertexShader");
      snprintf(fs_id, sizeof(fs_id), "%s", "normalMapFragmentShader");
    }
    else {
      snprintf(vs_id, sizeof(vs_id), "%s", "diffuseVertexShader");
      snprintf(fs_id, sizeof(fs_id), "%s", "diffuseFragmentShader");
    }

    program_t* program = NULL;
    snprintf(concat_id, sizeof(concat_id), "%s%s", vs_id, fs_id);
    if (aquarium_program_map_lookup_index(this, concat_id) != -1) {
      program = aquarium_program_map_lookup_program(this, concat_id);
    }
    else {
      char vs_id_path[STRMAX];
      char fs_id_path[STRMAX];
      snprintf(vs_id_path, sizeof(vs_id_path), "%s%s", program_path, vs_id);
      snprintf(fs_id_path, sizeof(fs_id_path), "%s%s", program_path, fs_id);
      program = context_create_program(&this->context, vs_id_path, fs_id_path);
      if (aquarium_settings.enable_alpha_blending
          && info->type != MODELGROUP_INNER
          && info->type != MODELGROUP_OUTSIDE) {
        program_set_options(program, true, this->g.alpha);
        program_compile_program(program);
      }
      else {
        program_set_options(program, false, this->g.alpha);
        program_compile_program(program);
      }
      aquarium_program_map_insert(this, concat_id, program);
    }

    model_set_program(model, program);
    model_init(model);
  }

load_model_end:
  cJSON_Delete(model_json);
  free(file_read_result.data);
  return status;
}

static void aquarium_update_and_draw(aquarium_t* this)
{
  global_t* g                      = &this->g;
  world_uniforms_t* world_uniforms = &this->world_uniforms;

  bool draw_per_model = aquarium_settings.draw_per_model;
  int32_t fish_begin  = aquarium_settings.enable_instanced_draw ?
                          MODELNAME_MODELSMALLFISHAINSTANCEDDRAWS :
                          MODELNAME_MODELSMALLFISHA;
  int32_t fish_end    = aquarium_settings.enable_instanced_draw ?
                          MODELNAME_MODELBIGFISHBINSTANCEDDRAWS :
                          MODELNAME_MODELBIGFISHB;

  for (uint32_t i = MODELNAME_MODELRUINCOLUMN; i <= MODELNAME_MODELSEAWEEDB;
       ++i) {
    model_t* model = this->aquarium_models[i];
    model_prepare_for_draw(model);

    for (uint32_t w = 0; w < model->world_matrix_count; ++i) {
      world_matrix_t* world_matrix = &model->world_matrices[i];
      memcpy(world_uniforms->world, *world_matrix, 16 * sizeof(float));
      matrix_mul_matrix_matrix4(
        world_uniforms->world_view_projection, world_uniforms->world,
        this->light_world_position_uniform.view_projection);
      matrix_inverse4(g->world_inverse, world_uniforms->world);
      matrix_transpose4(world_uniforms->world_inverse_transpose,
                        g->world_inverse);

      model_update_per_instance_uniforms(model, world_uniforms);
      if (!draw_per_model) {
        model_draw(model);
      }
    }
  }

  for (int i = fish_begin; i <= fish_end; ++i) {
    fish_model_t* model = (fish_model_t*)this->aquarium_models[i];
    model_prepare_for_draw((model_t*)model);

    const fish_t* fish_info = &fish_table[i - fish_begin];
    int numFish             = this->fish_count[i - fish_begin];
    float fish_base_clock   = g->mclock * g_settings.fish_speed;
    float fish_radius       = fish_info->radius;
    float fish_radius_range = fish_info->radius_range;
    float fish_speed        = fish_info->speed;
    float fish_speed_range  = fish_info->speed_range;
    float fish_tail_speed = fish_info->tail_speed * g_settings.fish_tail_speed;
    float fish_offset     = g_settings.fish_offset;
    // float fishClockSpeed  = g_fishSpeed;
    float fish_height = g_settings.fish_height + fish_info->height_offset;
    float fish_height_range
      = g_settings.fish_height_range * fish_info->height_range;
    float fish_xclock = g_settings.fish_xclock;
    float fish_yclock = g_settings.fish_yclock;
    float fish_zclock = g_settings.fish_zclock;

    for (int ii = 0; ii < numFish; ++ii) {
      float fish_clock = fish_base_clock + ii * fish_offset;
      float speed
        = fish_speed + ((float)matrix_pseudo_random()) * fish_speed_range;
      float scale = 1.0f + ((float)matrix_pseudo_random()) * 1.0f;
      float x_radius
        = fish_radius + ((float)matrix_pseudo_random()) * fish_radius_range;
      float y_radius
        = 2.0f + ((float)matrix_pseudo_random()) * fish_height_range;
      float z_radius
        = fish_radius + ((float)matrix_pseudo_random()) * fish_radius_range;
      float fishSpeedClock = fish_clock * speed;
      float x_clock        = fishSpeedClock * fish_xclock;
      float y_clock        = fishSpeedClock * fish_yclock;
      float z_clock        = fishSpeedClock * fish_zclock;

      fish_model_update_fish_per_uniforms(
        model, sin(x_clock) * x_radius, sin(y_clock) * y_radius + fish_height,
        cos(z_clock) * z_radius, sin(x_clock - 0.04f) * x_radius,
        sin(y_clock - 0.01f) * y_radius + fish_height,
        cos(z_clock - 0.04f) * z_radius, scale,
        fmod((g->mclock + ii * g_settings.tail_offset_mult) * fish_tail_speed
               * speed,
             PI * 2.0f),
        ii);

      if (!draw_per_model) {
        model_update_per_instance_uniforms((model_t*)model, world_uniforms);
        model_draw((model_t*)model);
      }
    }
  }

  if (draw_per_model) {
    context_update_all_fish_data(&this->context);
    context_begin_render_pass(&this->context);
    for (int i = 0; i <= MODELNAME_MODELMAX; ++i) {
      if (i >= MODELNAME_MODELSMALLFISHA && (i < fish_begin || i > fish_end)) {
        continue;
      }

      model_t* model = (model_t*)this->aquarium_models[i];
      model_draw(model);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Aquarium Example
 * -------------------------------------------------------------------------- */

// Other variables
static const char* example_title = "Aquarium";
static bool prepared             = false;
static aquarium_t aquarium       = {0};

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    aquarium_create(&aquarium);
    prepared = true;
    if (!aquarium_init(&aquarium)) {
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int example_draw(wgpu_example_context_t* context)
{
  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
}

void example_aquarium(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
      .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
