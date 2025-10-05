#include "webgpu/wgpu_common.h"

#include <stdbool.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Aquarium
 *
 * Aquarium is a complete port of the classic WebGL Aquarium to modern WebGPU,
 * showcasing advanced rendering techniques and efficient GPU programming.
 *
 * Ref:
 * https://github.com/webgfx/aquarium-web/tree/main/webgpu
 * https://github.com/webatintel/aquarium
 * https://webglsamples.org/aquarium/aquarium.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Config
 * -------------------------------------------------------------------------- */

typedef struct {
  float speed;
  float target_height;
  float target_radius;
  float eye_height;
  float eye_radius;
  float eye_speed;
  float field_of_view;
  float ambient_red;
  float ambient_green;
  float ambient_blue;
  float fog_power;
  float fog_mult;
  float fog_offset;
  float fog_red;
  float fog_green;
  float fog_blue;
} globals_t;

static globals_t default_globals = {
  .speed         = 1.0f,
  .target_height = 0.0f,
  .target_radius = 88.0f,
  .eye_height    = 38.0f,
  .eye_radius    = 69.0f,
  .eye_speed     = 0.06f,
  .field_of_view = 85.0f,
  .ambient_red   = 0.22f,
  .ambient_green = 0.25f,
  .ambient_blue  = 0.39f,
  .fog_power     = 14.5f,
  .fog_mult      = 1.66f,
  .fog_offset    = 0.53f,
  .fog_red       = 0.54f,
  .fog_green     = 0.86f,
  .fog_blue      = 1.0f,
};

static struct {
  float fish_height_range;
  float fish_height;
  float fish_speed;
  float fish_offset;
  float fish_xclock;
  float fish_yclock;
  float fish_zclock;
  float fish_tail_speed;
} default_fish = {
  .fish_height_range = 1.0f,
  .fish_height       = 25.0f,
  .fish_speed        = 0.124f,
  .fish_offset       = 0.52f,
  .fish_xclock       = 1.0f,
  .fish_yclock       = 0.556f,
  .fish_zclock       = 1.0f,
  .fish_tail_speed   = 1.0f,
};

typedef struct {
  float refraction_fudge;
  float eta;
  float tank_color_fudge;
} inner_const_t;

static inner_const_t default_inner_const = {
  .refraction_fudge = 3.0f,
  .eta              = 1.0f,
  .tank_color_fudge = 0.8f,
};

static struct {
  const char* id;
  const char* label;
  WGPUBool default_value
} option_definitions[8] = {
  // clang-format off
  { .id = "normalMaps", .label = "Normal Maps", .default_value = true  },
  { .id = "reflection", .label = "Reflection",  .default_value = true  },
  { .id = "tank",       .label = "Tank",        .default_value = true  },
  { .id = "museum",     .label = "Museum",      .default_value = true  },
  { .id = "fog",        .label = "Fog",         .default_value = true  },
  { .id = "bubbles",    .label = "Bubbles",     .default_value = true  },
  { .id = "lightRays",  .label = "Light Rays",  .default_value = true  },
  { .id = "lasers",     .label = "Lasers",      .default_value = false },
  // clang-format on
};

static uint32_t fish_count_presets[10]
  = {1, 100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000};

static struct {
  const char* name;
  globals_t globals;
  inner_const_t inner_const;
} view_presets[6] = {
  {
    .name = "Inside (A)",
    .globals = {
      .target_height = 63.3f,
      .target_radius = 91.6f,
      .eye_height    = 7.5f,
      .eye_radius    = 13.2f,
      .eye_speed     = 0.0258f,
      .field_of_view = 82.699f,
      .ambient_red   = 0.218f,
      .ambient_green = 0.502f,
      .ambient_blue  = 0.706f,
      .fog_power     = 16.5f,
      .fog_mult      = 1.5f,
      .fog_offset    = 0.738f,
      .fog_red       = 0.338f,
      .fog_green     = 0.81f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 0.796f,
    },
  },
  {
    .name = "Outside (A)",
    .globals = {
      .target_height = 17.1f,
      .target_radius = 69.2f,
      .eye_height    = 59.1f,
      .eye_radius    = 124.4f,
      .eye_speed     = 0.0258f,
      .field_of_view = 56.923f,
      .ambient_red   = 0.218f,
      .ambient_green = 0.246f,
      .ambient_blue  = 0.394f,
      .fog_power     = 27.1f,
      .fog_mult      = 1.46f,
      .fog_offset    = 0.53f,
      .fog_red       = 0.382f,
      .fog_green     = 0.602f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 1.0f,
    },
  },
  {
    .name = "Inside (Original)",
    .globals = {
      .target_height = 0.0f,
      .target_radius = 88.0f,
      .eye_height    = 38.0f,
      .eye_radius    = 69.0f,
      .eye_speed     = 0.0258f,
      .field_of_view = 64.0f,
      .ambient_red   = 0.218f,
      .ambient_green = 0.246f,
      .ambient_blue  = 0.394f,
      .fog_power     = 16.5f,
      .fog_mult      = 1.5f,
      .fog_offset    = 0.738f,
      .fog_red       = 0.338f,
      .fog_green     = 0.81f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 0.796f,
    },
  },
  {
    .name = "Outside (Original)",
    .globals = {
      .target_height = 72.0f,
      .target_radius = 73.0f,
      .eye_height    = 3.9f,
      .eye_radius    = 120.0f,
      .eye_speed     = 0.0258,
      .field_of_view = 74.0f,
      .ambient_red   = 0.218f,
      .ambient_green = 0.246f,
      .ambient_blue  = 0.394f,
      .fog_power     = 27.1f,
      .fog_mult      = 1.46f,
      .fog_offset    = 0.53f,
      .fog_red       = 0.382f,
      .fog_green     = 0.602f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 1.0f,
    },
  },
  {
    .name = "Center (LG)",
    .globals = {
      .target_height = 24.0f,
      .target_radius = 73.0f,
      .eye_height    = 24.0f,
      .eye_radius    = 0.0f,
      .eye_speed     = 0.06f,
      .field_of_view = 60.0f,
      .ambient_red   = 0.22f,
      .ambient_green = 0.25f,
      .ambient_blue  = 0.39f,
      .fog_power     = 14.5f,
      .fog_mult      = 1.3f,
      .fog_offset    = 0.53f,
      .fog_red       = 0.54f,
      .fog_green     = 0.86f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 0.8f,
    },
  },
  {
    .name = "Outside (LG)",
    .globals = {
      .target_height = 20.0f,
      .target_radius = 127.0f,
      .eye_height    = 39.9f,
      .eye_radius    = 124.0f,
      .eye_speed     = 0.06f,
      .field_of_view = 24.0f,
      .ambient_red   = 0.22f,
      .ambient_green = 0.25f,
      .ambient_blue  = 0.39f,
      .fog_power     = 27.1f,
      .fog_mult      = 1.2f,
      .fog_offset    = 0.53f,
      .fog_red       = 0.382f,
      .fog_green     = 0.602f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 1.0f,
    },
  },
};

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* bubble_shader_wgsl;
static const char* diffuse_shader_wgsl;
static const char* fish_shader_p1_wgsl;
static const char* fish_shader_p2_wgsl;
static const char* inner_shader_p1_wgsl;
static const char* inner_shader_p2_wgsl;
static const char* laser_shader_wgsl;
static const char* light_ray_shader_wgsl;
static const char* outer_shader_wgsl;
static const char* seaweed_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Aquarium example
 * -------------------------------------------------------------------------- */

int main(void)
{
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* bubble_shader_wgsl = CODE(
  // Bubble Particle Shader for WebGPU Aquarium
  // Billboarded particles with lifetime animation

  struct Uniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    time: f32,
    padding: vec3<f32>,
  };

  struct ParticleData {
    positionStartTime: vec4<f32>,      // xyz = position, w = start time
    velocityStartSize: vec4<f32>,      // xyz = velocity, w = start size
    accelerationEndSize: vec4<f32>,    // xyz = acceleration, w = end size
    colorMult: vec4<f32>,              // rgba multiplier
    lifetimeFrameSpinStart: vec4<f32>, // x = lifetime, y = frameStart, z = spinStart, w = spinSpeed
  };

  struct VertexInput {
    @location(0) corner: vec2<f32>,    // Corner position (-0.5 to 0.5)
    @location(1) positionStartTime: vec4<f32>,
    @location(2) velocityStartSize: vec4<f32>,
    @location(3) accelerationEndSize: vec4<f32>,
    @location(4) colorMult: vec4<f32>,
    @location(5) lifetimeFrameSpinStart: vec4<f32>,
  };

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) percentLife: f32,
    @location(2) colorMult: vec4<f32>,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Unpack particle data
    let position = input.positionStartTime.xyz;
    let startTime = input.positionStartTime.w;
    let velocity = input.velocityStartSize.xyz;
    let startSize = input.velocityStartSize.w;
    let acceleration = input.accelerationEndSize.xyz;
    let endSize = input.accelerationEndSize.w;
    let lifetime = input.lifetimeFrameSpinStart.x;
    let spinStart = input.lifetimeFrameSpinStart.z;
    let spinSpeed = input.lifetimeFrameSpinStart.w;

    // Calculate particle age and percent life
    let age = uniforms.time - startTime;
    let percentLife = age / lifetime;

    // Hide particles that are not alive
    var size = mix(startSize, endSize, percentLife);
    if (percentLife < 0.0 || percentLife > 1.0) {
      size = 0.0;
    }

    // Calculate particle position with physics
    let currentPosition = position + velocity * age + acceleration * age * age;

    // Calculate rotation
    let angle = spinStart + spinSpeed * age;
    let s = sin(angle);
    let c = cos(angle);
    let rotatedCorner = vec2<f32>(
      input.corner.x * c + input.corner.y * s,
      -input.corner.x * s + input.corner.y * c
    );

    // Billboard - face the camera
    let basisX = uniforms.viewInverse[0].xyz;
    let basisY = uniforms.viewInverse[1].xyz;
    let offsetPosition = (basisX * rotatedCorner.x + basisY * rotatedCorner.y) * size;

    // Final world position
    let worldPosition = currentPosition + offsetPosition;

    // Output
    output.position = uniforms.viewProjection * vec4<f32>(worldPosition, 1.0);
    output.texCoord = input.corner + vec2<f32>(0.5, 0.5);  // Convert from -0.5..0.5 to 0..1
    output.percentLife = percentLife;
    output.colorMult = input.colorMult;

    return output;
  }

  @group(1) @binding(0) var particleTexture: texture_2d<f32>;
  @group(1) @binding(1) var particleSampler: sampler;

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the particle texture
    let texColor = textureSample(particleTexture, particleSampler, input.texCoord);

    // Fade out at end of life
    let alpha = 1.0 - smoothstep(0.7, 1.0, input.percentLife);

    // Apply color multiplier and lifetime alpha
    var color = texColor * input.colorMult;
    color.a *= alpha;

    return color;
  }
);

static const char* diffuse_shader_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct ModelUniforms {
    world: mat4x4<f32>,
    worldInverse: mat4x4<f32>,
    worldInverseTranspose: mat4x4<f32>,
    extra: vec4<f32>,
  }

  struct MaterialUniforms {
    specular: vec4<f32>,
    shininess: f32,
    specularFactor: f32,
    pad0: vec2<f32>,
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var linearSampler: sampler;
  @group(2) @binding(2) var<uniform> materialUniforms: MaterialUniforms;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) surfaceToLight: vec3<f32>,
    @location(3) surfaceToView: vec3<f32>,
    @location(4) worldPosition: vec3<f32>,
    @location(5) clipPosition: vec4<f32>,
  }

  @vertex
  fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    output.position = frameUniforms.viewProjection * worldPosition;
    output.texCoord = input.texCoord;
    output.normal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.normal, 0.0)).xyz;
    output.surfaceToLight = frameUniforms.lightWorldPos.xyz - worldPosition.xyz;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - worldPosition.xyz;
    output.worldPosition = worldPosition.xyz;
    output.clipPosition = output.position;
    return output;
  }

  fn lit(l: f32, h: f32, shininess: f32) -> vec3<f32> {
    let ambient = 1.0;
    let diffuse = max(l, 0.0);
    let specular = select(0.0, pow(max(h, 0.0), shininess), l > 0.0);
    return vec3<f32>(ambient, diffuse, specular);
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuseColor = textureSample(diffuseTexture, linearSampler, input.texCoord);
    let normal = normalize(input.normal);
    let surfaceToLight = normalize(input.surfaceToLight);
    let surfaceToView = normalize(input.surfaceToView);
    let halfVector = normalize(surfaceToLight + surfaceToView);

    let lighting = lit(dot(normal, surfaceToLight), dot(normal, halfVector), materialUniforms.shininess);
    let lightColor = frameUniforms.lightColor.rgb;
    let ambientColor = frameUniforms.ambient.rgb;

    var color = vec3<f32>(0.0);
    color += lightColor * diffuseColor.rgb * lighting.y;
    color += diffuseColor.rgb * ambientColor;
    color += frameUniforms.lightColor.rgb * materialUniforms.specular.rgb * lighting.z * materialUniforms.specularFactor;

    var outColor = vec4<f32>(color, diffuseColor.a);

    if (frameUniforms.fogParams.w > 0.5) {
      let fogCoord = input.clipPosition.z / input.clipPosition.w;
      let fogFactor = clamp(pow(fogCoord, frameUniforms.fogParams.x) * frameUniforms.fogParams.y - frameUniforms.fogParams.z, 0.0, 1.0);
      let foggedColor = mix(outColor.rgb, frameUniforms.fogColor.rgb, fogFactor);
      outColor = vec4<f32>(foggedColor, outColor.a);
    }

    return outColor;
  }
);

static const char* fish_shader_p1_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct FishInstance {
    worldPosition: vec3<f32>,
    scale: f32,
    nextPosition: vec3<f32>,
    time: f32,
  }

  struct SpeciesUniforms {
    fishLength: f32,
    fishWaveLength: f32,
    fishBendAmount: f32,
    useNormalMap: f32,
    useReflectionMap: f32,
    shininess: f32,
    specularFactor: f32,
    padding: f32,
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<storage, read> fishInstances: array<FishInstance>;
  @group(1) @binding(1) var<uniform> speciesUniforms: SpeciesUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var normalTexture: texture_2d<f32>;
  @group(2) @binding(2) var reflectionTexture: texture_2d<f32>;
  @group(2) @binding(3) var linearSampler: sampler;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) surfaceToLight: vec3<f32>,
    @location(3) surfaceToView: vec3<f32>,
    @location(4) tangent: vec3<f32>,
    @location(5) binormal: vec3<f32>,
    @location(6) clipPosition: vec4<f32>,
  }

  fn safeForward(forward: vec3<f32>) -> vec3<f32> {
    let lenSq = dot(forward, forward);
    if (lenSq < 1e-6) {
      return vec3<f32>(0.0, 0.0, 1.0);
    }
    return forward / sqrt(lenSq);
  }

  fn computeBasis(forward: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    var right = cross(up, forward);
    var rightLenSq = dot(right, right);
    if (rightLenSq < 1e-6) {
      up = vec3<f32>(0.0, 0.0, 1.0);
      right = cross(up, forward);
      rightLenSq = dot(right, right);
      if (rightLenSq < 1e-6) {
        right = vec3<f32>(1.0, 0.0, 0.0);
      }
    }
    right = normalize(right);
    let realUp = normalize(cross(forward, right));
    return mat3x3<f32>(right, realUp, forward);
  }
);

static const char* fish_shader_p2_wgsl = CODE(
  @vertex
  fn vs_main(input: VertexInput, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
    let instance = fishInstances[instanceIndex];

    var forward = safeForward(instance.worldPosition - instance.nextPosition);
    let basis = computeBasis(forward);
    let right = basis[0];
    let trueUp = basis[1];

    let worldMatrix = mat4x4<f32>(
      vec4<f32>(right * instance.scale, 0.0),
      vec4<f32>(trueUp * instance.scale, 0.0),
      vec4<f32>(forward * instance.scale, 0.0),
      vec4<f32>(instance.worldPosition, 1.0)
    );

    var mult = input.position.z / max(speciesUniforms.fishLength, 0.0001);
    if (input.position.z <= 0.0) {
      mult = (-input.position.z / max(speciesUniforms.fishLength, 0.0001)) * 2.0;
    }

    let s = sin(instance.time + mult * speciesUniforms.fishWaveLength);
    let offset = (mult * mult) * s * speciesUniforms.fishBendAmount;
    let bentPosition = vec4<f32>(input.position + vec3<f32>(offset, 0.0, 0.0), 1.0);

    let worldPosition = worldMatrix * bentPosition;
    let normalMatrix = basis;

    var output: VertexOutput;
    output.position = frameUniforms.viewProjection * worldPosition;
    output.clipPosition = output.position;
    output.texCoord = input.texCoord;
    output.normal = normalize(normalMatrix * input.normal);
    output.tangent = normalize(normalMatrix * input.tangent);
    output.binormal = normalize(normalMatrix * input.binormal);
    output.surfaceToLight = frameUniforms.lightWorldPos.xyz - worldPosition.xyz;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - worldPosition.xyz;
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuseSample = textureSample(diffuseTexture, linearSampler, input.texCoord);
    let normalSample = textureSample(normalTexture, linearSampler, input.texCoord);

    var normal = normalize(input.normal);
    var specStrength = 0.0;
    if (speciesUniforms.useNormalMap > 0.5) {
      let tangent = normalize(input.tangent);
      let binormal = normalize(input.binormal);
      let tangentToWorld = mat3x3<f32>(tangent, binormal, normal);
      var tangentNormal = normalSample.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0);
      tangentNormal = normalize(tangentNormal + vec3<f32>(0.0, 0.0, 2.0));
      normal = normalize(tangentToWorld * tangentNormal);
      specStrength = normalSample.a;
    }

    let surfaceToLight = normalize(input.surfaceToLight);
    let surfaceToView = normalize(input.surfaceToView);
    let halfVector = normalize(surfaceToLight + surfaceToView);

    let diffuseFactor = max(dot(normal, surfaceToLight), 0.0);
    let specularTerm = select(0.0, pow(max(dot(normal, halfVector), 0.0), speciesUniforms.shininess), diffuseFactor > 0.0);

    let lightColor = frameUniforms.lightColor.rgb;
    let ambientColor = frameUniforms.ambient.rgb;

    var color = diffuseSample.rgb * ambientColor;
    color += diffuseSample.rgb * lightColor * diffuseFactor;
    color += lightColor * specularTerm * speciesUniforms.specularFactor * specStrength;

    if (speciesUniforms.useReflectionMap > 0.5) {
      let reflectionSample = textureSample(reflectionTexture, linearSampler, input.texCoord);
      let mixFactor = clamp(1.0 - reflectionSample.r, 0.0, 1.0);
      color = mix(reflectionSample.rgb, color, mixFactor);
    }

    if (frameUniforms.fogParams.w > 0.5) {
      let fogCoord = input.clipPosition.z / input.clipPosition.w;
      let fogFactor = clamp(pow(fogCoord, frameUniforms.fogParams.x) * frameUniforms.fogParams.y - frameUniforms.fogParams.z, 0.0, 1.0);
      color = mix(color, frameUniforms.fogColor.rgb, fogFactor);
    }

    return vec4<f32>(color, diffuseSample.a);
  }
);

static const char* inner_shader_p1_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct ModelUniforms {
    world: mat4x4<f32>,
    worldInverse: mat4x4<f32>,
    worldInverseTranspose: mat4x4<f32>,
    extra: vec4<f32>,
  }

  struct TankMaterialUniforms {
    specular: vec4<f32>,
    params0: vec4<f32>, // x: shininess, y: specularFactor, z: refractionFudge, w: eta
    params1: vec4<f32>, // x: tankColorFudge, y: useNormalMap, z: useReflectionMap, w: outerFudge (unused)
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var normalTexture: texture_2d<f32>;
  @group(2) @binding(2) var reflectionTexture: texture_2d<f32>;
  @group(2) @binding(3) var skyboxTexture: texture_cube<f32>;
  @group(2) @binding(4) var linearSampler: sampler;
  @group(2) @binding(5) var<uniform> tankUniforms: TankMaterialUniforms;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) binormal: vec3<f32>,
    @location(4) surfaceToLight: vec3<f32>,
    @location(5) surfaceToView: vec3<f32>,
    @location(6) clipPosition: vec4<f32>,
  }
);

static const char* inner_shader_p2_wgsl = CODE(
  @vertex
  fn vs_main(input: VertexInput) -> VertexOutput {
    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    var output: VertexOutput;
    output.position = frameUniforms.viewProjection * worldPosition;
    output.clipPosition = output.position;
    output.texCoord = input.texCoord;
    output.normal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.normal, 0.0)).xyz;
    output.tangent = (modelUniforms.worldInverseTranspose * vec4<f32>(input.tangent, 0.0)).xyz;
    output.binormal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.binormal, 0.0)).xyz;
    output.surfaceToLight = frameUniforms.lightWorldPos.xyz - worldPosition.xyz;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - worldPosition.xyz;
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var diffuseColor = textureSample(diffuseTexture, linearSampler, input.texCoord);
    let tankColorFudge = tankUniforms.params1.x;
    diffuseColor = vec4<f32>(diffuseColor.rgb + vec3<f32>(tankColorFudge, tankColorFudge, tankColorFudge), 1.0);

    var normal = normalize(input.normal);
    let useNormalMap = tankUniforms.params1.y;
    if (useNormalMap > 0.5) {
      let tangent = normalize(input.tangent);
      let binormal = normalize(input.binormal);
      let tangentToWorld = mat3x3<f32>(tangent, binormal, normal);
      let normalSample = textureSample(normalTexture, linearSampler, input.texCoord);
      var tangentNormal = normalSample.xyz - vec3<f32>(0.5, 0.5, 0.5);
      tangentNormal = normalize(tangentNormal + vec3<f32>(0.0, 0.0, tankUniforms.params0.z));
      normal = normalize(tangentToWorld * tangentNormal);
    }

    let surfaceToView = normalize(input.surfaceToView);
    let eta = max(tankUniforms.params0.w, 0.0001);
    var refractionDir = refract(surfaceToView, normal, eta);
    if (dot(refractionDir, refractionDir) < 1e-6) {
      refractionDir = -surfaceToView;
    }
    refractionDir = normalize(refractionDir);

    let skySample = textureSample(skyboxTexture, linearSampler, refractionDir);

    var refractionMask = 1.0;
    let useReflectionMap = tankUniforms.params1.z;
    if (useReflectionMap > 0.5) {
      refractionMask = textureSample(reflectionTexture, linearSampler, input.texCoord).r;
    }
    refractionMask = clamp(refractionMask, 0.0, 1.0);

    let skyContribution = skySample.rgb * diffuseColor.rgb;
    let mixedColor = mix(skyContribution, diffuseColor.rgb, refractionMask);
    var outColor = vec4<f32>(mixedColor, diffuseColor.a);

    if (frameUniforms.fogParams.w > 0.5) {
      let fogCoord = input.clipPosition.z / input.clipPosition.w;
      let fogFactor = clamp(pow(fogCoord, frameUniforms.fogParams.x) * frameUniforms.fogParams.y - frameUniforms.fogParams.z, 0.0, 1.0);
      let foggedColor = mix(outColor.rgb, frameUniforms.fogColor.rgb, fogFactor);
      outColor = vec4<f32>(foggedColor, outColor.a);
    }

    return outColor;
  }
);

static const char* laser_shader_wgsl = CODE(
  // Laser Beam Shader for WebGPU Aquarium
  // Simple textured beam with color modulation

  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
  };

  struct ModelUniforms {
    world: mat4x4<f32>,
  };

  struct MaterialUniforms {
    colorMult: vec4<f32>,
  };

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) texCoord: vec2<f32>,
  };

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
  };

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var beamTexture: texture_2d<f32>;
  @group(2) @binding(1) var beamSampler: sampler;
  @group(2) @binding(2) var<uniform> materialUniforms: MaterialUniforms;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    output.position = frameUniforms.viewProjection * worldPosition;
    output.texCoord = input.texCoord;

    return output;
  }

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    let texColor = textureSample(beamTexture, beamSampler, input.texCoord);
    return texColor * materialUniforms.colorMult;
  }
);

static const char* light_ray_shader_wgsl = CODE(
  // Light Ray (God Ray) Shader for WebGPU Aquarium
  // Animated volumetric light shafts from above

  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
  };

  struct ModelUniforms {
    world: mat4x4<f32>,
  };

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) texCoord: vec2<f32>,
  };

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
  };

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var lightRayTexture: texture_2d<f32>;
  @group(2) @binding(1) var lightRaySampler: sampler;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    output.position = frameUniforms.viewProjection * worldPosition;
    output.texCoord = input.texCoord;

    return output;
  }

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(lightRayTexture, lightRaySampler, input.texCoord);
  }
);

static const char* outer_shader_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct ModelUniforms {
    world: mat4x4<f32>,
    worldInverse: mat4x4<f32>,
    worldInverseTranspose: mat4x4<f32>,
    extra: vec4<f32>,
  }

  struct TankMaterialUniforms {
    specular: vec4<f32>,
    params0: vec4<f32>, // x: shininess, y: specularFactor, z: refractionFudge (unused), w: eta (unused)
    params1: vec4<f32>, // x: tankColorFudge (unused), y: useNormalMap, z: useReflectionMap, w: outerFudge
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var normalTexture: texture_2d<f32>;
  @group(2) @binding(2) var reflectionTexture: texture_2d<f32>;
  @group(2) @binding(3) var skyboxTexture: texture_cube<f32>;
  @group(2) @binding(4) var linearSampler: sampler;
  @group(2) @binding(5) var<uniform> tankUniforms: TankMaterialUniforms;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) binormal: vec3<f32>,
    @location(4) surfaceToView: vec3<f32>,
  }

  @vertex
  fn vs_main(input: VertexInput) -> VertexOutput {
    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    var output: VertexOutput;
    output.position = frameUniforms.viewProjection * worldPosition;
    output.texCoord = input.texCoord;
    output.normal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.normal, 0.0)).xyz;
    output.tangent = (modelUniforms.worldInverseTranspose * vec4<f32>(input.tangent, 0.0)).xyz;
    output.binormal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.binormal, 0.0)).xyz;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - worldPosition.xyz;
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuseColor = textureSample(diffuseTexture, linearSampler, input.texCoord);

    var normal = normalize(input.normal);
    if (tankUniforms.params1.y > 0.5) {
      let tangent = normalize(input.tangent);
      let binormal = normalize(input.binormal);
      let tangentToWorld = mat3x3<f32>(tangent, binormal, normal);
      let normalSample = textureSample(normalTexture, linearSampler, input.texCoord);
      var tangentNormal = normalSample.xyz - vec3<f32>(0.5, 0.5, 0.5);
      normal = normalize(tangentToWorld * tangentNormal);
    }

    let surfaceToView = normalize(input.surfaceToView);
    let reflectionDir = normalize(-reflect(surfaceToView, normal));
    var skyColor = textureSample(skyboxTexture, linearSampler, reflectionDir);

    let fudgeAmount = tankUniforms.params1.w;
    let fudge = skyColor.rgb * fudgeAmount;
    let bright = min(1.0, fudge.r * fudge.g * fudge.b);

    var reflectionAmount = 0.0;
    if (tankUniforms.params1.z > 0.5) {
      reflectionAmount = textureSample(reflectionTexture, linearSampler, input.texCoord).r;
    }
    reflectionAmount = clamp(reflectionAmount, 0.0, 1.0);

    let reflectColor = mix(vec4<f32>(skyColor.rgb, bright), diffuseColor, 1.0 - reflectionAmount);
    let viewDot = clamp(abs(dot(surfaceToView, normal)), 0.0, 1.0);
    var reflectMix = clamp((viewDot + 0.3) * reflectionAmount, 0.0, 1.0);
    if (tankUniforms.params1.z <= 0.5) {
      reflectMix = 1.0;
    }

    let finalColor = mix(skyColor.rgb, reflectColor.rgb, reflectMix);
    let alpha = clamp(1.0 - viewDot, 0.0, 1.0);
    return vec4<f32>(finalColor, alpha);
  }
);

static const char* seaweed_shader_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct ModelUniforms {
    world: mat4x4<f32>,
    worldInverse: mat4x4<f32>,
    worldInverseTranspose: mat4x4<f32>,
    extra: vec4<f32>,
  }

  struct MaterialUniforms {
    specular: vec4<f32>,
    shininess: f32,
    specularFactor: f32,
    pad0: vec2<f32>,
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var linearSampler: sampler;
  @group(2) @binding(2) var<uniform> materialUniforms: MaterialUniforms;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) surfaceToLight: vec3<f32>,
    @location(3) surfaceToView: vec3<f32>,
    @location(4) clipPosition: vec4<f32>,
  }

  fn safeNormalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    return select(fallback, v / len, len > 1e-5);
  }

  @vertex
  fn vs_main(input: VertexInput) -> VertexOutput {
    let worldPos = modelUniforms.world;
    let time = modelUniforms.extra.x;
    let toCamera = safeNormalize(frameUniforms.viewInverse[3].xyz - worldPos[3].xyz, vec3<f32>(0.0, 0.0, 1.0));
    let yAxis = vec3<f32>(0.0, 1.0, 0.0);
    let xAxis = safeNormalize(cross(yAxis, toCamera), vec3<f32>(1.0, 0.0, 0.0));
    let zAxis = safeNormalize(cross(xAxis, yAxis), vec3<f32>(0.0, 0.0, 1.0));

    let newWorld = mat4x4<f32>(
      vec4<f32>(xAxis, 0.0),
      vec4<f32>(yAxis, 0.0),
      vec4<f32>(zAxis, 0.0),
      vec4<f32>(worldPos[3].xyz, 1.0)
    );

    var bentPosition = vec4<f32>(input.position, 1.0);
    let sway = sin(time * 0.5) * pow(input.position.y * 0.07, 2.0);
    bentPosition.x += sway;
    bentPosition.y += -4.0;

    let worldPosition = newWorld * bentPosition;

    var output: VertexOutput;
    output.position = frameUniforms.viewProjection * worldPosition;
    output.clipPosition = output.position;
    output.texCoord = input.texCoord;
    let normalMatrix = mat3x3<f32>(newWorld[0].xyz, newWorld[1].xyz, newWorld[2].xyz);
    output.normal = normalize(normalMatrix * input.normal);
    let baseWorldPosition = (modelUniforms.world * vec4<f32>(input.position, 1.0)).xyz;
    output.surfaceToLight = frameUniforms.lightWorldPos.xyz - baseWorldPosition;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - baseWorldPosition;
    return output;
  }

  fn lit(l: f32, h: f32, shininess: f32) -> vec3<f32> {
    let diffuse = max(l, 0.0);
    let specular = select(0.0, pow(max(h, 0.0), shininess), l > 0.0);
    return vec3<f32>(1.0, diffuse, specular);
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuseSample = textureSample(diffuseTexture, linearSampler, input.texCoord);
    if (diffuseSample.a < 0.3) {
      discard;
    }

    let normal = normalize(input.normal);
    let surfaceToLight = normalize(input.surfaceToLight);
    let surfaceToView = normalize(input.surfaceToView);
    let halfVector = normalize(surfaceToLight + surfaceToView);
    let lighting = lit(dot(normal, surfaceToLight), dot(normal, halfVector), materialUniforms.shininess);

    let lightColor = frameUniforms.lightColor.rgb;
    let ambientColor = frameUniforms.ambient.rgb;

    var color = diffuseSample.rgb * ambientColor;
    color += diffuseSample.rgb * lightColor * lighting.y;
    color += lightColor * materialUniforms.specular.rgb * lighting.z * materialUniforms.specularFactor;

    if (frameUniforms.fogParams.w > 0.5) {
      let fogCoord = input.clipPosition.z / input.clipPosition.w;
      let fogFactor = clamp(pow(fogCoord, frameUniforms.fogParams.x) * frameUniforms.fogParams.y - frameUniforms.fogParams.z, 0.0, 1.0);
      color = mix(color, frameUniforms.fogColor.rgb, fogFactor);
    }

    return vec4<f32>(color, diffuseSample.a);
  }
);

// clang-format on
