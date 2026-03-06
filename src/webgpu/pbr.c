#include "pbr.h"

#include "core/image_loader.h"

#include <stdio.h>
#include <string.h>

/* -- Default IBL parameters ----------------------------------------------- */

#define IBL_DEFAULT_ENVIRONMENT_SIZE 1024
#define IBL_DEFAULT_IRRADIANCE_SIZE 64
#define IBL_DEFAULT_PREFILTERED_SIZE 512
#define IBL_DEFAULT_BRDF_LUT_SIZE 128
#define IBL_DEFAULT_NUM_SAMPLES 1024
#define IBL_MAX_PANORAMA_WIDTH 4096
#define IBL_MAX_PER_MIP_BUFFERS 16

/* -- Environment prefilter WGSL shader (part 1: bindings + utility fns) --- */

// clang-format off

static const char* environment_prefilter_shader_wgsl_part1 = CODE(
  const PI: f32 = 3.14159265359;

  @group(0) @binding(0) var environmentSampler: sampler;
  @group(0) @binding(1) var environmentTexture: texture_cube<f32>;
  @group(0) @binding(2) var<uniform> numSamples: u32;
  @group(0) @binding(3) var irradianceCube: texture_storage_2d_array<rgba16float, write>;
  @group(0) @binding(4) var brdfLut2D: texture_storage_2d<rgba16float, write>;

  @group(1) @binding(0) var<uniform> faceIndex: u32;

  @group(2) @binding(0) var<uniform> roughness: f32;
  @group(2) @binding(1) var prefilteredSpecularCube: texture_storage_2d_array<rgba16float, write>;

  fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
  }

  fn radicalInverseVdC(bitsIn: u32) -> f32 {
    var bits = bitsIn;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
  }

  fn hammersley2D(i: u32, N: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(N), radicalInverseVdC(i));
  }

  fn generateTBN(normal: vec3<f32>) -> mat3x3<f32> {
    var bitangent = vec3<f32>(0.0, 1.0, 0.0);
    let NdotUp = dot(normal, bitangent);
    let epsilon = 1e-7;
    if (1.0 - abs(NdotUp) <= epsilon) {
      if (NdotUp > 0.0) {
        bitangent = vec3<f32>(0.0, 0.0, 1.0);
      } else {
        bitangent = vec3<f32>(0.0, 0.0, -1.0);
      }
    }
    let tangent = normalize(cross(bitangent, normal));
    bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
  }

  fn uvToDirection(uv: vec2<f32>, face: u32) -> vec3<f32> {
    const faceDirs = array<vec3<f32>, 6>(
      vec3<f32>( 1.0,  0.0,  0.0),
      vec3<f32>(-1.0,  0.0,  0.0),
      vec3<f32>( 0.0,  1.0,  0.0),
      vec3<f32>( 0.0, -1.0,  0.0),
      vec3<f32>( 0.0,  0.0,  1.0),
      vec3<f32>( 0.0,  0.0, -1.0)
    );
    const upVectors = array<vec3<f32>, 6>(
      vec3<f32>( 0.0, -1.0,  0.0),
      vec3<f32>( 0.0, -1.0,  0.0),
      vec3<f32>( 0.0,  0.0,  1.0),
      vec3<f32>( 0.0,  0.0, -1.0),
      vec3<f32>( 0.0, -1.0,  0.0),
      vec3<f32>( 0.0, -1.0,  0.0)
    );
    const rightVectors = array<vec3<f32>, 6>(
      vec3<f32>( 0.0,  0.0, -1.0),
      vec3<f32>( 0.0,  0.0,  1.0),
      vec3<f32>( 1.0,  0.0,  0.0),
      vec3<f32>( 1.0,  0.0,  0.0),
      vec3<f32>( 1.0,  0.0,  0.0),
      vec3<f32>(-1.0,  0.0,  0.0)
    );
    let u = (uv.x * 2.0) - 1.0;
    let v = (uv.y * 2.0) - 1.0;
    return normalize(faceDirs[face] + (u * rightVectors[face]) + (v * upVectors[face]));
  }
);

/* -- Environment prefilter WGSL shader (part 2: sampling + irradiance) ---- */

static const char* environment_prefilter_shader_wgsl_part2 = CODE(
  fn importanceSampleLambertian(sampleIndex: u32, sampleCount: u32, normal: vec3<f32>) -> vec4<f32> {
    let xi = hammersley2D(sampleIndex, sampleCount);
    let phi = 2.0 * PI * xi.x;
    let cosTheta = sqrt(1.0 - xi.y);
    let sinTheta = sqrt(xi.y);
    let localDir = vec3<f32>(
      sinTheta * cos(phi),
      sinTheta * sin(phi),
      cosTheta
    );
    let worldDir = normalize(generateTBN(normal) * localDir);
    let pdf = cosTheta / PI;
    return vec4<f32>(worldDir, pdf);
  }

  fn dGGX(NdotH: f32, alpha: f32) -> f32 {
    let a = NdotH * alpha;
    let k = alpha / (1.0 - NdotH*NdotH + a*a);
    return (k * k) / PI;
  }

  fn vSmithGGXCorrelated(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    let GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
  }

  fn importanceSampleGGX(sampleIndex: u32, sampleCount: u32, normal: vec3<f32>, roughness: f32) -> vec4<f32> {
    let xi = hammersley2D(sampleIndex, sampleCount);
    let alpha = roughness * roughness;
    let cosTheta = saturate(sqrt((1.0 - xi.y) / (1.0 + ((alpha * alpha) - 1.0) * xi.y)));
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let phi = 2.0 * PI * xi.x;
    let localHalf = vec3<f32>(
      sinTheta * cos(phi),
      sinTheta * sin(phi),
      cosTheta
    );
    let tbn = generateTBN(normal);
    let halfVec = normalize(tbn * localHalf);
    let pdf = dGGX(cosTheta, alpha) / 4.0;
    return vec4<f32>(halfVec, pdf);
  }

  fn computeLOD(pdf: f32, faceSize: f32) -> f32 {
    return 0.5 * log2((6.0 * faceSize * faceSize) / (f32(numSamples) * pdf));
  }

  @compute @workgroup_size(8, 8)
  fn computeIrradiance(@builtin(global_invocation_id) id: vec3<u32>) {
    let outputSize = textureDimensions(irradianceCube).xy;
    if (id.x >= outputSize.x || id.y >= outputSize.y) { return; }

    let uv = vec2<f32>(f32(id.x) / f32(outputSize.x), f32(id.y) / f32(outputSize.y));
    let normal = normalize(uvToDirection(uv, faceIndex));

    var irradiance = vec3<f32>(0.0);
    var weightSum = 0.0;

    for (var i = 0u; i < numSamples; i++) {
      let sample = importanceSampleLambertian(i, numSamples, normal);
      let sampleDir = sample.xyz;
      let pdf = sample.w;
      let lod: f32 = computeLOD(pdf, f32(outputSize.x));
      let sampleColor = textureSampleLevel(environmentTexture, environmentSampler, sampleDir, lod).rgb;
      let weight = max(dot(normal, sampleDir), 0.0);
      irradiance += sampleColor * weight;
      weightSum += weight;
    }

    if (weightSum > 0.0) {
      irradiance /= weightSum;
    }

    textureStore(irradianceCube, id.xy, faceIndex, vec4<f32>(irradiance, 1.0));
  }
);

/* -- Environment prefilter WGSL shader (part 3: specular + BRDF LUT) ------ */

static const char* environment_prefilter_shader_wgsl_part3 = CODE(
  @compute @workgroup_size(8, 8)
  fn computePrefilteredSpecular(@builtin(global_invocation_id) id: vec3<u32>) {
    let outputSize = textureDimensions(prefilteredSpecularCube).xy;
    if (id.x >= outputSize.x || id.y >= outputSize.y) { return; }

    let uv = vec2<f32>(f32(id.x) / f32(outputSize.x), f32(id.y) / f32(outputSize.y));
    let N = normalize(uvToDirection(uv, faceIndex));

    var accumSpecular = vec3<f32>(0.0);
    var weightSum = 0.0;

    for (var i = 0u; i < numSamples; i++) {
      let sample = importanceSampleGGX(i, numSamples, N, roughness);
      let H = sample.xyz;
      let pdf = sample.w;
      let V = N;
      let L = normalize(reflect(-V, H));
      let NdotL = dot(N, L);
      if (NdotL > 0.0) {
        let lod: f32 = computeLOD(pdf, f32(outputSize.x));
        let sampleColor = textureSampleLevel(environmentTexture, environmentSampler, L, lod).rgb;
        accumSpecular += sampleColor * NdotL;
        weightSum += NdotL;
      }
    }

    if (weightSum > 0.0) {
      accumSpecular /= weightSum;
    }

    textureStore(prefilteredSpecularCube, id.xy, faceIndex, vec4<f32>(accumSpecular, 1.0));
  }

  @compute @workgroup_size(8, 8)
  fn computeLUT(@builtin(global_invocation_id) id: vec3<u32>) {
    let resolution = textureDimensions(brdfLut2D).xy;
    if (id.x >= resolution.x || id.y >= resolution.y) { return; }

    let eps = 1e-5;
    let minRough = 0.001;
    let NdotV = clamp((f32(id.x) + 0.5) / f32(resolution.x), eps, 1.0 - eps);
    let roughness = clamp((f32(id.y) + 0.5) / f32(resolution.y), minRough, 1.0 - eps);

    let V = vec3<f32>(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
    let N = vec3<f32>(0.0, 0.0, 1.0);

    var A = 0.0;
    var B = 0.0;

    for (var i = 0u; i < numSamples; i++) {
      let sample = importanceSampleGGX(i, numSamples, N, roughness);
      let H = sample.xyz;
      let L = normalize(2.0 * dot(V, H) * H - V);

      let NdotL = saturate(L.z);
      let NdotH = saturate(H.z);
      let VdotH = saturate(dot(V, H));

      if (NdotL > 0.0) {
        let G = vSmithGGXCorrelated(NdotV, NdotL, roughness);
        let Gv = (G * VdotH * NdotL) / NdotH;
        let Fc = pow(1.0 - VdotH, 5.0);
        A += (1.0 - Fc) * Gv;
        B += Fc * Gv;
      }
    }

    let scale = 4.0;
    A = (A * scale) / f32(numSamples);
    B = (B * scale) / f32(numSamples);

    textureStore(brdfLut2D, id.xy, vec4<f32>(A, B, 0.0, 1.0));
  }
);

// clang-format on

/* -- Internal: panorama downsampling -------------------------------------- */

/**
 * @brief Bilinear downsampling of an HDR panorama to max 4096×2048.
 */
static float* ibl_downsample_panorama(const float* src, int src_w, int src_h,
                                      uint32_t* out_w, uint32_t* out_h)
{
  const uint32_t new_w = IBL_MAX_PANORAMA_WIDTH;
  const uint32_t new_h = new_w / 2;
  float* dst           = (float*)malloc(new_w * new_h * 4 * sizeof(float));
  if (!dst) {
    return NULL;
  }

  const float scale_x = (float)(src_w - 1) / (float)(new_w - 1);
  const float scale_y = (float)(src_h - 1) / (float)(new_h - 1);

  for (uint32_t j = 0; j < new_h; ++j) {
    const float orig_y = j * scale_y;
    const int y0       = (int)floorf(orig_y);
    const int y1       = (y0 + 1 < src_h) ? y0 + 1 : src_h - 1;
    const float dy     = orig_y - y0;

    for (uint32_t i = 0; i < new_w; ++i) {
      const float orig_x = i * scale_x;
      const int x0       = (int)floorf(orig_x);
      const int x1       = (x0 + 1 < src_w) ? x0 + 1 : src_w - 1;
      const float dx     = orig_x - x0;

      for (int c = 0; c < 4; ++c) {
        const float c00              = src[(y0 * src_w + x0) * 4 + c];
        const float c10              = src[(y0 * src_w + x1) * 4 + c];
        const float c01              = src[(y1 * src_w + x0) * 4 + c];
        const float c11              = src[(y1 * src_w + x1) * 4 + c];
        const float top              = c00 + dx * (c10 - c00);
        const float bottom           = c01 + dx * (c11 - c01);
        dst[(j * new_w + i) * 4 + c] = top + dy * (bottom - top);
      }
    }
  }

  *out_w = new_w;
  *out_h = new_h;
  return dst;
}

/* -- Internal: helper to create IBL cubemap/2D textures ------------------- */

static WGPUTexture ibl_create_cubemap(WGPUDevice device, uint32_t size,
                                      uint32_t mip_levels,
                                      WGPUTextureUsage usage)
{
  WGPUTextureDescriptor desc = {
    .usage         = usage,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {size, size, 6},
    .format        = WGPUTextureFormat_RGBA16Float,
    .mipLevelCount = mip_levels,
    .sampleCount   = 1,
  };
  return wgpuDeviceCreateTexture(device, &desc);
}

static WGPUTexture ibl_create_texture_2d(WGPUDevice device, uint32_t size,
                                         WGPUTextureUsage usage)
{
  WGPUTextureDescriptor desc = {
    .usage         = usage,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {size, size, 1},
    .format        = WGPUTextureFormat_RGBA16Float,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  return wgpuDeviceCreateTexture(device, &desc);
}

static WGPUTextureView ibl_create_cube_view(WGPUTexture tex,
                                            uint32_t mip_levels)
{
  WGPUTextureViewDescriptor vd = {
    .format          = WGPUTextureFormat_RGBA16Float,
    .dimension       = WGPUTextureViewDimension_Cube,
    .baseMipLevel    = 0,
    .mipLevelCount   = mip_levels,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 6,
    .aspect          = WGPUTextureAspect_All,
  };
  return wgpuTextureCreateView(tex, &vd);
}

/* -- Internal: IBL precomputation (irradiance, specular, BRDF LUT) -------- */

static bool ibl_generate_maps(wgpu_context_t* ctx, WGPUTexture env_cubemap,
                              WGPUTexture irradiance_cubemap,
                              WGPUTexture prefiltered_cubemap,
                              WGPUTexture brdf_lut, uint32_t num_samples)
{
  WGPUDevice device = ctx->device;
  WGPUQueue queue   = ctx->queue;

  /* Concatenate the three shader parts */
  const size_t len1  = strlen(environment_prefilter_shader_wgsl_part1);
  const size_t len2  = strlen(environment_prefilter_shader_wgsl_part2);
  const size_t len3  = strlen(environment_prefilter_shader_wgsl_part3);
  const size_t total = len1 + 1 + len2 + 1 + len3 + 1;
  char* full_shader  = (char*)malloc(total);
  if (!full_shader) {
    return false;
  }
  size_t offset = 0;
  memcpy(full_shader + offset, environment_prefilter_shader_wgsl_part1, len1);
  offset += len1;
  full_shader[offset++] = '\n';
  memcpy(full_shader + offset, environment_prefilter_shader_wgsl_part2, len2);
  offset += len2;
  full_shader[offset++] = '\n';
  memcpy(full_shader + offset, environment_prefilter_shader_wgsl_part3, len3);
  offset += len3;
  full_shader[offset] = '\0';

  WGPUShaderModule shader = wgpu_create_shader_module(device, full_shader);
  free(full_shader);

  if (!shader) {
    fprintf(stderr, "ibl: failed to create prefilter shader module\n");
    return false;
  }

  /* Sampler (trilinear, repeat) */
  WGPUSampler env_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_Repeat,
              .addressModeW  = WGPUAddressMode_Repeat,
              .minFilter     = WGPUFilterMode_Linear,
              .magFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .maxAnisotropy = 1,
            });

  /* numSamples uniform buffer */
  WGPUBuffer num_samples_buf = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(uint32_t),
            });
  wgpuQueueWriteBuffer(queue, num_samples_buf, 0, &num_samples,
                       sizeof(uint32_t));

  /* Per-face uniform buffers + bind groups */
  WGPUBuffer face_bufs[6];
  for (uint32_t face = 0; face < 6; ++face) {
    face_bufs[face] = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                .size  = sizeof(uint32_t),
              });
    wgpuQueueWriteBuffer(queue, face_bufs[face], 0, &face, sizeof(uint32_t));
  }

  /* --- Bind group layouts --- */

  /* BG0: sampler + env cube + numSamples + irradiance storage + brdf storage */
  WGPUBindGroupLayoutEntry bgl0_entries[5] = {
    {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
    {
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .texture    = {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_Cube,
      },
    },
    {
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uint32_t),
      },
    },
    {
      .binding        = 3,
      .visibility     = WGPUShaderStage_Compute,
      .storageTexture = {
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA16Float,
        .viewDimension = WGPUTextureViewDimension_2DArray,
      },
    },
    {
      .binding        = 4,
      .visibility     = WGPUShaderStage_Compute,
      .storageTexture = {
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA16Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
  };
  WGPUBindGroupLayout bgl0
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .entryCount = 5,
                                                .entries    = bgl0_entries,
                                              });

  /* BG1: face index */
  WGPUBindGroupLayoutEntry bgl1_entries[1] = {
    {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uint32_t),
      },
    },
  };
  WGPUBindGroupLayout bgl1
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .entryCount = 1,
                                                .entries    = bgl1_entries,
                                              });

  /* BG2: roughness + prefiltered specular storage */
  WGPUBindGroupLayoutEntry bgl2_entries[2] = {
    {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(float),
      },
    },
    {
      .binding        = 1,
      .visibility     = WGPUShaderStage_Compute,
      .storageTexture = {
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA16Float,
        .viewDimension = WGPUTextureViewDimension_2DArray,
      },
    },
  };
  WGPUBindGroupLayout bgl2
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .entryCount = 2,
                                                .entries    = bgl2_entries,
                                              });

  /* Pipeline layout (3 bind groups) */
  WGPUBindGroupLayout bgls[] = {bgl0, bgl1, bgl2};
  WGPUPipelineLayout pipe_layout
    = wgpuDeviceCreatePipelineLayout(device, &(WGPUPipelineLayoutDescriptor){
                                               .bindGroupLayoutCount = 3,
                                               .bindGroupLayouts     = bgls,
                                             });

  /* 3 compute pipelines sharing the same shader module */
  WGPUComputePipelineDescriptor pipe_desc = {
    .layout  = pipe_layout,
    .compute = {.module = shader},
  };

  pipe_desc.compute.entryPoint = STRVIEW("computeIrradiance");
  WGPUComputePipeline pipeline_irradiance
    = wgpuDeviceCreateComputePipeline(device, &pipe_desc);

  pipe_desc.compute.entryPoint = STRVIEW("computePrefilteredSpecular");
  WGPUComputePipeline pipeline_prefilter
    = wgpuDeviceCreateComputePipeline(device, &pipe_desc);

  pipe_desc.compute.entryPoint = STRVIEW("computeLUT");
  WGPUComputePipeline pipeline_brdf_lut
    = wgpuDeviceCreateComputePipeline(device, &pipe_desc);

  /* Per-face bind groups */
  WGPUBindGroup face_bgs[6];
  for (uint32_t face = 0; face < 6; ++face) {
    WGPUBindGroupEntry e[1] = {
      {.binding = 0, .buffer = face_bufs[face], .size = sizeof(uint32_t)},
    };
    face_bgs[face]
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .layout     = bgl1,
                                            .entryCount = 1,
                                            .entries    = e,
                                          });
  }

  /* Per-mip uniform buffers + bind groups for specular prefiltering */
  const uint32_t pf_mip_count
    = wgpuTextureGetMipLevelCount(prefiltered_cubemap);
  ASSERT(pf_mip_count <= IBL_MAX_PER_MIP_BUFFERS);

  WGPUBuffer mip_bufs[IBL_MAX_PER_MIP_BUFFERS];
  WGPUBindGroup mip_bgs[IBL_MAX_PER_MIP_BUFFERS];

  for (uint32_t m = 0; m < pf_mip_count; ++m) {
    /* Roughness uniform */
    mip_bufs[m] = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                .size  = sizeof(float),
              });
    float roughness_val
      = (pf_mip_count > 1) ? (float)m / (float)(pf_mip_count - 1) : 0.0f;
    wgpuQueueWriteBuffer(queue, mip_bufs[m], 0, &roughness_val, sizeof(float));

    /* Per-mip texture view (single mip level, all 6 layers) */
    WGPUTextureView pf_view = wgpuTextureCreateView(
      prefiltered_cubemap, &(WGPUTextureViewDescriptor){
                             .format         = WGPUTextureFormat_RGBA16Float,
                             .dimension      = WGPUTextureViewDimension_2DArray,
                             .baseMipLevel   = m,
                             .mipLevelCount  = 1,
                             .baseArrayLayer = 0,
                             .arrayLayerCount = 6,
                             .aspect          = WGPUTextureAspect_All,
                           });

    WGPUBindGroupEntry e[2] = {
      {.binding = 0, .buffer = mip_bufs[m], .size = sizeof(float)},
      {.binding = 1, .textureView = pf_view},
    };
    mip_bgs[m] = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                                     .layout     = bgl2,
                                                     .entryCount = 2,
                                                     .entries    = e,
                                                   });

    WGPU_RELEASE_RESOURCE(TextureView, pf_view);
  }

  /* BG0: common bind group */
  WGPUTextureView env_view = ibl_create_cube_view(
    env_cubemap, wgpuTextureGetMipLevelCount(env_cubemap));
  WGPUTextureView irrad_view = wgpuTextureCreateView(
    irradiance_cubemap, &(WGPUTextureViewDescriptor){
                          .format          = WGPUTextureFormat_RGBA16Float,
                          .dimension       = WGPUTextureViewDimension_2DArray,
                          .baseMipLevel    = 0,
                          .mipLevelCount   = 1,
                          .baseArrayLayer  = 0,
                          .arrayLayerCount = 6,
                          .aspect          = WGPUTextureAspect_All,
                        });
  WGPUTextureView brdf_view = wgpuTextureCreateView(
    brdf_lut, &(WGPUTextureViewDescriptor){
                .format          = WGPUTextureFormat_RGBA16Float,
                .dimension       = WGPUTextureViewDimension_2D,
                .baseMipLevel    = 0,
                .mipLevelCount   = 1,
                .baseArrayLayer  = 0,
                .arrayLayerCount = 1,
                .aspect          = WGPUTextureAspect_All,
              });

  WGPUBindGroupEntry bg0_entries[5] = {
    {.binding = 0, .sampler = env_sampler},
    {.binding = 1, .textureView = env_view},
    {.binding = 2, .buffer = num_samples_buf, .size = sizeof(uint32_t)},
    {.binding = 3, .textureView = irrad_view},
    {.binding = 4, .textureView = brdf_view},
  };
  WGPUBindGroup bg0
    = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                          .layout     = bgl0,
                                          .entryCount = 5,
                                          .entries    = bg0_entries,
                                        });

  /* --- Dispatch all 3 passes in a single command buffer --- */
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    device, &(WGPUCommandEncoderDescriptor){
              .label = STRVIEW("IBL precomputation encoder"),
            });
  WGPUComputePassEncoder pass
    = wgpuCommandEncoderBeginComputePass(encoder, NULL);

  const uint32_t wg = 8;

  /* --- Pass 1: irradiance cubemap --- */
  {
    const uint32_t irrad_size = wgpuTextureGetWidth(irradiance_cubemap);
    const uint32_t wg_x       = (irrad_size + wg - 1) / wg;
    const uint32_t wg_y       = (irrad_size + wg - 1) / wg;

    wgpuComputePassEncoderSetPipeline(pass, pipeline_irradiance);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, NULL);
    wgpuComputePassEncoderSetBindGroup(pass, 2, mip_bgs[0], 0, NULL);

    for (uint32_t face = 0; face < 6; ++face) {
      wgpuComputePassEncoderSetBindGroup(pass, 1, face_bgs[face], 0, NULL);
      wgpuComputePassEncoderDispatchWorkgroups(pass, wg_x, wg_y, 1);
    }
  }

  /* --- Pass 2: prefiltered specular cubemap --- */
  {
    wgpuComputePassEncoderSetPipeline(pass, pipeline_prefilter);

    for (uint32_t face = 0; face < 6; ++face) {
      wgpuComputePassEncoderSetBindGroup(pass, 1, face_bgs[face], 0, NULL);

      for (uint32_t mip = 0; mip < pf_mip_count; ++mip) {
        wgpuComputePassEncoderSetBindGroup(pass, 2, mip_bgs[mip], 0, NULL);

        const uint32_t mip_w
          = MAX(1u, wgpuTextureGetWidth(prefiltered_cubemap) >> mip);
        const uint32_t mip_h
          = MAX(1u, wgpuTextureGetHeight(prefiltered_cubemap) >> mip);
        const uint32_t wg_x = (mip_w + wg - 1) / wg;
        const uint32_t wg_y = (mip_h + wg - 1) / wg;

        wgpuComputePassEncoderDispatchWorkgroups(pass, wg_x, wg_y, 1);
      }
    }
  }

  /* --- Pass 3: BRDF integration LUT --- */
  {
    const uint32_t lut_size = wgpuTextureGetWidth(brdf_lut);
    const uint32_t wg_x     = (lut_size + wg - 1) / wg;
    const uint32_t wg_y     = (lut_size + wg - 1) / wg;

    wgpuComputePassEncoderSetPipeline(pass, pipeline_brdf_lut);
    wgpuComputePassEncoderDispatchWorkgroups(pass, wg_x, wg_y, 1);
  }

  wgpuComputePassEncoderEnd(pass);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(queue, 1, &cmd);

  /* --- Cleanup --- */
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd);
  WGPU_RELEASE_RESOURCE(ComputePassEncoder, pass);
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder);
  WGPU_RELEASE_RESOURCE(BindGroup, bg0);
  WGPU_RELEASE_RESOURCE(TextureView, env_view);
  WGPU_RELEASE_RESOURCE(TextureView, irrad_view);
  WGPU_RELEASE_RESOURCE(TextureView, brdf_view);

  for (uint32_t m = 0; m < pf_mip_count; ++m) {
    WGPU_RELEASE_RESOURCE(BindGroup, mip_bgs[m]);
    WGPU_RELEASE_RESOURCE(Buffer, mip_bufs[m]);
  }
  for (uint32_t i = 0; i < 6; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, face_bgs[i]);
    WGPU_RELEASE_RESOURCE(Buffer, face_bufs[i]);
  }

  WGPU_RELEASE_RESOURCE(ComputePipeline, pipeline_irradiance);
  WGPU_RELEASE_RESOURCE(ComputePipeline, pipeline_prefilter);
  WGPU_RELEASE_RESOURCE(ComputePipeline, pipeline_brdf_lut);
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipe_layout);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl0);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl1);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl2);
  WGPU_RELEASE_RESOURCE(Buffer, num_samples_buf);
  WGPU_RELEASE_RESOURCE(Sampler, env_sampler);
  WGPU_RELEASE_RESOURCE(ShaderModule, shader);

  return true;
}

/* -- Public API: environment loading -------------------------------------- */

bool wgpu_environment_load_from_file(wgpu_environment_t* env,
                                     const char* filepath)
{
  if (!env || !filepath) {
    return false;
  }
  memset(env, 0, sizeof(wgpu_environment_t));

  int width = 0, height = 0, channels = 0;
  float* data
    = image_pixels_hdr_from_file(filepath, &width, &height, &channels, 4);
  if (!data) {
    fprintf(stderr, "wgpu_environment: failed to load '%s': %s\n", filepath,
            image_failure_reason());
    return false;
  }

  /* Validate 2:1 aspect ratio */
  if (width != 2 * height) {
    fprintf(stderr,
            "wgpu_environment: image must be 2:1 aspect ratio, got %dx%d\n",
            width, height);
    image_free(data);
    return false;
  }

  /* Downsample if wider than max */
  if (width > (int)IBL_MAX_PANORAMA_WIDTH) {
    uint32_t new_w = 0, new_h = 0;
    float* downsampled
      = ibl_downsample_panorama(data, width, height, &new_w, &new_h);
    image_free(data);
    if (!downsampled) {
      fprintf(stderr, "wgpu_environment: downsample allocation failed\n");
      return false;
    }
    env->data   = downsampled;
    env->width  = new_w;
    env->height = new_h;
  }
  else {
    env->data   = data;
    env->width  = (uint32_t)width;
    env->height = (uint32_t)height;
  }

  env->rotation = 0.0f;
  return true;
}

bool wgpu_environment_load_from_memory(wgpu_environment_t* env,
                                       const uint8_t* data, uint32_t size)
{
  if (!env || !data || size == 0) {
    return false;
  }
  memset(env, 0, sizeof(wgpu_environment_t));

  int width = 0, height = 0, channels = 0;
  float* pixels = image_pixels_hdr_from_memory(data, (int)size, &width, &height,
                                               &channels, 4);
  if (!pixels) {
    fprintf(stderr, "wgpu_environment: failed to load from memory: %s\n",
            image_failure_reason());
    return false;
  }

  /* Validate 2:1 aspect ratio */
  if (width != 2 * height) {
    fprintf(stderr,
            "wgpu_environment: image must be 2:1 aspect ratio, got %dx%d\n",
            width, height);
    image_free(pixels);
    return false;
  }

  /* Downsample if wider than max */
  if (width > (int)IBL_MAX_PANORAMA_WIDTH) {
    uint32_t new_w = 0, new_h = 0;
    float* downsampled
      = ibl_downsample_panorama(pixels, width, height, &new_w, &new_h);
    image_free(pixels);
    if (!downsampled) {
      fprintf(stderr, "wgpu_environment: downsample allocation failed\n");
      return false;
    }
    env->data   = downsampled;
    env->width  = new_w;
    env->height = new_h;
  }
  else {
    env->data   = pixels;
    env->width  = (uint32_t)width;
    env->height = (uint32_t)height;
  }

  env->rotation = 0.0f;
  return true;
}

void wgpu_environment_release(wgpu_environment_t* env)
{
  if (!env) {
    return;
  }
  if (env->data) {
    /* Data may come from stbi (image_free) or malloc (free).
     * Since ibl_downsample_panorama uses malloc and the original stbi data
     * is freed inside the load functions, we always use free() here. */
    free(env->data);
    env->data = NULL;
  }
  env->width    = 0;
  env->height   = 0;
  env->rotation = 0.0f;
}

/* -- Public API: IBL texture generation ----------------------------------- */

bool wgpu_ibl_textures_from_environment(wgpu_context_t* wgpu_context,
                                        const wgpu_environment_t* env,
                                        const wgpu_ibl_textures_desc_t* desc,
                                        wgpu_ibl_textures_t* ibl)
{
  if (!wgpu_context || !env || !env->data || !ibl) {
    return false;
  }
  memset(ibl, 0, sizeof(wgpu_ibl_textures_t));

  /* Resolve parameters with defaults */
  const uint32_t env_size
    = desc ? VALUE_OR(desc->environment_size, IBL_DEFAULT_ENVIRONMENT_SIZE) :
             IBL_DEFAULT_ENVIRONMENT_SIZE;
  const uint32_t irr_size
    = desc ? VALUE_OR(desc->irradiance_size, IBL_DEFAULT_IRRADIANCE_SIZE) :
             IBL_DEFAULT_IRRADIANCE_SIZE;
  const uint32_t pf_size
    = desc ? VALUE_OR(desc->prefiltered_size, IBL_DEFAULT_PREFILTERED_SIZE) :
             IBL_DEFAULT_PREFILTERED_SIZE;
  const uint32_t lut_size
    = desc ? VALUE_OR(desc->brdf_lut_size, IBL_DEFAULT_BRDF_LUT_SIZE) :
             IBL_DEFAULT_BRDF_LUT_SIZE;
  const uint32_t n_samples
    = desc ? VALUE_OR(desc->num_samples, IBL_DEFAULT_NUM_SAMPLES) :
             IBL_DEFAULT_NUM_SAMPLES;

  WGPUDevice device = wgpu_context->device;

  /* Compute mip level counts */
  const uint32_t env_mip_levels
    = wgpu_texture_mip_level_count(env_size, env_size);
  const uint32_t pf_mip_levels = wgpu_texture_mip_level_count(pf_size, pf_size);

  ibl->prefiltered_mip_levels = pf_mip_levels;

  /* --- Step 1: Create environment cubemap --- */
  ibl->environment_cubemap = ibl_create_cubemap(
    device, env_size, env_mip_levels,
    WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding
      | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst);

  if (!ibl->environment_cubemap) {
    fprintf(stderr, "ibl: failed to create environment cubemap\n");
    return false;
  }

  /* --- Step 2: Convert panorama to cubemap --- */
  wgpu_panorama_to_cubemap_converter_t* converter
    = wgpu_panorama_to_cubemap_converter_create(device);
  if (!converter) {
    fprintf(stderr, "ibl: failed to create panorama converter\n");
    wgpu_ibl_textures_destroy(ibl);
    return false;
  }
  bool convert_ok = wgpu_panorama_to_cubemap_converter_convert(
    converter, env->data, env->width, env->height, ibl->environment_cubemap);
  wgpu_panorama_to_cubemap_converter_destroy(converter);
  if (!convert_ok) {
    fprintf(stderr, "ibl: panorama-to-cubemap conversion failed\n");
    wgpu_ibl_textures_destroy(ibl);
    return false;
  }

  /* --- Step 3: Generate mipmaps for environment cubemap --- */
  wgpu_generate_mipmaps(wgpu_context, ibl->environment_cubemap,
                        WGPU_MIPMAP_VIEW_CUBE);

  /* --- Step 4: Create IBL output textures --- */
  ibl->irradiance_cubemap = ibl_create_cubemap(
    device, irr_size, 1,
    WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding);

  ibl->prefiltered_cubemap = ibl_create_cubemap(
    device, pf_size, pf_mip_levels,
    WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding);

  ibl->brdf_lut = ibl_create_texture_2d(device, lut_size,
                                        WGPUTextureUsage_TextureBinding
                                          | WGPUTextureUsage_StorageBinding);

  if (!ibl->irradiance_cubemap || !ibl->prefiltered_cubemap || !ibl->brdf_lut) {
    fprintf(stderr, "ibl: failed to create output textures\n");
    wgpu_ibl_textures_destroy(ibl);
    return false;
  }

  /* --- Step 5: Generate IBL maps --- */
  if (!ibl_generate_maps(wgpu_context, ibl->environment_cubemap,
                         ibl->irradiance_cubemap, ibl->prefiltered_cubemap,
                         ibl->brdf_lut, n_samples)) {
    fprintf(stderr, "ibl: IBL map generation failed\n");
    wgpu_ibl_textures_destroy(ibl);
    return false;
  }

  /* --- Step 6: Create texture views --- */
  ibl->environment_view
    = ibl_create_cube_view(ibl->environment_cubemap, env_mip_levels);
  ibl->irradiance_view = ibl_create_cube_view(ibl->irradiance_cubemap, 1);
  ibl->prefiltered_view
    = ibl_create_cube_view(ibl->prefiltered_cubemap, pf_mip_levels);
  ibl->brdf_lut_view = wgpuTextureCreateView(
    ibl->brdf_lut, &(WGPUTextureViewDescriptor){
                     .format          = WGPUTextureFormat_RGBA16Float,
                     .dimension       = WGPUTextureViewDimension_2D,
                     .baseMipLevel    = 0,
                     .mipLevelCount   = 1,
                     .baseArrayLayer  = 0,
                     .arrayLayerCount = 1,
                     .aspect          = WGPUTextureAspect_All,
                   });

  /* --- Step 7: Create samplers --- */
  ibl->environment_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_Repeat,
              .addressModeW  = WGPUAddressMode_Repeat,
              .minFilter     = WGPUFilterMode_Linear,
              .magFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .lodMinClamp   = 0.0f,
              .lodMaxClamp   = (float)env_mip_levels,
              .maxAnisotropy = 1,
            });

  ibl->brdf_lut_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .minFilter     = WGPUFilterMode_Linear,
              .magFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
              .maxAnisotropy = 1,
            });

  return true;
}

void wgpu_ibl_textures_destroy(wgpu_ibl_textures_t* ibl)
{
  if (!ibl) {
    return;
  }

  WGPU_RELEASE_RESOURCE(Sampler, ibl->brdf_lut_sampler);
  WGPU_RELEASE_RESOURCE(Sampler, ibl->environment_sampler);
  WGPU_RELEASE_RESOURCE(TextureView, ibl->brdf_lut_view);
  WGPU_RELEASE_RESOURCE(TextureView, ibl->prefiltered_view);
  WGPU_RELEASE_RESOURCE(TextureView, ibl->irradiance_view);
  WGPU_RELEASE_RESOURCE(TextureView, ibl->environment_view);
  WGPU_RELEASE_RESOURCE(Texture, ibl->brdf_lut);
  WGPU_RELEASE_RESOURCE(Texture, ibl->prefiltered_cubemap);
  WGPU_RELEASE_RESOURCE(Texture, ibl->irradiance_cubemap);
  WGPU_RELEASE_RESOURCE(Texture, ibl->environment_cubemap);

  memset(ibl, 0, sizeof(wgpu_ibl_textures_t));
}

/* -------------------------------------------------------------------------- *
 * Shared WGSL PBR shader snippets (runtime const-char* versions)
 *
 * These mirror the compile-time WGPU_PBR_WGSL_* macros in pbr.h and are
 * provided for callers that build shader source strings at runtime (e.g.
 * concatenating parts into a buffer before calling
 * wgpuDeviceCreateShaderModule).
 * -------------------------------------------------------------------------- */

// clang-format off
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

const char* wgpu_pbr_wgsl_srgb = CODE(
  // sRGB ↔ linear conversion (IEC 61966-2-1 accurate piecewise)
  fn SRGBtoLINEAR(srgbIn : vec4f) -> vec4f {
    let bLess = step(vec3f(0.04045), srgbIn.xyz);
    let linOut = mix(srgbIn.xyz / vec3f(12.92),
                     pow((srgbIn.xyz + vec3f(0.055)) / vec3f(1.055), vec3f(2.4)),
                     bLess);
    return vec4f(linOut, srgbIn.w);
  }
  fn linearToSRGB(linIn : vec3f) -> vec3f {
    let cutoff = step(vec3f(0.0031308), linIn);
    let higher = vec3f(1.055) * pow(linIn, vec3f(1.0 / 2.4)) - vec3f(0.055);
    let lower  = linIn * vec3f(12.92);
    return mix(lower, higher, cutoff);
  }
);

const char* wgpu_pbr_wgsl_tone_mapping = CODE(
  // Uncharted2 filmic tone mapping operator
  fn Uncharted2Tonemap(x : vec3f) -> vec3f {
    let A = 0.15; let B = 0.50; let C = 0.10;
    let D = 0.20; let E = 0.02; let F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F))
           - E / F;
  }

  // PBR Neutral tone mapping (Khronos specification)
  fn toneMapPBRNeutral(colorIn : vec3f) -> vec3f {
    let startCompression : f32 = 0.8 - 0.04;
    let desaturation : f32 = 0.15;
    let x : f32 = min(colorIn.r, min(colorIn.g, colorIn.b));
    let offset : f32 = select(0.04, x - 6.25 * x * x, x < 0.08);
    var color = colorIn - offset;
    let peak : f32 = max(color.r, max(color.g, color.b));
    if (peak < startCompression) { return color; }
    let d : f32 = 1.0 - startCompression;
    let newPeak : f32 = 1.0 - d * d / (peak + d - startCompression);
    color = color * (newPeak / peak);
    let g : f32 = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, newPeak * vec3f(1.0, 1.0, 1.0), g);
  }

  // Dispatched tone mapping with exposure and gamma correction
  // tmType: 0 = PBR Neutral, 1 = Uncharted2 filmic, 2 = Reinhard
  fn toneMap(colorIn : vec3f, exposure : f32, gamma : f32,
             tmType : i32) -> vec3f {
    let invGamma = 1.0 / gamma;
    var color = colorIn * exposure;
    if (tmType == 1) {
      let W = 11.2;
      color = Uncharted2Tonemap(color)
              * (1.0 / Uncharted2Tonemap(vec3f(W)));
    } else if (tmType == 2) {
      color = color / (color + vec3f(1.0));
    } else {
      color = toneMapPBRNeutral(color);
    }
    color = pow(color, vec3f(invGamma));
    return color;
  }
);

const char* wgpu_pbr_wgsl_brdf = CODE(
  // Fresnel-Schlick approximation
  fn FSchlick(f0 : vec3f, f90 : vec3f, vDotH : f32) -> vec3f {
    return f0 + (f90 - f0) * pow(clamp(1.0 - vDotH, 0.0, 1.0), 5.0);
  }

  // Height-correlated Smith visibility function (GGX)
  fn VGGX(nDotL : f32, nDotV : f32, alphaRoughness : f32) -> f32 {
    let a2 = alphaRoughness * alphaRoughness;
    let ggxV = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
    let ggxL = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
    let ggx = ggxV + ggxL;
    if (ggx > 0.0) { return 0.5 / ggx; }
    return 0.0;
  }

  // GGX / Trowbridge-Reitz Normal Distribution Function
  fn DGGX(nDotH : f32, alphaRoughness : f32) -> f32 {
    let alphaRoughnessSq = alphaRoughness * alphaRoughness;
    let f = (nDotH * nDotH) * (alphaRoughnessSq - 1.0) + 1.0;
    return alphaRoughnessSq / (3.141592653589793 * f * f);
  }

  // Lambertian diffuse BRDF with energy conservation via Fresnel weighting
  fn BRDFLambertian(f0 : vec3f, f90 : vec3f, diffuseColor : vec3f,
                    specularWeight : f32, vDotH : f32) -> vec3f {
    return (1.0 - specularWeight * FSchlick(f0, f90, vDotH))
           * (diffuseColor / 3.141592653589793);
  }

  // Cook-Torrance specular micro-facet BRDF
  fn BRDFSpecularGGX(f0 : vec3f, f90 : vec3f, alphaRoughness : f32,
                     specularWeight : f32, vDotH : f32, nDotL : f32,
                     nDotV : f32, nDotH : f32) -> vec3f {
    let F = FSchlick(f0, f90, vDotH);
    let V = VGGX(nDotL, nDotV, alphaRoughness);
    let D = DGGX(nDotH, alphaRoughness);
    return specularWeight * F * V * D;
  }
);

const char* wgpu_pbr_wgsl_ibl_fresnel = CODE(
  // IBL multi-scattering Fresnel (Fdez-Aguera approximation)
  // Requires: iblBRDFIntegrationLUTTexture (texture_2d<f32>) and
  //           iblBRDFIntegrationLUTSampler (sampler) in scope.
  fn getIBLGGXFresnel(n : vec3f, v : vec3f, roughness : f32,
                      F0 : vec3f, specularWeight : f32) -> vec3f {
    let NdotV = max(dot(n, v), 0.0);
    let brdfLUTCoords = vec2f(NdotV, roughness);
    let brdfLUTSample = textureSample(
      iblBRDFIntegrationLUTTexture,
      iblBRDFIntegrationLUTSampler,
      brdfLUTCoords);
    let brdfLUT = brdfLUTSample.rg;
    let fresnelPivot = max(vec3f(1.0 - roughness), F0) - F0;
    let fresnelSingleScatter = F0 + fresnelPivot * pow(1.0 - NdotV, 5.0);
    let FssEss = specularWeight
                 * (fresnelSingleScatter * brdfLUT.x + brdfLUT.y);
    let Ems = 1.0 - (brdfLUT.x + brdfLUT.y);
    let F_avg = specularWeight * (F0 + (1.0 - F0) / 21.0);
    let FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
    return FssEss + FmsEms;
  }
);

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
// clang-format on
