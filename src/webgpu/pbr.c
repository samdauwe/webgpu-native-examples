#include "pbr.h"

#include <cglm/cglm.h>
#include <math.h>

#include "../core/macro.h"
#include "buffer.h"
#include "shader.h"

// Shaders
// clang-format off
static const char* pbr_gen_brdf_lut_vertex_shader_wgsl = CODE(
  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) outUV : vec2<f32>,
  }

  @vertex
  fn main(
    @builtin(vertex_index) vertexIndex : u32
  ) -> Output {
    var output : Output;
    output.outUV = vec2<f32>(f32((vertexIndex << 1) & 2), f32(vertexIndex & 2));
    output.position = vec4<f32>(output.outUV * 2.0 - 1.0, 0.0, 1.0);
    return output;
  }
);

static const char* pbr_gen_brdf_lut_fragment_shader_wgsl = CODE(
  const NUM_SAMPLES = 1024u;
  const PI = 3.14159265359;

  // Based omn http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
  fn random(co : vec2<f32>) -> f32 {
    let a : f32 = 12.9898;
    let b : f32 = 78.233;
    let c : f32 = 43758.5453;
    let dt : f32 = dot(co.xy, vec2<f32>(a,b));
    let sn : f32 = dt % 3.14;
    return fract(sin(sn) * c);
  }

  fn hammersley2d(i : u32, N : u32) -> vec2f {
    // Radical inverse based on http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
    var bits : u32 = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    let rdi : f32 = f32(bits) * 2.3283064365386963e-10;
    return vec2<f32>(f32(i) /f32(N), rdi);
  }

  // Based on http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_slides.pdf
  fn importanceSample_GGX(Xi : vec2<f32>, roughness : f32, normal : vec3<f32>) -> vec3f {
    // Maps a 2D point to a hemisphere with spread based on roughness
    let alpha : f32 = roughness * roughness;
    let phi : f32 = 2.0 * PI * Xi.x + random(normal.xz) * 0.1;
    let cosTheta : f32 = sqrt((1.0 - Xi.y) / (1.0 + (alpha*alpha - 1.0) * Xi.y));
    let sinTheta : f32 = sqrt(1.0 - cosTheta * cosTheta);
    let H : vec3<f32> = vec3<f32>(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    // Tangent space
    let up : vec3<f32> = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
    let tangentX : vec3<f32> = normalize(cross(up, normal));
    let tangentY : vec3<f32> = normalize(cross(normal, tangentX));

    // Convert to world Space
    return normalize(tangentX * H.x + tangentY * H.y + normal * H.z);
  }

  // Geometric Shadowing function
  fn G_SchlicksmithGGX(dotNL : f32, dotNV : f32, roughness : f32) -> f32 {
    let k : f32 = (roughness * roughness) / 2.0;
    let GL : f32 = dotNL / (dotNL * (1.0 - k) + k);
    let GV : f32 = dotNV / (dotNV * (1.0 - k) + k);
    return GL * GV;
  }

  fn BRDF(NoV : f32, roughness : f32) -> vec2f {
    // Normal always points along z-axis for the 2D lookup
    let N : vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
    let V : vec3<f32> = vec3<f32>(sqrt(1.0 - NoV * NoV), 0.0, NoV);

    var LUT : vec2<f32> = vec2<f32>(0.0);
    for(var i : u32 = 0u; i < NUM_SAMPLES; i++) {
      let Xi : vec2<f32> = hammersley2d(i, NUM_SAMPLES);
      let H : vec3<f32> = importanceSample_GGX(Xi, roughness, N);
      let L : vec3<f32> = 2.0 * dot(V, H) * H - V;

      let dotNL : f32 = max(dot(N, L), 0.0);
      let dotNV : f32 = max(dot(N, V), 0.0);
      let dotVH : f32 = max(dot(V, H), 0.0);
      let dotNH : f32 = max(dot(H, N), 0.0);

      if (dotNL > 0.0) {
        let G : f32 = G_SchlicksmithGGX(dotNL, dotNV, roughness);
        let G_Vis : f32 = (G * dotVH) / (dotNH * dotNV);
        let Fc : f32 = pow(1.0 - dotVH, 5.0);
        LUT += vec2<f32>((1.0 - Fc) * G_Vis, Fc * G_Vis);
      }
    }
    return LUT / f32(NUM_SAMPLES);
  }

  @fragment
  fn main(
    @location(0) inUV : vec2<f32>,
  ) -> @location(0) vec4<f32> {
    return vec4<f32>(BRDF(inUV.x, 1.0 - inUV.y), 0.0, 1.0);
  }
);

static const char* pbr_filter_cube_vertex_shader_wgsl = CODE(
  struct Consts {
    mvp : mat4x4<f32>,
  };

  @group(0) @binding(0) var<uniform> consts : Consts;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) outUVW : vec3<f32>
  }

  @vertex
  fn main(
    @location(0) inPos : vec3<f32>
  ) -> Output {
    var output: Output;
    output.outUVW = inPos;
    output.position = consts.mvp * vec4<f32>(inPos.xyz, 1.0);
    return output;
  }
);

static const char* pbr_irradiance_cube_fragment_shader_wgsl = CODE(
  struct Consts {
    deltaPhi : f32,
    deltaTheta : f32,
  };

  @group(0) @binding(1) var<uniform> consts : Consts;
  @group(0) @binding(2) var textureEnv: texture_cube<f32>;
  @group(0) @binding(3) var samplerEnv: sampler;

  const PI = 3.1415926535897932384626433832795;

  @fragment
  fn main(
    @location(0) inPos : vec3<f32>,
  ) -> @location(0) vec4<f32> {
    var N : vec3<f32> = normalize(inPos);
    var up : vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(up, N));
    up = cross(N, right);

    let TWO_PI : f32 = PI * 2.0;
    let HALF_PI : f32 = PI * 0.5;

    var color : vec3<f32> = vec3<f32>(0.0);
    var sampleCount : u32 = 0u;
    for(var phi : f32 = 0.0; phi < TWO_PI; phi += consts.deltaPhi) {
      for(var theta : f32 = 0.0; theta < HALF_PI; theta += consts.deltaTheta) {
        let tempVec : vec3<f32> = cos(phi) * right + sin(phi) * up;
        let sampleVector : vec3<f32> = cos(theta) * N + sin(theta) * tempVec;
        color += textureSample(textureEnv, samplerEnv, sampleVector).rgb * cos(theta) * sin(theta);
        sampleCount++;
      }
    }

    return vec4<f32>(PI * color / f32(sampleCount), 1.0);
  }
);

static const char* pbr_prefilter_env_map_fragment_shader_wgsl = CODE(
  struct Consts {
    roughness : f32,
    numSamples : u32,
  };

  @group(0) @binding(1) var<uniform> consts : Consts;
  @group(0) @binding(2) var textureEnv: texture_cube<f32>;
  @group(0) @binding(3) var samplerEnv: sampler;

  const PI = 3.1415926536;

  // Based omn http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
  fn random(co : vec2<f32>) -> f32 {
    let a : f32 = 12.9898;
    let b : f32 = 78.233;
    let c : f32 = 43758.5453;
    let dt : f32 = dot(co.xy, vec2<f32>(a,b));
    let sn : f32 = dt % 3.14;
    return fract(sin(sn) * c);
  }

  fn hammersley2d(i : u32, N : u32) -> vec2f {
    // Radical inverse based on http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
    var bits : u32 = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    let rdi : f32 = f32(bits) * 2.3283064365386963e-10;
    return vec2<f32>(f32(i) /f32(N), rdi);
  }

  // Based on http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_slides.pdf
  fn importanceSample_GGX(Xi : vec2<f32>, roughness : f32, normal : vec3<f32>) -> vec3f {
    // Maps a 2D point to a hemisphere with spread based on roughness
    let alpha : f32 = roughness * roughness;
    let phi : f32 = 2.0 * PI * Xi.x + random(normal.xz) * 0.1;
    let cosTheta : f32 = sqrt((1.0 - Xi.y) / (1.0 + (alpha*alpha - 1.0) * Xi.y));
    let sinTheta : f32 = sqrt(1.0 - cosTheta * cosTheta);
    let H : vec3<f32> = vec3<f32>(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    // Tangent space
    let up : vec3<f32> = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
    let tangentX : vec3<f32> = normalize(cross(up, normal));
    let tangentY : vec3<f32> = normalize(cross(normal, tangentX));

    // Convert to world Space
    return normalize(tangentX * H.x + tangentY * H.y + normal * H.z);
  }

  // Normal Distribution function
  fn D_GGX(dotNH : f32, roughness : f32) -> f32 {
    let alpha : f32 = roughness * roughness;
    let alpha2 : f32 = alpha * alpha;
    let denom : f32 = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
    return (alpha2)/(PI * denom*denom);
  }

  fn prefilterEnvMap(R : vec3<f32>, roughness : f32) -> vec3f {
    let N : vec3<f32> = R;
    let V : vec3<f32> = R;
    var color : vec3<f32> = vec3<f32>(0.0);
    var totalWeight : f32 = 0.0;
    let envMapDim : f32 = f32(textureDimensions(textureEnv).x);
    for(var i : u32 = 0u; i < consts.numSamples; i++) {
        let Xi : vec2<f32> = hammersley2d(i, consts.numSamples);
        let H : vec3<f32> = importanceSample_GGX(Xi, roughness, N);
        let L : vec3<f32> = 2.0 * dot(V, H) * H - V;
        let dotNL : f32 = clamp(dot(N, L), 0.0, 1.0);
        if (dotNL > 0.0) {
          // Filtering based on https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
          let dotNH : f32 = clamp(dot(N, H), 0.0, 1.0);
          let dotVH : f32 = clamp(dot(V, H), 0.0, 1.0);
          // Probability Distribution Function
          let pdf : f32 = D_GGX(dotNH, roughness) * dotNH / (4.0 * dotVH) + 0.0001;
          // Slid angle of current smple
          let omegaS : f32 = 1.0 / (f32(consts.numSamples) * pdf);
          // Solid angle of 1 pixel across all cube faces
          let omegaP : f32 = 4.0 * PI / (6.0 * envMapDim * envMapDim);
          // Biased (+1.0) mip level for better result
          let mipLevel = select(max(0.5 * log2(omegaS / omegaP) + 1.0, 0.0f), 0.0, roughness == 0.0);
          color += textureSampleLevel(textureEnv, samplerEnv, L, mipLevel).rgb * dotNL;
          totalWeight += dotNL;
        }
    }
    return (color / totalWeight);
  }

  @fragment
  fn main(
    @location(0) inPos : vec3<f32>,
  ) -> @location(0) vec4<f32> {
    let N : vec3<f32> = normalize(inPos);
    return vec4<f32>(prefilterEnvMap(N, consts.roughness), 1.0);
  }
);
// clang-format on

texture_t pbr_generate_brdf_lut(wgpu_context_t* wgpu_context)
{
#define BRDF_LUT_DIM 512

  texture_t lut_brdf = {0};

  const WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
  const int32_t dim              = (int32_t)BRDF_LUT_DIM;

  /* Texture dimensions */
  WGPUExtent3D texture_extent = {
    .width              = dim,
    .height             = dim,
    .depthOrArrayLayers = 1,
  };

  /* Create the texture */
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "LUT BRDF texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = format,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    lut_brdf.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(lut_brdf.texture != NULL);
  }

  /* Create the texture view */
  {
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "LUT BRDF texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    lut_brdf.view = wgpuTextureCreateView(lut_brdf.texture, &texture_view_dec);
    ASSERT(lut_brdf.view != NULL);
  }

  /* Create the texture sampler */
  {
    lut_brdf.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = "LUT BRDF texture sampler",
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(lut_brdf.sampler != NULL);
  }

  /* Look-up-table (from BRDF) pipeline */
  WGPURenderPipeline pipeline = NULL;
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader WGSL
                .label            = "Gen BRDF LUT vertex shader",
                .wgsl_code.source = pbr_gen_brdf_lut_vertex_shader_wgsl,
                .entry            = "main",
              },
              .buffer_count = 0,
              .buffers      = NULL,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader WGSL
                .label            = "Gen BRDF LUT fragment shader",
                .wgsl_code.source = pbr_gen_brdf_lut_fragment_shader_wgsl,
                .entry            = "main",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create rendering pipeline using the specified states
    pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Gen BRDF LUT render pipeline",
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = NULL,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Create the actual renderpass */
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass = {
    .color_attachment[0]= (WGPURenderPassColorAttachment) {
        .view       = lut_brdf.view,
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
     },
  };
  render_pass.render_pass_descriptor = (WGPURenderPassDescriptor){
    .label                  = "Gen BRDF LUT render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachment,
    .depthStencilAttachment = NULL,
  };

  /* Render */
  {
    wgpu_context->cmd_enc
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.render_pass_descriptor);
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                     (float)dim, (float)dim, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u, dim,
                                        dim);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);

    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

    WGPUCommandBuffer command_buffer
      = wgpuCommandEncoderFinish(wgpu_context->cmd_enc, NULL);
    ASSERT(command_buffer != NULL);
    WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

    // Sumbit commmand buffer and cleanup
    wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)
  }

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline);

  return lut_brdf;
}

typedef enum pbr_cubemap_type_t {
  PBR_CUBEMAP_TYPE_IRRADIANCE,
  PBR_CUBEMAP_TYPE_PREFILTERED_ENV
} pbr_cubemap_type_t;

/**
 * @brief Offline generation for the cube maps used for PBR lighting
 * - Irradiance cube map
 * - Pre-filterd environment cubemap
 * @see
 * https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
 */
texture_t pbr_generate_cubemap(wgpu_context_t* wgpu_context,
                               pbr_cubemap_type_t cubemap_type,
                               struct gltf_model_t* skybox,
                               texture_t* skybox_texture)
{
#define ALIGNMENT 256 /* 256-byte alignment */
#define IRRADIANCE_CUBE_DIM 64
#define IRRADIANCE_CUBE_NUM_MIPS 7 /* ((uint32_t)(floor(log2(dim)))) + 1; */
#define PREFILTERED_CUBE_DIM 512
#define PREFILTERED_CUBE_NUM_MIPS 10 // ((uint32_t)(floor(log2(dim)))) + 1;

  texture_t cubemap = {0};

  const WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
  int32_t dim                    = 0;
  uint32_t num_mips              = 0;

  switch (cubemap_type) {
    case PBR_CUBEMAP_TYPE_IRRADIANCE: {
      dim      = (int32_t)IRRADIANCE_CUBE_DIM;
      num_mips = (uint32_t)IRRADIANCE_CUBE_NUM_MIPS;
    } break;
    case PBR_CUBEMAP_TYPE_PREFILTERED_ENV: {
      dim      = (int32_t)PREFILTERED_CUBE_DIM;
      num_mips = (uint32_t)PREFILTERED_CUBE_NUM_MIPS;
    } break;
  }

  ASSERT(num_mips == ((uint32_t)(floor(log2(dim)))) + 1);
  const uint32_t array_layer_count = 6u; /* Cube map */

  /** Pre-filtered cube map **/
  // Texture dimensions
  WGPUExtent3D texture_extent = {
    .width              = dim,
    .height             = dim,
    .depthOrArrayLayers = array_layer_count,
  };

  /* Create target cubemap */
  {
    /* Create the texture */
    {
      WGPUTextureDescriptor texture_desc = {
        .label         = "Cubemap texture",
        .size          = texture_extent,
        .mipLevelCount = num_mips,
        .sampleCount   = 1,
        .dimension     = WGPUTextureDimension_2D,
        .format        = format,
        .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst
                 | WGPUTextureUsage_TextureBinding,
      };
      cubemap.texture
        = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
      ASSERT(cubemap.texture != NULL);
    }

    /* Create the texture view */
    {
      WGPUTextureViewDescriptor texture_view_dec = {
        .label           = "Cubemap texture view",
        .dimension       = WGPUTextureViewDimension_Cube,
        .format          = format,
        .baseMipLevel    = 0,
        .mipLevelCount   = num_mips,
        .baseArrayLayer  = 0,
        .arrayLayerCount = array_layer_count,
      };
      cubemap.view = wgpuTextureCreateView(cubemap.texture, &texture_view_dec);
      ASSERT(cubemap.view != NULL);
    }

    /* Create the sampler */
    {
      cubemap.sampler = wgpuDeviceCreateSampler(
        wgpu_context->device, &(WGPUSamplerDescriptor){
                                .label         = "Cubemap texture sampler",
                                .addressModeU  = WGPUAddressMode_ClampToEdge,
                                .addressModeV  = WGPUAddressMode_ClampToEdge,
                                .addressModeW  = WGPUAddressMode_ClampToEdge,
                                .minFilter     = WGPUFilterMode_Linear,
                                .magFilter     = WGPUFilterMode_Linear,
                                .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                                .lodMinClamp   = 0.0f,
                                .lodMaxClamp   = (float)num_mips,
                                .maxAnisotropy = 1,
                              });
      ASSERT(cubemap.sampler != NULL);
    }
  }

  /* Framebuffer for offscreen rendering */
  struct {
    WGPUTexture texture;
    WGPUTextureView
      texture_views_irradiance[6 * (uint32_t)IRRADIANCE_CUBE_NUM_MIPS];
    WGPUTextureView
      texture_views_prefilter_env[6 * (uint32_t)PREFILTERED_CUBE_NUM_MIPS];
  } offscreen;

  WGPUTextureView* offscreen_texture_views = NULL;
  switch (cubemap_type) {
    case PBR_CUBEMAP_TYPE_IRRADIANCE:
      offscreen_texture_views = offscreen.texture_views_irradiance;
      break;
    case PBR_CUBEMAP_TYPE_PREFILTERED_ENV:
      offscreen_texture_views = offscreen.texture_views_prefilter_env;
      break;
  }

  /* Create offscreen framebuffer */
  {
    /* Color attachment */
    {
      // Create the texture
      WGPUTextureDescriptor texture_desc = {
        .label         = "Cubemap offscreen texture",
        .size          = (WGPUExtent3D) {
          .width              = dim,
          .height             = dim,
          .depthOrArrayLayers = array_layer_count,
        },
        .mipLevelCount = num_mips,
        .sampleCount   = 1,
        .dimension     = WGPUTextureDimension_2D,
        .format        = format,
        .usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding
                 | WGPUTextureUsage_RenderAttachment,
      };
      offscreen.texture
        = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
      ASSERT(offscreen.texture != NULL);

      /* Create the texture views */
      uint32_t idx = 0;
      for (uint32_t array_layer = 0; array_layer < array_layer_count;
           ++array_layer) {
        for (uint32_t i = 0; i < num_mips; ++i) {
          idx = (array_layer * num_mips) + i;
          WGPUTextureViewDescriptor texture_view_dec = {
            .label           = "Cube offscreen texture view",
            .aspect          = WGPUTextureAspect_All,
            .dimension       = WGPUTextureViewDimension_2D,
            .format          = texture_desc.format,
            .baseMipLevel    = i,
            .mipLevelCount   = 1,
            .baseArrayLayer  = array_layer,
            .arrayLayerCount = 1,
          };
          offscreen_texture_views[idx]
            = wgpuTextureCreateView(offscreen.texture, &texture_view_dec);
          ASSERT(offscreen_texture_views[idx] != NULL);
        }
      }
    }
  }

  struct push_block_irradiance_vs_t {
    mat4 mvp;
    uint8_t padding[192];
  } push_block_irradiance_vs[(uint32_t)IRRADIANCE_CUBE_NUM_MIPS * 6];

  struct push_block_irradiance_fs_t {
    float delta_phi;
    float delta_theta;
    uint8_t padding[248];
  } push_block_irradiance_fs[(uint32_t)IRRADIANCE_CUBE_NUM_MIPS * 6];

  struct push_block_prefilter_env_vs_t {
    mat4 mvp;
    uint8_t padding[192];
  } push_block_prefilter_env_vs[(uint32_t)PREFILTERED_CUBE_NUM_MIPS * 6];

  struct push_block_prefilter_env_fs_t {
    float roughness;
    uint32_t num_samples;
    uint8_t padding[248];
  } push_block_prefilter_env_fs[(uint32_t)PREFILTERED_CUBE_NUM_MIPS * 6];

  /* Update shader push constant block data */
  {
    mat4 matrices[6] = {
      GLM_MAT4_IDENTITY_INIT, /* POSITIVE_X */
      GLM_MAT4_IDENTITY_INIT, /* NEGATIVE_X */
      GLM_MAT4_IDENTITY_INIT, /* POSITIVE_Y */
      GLM_MAT4_IDENTITY_INIT, /* NEGATIVE_Y */
      GLM_MAT4_IDENTITY_INIT, /* POSITIVE_Z */
      GLM_MAT4_IDENTITY_INIT, /* NEGATIVE_Z */
    };
    /* NEGATIVE_X */
    glm_rotate(matrices[0], glm_rad(90.0f), (vec3){0.0f, 1.0f, 0.0f});
    glm_rotate(matrices[0], glm_rad(180.0f), (vec3){1.0f, 0.0f, 0.0f});
    /* NEGATIVE_X */
    glm_rotate(matrices[1], glm_rad(-90.0f), (vec3){0.0f, 1.0f, 0.0f});
    glm_rotate(matrices[1], glm_rad(180.0f), (vec3){1.0f, 0.0f, 0.0f});
    /* POSITIVE_Y */
    glm_rotate(matrices[2], glm_rad(90.0f), (vec3){1.0f, 0.0f, 0.0f});
    /* NEGATIVE_Y */
    glm_rotate(matrices[3], glm_rad(-90.0f), (vec3){1.0f, 0.0f, 0.0f});
    /* POSITIVE_Z */
    glm_rotate(matrices[4], glm_rad(180.0f), (vec3){1.0f, 0.0f, 0.0f});
    /* NEGATIVE_Z */
    glm_rotate(matrices[5], glm_rad(180.0f), (vec3){0.0f, 0.0f, 1.0f});

    mat4 projection = GLM_MAT4_IDENTITY_INIT;
    glm_perspective(PI / 2.0f, 1.0f, 0.1f, 512.0f, projection);
    // Sampling deltas
    const float delta_phi   = (2.0f * PI) / 180.0f;
    const float delta_theta = (0.5f * PI) / 64.0f;
    uint32_t idx            = 0;
    for (uint32_t m = 0; m < num_mips; ++m) {
      for (uint32_t f = 0; f < 6; ++f) {
        idx = (m * 6) + f;
        switch (cubemap_type) {
          case PBR_CUBEMAP_TYPE_IRRADIANCE: {
            // Set vertex shader push constant block
            glm_mat4_mul(projection, matrices[f],
                         push_block_irradiance_vs[idx].mvp);
            // Set fragment shader push constant block
            push_block_irradiance_fs[idx].delta_phi   = delta_phi;
            push_block_irradiance_fs[idx].delta_theta = delta_theta;
          } break;
          case PBR_CUBEMAP_TYPE_PREFILTERED_ENV: {
            // Set vertex shader push constant block
            glm_mat4_mul(projection, matrices[f],
                         push_block_prefilter_env_vs[idx].mvp);
            // Set fragment shader push constant block
            push_block_prefilter_env_fs[idx].roughness
              = (float)m / (float)(num_mips - 1);
            push_block_prefilter_env_fs[idx].num_samples = 32u;
          } break;
        }
      }
    }
  }

  static struct {
    // Vertex shader parameter uniform buffer
    struct {
      WGPUBuffer buffer;
      uint64_t buffer_size;
      uint64_t model_size;
    } vs;
    // Fragment parameter uniform buffer
    struct {
      WGPUBuffer buffer;
      uint64_t buffer_size;
      uint64_t model_size;
    } fs;
  } cube_ubos = {0};

  switch (cubemap_type) {
    case PBR_CUBEMAP_TYPE_IRRADIANCE: {
      /* Vertex shader parameter uniform buffer */
      {
        cube_ubos.vs.model_size = sizeof(mat4);
        cube_ubos.vs.buffer_size
          = calc_constant_buffer_byte_size(sizeof(push_block_irradiance_vs));
        cube_ubos.vs.buffer = wgpu_create_buffer_from_data(
          wgpu_context, push_block_irradiance_vs, cube_ubos.vs.buffer_size,
          WGPUBufferUsage_Uniform);
      }

      /* Fragment shader parameter uniform buffer */
      {
        cube_ubos.fs.model_size = sizeof(float) * 2;
        cube_ubos.fs.buffer_size
          = calc_constant_buffer_byte_size(sizeof(push_block_irradiance_fs));
        cube_ubos.fs.buffer = wgpu_create_buffer_from_data(
          wgpu_context, push_block_irradiance_fs, cube_ubos.fs.buffer_size,
          WGPUBufferUsage_Uniform);
      }
    } break;
    case PBR_CUBEMAP_TYPE_PREFILTERED_ENV: {
      /* Vertex shader parameter uniform buffer */
      {
        cube_ubos.vs.model_size = sizeof(mat4);
        cube_ubos.vs.buffer_size
          = calc_constant_buffer_byte_size(sizeof(push_block_prefilter_env_vs));
        cube_ubos.vs.buffer = wgpu_create_buffer_from_data(
          wgpu_context, push_block_prefilter_env_vs, cube_ubos.vs.buffer_size,
          WGPUBufferUsage_Uniform);
      }

      /* Fragment shader parameter uniform buffer */
      {
        cube_ubos.fs.model_size = sizeof(float) + sizeof(uint32_t);
        cube_ubos.fs.buffer_size
          = calc_constant_buffer_byte_size(sizeof(push_block_prefilter_env_fs));
        cube_ubos.fs.buffer = wgpu_create_buffer_from_data(
          wgpu_context, push_block_prefilter_env_fs, cube_ubos.fs.buffer_size,
          WGPUBufferUsage_Uniform);
      }
    } break;
  }

  /* Bind group layout */
  WGPUBindGroupLayout bind_group_layout = NULL;
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Vertex shader uniform UBO */
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = cube_ubos.vs.model_size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: Fragment shader uniform UBO */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = cube_ubos.fs.model_size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2: Fragment shader image view */
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        /* Binding 3: Fragment shader image sampler */
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };
    bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layout != NULL);
  }

  /* Bind group */
  WGPUBindGroup bind_group = NULL;
  {
    WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform UBO
        .binding = 0,
        .buffer  = cube_ubos.vs.buffer,
        .offset  = 0,
        .size    = cube_ubos.vs.model_size,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment shader uniform UBO
        .binding = 1,
        .buffer  = cube_ubos.fs.buffer,
        .offset  = 0,
        .size    = cube_ubos.fs.model_size,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader image view
        .binding     = 2,
        .textureView = skybox_texture->view
      },
      [3] = (WGPUBindGroupEntry) {
        // Binding 3: Fragment shader image sampler
        .binding = 3,
        .sampler = skybox_texture->sampler,
      },
    };
    bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Bind group",
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_group != NULL);
  }

  // Pipeline layout
  WGPUPipelineLayout pipeline_layout = NULL;
  {
    pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label                = "Pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &bind_group_layout,
                            });
    ASSERT(pipeline_layout != NULL);
  }

  /* Cubemap pipeline */
  WGPURenderPipeline pipeline = NULL;
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex buffer layout
    WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
      skybox,
      // Location 0: Position
      WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position));

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader WGSL
                .label            = "Cubemap vertex shader",
                .wgsl_code.source = pbr_filter_cube_vertex_shader_wgsl,
                .entry            = "main",
              },
             .buffer_count = 1,
             .buffers      = &skybox_vertex_buffer_layout,
            });

    // Fragment state
    const char* wgsl_code = (cubemap_type == PBR_CUBEMAP_TYPE_IRRADIANCE) ?
                              pbr_irradiance_cube_fragment_shader_wgsl :
                              pbr_prefilter_env_map_fragment_shader_wgsl;
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader WGSL
                .label            = "Cubemap fragment shader",
                .wgsl_code.source = wgsl_code,
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
    pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Cubemap render pipeline",
                              .layout       = pipeline_layout,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = NULL,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Create the actual renderpass */
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass = {
    .color_attachment[0]= (WGPURenderPassColorAttachment) {
        .view       = NULL, /* Assigned later */
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.2f,
          .a = 0.0f,
        },
     },
  };
  render_pass.render_pass_descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachment,
    .depthStencilAttachment = NULL,
  };

  /* Render */
  {
    wgpu_context->cmd_enc
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

    uint32_t idx         = 0;
    float viewport_width = 0.0f, viewport_height = 0.0f;
    for (uint32_t m = 0; m < num_mips; ++m) {
      viewport_width  = (float)(dim * pow(0.5f, m));
      viewport_height = (float)(dim * pow(0.5f, m));
      for (uint32_t f = 0; f < 6; ++f) {
        render_pass.color_attachment[0].view
          = offscreen_texture_views[(f * num_mips) + m];
        idx = (m * 6) + f;
        // Render scene from cube face's point of view
        wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
          wgpu_context->cmd_enc, &render_pass.render_pass_descriptor);
        wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                         viewport_width, viewport_height, 0.0f,
                                         1.0f);
        wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                            (uint32_t)viewport_width,
                                            (uint32_t)viewport_height);
        wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
        // Calculate the dynamic offsets
        uint32_t dynamic_offset     = idx * (uint32_t)ALIGNMENT;
        uint32_t dynamic_offsets[2] = {dynamic_offset, dynamic_offset};
        // Bind the bind group for rendering a mesh using the dynamic offset
        wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                          bind_group, 2, dynamic_offsets);
        // Draw object
        wgpu_gltf_model_draw(skybox, (wgpu_gltf_model_render_options_t){0});
        // End render pass
        wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
        WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
      }
    }

    // Copy region for transfer from framebuffer to cube face
    for (uint32_t m = 0; m < num_mips; ++m) {
      WGPUExtent3D copy_size = (WGPUExtent3D){
        .width              = (float)(dim * pow(0.5f, m)),
        .height             = (float)(dim * pow(0.5f, m)),
        .depthOrArrayLayers = array_layer_count,
      };
      wgpuCommandEncoderCopyTextureToTexture(wgpu_context->cmd_enc,
                                             // source
                                             &(WGPUImageCopyTexture){
                                               .texture  = offscreen.texture,
                                               .mipLevel = m,
                                             },
                                             // destination
                                             &(WGPUImageCopyTexture){
                                               .texture  = cubemap.texture,
                                               .mipLevel = m,
                                             },
                                             // copySize
                                             &copy_size);
    }

    WGPUCommandBuffer command_buffer
      = wgpuCommandEncoderFinish(wgpu_context->cmd_enc, NULL);
    ASSERT(command_buffer != NULL);
    WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

    // Sumbit commmand buffer and cleanup
    wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)
  }

  // Cleanup
  WGPU_RELEASE_RESOURCE(Texture, offscreen.texture)
  for (uint32_t i = 0; i < array_layer_count * num_mips; ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, offscreen_texture_views[i])
  }
  WGPU_RELEASE_RESOURCE(Buffer, cube_ubos.vs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, cube_ubos.fs.buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)

  return cubemap;
}

texture_t pbr_generate_irradiance_cube(wgpu_context_t* wgpu_context,
                                       struct gltf_model_t* skybox,
                                       texture_t* skybox_texture)
{
  return pbr_generate_cubemap(wgpu_context, PBR_CUBEMAP_TYPE_IRRADIANCE, skybox,
                              skybox_texture);
}

texture_t pbr_generate_prefiltered_env_cube(wgpu_context_t* wgpu_context,
                                            struct gltf_model_t* skybox,
                                            texture_t* skybox_texture)
{
  return pbr_generate_cubemap(wgpu_context, PBR_CUBEMAP_TYPE_PREFILTERED_ENV,
                              skybox, skybox_texture);
}
