#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - 3D Textures
 *
 * 3D texture loading (and generation using perlin noise) example.
 *
 * Generates a 3D texture on the cpu (using perlin noise), uploads it to the
 * device and samples it to render an animation. 3D textures store volumetric
 * data and interpolate in all three dimensions.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/texture3d/texture3d.cpp
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Translation of Ken Perlin's JAVA implementation
 * (http://mrl.nyu.edu/~perlin/noise/)
 * -------------------------------------------------------------------------- */

typedef struct {
  uint32_t permutations[512];
} perlin_noise_t;

static float fade(float t)
{
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static float lerp(float t, float a, float b)
{
  return a + t * (b - a);
}

static float grad(int hash, float x, float y, float z)
{
  // Convert LO 4 bits of hash code into 12 gradient directions
  const int h   = hash & 15;
  const float u = h < 8 ? x : y;
  const float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
  return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

/**
 * @brief Fisher–Yates shuffle implementation.
 * @see
 * https://stackoverflow.com/questions/42321370/fisher-yates-shuffling-algorithm-in-c
 * @see
 * https://stackoverflow.com/questions/3343797/is-this-c-implementation-of-fisher-yates-shuffle-correct
 * @ref https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
 */
static void fisher_yates_shuffle(uint8_t* values, int n)
{
  // implementation of Fisher
  int i = 0, j = 0; // create local variables to hold values for shuffle
  uint8_t tmp = 0;

  for (i = n - 1; i > 0; i--) {   // for loop to shuffle
    j         = rand() % (i + 1); // randomise j for shuffle with Fisher Yates
    tmp       = values[j];
    values[j] = values[i];
    values[i] = tmp;
  }
}

static void perlin_noise_init(perlin_noise_t* perlin_noise)
{
  // Generate random lookup for permutations containing all numbers from 0..255
  uint8_t plookup[256] = {0};
  for (uint32_t i = 0; i < 256; i++) {
    plookup[i] = (uint8_t)i;
  }
  fisher_yates_shuffle(plookup, 256);
  for (uint32_t i = 0; i < 256; i++) {
    perlin_noise->permutations[i]       = plookup[i];
    perlin_noise->permutations[256 + i] = plookup[i];
  }
}

static float perlin_noise_generate(perlin_noise_t* perlin_noise, float x,
                                   float y, float z)
{
  // Find unit cube that contains point
  const int32_t X = (int32_t)floor(x) & 255;
  const int32_t Y = (int32_t)floor(y) & 255;
  const int32_t Z = (int32_t)floor(z) & 255;
  // Find relative x,y,z of point in cube
  x -= floor(x);
  y -= floor(y);
  z -= floor(z);

  // Compute fade curves for each of x,y,z
  const float u = fade(x);
  const float v = fade(y);
  const float w = fade(z);

  // Hash coordinates of the 8 cube corners
  const uint32_t A  = perlin_noise->permutations[X] + Y;
  const uint32_t AA = perlin_noise->permutations[A] + Z;
  const uint32_t AB = perlin_noise->permutations[A + 1] + Z;
  const uint32_t B  = perlin_noise->permutations[X + 1] + Y;
  const uint32_t BA = perlin_noise->permutations[B] + Z;
  const uint32_t BB = perlin_noise->permutations[B + 1] + Z;

  // And add blended results for 8 corners of the cube;
  const float res = lerp(
    w,
    lerp(v,
         lerp(u, grad(perlin_noise->permutations[AA], x, y, z),
              grad(perlin_noise->permutations[BA], x - 1, y, z)),
         lerp(u, grad(perlin_noise->permutations[AB], x, y - 1, z),
              grad(perlin_noise->permutations[BB], x - 1, y - 1, z))),
    lerp(v,
         lerp(u, grad(perlin_noise->permutations[AA + 1], x, y, z - 1),
              grad(perlin_noise->permutations[BA + 1], x - 1, y, z - 1)),
         lerp(u, grad(perlin_noise->permutations[AB + 1], x, y - 1, z - 1),
              grad(perlin_noise->permutations[BB + 1], x - 1, y - 1, z - 1))));
  return res;
}

/* -------------------------------------------------------------------------- *
 * Fractal noise generator based on perlin noise above
 * -------------------------------------------------------------------------- */

typedef struct fractal_noise_t {
  perlin_noise_t* perlin_noise;
  uint32_t octaves;
  float frequency;
  float amplitude;
  float persistence;
} fractal_noise_t;

static void fractal_noise_init(fractal_noise_t* fractal_noise,
                               perlin_noise_t* perlin_noise)
{
  fractal_noise->perlin_noise = perlin_noise;
  fractal_noise->octaves      = 6;
  fractal_noise->persistence  = 0.5f;
}

static float fractal_noise_generate(fractal_noise_t* fractal_noise, float x,
                                    float y, float z)
{
  float sum       = 0.0f;
  float frequency = 1.0f;
  float amplitude = 1.0f;
  float max       = 0.0f;
  for (uint32_t i = 0; i < fractal_noise->octaves; i++) {
    sum += perlin_noise_generate(fractal_noise->perlin_noise, x * frequency,
                                 y * frequency, z * frequency)
           * amplitude;
    max += amplitude;
    amplitude *= fractal_noise->persistence;
    frequency *= 2.0f;
  }

  sum = sum / max;
  return (sum + 1.0f) / 2.0f;
}

/* -------------------------------------------------------------------------- *
 * WebGPU 3D textures example
 * -------------------------------------------------------------------------- */

#define NOISE_TEXTURE_WIDTH 128
#define NOISE_TEXTURE_HEIGHT 128
#define NOISE_TEXTURE_DEPTH 128
#define NOISE_TEXTURE_SIZE                                                     \
  (NOISE_TEXTURE_WIDTH * NOISE_TEXTURE_HEIGHT * NOISE_TEXTURE_DEPTH)

// Contains all Vulkan objects that are required to store and use a 3D texture
static struct {
  WGPUSampler sampler;
  WGPUTexture texture;
  WGPUTextureView view;
  WGPUTextureFormat format;
  uint32_t width, height, depth;
  uint32_t mip_levels;
  uint8_t data[NOISE_TEXTURE_SIZE];
  struct {
    perlin_noise_t perlin_noise;
    fractal_noise_t fractal_noise;
  } data_generation;
} noise_texture = {0};

// Vertex layout for this example
typedef struct vertex_t {
  vec3 pos;
  vec2 uv;
  vec3 normal;
} vertex_t;

// Vertex buffer
static wgpu_buffer_t vertices = {0};

// Index buffer
static wgpu_buffer_t indices = {0};

// Uniform buffer block object
static wgpu_buffer_t uniform_buffer_vs = {0};

static struct {
  mat4 projection;
  mat4 model_view;
  vec4 view_pos;
  float depth;
} ubo_vs = {
  .depth = 0.0f,
};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL; // solid pipeline layout

// Pipeline
static WGPURenderPipeline render_pipeline = NULL; // solid render pipeline

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

// Bind groups stores the resources bound to the binding points in a shader
static WGPUBindGroup bind_group              = NULL;
static WGPUBindGroupLayout bind_group_layout = NULL;

// Other variables
static const char* example_title = "3D Textures";
static bool prepared             = false;

// Setup a default look-at camera
static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -2.5f});
  camera_set_rotation(context->camera, (vec3){0.0f, 15.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

// Generate randomized noise and upload it to the 3D texture using staging
static void update_noise_texture(wgpu_context_t* wgpu_context)
{
  memset(noise_texture.data, 0, (uint64_t)NOISE_TEXTURE_SIZE);

  perlin_noise_init(&noise_texture.data_generation.perlin_noise);
  fractal_noise_init(&noise_texture.data_generation.fractal_noise,
                     &noise_texture.data_generation.perlin_noise);

  const float noiseScale = (float)(rand() % 10) + 4.0f;

#pragma omp parallel for
  for (int32_t z = 0; z < (int32_t)noise_texture.depth; z++) {
    for (int32_t y = 0; y < (int32_t)noise_texture.height; y++) {
      for (int32_t x = 0; x < (int32_t)noise_texture.width; x++) {
        float nx = (float)x / (float)noise_texture.width;
        float ny = (float)y / (float)noise_texture.height;
        float nz = (float)z / (float)noise_texture.depth;
#define FRACTAL
#ifdef FRACTAL
        float n = fractal_noise_generate(
          &noise_texture.data_generation.fractal_noise, nx * noiseScale,
          ny * noiseScale, nz * noiseScale);
#else
        float n = 20.0 * perlinNoise.noise(nx, ny, nz);
#endif
        n = n - floor(n);

        noise_texture.data[x + y * noise_texture.width
                           + z * noise_texture.width * noise_texture.height]
          = (uint8_t)(floor(n * 255));
      }
    }
  }

  // Copy 3D noise data to texture
  wgpu_image_to_texure(wgpu_context, noise_texture.texture, noise_texture.data,
                       (WGPUExtent3D){
                         .width              = noise_texture.width,
                         .height             = noise_texture.height,
                         .depthOrArrayLayers = noise_texture.depth,
                       },
                       1u);
}

// Prepare all Vulkan resources for the 3D texture
// Does not fill the texture with data
static void prepare_noise_texture(wgpu_context_t* wgpu_context, uint32_t width,
                                  uint32_t height, uint32_t depth)
{
  // A 3D texture is described as: width x height x depth
  noise_texture.width      = width;
  noise_texture.height     = height;
  noise_texture.depth      = depth;
  noise_texture.mip_levels = 1;
  noise_texture.format     = WGPUTextureFormat_R8Unorm;

  WGPUExtent3D texture_extent = {
    .width              = noise_texture.width,
    .height             = noise_texture.height,
    .depthOrArrayLayers = noise_texture.depth,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "3D noise - Texture",
    .size          = texture_extent,
    .mipLevelCount = noise_texture.mip_levels,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_3D,
    .format        = noise_texture.format,
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
  };
  noise_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(noise_texture.texture != NULL);

  // Create sampler
  noise_texture.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .maxAnisotropy = 1,
                          });
  ASSERT(noise_texture.sampler != NULL);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "3D noise - Texture view",
    .dimension       = WGPUTextureViewDimension_3D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  noise_texture.view
    = wgpuTextureCreateView(noise_texture.texture, &texture_view_dec);
  ASSERT(noise_texture.view != NULL);

  update_noise_texture(wgpu_context);
}

static void generate_quad(wgpu_context_t* wgpu_context)
{
  /* Setup vertices for a single uv-mapped quad made from two triangles */
  static const vertex_t vertices_data[4] = {
    [0] = {
      .pos    = {1.0f, 1.0f, 0.0f},
      .uv     = {1.0f, 1.0f},
      .normal = {0.0f, 0.0f, 1.0f},
    },
    [1] = {
      .pos    = {-1.0f, 1.0f, 0.0f},
      .uv     = {0.0f, 1.0f},
      .normal = {0.0f, 0.0f, 1.0f},
    },
    [2] = {
      .pos    = {-1.0f, -1.0f, 0.0f},
      .uv     = {0.0f, 0.0f},
      .normal = {0.0f, 0.0f, 1.0f},
    },
    [3] = {
      .pos    = {1.0f, -1.0f, 0.0f},
      .uv     = {1.0f, 0.0f},
      .normal = {0.0f, 0.0f, 1.0f},
    },
  };

  /* Create vertex buffer */
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertices_data),
                    .count = (uint32_t)ARRAY_SIZE(vertices_data),
                    .initial.data = vertices_data,
                  });

  /* Setup indices */
  static const uint16_t index_buffer[6] = {
    0, 1, 2, /* Vertex 1 */
    2, 3, 0  /* Vertex 2 */
  };

  /* Create index buffer */
  indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(index_buffer),
                    .count = (uint32_t)ARRAY_SIZE(index_buffer),
                    .initial.data = index_buffer,
                  });
}

static void update_uniform_buffers(wgpu_example_context_t* context,
                                   bool view_changed)
{
  if (view_changed) {
    // Pass view matrices to the shaders
    glm_mat4_copy(context->camera->matrices.perspective, ubo_vs.projection);
    glm_mat4_copy(context->camera->matrices.view, ubo_vs.model_view);
    glm_vec4_copy(context->camera->view_pos, ubo_vs.view_pos);
  }
  /* else */ {
    ubo_vs.depth += context->frame_timer * 0.15f;
    if (ubo_vs.depth > 1.0f) {
      ubo_vs.depth = ubo_vs.depth - 1.0f;
    }
  }

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &ubo_vs, sizeof(ubo_vs));
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Create vertex shader uniform buffer block
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_vs),
    });

  // Set uniform buffer block data
  update_uniform_buffers(context, true);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (Vertex shader)
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = uniform_buffer_vs.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Texture view (Fragment shader)
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_3D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      // Binding 2: Sampler (Fragment shader)
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_NonFiltering,
      },
      .texture = {0},
    }
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL)

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this descriptor set layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  // Bind Group
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0 : Vertex shader uniform buffer
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      // Binding 1 : Fragment shader texture view
      .binding     = 1,
      .textureView = noise_texture.view,
    },
    [2] = (WGPUBindGroupEntry) {
      // Binding 2: Fragment shader image sampler
      .binding = 2,
      .sampler = noise_texture.sampler,
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

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, // Assigned later
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

// Create the graphics pipeline
static void prepare_pipelines(wgpu_context_t* wgpu_context)
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
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    quad, sizeof(vertex_t),
    /* Attribute descriptions */
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    // Attribute location 1: Texture coordinates
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, offsetof(vertex_t, uv)),
    // Attribute location 2: Vertex normal
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, normal)))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader SPIR-V
                  .label = "Texture 3d - vertex shader",
                  .file  = "shaders/texture_3d/texture_3d.vert.spv",
                },
                .buffer_count = 1,
                .buffers = &quad_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader SPIR-V
                  .label = "Texture 3d - fragment shader",
                  .file  = "shaders/texture_3d/texture_3d.frag.spv",
                },
                .target_count = 1,
                .targets = &color_target_state,
              });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Texture 3d - quad render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(render_pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    generate_quad(context->wgpu_context);
    prepare_uniform_buffers(context);
    prepare_noise_texture(context->wgpu_context, (uint32_t)NOISE_TEXTURE_WIDTH,
                          (uint32_t)NOISE_TEXTURE_HEIGHT,
                          (uint32_t)NOISE_TEXTURE_DEPTH);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    if (imgui_overlay_button(context->imgui_overlay, "Generate New Texture")) {
      update_noise_texture(context->wgpu_context);
    }
  }
}

/* Build separate command buffer for the framebuffer image */
static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, render_pipeline);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Bind triangle vertex buffer (contains position and colors) */
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);

  /* Bind triangle index buffer */
  wgpuRenderPassEncoderSetIndexBuffer(wgpu_context->rpass_enc, indices.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);

  /* Draw indexed triangle */
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, indices.count, 1, 0,
                                   0, 0);

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  /* Submit to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  const int draw_result = example_draw(context);
  if (!context->paused || context->camera->updated) {
    update_uniform_buffers(context, context->camera->updated);
  }
  return draw_result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  WGPU_RELEASE_RESOURCE(Texture, noise_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, noise_texture.view)
  WGPU_RELEASE_RESOURCE(Sampler, noise_texture.sampler)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipeline)
}

void example_texture_3d(int argc, char* argv[])
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
