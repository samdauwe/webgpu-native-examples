#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Gerstner Waves
 *
 * This example is a WebGPU implementation of the Gerstner Waves algorithm.
 *
 * Ref:
 * https://github.com/artemhlezin/webgpu-gerstner-waves
 * https://en.wikipedia.org/wiki/Trochoidal_wave
 * https://www.reddit.com/r/webgpu/comments/s2elkb/webgpu_gerstner_waves_implementation
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* gerstner_waves_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Matrix utility functions
 * -------------------------------------------------------------------------- */

static void create_orbit_view_matrix(float radius, versor rotation, mat4* dest)
{
  // inv(R*T)
  mat4 view_matrix = GLM_MAT4_ZERO_INIT;
  glm_quat_mat4(rotation, view_matrix);
  glm_translate(view_matrix, (vec3){0.0f, 0.0f, radius});
  glm_mat4_inv(view_matrix, *dest);
}

static void position_from_view_matrix(mat4 view_matrix, vec3* dest)
{
  mat4 inv_view = GLM_MAT4_ZERO_INIT;
  glm_mat4_inv(view_matrix, inv_view);
  glm_vec3_copy((vec3){inv_view[3][0], inv_view[3][1], inv_view[3][2]}, *dest);
}

/**
 * @brief Creates a quaternion from the given euler angle x, y, z.
 *
 * @param {quat} out the receiving quaternion
 * @param {x} Angle to rotate around X axis in degrees.
 * @param {y} Angle to rotate around Y axis in degrees.
 * @param {z} Angle to rotate around Z axis in degrees.
 * @returns {quat} out
 * @function
 * @ref https://glmatrix.net/docs/module-quat.html
 * @see https://glmatrix.net/docs/quat.js.html#line459
 */
static void from_euler(float x, float y, float z, versor* dest)
{
  const float halfToRad = PI_2 / 180.0f;

  x *= halfToRad;
  y *= halfToRad;
  z *= halfToRad;

  const float sx = sin(x);
  const float cx = cos(x);
  const float sy = sin(y);
  const float cy = cos(y);
  const float sz = sin(z);
  const float cz = cos(z);

  (*dest)[0] = sx * cy * cz - cx * sy * sz;
  (*dest)[1] = cx * sy * cz + sx * cy * sz;
  (*dest)[2] = cx * cy * sz - sx * sy * cz;
  (*dest)[3] = cx * cy * cz + sx * sy * sz;
}

/* -------------------------------------------------------------------------- *
 * Gerstner Waves example.
 * -------------------------------------------------------------------------- */

#define SAMPLE_COUNT (4)

/* State struct */
static struct {
  plane_mesh_t plane_mesh;
  wgpu_buffer_t vertices;
  wgpu_buffer_t indices;
  struct {
    wgpu_buffer_t scene;
    wgpu_buffer_t gerstner_wave_params;
  } uniform_buffers;
  struct {
    float elapsed_time;
    float padding[3];
    mat4 model_matrix;
    mat4 view_projection_matrix;
    vec3 view_position;
  } scene_data;
  struct {
    mat4 view_matrix;
    versor rotation;
    mat4 projection_matrix;
  } tmp_mtx;
  vec2 current_mouse_position;
  struct {
    // Uniform storage requires that array elements be aligned to 16 bytes.
    // 4 bytes waveLength + 4 bytes amplitude + 4+4 bytes steepness
    // + 8+8 bytes direction = 32 Bytes
    struct {
      float wave_length; // 0 < L
      float amplitude;   // 0 < A
      float steepness;   // Steepness of the peak of the wave. 0 <= S <= 1
      float padding1;
      vec2 direction; // Normalized direction of the wave
      vec2 padding2;
    } waves[5];
    float amplitude_sum; // Sum of waves amplitudes
    float padding;       // The shader uses 168 bytes
  } gerstner_wave_params;
  WGPUBool gerstner_waves_normalized;
  wgpu_texture_t sea_color_texture;
  uint8_t file_buffer[128 * 1 * 10];
  WGPUSampler non_filtering_sampler;
  struct {
    WGPUBindGroupLayout uniforms;
    WGPUBindGroupLayout textures;
  } bind_group_layouts;
  struct {
    WGPUBindGroup uniforms;
    WGPUBindGroup textures;
  } bind_groups;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  uint32_t sample_count;
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
    // Multi-sampled texture
    struct {
      WGPUTexture texture;
      WGPUTextureView view;
      uint32_t sample_count;
    } multisampled_framebuffer;
  } render_pass;
  WGPUBool initialized;
} state = {
  .current_mouse_position = {50.0f, -25.0f},
  .tmp_mtx = {
    .view_matrix       = GLM_MAT4_ZERO_INIT,
    .rotation          = GLM_VEC4_ZERO_INIT,
    .projection_matrix = GLM_MAT4_ZERO_INIT,
  },
  .gerstner_wave_params = {
    .waves = {
      {
        .wave_length = 8.0f, // f32 - 4 bytes
        .amplitude   = 0.1f, // f32 - 4 bytes
        .steepness   = 1.0f, // f32 - 4 bytes, but 8 bytes will be reserved to match 32 bytes stride
        .direction   = {1.0f, 1.3f}, // vec2<f32> - 8 bytes but 16 bytes will be reserved
      },
      {
        .wave_length = 4.0f,
        .amplitude   = 0.1f,
        .steepness   = 0.8f,
        .direction   ={-0.7f, 0.0f},
      },
      {
        .wave_length = 5.0f,
        .amplitude   = 0.2f,
        .steepness   = 1.0f,
        .direction   = {0.3f, 0.2f},
      },
      {
        .wave_length = 10.f,
        .amplitude   = 0.5f,
        .steepness   = 1.0f,
        .direction   = {4.3f, 1.2f},
      },
      {
        .wave_length = 3.0f,
        .amplitude   = 0.1f,
        .steepness   = 1.0f,
        .direction   = {0.5f, 0.5f},
      },
    },
  },
  .sample_count = SAMPLE_COUNT,
  .render_pass = {
    .color_attachment = {
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.3, 0.3, 0.3, 1.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    },
    .depth_stencil_attachment = {
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    },
    .descriptor = {
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.render_pass.color_attachment,
      .depthStencilAttachment = &state.render_pass.depth_stencil_attachment,
    },
    .multisampled_framebuffer.sample_count = SAMPLE_COUNT,
  }
};

static void init_plane_mesh(void)
{
  plane_mesh_init(&state.plane_mesh, &(plane_mesh_init_options_t){
                                       .width   = 12.0f,
                                       .height  = 12.0f,
                                       .rows    = 100,
                                       .columns = 100,
                                     });
}

/* Initialize vertex and index buffers for an indexed plane mesh */
static void init_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  /* Create vertex buffer */
  state.vertices = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Plane mesh - Vertex buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
      .size         = state.plane_mesh.vertex_count * sizeof(plane_vertex_t),
      .count        = state.plane_mesh.vertex_count,
      .initial.data = state.plane_mesh.vertices,
    });

  /* Create index buffer */
  state.indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Plane mesh - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = state.plane_mesh.index_count * sizeof(uint32_t),
                    .count = state.plane_mesh.index_count,
                    .initial.data = state.plane_mesh.indices,
                  });
}

static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("File fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* The file data has been fetched, since we provided a big-enough buffer we
   * can be sure that all data has been loaded here */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc = (wgpu_texture_desc_t){
        .extent = (WGPUExtent3D) {
          .width              = img_width,
          .height             = img_height,
          .depthOrArrayLayers = 4,
      },
        .format = WGPUTextureFormat_RGBA8Unorm,
        .pixels = {
          .ptr  = pixels,
          .size = img_width * img_height * 4,
      },
    };
    texture->desc.is_dirty = true;
  }
}

static void init_texture(wgpu_context_t* wgpu_context)
{
  /* Dummy particle texture */
  state.sea_color_texture
    = wgpu_create_color_bars_texture(wgpu_context, 16, 16);

  /* Start loading the image file */
  const char* particle_texture_path = "assets/textures/sea-color.jpg";
  wgpu_texture_t* texture           = &state.sea_color_texture;
  sfetch_send(&(sfetch_request_t){
    .path      = particle_texture_path,
    .callback  = fetch_callback,
    .buffer    = SFETCH_RANGE(state.file_buffer),
    .user_data = {
      .ptr  = &texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });
}

static void init_texture_sampler(wgpu_context_t* wgpu_context)
{
  /* Create non-filtering sampler */
  WGPUSamplerDescriptor sampler_desc = {
    .label         = STRVIEW("Non-filtering texture - Sampler"),
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .maxAnisotropy = 1,
  };
  state.non_filtering_sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(state.non_filtering_sampler != NULL);
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout for Gerstner Waves mesh rendering & parameters */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Uniforms - Scene data */
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(state.scene_data),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: GerstnerWavesUniforms */
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(state.gerstner_wave_params),
        },
        .sampler = {0},
      },
    };

    /* Create the bind group layout */
    state.bind_group_layouts.uniforms = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Bind group layout - Gerstner Waves mesh"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(state.bind_group_layouts.uniforms != NULL);
  }

  /* Bind group layout for sea color texture */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Sampler */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_NonFiltering,
        },
        .texture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: Texture view */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      }
    };
    state.bind_group_layouts.textures = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Bind group layout - Sea color texture"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(state.bind_group_layouts.textures != NULL);
  }

  /* Create the pipeline layout from bind group layouts */
  WGPUBindGroupLayout bind_groups_layout_array[2] = {
    state.bind_group_layouts.uniforms, /* Group 0 */
    state.bind_group_layouts.textures  /* Group 1 */
  };
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_groups_layout_array),
      .bindGroupLayouts     = bind_groups_layout_array,
    });
  ASSERT(state.pipeline_layout != NULL);
}

/* Bind group for Gerstner Waves mesh rendering & parameters */
static void init_scene_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0: Uniforms */
      .binding = 0,
      .buffer  = state.uniform_buffers.scene.buffer,
      .offset  = 0,
      .size    = state.uniform_buffers.scene.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1: GerstnerWavesUniforms */
      .binding = 1,
      .buffer  = state.uniform_buffers.gerstner_wave_params.buffer,
      .offset  = 0,
      .size    = state.uniform_buffers.gerstner_wave_params.size,
    },
  };

  state.bind_groups.uniforms = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Mesh rendering & parameters - Bind group"),
      .layout     = state.bind_group_layouts.uniforms,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(state.bind_groups.uniforms != NULL);
}

/* Bind group for sea color texture */
static void init_texture_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
       /* Binding 0: Sampler */
      .binding = 0,
      .sampler = state.non_filtering_sampler,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1: Texture view */
      .binding     = 1,
      .textureView = state.sea_color_texture.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Bind group - Sea color texture"),
    .layout     = state.bind_group_layouts.textures,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.bind_groups.textures
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.bind_groups.textures != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{

  WGPUShaderModule shader_module = wgpu_create_shader_module(
    wgpu_context->device, gerstner_waves_shader_wgsl);

  /* Blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    plane, sizeof(plane_vertex_t),
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                       offsetof(plane_vertex_t, position)),
    // Attribute location 1: Normal
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       offsetof(plane_vertex_t, normal)),
    // Attribute location 2: UV
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x2,
                       offsetof(plane_vertex_t, uv)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Gerstner waves - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = shader_module,
      .entryPoint  = STRVIEW("vertex_main"),
      .bufferCount = 1,
      .buffers     = &plane_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("fragment_main"),
      .module      = shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_None,
      .frontFace = WGPUFrontFace_CCW
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
       .count = SAMPLE_COUNT,
       .mask  = 0xffffffff
    },
  };

  state.render_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.render_pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);
}

/* Create attachment for multisampling support */
static void init_multisampled_framebuffer(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(Texture,
                        state.render_pass.multisampled_framebuffer.texture)
  WGPU_RELEASE_RESOURCE(TextureView,
                        state.render_pass.multisampled_framebuffer.view)

  /* Create the multi-sampled texture */
  WGPUTextureDescriptor multisampled_frame_desc = {
    .label         = STRVIEW("Multi-sampled - Texture"),
    .size          = (WGPUExtent3D){
      .width               = wgpu_context->width,
      .height              = wgpu_context->height,
      .depthOrArrayLayers  = 1,
     },
    .mipLevelCount = 1,
    .sampleCount   = state.sample_count,
    .dimension     = WGPUTextureDimension_2D,
    .format        = wgpu_context->render_format,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.render_pass.multisampled_framebuffer.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &multisampled_frame_desc);
  ASSERT(state.render_pass.multisampled_framebuffer.texture != NULL);

  /* Create the multi-sampled texture view */
  state.render_pass.multisampled_framebuffer.view
    = wgpuTextureCreateView(state.render_pass.multisampled_framebuffer.texture,
                            &(WGPUTextureViewDescriptor){
                              .label  = STRVIEW("Multi-sampled - Texture view"),
                              .format = wgpu_context->render_format,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                            });
  ASSERT(state.render_pass.multisampled_framebuffer.view != NULL);
}

static void init_orbit_camera_matrices(void)
{
  // Model matrix
  glm_mat4_identity(state.scene_data.model_matrix);
  glm_rotate(state.scene_data.model_matrix, glm_rad(-90.0f),
             (vec3){1.0f, 0.0f, 0.0f});
  glm_translate(state.scene_data.model_matrix,
                (vec3){
                  -state.plane_mesh.width / 2.0f,  /* center plane x */
                  -state.plane_mesh.height / 2.0f, /* center plane y */
                  0.0f,                            /* center plane z */
                });
}

static void update_uniform_buffers_scene(wgpu_context_t* wgpu_context)
{
  /* Elapsed time */
  state.scene_data.elapsed_time = stm_sec(stm_now());

  /* MVP */
  from_euler(state.current_mouse_position[1], state.current_mouse_position[0],
             0.0f, &state.tmp_mtx.rotation);
  create_orbit_view_matrix(15, state.tmp_mtx.rotation,
                           &state.tmp_mtx.view_matrix);

  /* View position */
  position_from_view_matrix(state.tmp_mtx.view_matrix,
                            &state.scene_data.view_position);

  /* Projection matrix */
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;
  glm_perspective(glm_rad(50.0f), aspect_ratio, 0.1f, 100.0f,
                  state.tmp_mtx.projection_matrix);

  /* View projection matrix */
  glm_mat4_mul(state.tmp_mtx.projection_matrix, state.tmp_mtx.view_matrix,
               state.scene_data.view_projection_matrix);

  /* Update uniform buffer */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.scene.buffer,
                       0, &state.scene_data, state.uniform_buffers.scene.size);
}

static void update_uniform_buffers_gerstner_waves(wgpu_context_t* wgpu_context)
{
  // Normalize wave directions
  const uint32_t wave_count
    = (uint32_t)ARRAY_SIZE(state.gerstner_wave_params.waves);
  if (!state.gerstner_waves_normalized) {
    for (uint32_t i = 0; i < wave_count; ++i) {
      glm_vec2_normalize(state.gerstner_wave_params.waves[i].direction);
    }
    state.gerstner_waves_normalized = true;
  }

  // Calculate sum of wave amplitudes
  for (uint32_t i = 0; i < wave_count; ++i) {
    state.gerstner_wave_params.amplitude_sum
      += state.gerstner_wave_params.waves[i].amplitude;
  }

  // Update uniform buffer
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.uniform_buffers.gerstner_wave_params.buffer, 0,
                       &state.gerstner_wave_params,
                       state.uniform_buffers.gerstner_wave_params.size);
}

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  /* Scene uniform buffer */
  state.uniform_buffers.scene = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Gerstner Waves - Scene uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.scene_data),
                  });

  /* Gerstner Waves parameters buffer */
  state.uniform_buffers.gerstner_wave_params = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Gerstner Waves - Parameters uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.gerstner_wave_params),
                  });

  /* Initialize uniform buffers */
  update_uniform_buffers_scene(wgpu_context);
  update_uniform_buffers_gerstner_waves(wgpu_context);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
      .logger.func  = slog_func,
    });
    init_plane_mesh();
    init_orbit_camera_matrices();
    init_vertex_and_index_buffers(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_texture(wgpu_context);
    init_texture_sampler(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_scene_bind_group(wgpu_context);
    init_texture_bind_group(wgpu_context);
    init_pipeline(wgpu_context);
    init_multisampled_framebuffer(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_multisampled_framebuffer(wgpu_context);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE
           && input_event->mouse_btn_pressed
           && input_event->mouse_button == BUTTON_LEFT) {
    state.current_mouse_position[0] = input_event->mouse_x;
    state.current_mouse_position[1] = input_event->mouse_y;
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Recreate texture when pixel data loaded */
  if (state.sea_color_texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.sea_color_texture);
    FREE_TEXTURE_PIXELS(state.sea_color_texture);
    /* Upddate the bindgroup */
    init_texture_bind_group(wgpu_context);
  }

  /* Update matrix data */
  update_uniform_buffers_scene(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* Set target frame buffer */
  if (state.sample_count == 1) {
    state.render_pass.color_attachment.view = wgpu_context->swapchain_view;
    state.render_pass.color_attachment.resolveTarget = NULL;
  }
  else {
    state.render_pass.color_attachment.view
      = state.render_pass.multisampled_framebuffer.view;
    state.render_pass.color_attachment.resolveTarget
      = wgpu_context->swapchain_view;
  }
  state.render_pass.depth_stencil_attachment.view
    = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass.descriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.render_pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.indices.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_groups.uniforms, 0,
                                    0);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 1, state.bind_groups.textures, 0,
                                    0);
  wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.indices.count, 1, 0, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  wgpu_destroy_texture(&state.sea_color_texture);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.scene.buffer)
  WGPU_RELEASE_RESOURCE(Buffer,
                        state.uniform_buffers.gerstner_wave_params.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.uniforms)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.textures)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.uniforms)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.textures)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline)
  WGPU_RELEASE_RESOURCE(Sampler, state.non_filtering_sampler)
  WGPU_RELEASE_RESOURCE(Texture,
                        state.render_pass.multisampled_framebuffer.texture)
  WGPU_RELEASE_RESOURCE(TextureView,
                        state.render_pass.multisampled_framebuffer.view)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Gerstner Waves",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
    .sample_count   = SAMPLE_COUNT,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* gerstner_waves_shader_wgsl = CODE(
  struct Uniforms {
    elapsedTime: f32,
    @align(16) modelMatrix: mat4x4<f32>,  // Explicitly set alignment
    viewProjectionMatrix: mat4x4<f32>,
    cameraPosition: vec3<f32>
  }

  struct GerstnerWaveParameters {
    length: f32,  // 0 < L
    amplitude: f32, // 0 < A
    steepness: f32,  // Steepness of the peak of the wave. 0 <= S <= 1
    @size(16) @align(8) direction: vec2<f32>  // Normalized direction of the wave
  }

  struct GerstnerWavesUniforms {
    waves: array<GerstnerWaveParameters, 5>,
    amplitudeSum: f32  // Sum of waves amplitudes
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) worldPosition: vec4<f32>
  }

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<uniform> wavesUniforms: GerstnerWavesUniforms;

  @group(1) @binding(0) var seaSampler: sampler;
  @group(1) @binding(1) var seaColor: texture_2d<f32>;

  const pi = 3.14159;
  const gravity = 9.8; // m/sec^2
  const waveNumbers = 5;

  @vertex
  fn vertex_main(
    @location(0) position: vec3<f32>,
    // @location(1) normal: vec3<f32>,  // TODO: delete normals from plane geo
    @location(2) uv: vec2<f32>,
  ) -> VertexOutput {
    var output: VertexOutput;
    var worldPosition: vec4<f32> = uniforms.modelMatrix * vec4<f32>(position, 1.0);

    var wavesSum: vec3<f32> = vec3<f32>(0.0);
    var wavesSumNormal: vec3<f32>;
    for(var i: i32 = 0; i < waveNumbers; i = i + 1) {
      var wave = wavesUniforms.waves[i];
      var wavevectorMagnitude = 2.0 * pi / wave.length;
      var wavevector = wave.direction * wavevectorMagnitude;
      var temporalFrequency = sqrt(gravity * wavevectorMagnitude);
      var steepnessFactor = wave.steepness / (wave.amplitude * wavevectorMagnitude * f32(waveNumbers));

      var pos = dot(wavevector, worldPosition.xz) - temporalFrequency * uniforms.elapsedTime;
      var sinPosAmplitudeDirection = sin(pos) * wave.amplitude * wave.direction;

      var offset: vec3<f32>;
      offset.x = sinPosAmplitudeDirection.x * steepnessFactor;
      offset.z = sinPosAmplitudeDirection.y * steepnessFactor;
      offset.y = cos(pos) * wave.amplitude;

      var normal: vec3<f32>;
      normal.x = sinPosAmplitudeDirection.x * wavevectorMagnitude;
      normal.z = sinPosAmplitudeDirection.y * wavevectorMagnitude;
      normal.y = cos(pos) * wave.amplitude * wavevectorMagnitude * steepnessFactor;

      wavesSum = wavesSum + offset;
      wavesSumNormal = wavesSumNormal + normal;
    }
    wavesSumNormal.y = 1.0 - wavesSumNormal.y;
    wavesSumNormal = normalize(wavesSumNormal);

    worldPosition.x = worldPosition.x - wavesSum.x;
    worldPosition.z = worldPosition.z - wavesSum.z;
    worldPosition.y = wavesSum.y;

    output.worldPosition = worldPosition;
    output.position = uniforms.viewProjectionMatrix * worldPosition;
    output.normal = vec4<f32>(wavesSumNormal, 0.0);
    output.uv = uv;
    return output;
  }

  @fragment
  fn fragment_main(
    data: VertexOutput,
  ) -> @location(0) vec4<f32> {
    const lightColor = vec3<f32>(1.0, 0.8, 0.65);
    const skyColor = vec3<f32>(0.69, 0.84, 1.0);

    const lightPosition = vec3<f32>(-10.0, 1.0, -10.0);
    var light = normalize(lightPosition - data.worldPosition.xyz);  // Vector from surface to light
    var eye = normalize(uniforms.cameraPosition - data.worldPosition.xyz);  // Vector from surface to camera
    var reflection = reflect(data.normal.xyz, -eye);  // I - 2.0 * dot(N, I) * N

    var halfway = normalize(eye + light);  // Vector between View and Light
    const shininess = 30.0;
    var specular = clamp(pow(dot(data.normal.xyz, halfway), shininess), 0.0, 1.0) * lightColor;  // Blinn-Phong specular component

    var fresnel = clamp(pow(1.0 + dot(-eye, data.normal.xyz), 4.0), 0.0, 1.0);  // Cheap fresnel approximation

    // Normalize height to [0, 1]
    var normalizedHeight = (data.worldPosition.y + wavesUniforms.amplitudeSum) / (2.0 * wavesUniforms.amplitudeSum);
    var underwater = textureSample(seaColor, seaSampler, vec2<f32>(normalizedHeight, 0.0)).rgb;

    // Approximating Translucency (GPU Pro 2 article)
    const distortion = 0.1;
    const power = 4.0;
    const scale = 1.0;
    const ambient = 0.2;
    var thickness = smoothstep(0.0, 1.0, normalizedHeight);
    var distortedLight = light + data.normal.xyz * distortion;
    var translucencyDot = pow(clamp(dot(eye, -distortedLight), 0.0, 1.0), power);
    var translucency = (translucencyDot * scale + ambient) * thickness;
    var underwaterTranslucency = mix(underwater, lightColor, translucency) * translucency;

    var color = mix(underwater + underwaterTranslucency, skyColor, fresnel) + specular;

    return vec4<f32>(color, 1.0);
  }
);
// clang-format on
