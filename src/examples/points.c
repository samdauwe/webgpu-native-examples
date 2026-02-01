#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Points
 *
 * This example shows how to render points of various sizes using a quad and
 * instancing.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/points
 * https://webgpufundamentals.org/webgpu/lessons/webgpu-points.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* distance_sized_points_vert_wgsl;
static const char* fixed_size_points_vert_wgsl;
static const char* orange_frag_wgsl;
static const char* textured_frag_wgsl;

/* -------------------------------------------------------------------------- *
 * Points example
 * -------------------------------------------------------------------------- */

#define MAX_VERTICES_COUNT (1000u)
#define MAX_VERTICES_LEN (MAX_VERTICES_COUNT * 3)
#define TEXTURE_SIZE (64u)
#define DEPTH_FORMAT WGPUTextureFormat_Depth24Plus

/* State struct */
static struct {
  struct {
    mat4 matrix;
    vec2 resolution;
    float size;
  } ubo_vs;
  struct {
    float fov;
    vec3 position;
    vec3 target;
    vec3 up_vector;
    mat4 view_matrix;
    mat4 projection_matrix;
    mat4 view_projection_matrix;
  } view_info;
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t uniform_buffer_vs;
  wgpu_texture_t texture;
  struct {
    wgpu_texture_t texture;
    uint32_t width;
    uint32_t height;
  } depth_texture;
  WGPUBindGroup uniform_bind_group;
  WGPUBindGroupLayout bind_group_layout;
  // Pipelines
  //  - [0][0] : distance sized - orange
  //  - [0][1] : distance sized - textured
  //  - [1][0] : fixed sized - orange
  //  - [1][1] : fixed sized - textured
  WGPURenderPipeline render_pipelines[2][2];
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  const uint32_t num_samples;
  struct {
    bool fixed_size;
    bool textured;
    float size;
  } settings;
  WGPUBool initialized;
  uint64_t last_imgui_frame_time;
} state = {
  .view_info = {
    .fov                    = (90.0f * PI) / 180.0f,
    .position               = {0.0f, 0.0f, 1.5f},
    .target                 = {0.0f, 0.0f, 0.0f},
    .up_vector              = {0.0f, 1.0f, 0.0f},
    .view_matrix            = GLM_MAT4_ZERO_INIT,
    .projection_matrix      = GLM_MAT4_ZERO_INIT,
    .view_projection_matrix = GLM_MAT4_ZERO_INIT,
  },
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
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .num_samples = 1000,
  .settings = {
    .fixed_size = false,
    .textured   = false,
    .size       = 10.0,
  }
};

/* See: https://www.google.com/search?q=fibonacci+sphere */
static void
create_fibonacci_sphere_vertices(uint32_t num_samples, float radius,
                                 float (*vertices)[MAX_VERTICES_LEN])
{
  num_samples           = MIN(num_samples, MAX_VERTICES_COUNT);
  const float increment = PI * (3.0f - sqrtf(5.0f));
  float offset = 0.0f, y = 0.0f, r = 0.0f, phi = 0.0f, x = 0.0f, z = 0.0f;
  for (uint32_t i = 0; i < num_samples; ++i) {
    offset = 2.0f / (float)num_samples;
    y      = (float)i * offset - 1.0f + offset / 2.0f;
    r      = sqrtf(1.0f - powf(y, 2.0f));
    phi    = (float)(i % num_samples) * increment;
    x      = cosf(phi) * r;
    z      = sinf(phi) * r;
    /* Set vertex */
    (*vertices)[i * 3]     = x * radius;
    (*vertices)[i * 3 + 1] = y * radius;
    (*vertices)[i * 3 + 2] = z * radius;
  }
}

static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  float vertex_data[MAX_VERTICES_LEN] = {0};
  create_fibonacci_sphere_vertices(state.num_samples, 1.0f, &vertex_data);

  state.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex buffer - Vertices",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = state.num_samples * 3 * sizeof(float),
                    .initial.data = vertex_data,
                    .count        = state.num_samples,
                  });
}

static void update_uniform_buffer_data(struct wgpu_context_t* wgpu_context)
{
  /* Get current timestamp */
  const float now = stm_sec(stm_now());

  /* Set the size in the uniform values */
  state.ubo_vs.size = state.settings.size;

  /* Set the matrix value */
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;
  glm_perspective(state.view_info.fov, aspect_ratio, 0.1f, 50.0f,
                  state.view_info.projection_matrix);
  glm_lookat(state.view_info.position,   /* eye vector    */
             state.view_info.target,     /* center vector */
             state.view_info.up_vector,  /* up vector     */
             state.view_info.view_matrix /* result matrix */
  );
  glm_mat4_mul(state.view_info.projection_matrix, state.view_info.view_matrix,
               state.view_info.view_projection_matrix);
  glm_rotate_y(state.view_info.view_projection_matrix, now,
               state.ubo_vs.matrix);
  glm_rotate_x(state.ubo_vs.matrix, now * 0.1f, state.ubo_vs.matrix);

  /* Set the resolution in the uniform values */
  state.ubo_vs.resolution[0] = (float)wgpu_context->width;
  state.ubo_vs.resolution[1] = (float)wgpu_context->height;
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  /* Update the uniform buffer data */
  update_uniform_buffer_data(wgpu_context);

  /* Copy the uniform values to the GPU */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_vs.buffer, 0,
                       &state.ubo_vs, state.uniform_buffer_vs.size);
}

static void init_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  // Create vertex shader uniform buffer block
  state.uniform_buffer_vs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex shader - Uniform buffer block",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(state.ubo_vs),
                  });

  /* Initialize GPU buffer */
  update_uniform_buffer_data(wgpu_context);
}

static void init_texture(wgpu_context_t* wgpu_context)
{
  /* Texture */
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Quad - Texture"),
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding
                     | WGPUTextureUsage_RenderAttachment,
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .dimension     = WGPUTextureDimension_2D,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .size          = (WGPUExtent3D) {
      .width               = TEXTURE_SIZE,
      .height              = TEXTURE_SIZE,
      .depthOrArrayLayers  = 1,
    },
  };
  state.texture.handle
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(state.texture.handle != NULL);

  /* Set texture data - create a simple white circle pattern
   * The TypeScript version uses an offscreen canvas with ctx.fillText('🦋')
   * We create a simple visible pattern for testing */
  uint8_t data[TEXTURE_SIZE * TEXTURE_SIZE * 4] = {0};
  const uint32_t data_size                      = sizeof(data);
  const float center_x                          = TEXTURE_SIZE / 2.0f;
  const float center_y                          = TEXTURE_SIZE / 2.0f;

  for (uint32_t y = 0; y < TEXTURE_SIZE; ++y) {
    for (uint32_t x = 0; x < TEXTURE_SIZE; ++x) {
      const uint32_t index = (y * TEXTURE_SIZE + x) * 4;

      /* Calculate distance from center */
      const float dx     = (float)x - center_x;
      const float dy     = (float)y - center_y;
      const float dist   = sqrtf(dx * dx + dy * dy);
      const float radius = TEXTURE_SIZE / 2.0f;

      /* Create a white circle with soft edges */
      if (dist < radius * 0.8f) {
        /* White color */
        data[index + 0] = 255; /* R */
        data[index + 1] = 255; /* G */
        data[index + 2] = 255; /* B */
        data[index + 3] = 255; /* A */
      }
      else {
        /* Transparent background */
        data[index + 0] = 0;
        data[index + 1] = 0;
        data[index + 2] = 0;
        data[index + 3] = 0;
      }
    }
  }
  wgpuQueueWriteTexture(wgpu_context->queue,
    &(WGPUTexelCopyTextureInfo) {
      .texture  = state.texture.handle,
      .mipLevel = 0,
      .origin = (WGPUOrigin3D) {
          .x = 0,
          .y = 0,
          .z = 0,
      },
      .aspect = WGPUTextureAspect_All,
    },
    data, data_size,
    &(WGPUTexelCopyBufferLayout){
      .offset       = 0,
      .bytesPerRow  = TEXTURE_SIZE * 4,
      .rowsPerImage = TEXTURE_SIZE,
    },
    &(WGPUExtent3D){
      .width              = TEXTURE_SIZE,
      .height             = TEXTURE_SIZE,
      .depthOrArrayLayers = 1,
    });

  /* Texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = STRVIEW("Quad - Texture view"),
    .format          = texture_desc.format,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = texture_desc.mipLevelCount,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  state.texture.view
    = wgpuTextureCreateView(state.texture.handle, &texture_view_dec);
  ASSERT(state.texture.view != NULL);

  /* Create a sampler with linear filtering for smooth interpolation */
  state.texture.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device,
    &(WGPUSamplerDescriptor){
      .label         = STRVIEW("Quad - Sampler with linear filtering"),
      .addressModeU  = WGPUAddressMode_ClampToEdge,
      .addressModeV  = WGPUAddressMode_ClampToEdge,
      .addressModeW  = WGPUAddressMode_ClampToEdge,
      .magFilter     = WGPUFilterMode_Linear,
      .minFilter     = WGPUFilterMode_Linear,
      .mipmapFilter  = WGPUMipmapFilterMode_Linear,
      .maxAnisotropy = 16,
    });
  ASSERT(state.texture.sampler != NULL);
}

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_texture(&state.depth_texture.texture);

  /* Set texture size */
  state.depth_texture.width  = wgpu_context->width;
  state.depth_texture.height = wgpu_context->height;

  /* Create the depth texture */
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Depth - Texture"),
    .size          = (WGPUExtent3D) {
       .width              = state.depth_texture.width,
       .height             = state.depth_texture.height,
       .depthOrArrayLayers = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = DEPTH_FORMAT,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  state.depth_texture.texture.handle
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(state.depth_texture.texture.handle != NULL);

  /* Create the depth texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  state.depth_texture.texture.view = wgpuTextureCreateView(
    state.depth_texture.texture.handle, &texture_view_dec);
  ASSERT(state.depth_texture.texture.view != NULL);
}

static void init_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(state.ubo_vs),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
        /* Texture view */
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
      },
      .storageTexture = {0},
    }
  };
  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Cube - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.bind_group_layout != NULL);
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Render - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = state.uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .sampler = state.texture.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = state.texture.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Uniform buffer - Bind group"),
    .layout     = state.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.uniform_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.uniform_bind_group != NULL);
}

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  const char* vertex_shaders[2] = {
    distance_sized_points_vert_wgsl, /* Distance sized fragment shader */
    fixed_size_points_vert_wgsl,     /* Fixed size vertex shader       */
  };

  const char* fragment_shaders[2] = {
    orange_frag_wgsl,   /* Orange fragment shader   */
    textured_frag_wgsl, /* Textured fragment shader */
  };

  /* Make pipelines for each combination */
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(vertex_shaders); ++i) {
    for (uint32_t j = 0; j < (uint32_t)ARRAY_SIZE(fragment_shaders); ++j) {
      WGPUShaderModule vert_shader_module
        = wgpu_create_shader_module(wgpu_context->device, vertex_shaders[i]);
      WGPUShaderModule frag_shader_module
        = wgpu_create_shader_module(wgpu_context->device, fragment_shaders[j]);

      /* Color blend state */
      WGPUBlendState blend_state = (WGPUBlendState){
        .color.operation = WGPUBlendOperation_Add,
        .color.srcFactor = WGPUBlendFactor_One,
        .color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
        .alpha.operation = WGPUBlendOperation_Add,
        .alpha.srcFactor = WGPUBlendFactor_One,
        .alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
      };

      /* Depth stencil state */
      WGPUDepthStencilState depth_stencil_state
        = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
          .format              = DEPTH_FORMAT,
          .depth_write_enabled = true,
        });
      depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

      /* Vertex buffer layout */
      WGPU_VERTEX_BUFFER_LAYOUT(
        instanced_point, 3 * 4 /* 3 floats, 4 bytes each */,
        /* Attribute location 0: Position */
        WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0))
      instanced_point_vertex_buffer_layout.stepMode
        = WGPUVertexStepMode_Instance;

      WGPURenderPipelineDescriptor rp_desc = {
        .label  = STRVIEW("Points - Render pipeline"),
        .layout = state.pipeline_layout,
        .vertex = {
          .module      = vert_shader_module,
          .entryPoint  = STRVIEW("vs"),
          .bufferCount = 1,
          .buffers     = &instanced_point_vertex_buffer_layout,
        },
        .fragment = &(WGPUFragmentState) {
          .entryPoint  = STRVIEW("fs"),
          .module      = frag_shader_module,
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
           .count = 1,
           .mask  = 0xffffffff
        },
      };

      state.render_pipelines[i][j]
        = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
      ASSERT(state.render_pipelines[i][j] != NULL);

      wgpuShaderModuleRelease(vert_shader_module);
      wgpuShaderModuleRelease(frag_shader_module);
    }
  }
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_vertex_buffer(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_texture(wgpu_context);
    init_depth_texture(wgpu_context);
    init_bind_group_layout(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_bind_group(wgpu_context);
    init_pipelines(wgpu_context);
    imgui_overlay_init(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

/* Render GUI */
static void render_gui(wgpu_context_t* wgpu_context)
{
  const uint64_t now = stm_now();
  const float dt_sec
    = (float)stm_sec(stm_diff(now, state.last_imgui_frame_time));
  state.last_imgui_frame_time = now;

  imgui_overlay_new_frame(wgpu_context, dt_sec);

  /* Set window position closer to upper left corner */
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);
  igCheckbox("fixedSize", &state.settings.fixed_size);
  igCheckbox("textured", &state.settings.textured);
  imgui_overlay_slider_float("size", &state.settings.size, 0.0f, 80.0f, "%.1f");
  igEnd();
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update uniform data */
  update_uniform_buffers(wgpu_context);

  /* Render GUI */
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* If we don't have a depth texture OR if its size is different from the
   * canvasTexture when make a new depth texture */
  if (!state.depth_texture.texture.handle
      || state.depth_texture.width != (uint32_t)wgpu_context->width
      || state.depth_texture.height != (uint32_t)wgpu_context->height) {
    init_depth_texture(wgpu_context);
  }
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth_texture.texture.view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Bind the rendering pipeline */
  const WGPURenderPipeline pipeline
    = state.render_pipelines[state.settings.fixed_size ? 1 : 0]
                            [state.settings.textured ? 1 : 0];

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertex_buffer.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.uniform_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderDraw(rpass_enc, 6, state.vertex_buffer.count, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Render imgui overlay */
  imgui_overlay_render(wgpu_context);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  wgpu_destroy_buffer(&state.vertex_buffer);
  wgpu_destroy_buffer(&state.uniform_buffer_vs);
  wgpu_destroy_texture(&state.texture);
  wgpu_destroy_texture(&state.depth_texture.texture);
  WGPU_RELEASE_RESOURCE(BindGroup, state.uniform_bind_group)
  for (uint32_t i = 0; i < 2; ++i) {
    for (uint32_t j = 0; j < 2; ++j) {
      WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipelines[i][j])
    }
  }
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  imgui_overlay_shutdown();
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Points",
    .init_cb        = init,
    .frame_cb       = frame,
    .input_event_cb = input_event_cb,
    .shutdown_cb    = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* distance_sized_points_vert_wgsl = CODE(
  struct Vertex {
    @location(0) position: vec4f,
  };

  struct Uniforms {
    matrix: mat4x4f,
    resolution: vec2f,
    size: f32,
  };

  struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
  };

  @group(0) @binding(0) var<uniform> uni: Uniforms;

  @vertex fn vs(
      vert: Vertex,
      @builtin(vertex_index) vNdx: u32,
  ) -> VSOutput {
    let points = array(
      vec2f(-1, -1),
      vec2f( 1, -1),
      vec2f(-1,  1),
      vec2f(-1,  1),
      vec2f( 1, -1),
      vec2f( 1,  1),
    );
    var vsOut: VSOutput;
    let pos = points[vNdx];
    let clipPos = uni.matrix * vert.position;
    let pointPos = vec4f(pos * uni.size / uni.resolution * clipPos.w, 0, 0);
    vsOut.position = clipPos + pointPos;
    vsOut.texcoord = pos * 0.5 + 0.5;
    return vsOut;
  }
);

static const char* fixed_size_points_vert_wgsl = CODE(
  struct Vertex {
    @location(0) position: vec4f,
  };

  struct Uniforms {
    matrix: mat4x4f,
    resolution: vec2f,
    size: f32,
  };

  struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
  };

  @group(0) @binding(0) var<uniform> uni: Uniforms;

  @vertex fn vs(
      vert: Vertex,
      @builtin(vertex_index) vNdx: u32,
  ) -> VSOutput {
    let points = array(
      vec2f(-1, -1),
      vec2f( 1, -1),
      vec2f(-1,  1),
      vec2f(-1,  1),
      vec2f( 1, -1),
      vec2f( 1,  1),
    );
    var vsOut: VSOutput;
    let pos = points[vNdx];
    let clipPos = uni.matrix * vert.position;
    let pointPos = vec4f(pos * uni.size / uni.resolution, 0, 0);
    vsOut.position = clipPos + pointPos;
    vsOut.texcoord = pos * 0.5 + 0.5;
    return vsOut;
  }
);

static const char* orange_frag_wgsl = CODE(
  @fragment fn fs() -> @location(0) vec4f {
    return vec4f(1, 0.5, 0.2, 1);
  }
);

static const char* textured_frag_wgsl = CODE(
  struct VSOutput {
    @location(0) texcoord: vec2f,
  };

  @group(0) @binding(1) var s: sampler;
  @group(0) @binding(2) var t: texture_2d<f32>;

  @fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
    let color = textureSample(t, s, vsOut.texcoord);
    if (color.a < 0.1) {
      discard;
    }
    return color;
  }
);
// clang-format on
