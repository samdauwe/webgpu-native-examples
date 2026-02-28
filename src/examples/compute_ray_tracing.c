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
 * WebGPU Example - Compute Shader Ray Tracing
 *
 * Simple GPU ray tracer with shadows and reflections using a compute shader. No
 * scene geometry is rendered in the graphics pass.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/computeraytracing
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* raytracing_compute_shader_wgsl;
static const char* fullscreen_quad_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Compute Shader Ray Tracing example
 * -------------------------------------------------------------------------- */

#define TEX_DIM 1024u

/* SSBO sphere declaration */
typedef struct sphere_t {
  vec3 pos;
  float radius;
  vec3 diffuse;
  float specular;
  uint32_t id;
  int32_t _pad[3];
} sphere_t;

/* SSBO plane declaration */
typedef struct plane_t {
  vec3 normal;
  float distance;
  vec3 diffuse;
  float specular;
  uint32_t id;
  int32_t _pad[3];
} plane_t;

/* State struct */
static struct {
  /* Storage texture for ray traced output */
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
    WGPUSampler sampler;
  } storage_texture;
  /* Graphics pipeline (fullscreen quad display) */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
  } graphics;
  /* Compute pipeline (ray tracer) */
  struct {
    wgpu_buffer_t spheres_buffer;
    wgpu_buffer_t planes_buffer;
    wgpu_buffer_t uniform_buffer;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
    struct {
      vec3 lightPos;
      float aspectRatio;
      vec4 fogColor;
      struct {
        vec3 pos;
        float _pad;
        vec3 lookat;
        float fov;
      } camera;
    } ubo;
  } compute;
  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* Animation */
  float timer;
  bool paused;
  /* GUI timing */
  uint64_t last_frame_time;
  float frame_timer;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.025, 0.025, 0.025, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  .paused = false,
};

/* -- Storage texture ------------------------------------------------------- */

static void init_storage_texture(wgpu_context_t* wgpu_context)
{
  state.storage_texture.handle = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label         = STRVIEW("Ray trace target texture"),
      .usage         = WGPUTextureUsage_TextureBinding
                       | WGPUTextureUsage_StorageBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = (WGPUExtent3D){
        .width              = TEX_DIM,
        .height             = TEX_DIM,
        .depthOrArrayLayers = 1,
      },
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });
  ASSERT(state.storage_texture.handle != NULL);

  state.storage_texture.view
    = wgpuTextureCreateView(state.storage_texture.handle,
                            &(WGPUTextureViewDescriptor){
                              .label = STRVIEW("Ray trace target texture view"),
                              .format          = WGPUTextureFormat_RGBA8Unorm,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                            });
  ASSERT(state.storage_texture.view != NULL);

  state.storage_texture.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label        = STRVIEW("Ray trace target sampler"),
                            .addressModeU = WGPUAddressMode_ClampToEdge,
                            .addressModeV = WGPUAddressMode_ClampToEdge,
                            .addressModeW = WGPUAddressMode_ClampToEdge,
                            .minFilter    = WGPUFilterMode_Linear,
                            .magFilter    = WGPUFilterMode_Linear,
                            .mipmapFilter = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp  = 0.0f,
                            .lodMaxClamp  = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.storage_texture.sampler != NULL);
}

/* -- Storage buffers (scene geometry) -------------------------------------- */

static uint32_t current_id = 0;

static void init_sphere(sphere_t* sphere, vec3 pos, float radius, vec3 diffuse,
                        float specular)
{
  pos[1] *= -1.0f; /* Flip y */

  sphere->id = current_id++;
  glm_vec3_copy(pos, sphere->pos);
  sphere->radius = radius;
  glm_vec3_copy(diffuse, sphere->diffuse);
  sphere->specular = specular;
  memset(sphere->_pad, 0, sizeof(sphere->_pad));
}

static void init_plane(plane_t* plane, vec3 normal, float distance,
                       vec3 diffuse, float specular)
{
  plane->id = current_id++;
  glm_vec3_copy(normal, plane->normal);
  plane->distance = distance;
  glm_vec3_copy(diffuse, plane->diffuse);
  plane->specular = specular;
  memset(plane->_pad, 0, sizeof(plane->_pad));
}

static void init_storage_buffers(wgpu_context_t* wgpu_context)
{
  /* Spheres */
  static sphere_t spheres[3] = {0};
  init_sphere(&spheres[0], (vec3){1.75f, -0.5f, 0.0f}, 1.0f,
              (vec3){0.0f, 1.0f, 0.0f}, 32.0f);
  init_sphere(&spheres[1], (vec3){0.0f, 1.0f, -0.5f}, 1.0f,
              (vec3){0.65f, 0.77f, 0.97f}, 32.0f);
  init_sphere(&spheres[2], (vec3){-1.75f, -0.75f, -0.5f}, 1.25f,
              (vec3){0.9f, 0.76f, 0.46f}, 32.0f);

  state.compute.spheres_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Spheres storage buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
                    .size  = sizeof(spheres),
                    .initial.data = spheres,
                  });

  /* Planes */
  static plane_t planes[6] = {0};
  const float room_dim     = 4.0f;
  init_plane(&planes[0], (vec3){0.0f, 1.0f, 0.0f}, room_dim,
             (vec3){1.0f, 1.0f, 1.0f}, 32.0f);
  init_plane(&planes[1], (vec3){0.0f, -1.0f, 0.0f}, room_dim,
             (vec3){1.0f, 1.0f, 1.0f}, 32.0f);
  init_plane(&planes[2], (vec3){0.0f, 0.0f, 1.0f}, room_dim,
             (vec3){1.0f, 1.0f, 1.0f}, 32.0f);
  init_plane(&planes[3], (vec3){0.0f, 0.0f, -1.0f}, room_dim,
             (vec3){0.0f, 0.0f, 0.0f}, 32.0f);
  init_plane(&planes[4], (vec3){-1.0f, 0.0f, 0.0f}, room_dim,
             (vec3){1.0f, 0.0f, 0.0f}, 32.0f);
  init_plane(&planes[5], (vec3){1.0f, 0.0f, 0.0f}, room_dim,
             (vec3){0.0f, 1.0f, 0.0f}, 32.0f);

  state.compute.planes_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Planes storage buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
                    .size  = sizeof(planes),
                    .initial.data = planes,
                  });
}

/* -- Uniform buffer -------------------------------------------------------- */

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Default values */
  glm_vec4_zero(state.compute.ubo.fogColor);
  glm_vec3_copy((vec3){0.0f, 0.0f, 4.0f}, state.compute.ubo.camera.pos);
  state.compute.ubo.camera._pad = 0.0f;
  glm_vec3_copy((vec3){0.0f, 0.5f, 0.0f}, state.compute.ubo.camera.lookat);
  state.compute.ubo.camera.fov = 10.0f;
  state.compute.ubo.aspectRatio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  state.compute.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.compute.ubo),
                  });
}

static void update_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.compute.ubo.lightPos[0] = 0.0f
                                  + sinf(glm_rad(state.timer * 360.0f))
                                      * cosf(glm_rad(state.timer * 360.0f))
                                      * 2.0f;
  state.compute.ubo.lightPos[1]
    = 0.0f + sinf(glm_rad(state.timer * 360.0f)) * 2.0f;
  state.compute.ubo.lightPos[2]
    = 0.0f + cosf(glm_rad(state.timer * 360.0f)) * 2.0f;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.compute.uniform_buffer.buffer,
                       0, &state.compute.ubo, sizeof(state.compute.ubo));
}

/* -- Graphics pipeline (fullscreen quad display) --------------------------- */

static void init_graphics_pipeline(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };

  state.graphics.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Graphics bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.graphics.bind_group_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.storage_texture.view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.storage_texture.sampler,
    },
  };

  state.graphics.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Graphics bind group"),
                            .layout     = state.graphics.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.graphics.bind_group != NULL);

  /* Pipeline layout */
  state.graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Graphics pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.graphics.bind_group_layout,
    });
  ASSERT(state.graphics.pipeline_layout != NULL);

  /* Shader module */
  WGPUShaderModule shader_module = wgpu_create_shader_module(
    wgpu_context->device, fullscreen_quad_shader_wgsl);

  /* Blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  /* Pipeline */
  state.graphics.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Fullscreen quad render pipeline"),
      .layout = state.graphics.pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 0,
      },
      .fragment = &(WGPUFragmentState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff,
      },
    });
  ASSERT(state.graphics.pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);
}

/* -- Compute pipeline (ray tracer) ----------------------------------------- */

static void init_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .storageTexture = (WGPUStorageTextureBindingLayout){
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA8Unorm,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.compute.ubo),
      },
    },
    [2] = (WGPUBindGroupLayoutEntry){
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = state.compute.spheres_buffer.size,
      },
    },
    [3] = (WGPUBindGroupLayoutEntry){
      .binding    = 3,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = state.compute.planes_buffer.size,
      },
    },
  };

  state.compute.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Compute bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.compute.bind_group_layout != NULL);

  /* Pipeline layout */
  state.compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Compute pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.compute.bind_group_layout,
    });
  ASSERT(state.compute.pipeline_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.storage_texture.view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = state.compute.uniform_buffer.buffer,
      .offset  = 0,
      .size    = state.compute.uniform_buffer.size,
    },
    [2] = (WGPUBindGroupEntry){
      .binding = 2,
      .buffer  = state.compute.spheres_buffer.buffer,
      .offset  = 0,
      .size    = state.compute.spheres_buffer.size,
    },
    [3] = (WGPUBindGroupEntry){
      .binding = 3,
      .buffer  = state.compute.planes_buffer.buffer,
      .offset  = 0,
      .size    = state.compute.planes_buffer.size,
    },
  };

  state.compute.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Compute bind group"),
                            .layout     = state.compute.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.compute.bind_group != NULL);

  /* Compute shader module */
  WGPUShaderModule comp_shader_module = wgpu_create_shader_module(
    wgpu_context->device, raytracing_compute_shader_wgsl);

  /* Compute pipeline */
  state.compute.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Raytracing compute pipeline"),
      .layout  = state.compute.pipeline_layout,
      .compute = {
        .module     = comp_shader_module,
        .entryPoint = STRVIEW("main"),
      },
    });
  ASSERT(state.compute.pipeline != NULL);

  wgpuShaderModuleRelease(comp_shader_module);
}

/* -- GUI ------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){200.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);
  igCheckbox("Paused", &state.paused);
  igEnd();
}

/* -- Main callbacks -------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_storage_texture(wgpu_context);
    init_storage_buffers(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_graphics_pipeline(wgpu_context);
    init_compute_pipeline(wgpu_context);
    imgui_overlay_init(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Calculate delta time */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;
  state.frame_timer     = delta_time;

  /* Update timer (slower rotation for the light) */
  if (!state.paused) {
    state.timer += delta_time * 0.25f * 0.04f;
    if (state.timer > 1.0f) {
      state.timer -= 1.0f;
    }
    update_uniform_buffer(wgpu_context);
  }

  /* ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass: ray trace the scene */
  {
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass_enc, state.compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(cpass_enc, 0, state.compute.bind_group,
                                       0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cpass_enc, TEX_DIM / 16,
                                             TEX_DIM / 16, 1);
    wgpuComputePassEncoderEnd(cpass_enc);
    wgpuComputePassEncoderRelease(cpass_enc);
  }

  /* Render pass: display ray traced image as fullscreen quad */
  {
    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.graphics.bind_group,
                                      0, 0);
    wgpuRenderPassEncoderDraw(rpass_enc, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    wgpuRenderPassEncoderRelease(rpass_enc);
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Render imgui overlay */
  imgui_overlay_render(wgpu_context);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  imgui_overlay_shutdown();

  /* Storage texture */
  WGPU_RELEASE_RESOURCE(TextureView, state.storage_texture.view)
  WGPU_RELEASE_RESOURCE(Sampler, state.storage_texture.sampler)
  WGPU_RELEASE_RESOURCE(Texture, state.storage_texture.handle)

  /* Graphics pipeline */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)

  /* Compute pipeline */
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.spheres_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.planes_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipeline)
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Compute Shader Ray Tracing",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* fullscreen_quad_shader_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
  };

  @vertex
  fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    /* Generate fullscreen triangle */
    let uv = vec2<f32>(
      f32((vertex_index << 1u) & 2u),
      f32(vertex_index & 2u)
    );
    var output : VertexOutput;
    output.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    return output;
  }

  @group(0) @binding(0) var color_texture : texture_2d<f32>;
  @group(0) @binding(1) var color_sampler : sampler;

  @fragment
  fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(color_texture, color_sampler,
                         vec2<f32>(uv.x, 1.0 - uv.y));
  }
);

static const char* raytracing_compute_shader_wgsl = CODE(
  /* Ray tracing compute shader - based on Inigo Quilez's work */

  const EPSILON : f32 = 0.0001;
  const MAXLEN : f32 = 1000.0;
  const SHADOW_FACTOR : f32 = 0.5;
  const RAYBOUNCES : i32 = 2;
  const REFLECTIONSTRENGTH : f32 = 0.4;
  const REFLECTIONFALLOFF : f32 = 0.5;

  struct Camera {
    pos : vec3<f32>,
    _pad : f32,
    lookat : vec3<f32>,
    fov : f32,
  };

  struct UBO {
    lightPos : vec3<f32>,
    aspectRatio : f32,
    fogColor : vec4<f32>,
    camera : Camera,
  };

  struct Sphere {
    pos : vec3<f32>,
    radius : f32,
    diffuse : vec3<f32>,
    specular : f32,
    id : i32,
    _pad1 : i32,
    _pad2 : i32,
    _pad3 : i32,
  };

  struct Plane {
    normal : vec3<f32>,
    distance : f32,
    diffuse : vec3<f32>,
    specular : f32,
    id : i32,
    _pad1 : i32,
    _pad2 : i32,
    _pad3 : i32,
  };

  @group(0) @binding(0) var result_image : texture_storage_2d<rgba8unorm, write>;
  @group(0) @binding(1) var<uniform> ubo : UBO;
  @group(0) @binding(2) var<storage, read> spheres : array<Sphere>;
  @group(0) @binding(3) var<storage, read> planes : array<Plane>;

  /* Lighting */

  fn lightDiffuse(normal : vec3<f32>, lightDir : vec3<f32>) -> f32 {
    return clamp(dot(normal, lightDir), 0.1, 1.0);
  }

  fn lightSpecular(normal : vec3<f32>, lightDir : vec3<f32>,
                   specularFactor : f32) -> f32 {
    let viewVec = normalize(ubo.camera.pos);
    let halfVec = normalize(lightDir + viewVec);
    return pow(clamp(dot(normal, halfVec), 0.0, 1.0), specularFactor);
  }

  /* Sphere intersection */

  fn sphereIntersect(rayO : vec3<f32>, rayD : vec3<f32>,
                     sphere : Sphere) -> f32 {
    let oc = rayO - sphere.pos;
    let b = 2.0 * dot(oc, rayD);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let h = b * b - 4.0 * c;
    if (h < 0.0) {
      return -1.0;
    }
    let t = (-b - sqrt(h)) / 2.0;
    return t;
  }

  fn sphereNormal(pos : vec3<f32>, sphere : Sphere) -> vec3<f32> {
    return (pos - sphere.pos) / sphere.radius;
  }

  /* Plane intersection */

  fn planeIntersect(rayO : vec3<f32>, rayD : vec3<f32>,
                    plane : Plane) -> f32 {
    let d = dot(rayD, plane.normal);
    if (d == 0.0) {
      return 0.0;
    }
    let t = -(plane.distance + dot(rayO, plane.normal)) / d;
    if (t < 0.0) {
      return 0.0;
    }
    return t;
  }

  /* Scene intersection */

  fn intersect(rayO : vec3<f32>, rayD : vec3<f32>,
               resT_in : f32) -> vec2<f32> {
    /* Returns vec2(id, t) */
    var id : f32 = -1.0;
    var resT : f32 = resT_in;

    let sphere_count = arrayLength(&spheres);
    for (var i : u32 = 0u; i < sphere_count; i = i + 1u) {
      let tSphere = sphereIntersect(rayO, rayD, spheres[i]);
      if (tSphere > EPSILON && tSphere < resT) {
        id = f32(spheres[i].id);
        resT = tSphere;
      }
    }

    let plane_count = arrayLength(&planes);
    for (var i : u32 = 0u; i < plane_count; i = i + 1u) {
      let tPlane = planeIntersect(rayO, rayD, planes[i]);
      if (tPlane > EPSILON && tPlane < resT) {
        id = f32(planes[i].id);
        resT = tPlane;
      }
    }

    return vec2<f32>(id, resT);
  }

  /* Shadow calculation */

  fn calcShadow(rayO : vec3<f32>, rayD : vec3<f32>,
                objectId : i32, t_in : f32) -> vec2<f32> {
    /* Returns vec2(shadow_factor, t) */
    var t : f32 = t_in;
    let sphere_count = arrayLength(&spheres);
    for (var i : u32 = 0u; i < sphere_count; i = i + 1u) {
      if (spheres[i].id == objectId) {
        continue;
      }
      let tSphere = sphereIntersect(rayO, rayD, spheres[i]);
      if (tSphere > EPSILON && tSphere < t) {
        t = tSphere;
        return vec2<f32>(SHADOW_FACTOR, t);
      }
    }
    return vec2<f32>(1.0, t);
  }

  /* Fog */

  fn fog(t : f32, color : vec3<f32>) -> vec3<f32> {
    return mix(color, ubo.fogColor.rgb,
               clamp(sqrt(t * t) / 20.0, 0.0, 1.0));
  }

  /* Render scene - returns vec4(color.rgb, id) and modifies ray via out params */

  struct RenderResult {
    color : vec3<f32>,
    rayO : vec3<f32>,
    rayD : vec3<f32>,
    id : i32,
  };

  fn renderScene(rayO_in : vec3<f32>, rayD_in : vec3<f32>,
                 id_in : i32) -> RenderResult {
    var result : RenderResult;
    result.rayO = rayO_in;
    result.rayD = rayD_in;
    result.id = id_in;
    result.color = vec3<f32>(0.0, 0.0, 0.0);

    var t : f32 = MAXLEN;

    /* Get intersected object ID */
    let hit = intersect(rayO_in, rayD_in, t);
    let objectID = i32(hit.x);
    t = hit.y;

    if (objectID == -1) {
      return result;
    }

    let pos = rayO_in + t * rayD_in;
    let lightVec = normalize(ubo.lightPos - pos);
    var normal : vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
    var color : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    /* Check planes */
    let plane_count = arrayLength(&planes);
    for (var i : u32 = 0u; i < plane_count; i = i + 1u) {
      if (objectID == planes[i].id) {
        normal = planes[i].normal;
        let diffuse = lightDiffuse(normal, lightVec);
        let specular = lightSpecular(normal, lightVec, planes[i].specular);
        color = diffuse * planes[i].diffuse + specular;
      }
    }

    /* Check spheres */
    let sphere_count = arrayLength(&spheres);
    for (var i : u32 = 0u; i < sphere_count; i = i + 1u) {
      if (objectID == spheres[i].id) {
        normal = sphereNormal(pos, spheres[i]);
        let diffuse = lightDiffuse(normal, lightVec);
        let specular = lightSpecular(normal, lightVec, spheres[i].specular);
        color = diffuse * spheres[i].diffuse + specular;
      }
    }

    if (id_in == -1) {
      result.color = color;
      return result;
    }

    result.id = objectID;

    /* Shadows */
    var shadow_t : f32 = length(ubo.lightPos - pos);
    let shadow_result = calcShadow(pos, lightVec, result.id, shadow_t);
    color = color * shadow_result.x;
    shadow_t = shadow_result.y;

    /* Fog */
    color = fog(shadow_t, color);

    /* Reflect ray for next render pass */
    result.rayD = result.rayD + 2.0 * -dot(normal, result.rayD) * normal;
    result.rayO = pos;
    result.color = color;

    return result;
  }

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let dim = vec2<f32>(textureDimensions(result_image));
    let uv = vec2<f32>(global_id.xy) / dim;

    var rayO : vec3<f32> = ubo.camera.pos;
    var rayD : vec3<f32> = normalize(vec3<f32>(
      (-1.0 + 2.0 * uv) * vec2<f32>(ubo.aspectRatio, 1.0), -1.0));

    /* Basic color path */
    var id : i32 = 0;
    var r = renderScene(rayO, rayD, id);
    var finalColor = r.color;
    rayO = r.rayO;
    rayD = r.rayD;
    id = r.id;

    /* Reflections */
    var reflectionStrength : f32 = REFLECTIONSTRENGTH;
    for (var i : i32 = 0; i < RAYBOUNCES; i = i + 1) {
      r = renderScene(rayO, rayD, id);
      finalColor = (1.0 - reflectionStrength) * finalColor
                   + reflectionStrength
                     * mix(r.color, finalColor, 1.0 - reflectionStrength);
      reflectionStrength = reflectionStrength * REFLECTIONFALLOFF;
      rayO = r.rayO;
      rayD = r.rayD;
      id = r.id;
    }

    textureStore(result_image, vec2<i32>(global_id.xy),
                 vec4<f32>(finalColor, 0.0));
  }
);
// clang-format on
