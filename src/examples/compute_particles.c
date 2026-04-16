/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader Particle System
 *
 * Attraction based 2D GPU particle system using compute shaders. Particle data
 * is stored in a shader storage buffer. A compute shader updates particle
 * positions based on attraction/repulsion forces, and the particles are
 * rendered as instanced billboard quads with gradient coloring sampled from a
 * color ramp texture and a particle sprite texture.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/computeparticles
 * -------------------------------------------------------------------------- */

#include "core/image_loader.h"
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>
#include <string.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

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
 * WGSL Shaders (forward declarations)
 * -------------------------------------------------------------------------- */

static const char* particle_vertex_shader_wgsl;
static const char* particle_fragment_shader_wgsl;
static const char* particle_compute_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define PARTICLE_COUNT (256u * 1024u)

/* File buffer size for async texture loading */
#define TEXTURE_FILE_BUFFER_SIZE (256 * 1024)

/* -------------------------------------------------------------------------- *
 * Types
 * -------------------------------------------------------------------------- */

/* SSBO particle layout (matches Vulkan: vec2 pos, vec2 vel, vec4 gradientPos)
 * Total: 32 bytes per particle */
typedef struct particle_t {
  float pos[2];          /* Particle position */
  float vel[2];          /* Particle velocity */
  float gradient_pos[4]; /* Texture coord for gradient ramp map */
} particle_t;

/* Graphics uniform: screen dimensions for billboard sizing */
typedef struct graphics_ubo_t {
  float screen_width;
  float screen_height;
  float point_size; /* In pixels, matching Vulkan gl_PointSize */
  float _pad;       /* Padding to 16 bytes */
} graphics_ubo_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Textures */
  struct {
    wgpu_texture_t particle; /* Point sprite texture (particle01_rgba.png) */
    wgpu_texture_t
      gradient; /* Color gradient ramp (particle_gradient_rgba.png) */
    uint8_t particle_buf[TEXTURE_FILE_BUFFER_SIZE];
    uint8_t gradient_buf[TEXTURE_FILE_BUFFER_SIZE];
    bool particle_loaded;
    bool gradient_loaded;
  } textures;
  /* Graphics resources */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
    wgpu_buffer_t uniform_buffer;
    graphics_ubo_t ubo;
  } graphics;
  /* Compute resources */
  struct {
    wgpu_buffer_t storage_buffer;
    wgpu_buffer_t uniform_buffer;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
    struct {
      float delta_t;
      float dest_x;
      float dest_y;
      int32_t particle_count;
    } ubo;
  } compute;
  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* Animation */
  float timer;
  float anim_start;
  /* Settings */
  struct {
    bool attach_to_cursor;
  } settings;
  /* Mouse state (normalized -1..1) */
  float cursor_x;
  float cursor_y;
  /* Timing */
  uint64_t last_frame_time;
  float frame_timer;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  .anim_start = 20.0f,
  .settings = {
    .attach_to_cursor = false,
  },
};

/* -------------------------------------------------------------------------- *
 * Async texture loading
 * -------------------------------------------------------------------------- */

static void particle_texture_fetch_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("ERROR: Failed to fetch particle texture\n");
    return;
  }

  image_t img = {0};
  if (image_load_from_memory(response->data.ptr, (int)response->data.size, 4,
                             &img)) {
    state.textures.particle.desc = (wgpu_texture_desc_t){
      .extent       = {.width              = (uint32_t)img.width,
                       .height             = (uint32_t)img.height,
                       .depthOrArrayLayers = 1},
      .format       = WGPUTextureFormat_RGBA8Unorm,
      .address_mode = WGPUAddressMode_ClampToEdge,
      .pixels
      = {.ptr = img.pixels.u8, .size = (size_t)(img.width * img.height * 4)},
    };
    state.textures.particle.desc.is_dirty = true;
    state.textures.particle_loaded        = true;
  }
}

static void gradient_texture_fetch_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("ERROR: Failed to fetch gradient texture\n");
    return;
  }

  image_t img = {0};
  if (image_load_from_memory(response->data.ptr, (int)response->data.size, 4,
                             &img)) {
    state.textures.gradient.desc = (wgpu_texture_desc_t){
      .extent       = {.width              = (uint32_t)img.width,
                       .height             = (uint32_t)img.height,
                       .depthOrArrayLayers = 1},
      .format       = WGPUTextureFormat_RGBA8Unorm,
      .address_mode = WGPUAddressMode_ClampToEdge,
      .pixels
      = {.ptr = img.pixels.u8, .size = (size_t)(img.width * img.height * 4)},
    };
    state.textures.gradient.desc.is_dirty = true;
    state.textures.gradient_loaded        = true;
  }
}

static void init_textures(wgpu_context_t* wgpu_context)
{
  /* Create placeholder textures */
  state.textures.particle = wgpu_create_color_bars_texture(wgpu_context, NULL);
  state.textures.gradient = wgpu_create_color_bars_texture(wgpu_context, NULL);

  /* Kick off async loads */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/particle01_rgba.png",
    .callback = particle_texture_fetch_cb,
    .buffer   = SFETCH_RANGE(state.textures.particle_buf),
  });

  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/particle_gradient_rgba.png",
    .callback = gradient_texture_fetch_cb,
    .buffer   = SFETCH_RANGE(state.textures.gradient_buf),
  });
}

/* -------------------------------------------------------------------------- *
 * Storage buffer (particle SSBO)
 * -------------------------------------------------------------------------- */

static void init_storage_buffer(wgpu_context_t* wgpu_context)
{
  /* Match Vulkan: pos = random [-1,1], vel = 0, gradientPos.x = pos.x/2 */
  static particle_t particle_buffer[PARTICLE_COUNT];
  for (uint32_t i = 0; i < PARTICLE_COUNT; ++i) {
    particle_buffer[i] = (particle_t){
      .pos = {
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
      },
      .vel = {0.0f, 0.0f},
      .gradient_pos = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    particle_buffer[i].gradient_pos[0] = particle_buffer[i].pos[0] / 2.0f;
  }

  state.compute.storage_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Particle - Storage buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size         = PARTICLE_COUNT * sizeof(particle_t),
                    .initial.data = particle_buffer,
                  });
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Compute UBO */
  state.compute.ubo.particle_count = PARTICLE_COUNT;

  state.compute.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.compute.ubo),
                  });

  /* Graphics UBO (screen dimensions for billboard sizing) */
  state.graphics.ubo = (graphics_ubo_t){
    .screen_width  = (float)wgpu_context->width,
    .screen_height = (float)wgpu_context->height,
    .point_size    = 8.0f, /* Match Vulkan gl_PointSize */
  };

  state.graphics.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Graphics uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(graphics_ubo_t),
                    .initial.data = &state.graphics.ubo,
                  });
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Compute: match Vulkan — deltaT = frameTimer * 2.5 */
  state.compute.ubo.delta_t = state.frame_timer * 2.5f;

  if (state.settings.attach_to_cursor) {
    state.compute.ubo.dest_x = state.cursor_x;
    state.compute.ubo.dest_y = state.cursor_y;
  }
  else {
    state.compute.ubo.dest_x = sinf(state.timer * PI2) * 0.75f;
    state.compute.ubo.dest_y = 0.0f;
  }

  wgpuQueueWriteBuffer(wgpu_context->queue, state.compute.uniform_buffer.buffer,
                       0, &state.compute.ubo, sizeof(state.compute.ubo));

  /* Graphics: update screen dimensions for correct billboard sizing */
  state.graphics.ubo.screen_width  = (float)wgpu_context->width;
  state.graphics.ubo.screen_height = (float)wgpu_context->height;

  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.graphics.uniform_buffer.buffer, 0,
                       &state.graphics.ubo, sizeof(state.graphics.ubo));
}

/* -------------------------------------------------------------------------- *
 * Graphics pipeline
 * -------------------------------------------------------------------------- */

static void init_graphics_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
    [2] = (WGPUBindGroupLayoutEntry){
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
    [3] = (WGPUBindGroupLayoutEntry){
      .binding    = 3,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(graphics_ubo_t),
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
}

static void rebuild_graphics_bind_group(wgpu_context_t* wgpu_context)
{
  /* Release previous bind group if any */
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group)

  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.textures.particle.view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.textures.particle.sampler,
    },
    [2] = (WGPUBindGroupEntry){
      .binding     = 2,
      .textureView = state.textures.gradient.view,
    },
    [3] = (WGPUBindGroupEntry){
      .binding = 3,
      .buffer  = state.graphics.uniform_buffer.buffer,
      .size    = state.graphics.uniform_buffer.size,
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
}

static void init_graphics_pipeline(wgpu_context_t* wgpu_context)
{
  /* Pipeline layout */
  state.graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Graphics pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.graphics.bind_group_layout,
    });
  ASSERT(state.graphics.pipeline_layout != NULL);

  /* Shader modules */
  WGPUShaderModule vert_module = wgpu_create_shader_module(
    wgpu_context->device, particle_vertex_shader_wgsl);
  WGPUShaderModule frag_module = wgpu_create_shader_module(
    wgpu_context->device, particle_fragment_shader_wgsl);

  /* Additive blending (matches Vulkan: src=ONE, dst=ONE) */
  WGPUBlendState additive_blend = {
    .color = {
      .operation = WGPUBlendOperation_Add,
      .srcFactor = WGPUBlendFactor_One,
      .dstFactor = WGPUBlendFactor_One,
    },
    .alpha = {
      .operation = WGPUBlendOperation_Add,
      .srcFactor = WGPUBlendFactor_SrcAlpha,
      .dstFactor = WGPUBlendFactor_DstAlpha,
    },
  };

  /* Vertex buffer layout: per-instance particle data */
  WGPU_VERTEX_BUFFER_LAYOUT(
    particle, sizeof(particle_t),
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2,
                       offsetof(particle_t, pos)),
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                       offsetof(particle_t, gradient_pos)))

  /* Override step mode to per-instance (billboard quads via vertex_index) */
  particle_vertex_buffer_layout.stepMode = WGPUVertexStepMode_Instance;

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Particle render pipeline"),
    .layout = state.graphics.pipeline_layout,
    .vertex = {
      .module      = vert_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &particle_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState){
      .module      = frag_module,
      .entryPoint  = STRVIEW("fs_main"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &additive_blend,
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
      .mask  = 0xFFFFFFFF,
    },
  };

  state.graphics.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.graphics.pipeline != NULL);

  wgpuShaderModuleRelease(vert_module);
  wgpuShaderModuleRelease(frag_module);
}

/* -------------------------------------------------------------------------- *
 * Compute pipeline
 * -------------------------------------------------------------------------- */

static void init_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = PARTICLE_COUNT * sizeof(particle_t),
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
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.compute.storage_buffer.buffer,
      .size    = state.compute.storage_buffer.size,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = state.compute.uniform_buffer.buffer,
      .offset  = 0,
      .size    = state.compute.uniform_buffer.size,
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
  WGPUShaderModule comp_module = wgpu_create_shader_module(
    wgpu_context->device, particle_compute_shader_wgsl);

  /* Compute pipeline */
  state.compute.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Particle compute pipeline"),
      .layout  = state.compute.pipeline_layout,
      .compute = {
        .module     = comp_module,
        .entryPoint = STRVIEW("main"),
      },
    });
  ASSERT(state.compute.pipeline != NULL);

  wgpuShaderModuleRelease(comp_module);
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);
  igCheckbox("Attach Attractor to Cursor", &state.settings.attach_to_cursor);
  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Main callbacks
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 4,
    .num_channels = 2,
    .num_lanes    = 2,
    .logger.func  = slog_func,
  });

  init_storage_buffer(wgpu_context);
  init_uniform_buffers(wgpu_context);
  init_textures(wgpu_context);

  /* Graphics pipeline */
  init_graphics_bind_group_layout(wgpu_context);
  rebuild_graphics_bind_group(wgpu_context);
  init_graphics_pipeline(wgpu_context);

  /* Compute pipeline */
  init_compute_pipeline(wgpu_context);

  /* GUI */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async texture loads */
  sfetch_dowork();

  /* Recreate textures if loaded from disk */
  bool rebind = false;
  if (state.textures.particle.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.textures.particle);
    FREE_TEXTURE_PIXELS(state.textures.particle);
    rebind = true;
  }
  if (state.textures.gradient.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.textures.gradient);
    FREE_TEXTURE_PIXELS(state.textures.gradient);
    rebind = true;
  }
  if (rebind) {
    rebuild_graphics_bind_group(wgpu_context);
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

  /* Update animation (matches Vulkan exactly) */
  if (!state.settings.attach_to_cursor) {
    if (state.anim_start > 0.0f) {
      state.anim_start -= delta_time * 5.0f;
    }
    else if (state.anim_start <= 0.0f) {
      state.timer += delta_time * 0.04f;
      if (state.timer > 1.0f) {
        state.timer = 0.0f;
      }
    }
  }

  /* Update uniform buffers */
  update_uniform_buffers(wgpu_context);

  /* ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass: update particle positions */
  {
    WGPUComputePassEncoder cpass
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass, state.compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(cpass, 0, state.compute.bind_group, 0,
                                       NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cpass, PARTICLE_COUNT / 256, 1, 1);
    wgpuComputePassEncoderEnd(cpass);
    wgpuComputePassEncoderRelease(cpass);
  }

  /* Render pass: draw particles as instanced billboard quads */
  {
    WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(rpass, state.graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.graphics.bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass, 0, state.compute.storage_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    /* 6 vertices per quad (2 triangles), PARTICLE_COUNT instances */
    wgpuRenderPassEncoderDraw(rpass, 6, PARTICLE_COUNT, 0, 0);
    wgpuRenderPassEncoderEnd(rpass);
    wgpuRenderPassEncoderRelease(rpass);
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Render GUI overlay */
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

  /* Textures */
  wgpu_destroy_texture(&state.textures.particle);
  wgpu_destroy_texture(&state.textures.gradient);

  /* Graphics pipeline */
  WGPU_RELEASE_RESOURCE(Buffer, state.graphics.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)

  /* Compute pipeline */
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.storage_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipeline)

  sfetch_shutdown();
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  /* Track cursor position for attractor (normalized to -1..1).
   * Use wgpu_context dimensions which are always valid (unlike
   * input_event->window_width which is only set after a resize event). */
  if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    if (wgpu_context->width > 0 && wgpu_context->height > 0) {
      state.cursor_x
        = (input_event->mouse_x / (float)wgpu_context->width) * 2.0f - 1.0f;
      state.cursor_y
        = -((input_event->mouse_y / (float)wgpu_context->height) * 2.0f - 1.0f);
    }
  }
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Compute Shader Particle System",
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
static const char* particle_vertex_shader_wgsl = CODE(
  struct ScreenParams {
    screen_width : f32,
    screen_height : f32,
    point_size : f32,
  };

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) gradient_pos : f32,
    @location(1) uv : vec2<f32>,
  };

  @group(0) @binding(3) var<uniform> screen_params : ScreenParams;

  @vertex
  fn vs_main(
    @builtin(vertex_index) vertex_index : u32,
    @location(0) pos : vec2<f32>,
    @location(1) gradient_pos_in : vec4<f32>,
  ) -> VertexOutput {
    /* Billboard quad corners (2 triangles = 6 vertices) */
    var corners = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 1.0, -1.0),
      vec2<f32>( 1.0,  1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 1.0,  1.0),
      vec2<f32>(-1.0,  1.0),
    );

    /* UV coordinates matching Vulkan gl_PointCoord convention:
     * (0,0) at top-left, (1,1) at bottom-right */
    var uvs = array<vec2<f32>, 6>(
      vec2<f32>(0.0, 1.0),
      vec2<f32>(1.0, 1.0),
      vec2<f32>(1.0, 0.0),
      vec2<f32>(0.0, 1.0),
      vec2<f32>(1.0, 0.0),
      vec2<f32>(0.0, 0.0),
    );

    let corner = corners[vertex_index];

    /* Convert pixel-space point size to NDC offset.
     * In clip space with w=1: 1 pixel = 2.0 / screenDim
     * Half of point_size in NDC = point_size / screenDim */
    let pixel_scale = vec2<f32>(
      screen_params.point_size / screen_params.screen_width,
      screen_params.point_size / screen_params.screen_height,
    );

    var output : VertexOutput;
    output.position = vec4<f32>(pos + corner * pixel_scale, 1.0, 1.0);
    output.gradient_pos = gradient_pos_in.x;
    output.uv = uvs[vertex_index];
    return output;
  }
);

static const char* particle_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var color_map : texture_2d<f32>;
  @group(0) @binding(1) var color_sampler : sampler;
  @group(0) @binding(2) var gradient_ramp : texture_2d<f32>;

  @fragment
  fn fs_main(
    @location(0) gradient_pos : f32,
    @location(1) uv : vec2<f32>,
  ) -> @location(0) vec4<f32> {
    /* Sample particle sprite texture (equivalent of Vulkan gl_PointCoord) */
    let sprite_color = textureSample(color_map, color_sampler, uv);
    /* Sample gradient ramp for color animation */
    let gradient_color = textureSample(
      gradient_ramp, color_sampler, vec2<f32>(gradient_pos, 0.0)
    );
    /* Combine: sprite shape * gradient color (matches Vulkan fragment shader) */
    return vec4<f32>(sprite_color.rgb * gradient_color.rgb, sprite_color.a);
  }
);

static const char* particle_compute_shader_wgsl = CODE(
  struct Particle {
    pos : vec2<f32>,
    vel : vec2<f32>,
    gradient_pos : vec4<f32>,
  };

  struct Params {
    delta_t : f32,
    dest_x : f32,
    dest_y : f32,
    particle_count : i32,
  };

  @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
  @group(0) @binding(1) var<uniform> params : Params;

  fn attraction(pos : vec2<f32>, attract_pos : vec2<f32>) -> vec2<f32> {
    let delta = attract_pos - pos;
    let damp = 0.5;
    let d_damped_dot = dot(delta, delta) + damp;
    let inv_dist = 1.0 / sqrt(d_damped_dot);
    let inv_dist_cubed = inv_dist * inv_dist * inv_dist;
    return delta * inv_dist_cubed * 0.0035;
  }

  fn repulsion(pos : vec2<f32>, attract_pos : vec2<f32>) -> vec2<f32> {
    let delta = attract_pos - pos;
    let target_distance = sqrt(dot(delta, delta));
    return delta * (1.0 / (target_distance * target_distance * target_distance))
           * -0.000035;
  }

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;
    if (index >= u32(params.particle_count)) {
      return;
    }

    /* Read position and velocity */
    var v_vel = particles[index].vel;
    var v_pos = particles[index].pos;
    let g_pos = particles[index].gradient_pos;

    let dest_pos = vec2<f32>(params.dest_x, params.dest_y);

    /* Apply repulsion force only (matches Vulkan exactly) */
    v_vel = v_vel + repulsion(v_pos, dest_pos) * 0.05;

    /* Move by velocity */
    v_pos = v_pos + v_vel * params.delta_t;

    /* Collide with boundary */
    if (v_pos.x < -1.0 || v_pos.x > 1.0 || v_pos.y < -1.0 || v_pos.y > 1.0) {
      v_vel = (-v_vel * 0.1) + attraction(v_pos, dest_pos) * 12.0;
    } else {
      particles[index].pos = v_pos;
    }

    /* Write back velocity */
    particles[index].vel = v_vel;

    /* Animate gradient position */
    particles[index].gradient_pos.x = g_pos.x + 0.02 * params.delta_t;
    if (particles[index].gradient_pos.x > 1.0) {
      particles[index].gradient_pos.x =
        particles[index].gradient_pos.x - 1.0;
    }
  }
);
// clang-format on
