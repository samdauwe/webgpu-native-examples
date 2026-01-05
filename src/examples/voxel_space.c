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
 * WebGPU Example - Voxel Space
 *
 * This example shows a voxel-based terrain rendering technique using WebGPU
 * compute shaders. The terrain is rendered using a height map and color map,
 * similar to the classic Comanche game.
 *
 * The implementation uses:
 *  * Compute shaders for efficient terrain rendering
 *  * Texture loading with sokol_fetch
 *  * Flight simulator controls (keyboard + mouse)
 *  * Asynchronous image loading
 *
 * Ref:
 * https://github.com/s-macke/VoxelSpace
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* compute_shader_wgsl;
static const char* fullscreen_vertex_shader_wgsl;
static const char* fullscreen_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Voxel Space Example
 * -------------------------------------------------------------------------- */

#define MAP_SIZE 1024
#define CANVAS_WIDTH DEFAULT_WINDOW_WIDTH
#define CANVAS_HEIGHT DEFAULT_WINDOW_HEIGHT

/* State struct */
static struct {
  /* Camera parameters */
  struct {
    float x;
    float y;
    float height;
    float angle;
    float horizon;
    float distance;
  } camera;
  /* Input state */
  struct {
    float forward_backward;
    float left_right;
    float up_down;
    WGPUBool lookup;
    WGPUBool lookdown;
    WGPUBool mouse_pressed;
    float mouse_start_x;
    float mouse_start_y;
  } input;
  /* Time tracking */
  uint64_t last_time;
  /* Textures */
  struct {
    wgpu_texture_t height_map;
    wgpu_texture_t color_map;
    WGPUTexture render_texture;
    WGPUTextureView render_texture_view;
  } textures;
  /* Buffers */
  struct {
    WGPUBuffer uniform_buffer;
  } buffers;
  /* Samplers */
  WGPUSampler linear_sampler;
  /* Pipelines */
  struct {
    WGPUComputePipeline compute;
    WGPURenderPipeline render;
  } pipelines;
  /* Bind groups */
  struct {
    WGPUBindGroupLayout compute;
    WGPUBindGroupLayout render;
  } bind_group_layouts;
  struct {
    WGPUBindGroup compute;
    WGPUBindGroup render;
  } bind_groups;
  /* Pipeline layouts */
  struct {
    WGPUPipelineLayout compute;
    WGPUPipelineLayout render;
  } pipeline_layouts;
  /* Render pass */
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDescriptor descriptor;
  } render_pass;
  /* Image loading buffers */
  uint8_t height_map_buffer[MAP_SIZE * MAP_SIZE * 4];
  uint8_t color_map_buffer[MAP_SIZE * MAP_SIZE * 4];
  WGPUBool height_map_loaded;
  WGPUBool color_map_loaded;
  WGPUBool initialized;
} state = {
  .camera = {
    .x        = 512.0f,
    .y        = 512.0f,
    .height   = 150.0f,
    .angle    = 0.0f,
    .horizon  = 100.0f,
    .distance = 1000.0f,
  },
  .input = {
    .forward_backward = 0.0f,
    .left_right       = 0.0f,
    .up_down          = 0.0f,
    .lookup           = false,
    .lookdown         = false,
    .mouse_pressed    = false,
    .mouse_start_x    = 0.0f,
    .mouse_start_y    = 0.0f,
  },
  .height_map_loaded = false,
  .color_map_loaded  = false,
  .render_pass = {
    .color_attachment = {
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.5, 0.7, 0.9, 1.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    },
    .descriptor = {
      .colorAttachmentCount = 1,
      .colorAttachments     = &state.render_pass.color_attachment,
    },
  }
};

/* -------------------------------------------------------------------------- *
 * Image Loading
 * -------------------------------------------------------------------------- */

static void height_map_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Height map fetch failed, error: %d\n", response->error_code);
    return;
  }

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
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .pixels = {
        .ptr  = pixels,
        .size = img_width * img_height * 4,
      },
    };
    texture->desc.is_dirty  = true;
    state.height_map_loaded = true;
  }
}

static void color_map_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Color map fetch failed, error: %d\n", response->error_code);
    return;
  }

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
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .pixels = {
        .ptr  = pixels,
        .size = img_width * img_height * 4,
      },
    };
    texture->desc.is_dirty = true;
    state.color_map_loaded = true;
  }
}

static void init_textures(wgpu_context_t* wgpu_context)
{
  /* Create dummy textures initially */
  state.textures.height_map
    = wgpu_create_color_bars_texture(wgpu_context, NULL);
  state.textures.color_map = wgpu_create_color_bars_texture(wgpu_context, NULL);

  /* Start loading the height map */
  wgpu_texture_t* height_texture = &state.textures.height_map;
  sfetch_send(&(sfetch_request_t){
    .path      = "assets/textures/voxelspace_heightmap.png",
    .callback  = height_map_fetch_callback,
    .buffer    = SFETCH_RANGE(state.height_map_buffer),
    .user_data = {
      .ptr  = &height_texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });

  /* Start loading the color map */
  wgpu_texture_t* color_texture = &state.textures.color_map;
  sfetch_send(&(sfetch_request_t){
    .path      = "assets/textures/voxelspace_colormap.png",
    .callback  = color_map_fetch_callback,
    .buffer    = SFETCH_RANGE(state.color_map_buffer),
    .user_data = {
      .ptr  = &color_texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });

  /* Create render texture */
  state.textures.render_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Voxel Space - Render texture"),
      .size  = (WGPUExtent3D){
        .width              = CANVAS_WIDTH,
        .height             = CANVAS_HEIGHT,
        .depthOrArrayLayers = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .usage         = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding,
    });
  ASSERT(state.textures.render_texture != NULL);

  state.textures.render_texture_view
    = wgpuTextureCreateView(state.textures.render_texture, NULL);
  ASSERT(state.textures.render_texture_view != NULL);

  /* Create sampler */
  state.linear_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label = STRVIEW("Voxel Space - Linear sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.linear_sampler != NULL);
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Camera uniform buffer (8 floats: x, y, height, angle, horizon, distance,
   * screen_width, screen_height) */
  state.buffers.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Voxel Space - Uniform buffer"),
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = 32, /* 8 * 4 bytes */
      .mappedAtCreation = false,
    });
  ASSERT(state.buffers.uniform_buffer != NULL);
}

static void update_uniform_buffer(wgpu_context_t* wgpu_context)
{
  float uniform_data[8] = {
    state.camera.x,      state.camera.y,       state.camera.height,
    state.camera.angle,  state.camera.horizon, state.camera.distance,
    (float)CANVAS_WIDTH, (float)CANVAS_HEIGHT,
  };
  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.uniform_buffer, 0,
                       uniform_data, sizeof(uniform_data));
}

static void init_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = (WGPUBindGroupLayoutEntry){
      /* Binding 0: Uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = 32,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      /* Binding 1: Height map texture */
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [2] = (WGPUBindGroupLayoutEntry){
      /* Binding 2: Color map texture */
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [3] = (WGPUBindGroupLayoutEntry){
      /* Binding 3: Output storage texture */
      .binding    = 3,
      .visibility = WGPUShaderStage_Compute,
      .storageTexture = (WGPUStorageTextureBindingLayout){
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA8Unorm,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
  };

  state.bind_group_layouts.compute = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Voxel Space - Compute bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    });
  ASSERT(state.bind_group_layouts.compute != NULL);

  /* Pipeline layout */
  state.pipeline_layouts.compute = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Voxel Space - Compute pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.bind_group_layouts.compute,
    });
  ASSERT(state.pipeline_layouts.compute != NULL);

  /* Compute shader module */
  WGPUShaderModule compute_shader_module
    = wgpu_create_shader_module(wgpu_context->device, compute_shader_wgsl);

  /* Compute pipeline */
  state.pipelines.compute = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Voxel Space - Compute pipeline"),
      .layout  = state.pipeline_layouts.compute,
      .compute = {
        .module     = compute_shader_module,
        .entryPoint = STRVIEW("main"),
      },
    });
  ASSERT(state.pipelines.compute != NULL);

  /* Release shader module */
  WGPU_RELEASE_RESOURCE(ShaderModule, compute_shader_module);
}

static void init_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* Bind group layout for render pipeline */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      /* Binding 0: Render texture */
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      /* Binding 1: Sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };

  state.bind_group_layouts.render = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Voxel Space - Render bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    });
  ASSERT(state.bind_group_layouts.render != NULL);

  /* Pipeline layout */
  state.pipeline_layouts.render = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Voxel Space - Render pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.bind_group_layouts.render,
    });
  ASSERT(state.pipeline_layouts.render != NULL);

  /* Shader modules */
  WGPUShaderModule vertex_shader_module = wgpu_create_shader_module(
    wgpu_context->device, fullscreen_vertex_shader_wgsl);
  WGPUShaderModule fragment_shader_module = wgpu_create_shader_module(
    wgpu_context->device, fullscreen_fragment_shader_wgsl);

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Render pipeline */
  state.pipelines.render = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Voxel Space - Render pipeline"),
      .layout = state.pipeline_layouts.render,
      .vertex = (WGPUVertexState){
        .module     = vertex_shader_module,
        .entryPoint = STRVIEW("main"),
        .bufferCount = 0,
        .buffers     = NULL,
      },
      .primitive = (WGPUPrimitiveState){
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .multisample = (WGPUMultisampleState){
        .count                  = 1,
        .mask                   = 0xFFFFFFFF,
        .alphaToCoverageEnabled = false,
      },
      .fragment = &(WGPUFragmentState){
        .module      = fragment_shader_module,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets     = &color_target_state,
      },
    });
  ASSERT(state.pipelines.render != NULL);

  /* Release shader modules */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_shader_module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_shader_module);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Compute bind group */
  WGPUBindGroupEntry compute_bg_entries[4] = {
    [0] = (WGPUBindGroupEntry){
      /* Binding 0: Uniform buffer */
      .binding = 0,
      .buffer  = state.buffers.uniform_buffer,
      .offset  = 0,
      .size    = 32,
    },
    [1] = (WGPUBindGroupEntry){
      /* Binding 1: Height map texture */
      .binding     = 1,
      .textureView = state.textures.height_map.view,
    },
    [2] = (WGPUBindGroupEntry){
      /* Binding 2: Color map texture */
      .binding     = 2,
      .textureView = state.textures.color_map.view,
    },
    [3] = (WGPUBindGroupEntry){
      /* Binding 3: Output storage texture */
      .binding     = 3,
      .textureView = state.textures.render_texture_view,
    },
  };

  state.bind_groups.compute = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Voxel Space - Compute bind group"),
      .layout     = state.bind_group_layouts.compute,
      .entryCount = (uint32_t)ARRAY_SIZE(compute_bg_entries),
      .entries    = compute_bg_entries,
    });
  ASSERT(state.bind_groups.compute != NULL);

  /* Render bind group */
  WGPUBindGroupEntry render_bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      /* Binding 0: Render texture */
      .binding     = 0,
      .textureView = state.textures.render_texture_view,
    },
    [1] = (WGPUBindGroupEntry){
      /* Binding 1: Sampler */
      .binding = 1,
      .sampler = state.linear_sampler,
    },
  };

  state.bind_groups.render = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Voxel Space - Render bind group"),
      .layout     = state.bind_group_layouts.render,
      .entryCount = (uint32_t)ARRAY_SIZE(render_bg_entries),
      .entries    = render_bg_entries,
    });
  ASSERT(state.bind_groups.render != NULL);
}

static void update_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Release old bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.compute);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.render);

  /* Recreate bind groups with updated textures */
  init_bind_groups(wgpu_context);
}

static void update_camera(wgpu_context_t* wgpu_context, float delta_time)
{
  UNUSED_VAR(wgpu_context);
  const float speed          = 50.0f * delta_time;
  const float turn_speed     = 1.0f * delta_time;
  const float vertical_speed = 25.0f * delta_time;

  /* Apply turning */
  if (state.input.left_right != 0.0f) {
    state.camera.angle += state.input.left_right * turn_speed;
  }

  /* Forward/backward movement */
  if (state.input.forward_backward != 0.0f) {
    state.camera.x += cosf(state.camera.angle) * state.input.forward_backward
                      * (speed / 3.0f);
    state.camera.y += sinf(state.camera.angle) * state.input.forward_backward
                      * (speed / 3.0f);
  }

  /* Vertical movement */
  if (state.input.up_down != 0.0f) {
    state.camera.height += state.input.up_down * vertical_speed * 0.5f;
  }

  /* Look up/down */
  if (state.input.lookup) {
    state.camera.horizon += 100.0f * delta_time;
  }
  if (state.input.lookdown) {
    state.camera.horizon -= 100.0f * delta_time;
  }

  /* Keep camera within bounds */
  state.camera.x = fmodf(fmodf(state.camera.x, MAP_SIZE) + MAP_SIZE, MAP_SIZE);
  state.camera.y = fmodf(fmodf(state.camera.y, MAP_SIZE) + MAP_SIZE, MAP_SIZE);
  state.camera.height = CLAMP(state.camera.height, 10.0f, 500.0f);
  state.camera.horizon
    = CLAMP(state.camera.horizon, 0.0f, (float)CANVAS_HEIGHT);
}

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  UNUSED_VAR(wgpu_context);

  if (input_event->type == INPUT_EVENT_TYPE_KEY_DOWN) {
    switch (input_event->key_code) {
      case KEY_W:
      case KEY_UP:
        state.input.forward_backward = 3.0f;
        break;
      case KEY_S:
      case KEY_DOWN:
        state.input.forward_backward = -3.0f;
        break;
      case KEY_A:
      case KEY_LEFT:
        state.input.left_right = 1.0f;
        break;
      case KEY_D:
      case KEY_RIGHT:
        state.input.left_right = -1.0f;
        break;
      case KEY_R:
        state.input.up_down = 2.0f;
        break;
      case KEY_F:
        state.input.up_down = -2.0f;
        break;
      case KEY_Q:
        state.input.lookup = true;
        break;
      case KEY_E:
        state.input.lookdown = true;
        break;
      default:
        break;
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_KEY_UP) {
    switch (input_event->key_code) {
      case KEY_W:
      case KEY_UP:
      case KEY_S:
      case KEY_DOWN:
        state.input.forward_backward = 0.0f;
        break;
      case KEY_A:
      case KEY_LEFT:
      case KEY_D:
      case KEY_RIGHT:
        state.input.left_right = 0.0f;
        break;
      case KEY_R:
      case KEY_F:
        state.input.up_down = 0.0f;
        break;
      case KEY_Q:
        state.input.lookup = false;
        break;
      case KEY_E:
        state.input.lookdown = false;
        break;
      default:
        break;
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_DOWN
           && input_event->mouse_button == BUTTON_LEFT) {
    state.input.mouse_pressed    = true;
    state.input.forward_backward = 3.0f;
    state.input.mouse_start_x    = input_event->mouse_x;
    state.input.mouse_start_y    = input_event->mouse_y;
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_UP) {
    state.input.mouse_pressed    = false;
    state.input.forward_backward = 0.0f;
    state.input.left_right       = 0.0f;
    state.input.up_down          = 0.0f;
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE
           && state.input.mouse_pressed) {
    float dx = input_event->mouse_x - state.input.mouse_start_x;
    float dy = input_event->mouse_y - state.input.mouse_start_y;

    state.input.left_right = dx / (float)wgpu_context->width * 2.0f;
    state.camera.horizon
      = 100.0f + (-dy / (float)wgpu_context->height * 500.0f);
    state.input.up_down = -dy / (float)wgpu_context->height * 10.0f;
  }
}

static int init(wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 2,
      .num_channels = 1,
      .num_lanes    = 1,
      .logger.func  = slog_func,
    });
    stm_setup();
    state.last_time = stm_now();
    init_textures(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_compute_pipeline(wgpu_context);
    init_render_pipeline(wgpu_context);
    init_bind_groups(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async file loads */
  sfetch_dowork();

  /* Recreate textures when pixel data is loaded */
  if (state.textures.height_map.desc.is_dirty) {
    /* Destroy old texture */
    wgpu_destroy_texture(&state.textures.height_map);

    /* Create new texture */
    WGPUTextureDescriptor tex_desc = {
      .label = STRVIEW("Height map - Texture"),
      .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size = (WGPUExtent3D){
        .width = state.textures.height_map.desc.extent.width,
        .height = state.textures.height_map.desc.extent.height,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
    };
    state.textures.height_map.handle
      = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);

    /* Upload pixel data with correct bytesPerRow */
    const uint32_t width  = state.textures.height_map.desc.extent.width;
    const uint32_t height = state.textures.height_map.desc.extent.height;
    wgpuQueueWriteTexture(wgpu_context->queue,
                          &(WGPUTexelCopyTextureInfo){
                            .texture  = state.textures.height_map.handle,
                            .mipLevel = 0,
                            .origin   = (WGPUOrigin3D){.x = 0, .y = 0, .z = 0},
                            .aspect   = WGPUTextureAspect_All,
                          },
                          state.textures.height_map.desc.pixels.ptr,
                          state.textures.height_map.desc.pixels.size,
                          &(WGPUTexelCopyBufferLayout){
                            .offset       = 0,
                            .bytesPerRow  = width * 4,
                            .rowsPerImage = height,
                          },
                          &(WGPUExtent3D){
                            .width              = width,
                            .height             = height,
                            .depthOrArrayLayers = 1,
                          });

    /* Create texture view */
    state.textures.height_map.view
      = wgpuTextureCreateView(state.textures.height_map.handle, NULL);

    FREE_TEXTURE_PIXELS(state.textures.height_map);
    state.textures.height_map.desc.is_dirty = false;
    state.textures.height_map.initialized   = true;
  }

  if (state.textures.color_map.desc.is_dirty) {
    /* Destroy old texture */
    wgpu_destroy_texture(&state.textures.color_map);

    /* Create new texture */
    WGPUTextureDescriptor tex_desc = {
      .label = STRVIEW("Color map - Texture"),
      .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size = (WGPUExtent3D){
        .width = state.textures.color_map.desc.extent.width,
        .height = state.textures.color_map.desc.extent.height,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount = 1,
    };
    state.textures.color_map.handle
      = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);

    /* Upload pixel data with correct bytesPerRow */
    const uint32_t width  = state.textures.color_map.desc.extent.width;
    const uint32_t height = state.textures.color_map.desc.extent.height;
    wgpuQueueWriteTexture(wgpu_context->queue,
                          &(WGPUTexelCopyTextureInfo){
                            .texture  = state.textures.color_map.handle,
                            .mipLevel = 0,
                            .origin   = (WGPUOrigin3D){.x = 0, .y = 0, .z = 0},
                            .aspect   = WGPUTextureAspect_All,
                          },
                          state.textures.color_map.desc.pixels.ptr,
                          state.textures.color_map.desc.pixels.size,
                          &(WGPUTexelCopyBufferLayout){
                            .offset       = 0,
                            .bytesPerRow  = width * 4,
                            .rowsPerImage = height,
                          },
                          &(WGPUExtent3D){
                            .width              = width,
                            .height             = height,
                            .depthOrArrayLayers = 1,
                          });

    /* Create texture view */
    state.textures.color_map.view
      = wgpuTextureCreateView(state.textures.color_map.handle, NULL);

    FREE_TEXTURE_PIXELS(state.textures.color_map);
    state.textures.color_map.desc.is_dirty = false;
    state.textures.color_map.initialized   = true;
  }

  /* Update bind groups if textures were reloaded */
  if (state.height_map_loaded && state.color_map_loaded) {
    if (state.textures.height_map.initialized
        && state.textures.color_map.initialized) {
      update_bind_groups(wgpu_context);
      state.height_map_loaded = false;
      state.color_map_loaded  = false;
    }
  }

  /* Update camera based on input */
  uint64_t current_time = stm_now();
  float delta_time = (float)stm_sec(stm_diff(current_time, state.last_time));
  state.last_time  = current_time;

  update_camera(wgpu_context, delta_time);

  /* Update uniform buffer */
  update_uniform_buffer(wgpu_context);

  /* Get swap chain texture */
  WGPUTextureView swapchain_view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("Voxel Space - Command encoder"),
                          });

  /* Compute pass */
  WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(
    command_encoder, &(WGPUComputePassDescriptor){
                       .label = STRVIEW("Voxel Space - Compute pass"),
                     });
  wgpuComputePassEncoderSetPipeline(compute_pass, state.pipelines.compute);
  wgpuComputePassEncoderSetBindGroup(compute_pass, 0, state.bind_groups.compute,
                                     0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(compute_pass,
                                           (CANVAS_WIDTH + 63) / 64, 1, 1);
  wgpuComputePassEncoderEnd(compute_pass);
  WGPU_RELEASE_RESOURCE(ComputePassEncoder, compute_pass);

  /* Render pass */
  state.render_pass.color_attachment.view = swapchain_view;
  WGPURenderPassEncoder render_pass       = wgpuCommandEncoderBeginRenderPass(
    command_encoder, &state.render_pass.descriptor);
  wgpuRenderPassEncoderSetPipeline(render_pass, state.pipelines.render);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0, state.bind_groups.render, 0,
                                    NULL);
  wgpuRenderPassEncoderDraw(render_pass, 3, 1, 0, 0);
  wgpuRenderPassEncoderEnd(render_pass);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass);

  /* Submit command buffer */
  WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(
    command_encoder, &(WGPUCommandBufferDescriptor){
                       .label = STRVIEW("Voxel Space - Command buffer"),
                     });
  ASSERT(command_buffer != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, command_encoder);

  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Release buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.uniform_buffer);

  /* Release textures */
  wgpu_destroy_texture(&state.textures.height_map);
  wgpu_destroy_texture(&state.textures.color_map);
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.render_texture_view);
  WGPU_RELEASE_RESOURCE(Texture, state.textures.render_texture);

  /* Release sampler */
  WGPU_RELEASE_RESOURCE(Sampler, state.linear_sampler);

  /* Release bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.compute);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.render);

  /* Release bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.compute);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.render);

  /* Release pipelines */
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.pipelines.compute);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.render);

  /* Release pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.compute);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.render);

  /* Shutdown sokol_fetch */
  sfetch_shutdown();
}

int main(int argc, char* argv[])
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  wgpu_start(&(wgpu_desc_t){
    .title           = "Voxel Space",
    .width           = CANVAS_WIDTH,
    .height          = CANVAS_HEIGHT,
    .no_depth_buffer = true,
    .init_cb         = init,
    .frame_cb        = frame,
    .shutdown_cb     = shutdown,
    .input_event_cb  = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
/* Compute Shader - Voxel Space Rendering */
static const char* compute_shader_wgsl = CODE(
  struct Camera{
    x : f32,
    y : f32,
    height : f32,
    angle : f32,
    horizon : f32,
    distance : f32,
    screen_width : f32,
    screen_height : f32,
  };

  @group(0) @binding(0) var<uniform> camera : Camera;
  @group(0) @binding(1) var height_map : texture_2d<f32>;
  @group(0) @binding(2) var color_map : texture_2d<f32>;
  @group(0) @binding(3) var output_tex : texture_storage_2d<rgba8unorm, write>;

  @compute @workgroup_size(64, 1, 1) fn main(@builtin(global_invocation_id)
                                               global_id : vec3<u32>) {
    let screen_x = i32(global_id.x);
    let screen_w = i32(camera.screen_width);
    let screen_h = i32(camera.screen_height);

    if (screen_x >= screen_w) {
      return;
    }

    let fov       = 1.0;
    let ray_angle = camera.angle + (f32(screen_x) / f32(screen_w) - 0.5) * fov;
    let sin_a     = sin(ray_angle);
    let cos_a     = cos(ray_angle);

    // Clear column (Sky)
    for (var y = 0; y < screen_h; y++) {
      let sky_color
        = mix(vec4<f32>(0.5, 0.7, 0.9, 1.0), vec4<f32>(0.2, 0.4, 0.8, 1.0),
              f32(y) / f32(screen_h));
      textureStore(output_tex, vec2<i32>(screen_x, y), sky_color);
    }

    var y_buffer = f32(screen_h);

    let max_dist = i32(camera.distance);
    let map_dims = textureDimensions(height_map);
    let map_w    = i32(map_dims.x);
    let map_h    = i32(map_dims.y);

    var z  = 1.0;
    var dz = 1.0;

    for (var i = 0; i < max_dist; i++) {
      let map_x = (i32(floor(camera.x + cos_a * z)) % map_w + map_w) % map_w;
      let map_y = (i32(floor(camera.y + sin_a * z)) % map_h + map_h) % map_h;

      let h_val_norm = textureLoad(height_map, vec2<i32>(map_x, map_y), 0).r;
      let h_val      = h_val_norm * 255.0;
      let col_val    = textureLoad(color_map, vec2<i32>(map_x, map_y), 0);

      let scale_height = 240.0;
      let height_on_screen
        = (camera.height - h_val) / z * scale_height + camera.horizon;
      let draw_height = clamp(height_on_screen, 0.0, f32(screen_h));

      if (draw_height < y_buffer) {
        let top    = i32(draw_height);
        let bottom = i32(y_buffer);

        let fog_factor
          = clamp((z - 100.0) / (camera.distance - 100.0), 0.0, 1.0);
        let fog_color   = vec4<f32>(0.5, 0.7, 0.9, 1.0);
        let final_color = mix(col_val, fog_color, fog_factor);

        for (var y = top; y < bottom; y++) {
          textureStore(output_tex, vec2<i32>(screen_x, y), final_color);
        }
        y_buffer = draw_height;
      }

      dz += 0.005;
      z += dz;

      if (z > camera.distance) {
        break;
      }
    }
  }
);

/* Fullscreen Vertex Shader */
static const char* fullscreen_vertex_shader_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) position : vec4<f32>, @location(0) tex_coord : vec2<f32>,
  };

  @vertex fn main(@builtin(vertex_index) vertex_index : u32)->VertexOutput {
    var output : VertexOutput;
    let x            = f32((vertex_index << 1u) & 2u);
    let y            = f32(vertex_index & 2u);
    output.position  = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.tex_coord = vec2<f32>(x, y);
    return output;
  }
);

/* Fullscreen Fragment Shader */
static const char* fullscreen_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var render_texture : texture_2d<f32>;
  @group(0) @binding(1) var tex_sampler : sampler;

  @fragment fn main(@location(0) tex_coord : vec2<f32>)
    ->@location(0) vec4<f32> {
      return textureSample(render_texture, tex_sampler, tex_coord);
    }
);
// clang-format on
