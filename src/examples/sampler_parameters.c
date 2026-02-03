#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

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
 * WebGPU Example - Sampler Parameters
 *
 * Visualizes what all the sampler parameters do. Shows a textured plane at
 * various scales (rotated, head-on, in perspective, and in vanishing
 * perspective). The bottom-right view shows the raw contents of the 4 mipmap
 * levels of the test texture (16x16, 8x8, 4x4, and 2x2).
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/samplerParameters
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* textured_square_shader_wgsl;
static const char* show_texture_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define CANVAS_SIZE (200u)
#define VIEWPORT_GRID_SIZE (4u)
#define VIEWPORT_GRID_STRIDE (CANVAS_SIZE / VIEWPORT_GRID_SIZE)
#define VIEWPORT_SIZE (VIEWPORT_GRID_STRIDE - 2u)
#define TEXTURE_MIP_LEVELS (4u)
#define TEXTURE_BASE_SIZE (16u)
#define NUM_MATRICES (15u) /* 4 rows * 4 columns - 1 for the bottom right */
#define CAMERA_DIST (3.0f)

/* Address mode options */
typedef enum address_mode_t {
  ADDRESS_MODE_CLAMP_TO_EDGE = 0,
  ADDRESS_MODE_REPEAT        = 1,
  ADDRESS_MODE_MIRROR_REPEAT = 2,
  ADDRESS_MODE_COUNT         = 3,
} address_mode_t;

/* Filter mode options */
typedef enum filter_mode_t {
  FILTER_MODE_NEAREST = 0,
  FILTER_MODE_LINEAR  = 1,
  FILTER_MODE_COUNT   = 2,
} filter_mode_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Textures */
  struct {
    WGPUTexture checkerboard;
    WGPUTextureView checkerboard_view;
    WGPUTexture render_target;
    WGPUTextureView render_target_view;
  } textures;

  /* Buffers */
  struct {
    WGPUBuffer config;
    WGPUBuffer matrices;
  } buffers;

  /* Pipelines */
  WGPURenderPipeline textured_square_pipeline;
  WGPURenderPipeline show_texture_pipeline;
  WGPUBindGroupLayout textured_square_bind_group_layout;

  /* Bind groups */
  WGPUBindGroup show_texture_bind_group;

  /* Samplers - recreated each frame based on settings */
  WGPUSampler current_sampler;

  /* View projection matrix */
  mat4 view_proj;

  /* Config buffer data */
  struct {
    float animation_offset_x;
    float animation_offset_y;
    float flange_size;
    float highlight_flange;
  } config_data;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI settings */
  struct {
    /* Plane settings */
    float flange_log_size;
    bool highlight_flange;
    float animation;
    /* Sampler descriptor */
    int32_t address_mode_u;
    int32_t address_mode_v;
    int32_t mag_filter;
    int32_t min_filter;
    int32_t mipmap_filter;
    float lod_min_clamp;
    float lod_max_clamp;
    int32_t max_anisotropy;
  } settings;

  /* GUI string arrays */
  const char* address_modes_str[ADDRESS_MODE_COUNT];
  const char* filter_modes_str[FILTER_MODE_COUNT];

  /* Matrices for the 15 viewports */
  float matrices[NUM_MATRICES * 16];

  /* Frame timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.2f, 0.2f, 0.2f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  /* Initial settings matching TypeScript version */
  .settings = {
    .flange_log_size   = 1.0f,
    .highlight_flange  = false,
    .animation         = 0.1f,
    .address_mode_u    = ADDRESS_MODE_CLAMP_TO_EDGE,
    .address_mode_v    = ADDRESS_MODE_CLAMP_TO_EDGE,
    .mag_filter        = FILTER_MODE_LINEAR,
    .min_filter        = FILTER_MODE_LINEAR,
    .mipmap_filter     = FILTER_MODE_LINEAR,
    .lod_min_clamp     = 0.0f,
    .lod_max_clamp     = 4.0f,
    .max_anisotropy    = 1,
  },
  .address_modes_str = {
    "clamp-to-edge",
    "repeat",
    "mirror-repeat",
  },
  .filter_modes_str = {
    "nearest",
    "linear",
  },
};

/* -------------------------------------------------------------------------- *
 * Matrix initialization
 * -------------------------------------------------------------------------- */

static void init_matrices(void)
{
  /* Create matrices for different scales and rotations */
  /* Row 1: Scale by 2 */
  /* Row 2: Scale by 1 */
  /* Row 3: Scale by 0.9 */
  /* Row 4: Scale by 0.3 (first 3 columns only) */

  const float scales[4]    = {2.0f, 1.0f, 0.9f, 0.3f};
  const float rot_z_angle  = PI / 16.0f;
  const float rot_x_angle1 = -PI * 0.3f;
  const float rot_x_angle2 = -PI * 0.42f;

  int matrix_idx = 0;
  for (int row = 0; row < 4; ++row) {
    float scale = scales[row];
    int cols    = (row == 3) ? 3 : 4; /* Row 4 only has 3 columns */

    for (int col = 0; col < cols; ++col) {
      mat4 m;
      glm_mat4_identity(m);

      /* Apply rotation based on column */
      switch (col) {
        case 0: /* Rotated Z */
          glm_rotate_z(m, rot_z_angle, m);
          break;
        case 1: /* Identity (head-on) */
          /* No rotation */
          break;
        case 2: /* Perspective (rotated X) */
          glm_rotate_x(m, rot_x_angle1, m);
          break;
        case 3: /* Vanishing perspective */
          glm_rotate_x(m, rot_x_angle2, m);
          break;
      }

      /* Apply scale */
      glm_scale(m, (vec3){scale, scale, 1.0f});

      /* Copy to matrices array */
      memcpy(&state.matrices[matrix_idx * 16], m, sizeof(mat4));
      matrix_idx++;
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Texture initialization
 * -------------------------------------------------------------------------- */

static void init_checkerboard_texture(wgpu_context_t* wgpu_context)
{
  /* Create the checkerboard texture with 4 mip levels */
  WGPUTextureDescriptor tex_desc = {
    .label = STRVIEW("Checkerboard texture"),
    .size  = {
      .width              = TEXTURE_BASE_SIZE,
      .height             = TEXTURE_BASE_SIZE,
      .depthOrArrayLayers = 1,
    },
    .mipLevelCount = TEXTURE_MIP_LEVELS,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
  };
  state.textures.checkerboard
    = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);
  ASSERT(state.textures.checkerboard != NULL);

  /* Create texture view */
  WGPUTextureViewDescriptor view_desc = {
    .label           = STRVIEW("Checkerboard texture view"),
    .format          = WGPUTextureFormat_RGBA8Unorm,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = TEXTURE_MIP_LEVELS,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  state.textures.checkerboard_view
    = wgpuTextureCreateView(state.textures.checkerboard, &view_desc);
  ASSERT(state.textures.checkerboard_view != NULL);

  /* Colors for each mip level:
   * Level 0: white/black (16x16)
   * Level 1: blue/black (8x8)
   * Level 2: yellow/black (4x4)
   * Level 3: pink/black (2x2)
   */
  const uint8_t colors[TEXTURE_MIP_LEVELS][4] = {
    {255, 255, 255, 255}, /* white */
    {30, 136, 229, 255},  /* blue */
    {255, 193, 7, 255},   /* yellow */
    {216, 27, 96, 255},   /* pink */
  };

  /* Fill each mip level */
  for (uint32_t mip_level = 0; mip_level < TEXTURE_MIP_LEVELS; ++mip_level) {
    uint32_t size    = 1 << (TEXTURE_MIP_LEVELS - mip_level); /* 16, 8, 4, 2 */
    size_t data_size = size * size * 4;
    uint8_t* data    = (uint8_t*)malloc(data_size);
    ASSERT(data != NULL);

    for (uint32_t y = 0; y < size; ++y) {
      for (uint32_t x = 0; x < size; ++x) {
        size_t idx = (y * size + x) * 4;
        if ((x + y) % 2 != 0) {
          data[idx + 0] = colors[mip_level][0];
          data[idx + 1] = colors[mip_level][1];
          data[idx + 2] = colors[mip_level][2];
          data[idx + 3] = colors[mip_level][3];
        }
        else {
          data[idx + 0] = 0;
          data[idx + 1] = 0;
          data[idx + 2] = 0;
          data[idx + 3] = 255;
        }
      }
    }

    /* Upload to GPU */
    WGPUTexelCopyTextureInfo dest = {
      .texture  = state.textures.checkerboard,
      .mipLevel = mip_level,
    };
    WGPUTexelCopyBufferLayout layout = {
      .bytesPerRow  = size * 4,
      .rowsPerImage = size,
    };
    WGPUExtent3D extent = {
      .width              = size,
      .height             = size,
      .depthOrArrayLayers = 1,
    };
    wgpuQueueWriteTexture(wgpu_context->queue, &dest, data, data_size, &layout,
                          &extent);

    free(data);
  }
}

static void init_render_target(wgpu_context_t* wgpu_context)
{
  /* Release existing render target */
  if (state.textures.render_target_view != NULL) {
    wgpuTextureViewRelease(state.textures.render_target_view);
    state.textures.render_target_view = NULL;
  }
  if (state.textures.render_target != NULL) {
    wgpuTextureDestroy(state.textures.render_target);
    wgpuTextureRelease(state.textures.render_target);
    state.textures.render_target = NULL;
  }

  /* Create render target texture at fixed low resolution */
  WGPUTextureDescriptor tex_desc = {
    .label = STRVIEW("Render target texture"),
    .size  = {
      .width              = CANVAS_SIZE,
      .height             = CANVAS_SIZE,
      .depthOrArrayLayers = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = wgpu_context->render_format,
    .usage         = WGPUTextureUsage_RenderAttachment
                   | WGPUTextureUsage_TextureBinding,
  };
  state.textures.render_target
    = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);
  ASSERT(state.textures.render_target != NULL);

  /* Create texture view */
  state.textures.render_target_view
    = wgpuTextureCreateView(state.textures.render_target, NULL);
  ASSERT(state.textures.render_target_view != NULL);
}

/* -------------------------------------------------------------------------- *
 * Buffer initialization
 * -------------------------------------------------------------------------- */

static void init_buffers(wgpu_context_t* wgpu_context)
{
  /* Config buffer: viewProj (64 bytes) + animation offset/flange data (16
   * bytes) = 80 bytes, padded to 128 */
  state.buffers.config = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Config uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = 128,
    });
  ASSERT(state.buffers.config != NULL);

  /* Calculate view-projection matrix */
  mat4 proj, translate;
  glm_perspective(2.0f * atanf(1.0f / CAMERA_DIST), 1.0f, 0.1f, 100.0f, proj);
  glm_translate_make(translate, (vec3){0.0f, 0.0f, -CAMERA_DIST});
  glm_mat4_mul(proj, translate, state.view_proj);

  /* Upload view projection matrix */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.config, 0,
                       state.view_proj, sizeof(mat4));

  /* Matrices storage buffer */
  init_matrices();
  state.buffers.matrices = wgpuDeviceCreateBuffer(
    wgpu_context->device, &(WGPUBufferDescriptor){
                            .label = STRVIEW("Matrices storage buffer"),
                            .usage = WGPUBufferUsage_Storage,
                            .size  = sizeof(state.matrices),
                            .mappedAtCreation = true,
                          });
  ASSERT(state.buffers.matrices != NULL);

  /* Copy matrices to mapped buffer */
  void* mapped = wgpuBufferGetMappedRange(state.buffers.matrices, 0,
                                          sizeof(state.matrices));
  memcpy(mapped, state.matrices, sizeof(state.matrices));
  wgpuBufferUnmap(state.buffers.matrices);
}

/* -------------------------------------------------------------------------- *
 * Pipeline initialization
 * -------------------------------------------------------------------------- */

static void init_show_texture_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, show_texture_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Show texture pipeline"),
    .layout = NULL, /* auto layout */
    .vertex = {
      .module     = shader_module,
      .entryPoint = STRVIEW("vmain"),
    },
    .fragment = &(WGPUFragmentState){
      .module      = shader_module,
      .entryPoint  = STRVIEW("fmain"),
      .targetCount = 1,
      .targets     = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology = WGPUPrimitiveTopology_TriangleList,
    },
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.show_texture_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.show_texture_pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);

  /* Create bind group for show texture pipeline */
  WGPUBindGroupLayout bgl
    = wgpuRenderPipelineGetBindGroupLayout(state.show_texture_pipeline, 0);

  state.show_texture_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Show texture bind group"),
      .layout     = bgl,
      .entryCount = 1,
      .entries    = &(WGPUBindGroupEntry){
        .binding     = 0,
        .textureView = state.textures.checkerboard_view,
      },
    });
  ASSERT(state.show_texture_bind_group != NULL);

  wgpuBindGroupLayoutRelease(bgl);
}

static void init_textured_square_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule shader_module = wgpu_create_shader_module(
    wgpu_context->device, textured_square_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Pipeline constants */
  WGPUConstantEntry constants[2] = {
    {
      .key   = STRVIEW("kTextureBaseSize"),
      .value = (double)TEXTURE_BASE_SIZE,
    },
    {
      .key   = STRVIEW("kViewportSize"),
      .value = (double)VIEWPORT_SIZE,
    },
  };

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Textured square pipeline"),
    .layout = NULL, /* auto layout */
    .vertex = {
      .module        = shader_module,
      .entryPoint    = STRVIEW("vmain"),
      .constantCount = 2,
      .constants     = constants,
    },
    .fragment = &(WGPUFragmentState){
      .module      = shader_module,
      .entryPoint  = STRVIEW("fmain"),
      .targetCount = 1,
      .targets     = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology = WGPUPrimitiveTopology_TriangleList,
    },
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.textured_square_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.textured_square_pipeline != NULL);

  /* Get bind group layout for later bind group creation */
  state.textured_square_bind_group_layout
    = wgpuRenderPipelineGetBindGroupLayout(state.textured_square_pipeline, 0);
  ASSERT(state.textured_square_bind_group_layout != NULL);

  wgpuShaderModuleRelease(shader_module);
}

/* -------------------------------------------------------------------------- *
 * Config buffer update
 * -------------------------------------------------------------------------- */

static void update_config_buffer(wgpu_context_t* wgpu_context)
{
  float t = (float)stm_sec(stm_now()) * 0.5f;

  float data[4] = {
    cosf(t) * state.settings.animation,
    sinf(t) * state.settings.animation,
    (powf(2.0f, state.settings.flange_log_size) - 1.0f) / 2.0f,
    state.settings.highlight_flange ? 1.0f : 0.0f,
  };

  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.config, 64, data,
                       sizeof(data));
}

/* -------------------------------------------------------------------------- *
 * Address mode / filter mode conversion
 * -------------------------------------------------------------------------- */

static WGPUAddressMode get_address_mode(int32_t mode)
{
  switch (mode) {
    case ADDRESS_MODE_REPEAT:
      return WGPUAddressMode_Repeat;
    case ADDRESS_MODE_MIRROR_REPEAT:
      return WGPUAddressMode_MirrorRepeat;
    default:
      return WGPUAddressMode_ClampToEdge;
  }
}

static WGPUFilterMode get_filter_mode(int32_t mode)
{
  return mode == FILTER_MODE_LINEAR ? WGPUFilterMode_Linear :
                                      WGPUFilterMode_Nearest;
}

static WGPUMipmapFilterMode get_mipmap_filter_mode(int32_t mode)
{
  return mode == FILTER_MODE_LINEAR ? WGPUMipmapFilterMode_Linear :
                                      WGPUMipmapFilterMode_Nearest;
}

/* -------------------------------------------------------------------------- *
 * GUI rendering
 * -------------------------------------------------------------------------- */

static void reset_to_initial(void)
{
  state.settings.flange_log_size  = 1.0f;
  state.settings.highlight_flange = false;
  state.settings.animation        = 0.1f;
  state.settings.address_mode_u   = ADDRESS_MODE_CLAMP_TO_EDGE;
  state.settings.address_mode_v   = ADDRESS_MODE_CLAMP_TO_EDGE;
  state.settings.mag_filter       = FILTER_MODE_LINEAR;
  state.settings.min_filter       = FILTER_MODE_LINEAR;
  state.settings.mipmap_filter    = FILTER_MODE_LINEAR;
  state.settings.lod_min_clamp    = 0.0f;
  state.settings.lod_max_clamp    = 4.0f;
  state.settings.max_anisotropy   = 1;
}

static void set_checkered_floor(void)
{
  state.settings.flange_log_size = 10.0f;
  state.settings.address_mode_u  = ADDRESS_MODE_REPEAT;
  state.settings.address_mode_v  = ADDRESS_MODE_REPEAT;
}

static void set_smooth(void)
{
  state.settings.mag_filter    = FILTER_MODE_LINEAR;
  state.settings.min_filter    = FILTER_MODE_LINEAR;
  state.settings.mipmap_filter = FILTER_MODE_LINEAR;
}

static void set_crunchy(void)
{
  state.settings.mag_filter    = FILTER_MODE_NEAREST;
  state.settings.min_filter    = FILTER_MODE_NEAREST;
  state.settings.mipmap_filter = FILTER_MODE_NEAREST;
}

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Sampler Parameters", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Presets */
  if (igCollapsingHeaderBoolPtr("Presets", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    if (igButton("reset to initial", (ImVec2){0, 0})) {
      reset_to_initial();
    }
    igSameLine(0.0f, -1.0f);
    if (igButton("checkered floor", (ImVec2){0, 0})) {
      set_checkered_floor();
    }
    if (igButton("smooth (linear)", (ImVec2){0, 0})) {
      set_smooth();
    }
    igSameLine(0.0f, -1.0f);
    if (igButton("crunchy (nearest)", (ImVec2){0, 0})) {
      set_crunchy();
    }
  }

  /* Plane settings */
  if (igCollapsingHeaderBoolPtr("Plane settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_slider_float("size = 2**", &state.settings.flange_log_size,
                               0.0f, 10.0f, "%.1f");
    igCheckbox("highlightFlange", &state.settings.highlight_flange);
    imgui_overlay_slider_float("animation", &state.settings.animation, 0.0f,
                               0.5f, "%.2f");
  }

  /* GPUSamplerDescriptor */
  if (igCollapsingHeaderBoolPtr("GPUSamplerDescriptor", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_combo_box("addressModeU", &state.settings.address_mode_u,
                            state.address_modes_str, ADDRESS_MODE_COUNT);
    imgui_overlay_combo_box("addressModeV", &state.settings.address_mode_v,
                            state.address_modes_str, ADDRESS_MODE_COUNT);
    imgui_overlay_combo_box("magFilter", &state.settings.mag_filter,
                            state.filter_modes_str, FILTER_MODE_COUNT);
    imgui_overlay_combo_box("minFilter", &state.settings.min_filter,
                            state.filter_modes_str, FILTER_MODE_COUNT);
    imgui_overlay_combo_box("mipmapFilter", &state.settings.mipmap_filter,
                            state.filter_modes_str, FILTER_MODE_COUNT);

    /* LOD clamp with interaction */
    if (imgui_overlay_slider_float("lodMinClamp", &state.settings.lod_min_clamp,
                                   0.0f, 4.0f, "%.1f")) {
      if (state.settings.lod_max_clamp < state.settings.lod_min_clamp) {
        state.settings.lod_max_clamp = state.settings.lod_min_clamp;
      }
    }
    if (imgui_overlay_slider_float("lodMaxClamp", &state.settings.lod_max_clamp,
                                   0.0f, 4.0f, "%.1f")) {
      if (state.settings.lod_min_clamp > state.settings.lod_max_clamp) {
        state.settings.lod_min_clamp = state.settings.lod_max_clamp;
      }
    }

    /* maxAnisotropy */
    if (igCollapsingHeaderBoolPtr("maxAnisotropy (set only if all \"linear\")",
                                  NULL, ImGuiTreeNodeFlags_DefaultOpen)) {
      imgui_overlay_slider_int("maxAnisotropy", &state.settings.max_anisotropy,
                               1, 16);
    }
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_render_target(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_checkerboard_texture(wgpu_context);
    init_render_target(wgpu_context);
    init_buffers(wgpu_context);
    init_show_texture_pipeline(wgpu_context);
    init_textured_square_pipeline(wgpu_context);
    imgui_overlay_init(wgpu_context);
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

  /* Update config buffer */
  update_config_buffer(wgpu_context);

  /* Calculate delta time for ImGui */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);

  /* Render GUI controls */
  render_gui(wgpu_context);

  /* Create sampler based on current settings */
  /* Only use anisotropy if all filters are linear */
  uint16_t anisotropy = 1;
  if (state.settings.min_filter == FILTER_MODE_LINEAR
      && state.settings.mag_filter == FILTER_MODE_LINEAR
      && state.settings.mipmap_filter == FILTER_MODE_LINEAR) {
    anisotropy = (uint16_t)state.settings.max_anisotropy;
  }

  /* Release previous sampler */
  if (state.current_sampler != NULL) {
    wgpuSamplerRelease(state.current_sampler);
    state.current_sampler = NULL;
  }

  state.current_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device,
    &(WGPUSamplerDescriptor){
      .label         = STRVIEW("Current sampler"),
      .addressModeU  = get_address_mode(state.settings.address_mode_u),
      .addressModeV  = get_address_mode(state.settings.address_mode_v),
      .addressModeW  = WGPUAddressMode_ClampToEdge,
      .magFilter     = get_filter_mode(state.settings.mag_filter),
      .minFilter     = get_filter_mode(state.settings.min_filter),
      .mipmapFilter  = get_mipmap_filter_mode(state.settings.mipmap_filter),
      .lodMinClamp   = state.settings.lod_min_clamp,
      .lodMaxClamp   = state.settings.lod_max_clamp,
      .maxAnisotropy = anisotropy,
    });
  ASSERT(state.current_sampler != NULL);

  /* Create bind group for textured square */
  WGPUBindGroupEntry bg_entries[4] = {
    {.binding = 0, .buffer = state.buffers.config, .size = 128},
    {.binding = 1,
     .buffer  = state.buffers.matrices,
     .size    = sizeof(state.matrices)},
    {.binding = 2, .sampler = state.current_sampler},
    {.binding = 3, .textureView = state.textures.checkerboard_view},
  };
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label  = STRVIEW("Textured square bind group"),
                            .layout = state.textured_square_bind_group_layout,
                            .entryCount = 4,
                            .entries    = bg_entries,
                          });
  ASSERT(bind_group != NULL);

  /* Begin render pass to render target */
  state.color_attachment.view = state.textures.render_target_view;

  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Draw test squares (15 viewports) */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.textured_square_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, bind_group, 0, 0);

  for (uint32_t i = 0; i < VIEWPORT_GRID_SIZE * VIEWPORT_GRID_SIZE - 1; ++i) {
    float vp_x = (float)(VIEWPORT_GRID_STRIDE * (i % VIEWPORT_GRID_SIZE) + 1);
    float vp_y = (float)(VIEWPORT_GRID_STRIDE * (i / VIEWPORT_GRID_SIZE) + 1);
    wgpuRenderPassEncoderSetViewport(rpass_enc, vp_x, vp_y,
                                     (float)VIEWPORT_SIZE, (float)VIEWPORT_SIZE,
                                     0.0f, 1.0f);
    wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, i);
  }

  /* Show texture contents in bottom-right viewport */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.show_texture_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.show_texture_bind_group,
                                    0, 0);

  float last_vp = (float)((VIEWPORT_GRID_SIZE - 1) * VIEWPORT_GRID_STRIDE + 1);

  /* Level 0: 16x16 at position (0, 0) in the last viewport */
  wgpuRenderPassEncoderSetViewport(rpass_enc, last_vp, last_vp, 32.0f, 32.0f,
                                   0.0f, 1.0f);
  wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 0);

  /* Level 1: 8x8 at position (32, 0) */
  wgpuRenderPassEncoderSetViewport(rpass_enc, last_vp + 32.0f, last_vp, 16.0f,
                                   16.0f, 0.0f, 1.0f);
  wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 1);

  /* Level 2: 4x4 at position (32, 16) */
  wgpuRenderPassEncoderSetViewport(rpass_enc, last_vp + 32.0f, last_vp + 16.0f,
                                   8.0f, 8.0f, 0.0f, 1.0f);
  wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 2);

  /* Level 3: 2x2 at position (32, 24) */
  wgpuRenderPassEncoderSetViewport(rpass_enc, last_vp + 32.0f, last_vp + 24.0f,
                                   4.0f, 4.0f, 0.0f, 1.0f);
  wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 3);

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  /* Cleanup render pass resources */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);
  wgpuBindGroupRelease(bind_group);

  /* Now blit the render target to the swapchain at low resolution */
  /* For the pixelated effect, we render a fullscreen quad with the render
   * target texture, but since the render target is low-res (200x200),
   * it will appear pixelated when stretched to the window size */

  /* Create a second render pass to blit to swapchain */
  WGPURenderPassColorAttachment blit_color_attachment = {
    .view       = wgpu_context->swapchain_view,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.2f, 0.2f, 0.2f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  };
  WGPURenderPassDescriptor blit_render_pass_desc = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &blit_color_attachment,
  };

  /* Create a simple blit pass - for now just clear and let ImGui render */
  /* The low-res render target content is already displayed via ImGui or
   * could be blitted with a fullscreen shader */
  WGPUCommandEncoder blit_cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder blit_rpass_enc
    = wgpuCommandEncoderBeginRenderPass(blit_cmd_enc, &blit_render_pass_desc);

  /* Copy the low-res render to the screen by drawing it as a textured quad */
  /* For simplicity, we'll render the same scene directly to swapchain */
  /* Recreate bind group for swapchain rendering */
  WGPUBindGroup swapchain_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label  = STRVIEW("Swapchain bind group"),
                            .layout = state.textured_square_bind_group_layout,
                            .entryCount = 4,
                            .entries    = bg_entries,
                          });

  /* Draw test squares to swapchain */
  wgpuRenderPassEncoderSetPipeline(blit_rpass_enc,
                                   state.textured_square_pipeline);
  wgpuRenderPassEncoderSetBindGroup(blit_rpass_enc, 0, swapchain_bind_group, 0,
                                    0);

  /* Calculate scale factor for pixelated rendering */
  float scale_x = (float)wgpu_context->width / (float)CANVAS_SIZE;
  float scale_y = (float)wgpu_context->height / (float)CANVAS_SIZE;
  float scale   = MIN(scale_x, scale_y);

  /* Center the rendering */
  float offset_x = ((float)wgpu_context->width - CANVAS_SIZE * scale) / 2.0f;
  float offset_y = ((float)wgpu_context->height - CANVAS_SIZE * scale) / 2.0f;

  for (uint32_t i = 0; i < VIEWPORT_GRID_SIZE * VIEWPORT_GRID_SIZE - 1; ++i) {
    float vp_x
      = offset_x
        + scale * (float)(VIEWPORT_GRID_STRIDE * (i % VIEWPORT_GRID_SIZE) + 1);
    float vp_y
      = offset_y
        + scale * (float)(VIEWPORT_GRID_STRIDE * (i / VIEWPORT_GRID_SIZE) + 1);
    wgpuRenderPassEncoderSetViewport(blit_rpass_enc, vp_x, vp_y,
                                     (float)VIEWPORT_SIZE * scale,
                                     (float)VIEWPORT_SIZE * scale, 0.0f, 1.0f);
    wgpuRenderPassEncoderDraw(blit_rpass_enc, 6, 1, 0, i);
  }

  /* Show texture contents in bottom-right viewport */
  wgpuRenderPassEncoderSetPipeline(blit_rpass_enc, state.show_texture_pipeline);
  wgpuRenderPassEncoderSetBindGroup(blit_rpass_enc, 0,
                                    state.show_texture_bind_group, 0, 0);

  float scaled_last_vp
    = offset_y
      + scale * (float)((VIEWPORT_GRID_SIZE - 1) * VIEWPORT_GRID_STRIDE + 1);
  float scaled_last_vp_x
    = offset_x
      + scale * (float)((VIEWPORT_GRID_SIZE - 1) * VIEWPORT_GRID_STRIDE + 1);

  wgpuRenderPassEncoderSetViewport(blit_rpass_enc, scaled_last_vp_x,
                                   scaled_last_vp, 32.0f * scale, 32.0f * scale,
                                   0.0f, 1.0f);
  wgpuRenderPassEncoderDraw(blit_rpass_enc, 6, 1, 0, 0);

  wgpuRenderPassEncoderSetViewport(
    blit_rpass_enc, scaled_last_vp_x + 32.0f * scale, scaled_last_vp,
    16.0f * scale, 16.0f * scale, 0.0f, 1.0f);
  wgpuRenderPassEncoderDraw(blit_rpass_enc, 6, 1, 0, 1);

  wgpuRenderPassEncoderSetViewport(
    blit_rpass_enc, scaled_last_vp_x + 32.0f * scale,
    scaled_last_vp + 16.0f * scale, 8.0f * scale, 8.0f * scale, 0.0f, 1.0f);
  wgpuRenderPassEncoderDraw(blit_rpass_enc, 6, 1, 0, 2);

  wgpuRenderPassEncoderSetViewport(
    blit_rpass_enc, scaled_last_vp_x + 32.0f * scale,
    scaled_last_vp + 24.0f * scale, 4.0f * scale, 4.0f * scale, 0.0f, 1.0f);
  wgpuRenderPassEncoderDraw(blit_rpass_enc, 6, 1, 0, 3);

  wgpuRenderPassEncoderEnd(blit_rpass_enc);
  WGPUCommandBuffer blit_cmd_buffer
    = wgpuCommandEncoderFinish(blit_cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &blit_cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(blit_rpass_enc);
  wgpuCommandBufferRelease(blit_cmd_buffer);
  wgpuCommandEncoderRelease(blit_cmd_enc);
  wgpuBindGroupRelease(swapchain_bind_group);

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  WGPU_RELEASE_RESOURCE(Sampler, state.current_sampler)
  WGPU_RELEASE_RESOURCE(BindGroup, state.show_texture_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.textured_square_bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.show_texture_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.textured_square_pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.config)
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.matrices)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.checkerboard_view)
  WGPU_RELEASE_RESOURCE(Texture, state.textures.checkerboard)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.render_target_view)
  if (state.textures.render_target != NULL) {
    wgpuTextureDestroy(state.textures.render_target);
    wgpuTextureRelease(state.textures.render_target);
    state.textures.render_target = NULL;
  }
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Sampler Parameters",
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
static const char* textured_square_shader_wgsl = CODE(
  struct Config {
    viewProj: mat4x4f,
    animationOffset: vec2f,
    flangeSize: f32,
    highlightFlange: f32,
  };

  @group(0) @binding(0) var<uniform> config: Config;
  @group(0) @binding(1) var<storage, read> matrices: array<mat4x4f>;
  @group(0) @binding(2) var samp: sampler;
  @group(0) @binding(3) var tex: texture_2d<f32>;

  struct Varying {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
  }

  override kTextureBaseSize: f32;
  override kViewportSize: f32;

  @vertex
  fn vmain(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
  ) -> Varying {
    let flange = config.flangeSize;
    var uvs = array(
      vec2(-flange, -flange), vec2(-flange, 1 + flange), vec2(1 + flange, -flange),
      vec2(1 + flange, -flange), vec2(-flange, 1 + flange), vec2(1 + flange, 1 + flange),
    );
    // Default size (if matrix is the identity) makes 1 texel = 1 pixel.
    let radius = (1 + 2 * flange) * kTextureBaseSize / kViewportSize;
    var positions = array(
      vec2(-radius, -radius), vec2(-radius, radius), vec2(radius, -radius),
      vec2(radius, -radius), vec2(-radius, radius), vec2(radius, radius),
    );

    let modelMatrix = matrices[instance_index];
    let pos = config.viewProj * modelMatrix * vec4f(positions[vertex_index] + config.animationOffset, 0, 1);
    return Varying(pos, uvs[vertex_index]);
  }

  @fragment
  fn fmain(vary: Varying) -> @location(0) vec4f {
    let uv = vary.uv;
    var color = textureSample(tex, samp, uv);

    let outOfBounds = uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1;
    if config.highlightFlange > 0 && outOfBounds {
      color += vec4(0.7, 0, 0, 0);
    }

    return color;
  }
);

static const char* show_texture_shader_wgsl = CODE(
  @group(0) @binding(0) var tex: texture_2d<f32>;

  struct Varying {
    @builtin(position) pos: vec4f,
    @location(0) texelCoord: vec2f,
    @location(1) mipLevel: f32,
  }

  const kMipLevels = 4;
  const baseMipSize: u32 = 16;

  @vertex
  fn vmain(
    @builtin(instance_index) instance_index: u32, // used as mipLevel
    @builtin(vertex_index) vertex_index: u32,
  ) -> Varying {
    var square = array(
      vec2f(0, 0), vec2f(0, 1), vec2f(1, 0),
      vec2f(1, 0), vec2f(0, 1), vec2f(1, 1),
    );
    let uv = square[vertex_index];
    let pos = vec4(uv * 2 - vec2(1, 1), 0.0, 1.0);

    let mipLevel = instance_index;
    let mipSize = f32(1 << (kMipLevels - mipLevel));
    let texelCoord = uv * mipSize;
    return Varying(pos, texelCoord, f32(mipLevel));
  }

  @fragment
  fn fmain(vary: Varying) -> @location(0) vec4f {
    return textureLoad(tex, vec2u(vary.texelCoord), u32(vary.mipLevel));
  }
);
// clang-format on
