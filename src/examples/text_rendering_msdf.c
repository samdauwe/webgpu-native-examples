#include "common_shaders.h"
#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cJSON.h>
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
 * WebGPU Example - Text Rendering MSDF
 *
 * This example demonstrates text rendering using Multichannel Signed Distance
 * Fields (MSDF). MSDF allows for high-quality, resolution-independent text
 * rendering with smooth edges at any scale.
 *
 * The example renders:
 * - A rotating cube with face labels ("Front", "Back", "Left", "Right",
 *   "Top", "Bottom")
 * - A scrolling text crawl displaying information about WebGPU
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/textRenderingMsdf
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* msdf_text_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define MAX_TEXT_CHARS 2048
#define MAX_FORMATTED_TEXTS 16
#define MAX_KERNINGS 512
#define MAX_FONT_CHARS 128
#define FONT_JSON_BUFFER_SIZE (32 * 1024)
#define FONT_TEXTURE_SIZE (512 * 512 * 4)

/* -------------------------------------------------------------------------- *
 * MSDF Font Character structure
 * -------------------------------------------------------------------------- */

typedef struct msdf_char_t {
  int32_t id;
  int32_t index;
  float width;
  float height;
  float xoffset;
  float yoffset;
  float xadvance;
  float x;
  float y;
  int32_t page;
  int32_t char_index;
} msdf_char_t;

/* -------------------------------------------------------------------------- *
 * MSDF Font Kerning structure
 * -------------------------------------------------------------------------- */

typedef struct msdf_kerning_t {
  int32_t first;
  int32_t second;
  int32_t amount;
} msdf_kerning_t;

/* -------------------------------------------------------------------------- *
 * MSDF Font structure
 * -------------------------------------------------------------------------- */

typedef struct msdf_font_t {
  WGPURenderPipeline pipeline;
  WGPUBindGroup bind_group;
  float line_height;
  msdf_char_t chars[MAX_FONT_CHARS];
  int32_t char_count;
  msdf_kerning_t kernings[MAX_KERNINGS];
  int32_t kerning_count;
  msdf_char_t default_char;
  float scale_w;
  float scale_h;
} msdf_font_t;

/* -------------------------------------------------------------------------- *
 * MSDF Text Measurements structure
 * -------------------------------------------------------------------------- */

typedef struct msdf_text_measurements_t {
  float width;
  float height;
  float line_widths[64];
  int32_t line_count;
  int32_t printed_char_count;
} msdf_text_measurements_t;

/* -------------------------------------------------------------------------- *
 * MSDF Formatted Text structure
 * -------------------------------------------------------------------------- */

typedef struct msdf_text_t {
  WGPURenderBundle render_bundle;
  msdf_text_measurements_t measurements;
  WGPUBuffer text_buffer;
  float buffer_array[24]; /* 16 for transform + 4 for color + 1 for scale + 3
                             padding */
  bool buffer_dirty;
} msdf_text_t;

/* -------------------------------------------------------------------------- *
 * Text formatting options
 * -------------------------------------------------------------------------- */

typedef struct msdf_text_format_options_t {
  bool centered;
  float pixel_scale;
  float color[4];
  bool has_color;
} msdf_text_format_options_t;

/* -------------------------------------------------------------------------- *
 * State struct
 * -------------------------------------------------------------------------- */

static struct {
  /* WebGPU context pointer for callbacks */
  wgpu_context_t* wgpu_context;

  /* Cube geometry */
  cube_mesh_t cube_mesh;
  wgpu_buffer_t cube_vertices;

  /* Cube rendering */
  WGPURenderPipeline cube_pipeline;
  WGPUBuffer cube_uniform_buffer;
  WGPUBindGroup cube_bind_group;

  /* MSDF text renderer */
  struct {
    WGPUBindGroupLayout font_bind_group_layout;
    WGPUBindGroupLayout text_bind_group_layout;
    WGPURenderPipeline pipeline;
    WGPUSampler sampler;
    WGPUBuffer camera_uniform_buffer;
    float camera_array[32]; /* projection + view matrices */
    WGPURenderBundleEncoderDescriptor render_bundle_desc;
    WGPUTextureFormat color_formats[1];
  } text_renderer;

  /* Font data */
  struct {
    msdf_font_t font;
    wgpu_texture_t texture;
    WGPUBuffer chars_buffer;
    bool loaded;
  } font;

  /* Formatted texts */
  msdf_text_t texts[MAX_FORMATTED_TEXTS];
  int32_t text_count;

  /* Text transforms for cube faces */
  mat4 text_transforms[6];

  /* Indices into texts array */
  int32_t title_text_idx;
  int32_t large_text_idx;

  /* View matrices */
  struct {
    mat4 projection;
    mat4 view;
    mat4 model;
    mat4 model_view_projection;
    mat4 text_matrix;
  } view_matrices;

  /* Async loading */
  uint8_t font_json_buffer[FONT_JSON_BUFFER_SIZE];
  uint8_t font_texture_buffer[FONT_TEXTURE_SIZE];

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Timing */
  uint64_t start_time;

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
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
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .title_text_idx = -1,
  .large_text_idx = -1,
};

/* -------------------------------------------------------------------------- *
 * MSDF Font helper functions
 * -------------------------------------------------------------------------- */

static msdf_char_t* font_get_char(msdf_font_t* font, int32_t char_code)
{
  for (int32_t i = 0; i < font->char_count; ++i) {
    if (font->chars[i].id == char_code) {
      return &font->chars[i];
    }
  }
  return &font->default_char;
}

static float font_get_x_advance(msdf_font_t* font, int32_t char_code,
                                int32_t next_char_code)
{
  msdf_char_t* c = font_get_char(font, char_code);
  float advance  = c->xadvance;

  if (next_char_code >= 0) {
    for (int32_t i = 0; i < font->kerning_count; ++i) {
      if (font->kernings[i].first == char_code
          && font->kernings[i].second == next_char_code) {
        advance += (float)font->kernings[i].amount;
        break;
      }
    }
  }

  return advance;
}

/* -------------------------------------------------------------------------- *
 * MSDF Text helper functions
 * -------------------------------------------------------------------------- */

static void msdf_text_set_transform(msdf_text_t* text, mat4 matrix)
{
  memcpy(text->buffer_array, matrix, sizeof(mat4));
  text->buffer_dirty = true;
}

static void msdf_text_set_color(msdf_text_t* text, float r, float g, float b,
                                float a)
{
  text->buffer_array[16] = r;
  text->buffer_array[17] = g;
  text->buffer_array[18] = b;
  text->buffer_array[19] = a;
  text->buffer_dirty     = true;
}

static void msdf_text_set_pixel_scale(msdf_text_t* text, float pixel_scale)
{
  text->buffer_array[20] = pixel_scale;
  text->buffer_dirty     = true;
}

static WGPURenderBundle
msdf_text_get_render_bundle(wgpu_context_t* wgpu_context, msdf_text_t* text)
{
  if (text->buffer_dirty) {
    text->buffer_dirty = false;
    wgpuQueueWriteBuffer(wgpu_context->queue, text->text_buffer, 0,
                         text->buffer_array, sizeof(text->buffer_array));
  }
  return text->render_bundle;
}

/* -------------------------------------------------------------------------- *
 * Measure text function
 * -------------------------------------------------------------------------- */

typedef void (*char_callback_fn)(float x, float y, int32_t line, msdf_char_t* c,
                                 void* user_data);

static msdf_text_measurements_t measure_text(msdf_font_t* font,
                                             const char* text,
                                             char_callback_fn callback,
                                             void* user_data)
{
  msdf_text_measurements_t measurements = {0};
  float max_width                       = 0.0f;
  float text_offset_x                   = 0.0f;
  float text_offset_y                   = 0.0f;
  int32_t line                          = 0;
  int32_t printed_char_count            = 0;

  size_t len             = strlen(text);
  int32_t next_char_code = (len > 0) ? (int32_t)(unsigned char)text[0] : -1;

  for (size_t i = 0; i < len; ++i) {
    int32_t char_code = next_char_code;
    next_char_code = (i < len - 1) ? (int32_t)(unsigned char)text[i + 1] : -1;

    switch (char_code) {
      case 10: /* Newline */
        if (measurements.line_count < 64) {
          measurements.line_widths[measurements.line_count++] = text_offset_x;
        }
        line++;
        if (text_offset_x > max_width) {
          max_width = text_offset_x;
        }
        text_offset_x = 0.0f;
        text_offset_y -= font->line_height;
        break;
      case 13: /* CR */
        break;
      case 32: /* Space */
        text_offset_x += font_get_x_advance(font, char_code, -1);
        break;
      default: {
        msdf_char_t* c = font_get_char(font, char_code);
        if (callback) {
          callback(text_offset_x, text_offset_y, line, c, user_data);
        }
        text_offset_x += font_get_x_advance(font, char_code, next_char_code);
        printed_char_count++;
        break;
      }
    }
  }

  if (measurements.line_count < 64) {
    measurements.line_widths[measurements.line_count++] = text_offset_x;
  }
  if (text_offset_x > max_width) {
    max_width = text_offset_x;
  }

  measurements.width  = max_width;
  measurements.height = (float)measurements.line_count * font->line_height;
  measurements.printed_char_count = printed_char_count;

  return measurements;
}

/* -------------------------------------------------------------------------- *
 * Format text callback data
 * -------------------------------------------------------------------------- */

typedef struct format_text_callback_data_t {
  float* text_array;
  int32_t offset;
  msdf_text_measurements_t* measurements;
  bool centered;
} format_text_callback_data_t;

static void format_text_callback(float x, float y, int32_t line, msdf_char_t* c,
                                 void* user_data)
{
  format_text_callback_data_t* data = (format_text_callback_data_t*)user_data;

  float actual_x = x;
  float actual_y = y;

  if (data->centered && data->measurements) {
    float line_offset
      = data->measurements->width * -0.5f
        - (data->measurements->width - data->measurements->line_widths[line])
            * -0.5f;
    actual_x = x + line_offset;
    actual_y = y + data->measurements->height * 0.5f;
  }

  data->text_array[data->offset]     = actual_x;
  data->text_array[data->offset + 1] = actual_y;
  data->text_array[data->offset + 2] = (float)c->char_index;
  data->text_array[data->offset + 3] = 0.0f; /* padding */
  data->offset += 4;
}

/* -------------------------------------------------------------------------- *
 * Format text function
 * -------------------------------------------------------------------------- */

static int32_t format_text(wgpu_context_t* wgpu_context, msdf_font_t* font,
                           const char* text,
                           msdf_text_format_options_t* options)
{
  if (state.text_count >= MAX_FORMATTED_TEXTS) {
    return -1;
  }

  size_t text_len    = strlen(text);
  size_t buffer_size = (text_len + 6) * sizeof(float) * 4;
  float* text_array  = malloc(buffer_size);
  if (!text_array) {
    return -1;
  }
  memset(text_array, 0, buffer_size);

  /* Initialize text uniform data (first 24 floats) */
  /* Identity transform matrix (16 floats) */
  glm_mat4_identity((vec4*)text_array);

  /* Color (4 floats) */
  if (options && options->has_color) {
    text_array[16] = options->color[0];
    text_array[17] = options->color[1];
    text_array[18] = options->color[2];
    text_array[19] = options->color[3];
  }
  else {
    text_array[16] = 1.0f;
    text_array[17] = 1.0f;
    text_array[18] = 1.0f;
    text_array[19] = 1.0f;
  }

  /* Pixel scale */
  float pixel_scale = (options && options->pixel_scale > 0.0f) ?
                        options->pixel_scale :
                        (1.0f / 512.0f);
  text_array[20]    = pixel_scale;
  text_array[21]    = 0.0f; /* padding */
  text_array[22]    = 0.0f; /* padding */
  text_array[23]    = 0.0f; /* padding */

  /* Measure and format text */
  msdf_text_measurements_t measurements;
  format_text_callback_data_t cb_data = {
    .text_array   = text_array,
    .offset       = 24,
    .measurements = NULL,
    .centered     = options && options->centered,
  };

  if (options && options->centered) {
    /* First pass: measure */
    measurements         = measure_text(font, text, NULL, NULL);
    cb_data.measurements = &measurements;
    /* Second pass: format with centering */
    measure_text(font, text, format_text_callback, &cb_data);
  }
  else {
    measurements = measure_text(font, text, format_text_callback, &cb_data);
  }

  /* Create text buffer */
  WGPUBuffer text_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("MSDF text buffer"),
      .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
      .size             = buffer_size,
      .mappedAtCreation = false,
    });

  /* Write initial data */
  wgpuQueueWriteBuffer(wgpu_context->queue, text_buffer, 0, text_array,
                       buffer_size);

  /* Create bind group */
  WGPUBindGroup text_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("MSDF text bind group"),
      .layout     = state.text_renderer.text_bind_group_layout,
      .entryCount = 2,
      .entries = (WGPUBindGroupEntry[]){
        {
          .binding = 0,
          .buffer  = state.text_renderer.camera_uniform_buffer,
          .offset  = 0,
          .size    = sizeof(state.text_renderer.camera_array),
        },
        {
          .binding = 1,
          .buffer  = text_buffer,
          .offset  = 0,
          .size    = buffer_size,
        },
      },
    });

  /* Create render bundle */
  WGPURenderBundleEncoder encoder = wgpuDeviceCreateRenderBundleEncoder(
    wgpu_context->device,
    &(WGPURenderBundleEncoderDescriptor){
      .label              = STRVIEW("MSDF text render bundle encoder"),
      .colorFormatCount   = 1,
      .colorFormats       = state.text_renderer.color_formats,
      .depthStencilFormat = wgpu_context->depth_stencil_format,
      .sampleCount        = 1,
    });

  wgpuRenderBundleEncoderSetPipeline(encoder, state.text_renderer.pipeline);
  wgpuRenderBundleEncoderSetBindGroup(encoder, 0, state.font.font.bind_group, 0,
                                      NULL);
  wgpuRenderBundleEncoderSetBindGroup(encoder, 1, text_bind_group, 0, NULL);
  wgpuRenderBundleEncoderDraw(encoder, 4, measurements.printed_char_count, 0,
                              0);

  WGPURenderBundle render_bundle = wgpuRenderBundleEncoderFinish(encoder, NULL);

  wgpuRenderBundleEncoderRelease(encoder);
  wgpuBindGroupRelease(text_bind_group);

  /* Store in state */
  int32_t idx            = state.text_count++;
  msdf_text_t* msdf_text = &state.texts[idx];

  msdf_text->render_bundle = render_bundle;
  msdf_text->measurements  = measurements;
  msdf_text->text_buffer   = text_buffer;
  msdf_text->buffer_dirty  = false;

  /* Copy initial values to buffer_array */
  memcpy(msdf_text->buffer_array, text_array, sizeof(msdf_text->buffer_array));

  free(text_array);

  return idx;
}

/* -------------------------------------------------------------------------- *
 * Initialize cube mesh
 * -------------------------------------------------------------------------- */

static void init_cube_mesh(void)
{
  cube_mesh_init(&state.cube_mesh);
}

/* -------------------------------------------------------------------------- *
 * Initialize cube vertex buffer
 * -------------------------------------------------------------------------- */

static void init_cube_vertex_buffer(wgpu_context_t* wgpu_context)
{
  state.cube_vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Vertices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(state.cube_mesh.vertex_array),
                    .initial.data = state.cube_mesh.vertex_array,
                  });
}

/* -------------------------------------------------------------------------- *
 * Initialize cube pipeline
 * -------------------------------------------------------------------------- */

static void init_cube_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, basic_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, vertex_position_color_fragment_shader_wgsl);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(cube, state.cube_mesh.vertex_size,
                            /* Attribute location 0: Position */
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                                               state.cube_mesh.position_offset),
                            /* Attribute location 1: UV */
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2,
                                               state.cube_mesh.uv_offset))

  state.cube_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Cube - Render pipeline"),
      .layout = NULL, /* auto layout */
      .vertex = {
        .module      = vert_shader_module,
        .entryPoint  = STRVIEW("main"),
        .bufferCount = 1,
        .buffers     = &cube_vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = frag_shader_module,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_Back,
        .frontFace = WGPUFrontFace_CCW,
      },
      .depthStencil = &depth_stencil_state,
      .multisample = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

/* -------------------------------------------------------------------------- *
 * Initialize cube uniform buffer
 * -------------------------------------------------------------------------- */

static void init_cube_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.cube_uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Cube - Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4),
    });

  state.cube_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Cube - Bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.cube_pipeline, 0),
      .entryCount = 1,
      .entries = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.cube_uniform_buffer,
        .offset  = 0,
        .size    = sizeof(mat4),
      },
    });
}

/* -------------------------------------------------------------------------- *
 * Initialize text renderer
 * -------------------------------------------------------------------------- */

static void init_text_renderer(wgpu_context_t* wgpu_context)
{
  /* Store color format for render bundles */
  state.text_renderer.color_formats[0] = wgpu_context->render_format;

  /* Create sampler */
  state.text_renderer.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("MSDF text sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .maxAnisotropy = 16,
                          });

  /* Create camera uniform buffer */
  state.text_renderer.camera_uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("MSDF camera uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.text_renderer.camera_array),
    });

  /* Create font bind group layout */
  state.text_renderer.font_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("MSDF font group layout"),
      .entryCount = 3,
      .entries = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Fragment,
          .texture = {
            .sampleType    = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
            .multisampled  = false,
          },
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Fragment,
          .sampler = {
            .type = WGPUSamplerBindingType_Filtering,
          },
        },
        {
          .binding    = 2,
          .visibility = WGPUShaderStage_Vertex,
          .buffer = {
            .type           = WGPUBufferBindingType_ReadOnlyStorage,
            .minBindingSize = 0,
          },
        },
      },
    });

  /* Create text bind group layout */
  state.text_renderer.text_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("MSDF text group layout"),
      .entryCount = 2,
      .entries = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Vertex,
          .buffer = {
            .type           = WGPUBufferBindingType_Uniform,
            .minBindingSize = sizeof(state.text_renderer.camera_array),
          },
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
          .buffer = {
            .type           = WGPUBufferBindingType_ReadOnlyStorage,
            .minBindingSize = 0,
          },
        },
      },
    });

  /* Create pipeline layout */
  WGPUBindGroupLayout bind_group_layouts[2] = {
    state.text_renderer.font_bind_group_layout,
    state.text_renderer.text_bind_group_layout,
  };

  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("MSDF text pipeline layout"),
                            .bindGroupLayoutCount = 2,
                            .bindGroupLayouts     = bind_group_layouts,
                          });

  /* Create shader module */
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, msdf_text_shader_wgsl);

  /* Create render pipeline */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = false,
    });

  state.text_renderer.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("MSDF text pipeline"),
      .layout = pipeline_layout,
      .vertex = {
        .module     = shader_module,
        .entryPoint = STRVIEW("vertexMain"),
      },
      .fragment = &(WGPUFragmentState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = wgpu_context->render_format,
          .blend = &(WGPUBlendState){
            .color = {
              .srcFactor = WGPUBlendFactor_SrcAlpha,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One,
              .operation = WGPUBlendOperation_Add,
            },
          },
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology         = WGPUPrimitiveTopology_TriangleStrip,
        .stripIndexFormat = WGPUIndexFormat_Uint32,
      },
      .depthStencil = &depth_stencil_state,
      .multisample = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });

  wgpuShaderModuleRelease(shader_module);
  wgpuPipelineLayoutRelease(pipeline_layout);
}

/* -------------------------------------------------------------------------- *
 * Font texture fetch callback
 * -------------------------------------------------------------------------- */

static void font_texture_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Font texture fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);

  if (pixels) {
    wgpu_texture_t* texture = &state.font.texture;
    texture->desc = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = img_width,
        .height             = img_height,
        .depthOrArrayLayers = 1,
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

/* -------------------------------------------------------------------------- *
 * Initialize font bind group (called after texture loads)
 * -------------------------------------------------------------------------- */

static void init_font_bind_group(wgpu_context_t* wgpu_context)
{
  /* Release old bind group if exists */
  WGPU_RELEASE_RESOURCE(BindGroup, state.font.font.bind_group)

  state.font.font.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("MSDF font bind group"),
      .layout     = state.text_renderer.font_bind_group_layout,
      .entryCount = 3,
      .entries = (WGPUBindGroupEntry[]){
        {
          .binding     = 0,
          .textureView = state.font.texture.view,
        },
        {
          .binding = 1,
          .sampler = state.text_renderer.sampler,
        },
        {
          .binding = 2,
          .buffer  = state.font.chars_buffer,
          .offset  = 0,
          .size    = state.font.font.char_count * 8 * sizeof(float),
        },
      },
    });
}

/* -------------------------------------------------------------------------- *
 * Font JSON fetch callback
 * -------------------------------------------------------------------------- */

static void font_json_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Font JSON fetch failed, error: %d\n", response->error_code);
    return;
  }

  wgpu_context_t* wgpu_context = state.wgpu_context;

  /* Parse JSON */
  cJSON* json = cJSON_ParseWithLength((const char*)response->data.ptr,
                                      response->data.size);
  if (!json) {
    printf("Failed to parse font JSON\n");
    return;
  }

  /* Parse common info */
  cJSON* common = cJSON_GetObjectItem(json, "common");
  if (common) {
    state.font.font.line_height
      = (float)cJSON_GetObjectItem(common, "lineHeight")->valuedouble;
    state.font.font.scale_w
      = (float)cJSON_GetObjectItem(common, "scaleW")->valuedouble;
    state.font.font.scale_h
      = (float)cJSON_GetObjectItem(common, "scaleH")->valuedouble;
  }

  float u = 1.0f / state.font.font.scale_w;
  float v = 1.0f / state.font.font.scale_h;

  /* Parse characters */
  cJSON* chars = cJSON_GetObjectItem(json, "chars");
  if (chars) {
    int char_count = cJSON_GetArraySize(chars);
    state.font.font.char_count
      = (char_count > MAX_FONT_CHARS) ? MAX_FONT_CHARS : char_count;

    /* Create chars buffer for GPU */
    float* chars_array = malloc(state.font.font.char_count * 8 * sizeof(float));

    int array_offset = 0;
    for (int i = 0; i < state.font.font.char_count; ++i) {
      cJSON* c = cJSON_GetArrayItem(chars, i);

      msdf_char_t* fc = &state.font.font.chars[i];
      fc->id          = cJSON_GetObjectItem(c, "id")->valueint;
      fc->index       = cJSON_GetObjectItem(c, "index")->valueint;
      fc->width       = (float)cJSON_GetObjectItem(c, "width")->valuedouble;
      fc->height      = (float)cJSON_GetObjectItem(c, "height")->valuedouble;
      fc->xoffset     = (float)cJSON_GetObjectItem(c, "xoffset")->valuedouble;
      fc->yoffset     = (float)cJSON_GetObjectItem(c, "yoffset")->valuedouble;
      fc->xadvance    = (float)cJSON_GetObjectItem(c, "xadvance")->valuedouble;
      fc->x           = (float)cJSON_GetObjectItem(c, "x")->valuedouble;
      fc->y           = (float)cJSON_GetObjectItem(c, "y")->valuedouble;
      fc->page        = cJSON_GetObjectItem(c, "page")->valueint;
      fc->char_index  = i;

      /* Store char data for GPU */
      chars_array[array_offset]     = fc->x * u;      /* texOffset.x */
      chars_array[array_offset + 1] = fc->y * v;      /* texOffset.y */
      chars_array[array_offset + 2] = fc->width * u;  /* texExtent.x */
      chars_array[array_offset + 3] = fc->height * v; /* texExtent.y */
      chars_array[array_offset + 4] = fc->width;      /* size.x */
      chars_array[array_offset + 5] = fc->height;     /* size.y */
      chars_array[array_offset + 6] = fc->xoffset;    /* offset.x */
      chars_array[array_offset + 7] = -fc->yoffset;   /* offset.y */
      array_offset += 8;
    }

    /* Set default char */
    if (state.font.font.char_count > 0) {
      state.font.font.default_char = state.font.font.chars[0];
    }

    /* Create GPU buffer for char data */
    state.font.chars_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label            = STRVIEW("MSDF character layout buffer"),
        .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size             = state.font.font.char_count * 8 * sizeof(float),
        .mappedAtCreation = false,
      });

    wgpuQueueWriteBuffer(wgpu_context->queue, state.font.chars_buffer, 0,
                         chars_array,
                         state.font.font.char_count * 8 * sizeof(float));

    free(chars_array);
  }

  /* Parse kernings */
  cJSON* kernings = cJSON_GetObjectItem(json, "kernings");
  if (kernings) {
    int kerning_count = cJSON_GetArraySize(kernings);
    state.font.font.kerning_count
      = (kerning_count > MAX_KERNINGS) ? MAX_KERNINGS : kerning_count;

    for (int i = 0; i < state.font.font.kerning_count; ++i) {
      cJSON* k = cJSON_GetArrayItem(kernings, i);
      state.font.font.kernings[i].first
        = cJSON_GetObjectItem(k, "first")->valueint;
      state.font.font.kernings[i].second
        = cJSON_GetObjectItem(k, "second")->valueint;
      state.font.font.kernings[i].amount
        = cJSON_GetObjectItem(k, "amount")->valueint;
    }
  }

  /* Get texture page URL */
  cJSON* pages = cJSON_GetObjectItem(json, "pages");
  if (pages && cJSON_GetArraySize(pages) > 0) {
    const char* page_url = cJSON_GetArrayItem(pages, 0)->valuestring;

    /* Build texture path */
    char texture_path[256];
    snprintf(texture_path, sizeof(texture_path), "assets/font/%s", page_url);

    /* Create placeholder texture */
    state.font.texture = wgpu_create_color_bars_texture(
      wgpu_context,
      &(wgpu_texture_desc_t){
        .format = WGPUTextureFormat_RGBA8Unorm,
        .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                 | WGPUTextureUsage_RenderAttachment,
      });

    /* Start loading the texture */
    sfetch_send(&(sfetch_request_t){
      .path     = texture_path,
      .callback = font_texture_fetch_callback,
      .buffer   = SFETCH_RANGE(state.font_texture_buffer),
    });
  }

  cJSON_Delete(json);

  state.font.loaded = true;
}

/* -------------------------------------------------------------------------- *
 * Initialize font loading
 * -------------------------------------------------------------------------- */

static void init_font(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/font/ya-hei-ascii-msdf.json",
    .callback = font_json_fetch_callback,
    .buffer   = SFETCH_RANGE(state.font_json_buffer),
  });
}

/* -------------------------------------------------------------------------- *
 * Get text transform matrix
 * -------------------------------------------------------------------------- */

static void get_text_transform(vec3 position, vec3 rotation, mat4 dest)
{
  glm_mat4_identity(dest);
  glm_translate(dest, position);

  if (rotation[0] != 0.0f) {
    glm_rotate_x(dest, rotation[0], dest);
  }
  if (rotation[1] != 0.0f) {
    glm_rotate_y(dest, rotation[1], dest);
  }
  if (rotation[2] != 0.0f) {
    glm_rotate_z(dest, rotation[2], dest);
  }
}

/* -------------------------------------------------------------------------- *
 * Initialize text transforms
 * -------------------------------------------------------------------------- */

static void init_text_transforms(void)
{
  /* Front */
  get_text_transform((vec3){0.0f, 0.0f, 1.1f}, (vec3){0.0f, 0.0f, 0.0f},
                     state.text_transforms[0]);
  /* Back */
  get_text_transform((vec3){0.0f, 0.0f, -1.1f}, (vec3){0.0f, GLM_PI, 0.0f},
                     state.text_transforms[1]);
  /* Right */
  get_text_transform((vec3){1.1f, 0.0f, 0.0f},
                     (vec3){0.0f, GLM_PI / 2.0f, 0.0f},
                     state.text_transforms[2]);
  /* Left */
  get_text_transform((vec3){-1.1f, 0.0f, 0.0f},
                     (vec3){0.0f, -GLM_PI / 2.0f, 0.0f},
                     state.text_transforms[3]);
  /* Top */
  get_text_transform((vec3){0.0f, 1.1f, 0.0f},
                     (vec3){-GLM_PI / 2.0f, 0.0f, 0.0f},
                     state.text_transforms[4]);
  /* Bottom */
  get_text_transform((vec3){0.0f, -1.1f, 0.0f},
                     (vec3){GLM_PI / 2.0f, 0.0f, 0.0f},
                     state.text_transforms[5]);
}

/* -------------------------------------------------------------------------- *
 * Initialize formatted texts
 * -------------------------------------------------------------------------- */

static bool texts_initialized = false;

static void init_texts(wgpu_context_t* wgpu_context)
{
  if (!state.font.loaded || !state.font.font.bind_group) {
    return;
  }

  if (texts_initialized) {
    return;
  }

  texts_initialized = true;

  /* Face labels */
  static const char* face_labels[]
    = {"Front", "Back", "Right", "Left", "Top", "Bottom"};
  static const float face_colors[][4] = {
    {1.0f, 0.0f, 0.0f, 1.0f}, /* Front - Red */
    {0.0f, 1.0f, 1.0f, 1.0f}, /* Back - Cyan */
    {0.0f, 1.0f, 0.0f, 1.0f}, /* Right - Green */
    {1.0f, 0.0f, 1.0f, 1.0f}, /* Left - Magenta */
    {0.0f, 0.0f, 1.0f, 1.0f}, /* Top - Blue */
    {1.0f, 1.0f, 0.0f, 1.0f}, /* Bottom - Yellow */
  };

  for (int i = 0; i < 6; ++i) {
    msdf_text_format_options_t options = {
      .centered    = true,
      .pixel_scale = 1.0f / 128.0f,
      .has_color   = true,
    };
    memcpy(options.color, face_colors[i], sizeof(options.color));
    format_text(wgpu_context, &state.font.font, face_labels[i], &options);
  }

  /* Title text */
  msdf_text_format_options_t title_options = {
    .centered    = true,
    .pixel_scale = 1.0f / 128.0f,
    .has_color   = false,
  };
  state.title_text_idx
    = format_text(wgpu_context, &state.font.font, "WebGPU", &title_options);

  /* Large text */
  static const char* large_text
    = "\n"
      "WebGPU exposes an API for performing operations, such as rendering\n"
      "and computation, on a Graphics Processing Unit.\n"
      "\n"
      "Graphics Processing Units, or GPUs for short, have been essential\n"
      "in enabling rich rendering and computational applications in personal\n"
      "computing. WebGPU is an API that exposes the capabilities of GPU\n"
      "hardware for the Web. The API is designed from the ground up to\n"
      "efficiently map to (post-2014) native GPU APIs. WebGPU is not related\n"
      "to WebGL and does not explicitly target OpenGL ES.\n"
      "\n"
      "WebGPU sees physical GPU hardware as GPUAdapters. It provides a\n"
      "connection to an adapter via GPUDevice, which manages resources, and\n"
      "the device's GPUQueues, which execute commands. GPUDevice may have\n"
      "its own memory with high-speed access to the processing units.\n"
      "GPUBuffer and GPUTexture are the physical resources backed by GPU\n"
      "memory. GPUCommandBuffer and GPURenderBundle are containers for\n"
      "user-recorded commands. GPUShaderModule contains shader code. The\n"
      "other resources, such as GPUSampler or GPUBindGroup, configure the\n"
      "way physical resources are used by the GPU.\n"
      "\n"
      "GPUs execute commands encoded in GPUCommandBuffers by feeding data\n"
      "through a pipeline, which is a mix of fixed-function and programmable\n"
      "stages. Programmable stages execute shaders, which are special\n"
      "programs designed to run on GPU hardware. Most of the state of a\n"
      "pipeline is defined by a GPURenderPipeline or a GPUComputePipeline\n"
      "object. The state not included in these pipeline objects is set\n"
      "during encoding with commands, such as beginRenderPass() or\n"
      "setBlendConstant().";

  msdf_text_format_options_t large_options = {
    .centered    = false,
    .pixel_scale = 1.0f / 256.0f,
    .has_color   = false,
  };
  state.large_text_idx
    = format_text(wgpu_context, &state.font.font, large_text, &large_options);
}

/* -------------------------------------------------------------------------- *
 * Initialize view matrices
 * -------------------------------------------------------------------------- */

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  state.view_matrices.projection);
}

/* -------------------------------------------------------------------------- *
 * Update transformation matrix
 * -------------------------------------------------------------------------- */

static void update_transformation_matrix(wgpu_context_t* wgpu_context)
{
  const float now = (float)stm_sec(stm_now()) / 5.0f;

  /* View matrix */
  glm_mat4_identity(state.view_matrices.view);
  glm_translate(state.view_matrices.view, (vec3){0.0f, 0.0f, -5.0f});

  /* Model matrix */
  glm_mat4_identity(state.view_matrices.model);
  glm_translate(state.view_matrices.model, (vec3){0.0f, 2.0f, -3.0f});
  glm_rotate(state.view_matrices.model, 1.0f,
             (vec3){sinf(now), cosf(now), 0.0f});

  /* Model-view-projection matrix for the cube */
  glm_mat4_mul(state.view_matrices.projection, state.view_matrices.view,
               state.view_matrices.model_view_projection);
  glm_mat4_mul(state.view_matrices.model_view_projection,
               state.view_matrices.model,
               state.view_matrices.model_view_projection);

  /* Update cube uniform buffer */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.cube_uniform_buffer, 0,
                       state.view_matrices.model_view_projection, sizeof(mat4));

  /* Update text renderer camera */
  memcpy(state.text_renderer.camera_array, state.view_matrices.projection,
         sizeof(mat4));
  memcpy(state.text_renderer.camera_array + 16, state.view_matrices.view,
         sizeof(mat4));
  wgpuQueueWriteBuffer(
    wgpu_context->queue, state.text_renderer.camera_uniform_buffer, 0,
    state.text_renderer.camera_array, sizeof(state.text_renderer.camera_array));

  /* Update text transforms for cube faces */
  for (int i = 0; i < 6 && i < state.text_count; ++i) {
    glm_mat4_mul(state.view_matrices.model, state.text_transforms[i],
                 state.view_matrices.text_matrix);
    msdf_text_set_transform(&state.texts[i], state.view_matrices.text_matrix);
  }

  /* Update crawling text transform */
  if (state.title_text_idx >= 0 && state.large_text_idx >= 0) {
    uint64_t elapsed = stm_diff(stm_now(), state.start_time);
    float crawl      = fmodf((float)stm_sec(elapsed) / 2.5f, 14.0f);

    glm_mat4_identity(state.view_matrices.text_matrix);
    glm_rotate_x(state.view_matrices.text_matrix, -GLM_PI / 8.0f,
                 state.view_matrices.text_matrix);
    glm_translate(state.view_matrices.text_matrix,
                  (vec3){0.0f, crawl - 3.0f, 0.0f});

    msdf_text_set_transform(&state.texts[state.title_text_idx],
                            state.view_matrices.text_matrix);

    glm_translate(state.view_matrices.text_matrix, (vec3){-3.0f, -0.1f, 0.0f});
    msdf_text_set_transform(&state.texts[state.large_text_idx],
                            state.view_matrices.text_matrix);
  }
}

/* -------------------------------------------------------------------------- *
 * Init function
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  UNUSED_FUNCTION(msdf_text_set_color);
  UNUSED_FUNCTION(msdf_text_set_pixel_scale);

  if (wgpu_context) {
    /* Store global context for async callbacks */
    state.wgpu_context = wgpu_context;

    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 4,
      .num_channels = 2,
      .num_lanes    = 2,
    });

    state.start_time = stm_now();

    init_cube_mesh();
    init_cube_vertex_buffer(wgpu_context);
    init_cube_pipeline(wgpu_context);
    init_cube_uniform_buffer(wgpu_context);
    init_text_renderer(wgpu_context);
    init_font(wgpu_context);
    init_text_transforms();
    init_view_matrices(wgpu_context);

    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

/* -------------------------------------------------------------------------- *
 * Frame function
 * -------------------------------------------------------------------------- */

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async loading */
  sfetch_dowork();

  /* Recreate font texture when loaded */
  if (state.font.texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.font.texture);
    FREE_TEXTURE_PIXELS(state.font.texture);
    init_font_bind_group(wgpu_context);
  }

  /* Initialize texts once font is ready */
  if (state.font.loaded && state.font.font.bind_group && !texts_initialized) {
    init_texts(wgpu_context);
  }

  /* Update transformation matrices */
  update_transformation_matrix(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Render cube */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.cube_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.cube_bind_group, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.cube_vertices.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(rpass_enc, state.cube_mesh.vertex_count, 1, 0, 0);

  /* Render text using render bundles */
  if (texts_initialized) {
    WGPURenderBundle bundles[MAX_FORMATTED_TEXTS];
    int bundle_count = 0;

    for (int i = 0; i < state.text_count && bundle_count < MAX_FORMATTED_TEXTS;
         ++i) {
      bundles[bundle_count++]
        = msdf_text_get_render_bundle(wgpu_context, &state.texts[i]);
    }

    if (bundle_count > 0) {
      wgpuRenderPassEncoderExecuteBundles(rpass_enc, bundle_count, bundles);
    }
  }

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Shutdown function
 * -------------------------------------------------------------------------- */

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  /* Release text resources */
  for (int i = 0; i < state.text_count; ++i) {
    WGPU_RELEASE_RESOURCE(RenderBundle, state.texts[i].render_bundle)
    WGPU_RELEASE_RESOURCE(Buffer, state.texts[i].text_buffer)
  }

  /* Release font resources */
  wgpu_destroy_texture(&state.font.texture);
  WGPU_RELEASE_RESOURCE(Buffer, state.font.chars_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.font.font.bind_group)

  /* Release text renderer resources */
  WGPU_RELEASE_RESOURCE(Sampler, state.text_renderer.sampler)
  WGPU_RELEASE_RESOURCE(Buffer, state.text_renderer.camera_uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.text_renderer.font_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.text_renderer.text_bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.text_renderer.pipeline)

  /* Release cube resources */
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.cube_bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.cube_pipeline)
}

/* -------------------------------------------------------------------------- *
 * Main function
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Text Rendering MSDF",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* msdf_text_shader_wgsl = CODE(
  // Positions for simple quad geometry
  const pos = array(vec2f(0, -1), vec2f(1, -1), vec2f(0, 0), vec2f(1, 0));

  struct VertexInput {
    @builtin(vertex_index) vertex : u32,
    @builtin(instance_index) instance : u32,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) texcoord : vec2f,
  };

  struct Char {
    texOffset: vec2f,
    texExtent: vec2f,
    size: vec2f,
    offset: vec2f,
  };

  struct FormattedText {
    transform: mat4x4f,
    color: vec4f,
    scale: f32,
    chars: array<vec3f>,
  };

  struct Camera {
    projection: mat4x4f,
    view: mat4x4f,
  };

  // Font bindings
  @group(0) @binding(0) var fontTexture: texture_2d<f32>;
  @group(0) @binding(1) var fontSampler: sampler;
  @group(0) @binding(2) var<storage> chars: array<Char>;

  // Text bindings
  @group(1) @binding(0) var<uniform> camera: Camera;
  @group(1) @binding(1) var<storage> text: FormattedText;

  @vertex
  fn vertexMain(input : VertexInput) -> VertexOutput {
    let textElement = text.chars[input.instance];
    let char = chars[u32(textElement.z)];
    let charPos = (pos[input.vertex] * char.size + textElement.xy + char.offset) * text.scale;

    var output : VertexOutput;
    output.position = camera.projection * camera.view * text.transform * vec4f(charPos, 0, 1);

    output.texcoord = pos[input.vertex] * vec2f(1, -1);
    output.texcoord *= char.texExtent;
    output.texcoord += char.texOffset;
    return output;
  }

  fn sampleMsdf(texcoord: vec2f) -> f32 {
    let c = textureSample(fontTexture, fontSampler, texcoord);
    return max(min(c.r, c.g), min(max(c.r, c.g), c.b));
  }

  // Antialiasing technique from Paul Houx
  // https://github.com/Chlumsky/msdfgen/issues/22#issuecomment-234958005
  @fragment
  fn fragmentMain(input : VertexOutput) -> @location(0) vec4f {
    // pxRange (AKA distanceRange) comes from the msdfgen tool. Don McCurdy's tool
    // uses the default which is 4.
    let pxRange = 4.0;
    let sz = vec2f(textureDimensions(fontTexture, 0));
    let dx = sz.x*length(vec2f(dpdx(input.texcoord.x), dpdy(input.texcoord.x)));
    let dy = sz.y*length(vec2f(dpdx(input.texcoord.y), dpdy(input.texcoord.y)));
    let toPixels = pxRange * inverseSqrt(dx * dx + dy * dy);
    let sigDist = sampleMsdf(input.texcoord) - 0.5;
    let pxDist = sigDist * toPixels;

    let edgeWidth = 0.5;

    let alpha = smoothstep(-edgeWidth, edgeWidth, pxDist);

    if (alpha < 0.001) {
      discard;
    }

    return vec4f(text.color.rgb, text.color.a * alpha);
  }
);
// clang-format on
