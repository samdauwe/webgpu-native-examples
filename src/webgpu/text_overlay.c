#include "text_overlay.h"

#include <stdarg.h>
#include <string.h>

/* https://nothings.org/stb/font/ */
/* https://www.nothings.org/stb/font/latin1/consolas/ */
#include <stb_font_consolas_24_latin1.h>

#include <stdio.h>

/* Max. number of chars the text overlay buffer can hold */
#define TEXTOVERLAY_MAX_CHAR_COUNT (2048)

/* Format string buffer size */
#define TEXT_OVERLAY_STRMAX (256)

/* -------------------------------------------------------------------------- *
 * WGSL shader (declared here, defined at end of file)
 * -------------------------------------------------------------------------- */

static const char* text_overlay_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Text vertex structure
 * -------------------------------------------------------------------------- */

typedef struct text_vertex_t {
  float position[2];
  float uv[2];
} text_vertex_t;

/* -------------------------------------------------------------------------- *
 * Text overlay internal structure
 * -------------------------------------------------------------------------- */

typedef struct text_overlay {
  wgpu_context_t* wgpu_context;
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  wgpu_buffer_t vertex_buffer;
  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;
  struct {
    WGPUTexture texture;
    WGPUTextureView texture_view;
    WGPUSampler sampler;
  } font;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  struct {
    text_vertex_t data[TEXTOVERLAY_MAX_CHAR_COUNT * 4];
  } draw_buffer;
  stb_fontchar stb_font_data[STB_FONT_consolas_24_latin1_NUM_CHARS];
  uint32_t num_letters;
} text_overlay;

/* -------------------------------------------------------------------------- *
 * Font texture creation
 * -------------------------------------------------------------------------- */

static void text_overlay_create_fonts_texture(text_overlay_t* text_overlay)
{
  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  const uint32_t font_width  = STB_FONT_consolas_24_latin1_BITMAP_WIDTH;
  const uint32_t font_height = STB_FONT_consolas_24_latin1_BITMAP_HEIGHT;

  static unsigned char font24pixels[STB_FONT_consolas_24_latin1_BITMAP_HEIGHT]
                                   [STB_FONT_consolas_24_latin1_BITMAP_WIDTH];
  stb_font_consolas_24_latin1(text_overlay->stb_font_data, font24pixels,
                              font_height);

  /* Upload font texture via wgpuQueueWriteTexture */
  WGPUExtent3D texture_size = {
    .width              = font_width,
    .height             = font_height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Text overlay font texture"),
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
    .dimension     = WGPUTextureDimension_2D,
    .size          = texture_size,
    .format        = WGPUTextureFormat_R8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  text_overlay->font.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(text_overlay->font.texture);

  /* Write pixel data directly to texture (no staging buffer needed) */
  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = text_overlay->font.texture,
                          .mipLevel = 0,
                          .origin   = (WGPUOrigin3D){0, 0, 0},
                          .aspect   = WGPUTextureAspect_All,
                        },
                        font24pixels, font_width * font_height,
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = font_width,
                          .rowsPerImage = font_height,
                        },
                        &texture_size);

  /* Create texture view */
  text_overlay->font.texture_view
    = wgpuTextureCreateView(text_overlay->font.texture, NULL);
  ASSERT(text_overlay->font.texture_view);

  /* Create sampler */
  text_overlay->font.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label = STRVIEW("Text overlay font sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(text_overlay->font.sampler);
}

/* -------------------------------------------------------------------------- *
 * Pipeline layout and bind group setup
 * -------------------------------------------------------------------------- */

static void text_overlay_setup_pipeline_layout(text_overlay_t* text_overlay)
{
  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };

  text_overlay->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Text overlay bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(text_overlay->bind_group_layout != NULL);

  text_overlay->pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Text overlay pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &text_overlay->bind_group_layout,
    });
  ASSERT(text_overlay->pipeline_layout != NULL);
}

static void text_overlay_setup_bind_group(text_overlay_t* text_overlay)
{
  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = text_overlay->font.texture_view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = text_overlay->font.sampler,
    },
  };

  text_overlay->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Text overlay bind group"),
                            .layout     = text_overlay->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(text_overlay->bind_group != NULL);
}

/* -------------------------------------------------------------------------- *
 * Render pipeline
 * -------------------------------------------------------------------------- */

static void text_overlay_prepare_pipeline(text_overlay_t* text_overlay)
{
  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;
  WGPUDevice device            = wgpu_context->device;

  WGPUShaderModule shader
    = wgpu_create_shader_module(device, text_overlay_shader_wgsl);

  /* Blending: use alpha from red channel of the font texture */
  WGPUBlendState blend_state = {
    .color = (WGPUBlendComponent){
      .operation = WGPUBlendOperation_Add,
      .srcFactor = WGPUBlendFactor_SrcAlpha,
      .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
    },
    .alpha = (WGPUBlendComponent){
      .operation = WGPUBlendOperation_Add,
      .srcFactor = WGPUBlendFactor_OneMinusSrcAlpha,
      .dstFactor = WGPUBlendFactor_Zero,
    },
  };

  WGPUColorTargetState color_target = {
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUVertexAttribute vertex_attrs[2] = {
    [0] = {
      .shaderLocation = 0,
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = offsetof(text_vertex_t, position),
    },
    [1] = {
      .shaderLocation = 1,
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = offsetof(text_vertex_t, uv),
    },
  };

  WGPUVertexBufferLayout vb_layout = {
    .arrayStride    = sizeof(text_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = (uint32_t)ARRAY_SIZE(vertex_attrs),
    .attributes     = vertex_attrs,
  };

  text_overlay->pipeline = wgpuDeviceCreateRenderPipeline(
    device, &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Text overlay render pipeline"),
      .layout = text_overlay->pipeline_layout,
      .vertex = (WGPUVertexState){
        .module      = shader,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &vb_layout,
      },
      .primitive = (WGPUPrimitiveState){
        .topology         = WGPUPrimitiveTopology_TriangleStrip,
        .stripIndexFormat  = WGPUIndexFormat_Uint32,
        .frontFace        = WGPUFrontFace_CCW,
        .cullMode         = WGPUCullMode_None,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
      .fragment = &(WGPUFragmentState){
        .module      = shader,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &color_target,
      },
    });
  ASSERT(text_overlay->pipeline);

  WGPU_RELEASE_RESOURCE(ShaderModule, shader);
}

/* -------------------------------------------------------------------------- *
 * Public API
 * -------------------------------------------------------------------------- */

text_overlay_t* text_overlay_create(wgpu_context_t* wgpu_context)
{
  text_overlay_t* text_overlay
    = (text_overlay_t*)malloc(sizeof(text_overlay_t));
  memset(text_overlay, 0, sizeof(text_overlay_t));

  text_overlay->wgpu_context = wgpu_context;

  /* Create vertex buffer for text quads */
  text_overlay->vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Text overlay vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(text_overlay->draw_buffer.data),
                  });

  /* Setup pipeline layout and bind group */
  text_overlay_setup_pipeline_layout(text_overlay);
  text_overlay_create_fonts_texture(text_overlay);
  text_overlay_setup_bind_group(text_overlay);
  text_overlay_prepare_pipeline(text_overlay);

  /* Render pass descriptor: load existing content, no depth for overlay */
  text_overlay->color_attachment = (WGPURenderPassColorAttachment){
    .view       = NULL,
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    .loadOp     = WGPULoadOp_Load,
    .storeOp    = WGPUStoreOp_Store,
  };
  text_overlay->render_pass_descriptor = (WGPURenderPassDescriptor){
    .label                = STRVIEW("Text overlay render pass"),
    .colorAttachmentCount = 1,
    .colorAttachments     = &text_overlay->color_attachment,
  };

  return text_overlay;
}

void text_overlay_release(text_overlay_t* text_overlay)
{
  if (!text_overlay) {
    return;
  }

  WGPU_RELEASE_RESOURCE(RenderPipeline, text_overlay->pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, text_overlay->pipeline_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, text_overlay->bind_group);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, text_overlay->bind_group_layout);
  wgpu_destroy_buffer(&text_overlay->vertex_buffer);
  WGPU_RELEASE_RESOURCE(Texture, text_overlay->font.texture);
  WGPU_RELEASE_RESOURCE(TextureView, text_overlay->font.texture_view);
  WGPU_RELEASE_RESOURCE(Sampler, text_overlay->font.sampler);

  free(text_overlay);
}

void text_overlay_begin_text_update(text_overlay_t* text_overlay)
{
  text_overlay->num_letters = 0;
}

void text_overlay_add_text(text_overlay_t* text_overlay, const char* text,
                           float x, float y, text_overlay_text_align_enum align)
{
  const uint32_t first_char = STB_FONT_consolas_24_latin1_FIRST_CHAR;

  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  const uint32_t frame_buffer_width  = (uint32_t)wgpu_context->width;
  const uint32_t frame_buffer_height = (uint32_t)wgpu_context->height;

  if (frame_buffer_width == 0 || frame_buffer_height == 0) {
    return;
  }

  /* Character dimensions in NDC space */
  const float charW = 1.5f / (float)frame_buffer_width;
  const float charH = 1.5f / (float)frame_buffer_height;

  const float fbW = (float)frame_buffer_width;
  const float fbH = (float)frame_buffer_height;

  /* Convert pixel coordinates to NDC [-1, 1]
   * WebGPU: +Y = up, so y goes from +1 (top) to -1 (bottom) */
  x = (x / fbW * 2.0f) - 1.0f;
  y = 1.0f - (y / fbH * 2.0f);

  /* Calculate text width for alignment */
  float text_width      = 0.0f;
  const size_t text_len = strlen(text);
  for (size_t i = 0; i < text_len; ++i) {
    uint32_t ch = (uint32_t)(unsigned char)text[i];
    if (ch < first_char
        || ch >= first_char + STB_FONT_consolas_24_latin1_NUM_CHARS) {
      continue;
    }
    stb_fontchar* cd = &text_overlay->stb_font_data[ch - first_char];
    text_width += cd->advance * charW;
  }

  switch (align) {
    case TextOverlay_Text_AlignRight:
      x -= text_width;
      break;
    case TextOverlay_Text_AlignCenter:
      x -= text_width / 2.0f;
      break;
    default:
      break;
  }

  /* Generate a UV-mapped quad (triangle strip) per character */
  text_vertex_t* mapped
    = &text_overlay->draw_buffer.data[text_overlay->num_letters * 4];

  for (size_t i = 0; i < text_len; ++i) {
    if (text_overlay->num_letters >= TEXTOVERLAY_MAX_CHAR_COUNT) {
      break;
    }

    uint32_t ch = (uint32_t)(unsigned char)text[i];
    if (ch < first_char
        || ch >= first_char + STB_FONT_consolas_24_latin1_NUM_CHARS) {
      continue;
    }
    stb_fontchar* cd = &text_overlay->stb_font_data[ch - first_char];

    /* Quad vertices: top-left, top-right, bottom-left, bottom-right
     * Y is negated because stb font y0 < y1, but in WebGPU clip +Y = up */
    mapped[0].position[0] = x + (float)cd->x0 * charW;
    mapped[0].position[1] = y - (float)cd->y0 * charH;
    mapped[0].uv[0]       = cd->s0;
    mapped[0].uv[1]       = cd->t0;

    mapped[1].position[0] = x + (float)cd->x1 * charW;
    mapped[1].position[1] = y - (float)cd->y0 * charH;
    mapped[1].uv[0]       = cd->s1;
    mapped[1].uv[1]       = cd->t0;

    mapped[2].position[0] = x + (float)cd->x0 * charW;
    mapped[2].position[1] = y - (float)cd->y1 * charH;
    mapped[2].uv[0]       = cd->s0;
    mapped[2].uv[1]       = cd->t1;

    mapped[3].position[0] = x + (float)cd->x1 * charW;
    mapped[3].position[1] = y - (float)cd->y1 * charH;
    mapped[3].uv[0]       = cd->s1;
    mapped[3].uv[1]       = cd->t1;

    mapped += 4;
    x += cd->advance * charW;
    text_overlay->num_letters++;
  }
}

void text_overlay_add_formatted_text(text_overlay_t* text_overlay, float x,
                                     float y,
                                     text_overlay_text_align_enum align,
                                     const char* format_str, ...)
{
  char text[TEXT_OVERLAY_STRMAX];
  va_list args;
  va_start(args, format_str);
  vsnprintf(text, sizeof(text), format_str, args);
  va_end(args);
  text_overlay_add_text(text_overlay, text, x, y, align);
}

void text_overlay_end_text_update(text_overlay_t* text_overlay)
{
  if (text_overlay->num_letters == 0) {
    return;
  }

  const uint32_t data_size
    = text_overlay->num_letters * (uint32_t)sizeof(text_vertex_t) * 4;

  wgpuQueueWriteBuffer(text_overlay->wgpu_context->queue,
                       text_overlay->vertex_buffer.buffer, 0,
                       text_overlay->draw_buffer.data, data_size);
}

void text_overlay_draw_frame(text_overlay_t* text_overlay, WGPUTextureView view)
{
  if (!text_overlay || text_overlay->num_letters == 0) {
    return;
  }

  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  /* Create a dedicated command encoder for the overlay pass */
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  text_overlay->color_attachment.view = view;
  WGPURenderPassEncoder rpass_enc     = wgpuCommandEncoderBeginRenderPass(
    encoder, &text_overlay->render_pass_descriptor);

  wgpuRenderPassEncoderSetPipeline(rpass_enc, text_overlay->pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, text_overlay->bind_group, 0,
                                    NULL);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 0, text_overlay->vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);

  /* Draw each character as a 4-vertex triangle strip */
  for (uint32_t j = 0; j < text_overlay->num_letters; ++j) {
    wgpuRenderPassEncoderDraw(rpass_enc, 4, 1, j * 4, 0);
  }

  wgpuRenderPassEncoderEnd(rpass_enc);

  WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buf);

  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buf);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder);
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* text_overlay_shader_wgsl = CODE(
  struct VertexInput {
    @location(0) position: vec2f,
    @location(1) uv: vec2f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
  }

  @group(0) @binding(0) var fontTexture: texture_2d<f32>;
  @group(0) @binding(1) var fontSampler: sampler;

  @vertex
  fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
  }

  @fragment
  fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let alpha = textureSample(fontTexture, fontSampler, in.uv).r;
    return vec4f(1.0, 1.0, 1.0, alpha);
  }
);
// clang-format on
