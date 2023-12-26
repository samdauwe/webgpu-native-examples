#include "text_overlay.h"

#include <stdarg.h>
#include <string.h>

#include <cglm/cglm.h>

#include "../core/macro.h"
#include "buffer.h"
#include "shader.h"

// https://nothings.org/stb/font/
// https://www.nothings.org/stb/font/latin1/consolas/
#include <stb_font_consolas_24_latin1.h>

/* Max. number of chars the text overlay buffer can hold */
#define TEXTOVERLAY_MAX_CHAR_COUNT 2048

/**
 * @brief Text vertex class
 */
typedef struct text_vertex_t {
  vec2 position;
  vec2 uv;
} text_vertex_t;

/**
 * @brief Text overlay class
 */
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
  struct {
    WGPUTextureFormat format;
  } color, depth_stencil;
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass;
  struct {
    struct {
      text_vertex_t data[TEXTOVERLAY_MAX_CHAR_COUNT * 4];
      size_t size;
    } vertex;
  } draw_buffer;
  stb_fontchar stb_font_data[STB_FONT_consolas_24_latin1_NUM_CHARS];
  uint32_t num_letters;
  bool flip_y; /* false: Y-axis up / true: Y-axis down */
} text_overlay;

static void text_overlay_init(text_overlay_t* text_overlay,
                              wgpu_context_t* wgpu_context)
{
  text_overlay->wgpu_context         = wgpu_context;
  text_overlay->color.format         = wgpu_context->swap_chain.format;
  text_overlay->depth_stencil.format = WGPUTextureFormat_Depth24PlusStencil8;
  text_overlay->flip_y               = false;
}

static void text_overlay_create_vertex_buffer(text_overlay_t* text_overlay)
{
  text_overlay->draw_buffer.vertex.size
    = sizeof(text_overlay->draw_buffer.vertex.data);

  text_overlay->vertex_buffer = wgpu_create_buffer(
    text_overlay->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "text-overlay-vertex-buffer",
      .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size  = text_overlay->draw_buffer.vertex.size,
    });
}

static void text_overlay_create_fonts_texture(text_overlay_t* text_overlay)
{
  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  const uint32_t font_width  = STB_FONT_consolas_24_latin1_BITMAP_WIDTH;
  const uint32_t font_height = STB_FONT_consolas_24_latin1_BITMAP_WIDTH;

  static unsigned char font24pixels[STB_FONT_consolas_24_latin1_BITMAP_WIDTH]
                                   [STB_FONT_consolas_24_latin1_BITMAP_WIDTH];
  stb_font_consolas_24_latin1(text_overlay->stb_font_data, font24pixels,
                              font_height);
  /* Size of the font texture is WIDTH * HEIGHT * 1 byte (only one channel) */
  size_t bytes_per_pixel   = 1;
  size_t font24pixels_size = font_width * font_height * bytes_per_pixel;

  /* Upload font texture to graphics system */
  WGPUExtent3D texture_size = {
    .width              = font_width,
    .height             = font_height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "text-overlay-font-texture",
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

  /* Staging buffer */
  wgpu_buffer_t gpu_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
        .label   = "text-overlay-font-texture_staging_buffer",
        .usage   = WGPUBufferUsage_CopySrc,
        .size    = font24pixels_size,
        .initial = {
          .data  = font24pixels,
          .size  = font24pixels_size,
        },
    });

  /* Copy buffer to texture */
  WGPUImageCopyBuffer buffer_copy_view    = {
      .buffer = gpu_buffer.buffer,
      .layout = (WGPUTextureDataLayout){
          .offset       = 0,
          .bytesPerRow  = font_width * bytes_per_pixel,
          .rowsPerImage = font_height,
      },
    };

  WGPUImageCopyTexture texture_copy_view = {
      .texture = text_overlay->font.texture,
      .mipLevel = 0,
      .origin = (WGPUOrigin3D){
          .x = 0u,
          .y = 0u,
          .z = 0u,
        },
      .aspect = WGPUTextureAspect_All,
    };

  WGPUCommandBuffer copy_command = wgpu_copy_buffer_to_texture(
    wgpu_context, &buffer_copy_view, &texture_copy_view, &texture_size);
  /* Submit to the queue */
  wgpuQueueSubmit(wgpu_context->queue, 1, &copy_command);

  /* Release command buffer and staging buffer */
  WGPU_RELEASE_RESOURCE(CommandBuffer, copy_command)
  wgpu_destroy_buffer(&gpu_buffer);

  /* Create texture view */
  WGPUTextureViewDescriptor texture_view_desc = {
    .label           = "text-overlay-texture-view",
    .format          = texture_desc.format,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };

  text_overlay->font.texture_view
    = wgpuTextureCreateView(text_overlay->font.texture, &texture_view_desc);
  ASSERT(text_overlay->font.texture_view);

  /* Create the sampler */
  WGPUSamplerDescriptor sampler_desc = {
    .label         = "imgui-font-sampler",
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
    .addressModeW  = WGPUAddressMode_Repeat,
    .magFilter     = WGPUFilterMode_Linear,
    .minFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUMipmapFilterMode_Linear,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
    .compare       = WGPUCompareFunction_Undefined,
  };

  text_overlay->font.sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(text_overlay->font.sampler);
}

static void text_overlay_setup_pipeline_layout(text_overlay_t* text_overlay)
{
  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Texture view (Fragment shader)
      .binding = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Sampler (Fragment shader)
      .binding = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type=WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    }
  };
  text_overlay->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(text_overlay->bind_group_layout != NULL)

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this descriptor set layout
  text_overlay->pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &text_overlay->bind_group_layout,
    });
  ASSERT(text_overlay->pipeline_layout != NULL)
}

static void text_overlay_setup_bind_group(text_overlay_t* text_overlay)
{
  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  // Bind Group
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0 : Fragment shader texture view
      .binding = 0,
      .textureView =text_overlay->font.texture_view,
    },
    [1] = (WGPUBindGroupEntry) {
      // Binding 1: Fragment shader image sampler
      .binding = 1,
      .sampler = text_overlay->font.sampler,
    },
  };

  text_overlay->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .layout     = text_overlay->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(text_overlay->bind_group != NULL)
}

// Prepare a separate pipeline for the font rendering decoupled from the main
// application
static void text_overlay_prepare_pipeline(text_overlay_t* text_overlay)
{
  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleStrip,
    .frontFace = WGPUFrontFace_CW,
    .cullMode  = text_overlay->flip_y ? WGPUCullMode_Front : WGPUCullMode_Back,
  };

  // Enable blending, using alpha from red channel of the font texture (see
  // text.frag)
  WGPUBlendState blend_state                   = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = text_overlay->color.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = text_overlay->depth_stencil.format,
      .depth_write_enabled = true,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    text_overlay, sizeof(text_vertex_t),
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2,
                       offsetof(text_vertex_t, position)),
    // Attribute location 1: UV
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2,
                       offsetof(text_vertex_t, uv)))

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
        wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Vertex shader SPIR-V
          .file = "shaders/text_overlay/text.vert.spv",
        },
        .buffer_count = 1,
        .buffers = &text_overlay_vertex_buffer_layout,
      });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
        wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Fragment shader SPIR-V
          .file = "shaders/text_overlay/text.frag.spv",
        },
        .target_count = 1,
        .targets = &color_target_state_desc,
      });

  // Multisample state
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  text_overlay->pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "text_overlay_render_pipeline",
                            .layout       = text_overlay->pipeline_layout,
                            .primitive    = primitive_state_desc,
                            .vertex       = vertex_state_desc,
                            .fragment     = &fragment_state_desc,
                            .depthStencil = &depth_stencil_state_desc,
                            .multisample  = multisample_state_desc,
                          });
  ASSERT(text_overlay->pipeline);

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

// Prepare a separate render pass for rendering the text as an overlay
static void text_overlay_setup_render_pass(text_overlay_t* text_overlay)
{
  wgpu_context_t* wgpu_context = text_overlay->wgpu_context;

  // Color attachment
  text_overlay->render_pass.color_attachment[0]
    = (WGPURenderPassColorAttachment) {
      .view       = NULL,
      .depthSlice = ~0,
      // Don't clear the framebuffer (like the renderpass from the example does)
      .loadOp     = WGPULoadOp_Load,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 0.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Render pass descriptor
  text_overlay->render_pass.render_pass_descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = text_overlay->render_pass.color_attachment,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

text_overlay_t* text_overlay_create(wgpu_context_t* wgpu_context)
{
  text_overlay_t* text_overlay
    = (text_overlay_t*)malloc(sizeof(text_overlay_t));
  memset(text_overlay, 0, sizeof(text_overlay_t));

  // Prepare ImGui overlay
  text_overlay_init(text_overlay, wgpu_context);
  // Create the vertex buffer containing the text overloy vertices
  text_overlay_create_vertex_buffer(text_overlay);
  // Create the pipeline layout that is used to generate the rendering
  // pipelines
  text_overlay_setup_pipeline_layout(text_overlay);
  // Create the fonts texture
  text_overlay_create_fonts_texture(text_overlay);
  // Setup the bind group containing the texture bindings
  text_overlay_setup_bind_group(text_overlay);
  // Create the graphics pipeline
  text_overlay_prepare_pipeline(text_overlay);
  // Setup render pass
  text_overlay_setup_render_pass(text_overlay);

  return text_overlay;
}

void text_overlay_release(text_overlay_t* text_overlay)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, text_overlay->pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, text_overlay->pipeline_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, text_overlay->bind_group);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, text_overlay->bind_group_layout);

  WGPU_RELEASE_RESOURCE(Buffer, text_overlay->vertex_buffer.buffer);
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

  const bool flip_y                  = text_overlay->flip_y;
  const uint32_t frame_buffer_width  = wgpu_context->surface.width;
  const uint32_t frame_buffer_height = wgpu_context->surface.height;

  const float charW = 1.5f / frame_buffer_width;
  const float charH = 1.5f / frame_buffer_height;

  float fbW      = (float)frame_buffer_width;
  float fbH      = (float)frame_buffer_height;
  x              = (x / fbW * 2.0f) - 1.0f;
  y              = flip_y ? (y / fbH * 2.0f) - 1.0f : 1.0f - (y / fbH * 2.0f);
  float fbYScale = flip_y ? -1.0f : 1.0f;

  // Calculate text width
  float textWidth    = 0.0f;
  size_t text_length = strlen(text);
  for (size_t i = 0; i < text_length; ++i) {
    stb_fontchar* charData
      = &text_overlay->stb_font_data[(uint32_t)text[i] - first_char];
    textWidth += charData->advance * charW;
  }

  switch (align) {
    case TextOverlay_Text_AlignRight:
      x -= textWidth;
      break;
    case TextOverlay_Text_AlignCenter:
      x -= textWidth / 2.0f;
      break;
    default:
      break;
  }

  // Generate a uv mapped quad per char in the new text
  text_vertex_t* mapped
    = &text_overlay->draw_buffer.vertex.data[text_overlay->num_letters * 4];
  for (size_t i = 0; i < text_length; ++i) {
    stb_fontchar* charData
      = &text_overlay->stb_font_data[(uint32_t)text[i] - first_char];

    mapped->position[0] = (x + (float)charData->x0 * charW);
    mapped->position[1] = (y - (float)charData->y0 * charH * fbYScale);
    mapped->uv[0]       = charData->s0;
    mapped->uv[1]       = charData->t0;
    mapped++;

    mapped->position[0] = (x + (float)charData->x1 * charW);
    mapped->position[1] = (y - (float)charData->y0 * charH * fbYScale);
    mapped->uv[0]       = charData->s1;
    mapped->uv[1]       = charData->t0;
    mapped++;

    mapped->position[0] = (x + (float)charData->x0 * charW);
    mapped->position[1] = (y - (float)charData->y1 * charH * fbYScale);
    mapped->uv[0]       = charData->s0;
    mapped->uv[1]       = charData->t1;
    mapped++;

    mapped->position[0] = (x + (float)charData->x1 * charW);
    mapped->position[1] = (y - (float)charData->y1 * charH * fbYScale);
    mapped->uv[0]       = charData->s1;
    mapped->uv[1]       = charData->t1;
    mapped++;

    x += charData->advance * charW;

    text_overlay->num_letters++;
  }
}

void text_overlay_add_formatted_text(text_overlay_t* text_overlay, float x,
                                     float y,
                                     text_overlay_text_align_enum align,
                                     const char* format_str, ...)
{
  char text[STRMAX];
  va_list args;
  va_start(args, format_str);
  vsnprintf(text, sizeof(text), format_str, args);
  va_end(args);
  text_overlay_add_text(text_overlay, text, x, y, align);
}

// Unmap buffer
void text_overlay_end_text_update(text_overlay_t* text_overlay)
{
  uint32_t data_size = text_overlay->num_letters * sizeof(text_vertex_t)
                       * 4 /* => 4 vertices per character */;

  wgpu_record_copy_data_to_buffer(
    text_overlay->wgpu_context, &text_overlay->vertex_buffer, 0, data_size,
    text_overlay->draw_buffer.vertex.data, data_size);
}

void text_overlay_draw_frame(text_overlay_t* text_overlay, WGPUTextureView view)
{
  text_overlay->render_pass.color_attachment[0].view = view;
  text_overlay->wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    text_overlay->wgpu_context->cmd_enc,
    &text_overlay->render_pass.render_pass_descriptor);
  WGPURenderPassEncoder rpass_enc = text_overlay->wgpu_context->rpass_enc;

  wgpuRenderPassEncoderSetPipeline(rpass_enc, text_overlay->pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, text_overlay->bind_group, 0,
                                    NULL);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 0, text_overlay->vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  for (uint32_t j = 0; j < text_overlay->num_letters; ++j) {
    wgpuRenderPassEncoderDraw(rpass_enc, 4, 1, j * 4, 0);
  }

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
}
