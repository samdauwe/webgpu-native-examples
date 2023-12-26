#include "imgui_overlay.h"

#include <stdlib.h>
#include <string.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#define ImDrawCallback_ResetRenderState (ImDrawCallback)(-1)

#include "../core/macro.h"
#include "shader.h"

#define _IMGUI_MAX_VERTEX_DATA_SIZE_DEFAULT 40000
#define _IMGUI_MAX_INDEX_DATA_SIZE_DEFAULT 10000

// Vertex buffer and attributes
typedef struct vertex_uniform_buffer_t {
  float mvp[4][4];
} vertex_uniform_buffer_t;

/**
 * @brief ImGui overlay class
 */
typedef struct imgui_overlay {
  wgpu_context_t* wgpu_context;
  struct {
    bool enable_alpha_blending;
    uint32_t msaa_sample_count;
    float scale;
    bool enable_scissor;
  } settings;
  struct {
    WGPUTexture texture;
    WGPUTextureView texture_view;
    WGPUSampler sampler;
  } font;
  WGPUTextureFormat depth_stencil_format;
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  wgpu_buffer_t uniform_buffer;
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;
  // Render pass descriptor for frame buffer writes
  WGPURenderPassColorAttachment rp_color_att_descriptors[1];
  WGPURenderPassDescriptor render_pass_desc;
  struct {
    struct {
      ImDrawVert data[_IMGUI_MAX_VERTEX_DATA_SIZE_DEFAULT];
      int32_t size;
    } vertex;
    struct {
      ImDrawIdx data[_IMGUI_MAX_INDEX_DATA_SIZE_DEFAULT];
      int32_t size;
    } index;
  } draw_buffers;
  bool visible;
  bool updated;
  float scale;
} imgui_overlay;

// Initialize styles, keys, etc.
static void imgui_overlay_init(imgui_overlay_t* imgui_overlay,
                               wgpu_context_t* wgpu_context,
                               WGPUTextureFormat format)
{
  // Configure ImGUI overlay settings
  imgui_overlay->wgpu_context = wgpu_context;

  imgui_overlay->depth_stencil_format = format;
  imgui_overlay->vertex_buffer.size   = 0;
  imgui_overlay->index_buffer.size    = 0;

  imgui_overlay->draw_buffers.index.size        = 3000;
  imgui_overlay->draw_buffers.vertex.size       = 3000;
  imgui_overlay->settings.enable_alpha_blending = true;
  imgui_overlay->settings.msaa_sample_count     = 1;
  imgui_overlay->settings.scale                 = 1.0f;
  imgui_overlay->settings.enable_scissor        = false;

  imgui_overlay->visible = true;
  imgui_overlay->updated = false;
  imgui_overlay->scale   = imgui_overlay->settings.scale;

  // Setup Dear ImGui context
  igCreateContext(NULL);

  // Setup Dear ImGui style
  igStyleColorsDark(igGetStyle());

  // Setup back-end capabilities flags
  ImGuiIO* io = igGetIO();
  ImFontAtlas_AddFontDefault(io->Fonts, NULL);
  io->BackendRendererName = "imgui_overlay";
  io->Fonts->TexID        = 0;
  io->FontGlobalScale     = imgui_overlay->settings.scale;
  io->BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
  io->DisplaySize.x             = (float)wgpu_context->surface.width;
  io->DisplaySize.y             = (float)wgpu_context->surface.height;
  io->DisplayFramebufferScale.x = 1.0f;
  io->DisplayFramebufferScale.y = 1.0f;
}

static void imgui_overlay_setup_render_state(imgui_overlay_t* imgui_overlay)
{
  WGPURenderPassEncoder rpass_enc = imgui_overlay->wgpu_context->rpass_enc;
  wgpuRenderPassEncoderSetPipeline(rpass_enc, imgui_overlay->pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, imgui_overlay->bind_group, 0,
                                    NULL);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 0, imgui_overlay->vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rpass_enc, imgui_overlay->index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
}

static void imgui_overlay_create_fonts_texture(imgui_overlay_t* imgui_overlay)
{
  wgpu_context_t* wgpu_context = imgui_overlay->wgpu_context;

  // Build texture atlas
  ImGuiIO* io = igGetIO();
  unsigned char* font_pixels;
  int font_width;
  int font_height;
  int bytes_per_pixel;
  ImFontAtlas_GetTexDataAsRGBA32(io->Fonts, &font_pixels, &font_width,
                                 &font_height, &bytes_per_pixel);
  const uint32_t pixels_size_bytes = font_width * font_height * bytes_per_pixel;

  // Upload texture to graphics system
  {
    WGPUExtent3D texture_size = {
      .width              = font_width,
      .height             = font_height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label     = "imgui-font-texture",
      .usage     = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
      .dimension = WGPUTextureDimension_2D,
      .size      = texture_size,
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    imgui_overlay->font.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(imgui_overlay->font.texture);

    wgpu_buffer_t gpu_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){.usage   = WGPUBufferUsage_CopySrc,
                                          .size    = pixels_size_bytes,
                                          .initial = {
                                            .data = font_pixels,
                                            .size = pixels_size_bytes,
                                          }});

    WGPUImageCopyBuffer buffer_copy_view
      = {.buffer = gpu_buffer.buffer,
         .layout = (WGPUTextureDataLayout){
           .offset       = 0,
           .bytesPerRow  = font_width * bytes_per_pixel,
           .rowsPerImage = font_height,
         }};

    WGPUImageCopyTexture texture_copy_view = {
      .texture = imgui_overlay->font.texture,
      .mipLevel = 0,
      .origin = (WGPUOrigin3D){
          .x=0u,
          .y=0u,
          .z=0u,
        },
      .aspect = WGPUTextureAspect_All,
    };

    WGPUCommandBuffer copy_command = wgpu_copy_buffer_to_texture(
      wgpu_context, &buffer_copy_view, &texture_copy_view, &texture_size);
    // Submit to the queue
    wgpuQueueSubmit(wgpu_context->queue, 1, &copy_command);

    // Release command buffer and staging buffer
    WGPU_RELEASE_RESOURCE(CommandBuffer, copy_command)
    wgpu_destroy_buffer(&gpu_buffer);

    // Create texture view
    WGPUTextureViewDescriptor texture_view_desc = {
      .label           = "imgui-texture-view",
      .format          = WGPUTextureFormat_RGBA8Unorm,
      .dimension       = WGPUTextureViewDimension_2D,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };

    imgui_overlay->font.texture_view
      = wgpuTextureCreateView(imgui_overlay->font.texture, &texture_view_desc);
    ASSERT(imgui_overlay->font.texture_view);

    // Create the sampler
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

    imgui_overlay->font.sampler
      = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
    ASSERT(imgui_overlay->font.sampler);
  }

  io->Fonts->TexID = (ImTextureID)imgui_overlay->font.texture_view;
}

static void imgui_overlay_setup_render_pass(imgui_overlay_t* imgui_overlay)
{
  wgpu_context_t* wgpu_context = imgui_overlay->wgpu_context;

  // Color attachment
  imgui_overlay->rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL,
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Load,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context,
                          &(deph_stencil_texture_creation_options){
                            .format = imgui_overlay->depth_stencil_format,
                          });

  // Render pass descriptor
  imgui_overlay->render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = imgui_overlay->rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void imgui_overlay_setup_pipeline_layout(imgui_overlay_t* imgui_overlay)
{
  wgpu_context_t* wgpu_context = imgui_overlay->wgpu_context;

  // Create bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[3]
      = {
      [0] = (WGPUBindGroupLayoutEntry){
        // Binding 0: Uniform buffer (Vertex shader)
        .binding              = 0,
        .visibility           = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = 16 * 4,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Sampler (Fragment shader)
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Texture view (Fragment shader)
        .binding = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      }
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    imgui_overlay->bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(imgui_overlay->bind_group_layout)
  }

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  {
    WGPUBindGroupLayout bgls[1] = {imgui_overlay->bind_group_layout};
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = bgls,
    };
    imgui_overlay->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &pipeline_layout_desc);
    ASSERT(imgui_overlay->pipeline_layout)
  }
}

static void imgui_overlay_prepare_pipeline(imgui_overlay_t* imgui_overlay)
{
  wgpu_context_t* wgpu_context = imgui_overlay->wgpu_context;

  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state
    = wgpu_create_blend_state(imgui_overlay->settings.enable_alpha_blending);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = imgui_overlay->depth_stencil_format,
      .depth_write_enabled = false,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    imgui, sizeof(ImDrawVert),
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2,
                       offsetof(ImDrawVert, pos)),
    // Attribute location 1: UV
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, offsetof(ImDrawVert, uv)),
    // Attribute location 2: Color
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Unorm8x4, offsetof(ImDrawVert, col)))

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader SPIR-V
                  .file = "shaders/imgui_overlay/imgui.vert.spv",
                },
                .buffer_count = 1,
                .buffers = &imgui_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader SPIR-V
                  .file = "shaders/imgui_overlay/imgui.frag.spv",
                },
                .target_count = 1,
                .targets = &color_target_state_desc,
              });

  // Multisample state
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = imgui_overlay->settings.msaa_sample_count,
      });

  // Create rendering pipeline using the specified states
  imgui_overlay->pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "imgui_render_pipeline",
                            .layout       = imgui_overlay->pipeline_layout,
                            .primitive    = primitive_state_desc,
                            .vertex       = vertex_state_desc,
                            .fragment     = &fragment_state_desc,
                            .depthStencil = &depth_stencil_state_desc,
                            .multisample  = multisample_state_desc,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static void imgui_overlay_prepare_uniform_buffer(imgui_overlay_t* imgui_overlay)
{
  wgpu_context_t* wgpu_context = imgui_overlay->wgpu_context;

  // Create uniform buffer
  {
    imgui_overlay->uniform_buffer = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "imgui-uniform-buffer",
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
        .size  = sizeof(vertex_uniform_buffer_t),
      });
  }

  // Create uniform bind group
  {
    WGPUBindGroupEntry bg_entries[3]= {
       [0] = (WGPUBindGroupEntry) {
         .binding = 0,
         .offset  = 0,
         .size    = imgui_overlay->uniform_buffer.size,
         .buffer  = imgui_overlay->uniform_buffer.buffer,
       },
       [1] = (WGPUBindGroupEntry) {
         .binding = 1,
         .sampler = imgui_overlay->font.sampler,
       },
       [2] = (WGPUBindGroupEntry) {
         .binding = 2,
         .textureView = imgui_overlay->font.texture_view,
       },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .layout     = imgui_overlay->bind_group_layout,
      .entryCount = ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };

    imgui_overlay->bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(imgui_overlay->bind_group);
  }
}

imgui_overlay_t* imgui_overlay_create(wgpu_context_t* wgpu_context,
                                      WGPUTextureFormat format)
{
  imgui_overlay_t* imgui_overlay
    = (imgui_overlay_t*)malloc(sizeof(imgui_overlay_t));

  // Prepare ImGui overlay
  imgui_overlay_init(imgui_overlay, wgpu_context, format);
  // Create the pipeline layout that is used to generate the rendering
  // pipelines
  imgui_overlay_setup_pipeline_layout(imgui_overlay);
  // Create the fonts texture
  imgui_overlay_create_fonts_texture(imgui_overlay);
  // Prepare and initialize a uniform buffer block containing shader uniforms
  imgui_overlay_prepare_uniform_buffer(imgui_overlay);
  // Create the graphics pipeline
  imgui_overlay_prepare_pipeline(imgui_overlay);
  // Setup render pass
  imgui_overlay_setup_render_pass(imgui_overlay);

  return imgui_overlay;
}

void imgui_overlay_release(imgui_overlay_t* imgui_overlay)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, imgui_overlay->pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, imgui_overlay->pipeline_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, imgui_overlay->bind_group);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, imgui_overlay->bind_group_layout);

  WGPU_RELEASE_RESOURCE(Buffer, imgui_overlay->index_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, imgui_overlay->vertex_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Texture, imgui_overlay->font.texture);
  WGPU_RELEASE_RESOURCE(TextureView, imgui_overlay->font.texture_view);
  WGPU_RELEASE_RESOURCE(Sampler, imgui_overlay->font.sampler);
  WGPU_RELEASE_RESOURCE(Buffer, imgui_overlay->uniform_buffer.buffer);

  free(imgui_overlay);
}

float imgui_overlay_get_scale(imgui_overlay_t* imgui_overlay)
{
  return imgui_overlay->scale;
}

// Starts a new imGui frame
void imgui_overlay_new_frame(imgui_overlay_t* imgui_overlay,
                             wgpu_example_context_t* context)
{
  ImGuiIO* io = igGetIO();

  io->DisplaySize.x = (float)imgui_overlay->wgpu_context->surface.width;
  io->DisplaySize.y = (float)imgui_overlay->wgpu_context->surface.height;
  io->DeltaTime     = context->frame_timer;

  io->MousePos.x   = context->mouse_position[0];
  io->MousePos.y   = context->mouse_position[1];
  io->MouseDown[0] = context->mouse_buttons.left;
  io->MouseDown[1] = context->mouse_buttons.right;

  igNewFrame();
}

// UI scale and translate
static void imgui_overlay_update_uniform_buffers(imgui_overlay_t* imgui_overlay,
                                                 ImDrawData* draw_data)
{
  // Setup orthographic projection matrix into our constant buffer
  // Our visible imgui space lies from draw_data->DisplayPos (top left) to
  // draw_data->DisplayPos+data_data->DisplaySize (bottom right).
  vertex_uniform_buffer_t vertex_constant_buffer = {0};
  {
    const float L         = draw_data->DisplayPos.x;
    const float R         = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    const float T         = draw_data->DisplayPos.y;
    const float B         = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    const float mvp[4][4] = {
      {2.0f / (R - L), 0.0f, 0.0f, 0.0f},                 //
      {0.0f, 2.0f / (T - B), 0.0f, 0.0f},                 //
      {0.0f, 0.0f, 0.5f, 0.0f},                           //
      {(R + L) / (L - R), (T + B) / (B - T), 0.5f, 1.0f}, //
    };
    memcpy(&vertex_constant_buffer.mvp, mvp, sizeof(mvp));
  }

  wgpu_record_copy_data_to_buffer(
    imgui_overlay->wgpu_context, &imgui_overlay->uniform_buffer, 0,
    sizeof(vertex_uniform_buffer_t), &vertex_constant_buffer.mvp,
    sizeof(vertex_uniform_buffer_t));
}

// Draw current imGui frame into a command buffer
void imgui_overlay_draw_frame(imgui_overlay_t* imgui_overlay,
                              WGPUTextureView view)
{
  ImDrawData* draw_data = igGetDrawData();

  // Check if there is content to tbe rendered
  if (!draw_data || draw_data->CmdListsCount == 0) {
    return;
  }

  // UI scale and translate
  imgui_overlay_update_uniform_buffers(imgui_overlay, draw_data);

  // Set texture view
  imgui_overlay->rp_color_att_descriptors[0].view = view;
  imgui_overlay->wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    imgui_overlay->wgpu_context->cmd_enc, &imgui_overlay->render_pass_desc);
  WGPURenderPassEncoder rpass_enc = imgui_overlay->wgpu_context->rpass_enc;

  // Setup desired Dawn state
  imgui_overlay_setup_render_state(imgui_overlay);

  // Render pass
  // (Because we merged all buffers into a single one, we maintain our own
  // offset into them)
  int global_vtx_offset = 0;
  int global_idx_offset = 0;
  ImVec2 clip_off       = draw_data->DisplayPos;
  ImVec2 clip_scale     = draw_data->FramebufferScale;
  for (int n = 0; n < draw_data->CmdListsCount; n++) {
    const ImDrawList* cmd_list = draw_data->CmdLists[n];
    for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; ++cmd_i) {
      const ImDrawCmd* pcmd = &cmd_list->CmdBuffer.Data[cmd_i];
      if (pcmd->UserCallback != NULL) {
        // User callback, registered via ImDrawList::AddCallback()
        // (ImDrawCallback_ResetRenderState is a special callback value used by
        // the user to request the renderer to reset render state.)
        if (pcmd->UserCallback == ImDrawCallback_ResetRenderState) {
          imgui_overlay_setup_render_state(imgui_overlay);
        }
        else {
          pcmd->UserCallback(cmd_list, pcmd);
        }
      }
      else {
        // Apply Scissor, Bind texture, Draw
        if (imgui_overlay->settings.enable_scissor) {
          ImVec4 clip_rect;
          clip_rect.x = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
          clip_rect.y = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
          clip_rect.z = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
          clip_rect.w = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;
          wgpuRenderPassEncoderSetScissorRect(
            rpass_enc, (uint32_t)clip_rect.x, (uint32_t)clip_rect.y,
            (uint32_t)clip_rect.z, (uint32_t)clip_rect.w);
        }
        wgpuRenderPassEncoderDrawIndexed(
          rpass_enc, pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset,
          pcmd->VtxOffset + global_vtx_offset, 0);
      }
    }
    global_idx_offset += cmd_list->IdxBuffer.Size;
    global_vtx_offset += cmd_list->VtxBuffer.Size;
  }

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
}

// Update vertex and index buffer containing the imGui elements when required
static void imgui_overlay_update_buffers(imgui_overlay_t* imgui_overlay)
{
  ImDrawData* draw_data = igGetDrawData();

  // Check if there is content to tbe rendered
  if (!draw_data || draw_data->CmdListsCount == 0) {
    return;
  }

  // Avoid rendering when minimized
  if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f) {
    return;
  }

  wgpu_context_t* wgpu_context = imgui_overlay->wgpu_context;

  // Update buffers only if vertex or index count has been changed compared to
  // current buffer size

  // Create and grow index buffers if needed
  int32_t vertex_buffer_size = imgui_overlay->draw_buffers.vertex.size;
  if (imgui_overlay->vertex_buffer.size == 0
      || vertex_buffer_size < draw_data->TotalVtxCount) {
    vertex_buffer_size = draw_data->TotalVtxCount + 5000;
    vertex_buffer_size = vertex_buffer_size % 4 == 0 ?
                           vertex_buffer_size :
                           vertex_buffer_size + 4 - vertex_buffer_size % 4;

    if (imgui_overlay->vertex_buffer.size > 0) {
      wgpu_destroy_buffer(&imgui_overlay->vertex_buffer);
    }

    imgui_overlay->vertex_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "imgui-vertex-buffer",
                      .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                      .size  = vertex_buffer_size * sizeof(ImDrawVert),
                    });
    imgui_overlay->draw_buffers.vertex.size = vertex_buffer_size;
  }

  // Create and grow index buffers if needed
  int32_t indexBuffer_size = imgui_overlay->draw_buffers.index.size;
  if (imgui_overlay->index_buffer.size == 0
      || indexBuffer_size < draw_data->TotalIdxCount) {
    indexBuffer_size = draw_data->TotalIdxCount + 10000;
    indexBuffer_size = indexBuffer_size % 4 == 0 ?
                         indexBuffer_size :
                         indexBuffer_size + 4 - indexBuffer_size % 4;

    if (imgui_overlay->index_buffer.size > 0) {
      wgpu_destroy_buffer(&imgui_overlay->index_buffer);
    }

    imgui_overlay->index_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "imgui-index-buffer",
                      .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                      .size  = indexBuffer_size * sizeof(ImDrawIdx),
                    });
    imgui_overlay->draw_buffers.index.size = indexBuffer_size;
  }

  // Upload vertex/index data into a single contiguous GPU buffer
  uint32_t vtx_dst    = 0;
  uint32_t idx_dst    = 0;
  ImDrawVert* pVertex = imgui_overlay->draw_buffers.vertex.data;
  ImDrawIdx* pIndex   = imgui_overlay->draw_buffers.index.data;
  for (int n = 0; n < draw_data->CmdListsCount; ++n) {
    const ImDrawList* cmd_list = draw_data->CmdLists[n];
    memcpy(pVertex, cmd_list->VtxBuffer.Data,
           cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
    memcpy(pIndex, cmd_list->IdxBuffer.Data,
           cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));

    pVertex += cmd_list->VtxBuffer.Size;
    pIndex += cmd_list->IdxBuffer.Size;
    vtx_dst += cmd_list->VtxBuffer.Size * sizeof(ImDrawVert);
    idx_dst += cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx);
  }
  vtx_dst = vtx_dst % 4 == 0 ? vtx_dst : vtx_dst + 4 - vtx_dst % 4;
  idx_dst = idx_dst % 4 == 0 ? idx_dst : idx_dst + 4 - idx_dst % 4;

  if (vtx_dst != 0 && idx_dst != 0) {
    wgpu_record_copy_data_to_buffer(
      wgpu_context, &imgui_overlay->vertex_buffer, 0, vtx_dst,
      imgui_overlay->draw_buffers.vertex.data,
      (pVertex - imgui_overlay->draw_buffers.vertex.data) * sizeof(ImDrawVert));
    wgpu_record_copy_data_to_buffer(
      wgpu_context, &imgui_overlay->index_buffer, 0, idx_dst,
      imgui_overlay->draw_buffers.index.data,
      (pIndex - imgui_overlay->draw_buffers.index.data) * sizeof(ImDrawIdx));
  }
}

// Render function
// (this used to be set in io.RenderDrawListsFn and called by ImGui::Render(),
// but you can now call this directly from your main loop)
void imgui_overlay_render(imgui_overlay_t* imgui_overlay)
{
  // Render to generate draw buffers
  igRender();

  // Update vertex and index buffer containing the imGui elements when required
  imgui_overlay_update_buffers(imgui_overlay);
}

bool imgui_overlay_want_capture_mouse(void)
{
  ImGuiIO* io = igGetIO();
  return io->WantCaptureMouse;
}

bool imgui_overlay_header(const char* caption)
{
  return igCollapsingHeader_TreeNodeFlags(caption,
                                          ImGuiTreeNodeFlags_DefaultOpen);
}

bool imgui_overlay_checkBox(imgui_overlay_t* imgui_overlay, const char* caption,
                            bool* value)
{
  bool res = igCheckbox(caption, value);
  if (res) {
    imgui_overlay->updated = true;
  };
  return res;
}

bool imgui_overlay_input_float(imgui_overlay_t* imgui_overlay,
                               const char* caption, float* value, float step,
                               const char* format)
{
  bool res = igInputFloat(caption, value, step, step * 10.0f, format, 0);
  if (res) {
    imgui_overlay->updated = true;
  };
  return res;
}

bool imgui_overlay_slider_float(imgui_overlay_t* imgui_overlay,
                                const char* caption, float* value, float min,
                                float max, const char* format)
{
  bool res = igSliderFloat(caption, value, min, max, format, 0);
  if (res) {
    imgui_overlay->updated = true;
  };
  return res;
}

bool imgui_overlay_slider_int(imgui_overlay_t* imgui_overlay,
                              const char* caption, int32_t* value, int32_t min,
                              int32_t max)
{
  bool res = igSliderInt(caption, value, min, max, "%d", 0);
  if (res) {
    imgui_overlay->updated = true;
  };
  return res;
}

bool imgui_overlay_combo_box(imgui_overlay_t* imgui_overlay,
                             const char* caption, int32_t* item_index,
                             const char** items, uint32_t item_count)
{
  if (item_count == 0) {
    return false;
  }
  bool res
    = igCombo_Str_arr(caption, item_index, &items[0], item_count, item_count);
  if (res) {
    imgui_overlay->updated = true;
  };
  return res;
}

bool imgui_overlay_button(imgui_overlay_t* imgui_overlay, const char* caption)
{
  bool res = igSmallButton(caption);
  if (res) {
    imgui_overlay->updated = true;
  };
  return res;
}

void imgui_overlay_text(const char* format_str, ...)
{
  va_list args;
  va_start(args, format_str);
  igTextV(format_str, args);
  va_end(args);
}

bool imgui_overlay_color_edit4(imgui_overlay_t* imgui_overlay,
                               const char* caption, float color[4])
{
  bool res = igColorEdit4(caption, color, ImGuiColorEditFlags_Float);
  if (res) {
    imgui_overlay->updated = true;
  };
  return res;
}
