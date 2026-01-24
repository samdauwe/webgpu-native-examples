/**
 * @file imgui_overlay.c
 * @brief Modular ImGui overlay implementation for WebGPU examples
 *
 * A clean, efficient, and modular ImGui overlay that can be easily integrated
 * into any WebGPU example.
 */

#include "imgui_overlay.h"

#include <stdarg.h>
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

/* -------------------------------------------------------------------------- *
 * WGSL Shader Code
 * -------------------------------------------------------------------------- */

static const char* imgui_vertex_shader_wgsl;
static const char* imgui_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Internal State
 * -------------------------------------------------------------------------- */

static struct {
  /* ImGui context */
  struct ImGuiContext* imgui_context;
  bool initialized;

  /* Font texture */
  WGPUTexture font_texture;
  WGPUTextureView font_texture_view;
  WGPUSampler font_sampler;

  /* Render pipeline */
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  /* Buffers */
  WGPUBuffer vertex_buffer;
  uint64_t vertex_buffer_size;
  WGPUBuffer index_buffer;
  uint64_t index_buffer_size;
  WGPUBuffer uniform_buffer;

  /* Render pass descriptor (pre-allocated for performance) */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
} overlay_state = {0};

/* -------------------------------------------------------------------------- *
 * Internal Helper Functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Create font texture from ImGui font atlas
 */
static void create_font_texture(wgpu_context_t* wgpu_context)
{
  ImGuiIO* io = igGetIO();

  /* Build texture atlas */
  unsigned char* font_pixels;
  int font_width, font_height, bytes_per_pixel;
  ImFontAtlas_GetTexDataAsRGBA32(io->Fonts, &font_pixels, &font_width,
                                 &font_height, &bytes_per_pixel);
  uint32_t pixels_size_bytes
    = (uint32_t)(font_width * font_height * bytes_per_pixel);

  /* Create texture */
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("ImGui font texture"),
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
    .dimension     = WGPUTextureDimension_2D,
    .size          = (WGPUExtent3D){
      .width              = (uint32_t)font_width,
      .height             = (uint32_t)font_height,
      .depthOrArrayLayers = 1,
    },
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  overlay_state.font_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(overlay_state.font_texture);

  /* Upload texture data */
  wgpuQueueWriteTexture(
    wgpu_context->queue,
    &(WGPUTexelCopyTextureInfo){
      .texture  = overlay_state.font_texture,
      .mipLevel = 0,
      .origin   = (WGPUOrigin3D){0, 0, 0},
      .aspect   = WGPUTextureAspect_All,
    },
    font_pixels, pixels_size_bytes,
    &(WGPUTexelCopyBufferLayout){
      .offset       = 0,
      .bytesPerRow  = (uint32_t)(font_width * bytes_per_pixel),
      .rowsPerImage = (uint32_t)font_height,
    },
    &(WGPUExtent3D){
      .width              = (uint32_t)font_width,
      .height             = (uint32_t)font_height,
      .depthOrArrayLayers = 1,
    });

  /* Create texture view */
  overlay_state.font_texture_view = wgpuTextureCreateView(
    overlay_state.font_texture, &(WGPUTextureViewDescriptor){
                                  .label  = STRVIEW("ImGui font texture view"),
                                  .format = WGPUTextureFormat_RGBA8Unorm,
                                  .dimension      = WGPUTextureViewDimension_2D,
                                  .baseMipLevel   = 0,
                                  .mipLevelCount  = 1,
                                  .baseArrayLayer = 0,
                                  .arrayLayerCount = 1,
                                  .aspect          = WGPUTextureAspect_All,
                                });
  ASSERT(overlay_state.font_texture_view);

  /* Create sampler */
  overlay_state.font_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("ImGui font sampler"),
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_Repeat,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(overlay_state.font_sampler);

  /* Store texture ID in ImGui */
  io->Fonts->TexID = (ImTextureID)(intptr_t)overlay_state.font_texture_view;
}

/**
 * @brief Create ImGui render pipeline
 */
static void create_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = 64,
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
        .multisampled  = false,
      },
    },
  };

  overlay_state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("ImGui bind group layout"),
                            .entryCount = 3,
                            .entries    = bgl_entries,
                          });
  ASSERT(overlay_state.bind_group_layout);

  /* Create pipeline layout */
  overlay_state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("ImGui pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &overlay_state.bind_group_layout,
    });
  ASSERT(overlay_state.pipeline_layout);

  /* Vertex state */
  WGPUVertexAttribute vertex_attributes[3] = {
    [0] = (WGPUVertexAttribute){
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = offsetof(ImDrawVert, pos),
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute){
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = offsetof(ImDrawVert, uv),
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute){
      .format         = WGPUVertexFormat_Unorm8x4,
      .offset         = offsetof(ImDrawVert, col),
      .shaderLocation = 2,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = sizeof(ImDrawVert),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 3,
    .attributes     = vertex_attributes,
  };

  /* Create shader modules */
  WGPUShaderModule vs_module
    = wgpu_create_shader_module(wgpu_context->device, imgui_vertex_shader_wgsl);
  ASSERT(vs_module);

  WGPUShaderModule fs_module = wgpu_create_shader_module(
    wgpu_context->device, imgui_fragment_shader_wgsl);
  ASSERT(fs_module);

  /* Color target state with blending */
  WGPUBlendState blend_state = {
    .color = (WGPUBlendComponent){
      .operation = WGPUBlendOperation_Add,
      .srcFactor = WGPUBlendFactor_SrcAlpha,
      .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
    },
    .alpha = (WGPUBlendComponent){
      .operation = WGPUBlendOperation_Add,
      .srcFactor = WGPUBlendFactor_One,
      .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
    },
  };

  WGPUColorTargetState color_target_state = {
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Create render pipeline */
  overlay_state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("ImGui render pipeline"),
      .layout = overlay_state.pipeline_layout,
      .vertex = (WGPUVertexState){
        .module      = vs_module,
        .entryPoint  = STRVIEW("main"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .primitive = (WGPUPrimitiveState){
        .topology         = WGPUPrimitiveTopology_TriangleList,
        .stripIndexFormat = WGPUIndexFormat_Undefined,
        .frontFace        = WGPUFrontFace_CW,
        .cullMode         = WGPUCullMode_None,
      },
      .multisample = (WGPUMultisampleState){
        .count                  = 1,
        .mask                   = 0xFFFFFFFF,
        .alphaToCoverageEnabled = false,
      },
      .fragment = &(WGPUFragmentState){
        .module      = fs_module,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets     = &color_target_state,
      },
    });
  ASSERT(overlay_state.pipeline);

  /* Release shader modules */
  WGPU_RELEASE_RESOURCE(ShaderModule, vs_module)
  WGPU_RELEASE_RESOURCE(ShaderModule, fs_module)
}

/**
 * @brief Create uniform buffer for projection matrix
 */
static void create_uniform_buffer(wgpu_context_t* wgpu_context)
{
  overlay_state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("ImGui uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = 64,
    });
  ASSERT(overlay_state.uniform_buffer);
}

/**
 * @brief Create bind group
 */
static void create_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = overlay_state.uniform_buffer,
      .size    = 64,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = overlay_state.font_sampler,
    },
    [2] = (WGPUBindGroupEntry){
      .binding     = 2,
      .textureView = overlay_state.font_texture_view,
    },
  };

  overlay_state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("ImGui bind group"),
                            .layout     = overlay_state.bind_group_layout,
                            .entryCount = 3,
                            .entries    = bg_entries,
                          });
  ASSERT(overlay_state.bind_group);
}

/**
 * @brief Update vertex and index buffers
 */
static void update_buffers(wgpu_context_t* wgpu_context, ImDrawData* draw_data)
{
  uint64_t vertex_size
    = (uint64_t)draw_data->TotalVtxCount * sizeof(ImDrawVert);
  uint64_t index_size = (uint64_t)draw_data->TotalIdxCount * sizeof(ImDrawIdx);

  /* Create or resize vertex buffer */
  if (!overlay_state.vertex_buffer
      || overlay_state.vertex_buffer_size < vertex_size) {
    if (overlay_state.vertex_buffer) {
      WGPU_RELEASE_RESOURCE(Buffer, overlay_state.vertex_buffer)
    }
    overlay_state.vertex_buffer_size = vertex_size + 5000 * sizeof(ImDrawVert);
    overlay_state.vertex_buffer      = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
             .label = STRVIEW("ImGui vertex buffer"),
             .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
             .size  = overlay_state.vertex_buffer_size,
      });
    ASSERT(overlay_state.vertex_buffer);
  }

  /* Create or resize index buffer */
  if (!overlay_state.index_buffer
      || overlay_state.index_buffer_size < index_size) {
    if (overlay_state.index_buffer) {
      WGPU_RELEASE_RESOURCE(Buffer, overlay_state.index_buffer)
    }
    overlay_state.index_buffer_size = index_size + 10000 * sizeof(ImDrawIdx);
    overlay_state.index_buffer      = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
             .label = STRVIEW("ImGui index buffer"),
             .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
             .size  = overlay_state.index_buffer_size,
      });
    ASSERT(overlay_state.index_buffer);
  }

  /* Upload data using staging buffers */
  ImDrawVert* vtx_staging = (ImDrawVert*)malloc(vertex_size);
  ImDrawIdx* idx_staging  = (ImDrawIdx*)malloc(index_size);

  ImDrawVert* vtx_dst = vtx_staging;
  ImDrawIdx* idx_dst  = idx_staging;

  for (int n = 0; n < draw_data->CmdListsCount; n++) {
    const ImDrawList* cmd_list = draw_data->CmdLists[n];
    memcpy(vtx_dst, cmd_list->VtxBuffer.Data,
           (size_t)cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
    memcpy(idx_dst, cmd_list->IdxBuffer.Data,
           (size_t)cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
    vtx_dst += cmd_list->VtxBuffer.Size;
    idx_dst += cmd_list->IdxBuffer.Size;
  }

  /* Calculate actual data size and align to 4 bytes */
  uint64_t vtx_data_size
    = (uint64_t)(vtx_dst - vtx_staging) * sizeof(ImDrawVert);
  uint64_t idx_data_size
    = (uint64_t)(idx_dst - idx_staging) * sizeof(ImDrawIdx);
  uint64_t vtx_size_aligned = (vtx_data_size + 3) & ~(uint64_t)3;
  uint64_t idx_size_aligned = (idx_data_size + 3) & ~(uint64_t)3;

  /* Upload to GPU */
  if (vtx_data_size > 0 && idx_data_size > 0) {
    wgpuQueueWriteBuffer(wgpu_context->queue, overlay_state.vertex_buffer, 0,
                         vtx_staging, vtx_size_aligned);
    wgpuQueueWriteBuffer(wgpu_context->queue, overlay_state.index_buffer, 0,
                         idx_staging, idx_size_aligned);
  }

  free(vtx_staging);
  free(idx_staging);
}

/* -------------------------------------------------------------------------- *
 * Public API Implementation
 * -------------------------------------------------------------------------- */

int imgui_overlay_init(wgpu_context_t* wgpu_context)
{
  if (overlay_state.initialized) {
    return EXIT_SUCCESS;
  }

  /* Create ImGui context */
  overlay_state.imgui_context = igCreateContext(NULL);
  igSetCurrentContext(overlay_state.imgui_context);

  /* Setup ImGui IO */
  ImGuiIO* io                   = igGetIO();
  io->BackendRendererName       = "imgui_impl_webgpu";
  io->DisplaySize.x             = (float)wgpu_context->width;
  io->DisplaySize.y             = (float)wgpu_context->height;
  io->DisplayFramebufferScale.x = 1.0f;
  io->DisplayFramebufferScale.y = 1.0f;

  /* Setup keyboard mapping */
  io->KeyMap[ImGuiKey_Tab]        = KEY_TAB;
  io->KeyMap[ImGuiKey_LeftArrow]  = KEY_LEFT;
  io->KeyMap[ImGuiKey_RightArrow] = KEY_RIGHT;
  io->KeyMap[ImGuiKey_UpArrow]    = KEY_UP;
  io->KeyMap[ImGuiKey_DownArrow]  = KEY_DOWN;
  io->KeyMap[ImGuiKey_PageUp]     = KEY_PAGE_UP;
  io->KeyMap[ImGuiKey_PageDown]   = KEY_PAGE_DOWN;
  io->KeyMap[ImGuiKey_Home]       = KEY_HOME;
  io->KeyMap[ImGuiKey_End]        = KEY_END;
  io->KeyMap[ImGuiKey_Insert]     = KEY_INSERT;
  io->KeyMap[ImGuiKey_Delete]     = KEY_DELETE;
  io->KeyMap[ImGuiKey_Backspace]  = KEY_BACKSPACE;
  io->KeyMap[ImGuiKey_Space]      = KEY_SPACE;
  io->KeyMap[ImGuiKey_Enter]      = KEY_ENTER;
  io->KeyMap[ImGuiKey_Escape]     = KEY_ESCAPE;
  io->KeyMap[ImGuiKey_A]          = KEY_A;
  io->KeyMap[ImGuiKey_C]          = KEY_C;
  io->KeyMap[ImGuiKey_V]          = KEY_V;
  io->KeyMap[ImGuiKey_X]          = KEY_X;
  io->KeyMap[ImGuiKey_Y]          = KEY_Y;
  io->KeyMap[ImGuiKey_Z]          = KEY_Z;

  /* Add default font */
  ImFontAtlas_AddFontDefault(io->Fonts, NULL);

  /* Create resources */
  create_font_texture(wgpu_context);
  create_render_pipeline(wgpu_context);
  create_uniform_buffer(wgpu_context);
  create_bind_group(wgpu_context);

  /* Initialize render pass descriptor */
  overlay_state.color_attachment = (WGPURenderPassColorAttachment){
    .view       = NULL,
    .loadOp     = WGPULoadOp_Load,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor){0.0f, 0.0f, 0.0f, 0.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  };

  overlay_state.render_pass_descriptor = (WGPURenderPassDescriptor){
    .label                = STRVIEW("ImGui render pass"),
    .colorAttachmentCount = 1,
    .colorAttachments     = &overlay_state.color_attachment,
  };

  overlay_state.initialized = true;
  return EXIT_SUCCESS;
}

void imgui_overlay_new_frame(wgpu_context_t* wgpu_context, float delta_time)
{
  if (!overlay_state.initialized) {
    return;
  }

  ImGuiIO* io       = igGetIO();
  io->DisplaySize.x = (float)wgpu_context->width;
  io->DisplaySize.y = (float)wgpu_context->height;
  io->DeltaTime     = delta_time > 0.0f ? delta_time : (1.0f / 60.0f);

  igNewFrame();
}

void imgui_overlay_render(wgpu_context_t* wgpu_context)
{
  if (!overlay_state.initialized) {
    return;
  }

  /* Render ImGui */
  igRender();
  ImDrawData* draw_data = igGetDrawData();

  if (draw_data->TotalVtxCount == 0) {
    return;
  }

  /* Update buffers */
  update_buffers(wgpu_context, draw_data);

  /* Update uniform buffer (projection matrix) */
  {
    float L         = draw_data->DisplayPos.x;
    float R         = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    float T         = draw_data->DisplayPos.y;
    float B         = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    float mvp[4][4] = {
      {2.0f / (R - L), 0.0f, 0.0f, 0.0f},
      {0.0f, 2.0f / (T - B), 0.0f, 0.0f},
      {0.0f, 0.0f, 0.5f, 0.0f},
      {(R + L) / (L - R), (T + B) / (B - T), 0.5f, 1.0f},
    };
    wgpuQueueWriteBuffer(wgpu_context->queue, overlay_state.uniform_buffer, 0,
                         mvp, sizeof(mvp));
  }

  /* Create command encoder */
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Render pass */
  overlay_state.color_attachment.view = wgpu_context->swapchain_view;
  WGPURenderPassEncoder pass          = wgpuCommandEncoderBeginRenderPass(
    encoder, &overlay_state.render_pass_descriptor);

  /* Render ImGui draw data */
  wgpuRenderPassEncoderSetPipeline(pass, overlay_state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, overlay_state.bind_group, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, overlay_state.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    pass, overlay_state.index_buffer,
    sizeof(ImDrawIdx) == 2 ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32, 0,
    WGPU_WHOLE_SIZE);

  /* Render command lists */
  int global_vtx_offset = 0;
  int global_idx_offset = 0;
  ImVec2 clip_off       = draw_data->DisplayPos;
  for (int n = 0; n < draw_data->CmdListsCount; n++) {
    const ImDrawList* cmd_list = draw_data->CmdLists[n];
    for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
      const ImDrawCmd* pcmd = &cmd_list->CmdBuffer.Data[cmd_i];
      if (pcmd->UserCallback) {
        pcmd->UserCallback(cmd_list, pcmd);
      }
      else {
        /* Set scissor rectangle */
        ImVec2 clip_min
          = {pcmd->ClipRect.x - clip_off.x, pcmd->ClipRect.y - clip_off.y};
        ImVec2 clip_max
          = {pcmd->ClipRect.z - clip_off.x, pcmd->ClipRect.w - clip_off.y};
        if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y) {
          continue;
        }

        wgpuRenderPassEncoderSetScissorRect(
          pass, (uint32_t)clip_min.x, (uint32_t)clip_min.y,
          (uint32_t)(clip_max.x - clip_min.x),
          (uint32_t)(clip_max.y - clip_min.y));

        /* Draw */
        wgpuRenderPassEncoderDrawIndexed(pass, pcmd->ElemCount, 1,
                                         (uint32_t)global_idx_offset,
                                         global_vtx_offset, 0);
      }
      global_idx_offset += (int)pcmd->ElemCount;
    }
    global_vtx_offset += cmd_list->VtxBuffer.Size;
  }

  wgpuRenderPassEncoderEnd(pass);

  /* Submit */
  WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command);

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(CommandBuffer, command)
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)
}

void imgui_overlay_handle_input(wgpu_context_t* wgpu_context,
                                const input_event_t* event)
{
  UNUSED_VAR(wgpu_context);

  if (!overlay_state.initialized) {
    return;
  }

  ImGuiIO* io = igGetIO();

  switch (event->type) {
    case INPUT_EVENT_TYPE_KEY_DOWN:
      if (event->key_code < KEY_NUM) {
        io->KeysDown[event->key_code] = true;
      }
      io->KeyCtrl  = (event->key_code == KEY_LEFT_CONTROL
                     || event->key_code == KEY_RIGHT_CONTROL);
      io->KeyShift = (event->key_code == KEY_LEFT_SHIFT
                      || event->key_code == KEY_RIGHT_SHIFT);
      io->KeyAlt
        = (event->key_code == KEY_LEFT_ALT || event->key_code == KEY_RIGHT_ALT);
      io->KeySuper = (event->key_code == KEY_LEFT_SUPER
                      || event->key_code == KEY_RIGHT_SUPER);
      break;

    case INPUT_EVENT_TYPE_KEY_UP:
      if (event->key_code < KEY_NUM) {
        io->KeysDown[event->key_code] = false;
      }
      io->KeyCtrl  = false;
      io->KeyShift = false;
      io->KeyAlt   = false;
      io->KeySuper = false;
      break;

    case INPUT_EVENT_TYPE_CHAR:
      if (event->char_code > 0 && event->char_code < 0x10000) {
        ImGuiIO_AddInputCharacter(io, (unsigned short)event->char_code);
      }
      break;

    case INPUT_EVENT_TYPE_MOUSE_DOWN:
      if (event->mouse_button == BUTTON_LEFT) {
        io->MouseDown[0] = true;
      }
      else if (event->mouse_button == BUTTON_RIGHT) {
        io->MouseDown[1] = true;
      }
      else if (event->mouse_button == BUTTON_MIDDLE) {
        io->MouseDown[2] = true;
      }
      break;

    case INPUT_EVENT_TYPE_MOUSE_UP:
      if (event->mouse_button == BUTTON_LEFT) {
        io->MouseDown[0] = false;
      }
      else if (event->mouse_button == BUTTON_RIGHT) {
        io->MouseDown[1] = false;
      }
      else if (event->mouse_button == BUTTON_MIDDLE) {
        io->MouseDown[2] = false;
      }
      break;

    case INPUT_EVENT_TYPE_MOUSE_MOVE:
      io->MousePos.x = event->mouse_x;
      io->MousePos.y = event->mouse_y;
      break;

    case INPUT_EVENT_TYPE_MOUSE_SCROLL:
      io->MouseWheelH += event->scroll_x;
      io->MouseWheel += event->scroll_y;
      break;

    case INPUT_EVENT_TYPE_RESIZED:
      io->DisplaySize.x = (float)event->window_width;
      io->DisplaySize.y = (float)event->window_height;
      break;

    default:
      break;
  }
}

void imgui_overlay_shutdown(void)
{
  if (!overlay_state.initialized) {
    return;
  }

  /* Release buffers */
  WGPU_RELEASE_RESOURCE(Buffer, overlay_state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, overlay_state.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, overlay_state.uniform_buffer)

  /* Release pipeline resources */
  WGPU_RELEASE_RESOURCE(RenderPipeline, overlay_state.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, overlay_state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, overlay_state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, overlay_state.bind_group)

  /* Release font resources */
  WGPU_RELEASE_RESOURCE(Sampler, overlay_state.font_sampler)
  WGPU_RELEASE_RESOURCE(TextureView, overlay_state.font_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, overlay_state.font_texture)

  /* Destroy ImGui context */
  if (overlay_state.imgui_context) {
    igDestroyContext(overlay_state.imgui_context);
    overlay_state.imgui_context = NULL;
  }

  overlay_state.initialized = false;
}

bool imgui_overlay_want_capture_mouse(void)
{
  if (!overlay_state.initialized) {
    return false;
  }
  return igGetIO()->WantCaptureMouse;
}

bool imgui_overlay_want_capture_keyboard(void)
{
  if (!overlay_state.initialized) {
    return false;
  }
  return igGetIO()->WantCaptureKeyboard;
}

/* -------------------------------------------------------------------------- *
 * Convenience Widget Functions
 * -------------------------------------------------------------------------- */

bool imgui_overlay_header(const char* caption)
{
  return igCollapsingHeader(caption, ImGuiTreeNodeFlags_DefaultOpen);
}

bool imgui_overlay_checkbox(const char* caption, bool* value)
{
  return igCheckbox(caption, value);
}

bool imgui_overlay_slider_float(const char* caption, float* value, float min,
                                float max, const char* format)
{
  return igSliderFloat(caption, value, min, max, format ? format : "%.3f", 0);
}

bool imgui_overlay_slider_int(const char* caption, int32_t* value, int32_t min,
                              int32_t max)
{
  return igSliderInt(caption, value, min, max, "%d");
}

bool imgui_overlay_input_float(const char* caption, float* value, float step,
                               const char* format)
{
  return igInputFloat(caption, value, step, step * 10.0f,
                      format ? format : "%.3f", 0);
}

bool imgui_overlay_combo_box(const char* caption, int32_t* item_index,
                             const char** items, uint32_t item_count)
{
  if (item_count == 0) {
    return false;
  }
  return igCombo(caption, item_index, items, (int)item_count, (int)item_count);
}

bool imgui_overlay_button(const char* caption)
{
  return igSmallButton(caption);
}

void imgui_overlay_text(const char* format_str, ...)
{
  va_list args;
  va_start(args, format_str);
  igTextV(format_str, args);
  va_end(args);
}

bool imgui_overlay_color_edit4(const char* caption, float color[4])
{
  return igColorEdit4(caption, color, ImGuiColorEditFlags_Float);
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* imgui_vertex_shader_wgsl = CODE(
  struct VertexInput {
    @location(0) position : vec2f,
    @location(1) uv : vec2f,
    @location(2) color : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
    @location(1) color : vec4f,
  };

  struct Uniforms { projection_matrix : mat4x4f, };

  @group(0) @binding(0) var<uniform> uniforms : Uniforms;

  @vertex fn main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.position = uniforms.projection_matrix * vec4f(in.position, 0.0, 1.0);
    out.uv       = in.uv;
    out.color    = in.color;
    return out;
  }
);

static const char* imgui_fragment_shader_wgsl = CODE(
  struct FragmentInput {
    @location(0) uv : vec2f,
    @location(1) color : vec4f,
  };

  @group(0) @binding(1) var texture_sampler : sampler;
  @group(0) @binding(2) var texture_view : texture_2d<f32>;

  @fragment fn main(in : FragmentInput) -> @location(0) vec4f {
    return in.color * textureSample(texture_view, texture_sampler, in.uv);
  }
);
// clang-format on
