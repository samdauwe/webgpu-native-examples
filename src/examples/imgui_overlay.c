#include "webgpu/wgpu_common.h"

#include <string.h>

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
 * WebGPU Example - ImGui Overlay
 *
 * Generates and renders a complex user interface with multiple windows,
 * controls and user interaction on top of a 3D scene. The UI is generated using
 * Dear ImGUI and updated each frame.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/imgui
 * https://github.com/ocornut/imgui
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader Code
 * -------------------------------------------------------------------------- */

static const char* imgui_vertex_shader_wgsl;
static const char* imgui_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * ImGui Overlay State
 * -------------------------------------------------------------------------- */

static struct {
  /* ImGui context */
  struct ImGuiContext* imgui_context;

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

  /* UI state */
  bool show_demo_window;
  bool show_another_window;
  float clear_color[4];
  float demo_float;
  int demo_counter;

  /* Timing */
  uint64_t last_frame_time;

  /* Render pass descriptor (pre-allocated for performance) */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
} state = {
  .show_demo_window    = true,
  .show_another_window = true,
  .clear_color         = {0.45f, 0.55f, 0.60f, 1.00f},
  .demo_float          = 0.0f,
  .demo_counter        = 0,
  .last_frame_time     = 0,
};

/* -------------------------------------------------------------------------- *
 * ImGui Helper Functions
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
  uint32_t pixels_size_bytes = font_width * font_height * bytes_per_pixel;

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
  state.font_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(state.font_texture);

  /* Upload texture data */
  wgpuQueueWriteTexture(
    wgpu_context->queue,
    &(WGPUTexelCopyTextureInfo){
      .texture  = state.font_texture,
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
  state.font_texture_view = wgpuTextureCreateView(
    state.font_texture, &(WGPUTextureViewDescriptor){
                          .label           = STRVIEW("ImGui font texture view"),
                          .format          = WGPUTextureFormat_RGBA8Unorm,
                          .dimension       = WGPUTextureViewDimension_2D,
                          .baseMipLevel    = 0,
                          .mipLevelCount   = 1,
                          .baseArrayLayer  = 0,
                          .arrayLayerCount = 1,
                          .aspect          = WGPUTextureAspect_All,
                        });
  ASSERT(state.font_texture_view);

  /* Create sampler */
  state.font_sampler = wgpuDeviceCreateSampler(
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
  ASSERT(state.font_sampler);

  /* Store texture ID in ImGui */
  io->Fonts->TexID = (ImTextureID)(intptr_t)state.font_texture_view;
}

/**
 * @brief Create ImGui render pipeline
 */
static void create_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry){
      /* Binding 0: Uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = 64, /* mat4x4 */
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
    [2] = (WGPUBindGroupLayoutEntry){
      /* Binding 2: Texture view */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("ImGui bind group layout"),
                            .entryCount = 3,
                            .entries    = bgl_entries,
                          });
  ASSERT(state.bind_group_layout);

  /* Create pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("ImGui pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout);

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

  /* Color target state */
  WGPUBlendState blend_state = {
    .color
    = (WGPUBlendComponent){
      .operation = WGPUBlendOperation_Add,
      .srcFactor = WGPUBlendFactor_SrcAlpha,
      .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
    },
    .alpha
    = (WGPUBlendComponent){
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
  state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("ImGui render pipeline"),
      .layout = state.pipeline_layout,
      .vertex
      = (WGPUVertexState){
        .module      = vs_module,
        .entryPoint  = STRVIEW("main"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .primitive
      = (WGPUPrimitiveState){
        .topology         = WGPUPrimitiveTopology_TriangleList,
        .stripIndexFormat = WGPUIndexFormat_Undefined,
        .frontFace        = WGPUFrontFace_CW,
        .cullMode         = WGPUCullMode_None,
      },
      .multisample
      = (WGPUMultisampleState){
        .count                  = 1,
        .mask                   = 0xFFFFFFFF,
        .alphaToCoverageEnabled = false,
      },
      .fragment
      = &(WGPUFragmentState){
        .module      = fs_module,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets     = &color_target_state,
      },
    });
  ASSERT(state.pipeline);

  /* Release shader modules */
  WGPU_RELEASE_RESOURCE(ShaderModule, vs_module)
  WGPU_RELEASE_RESOURCE(ShaderModule, fs_module)
}

/**
 * @brief Create uniform buffer for projection matrix
 */
static void create_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("ImGui uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = 64, /* mat4x4 */
    });
  ASSERT(state.uniform_buffer);
}

/**
 * @brief Create bind group
 */
static void create_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.uniform_buffer,
      .size    = 64,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.font_sampler,
    },
    [2] = (WGPUBindGroupEntry){
      .binding     = 2,
      .textureView = state.font_texture_view,
    },
  };

  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("ImGui bind group"),
                            .layout     = state.bind_group_layout,
                            .entryCount = 3,
                            .entries    = bg_entries,
                          });
  ASSERT(state.bind_group);
}

/**
 * @brief Update vertex and index buffers
 */
static void update_buffers(wgpu_context_t* wgpu_context, ImDrawData* draw_data)
{
  uint64_t vertex_size = draw_data->TotalVtxCount * sizeof(ImDrawVert);
  uint64_t index_size  = draw_data->TotalIdxCount * sizeof(ImDrawIdx);

  /* Create or resize vertex buffer */
  if (!state.vertex_buffer || state.vertex_buffer_size < vertex_size) {
    if (state.vertex_buffer) {
      WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
    }
    state.vertex_buffer_size = vertex_size + 5000 * sizeof(ImDrawVert);
    state.vertex_buffer      = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
             .label = STRVIEW("ImGui vertex buffer"),
             .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
             .size  = state.vertex_buffer_size,
      });
    ASSERT(state.vertex_buffer);
  }

  /* Create or resize index buffer */
  if (!state.index_buffer || state.index_buffer_size < index_size) {
    if (state.index_buffer) {
      WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)
    }
    state.index_buffer_size = index_size + 10000 * sizeof(ImDrawIdx);
    state.index_buffer      = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
             .label = STRVIEW("ImGui index buffer"),
             .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
             .size  = state.index_buffer_size,
      });
    ASSERT(state.index_buffer);
  }

  /* Upload data - use staging buffers like the old implementation */
  ImDrawVert* vtx_staging = (ImDrawVert*)malloc(vertex_size);
  ImDrawIdx* idx_staging  = (ImDrawIdx*)malloc(index_size);

  ImDrawVert* vtx_dst = vtx_staging;
  ImDrawIdx* idx_dst  = idx_staging;

  for (int n = 0; n < draw_data->CmdListsCount; n++) {
    const ImDrawList* cmd_list = draw_data->CmdLists[n];
    memcpy(vtx_dst, cmd_list->VtxBuffer.Data,
           cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
    memcpy(idx_dst, cmd_list->IdxBuffer.Data,
           cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
    vtx_dst += cmd_list->VtxBuffer.Size;
    idx_dst += cmd_list->IdxBuffer.Size;
  }

  /* Calculate actual data size and align to 4 bytes */
  uint64_t vtx_data_size    = (vtx_dst - vtx_staging) * sizeof(ImDrawVert);
  uint64_t idx_data_size    = (idx_dst - idx_staging) * sizeof(ImDrawIdx);
  uint64_t vtx_size_aligned = (vtx_data_size + 3) & ~3;
  uint64_t idx_size_aligned = (idx_data_size + 3) & ~3;

  /* Upload to GPU */
  if (vtx_data_size > 0 && idx_data_size > 0) {
    wgpuQueueWriteBuffer(wgpu_context->queue, state.vertex_buffer, 0,
                         vtx_staging, vtx_size_aligned);
    wgpuQueueWriteBuffer(wgpu_context->queue, state.index_buffer, 0,
                         idx_staging, idx_size_aligned);
  }

  free(vtx_staging);
  free(idx_staging);
}

/**
 * @brief Initialize ImGui context and resources
 */
static int init(wgpu_context_t* wgpu_context)
{
  /* Initialize sokol_time */
  stm_setup();

  /* Create ImGui context */
  state.imgui_context = igCreateContext(NULL);
  igSetCurrentContext(state.imgui_context);

  /* Setup ImGui IO */
  ImGuiIO* io                   = igGetIO();
  io->BackendRendererName       = "imgui_impl_webgpu";
  io->DisplaySize.x             = (float)wgpu_context->width;
  io->DisplaySize.y             = (float)wgpu_context->height;
  io->DisplayFramebufferScale.x = 1.0f;
  io->DisplayFramebufferScale.y = 1.0f;

  /* Setup keyboard mapping for ImGui */
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

  /* Create font texture */
  create_font_texture(wgpu_context);

  /* Create render pipeline */
  create_render_pipeline(wgpu_context);

  /* Create uniform buffer */
  create_uniform_buffer(wgpu_context);

  /* Create bind group */
  create_bind_group(wgpu_context);

  /* Initialize render pass descriptor */
  state.color_attachment = (WGPURenderPassColorAttachment){
    .view       = NULL, /* Set each frame */
    .loadOp     = WGPULoadOp_Load,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor){0.0f, 0.0f, 0.0f, 0.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  };

  state.render_pass_descriptor = (WGPURenderPassDescriptor){
    .label                = STRVIEW("ImGui - Render pass"),
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  };

  return EXIT_SUCCESS;
}

/**
 * @brief Frame callback - render ImGui interface
 */
static int frame(wgpu_context_t* wgpu_context)
{
  /* Calculate delta time */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Update ImGui IO */
  ImGuiIO* io       = igGetIO();
  io->DisplaySize.x = (float)wgpu_context->width;
  io->DisplaySize.y = (float)wgpu_context->height;
  io->DeltaTime     = delta_time > 0.0f ? delta_time : (1.0f / 60.0f);

  /* Start new ImGui frame */
  igNewFrame();

  /* Show demo window */
  if (state.show_demo_window) {
    igShowDemoWindow(&state.show_demo_window);
  }

  /* Show custom window */
  {
    igBegin("Hello, world!", NULL, 0);
    igText("This is some useful text");
    igCheckbox("Demo window", &state.show_demo_window);
    igCheckbox("Another window", &state.show_another_window);

    igSliderFloat("Float", &state.demo_float, 0.0f, 1.0f, "%.3f", 0);
    igColorEdit3("clear color", state.clear_color, 0);

    ImVec2 button_size = {0, 0};
    if (igButton("Button", button_size)) {
      state.demo_counter++;
    }
    igSameLine(0.0f, -1.0f);
    igText("counter = %d", state.demo_counter);

    igText("Application average %.3f ms/frame (%.1f FPS)",
           1000.0f / igGetIO()->Framerate, igGetIO()->Framerate);
    igEnd();
  }

  /* Show another window */
  if (state.show_another_window) {
    igBegin("imgui Another Window", &state.show_another_window, 0);
    igText("Hello from imgui");
    ImVec2 button_size = {0, 0};
    if (igButton("Close me", button_size)) {
      state.show_another_window = false;
    }
    igEnd();
  }

  /* Render ImGui */
  igRender();
  ImDrawData* draw_data = igGetDrawData();

  /* Update vertex and index buffers */
  if (draw_data->TotalVtxCount > 0) {
    update_buffers(wgpu_context, draw_data);
  }

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
    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0, mvp,
                         sizeof(mvp));
  }

  /* Create command encoder */
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("ImGui - Command encoder"),
                          });

  /* Render pass */
  state.color_attachment.view
    = wgpu_context->swapchain_view; /* Set current frame's view */
  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(encoder, &state.render_pass_descriptor);

  /* Render ImGui draw data */
  if (draw_data->TotalVtxCount > 0) {
    wgpuRenderPassEncoderSetPipeline(pass, state.pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_group, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.index_buffer,
      sizeof(ImDrawIdx) == 2 ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32,
      0, WGPU_WHOLE_SIZE);

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
          wgpuRenderPassEncoderDrawIndexed(
            pass, pcmd->ElemCount, 1, global_idx_offset, global_vtx_offset, 0);
        }
        global_idx_offset += pcmd->ElemCount;
      }
      global_vtx_offset += cmd_list->VtxBuffer.Size;
    }
  }

  wgpuRenderPassEncoderEnd(pass);

  /* Get command buffer */
  WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);

  /* Submit command buffer */
  wgpuQueueSubmit(wgpu_context->queue, 1, &command);

  /* Release resources */
  WGPU_RELEASE_RESOURCE(CommandBuffer, command)
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)

  return EXIT_SUCCESS;
}

/**
 * @brief Input event callback for window resize and ImGui interaction
 */
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* event)
{
  UNUSED_VAR(wgpu_context);

  ImGuiIO* io = igGetIO();

  switch (event->type) {
    case INPUT_EVENT_TYPE_KEY_DOWN:
      if (event->key_code < KEY_NUM) {
        io->KeysDown[event->key_code] = true;
      }
      /* Update modifier keys */
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
      /* Update modifier keys */
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
      /* Map buttons: ImGui expects [0]=Left, [1]=Right, [2]=Middle */
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
      /* Map buttons: ImGui expects [0]=Left, [1]=Right, [2]=Middle */
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
      /* Window resize handled automatically by framework */
      io->DisplaySize.x = (float)event->window_width;
      io->DisplaySize.y = (float)event->window_height;
      break;

    default:
      break;
  }
}

/**
 * @brief Cleanup and release resources
 */
static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Release buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)

  /* Release pipeline resources */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)

  /* Release font resources */
  WGPU_RELEASE_RESOURCE(Sampler, state.font_sampler)
  WGPU_RELEASE_RESOURCE(TextureView, state.font_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, state.font_texture)

  /* Destroy ImGui context */
  if (state.imgui_context) {
    igDestroyContext(state.imgui_context);
    state.imgui_context = NULL;
  }
}

/**
 * @brief Main entry point
 */
int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "ImGui Overlay",
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
/* Vertex shader */
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

  @vertex fn main(in : VertexInput)->VertexOutput {
    var out : VertexOutput;
    out.position = uniforms.projection_matrix * vec4f(in.position, 0.0, 1.0);
    out.uv       = in.uv;
    out.color    = in.color;
    return out;
  }
);

/* Fragment shader */
static const char* imgui_fragment_shader_wgsl = CODE(
  struct FragmentInput {
    @location(0) uv : vec2f,
    @location(1) color : vec4f,
  };

  @group(0) @binding(1) var texture_sampler : sampler;
  @group(0) @binding(2) var texture_view : texture_2d<f32>;

  @fragment fn main(in : FragmentInput)->@location(0) vec4f {
    return in.color * textureSample(texture_view, texture_sampler, in.uv);
  }
);
// clang-format on
