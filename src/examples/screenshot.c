#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Saving Framebuffer To Screenshot
 *
 * This example shows how to capture an image by rendering a scene to a texture,
 * copying the texture to a buffer, and retrieving the image from the buffer so
 * that it can be stored into a png image. Two render pipelines are used in this
 * example: one for rendering the scene in a window and another pipeline for
 * offscreen rendering. Note that a single offscreen render pipeline would be
 * sufficient for "taking a screenshot," with the added benefit that this method
 * would not require a window to be created.
 *
 * Ref:
 * https://github.com/gfx-rs/wgpu/tree/master/wgpu/examples/capture
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/screenshot
 * -------------------------------------------------------------------------- */

#define COPY_BYTES_PER_ROW_ALIGNMENT 256u

static struct gltf_model_t* dragon;
static wgpu_buffer_t uniform_buffer;

static struct {
  mat4 projection;
  mat4 model;
  mat4 view;
} ubo_vs = {0};

static WGPUPipelineLayout pipeline_layout    = NULL;
static WGPUBindGroupLayout bind_group_layout = NULL;
static WGPUBindGroup bind_group              = NULL;

static const char* screenshot_filename = "Screenshot.png";
static bool screenshot_requested       = false;
static bool screenshot_saved           = false;

static struct scene_rendering_t {
  WGPURenderPipeline pipeline;
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass;
} scene_rendering = {0};

static struct offscreen_rendering_t {
  /* Framebuffer for offscreen rendering */
  struct {
    WGPUTextureFormat format;
    WGPUTexture texture;
    WGPUTextureView texture_view;
  } color, depth_stencil;
  WGPURenderPipeline pipeline;
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass;
  /* The pixel buffer lets us retrieve the framebuffer data as an array */
  struct {
    wgpu_buffer_t buffer;
    bool buffer_mapped;
    struct {
      uint64_t width;
      uint64_t height;
      uint64_t unpadded_bytes_per_row;
      uint64_t padded_bytes_per_row;
    } buffer_dimensions;
  } pixel_data;
} offscreen_rendering = {0};

static const char* example_title = "Saving Framebuffer To Screenshot";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 512.0f);
  camera_set_rotation(context->camera, (vec3){25.0f, 23.75f, 0.0f});
  camera_set_translation(context->camera, (vec3){0.0f, 0.0f, -3.0f});
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors;
  dragon = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/chinesedragon.gltf",
    .file_loading_flags = gltf_loading_flags,
  });
  ASSERT(dragon != NULL);
}

// It is a WebGPU requirement that ImageCopyBuffer.layout.bytes_per_row %
// wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0 So we calculate padded_bytes_per_row
// by rounding unpadded_bytes_per_row up to the next multiple of
// wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.
// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
static void calculate_buffer_dimensions(uint32_t width, uint32_t height)
{
  const uint32_t bytes_per_pixel        = sizeof(uint8_t) * 4;
  const uint32_t unpadded_bytes_per_row = width * bytes_per_pixel;
  const uint32_t align = (uint32_t)COPY_BYTES_PER_ROW_ALIGNMENT;
  const uint32_t padded_bytes_per_row_padding
    = (align - unpadded_bytes_per_row % align) % align;
  const uint32_t padded_bytes_per_row
    = unpadded_bytes_per_row + padded_bytes_per_row_padding;
  {
    offscreen_rendering.pixel_data.buffer_dimensions.width  = width;
    offscreen_rendering.pixel_data.buffer_dimensions.height = height;
    offscreen_rendering.pixel_data.buffer_dimensions.unpadded_bytes_per_row
      = unpadded_bytes_per_row;
    offscreen_rendering.pixel_data.buffer_dimensions.padded_bytes_per_row
      = padded_bytes_per_row;
  }
}

static void prepare_offscreen(wgpu_context_t* wgpu_context)
{
  /* Prepare pixel output buffer */
  calculate_buffer_dimensions(wgpu_context->surface.width,
                              wgpu_context->surface.height);

  /* The output buffer lets us retrieve the data as an array */
  offscreen_rendering.pixel_data.buffer = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Pixel output buffer",
      .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
      .size
      = (uint64_t)(offscreen_rendering.pixel_data.buffer_dimensions
                     .padded_bytes_per_row
                   * offscreen_rendering.pixel_data.buffer_dimensions.height),
    });

  /* Attachment formats */
  offscreen_rendering.color.format = WGPUTextureFormat_RGBA8UnormSrgb;
  offscreen_rendering.depth_stencil.format
    = WGPUTextureFormat_Depth24PlusStencil8;

  /* Create the texture */
  WGPUExtent3D texture_extent = {
    .width  = offscreen_rendering.pixel_data.buffer_dimensions.width,
    .height = offscreen_rendering.pixel_data.buffer_dimensions.height,
    .depthOrArrayLayers = 1,
  };

  /* Color attachment */
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Color attachment - Texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = offscreen_rendering.color.format,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
    };
    offscreen_rendering.color.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(offscreen_rendering.color.texture != NULL);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_desc = {
      .label           = "Color attachment - Texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    offscreen_rendering.color.texture_view = wgpuTextureCreateView(
      offscreen_rendering.color.texture, &texture_view_desc);
    ASSERT(offscreen_rendering.color.texture_view != NULL);
  }

  /* Depth stencil attachment */
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Depth stencil attachment - Texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = offscreen_rendering.depth_stencil.format,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
    };
    offscreen_rendering.depth_stencil.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(offscreen_rendering.depth_stencil.texture != NULL);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_desc = {
      .label           = "Depth stencil attachment - Texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    offscreen_rendering.depth_stencil.texture_view = wgpuTextureCreateView(
      offscreen_rendering.depth_stencil.texture, &texture_view_desc);
    ASSERT(offscreen_rendering.depth_stencil.texture_view != NULL);
  }

  // Create a separate render pass for the offscreen rendering as it may differ
  // from the one used for scene rendering
  /* Color attachment */
  offscreen_rendering.render_pass.color_attachment[0]
    = (WGPURenderPassColorAttachment) {
      .view       = offscreen_rendering.color.texture_view,
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  /* Depth attachment */
  offscreen_rendering.render_pass.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view           = offscreen_rendering.depth_stencil.texture_view,
      .depthLoadOp    = WGPULoadOp_Clear,
      .depthStoreOp   = WGPUStoreOp_Store,
      .stencilLoadOp  = WGPULoadOp_Clear,
      .stencilStoreOp = WGPUStoreOp_Store,
    };

  /* Render pass descriptor */
  offscreen_rendering.render_pass.render_pass_descriptor
    = (WGPURenderPassDescriptor){
      .label                = "Render pass descriptor",
      .colorAttachmentCount = 1,
      .colorAttachments     = offscreen_rendering.render_pass.color_attachment,
      .depthStencilAttachment
      = &offscreen_rendering.render_pass.depth_stencil_attachment,
    };
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout entries */
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0: Vertex shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(ubo_vs),
      },
      .sampler = {0},
    }
  };

  /* Create the bind group layout */
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL)

  /* Create the pipeline layout */
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0: Vertex shader uniform buffer */
        .binding = 0,
        .buffer  = uniform_buffer.buffer,
        .offset  = 0,
        .size    = uniform_buffer.size,
      },
    };

  bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "Bind group",
                            .layout     = bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(bind_group != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
  scene_rendering.render_pass.color_attachment[0]
    = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 0.0f, /* Set to 1.0f for transparent background */
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  scene_rendering.render_pass.render_pass_descriptor
    = (WGPURenderPassDescriptor){
      .label                  = "Render pass descriptor",
      .colorAttachmentCount   = 1,
      .colorAttachments       = scene_rendering.render_pass.color_attachment,
      .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
    };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  // Vertex buffer layout
  WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
    gltf_model,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal),
    // Location 2: Vertex color
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_Color));

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                /* Vertex shader SPIR-V */
                .label = "Mesh - Vertex shader SPIR-V",
                .file  = "shaders/screenshot/mesh.vert.spv",
              },
              .buffer_count = 1,
              .buffers      = &gltf_model_vertex_buffer_layout,
            });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                /* Fragment shader SPIR-V */
                .label = "Mesh - Fragment shader SPIR-V",
                .file  = "shaders/screenshot/mesh.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create scene rendering pipeline using the specified states
  scene_rendering.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "GLTF model - Render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(scene_rendering.pipeline != NULL);

  // Create offscreen rendering pipeline using the specified states
  color_target_state.format    = offscreen_rendering.color.format;
  offscreen_rendering.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Offscreen render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(offscreen_rendering.pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  glm_mat4_copy(context->camera->matrices.perspective, ubo_vs.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_vs.view);
  glm_mat4_identity(ubo_vs.model);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer.buffer, 0,
                          &ubo_vs, uniform_buffer.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Mesh vertex shader uniform buffer block */
  uniform_buffer = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Mesh vertex shader - Uniform buffer block",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_vs),
    });

  /* Update uniform buffer block data and uniform buffer */
  update_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_offscreen(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    if (imgui_overlay_button(context->imgui_overlay, "Take screenshot")) {
      screenshot_requested = true;
    }
    if (screenshot_saved) {
      imgui_overlay_text("Screenshot saved as: %s", screenshot_filename);
    }
  }
}

static WGPUCommandBuffer
build_command_buffer(wgpu_context_t* wgpu_context,
                     WGPURenderPassDescriptor* render_pass_descriptor,
                     WGPURenderPipeline pipeline, bool overlay_ui)
{
  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, render_pass_descriptor);

  // Render glTF model
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);
  wgpu_gltf_model_draw(dragon, (wgpu_gltf_model_render_options_t){0});

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay
  if (overlay_ui) {
    draw_ui(wgpu_context->context, example_on_update_ui_overlay);
  }

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static WGPUCommandBuffer
build_copy_texture_to_buffer_command_buffer(wgpu_context_t* wgpu_context)
{
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  wgpuCommandEncoderCopyTextureToBuffer(wgpu_context->cmd_enc,
    // Source
    &(WGPUImageCopyTexture){
      .texture  = offscreen_rendering.color.texture,
      .mipLevel =0,
    },
    // Destination
    &(WGPUImageCopyBuffer){
      .buffer  = offscreen_rendering.pixel_data.buffer.buffer,
      .layout = (WGPUTextureDataLayout) {
        .offset      = 0,
        .bytesPerRow = offscreen_rendering.pixel_data.buffer_dimensions.padded_bytes_per_row,
        .rowsPerImage = offscreen_rendering.pixel_data.buffer_dimensions.height,
      },
    },
    // CopySize
    &(WGPUExtent3D) {
      .width  = offscreen_rendering.pixel_data.buffer_dimensions.width,
      .height = offscreen_rendering.pixel_data.buffer_dimensions.height,
      .depthOrArrayLayers = 1,
    });

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static void read_buffer_map_cb(WGPUBufferMapAsyncStatus status, void* user_data)
{
  UNUSED_VAR(user_data);

  if (status == WGPUBufferMapAsyncStatus_Success) {
    int32_t w = offscreen_rendering.pixel_data.buffer_dimensions.width;
    int32_t h = offscreen_rendering.pixel_data.buffer_dimensions.height;
    int32_t channels_num = 4;

    size_t pixels_size     = w * h * channels_num;
    uint8_t* pixels        = (uint8_t*)malloc(pixels_size);
    uint8_t const* mapping = (uint8_t*)wgpuBufferGetConstMappedRange(
      offscreen_rendering.pixel_data.buffer.buffer, 0,
      sizeof(offscreen_rendering.pixel_data.buffer.size));
    ASSERT(mapping);
    memcpy(pixels, mapping, pixels_size);
    stbi_write_png(screenshot_filename, w, h, channels_num, pixels,
                   w * sizeof(int));
    wgpuBufferUnmap(offscreen_rendering.pixel_data.buffer.buffer);
    free(pixels);

    offscreen_rendering.pixel_data.buffer_mapped = false;
    screenshot_requested                         = false;
    screenshot_saved                             = true;
  }
}

static int example_draw(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  // Prepare frame
  prepare_frame(context);

  // Set target frame buffer
  scene_rendering.render_pass.color_attachment[0].view
    = wgpu_context->swap_chain.frame_buffer;
  offscreen_rendering.render_pass.color_attachment[0].view
    = offscreen_rendering.color.texture_view;

  // Command buffer to be submitted to the queue
  bool buffer_mapped   = offscreen_rendering.pixel_data.buffer_mapped;
  bool save_screenshot = !buffer_mapped && screenshot_requested;
  wgpu_context->submit_info.command_buffer_count = save_screenshot ? 3 : 1;
  wgpu_context->submit_info.command_buffers[0]   = build_command_buffer(
    wgpu_context, &scene_rendering.render_pass.render_pass_descriptor,
    scene_rendering.pipeline, true);
  if (save_screenshot) {
    wgpu_context->submit_info.command_buffers[1] = build_command_buffer(
      wgpu_context, &offscreen_rendering.render_pass.render_pass_descriptor,
      offscreen_rendering.pipeline, false);
    wgpu_context->submit_info.command_buffers[2]
      = build_copy_texture_to_buffer_command_buffer(wgpu_context);
  }

  // Submit command buffers to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  // Read query results for displaying in next frame
  if (save_screenshot) {
    offscreen_rendering.pixel_data.buffer_mapped = true;
    wgpuBufferMapAsync(
      offscreen_rendering.pixel_data.buffer.buffer, WGPUMapMode_Read, 0,
      offscreen_rendering.pixel_data.buffer.size, read_buffer_map_cb, NULL);
  }

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context);
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  // Update the uniform buffer when the view is changed by user input
  update_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  wgpu_gltf_model_destroy(dragon);

  WGPU_RELEASE_RESOURCE(Texture, offscreen_rendering.color.texture)
  WGPU_RELEASE_RESOURCE(Texture, offscreen_rendering.depth_stencil.texture)

  WGPU_RELEASE_RESOURCE(TextureView, offscreen_rendering.color.texture_view)
  WGPU_RELEASE_RESOURCE(TextureView,
                        offscreen_rendering.depth_stencil.texture_view)

  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, offscreen_rendering.pixel_data.buffer.buffer)

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)

  WGPU_RELEASE_RESOURCE(RenderPipeline, scene_rendering.pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, offscreen_rendering.pipeline)
}

void example_screenshot(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
