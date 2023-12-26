#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Sampler Parameters
 *
 * Visualizes what all the sampler parameters do. Shows a textured plane at
 * various scales (rotated, head-on, in perspective, and in vanishing
 * perspective). The bottom-right view shows the raw contents of the 4 mipmap
 * levels of the test texture (16x16, 8x8, 4x4, and 2x2).
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/samplerParameters
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* show_texture_wgsl;
static const char* textured_square_wgsl;

/* -------------------------------------------------------------------------- *
 * Sampler Parameters example
 * -------------------------------------------------------------------------- */

#define CANVAS_SIZE 600u

static struct {
  float flange_log_size;
  bool highlight_flange;
  float animation;
} config = {
  .flange_log_size  = 1.0f,
  .highlight_flange = false,
  .animation        = 0.1f,
};

// Uniform buffer
static wgpu_buffer_t buf_config = {0};

// Storage buffer
static wgpu_buffer_t buf_matrices = {0};

// Checkerboard texture
static texture_t checkerboard = {0};

// Render pipelines
static WGPURenderPipeline show_texture_render_pipeline    = NULL;
static WGPURenderPipeline textured_square_render_pipeline = NULL;

// Bind groups
static WGPUBindGroup show_texture_bind_group    = NULL;
static WGPUBindGroup textured_square_bind_group = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Viewport properties
static uint32_t texture_base_size    = 0;
static uint32_t viewport_size        = 0;
static uint32_t viewport_grid_size   = 0;
static uint32_t viewport_grid_stride = 0;

// Other variables
static const char* example_title = "Sampler Parameters";
static bool prepared             = false;

//
// GUI controls
//

static WGPUSamplerDescriptor sampler_descriptor = {
  .label         = "Checkerboard texture sampler",
  .addressModeU  = WGPUAddressMode_ClampToEdge,
  .addressModeV  = WGPUAddressMode_ClampToEdge,
  .addressModeW  = WGPUAddressMode_ClampToEdge,
  .magFilter     = WGPUFilterMode_Linear,
  .minFilter     = WGPUFilterMode_Linear,
  .mipmapFilter  = WGPUMipmapFilterMode_Linear,
  .lodMinClamp   = 0.0f,
  .lodMaxClamp   = 4.0f,
  .maxAnisotropy = 1,
};

typedef enum preset_enum_t {
  Preset_Initial      = 0,
  Preset_Checkerboard = 1,
  Preset_Smooth       = 2,
  Preset_Crunchy      = 3,
  Preset_Count        = 4,
} preset_enum_t;

static WGPUSamplerDescriptor presets[4] = {
  [Preset_Checkerboard] = {
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
  },
};

static void prepare_canvas(void)
{
  const uint32_t canvas_size = CANVAS_SIZE;
  viewport_grid_size         = 4;
  viewport_grid_stride       = floor(canvas_size / (float)viewport_grid_size);
  viewport_size              = viewport_grid_stride - 2;
}

static void prepare_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Create uniform buffer */
  buf_config = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniform bufer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = 128,
                  });
  ASSERT(buf_config.buffer != NULL);

  /* View-projection matrix set up so it doesn't transform anything at z=0. */
  const uint32_t camera_dist = 3;
  mat4 view_proj             = GLM_MAT4_IDENTITY_INIT;
  glm_perspective(2.0f * atan(1.0f / camera_dist), 1, 0.1f, 100.0f, view_proj);
  glm_translate(view_proj, (vec3){0.0f, 0.0f, -camera_dist});

  /* Update uniform buffer data */
  wgpu_queue_write_buffer(wgpu_context, buf_config.buffer, 0, view_proj,
                          sizeof(mat4));
}

static void update_config_buffer(wgpu_example_context_t* context)
{
  const float t       = (context->frame.timestamp_millis / 1000.0f) * 0.5f;
  const float data[4] = {
    cos(t) * config.animation,                   //
    sin(t) * config.animation,                   //
    (pow(2, config.flange_log_size) - 1) / 2.0f, //
    config.highlight_flange,                     //
  };

  wgpu_queue_write_buffer(context->wgpu_context, buf_config.buffer, 64,
                          &data[0], sizeof(data));
}

static void prepare_storage_buffer(wgpu_context_t* wgpu_context)
{
  mat4 matrices[15] = {0};

  /* Row 1: Scale by 2 */
  {
    /* Column 1 */
    glm_mat4_identity(matrices[0]);
    glm_rotate_z(matrices[0], PI / 16, matrices[0]);
    glm_scale(matrices[0], (vec3){2, 2, 1});
    /* Column 2 */
    glm_mat4_identity(matrices[1]);
    glm_scale(matrices[1], (vec3){2, 2, 1});
    /* Column 3 */
    glm_mat4_identity(matrices[2]);
    glm_rotate_x(matrices[2], -PI * 0.3f, matrices[2]);
    glm_scale(matrices[2], (vec3){2, 2, 1});
    /* Column 4 */
    glm_mat4_identity(matrices[3]);
    glm_rotate_x(matrices[3], -PI * 0.42f, matrices[3]);
    glm_scale(matrices[3], (vec3){2, 2, 1});
  }

  /* Row 2: Scale by 1 */
  {
    /* Column 1 */
    glm_mat4_identity(matrices[4]);
    glm_rotate_z(matrices[4], PI / 16, matrices[4]);
    /* Column 2 */
    glm_mat4_identity(matrices[5]);
    /* Column 3 */
    glm_mat4_identity(matrices[6]);
    glm_rotate_x(matrices[6], -PI * 0.3f, matrices[6]);
    /* Column 4 */
    glm_mat4_identity(matrices[7]);
    glm_rotate_x(matrices[7], -PI * 0.42f, matrices[7]);
  }

  /* Row 3: Scale by 0.9 */
  {
    /* Column 1 */
    glm_mat4_identity(matrices[8]);
    glm_rotate_z(matrices[8], PI / 16, matrices[8]);
    glm_scale(matrices[8], (vec3){0.9f, 0.9f, 1.0f});
    /* Column 2 */
    glm_mat4_identity(matrices[9]);
    glm_scale(matrices[9], (vec3){0.9f, 0.9f, 1});
    /* Column 3 */
    glm_mat4_identity(matrices[10]);
    glm_rotate_x(matrices[10], -PI * 0.3f, matrices[10]);
    glm_scale(matrices[10], (vec3){0.9f, 0.9f, 1.0f});
    /* Column 4 */
    glm_mat4_identity(matrices[11]);
    glm_rotate_x(matrices[11], -PI * 0.42f, matrices[11]);
    glm_scale(matrices[11], (vec3){0.9f, 0.9f, 1.0f});
  }

  /* Row 4: Scale by 0.3 */
  {
    /* Column 1 */
    glm_mat4_identity(matrices[12]);
    glm_rotate_z(matrices[12], PI / 16, matrices[12]);
    glm_scale(matrices[12], (vec3){0.3f, 0.3f, 1.0f});
    /* Column 2 */
    glm_mat4_identity(matrices[13]);
    glm_scale(matrices[13], (vec3){0.3f, 0.3f, 1.0f});
    /* Column 3 */
    glm_mat4_identity(matrices[14]);
    glm_rotate_x(matrices[14], -PI * 0.3f, matrices[14]);
    glm_scale(matrices[14], (vec3){0.3f, 0.3f, 1.0f});
  }

  buf_matrices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Matrices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
                    .size  = sizeof(matrices),
                    .initial.data = matrices,
                  });
}

//
// Initialize test texture
//
// Set up a texture with 4 mip levels, each containing a differently-colored
// checkerboard with 1x1 pixels (so when rendered the checkerboards are
// different sizes). This is different from a normal mipmap where each level
// would look like a lower-resolution version of the previous one.
// Level 0 is 16x16 white/black
// Level 1 is 8x8 blue/black
// Level 2 is 4x4 yellow/black
// Level 3 is 2x2 pink/black
static void initialize_test_texture(wgpu_context_t* wgpu_context)
{
  const uint32_t texture_mip_levels = 4;
  texture_base_size                 = 16;

  /* Checkerboard texture */
  WGPUTextureDescriptor texture_desc = {
    .label         = "Checkerboard texture",
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .dimension     = WGPUTextureDimension_2D,
    .mipLevelCount = texture_mip_levels,
    .sampleCount   = 1,
      .size          = (WGPUExtent3D) {
      .width               = texture_base_size,
      .height              = texture_base_size,
      .depthOrArrayLayers  = 1,
    },
  };
  checkerboard.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(checkerboard.texture != NULL);

  /* Checkerboard texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "Checkerboard texture view",
    .format          = texture_desc.format,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = texture_desc.mipLevelCount,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  checkerboard.view
    = wgpuTextureCreateView(checkerboard.texture, &texture_view_dec);
  ASSERT(checkerboard.view != NULL);

  /* Checkerboard texture data */
  const uint8_t color_for_level[4][4] = {
    {255, 255, 255, 255}, /*        */
    {30, 136, 229, 255},  /* blue   */
    {255, 193, 7, 255},   /* yellow */
    {216, 27, 96, 255},   /* pink   */
  };
  const uint8_t color_black[4] = {0, 0, 0, 255};
  uint32_t index               = 0;
  for (uint32_t mip_level = 0; mip_level < texture_mip_levels; ++mip_level) {
    /* Sizes: 16, 8, 4, 2 */
    const uint32_t size      = pow(2, texture_mip_levels - mip_level);
    const uint32_t data_size = size * size * 4 * sizeof(uint8_t);
    uint8_t* data            = (uint8_t*)malloc(data_size);
    memset(data, 0, sizeof(data_size));
    for (uint32_t y = 0; y < size; ++y) {
      for (uint32_t x = 0; x < size; ++x) {
        index = (y * size + x) * 4;
        for (uint8_t c = 0; c < 4; ++c) {
          data[index + c]
            = (x + y) % 2 ? color_for_level[mip_level][c] : color_black[c];
        }
      }
    }
    wgpuQueueWriteTexture(wgpu_context->queue,
      &(WGPUImageCopyTexture) {
        .texture = checkerboard.texture,
        .mipLevel = mip_level,
        .origin = (WGPUOrigin3D) {
          .x = 0,
          .y = 0,
          .z = 0,
      },
      .aspect = WGPUTextureAspect_All,
      },
      data, data_size,
      &(WGPUTextureDataLayout){
        .offset       = 0,
        .bytesPerRow  = size * 4,
        .rowsPerImage = size,
      },
      &(WGPUExtent3D){
        .width              = size,
        .height             = size,
        .depthOrArrayLayers = 1,
      });
    free(data);
  }
}

static void update_textured_square_sampler(wgpu_context_t* wgpu_context)
{
  /* Destroy current sampler */
  WGPU_RELEASE_RESOURCE(Sampler, checkerboard.sampler)

  /* Create sampler */
  sampler_descriptor.maxAnisotropy
    = (sampler_descriptor.minFilter == WGPUFilterMode_Linear
       && sampler_descriptor.magFilter == WGPUFilterMode_Linear
       && sampler_descriptor.minFilter == WGPUFilterMode_Linear) ?
        sampler_descriptor.maxAnisotropy :
        1;
  checkerboard.sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_descriptor);
  ASSERT(checkerboard.sampler != NULL);
}

static void update_textured_square_bind_group(wgpu_context_t* wgpu_context)
{
  /* Destroy current bind group */
  WGPU_RELEASE_RESOURCE(BindGroup, textured_square_bind_group)

  /* Create bind group */
  WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = buf_config.buffer,
        .size    = buf_config.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = buf_matrices.buffer,
        .size    = buf_matrices.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding  = 2,
        .sampler  = checkerboard.sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding     = 3,
        .textureView = checkerboard.view
      },
    };
  textured_square_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label  = "Textured square bind group",
                            .layout = wgpuRenderPipelineGetBindGroupLayout(
                              textured_square_render_pipeline, 0),
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(textured_square_bind_group != NULL);
}

static void setup_render_pass(void)
{
  /* Color attachment */
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
    .view       = NULL, /* Assigned later */
    .depthSlice = ~0,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor) {
      .r = 0.2f,
      .g = 0.2f,
      .b = 0.2f,
      .a = 1.0f,
    },
  };

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = NULL,
  };
}

//
// "Debug" view of the actual texture contents
//
static void prepare_debug_view_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      // Vertex shader WGSL
      .label            = "Debug view vertex shader WGSL",
      .wgsl_code.source = show_texture_wgsl,
      .entry            = "vmain"
    },
  });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      // Fragment shader WGSL
      .label            = "Debug view fragment shader WGSL",
      .wgsl_code.source = show_texture_wgsl,
      .entry            = "fmain"
    },
    .target_count = 1,
    .targets = &color_target_state,
  });

  // Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Create rendering pipeline using the specified states */
  show_texture_render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Show texture render pipeline",
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(show_texture_render_pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void prepare_show_texture_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entry = (WGPUBindGroupEntry){
    .binding     = 0,
    .textureView = checkerboard.view,
  };
  show_texture_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label  = "Show texture bind group",
                            .layout = wgpuRenderPipelineGetBindGroupLayout(
                              show_texture_render_pipeline, 0),
                            .entryCount = 1,
                            .entries    = &bg_entry,
                          });
  ASSERT(show_texture_bind_group != NULL);
}

//
// Pipeline for drawing the test squares
//
static void
prepare_textured_square_render_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Constants
  WGPUConstantEntry constant_entries[2] = {
    [0] = (WGPUConstantEntry){
      .key   = "kTextureBaseSize",
      .value = (double)texture_base_size,
    },
    [1] = (WGPUConstantEntry){
      .key   = "kViewportSize",
      .value =(double)viewport_size,
    },
  };

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      // Vertex shader WGSL
      .label            = "Textured square vertex shader WGSL",
      .wgsl_code.source = textured_square_wgsl,
      .entry            = "vmain"
    },
    .constant_count = (uint32_t)ARRAY_SIZE(constant_entries),
    .constants      = constant_entries,
  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      // Fragment shader WGSL
      .label            = "Textured square fragment shader WGSL",
      .wgsl_code.source = textured_square_wgsl,
      .entry            = "fmain"
    },
    .target_count   = 1,
    .targets        = &color_target_state,
  });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  textured_square_render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Textured square render pipeline",
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(textured_square_render_pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Draw test squares */
  {
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     textured_square_render_pipeline);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      textured_square_bind_group, 0, 0);
    for (uint32_t i = 0; i < pow(viewport_grid_size, 2) - 1; ++i) {
      const uint32_t vp_x = viewport_grid_stride * (i % viewport_grid_size) + 1;
      const uint32_t vp_y
        = viewport_grid_stride * floor(i / (float)viewport_grid_size) + 1;
      wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, vp_x, vp_y,
                                       viewport_size, viewport_size, 0.0f,
                                       1.0f);
      wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, 1, 0, i);
    }
  }

  /* Show texture contents */
  {
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     show_texture_render_pipeline);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      show_texture_bind_group, 0, 0);
    const uint32_t last_viewport
      = (viewport_grid_size - 1) * viewport_grid_stride + 1;
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, last_viewport,
                                     last_viewport, 32, 32, 0.0f, 1.0f);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, 1, 0, 0);
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc,
                                     last_viewport + 32, last_viewport, 16, 16,
                                     0.0f, 1.0f);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, 1, 0, 1);
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc,
                                     last_viewport + 32, last_viewport + 16, 8,
                                     8, 0.0f, 1.0f);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, 1, 0, 2);
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc,
                                     last_viewport + 32, last_viewport + 24, 4,
                                     4, 0.0f, 1.0f);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, 1, 0, 3);
  }

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_canvas();
    initialize_test_texture(context->wgpu_context);
    prepare_debug_view_render_pipeline(context->wgpu_context);
    prepare_show_texture_bind_group(context->wgpu_context);
    prepare_textured_square_render_pipeline(context->wgpu_context);
    prepare_uniform_buffer(context->wgpu_context);
    prepare_storage_buffer(context->wgpu_context);
    update_config_buffer(context);
    update_textured_square_sampler(context->wgpu_context);
    update_textured_square_bind_group(context->wgpu_context);
    setup_render_pass();
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int example_draw(wgpu_context_t* wgpu_context)
{
  // Get next image in the swap chain (back/front buffer)
  wgpu_swap_chain_get_current_image(wgpu_context);

  // Create command buffer
  WGPUCommandBuffer command_buffer = build_command_buffer(wgpu_context);
  ASSERT(command_buffer != NULL);

  // Submit command buffer to the queue
  wgpu_flush_command_buffers(wgpu_context, &command_buffer, 1);

  // Present the current buffer to the swap chain
  wgpu_swap_chain_present(wgpu_context);

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  if (!context->paused) {
    update_config_buffer(context);
  }
  return example_draw(context->wgpu_context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  wgpu_destroy_texture(&checkerboard);
  WGPU_RELEASE_RESOURCE(Buffer, buf_config.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, buf_matrices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, show_texture_render_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, textured_square_render_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, show_texture_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, textured_square_bind_group)
}

void example_sampler_parameters(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
      .example_settings = (wgpu_example_settings_t){
        .title   = example_title,
        .overlay = true,
    },
    .example_window_config = (window_config_t){
      .width  = CANVAS_SIZE,
      .height = CANVAS_SIZE,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* show_texture_wgsl = CODE(
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

static const char* textured_square_wgsl = CODE(
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
// clang-format on
