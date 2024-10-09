#include "example_base.h"
#include "meshes.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Points
 *
 * This example shows how to render points of various sizes using a quad and
 * instancing.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/points
 * https://webgpufundamentals.org/webgpu/lessons/webgpu-points.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* distance_sized_points_vert_wgsl;
static const char* fixed_size_points_vert_wgsl;
static const char* orange_frag_wgsl;
static const char* textured_frag_wgsl;

/* -------------------------------------------------------------------------- *
 * Points example
 * -------------------------------------------------------------------------- */

#define MAX_VERTICES_COUNT 1000u
#define MAX_VERTICES_LEN (MAX_VERTICES_COUNT * 3)
#define TEXTURE_SIZE 64u
#define DEPTH_FORMAT WGPUTextureFormat_Depth24Plus

const uint32_t num_samples = 1000;

// WGPU buffer
static wgpu_buffer_t vertex_buffer     = {0};
static wgpu_buffer_t uniform_buffer_vs = {0};

// textures
static texture_t texture = {0};
static struct {
  texture_t texture;
  uint32_t width;
  uint32_t height;
} depth_texture = {0};

/* Bind group layout and bind group */
static WGPUBindGroup uniform_bind_group;
static WGPUBindGroupLayout bind_group_layout;

// Pipelines
//  - [0][0] : distance sized - orange
//  - [0][1] : distance sized - textured
//  - [1][0] : fixed sized - orange
//  - [1][1] : fixed sized - textured
static WGPURenderPipeline render_pipelines[2][2] = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

static struct {
  mat4 matrix;
  vec2 resolution;
  float size;
} ubo_vs = {0};

static struct {
  float fov;
  vec3 position;
  vec3 target;
  vec3 up_vector;
  mat4 view_matrix;
  mat4 projection_matrix;
  mat4 view_projection_matrix;
} view_info = {
  .fov                    = (90.0f * PI) / 180.0f,
  .position               = {0.0f, 0.0f, 1.5f},
  .target                 = {0.0f, 0.0f, 0.0f},
  .up_vector              = {0.0f, 1.0f, 0.0f},
  .view_matrix            = GLM_MAT4_ZERO_INIT,
  .projection_matrix      = GLM_MAT4_ZERO_INIT,
  .view_projection_matrix = GLM_MAT4_ZERO_INIT,
};

/* GUI parameters */
static struct {
  bool fixed_size;
  bool textured;
  float size;
} settings = {
  .fixed_size = false,
  .textured   = false,
  .size       = 10.0,
};

// Other variables
static const char* example_title = "Points";
static bool prepared             = false;

/* See: https://www.google.com/search?q=fibonacci+sphere */
static void
create_fibonacci_sphere_vertices(uint32_t num_samples, float radius,
                                 float (*vertices)[MAX_VERTICES_LEN])
{
  num_samples              = MIN(num_samples, MAX_VERTICES_COUNT);
  const uint32_t increment = PI * (3.0f - sqrt(5.0f));
  float offset = 0.0f, y = 0.0f, r = 0.0f, phi = 0.0f, x = 0.0f, z = 0.0f;
  for (uint32_t i = 0; i < num_samples; ++i) {
    offset = 2.0f / num_samples;
    y      = i * offset - 1.0f + offset / 2.0f;
    r      = sqrt(1.0f - pow(y, 2.0f));
    phi    = (i % num_samples) * increment;
    x      = cos(phi) * r;
    z      = sin(phi) * r;
    // Ste vertex
    (*vertices)[i * 3]     = x * radius;
    (*vertices)[i * 3 + 1] = y * radius;
    (*vertices)[i * 3 + 2] = z * radius;
  }
}

static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  float vertex_data[MAX_VERTICES_LEN] = {0};
  create_fibonacci_sphere_vertices(num_samples, 1.0f, &vertex_data);

  vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex buffer - Vertices",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = num_samples * 3 * sizeof(float),
                    .initial.data = vertex_data,
                    .count        = num_samples,
                  });
}

static void update_uniform_buffer_data(wgpu_example_context_t* context)
{
  /* Convert to seconds */
  float time = context->frame.timestamp_millis * 0.001f;

  /* Set the size in the uniform values */
  ubo_vs.size = settings.size;

  /* Set the matrix value */
  const float aspect_ratio = (float)context->wgpu_context->surface.width
                             / (float)context->wgpu_context->surface.height;
  glm_perspective(view_info.fov, aspect_ratio, 0.1f, 50.0f,
                  view_info.projection_matrix);
  glm_lookat(view_info.position,   /* eye vector    */
             view_info.target,     /* center vector */
             view_info.up_vector,  /* up vector     */
             view_info.view_matrix /* result matrix */
  );
  glm_mat4_mul(view_info.projection_matrix, view_info.view_matrix,
               view_info.view_projection_matrix);
  glm_rotate_y(view_info.view_projection_matrix, time, ubo_vs.matrix);
  glm_rotate_x(ubo_vs.matrix, time * 0.1f, ubo_vs.matrix);

  /* Set the resolution in the uniform values */
  ubo_vs.resolution[0] = (float)context->wgpu_context->surface.width;
  ubo_vs.resolution[1] = (float)context->wgpu_context->surface.height;
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Update the uniform buffer data
  update_uniform_buffer_data(context);

  // Copy the uniform values to the GPU
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &ubo_vs, uniform_buffer_vs.size);
}

static void prepare_uniform_buffer(wgpu_example_context_t* context)
{
  // Create vertex shader uniform buffer block
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Vertex shader - Uniform buffer block",
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(ubo_vs),
    });

  // Initialize GPU buffer
  update_uniform_buffer_data(context);
}

static void prepare_texture(wgpu_context_t* wgpu_context)
{
  /* Texture */
  WGPUTextureDescriptor texture_desc = {
    .label         = "Texture",
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding
                     | WGPUTextureUsage_RenderAttachment,
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .dimension     = WGPUTextureDimension_2D,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .size          = (WGPUExtent3D) {
      .width               = TEXTURE_SIZE,
      .height              = TEXTURE_SIZE,
      .depthOrArrayLayers  = 1,
    },
  };
  texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(texture.texture != NULL);

  /* Set texture data */
  uint8_t data[TEXTURE_SIZE * TEXTURE_SIZE * 4] = {0};
  const uint32_t data_size                      = sizeof(data);
  const uint8_t color_white[4]                  = {255, 255, 255, 255};
  const uint8_t color_black[4]                  = {0, 0, 0, 255};
  uint32_t index                                = 0;
  for (uint32_t y = 0; y < TEXTURE_SIZE; ++y) {
    for (uint32_t x = 0; x < TEXTURE_SIZE; ++x) {
      index = (y * TEXTURE_SIZE + x) * 4;
      for (uint8_t c = 0; c < 4; ++c) {
        data[index + c] = (x + y) % 2 ? color_white[c] : color_black[c];
      }
    }
  }
  wgpuQueueWriteTexture(wgpu_context->queue,
    &(WGPUImageCopyTexture) {
      .texture  = texture.texture,
      .mipLevel = 0,
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
      .bytesPerRow  = TEXTURE_SIZE * 4,
      .rowsPerImage = TEXTURE_SIZE,
    },
    &(WGPUExtent3D){
      .width              = TEXTURE_SIZE,
      .height             = TEXTURE_SIZE,
      .depthOrArrayLayers = 1,
    });

  /* Texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "Texture view",
    .format          = texture_desc.format,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = texture_desc.mipLevelCount,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  texture.view = wgpuTextureCreateView(texture.texture, &texture_view_dec);
  ASSERT(texture.view != NULL);

  /* Create a sampler with linear filtering for smooth interpolation */
  texture.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Sampler with linear filtering",
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .maxAnisotropy = 16,
                          });
  ASSERT(texture.sampler != NULL);
}

static void prepare_depth_texture(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_texture(&depth_texture.texture);

  /* Set texture size */
  depth_texture.width  = wgpu_context->surface.width;
  depth_texture.height = wgpu_context->surface.height;

  /* Create the depth texture */
  WGPUTextureDescriptor texture_desc = {
    .label         = "Depth texture",
    .size          = (WGPUExtent3D) {
       .width              = depth_texture.width,
       .height             = depth_texture.height,
       .depthOrArrayLayers = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = DEPTH_FORMAT,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  depth_texture.texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(depth_texture.texture.texture != NULL);

  /* Create the depth texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  depth_texture.texture.view
    = wgpuTextureCreateView(depth_texture.texture.texture, &texture_view_dec);
  ASSERT(depth_texture.texture.view != NULL);
}

// Create a bind group layout so we can share the bind groups with multiple
// pipelines.
static void setup_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(ubo_vs),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
        /* Texture view */
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
      },
      .storageTexture = {0},
    }
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Cube - Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Render - Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .sampler = texture.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = texture.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Uniform buffer - Bind group",
    .layout     = bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  uniform_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(uniform_bind_group != NULL);
}

static void setup_render_pass(void)
{
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
    .view       = NULL, /* Assigned later */
    .depthSlice = ~0,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor) {
      .r = 0.3f,
      .g = 0.3f,
      .b = 0.3f,
      .a = 1.0f,
    },
  };

  /* Render pass descriptor */
  render_pass.depth_stencil_attachment = (WGPURenderPassDepthStencilAttachment){
    .view            = NULL, /* To be filled out when we render */
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  };
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Our basic canvas render pass",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &render_pass.depth_stencil_attachment,
    .occlusionQuerySet      = NULL,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  const char* vertex_shaders[2] = {
    distance_sized_points_vert_wgsl, /* Distance sized fragment shader */
    fixed_size_points_vert_wgsl,     /* Fixed size vertex shader       */
  };

  const char* fragment_shaders[2] = {
    orange_frag_wgsl,   /* Orange fragment shader   */
    textured_frag_wgsl, /* Textured fragment shader */
  };

  /* Make pipelines for each combination */
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(vertex_shaders); ++i) {
    for (uint32_t j = 0; j < (uint32_t)ARRAY_SIZE(fragment_shaders); ++j) {
      // Primitive state
      WGPUPrimitiveState primitive_state = {
        .topology  = WGPUPrimitiveTopology_PointList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      };

      // Color target state
      WGPUBlendState blend_state = (WGPUBlendState){
        .color.operation = WGPUBlendOperation_Add,
        .color.srcFactor = WGPUBlendFactor_One,
        .color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
        .alpha.operation = WGPUBlendOperation_Add,
        .alpha.srcFactor = WGPUBlendFactor_One,
        .alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
      };
      WGPUColorTargetState color_target_state = (WGPUColorTargetState){
        .format    = wgpu_context->swap_chain.format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      };

      // Depth stencil state
      WGPUDepthStencilState depth_stencil_state
        = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
          .format              = DEPTH_FORMAT,
          .depth_write_enabled = true,
        });
      depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

      // Vertex buffer layout
      WGPU_VERTEX_BUFFER_LAYOUT(
        instanced_point, 3 * 4 /* 3 floats, 4 bytes each */,
        // Attribute location 0: Position
        WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0))
      instanced_point_vertex_buffer_layout.stepMode
        = WGPUVertexStepMode_Instance;

      // Vertex state
      WGPUVertexState vertex_state = wgpu_create_vertex_state(
        wgpu_context, &(wgpu_vertex_state_t){
                        .shader_desc = (wgpu_shader_desc_t){
                          /* Vertex shader WGSL */
                          .label            = "Vertex shader WGSL",
                          .wgsl_code.source = vertex_shaders[i],
                          .entry            = "vs",
                        },
                        .buffer_count = 1,
                        .buffers      = &instanced_point_vertex_buffer_layout,
                      });

      // Fragment state
      WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        wgpu_context, &(wgpu_fragment_state_t){
                        .shader_desc = (wgpu_shader_desc_t){
                          /* Fragment shader WGSL */
                          .label            = "Fragment shader WGSL",
                          .wgsl_code.source = fragment_shaders[j],
                          .entry            = "fs",
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

      // Create rendering pipeline using the specified states
      render_pipelines[i][j] = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                .label        = "Points - Render pipeline",
                                .layout       = pipeline_layout,
                                .primitive    = primitive_state,
                                .vertex       = vertex_state,
                                .fragment     = &fragment_state,
                                .depthStencil = &depth_stencil_state,
                                .multisample  = multisample_state,
                              });
      ASSERT(render_pipelines[i][j] != NULL);

      // Partial cleanup
      WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
      WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
    }
  }
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertex_buffer(context->wgpu_context);
    prepare_uniform_buffer(context);
    prepare_texture(context->wgpu_context);
    prepare_depth_texture(context->wgpu_context);
    setup_bind_group_layout(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass();
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    imgui_overlay_checkBox(context->imgui_overlay, "Fixed Size",
                           &settings.fixed_size);
    imgui_overlay_checkBox(context->imgui_overlay, "Textured",
                           &settings.textured);
    if (imgui_overlay_slider_float(context->imgui_overlay, "Size",
                                   &settings.size, 0.0f, 80.0f, "%.1f")) {
      update_uniform_buffers(context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* If we don't have a depth texture OR if its size is different from the
   * canvasTexture when make a new depth texture */
  if (!depth_texture.texture.texture
      || depth_texture.width != wgpu_context->surface.width
      || depth_texture.height != wgpu_context->surface.height) {
    prepare_depth_texture(wgpu_context);
  }
  render_pass.depth_stencil_attachment.view = depth_texture.texture.view;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Bind the rendering pipeline */
  const WGPURenderPipeline pipeline
    = render_pipelines[settings.fixed_size ? 1 : 0][settings.textured ? 1 : 0];
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  // Bind vertex buffer (contains positions)
  wgpuRenderPassEncoderSetVertexBuffer(
    wgpu_context->rpass_enc, 0, vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    uniform_bind_group, 0, 0);

  /* Draw */
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, vertex_buffer.count, 0,
                            0);

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

static int example_draw(wgpu_example_context_t* context)
{
  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  /* Submit to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  wgpu_destroy_buffer(&vertex_buffer);
  wgpu_destroy_buffer(&uniform_buffer_vs);
  wgpu_destroy_texture(&texture);
  wgpu_destroy_texture(&depth_texture.texture);
  WGPU_RELEASE_RESOURCE(BindGroup, uniform_bind_group)
  for (uint32_t i = 0; i < 2; ++i) {
    for (uint32_t j = 0; j < 2; ++j) {
      WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines[i][j])
    }
  }
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
}

void example_points(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* distance_sized_points_vert_wgsl = CODE(
  struct Vertex {
    @location(0) position: vec4f,
  };

  struct Uniforms {
    matrix: mat4x4f,
    resolution: vec2f,
    size: f32,
  };

  struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
  };

  @group(0) @binding(0) var<uniform> uni: Uniforms;

  @vertex fn vs(
      vert: Vertex,
      @builtin(vertex_index) vNdx: u32,
  ) -> VSOutput {
    let points = array(
      vec2f(-1, -1),
      vec2f( 1, -1),
      vec2f(-1,  1),
      vec2f(-1,  1),
      vec2f( 1, -1),
      vec2f( 1,  1),
    );
    var vsOut: VSOutput;
    let pos = points[vNdx];
    let clipPos = uni.matrix * vert.position;
    let pointPos = vec4f(pos * uni.size / uni.resolution, 0, 0);
    vsOut.position = clipPos + pointPos;
    vsOut.texcoord = pos * 0.5 + 0.5;
    return vsOut;
  }
);

static const char* fixed_size_points_vert_wgsl = CODE(
  struct Vertex {
    @location(0) position: vec4f,
  };

  struct Uniforms {
    matrix: mat4x4f,
    resolution: vec2f,
    size: f32,
  };

  struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
  };

  @group(0) @binding(0) var<uniform> uni: Uniforms;

  @vertex fn vs(
      vert: Vertex,
      @builtin(vertex_index) vNdx: u32,
  ) -> VSOutput {
    let points = array(
      vec2f(-1, -1),
      vec2f( 1, -1),
      vec2f(-1,  1),
      vec2f(-1,  1),
      vec2f( 1, -1),
      vec2f( 1,  1),
    );
    var vsOut: VSOutput;
    let pos = points[vNdx];
    let clipPos = uni.matrix * vert.position;
    let pointPos = vec4f(pos * uni.size / uni.resolution * clipPos.w, 0, 0);
    vsOut.position = clipPos + pointPos;
    vsOut.texcoord = pos * 0.5 + 0.5;
    return vsOut;
  }
);

static const char* orange_frag_wgsl = CODE(
  @fragment fn fs() -> @location(0) vec4f {
    return vec4f(1, 0.5, 0.2, 1);
  }
);

static const char* textured_frag_wgsl = CODE(
  struct VSOutput {
    @location(0) texcoord: vec2f,
  };

  @group(0) @binding(1) var s: sampler;
  @group(0) @binding(2) var t: texture_2d<f32>;

  @fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
    let color = textureSample(t, s, vsOut.texcoord);
    if (color.a < 0.1) {
      discard;
    }
    return color;
  }
);
// clang-format on
