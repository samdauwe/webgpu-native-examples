#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Volume Rendering - Texture 3D
 *
 * This example shows how to render volumes with WebGPU using a 3D texture. It
 * demonstrates simple direct volume rendering for photometric content through
 * ray marching in a fragment shader, where a full-screen triangle determines
 * the color from ray start and step size values as set in the vertex shader.
 * This implementation employs data from the BrainWeb Simulated Brain Database,
 * with decompression streams, to save disk space and network traffic.
 *
 * The original raw data is generated using the BrainWeb Simulated Brain
 * Database:
 * https://brainweb.bic.mni.mcgill.ca/brainweb/
 * before processingin a custom Python script:
 * https://github.com/webgpu/webgpu-samples/tree/main/public/assets/img/volume/t1_icbm_normal_1mm_pn0_rf0.py).
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/volumeRenderingTexture3D
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* volume_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Volume Rendering - Texture 3D example
 * -------------------------------------------------------------------------- */

#define VOLUME_TEXTURE_WIDTH 180
#define VOLUME_TEXTURE_HEIGHT 216
#define VOLUME_TEXTURE_DEPTH 180
#define VOLUME_TEXTURE_SIZE                                                    \
  (VOLUME_TEXTURE_WIDTH * VOLUME_TEXTURE_HEIGHT * VOLUME_TEXTURE_DEPTH)

/* GUI parameters */
static struct {
  bool rotate_camera;
  float near;
  float far;
} params = {
  .rotate_camera = true,
  .near          = 2.0f,
  .far           = 7.0,
};

static struct {
  mat4 inverse_model_view_projection_matrix;
} ubo_vs = {
  .inverse_model_view_projection_matrix = GLM_MAT4_ZERO_INIT,
};

// Uniform buffer block object
static wgpu_buffer_t uniform_buffer_vs = {0};

static struct {
  mat4 projection;
  mat4 view;
} view_matrices = {0};

static float last_frame_ms         = 0.0f;
static float rotation              = 0.0f;
static const uint32_t sample_count = 4;

// Contains all WebGPU objects that are required to store and use the 3D texture
static struct {
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
    WGPUSampler sampler;
    uint8_t data[VOLUME_TEXTURE_SIZE];
  } volume;
  struct {
    WGPUTexture texture;
    WGPUTextureView framebuffer;
  } multisampled;
} textures = {0};

// Pipeline
static WGPURenderPipeline render_pipeline = NULL;

// Bind groups stores the resources bound to the binding points in a shader
static WGPUBindGroup uniform_bind_group = NULL;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

// Other variables
static const char* example_title = "Volume Rendering - Texture 3D";
static bool prepared             = false;

static void
get_inverse_model_view_projection_matrix(wgpu_context_t* wgpu_context,
                                         float delta_time, mat4* dest)
{
  /* View matrix */
  glm_mat4_identity(view_matrices.view);
  glm_translate(view_matrices.view, (vec3){0.0f, 0.0f, -4.0f});
  if (params.rotate_camera) {
    rotation += delta_time;
  }
  glm_rotate(view_matrices.view, 1.0f,
             (vec3){sin(rotation), cos(rotation), 0.0f});

  /* Projection matrix */
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;
  glm_mat4_identity(view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, params.near, params.far,
                  view_matrices.projection);
  glm_mat4_mul(view_matrices.projection, view_matrices.view, *dest);
  glm_mat4_inv(*dest, *dest);
}

static void
update_inverse_model_view_projection_matrix(wgpu_example_context_t* context)
{
  const float now        = context->frame.timestamp_millis;
  const float delta_time = (now - last_frame_ms) / 1000.0f;
  last_frame_ms          = now;

  get_inverse_model_view_projection_matrix(
    context->wgpu_context, delta_time,
    &ubo_vs.inverse_model_view_projection_matrix);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Update the inverse model-view-projection matrix
  update_inverse_model_view_projection_matrix(context);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &ubo_vs.inverse_model_view_projection_matrix,
                          uniform_buffer_vs.size);
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffer(wgpu_example_context_t* context)
{
  /* Create vertex shader uniform buffer block */
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Vertex shader - Uniform buffer block",
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(ubo_vs),
    });

  /* Set uniform buffer block data */
  last_frame_ms = context->frame.timestamp_millis;
  update_uniform_buffers(context);
}

static void prepare_multisampled_framebuffer(wgpu_context_t* wgpu_context)
{
  /* Create the multi-sampled texture */
  WGPUTextureDescriptor multisampled_frame_desc = {
    .label         = "Multi-sampled texture",
    .size          = (WGPUExtent3D){
       .width               = wgpu_context->surface.width,
       .height              = wgpu_context->surface.height,
       .depthOrArrayLayers  = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = sample_count,
    .dimension     = WGPUTextureDimension_2D,
    .format        = wgpu_context->swap_chain.format,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  textures.multisampled.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &multisampled_frame_desc);
  ASSERT(textures.multisampled.texture != NULL);

  /* Create the multi-sampled texture view */
  textures.multisampled.framebuffer = wgpuTextureCreateView(
    textures.multisampled.texture, &(WGPUTextureViewDescriptor){
                                     .label  = "Multi-sampled texture view",
                                     .format = wgpu_context->swap_chain.format,
                                     .dimension = WGPUTextureViewDimension_2D,
                                     .baseMipLevel    = 0,
                                     .mipLevelCount   = 1,
                                     .baseArrayLayer  = 0,
                                     .arrayLayerCount = 1,
                                   });
  ASSERT(textures.multisampled.framebuffer != NULL);
}

static void prepare_volume_texture(wgpu_context_t* wgpu_context)
{
  const uint32_t width           = VOLUME_TEXTURE_WIDTH;
  const uint32_t height          = VOLUME_TEXTURE_HEIGHT;
  const uint32_t depth           = VOLUME_TEXTURE_DEPTH;
  const uint32_t mip_levels      = 1;
  const WGPUTextureFormat format = WGPUTextureFormat_R8Unorm;
  const uint32_t block_length    = 1;
  const uint32_t bytes_per_block = 1;
  const uint32_t blocks_wide     = ceil(width / (float)block_length);
  const uint32_t blocks_high     = ceil(height / (float)block_length);
  const uint32_t bytes_per_row   = blocks_wide * bytes_per_block;

  /* Read volume data from file */
  {
    file_read_result_t file_read_result = {0};
    const char* data_path
      = "textures/volume/t1_icbm_normal_1mm_pn0_rf0_180x216x180_uint8_1x1.bin";
    read_file(data_path, &file_read_result, false);
    if (file_read_result.size == VOLUME_TEXTURE_SIZE) {
      memcpy(textures.volume.data, file_read_result.data,
             file_read_result.size);
    }
    if (file_read_result.data) {
      free(file_read_result.data);
    }
  }

  /* Create the volume texture */
  WGPUTextureDescriptor texture_desc = {
    .label         = "Volume texture",
    .size          =   (WGPUExtent3D) {
      .width              = width,
      .height             = height,
      .depthOrArrayLayers = depth,
    },
    .mipLevelCount = mip_levels,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_3D,
    .format        = format,
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
  };
  textures.volume.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(textures.volume.texture != NULL);

  /* Copy volume data to texture */
  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUImageCopyTexture) {
                          .texture = textures.volume.texture,
                          .mipLevel = 0,
                          .origin = (WGPUOrigin3D) {
                              .x = 0,
                              .y = 0,
                              .z = 0,
                          },
                          .aspect = WGPUTextureAspect_All,
                        },
                        textures.volume.data, ARRAY_SIZE(textures.volume.data),
                        &(WGPUTextureDataLayout){
                          .offset       = 0,
                          .bytesPerRow  = bytes_per_row,
                          .rowsPerImage = blocks_high,
                        },
                        &(WGPUExtent3D){
                          .width              = width,
                          .height             = height,
                          .depthOrArrayLayers = depth,
                        });

  /* Create a sampler with linear filtering for smooth interpolation. */
  textures.volume.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Volume texture sampler",
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .maxAnisotropy = 16,
                          });
  ASSERT(textures.volume.sampler != NULL);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "Volume texture view",
    .dimension       = WGPUTextureViewDimension_3D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  textures.volume.view
    = wgpuTextureCreateView(textures.volume.texture, &texture_view_dec);
  ASSERT(textures.volume.view != NULL);
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  /* Bind Group */
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Vertex shader uniform buffer */
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1: Fragment shader image sampler */
      .binding = 1,
      .sampler = textures.volume.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      /* Binding 2 : Fragment shader texture view */
      .binding     = 2,
      .textureView = textures.volume.view,
    },
  };

  uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = "Uniform bind group",
      .layout     = wgpuRenderPipelineGetBindGroupLayout(render_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(uniform_bind_group != NULL);
}

static void setup_render_pass(void)
{
  /* Color attachment */
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
    .view          = NULL, /* Assigned later */
    .resolveTarget = NULL,
    .depthSlice    = ~0,
    .loadOp        = WGPULoadOp_Clear,
    .storeOp       = WGPUStoreOp_Discard,
    .clearValue    = (WGPUColor) {
      .r = 0.5f,
      .g = 0.5f,
      .b = 0.5f,
      .a = 1.0f,
    },
  };

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                = "Render pass descriptor",
    .colorAttachmentCount = 1,
    .colorAttachments     = rp_color_att_descriptors,
  };
}

/* Create the graphics pipeline */
static void prepare_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      /* Vertex shader WGSL */
                      .label            = "Volume texture 3d - Vertex shader WGSL",
                      .wgsl_code.source = volume_shader_wgsl,
                      .entry            = "vertex_main",
                    },
                  });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      /* Fragment shader WGSL */
                      .label            = "Volume texture 3d - Fragment shader WGSL",
                      .wgsl_code.source = volume_shader_wgsl,
                      .entry            = "fragment_main",
                    },
                    .target_count = 1,
                    .targets = &color_target_state,
                  });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = sample_count,
      });

  /* Create rendering pipeline using the specified states */
  render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label     = "Volume texture 3d - Render pipeline",
                            .primitive = primitive_state,
                            .vertex    = vertex_state,
                            .fragment  = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(render_pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_uniform_buffer(context);
    prepare_multisampled_framebuffer(context->wgpu_context);
    prepare_volume_texture(context->wgpu_context);
    prepare_pipeline(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
    setup_render_pass();
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    if (imgui_overlay_checkBox(context->imgui_overlay, "Rotate Camera",
                               &params.rotate_camera)) {
      update_uniform_buffers(context);
    }
    if (imgui_overlay_slider_float(context->imgui_overlay, "Near", &params.near,
                                   2.0f, 7.0f, "%.1f")) {
      update_uniform_buffers(context);
    }
    if (imgui_overlay_slider_float(context->imgui_overlay, "Far", &params.far,
                                   2.0f, 7.0f, "%.1f")) {
      update_uniform_buffers(context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  rp_color_att_descriptors[0].view = textures.multisampled.framebuffer;
  rp_color_att_descriptors[0].resolveTarget
    = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, render_pipeline);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    uniform_bind_group, 0, 0);

  /* Draw */
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);

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
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  if (params.rotate_camera) {
    update_uniform_buffers(context);
  }
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  WGPU_RELEASE_RESOURCE(Texture, textures.volume.texture)
  WGPU_RELEASE_RESOURCE(TextureView, textures.volume.view)
  WGPU_RELEASE_RESOURCE(Sampler, textures.volume.sampler)
  WGPU_RELEASE_RESOURCE(Texture, textures.multisampled.texture)
  WGPU_RELEASE_RESOURCE(TextureView, textures.multisampled.framebuffer)
  WGPU_RELEASE_RESOURCE(BindGroup, uniform_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipeline)
}

void example_volume_rendering_texture_3d(int argc, char* argv[])
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
static const char* volume_shader_wgsl = CODE(
  struct Uniforms {
    inverseModelViewProjectionMatrix : mat4x4f,
  }

  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  @group(0) @binding(1) var mySampler: sampler;
  @group(0) @binding(2) var myTexture: texture_3d<f32>;

  struct VertexOutput {
    @builtin(position) Position : vec4f,
    @location(0) near : vec3f,
    @location(1) step : vec3f,
  }

  const NumSteps = 64u;

  @vertex
  fn vertex_main(
    @builtin(vertex_index) VertexIndex : u32
  ) -> VertexOutput {
    var pos = array<vec2f, 3>(
      vec2(-1.0, 3.0),
      vec2(-1.0, -1.0),
      vec2(3.0, -1.0)
    );
    var xy = pos[VertexIndex];
    var near = uniforms.inverseModelViewProjectionMatrix * vec4f(xy, 0.0, 1);
    var far = uniforms.inverseModelViewProjectionMatrix * vec4f(xy, 1, 1);
    near /= near.w;
    far /= far.w;
    return VertexOutput(
      vec4f(xy, 0.0, 1.0),
      near.xyz,
      (far.xyz - near.xyz) / f32(NumSteps)
    );
  }

  @fragment
  fn fragment_main(
    @location(0) near: vec3f,
    @location(1) step: vec3f
  ) -> @location(0) vec4f {
    var rayPos = near;
    var result = 0.0;
    for (var i = 0u; i < NumSteps; i++) {
      let texCoord = (rayPos.xyz + 1.0) * 0.5;
      let sample =
        textureSample(myTexture, mySampler, texCoord).r * 4.0 / f32(NumSteps);
      let intersects =
        all(rayPos.xyz < vec3f(1.0)) && all(rayPos.xyz > vec3f(-1.0));
      result += select(0.0, (1.0 - result) * sample, intersects && result < 1.0);
      rayPos += step;
    }
    return vec4f(vec3f(result), 1.0);
  }
);
// clang-format on
