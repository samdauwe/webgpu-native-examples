#include "example_base.h"

#include <string.h>

#include "../core/video_decode.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Video Texture
 *
 * This example shows how to upload video frames to WebGPU.
 *
 * Note:
 * - Uses FFMPEG for decoding video frames
 * - Video loops by default
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/blob/main/src/pages/samples/videoUploading.ts
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Video Texture example
 * -------------------------------------------------------------------------- */

// Vertex buffer
static wgpu_buffer_t vertices = {0};

// Pipeline
static WGPURenderPipeline pipeline = {0};

// Bind groups stores the resources bound to the binding points in a shader
static WGPUBindGroup uniform_bind_group = {0};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Texture and sampler
static struct {
  WGPUSampler sampler;
  WGPUTexture texture;
  WGPUTextureView view;
} video_texture = {0};

static struct {
  struct {
    int32_t width;
    int32_t height;
  } frame_size;
} video_info = {0};

static const char* video_file_location
  = "videos/video_uploading/big_buck_bunny_trailer.mp4";

// Other variables
static const char* example_title = "Video Texture";
static bool prepared             = false;

// clang-format off
static const float rectangle_vertices[30] = {
   1.0f,  1.0f, 0.0f, 1.0f, 0.0f, //
   1.0f, -1.0f, 0.0f, 1.0f, 1.0f, //
  -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, //
   1.0f,  1.0f, 0.0f, 1.0f, 0.0f, //
  -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, //
  -1.0f,  1.0f, 0.0f, 0.0f, 0.0f, //
};
// clang-format on

// Prepare vertex buffer
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(rectangle_vertices),
                    .count = 6,
                    .initial.data = rectangle_vertices,
                  });
}

static void prepare_video_texture(wgpu_context_t* wgpu_context)
{
  // Create the texture
  video_texture.texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = "Video texture",
      .size          = (WGPUExtent3D){
        .width               = video_info.frame_size.width,
        .height              = video_info.frame_size.height,
        .depthOrArrayLayers  = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
  });

  // Create the texture view
  video_texture.view = wgpuTextureCreateView(
    video_texture.texture, &(WGPUTextureViewDescriptor){
                             .label           = "Video texture view",
                             .format          = WGPUTextureFormat_RGBA8Unorm,
                             .dimension       = WGPUTextureViewDimension_2D,
                             .baseMipLevel    = 0,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = 0,
                             .arrayLayerCount = 1,
                           });
  ASSERT(video_texture.view != NULL);

  // Create the sampler
  video_texture.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Video texture sampler",
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                            .maxAnisotropy = 1,
                          });
  ASSERT(video_texture.sampler != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
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

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                = "Render pass descriptor",
    .colorAttachmentCount = 1,
    .colorAttachments     = render_pass.color_attachments,
  };
}

static void prepare_uniform_bind_group(wgpu_context_t* wgpu_context)
{
  // Uniform bind group
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : video texture sampler */
      .binding = 0,
      .sampler = video_texture.sampler,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1 : video texture view */
      .binding     = 1,
      .textureView = video_texture.view,
    },
  };
  uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = "Uniform - Bind group",
      .layout     = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(uniform_bind_group != NULL)
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    video_uploading, 20,
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    /* Attribute location 1: UV */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 12))

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  /* Vertex shader WGSL */
                  .label            = "Vertex shader WGSL",
                  .wgsl_code.source = vertex_shader_wgsl,
                },
                .buffer_count = 1,
                .buffers      = &video_uploading_vertex_buffer_layout,
              });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  /* Fragment shader WGSL */
                  .label            = "Fragment shader WGSL",
                  .wgsl_code.source = fragment_shader_wgsl,
                },
                .target_count = 1,
                .targets      = &color_target_state,
              });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Create rendering pipeline using the specified states */
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Video uploading render pipeline",
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int prepare_video(const char* fname)
{
  init_video_decode();
  open_video_file(fname);

  get_video_dimension(&video_info.frame_size.width,
                      &video_info.frame_size.height);

  start_video_decode();

  return EXIT_SUCCESS;
}

static int update_capture_texture(wgpu_context_t* wgpu_context)
{
  int video_w = 0, video_h = 0;
  void* video_buf = NULL;

  get_video_dimension(&video_w, &video_h);
  get_video_buffer(&video_buf);

  if (video_buf) {
    wgpu_image_to_texure(wgpu_context, video_texture.texture,
                         (uint8_t*)video_buf,
                         (WGPUExtent3D){
                           .width              = video_info.frame_size.width,
                           .height             = video_info.frame_size.height,
                           .depthOrArrayLayers = 1,
                         },
                         4u);
  }

  return EXIT_SUCCESS;
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_video(video_file_location);
    prepare_vertex_buffer(context->wgpu_context);
    prepare_video_texture(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    prepare_uniform_bind_group(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    uniform_bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, vertices.count, 1, 0, 0);

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  update_capture_texture(wgpu_context);

  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
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
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, uniform_bind_group)
  WGPU_RELEASE_RESOURCE(Sampler, video_texture.sampler)
  WGPU_RELEASE_RESOURCE(Texture, video_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, video_texture.view)
}

void example_video_uploading(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title           = example_title,
     .vsync           = true,
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
static const char* vertex_shader_wgsl = CODE(
  struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) uv : vec2<f32>
  };

  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragUV : vec2<f32>
  };

  @vertex
  fn main(input : VertexInput) -> VertexOutput {
    return VertexOutput(vec4<f32>(input.position, 1.0), input.uv);
  }
);

static const char* fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var mySampler: sampler;
  @group(0) @binding(1) var myTexture: texture_2d<f32>;

  @fragment
  fn main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(myTexture, mySampler, fragUV);
  }
);
// clang-format on
