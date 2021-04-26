#include "example_base.h"
#include "examples.h"

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

// Vertex buffer
static struct vertices_t {
  WGPUBuffer buffer;
  uint64_t count;
  uint64_t size;
} vertices = {0};

// Pipeline
static WGPURenderPipeline pipeline;

// Bind groups stores the resources bound to the binding points in a shader
static WGPUBindGroup uniform_bind_group;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachmentDescriptor rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

// Texture and sampler
static struct video_texture_t {
  WGPUSampler sampler;
  WGPUTexture texture;
  WGPUTextureView view;
} video_texture = {0};

static struct video_info_t {
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

static const float rectangle_vertices[30] = {
  1.0f,  1.0f,  0.0f, 1.0f, 0.0f, //
  1.0f,  -1.0f, 0.0f, 1.0f, 1.0f, //
  -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, //
  1.0f,  1.0f,  0.0f, 1.0f, 0.0f, //
  -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, //
  -1.0f, 1.0f,  0.0f, 0.0f, 0.0f, //
};

// Prepare vertex buffer
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  vertices.count = 6;
  vertices.size  = sizeof(rectangle_vertices);

  // Vertex buffer
  vertices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, rectangle_vertices, vertices.size, WGPUBufferUsage_Vertex);
}

static void prepare_video_texture(wgpu_context_t* wgpu_context)
{
  // Create the texture
  video_texture.texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .size          = (WGPUExtent3D){
        .width  = video_info.frame_size.width,
        .height = video_info.frame_size.height,
        .depth  = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_Sampled,
  });

  // Create the texture view
  video_texture.view = wgpuTextureCreateView(
    video_texture.texture, &(WGPUTextureViewDescriptor){
                             .format          = WGPUTextureFormat_RGBA8Unorm,
                             .dimension       = WGPUTextureViewDimension_2D,
                             .baseMipLevel    = 0,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = 0,
                             .arrayLayerCount = 1,
                           });

  // Create the sampler
  video_texture.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUFilterMode_Nearest,
                            .maxAnisotropy = 1,
                          });
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachmentDescriptor) {
      .attachment = NULL,
      .loadOp = WGPULoadOp_Clear,
      .storeOp = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount = 1,
    .colorAttachments     = rp_color_att_descriptors,
  };
}

static void prepare_uniform_bind_group(wgpu_context_t* wgpu_context)
{
  // Uniform bind group
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .sampler = video_texture.sampler,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .textureView = video_texture.view,
    },
  };
  uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .layout     = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(uniform_bind_group != NULL)
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Construct the different states making up the pipeline

  // Rasterization state
  WGPURasterizationStateDescriptor rasterization_state_desc
    = wgpu_create_rasterization_state_descriptor(
      &(create_rasterization_state_desc_t){
        .front_face = WGPUFrontFace_CW,
        .cull_mode  = WGPUCullMode_Back,
      });

  // Color blend state
  WGPUColorStateDescriptor color_state_desc
    = wgpu_create_color_state_descriptor(&(create_color_state_desc_t){
      .format       = wgpu_context->swap_chain.format,
      .enable_blend = true,
    });

  // Vertex input binding
  WGPU_VERTSTATE(video_uploading, 20,
                 // Attribute location 0: Position
                 WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
                 // Attribute location 1: UV
                 WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 12))

  // Shaders
  // Vertex shader
  wgpu_shader_t vert_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Vertex shader SPIR-V
                    .file = "shaders/video_uploading/shader.vert.spv",
                  });
  // Fragment shader
  wgpu_shader_t frag_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Fragment shader SPIR-V
                    .file = "shaders/video_uploading/shader.frag.spv",
                  });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      // Vertex shader
      .vertexStage = vert_shader.programmable_stage_descriptor,
      // Fragment shader
      .fragmentStage = &frag_shader.programmable_stage_descriptor,
      // Rasterization state
      .rasterizationState     = &rasterization_state_desc,
      .primitiveTopology      = WGPUPrimitiveTopology_TriangleList,
      .colorStateCount        = 1,
      .colorStates            = &color_state_desc,
      .depthStencilState      = NULL,
      .vertexState            = &vert_state_video_uploading,
      .sampleCount            = 1,
      .sampleMask             = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    });

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  wgpu_shader_release(&frag_shader);
  wgpu_shader_release(&vert_shader);
}

static int prepare_video(const char* fname)
{
  init_video_decode();
  open_video_file(fname);

  get_video_dimension(&video_info.frame_size.width,
                      &video_info.frame_size.height);

  start_video_decode();

  return 0;
}

static int update_capture_texture(wgpu_context_t* wgpu_context)
{
  int video_w, video_h;
  void* video_buf;

  get_video_dimension(&video_w, &video_h);
  get_video_buffer(&video_buf);

  if (video_buf) {
    wgpu_image_to_texure(wgpu_context, &(texture_image_desc_t){
                                         .width  = video_info.frame_size.width,
                                         .height = video_info.frame_size.height,
                                         .channels = 4u,
                                         .pixels   = (uint8_t*)video_buf,
                                         .texture  = video_texture.texture,
                                       });
  }

  return 0;
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
    return 0;
  }

  return 1;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  rp_color_att_descriptors[0].attachment
    = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    uniform_bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, vertices.count, 1, 0, 0);

  // End render pass
  wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Get command buffer
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

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
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
     .title  = example_title,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
  });
  // clang-format on
}
