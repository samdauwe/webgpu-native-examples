#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#include <stdbool.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - MSAA Line
 *
 * The parts of this example enabling MSAA are:
 * *    The render pipeline is created with a sample_count > 1.
 * *    A new texture with a sample_count > 1 is created and set as the
 *      color_attachment instead of the swapchain.
 * *    The swapchain is now specified as a resolve_target.
 *
 * The parts of this example enabling LineList are:
 * *   Set the primitive_topology to PrimitiveTopology::LineList.
 * *   Vertices and Indices describe the two points that make up a line.
 *
 * Ref:
 * https://github.com/gfx-rs/wgpu/tree/trunk/examples/features/src/msaa_line
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * MSAA Line example
 * -------------------------------------------------------------------------- */

#define NUMBER_OF_LINES (50u)
#define SAMPLE_COUNT (4u)

typedef struct {
  vec2 position;
  vec4 color;
} vertex_t;

/* State struct */
static struct {
  wgpu_buffer_t vertices;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  WGPURenderBundle render_bundle;
  WGPUTexture multisampled_texture;
  WGPUTextureView multisampled_framebuffer;
  uint32_t sample_count;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  bool initialized;
} state = {
  .sample_count = SAMPLE_COUNT,
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
  },
};

static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  const uint32_t vertex_count = NUMBER_OF_LINES * 2;
  vertex_t vertex_data[vertex_count];
  float percent = 0.0f, sin_value = 0.0f, cos_value = 0.0f;
  for (uint32_t i = 0; i < NUMBER_OF_LINES; ++i) {
    percent                = (float)i / (float)NUMBER_OF_LINES;
    sin_value              = sinf(percent * PI2);
    cos_value              = cosf(percent * PI2);
    vertex_data[i * 2 + 0] = (vertex_t){
      .position = {0.f, 0.f},
      .color    = {1.f, -sin_value, cos_value, 1.f},
    };
    vertex_data[i * 2 + 1] = (vertex_t){
      .position = {1.f * cos_value, 1.f * sin_value},
      .color    = {sin_value, -cos_value, 1.f, 1.f},
    };
  }

  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_data),
                    .count = (uint32_t)ARRAY_SIZE(vertex_data),
                    .initial.data = vertex_data,
                  });
}

static void init_multisampled_framebuffer(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(Texture, state.multisampled_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.multisampled_framebuffer)

  /* Create the multi-sampled texture */
  WGPUTextureDescriptor multisampled_frame_desc = {
    .label         = STRVIEW("Multi-sampled - Texture"),
    .size          = (WGPUExtent3D){
      .width               = wgpu_context->width,
      .height              = wgpu_context->height,
      .depthOrArrayLayers  = 1,
     },
    .mipLevelCount = 1,
    .sampleCount   = state.sample_count,
    .dimension     = WGPUTextureDimension_2D,
    .format        = wgpu_context->render_format,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.multisampled_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &multisampled_frame_desc);
  ASSERT(state.multisampled_texture != NULL);

  /* Create the multi-sampled texture view */
  state.multisampled_framebuffer
    = wgpuTextureCreateView(state.multisampled_texture,
                            &(WGPUTextureViewDescriptor){
                              .label  = STRVIEW("Multi-sampled - Texture view"),
                              .format = wgpu_context->render_format,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                            });
  ASSERT(state.multisampled_framebuffer != NULL);
}

static void init_render_bundle(wgpu_context_t* wgpu_context)
{
  WGPURenderBundleEncoderDescriptor rbe_desc = {
    .colorFormatCount = 1,
    .colorFormats     = &wgpu_context->render_format,
    .sampleCount      = state.sample_count,
  };
  WGPURenderBundleEncoder encoder
    = wgpuDeviceCreateRenderBundleEncoder(wgpu_context->device, &rbe_desc);
  wgpuRenderBundleEncoderSetPipeline(encoder, state.render_pipeline);
  wgpuRenderBundleEncoderSetVertexBuffer(encoder, 0, state.vertices.buffer, 0,
                                         WGPU_WHOLE_SIZE);
  wgpuRenderBundleEncoderDraw(encoder, state.vertices.count, 1, 0, 0);
  state.render_bundle
    = wgpuRenderBundleEncoderFinish(encoder, &(WGPURenderBundleDescriptor){
                                               .label = STRVIEW("main"),
                                             });
  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, encoder);
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {0};
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                         &pipeline_layout_desc);
  ASSERT(state.pipeline_layout != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(msaa_line, sizeof(vertex_t),
                            // Attribute location 0: Position
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2,
                                               offsetof(vertex_t, position)),
                            // Attribute location 1: Color
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                                               offsetof(vertex_t, color)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Two cubes - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &msaa_line_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_LineList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    },
    .multisample = {
       .count = state.sample_count,
       .mask  = 0xffffffff
    },
  };

  state.render_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.render_pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    init_vertex_buffer(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_pipeline(wgpu_context);
    init_multisampled_framebuffer(wgpu_context);
    init_render_bundle(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_multisampled_framebuffer(wgpu_context);
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* Set target frame buffer */
  if (state.sample_count == 1) {
    state.color_attachment.view          = wgpu_context->swapchain_view;
    state.color_attachment.resolveTarget = NULL;
  }
  else {
    state.color_attachment.view          = state.multisampled_framebuffer;
    state.color_attachment.resolveTarget = wgpu_context->swapchain_view;
  }

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Execute bundles. */
  wgpuRenderPassEncoderExecuteBundles(rpass_enc, 1, &state.render_bundle);

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Texture, state.multisampled_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.multisampled_framebuffer)
  WGPU_RELEASE_RESOURCE(RenderBundle, state.render_bundle)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "MSAA Line",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
    .sample_count   = state.sample_count,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* vertex_shader_wgsl = CODE(
  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) f_Color : vec4<f32>,
  };

  @vertex
  fn main(
    @location(0) a_Pos : vec2<f32>,
    @location(1) a_Color : vec4<f32>
  ) -> Output {
    var output : Output;
    output.position = vec4(a_Pos, 0.0, 1.0);
    output.f_Color = a_Color;
    return output;
  }
);

static const char* fragment_shader_wgsl = CODE(
  @fragment
  fn main(@location(0) v_Color : vec4<f32>) -> @location(0) vec4<f32> {
    return v_Color;
  }
);
// clang-format on
