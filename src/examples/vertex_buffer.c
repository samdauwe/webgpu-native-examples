#include "webgpu/wgpu_common.h"

#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Vertex Buffer
 *
 * This example shows how to map a GPU buffer and use the function
 * wgpuBufferGetMappedRange.
 *
 * Ref:
 * https://github.com/juj/wasm_webgpu
 * https://github.com/juj/wasm_webgpu/blob/main/samples/vertex_buffer/vertex_buffer.c
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Vertex Buffer example
 * -------------------------------------------------------------------------- */

#define RECURSION_LIMIT 7u

static struct {
  wgpu_buffer_t vertices_buffer;
  WGPURenderPipeline pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  int8_t initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_dscriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
};

typedef struct float2 {
  float x, y;
} float2;

typedef struct vertex {
  float2 pos;
  float color;
} vertex_t;

float min(float a, float b, float c)
{
  return a < b && a < c ? a : (b < c ? b : c);
}

float max(float a, float b, float c)
{
  return a > b && a > c ? a : (b > c ? b : c);
}

float2 avg(const float2* v0, const float2* v1)
{
  return (float2){
    (v0->x + v1->x) * 0.5f, /* x */
    (v0->y + v1->y) * 0.5f, /* y */
  };
}

float2 avg8(const float2* v0, const float2* v1)
{
  return (float2){
    v0->x * 0.8f + v1->x * 0.2f, /* x */
    v0->y * 0.8f + v1->y * 0.2f, /* y */
  };
}

static void wgpu_buffer_write_mapped_range(WGPUBuffer buffer, size_t offset,
                                           void* data, size_t data_size)
{
  void* mapping = wgpuBufferGetMappedRange(buffer, offset, data_size);
  ASSERT(mapping)
  memcpy(mapping, data, data_size);
}

static void divide(const float2* v0, const float2* v1, const float2* v2,
                   int recursion_limit)
{
  if (min(v0->x, v1->x, v2->x) > 1.f) {
    return;
  }
  if (min(v0->y, v1->y, v2->y) > 1.f) {
    return;
  }
  if (max(v0->x, v1->x, v2->x) < -1.f) {
    return;
  }
  if (max(v0->y, v1->y, v2->y) < -1.f) {
    return;
  }

  float2 w1 = avg(v0, v2);
  float2 w2 = avg8(v1, v2);
  float2 w0 = avg(v2, &w2);
  float2 w3 = avg(v0, &w2);

#define COLOR(z)                                                               \
  ((recursion_limit == 3 ? 0.7f : 0.4f)                                        \
   * ((z).y * -1.f + 0.3f + (z).x + sin((z).x * 5.f) * 0.3f))

  vertex_t data[] = {
    {*v0, COLOR(*v0)}, //
    {w2, COLOR(w2)},   //
    {w2, COLOR(w2)},   //
    {w1, COLOR(w1)},   //
    {w1, COLOR(w1)},   //
    {w0, COLOR(w0)},   //
    {w1, COLOR(w1)},   //
    {w3, COLOR(w3)},   //
  };
  wgpu_buffer_write_mapped_range(state.vertices_buffer.buffer,
                                 state.vertices_buffer.count * sizeof(vertex_t),
                                 data, sizeof(data));
  state.vertices_buffer.count += 8;

  if (--recursion_limit > 0) {
    divide(&w2, v1, v0, recursion_limit);
    divide(&w3, v0, &w1, recursion_limit);
    divide(&w3, &w2, &w1, recursion_limit);
    divide(&w0, &w1, &w2, recursion_limit);
    divide(&w0, &w1, v2, recursion_limit);
  }
}

static void create_geometry(wgpu_context_t* wgpu_context)
{
// Upper limit of num vertices written = 8*(1 + 5 + 5^2 + 5^3 + ... +
// 5^recursionLimit) = 2 * 5^r - 2
#define MAX_VERTICES (2 * (uint64_t)pow(5, RECURSION_LIMIT) - 2)

  float2 v[3] = {
    // clang-format off
    {-4.0f, -4.0f}, //
    {-4.0f,  4.0f}, //
    {12.0f, -4.0f}, //
    // clang-format on
  };

  state.vertices_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Geometry - Vertices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = MAX_VERTICES * sizeof(vertex_t),
                    .count = 4,
                    .mapped_at_creation = true,
                  });

  float viewport_x_scale = (float)wgpu_context->height / wgpu_context->width;
  for (int i = 0; i < 3; ++i) {
    v[i].x *= viewport_x_scale;
  }
  divide(&v[0], &v[1], &v[2], RECURSION_LIMIT);
  wgpuBufferUnmap(state.vertices_buffer.buffer);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    vertex_buffer, sizeof(vertex_t),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2, offsetof(vertex_t, pos)),
    /* Attribute location 1: Color */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32, offsetof(vertex_t, color)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label = STRVIEW("Vertex buffer - Render pipeline"),
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &vertex_buffer_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_LineList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    },
    .multisample = {
      .count = 1,
      .mask  = 0xffffffff
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    create_geometry(wgpu_context);
    init_pipeline(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_dscriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetViewport(rpass_enc, 0.0f, 0.0f,
                                   (float)wgpu_context->width,
                                   (float)wgpu_context->height, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(rpass_enc, 0u, 0u, wgpu_context->width,
                                      wgpu_context->height);

  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 0, state.vertices_buffer.buffer, 0,
    state.vertices_buffer.count * sizeof(vertex_t));
  wgpuRenderPassEncoderDraw(rpass_enc, state.vertices_buffer.count, 1, 0, 0);
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
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices_buffer.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Vertex Buffer",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* vertex_shader_wgsl = CODE(
  struct In {
    @location(0) pos : vec2<f32>,
    @location(1) color : f32
  };

  struct Out {
    @builtin(position) pos : vec4<f32>,
    @location(0) color : f32
  };

  @vertex
  fn main(in: In) -> Out {
    var out: Out;
    out.pos = vec4<f32>(in.pos, 0.0, 1.0);
    out.color = in.color;
    return out;
  }
);

static const char* fragment_shader_wgsl = CODE(
  @fragment
  fn main(@location(0) inColor : f32) -> @location(0) vec4<f32> {
    return vec4<f32>(inColor, inColor, abs(inColor), 1.0);
  }
);
// clang-format on
