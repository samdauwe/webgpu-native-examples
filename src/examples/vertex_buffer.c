#include "example_base.h"

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

static uint64_t num_vertices = 0ull;

// Buffer containing the vertices
static WGPUBuffer vertices_buffer = {0};

// Rendering
static WGPURenderPipeline render_pipeline = NULL;

// Other variables
static const char* example_title = "Vertex Buffer";
static bool prepared             = false;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass;

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
  wgpu_buffer_write_mapped_range(
    vertices_buffer, num_vertices * sizeof(vertex_t), data, sizeof(data));
  num_vertices += 8;

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

  num_vertices = 0;

  float2 v[3] = {
    // clang-format off
    {-4.0f, -4.0f}, //
    {-4.0f,  4.0f}, //
    {12.0f, -4.0f}, //
    // clang-format on
  };

  WGPUBufferDescriptor buffer_desc = {
    .usage            = WGPUBufferUsage_Vertex,
    .size             = MAX_VERTICES * sizeof(vertex_t),
    .mappedAtCreation = true,
  };
  vertices_buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
  ASSERT(vertices_buffer);

  float viewport_x_scale
    = (float)wgpu_context->surface.height / wgpu_context->surface.width;
  for (int i = 0; i < 3; ++i) {
    v[i].x *= viewport_x_scale;
  }
  divide(&v[0], &v[1], &v[2], RECURSION_LIMIT);

  wgpuBufferUnmap(vertices_buffer);
}

static void prepare_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_LineList,
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

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    vertex_buffer, sizeof(vertex_t),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2, offsetof(vertex_t, pos)),
    /* Attribute location 1: Color */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32, offsetof(vertex_t, color)))

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
     .shader_desc = (wgpu_shader_desc_t){
       /* Vertex shader WGSL */
       .label            = "Vertex shader WGSL",
       .wgsl_code.source = vertex_shader_wgsl,
       },
     .buffer_count = 1,
     .buffers      = &vertex_buffer_vertex_buffer_layout,
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
  render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Vertex buffer - Render pipeline",
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(render_pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
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
     .r = 0.0f,
     .g = 0.0f,
     .b = 0.0f,
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

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    create_geometry(context->wgpu_context);
    prepare_pipeline(context->wgpu_context);
    setup_render_pass();
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, render_pipeline);

  // Set viewport
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Bind vertex buffer (contains positions and colors)
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices_buffer, 0,
                                       num_vertices * sizeof(vertex_t));

  // Draw geometry
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, num_vertices, 1, 0, 0);

  // Create command buffer and cleanup
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_context_t* wgpu_context)
{
  /* Get next image in the swap chain (back/front buffer) */
  wgpu_swap_chain_get_current_image(wgpu_context);

  /* Create command buffer */
  WGPUCommandBuffer command_buffer = build_command_buffer(wgpu_context);
  ASSERT(command_buffer != NULL);

  /* Submit command buffer to the queue */
  wgpu_flush_command_buffers(wgpu_context, &command_buffer, 1);

  /* Present the current buffer to the swap chain */
  wgpu_swap_chain_present(wgpu_context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context->wgpu_context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(Buffer, vertices_buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipeline)
}

void example_vertex_buffer(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .vsync = true,
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
