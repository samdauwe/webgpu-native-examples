#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Pseudorandom Number Generation
 *
 * A WebGPU example demonstrating pseudorandom number generation on the GPU. A
 * 32-bit PCG hash is used which is fast enough to be useful for real-time,
 * while also being high-quality enough for almost any graphics use-case.
 *
 * A pseudorandom number generator (PRNG), also known as a deterministic random
 * bit generator (DRBG), is an algorithm for generating a sequence of numbers
 * whose properties approximate the properties of sequences of random numbers.
 *
 * Ref:
 * https://en.wikipedia.org/wiki/Pseudorandom_number_generator
 * https://github.com/wwwtyro/webgpu-prng-example
 * https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* prng_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Pseudorandom Number Generation example
 * -------------------------------------------------------------------------- */

/* Vertex layout used in this example */
typedef struct {
  vec2 position;
} vertex_t;

static struct {
  /* Vertex buffer and attributes */
  wgpu_buffer_t vertices;
  /* Uniform buffer block object */
  wgpu_buffer_t uniform_buffer_fs;
  /* Uniform block fragment shader */
  struct {
    uint32_t offset;
  } ubo_fs;
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroupLayout uniform_bind_group_layout;
  WGPUBindGroup uniform_bind_group;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  int8_t initialized;
  int8_t paused;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.125, 0.125, 0.250, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_dscriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
};

static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  /* Vertices */
  static const vertex_t vertex_buffer[6] = {
    {
      .position = {-1.0f, -1.0f},
    },
    {
      .position = {1.0f, -1.0f},
    },
    {
      .position = {1.0f, 1.0f},
    },
    {
      .position = {-1.0f, -1.0f},
    },
    {
      .position = {1.0f, 1.0f},
    },
    {
      .position = {-1.0f, 1.0f},
    },
  };

  /* Create vertex buffer */
  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_buffer),
                    .count = (uint32_t)ARRAY_SIZE(vertex_buffer),
                    .initial.data = vertex_buffer,
                  });
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  state.uniform_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor) {
      .label      = STRVIEW("Bind group layout"),
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Fragment shader)
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout){
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(state.ubo_fs),
        },
        .sampler = {0},
      }
    }
  );
  ASSERT(state.uniform_bind_group_layout != NULL);

  // Pipeline layout
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.uniform_bind_group_layout,
    });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  // Bind Group
  state.uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor) {
     .label      = STRVIEW("Bind group"),
     .layout     = state.uniform_bind_group_layout,
     .entryCount = 1,
     .entries    = &(WGPUBindGroupEntry) {
       // Binding 0: Uniform buffer (Fragment shader)
       .binding = 0,
       .buffer  = state.uniform_buffer_fs.buffer,
       .offset  = 0,
       .size    = state.uniform_buffer_fs.size,
     },
   }
  );
  ASSERT(state.uniform_bind_group != NULL);
}

void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.ubo_fs.offset = (uint32_t)roundf(random_float() * 4294967295.f);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_fs.buffer, 0,
                       &state.ubo_fs, state.uniform_buffer_fs.size);
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Create a uniform buffer */
  state.uniform_buffer_fs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.ubo_fs), /* One u32, 4 bytes each */
                  });

  /* Upload the uniform buffer to the GPU */
  update_uniform_buffers(wgpu_context);
}

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, prng_shader_wgsl);

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(quad, sizeof(vertex_t),
                            // Attribute location 0: Position
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2,
                                               offsetof(vertex_t, position)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("PRNG - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &quad_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("fs_main"),
      .module      = shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState)  {
        .format    = wgpu_context->render_format,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_Back,
      .frontFace = WGPUFrontFace_CCW
    },
    .multisample = {
      .count = 1,
      .mask  = 0xffffffff
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    init_vertex_buffer(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_bind_group(wgpu_context);
    init_pipelines(wgpu_context);
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

  if (!state.paused) {
    update_uniform_buffers(wgpu_context);
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_dscriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.uniform_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderDraw(rpass_enc, state.vertices.count, 1, 0, 0);
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
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_fs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.uniform_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.uniform_bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Pseudorandom Number Generation",
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
static const char* prng_shader_wgsl = CODE(
  struct Uniforms {
    offset: u32
  }

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;

  var<private> state: u32;

  // From https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/\n"
  fn pcg_hash(input: u32) -> u32 {
    state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
  }

  @vertex
  fn vs_main(@location(0) position : vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
  }

  @fragment
  fn fs_main(
    @builtin(position) position: vec4<f32>,
  ) -> @location(0) vec4<f32> {
    var seed = u32(512.0 * position.y + position.x) + uniforms.offset;
    var pcg = pcg_hash(seed);
    var v = f32(pcg) * (1.0 / 4294967295.0);
    return vec4<f32>(v, v, v, 1.0);
  }
);
// clang-format on
