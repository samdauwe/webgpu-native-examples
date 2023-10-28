#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Pristine Grid
 *
 * A simple WebGPU implementation of the "Pristine Grid" technique described in
 * this wonderful little blog post:
 * https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
 *
 * Ref:
 * https://github.com/toji/pristine-grid-webgpu
 * https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// A WebGPU implementation of the "Pristine Grid" shader described at
// https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
static const char* grid_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Pristine Grid example
 * -------------------------------------------------------------------------- */

static const uint32_t msaa_sample_count = 4u;
static bool use_msaa                    = false;

// Vertex layout used in this example
typedef struct {
  vec3 position;
  vec2 uv;
} vertex_t;

static struct {
  vec4 line_color;
  vec4 base_color;
  vec2 line_width;
  vec4 padding;
} uniform_array;

static struct {
  WGPUColor clear_color;
  WGPUColor line_color;
  WGPUColor base_color;
  float line_width_x;
  float line_width_y;
} grid_options = {
  .clear_color = (WGPUColor) {
    .r = 0.0f,
    .g = 0.0f,
    .b = 0.2f,
    .a = 1.0f,
  },
  .line_color = (WGPUColor) {
    .r = 1.0f,
    .g = 1.0f,
    .b = 1.2f,
    .a = 1.0f,
  },
  .base_color = (WGPUColor) {
    .r = 0.0f,
    .g = 0.0f,
    .b = 0.0f,
    .a = 1.0f,
  },
  .line_width_x = 0.05f,
  .line_width_y = 0.05f,
};

static WGPUColor clear_color          = {0};
static WGPUTextureFormat depth_format = WGPUTextureFormat_Depth24Plus;

static wgpu_buffer_t vertex_buffer  = {0};
static wgpu_buffer_t index_buffer   = {0};
static wgpu_buffer_t uniform_buffer = {0};

static WGPUBindGroupLayout frame_bind_group_layout = NULL;
static WGPUBindGroupLayout bind_group_layout       = NULL;

static WGPUBindGroup frame_bind_group = NULL;
static WGPUBindGroup bind_group       = NULL;

static WGPUPipelineLayout pipeline_layout = NULL;
static WGPURenderPipeline pipeline        = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Other variables
static const char* example_title = "Pristine Grid";
static bool prepared             = false;

static void prepare_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  /* Setup vertices (x, y, z, u,v) */
  {
    static const vertex_t vertex_array[4] = {
      {
        .position = {-20.0f, -0.5f, -20.0f},
        .uv       = {0.0f, 0.0f},
      },
      {
        .position = {20.0f, -0.5f, -20.0f},
        .uv       = {200.0f, 0.0f},
      },
      {
        .position = {-20.0f, -0.5f, 20.0f},
        .uv       = {0.0f, 200.0f},
      },
      {
        .position = {20.0f, -0.5f, 20.0f},
        .uv       = {200.0f, 200.0f},
      },
    };
    vertex_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Vertex buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = sizeof(vertex_array),
                      .count = (uint32_t)ARRAY_SIZE(vertex_array),
                      .initial.data = vertex_array,
                    });
  }

  /* Setup indices */
  {
    static const uint16_t index_array[6] = {
      0, 1, 2, /* */
      1, 2, 3  /* */
    };
    index_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Index buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                      .size  = sizeof(index_array),
                      .count = (uint32_t)ARRAY_SIZE(index_array),
                      .initial.data = index_array,
                    });
  }
}

static void update_uniforms(wgpu_context_t* wgpu_context)
{
  /* Update uniforms data */
  memcpy(&clear_color, &grid_options.clear_color, sizeof(WGPUColor));
  memcpy(&uniform_array.line_color, &grid_options.line_color,
         sizeof(WGPUColor));
  memcpy(&uniform_array.base_color, &grid_options.base_color,
         sizeof(WGPUColor));
  glm_vec2_copy((vec2){grid_options.line_width_x, grid_options.line_width_y},
                uniform_array.line_width);

  /* Update uniform buffer */
  wgpu_queue_write_buffer(wgpu_context, uniform_buffer.buffer, 0,
                          &uniform_array, sizeof(uniform_array));
}

static void prepare_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Create uniform buffer */
  uniform_buffer.buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = "Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(uniform_array),
    });
  ASSERT(uniform_buffer.buffer != NULL);

  /* Update uniform buffer */
  update_uniforms(wgpu_context);
}

static void setup_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Pristine Grid bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0 : uniform array */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniform_array),
        },
        .sampler = {0},
      },
    };

    bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Pristine Grid bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layout != NULL);
  }
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayout bind_group_layouts[2] = {
    frame_bind_group_layout, /* Group 0 */
    bind_group_layout,       /* Group 1 */
  };
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = "Pristine Grid pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Pristine Grid bind group */
  {
    bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
        .label      = "Pristine Grid bind group",
        .layout     = bind_group_layout,
        .entryCount = 1,
        .entries    = &(WGPUBindGroupEntry) {
          /* Binding 0 : Uniform buffer */
          .binding = 0,
          .buffer  = uniform_buffer.buffer,
          .offset  = 0,
          .size    = uniform_buffer.size,
        },
      }
      );
    ASSERT(bind_group != NULL);
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
}

static void prepare_render_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
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
      .format              = depth_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_LessEqual;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    triangle, 20,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    // Attribute location 1: UV
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, sizeof(float) * 3))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      // Vertex shader WGSL
      .label             = "grid_shader_wgsl",
      .wgsl_code.source  = grid_shader_wgsl,
      .entry             = "vertexMain",
    },
    .buffer_count = 1,
    .buffers = &triangle_vertex_buffer_layout,
  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      // Fragment shader WGSL
      .label             = "grid_shader_wgsl",
      .wgsl_code.source  = grid_shader_wgsl,
      .entry             = "fragmentMain",
    },
    .target_count = 1,
    .targets = &color_target_state,
  });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = use_msaa ? msaa_sample_count : 1u,
      });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Pristine Grid render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertex_and_index_buffers(context->wgpu_context);
    prepare_uniform_buffer(context->wgpu_context);
    setup_bind_group_layouts(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_render_pipeline(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
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

  if (pipeline) {
    // Bind the rendering pipeline
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

    // Set viewport
    wgpuRenderPassEncoderSetViewport(
      wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
      (float)wgpu_context->surface.height, 0.0f, 1.0f);

    // Set scissor rectangle
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        wgpu_context->surface.width,
                                        wgpu_context->surface.height);

    // Set the bind groups
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      frame_bind_group, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 1, bind_group, 0,
                                      0);

    // Bind vertex buffer (contains positions & uvs)
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 0, vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);

    // Bind index buffer
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, index_buffer.buffer, WGPUIndexFormat_Uint32, 0,
      WGPU_WHOLE_SIZE);

    // Draw quad
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, index_buffer.count, 1, 0,
                              0);
  }

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
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

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context->wgpu_context);
}

/* Clean up used resources */
static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  WGPU_RELEASE_RESOURCE(Buffer, vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, index_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, frame_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, frame_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_pristine_grid(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* grid_shader_wgsl = CODE(
  // grid function from Best Darn Grid article
  fn PristineGrid(uv: vec2f, lineWidth: vec2f) -> f32 {
    let uvDDXY = vec4f(dpdx(uv), dpdy(uv));
    let uvDeriv = vec2f(length(uvDDXY.xz), length(uvDDXY.yw));
    let invertLine: vec2<bool> = lineWidth > vec2f(0.5);
    let targetWidth: vec2f = select(lineWidth, 1 - lineWidth, invertLine);
    let drawWidth: vec2f = clamp(targetWidth, uvDeriv, vec2f(0.5));
    let lineAA: vec2f = uvDeriv * 1.5;
    var gridUV: vec2f = abs(fract(uv) * 2.0 - 1.0);
    gridUV = select(1 - gridUV, gridUV, invertLine);
    var grid2: vec2f = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    grid2 *= saturate(targetWidth / drawWidth);
    grid2 = mix(grid2, targetWidth, saturate(uvDeriv * 2.0 - 1.0));
    grid2 = select(grid2, 1.0 - grid2, invertLine);
    return mix(grid2.x, 1.0, grid2.y);
  }

  struct VertexIn {
    @location(0) pos: vec4f,
    @location(1) uv: vec2f,
  }

  struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
  }

  struct Camera {
    projection: mat4x4f,
    view: mat4x4f,
  }
  @group(0) @binding(0) var<uniform> camera: Camera;

  struct GridArgs {
    lineColor: vec4f,
    baseColor: vec4f,
    lineWidth: vec2f,
  }
  @group(1) @binding(0) var<uniform> gridArgs: GridArgs;

  @vertex
  fn vertexMain(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.pos = camera.projection * camera.view * in.pos;
    out.uv = in.uv;
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
    var grid = PristineGrid(in.uv, gridArgs.lineWidth);

    // lerp between base and line color
    return mix(gridArgs.baseColor, gridArgs.lineColor, grid * gridArgs.lineColor.a);
  }
);
// clang-format on
