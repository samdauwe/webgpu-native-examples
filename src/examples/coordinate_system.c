#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Coordinate System
 *
 * Illustrates the coordinate systems used in WebGPU. WebGPU’s coordinate
 * systems match DirectX and Metal’s coordinate systems in a graphics pipeline.
 * Y-axis is up in normalized device coordinate (NDC): point(-1.0, -1.0) in NDC
 * is located at the bottom-left corner of NDC. This example has several options
 * for changing relevant pipeline state, and displaying meshes with WebGPU or
 * Vulkan style coordinates.
 *
 * Ref:
 * https://gpuweb.github.io/gpuweb/
 * https://github.com/gpuweb/gpuweb/issues/416
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/negativeviewportheight/negativeviewportheight.cpp
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* quad_vertex_shader_wgsl;
static const char* quad_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Coordinate System example
 * -------------------------------------------------------------------------- */

// Settings
static struct {
  int32_t winding_order;
  int32_t cull_mode;
  int32_t quad_type;
} settings = {
  .winding_order = 1,
  .cull_mode     = (int32_t)WGPUCullMode_Back,
  .quad_type     = 1,
};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// The bind group layout
static WGPUBindGroupLayout bind_group_layout;

static struct {
  WGPUBindGroup cw;
  WGPUBindGroup ccw;
} bind_groups = {0};

static struct {
  texture_t cw;
  texture_t ccw;
} textures = {0};

static struct {
  wgpu_buffer_t vertices_y_up;
  wgpu_buffer_t vertices_y_down;
  wgpu_buffer_t indices_ccw;
  wgpu_buffer_t indices_cw;
} quad = {0};

// Other variables
static const char* example_title = "Coordinate System";
static bool prepared             = false;

static void load_assets(wgpu_context_t* wgpu_context)
{
  textures.cw = wgpu_create_texture_from_file(
    wgpu_context, "textures/texture_orientation_cw_rgba.png", NULL);
  textures.ccw = wgpu_create_texture_from_file(
    wgpu_context, "textures/texture_orientation_ccw_rgba.png", NULL);

  // Create two quads with different Y orientations
  struct vertex_t {
    vec3 pos;
    vec2 uv;
  };

  const float ar
    = (float)wgpu_context->surface.height / (float)wgpu_context->surface.width;

  // WebGPU style (y points upwards)
  // clang-format off
  struct vertex_t vertices_y_pos[4] = {
    {.pos = {-1.0f * ar, -1.0f, 1.0f}, .uv = {0.0f, 1.0f}},
    {.pos = {-1.0f * ar,  1.0f, 1.0f}, .uv = {0.0f, 0.0f}},
    {.pos = { 1.0f * ar,  1.0f, 1.0f}, .uv = {1.0f, 0.0f}},
    {.pos = { 1.0f * ar, -1.0f, 1.0f}, .uv = {1.0f, 1.0f}},
  };
  // clang-format on

  // Vulkan style (y points downwards)
  // clang-format off
  struct vertex_t vertices_y_neg[4] = {
    {.pos = {-1.0f * ar,  1.0f, 1.0f}, .uv = {0.0f, 1.0f}},
    {.pos = {-1.0f * ar, -1.0f, 1.0f}, .uv = {0.0f, 0.0f}},
    {.pos = { 1.0f * ar, -1.0f, 1.0f}, .uv = {1.0f, 0.0f}},
    {.pos = { 1.0f * ar,  1.0f, 1.0f}, .uv = {1.0f, 1.0f}},
  };
  // clang-format on

  quad.vertices_y_up = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad vertices buffer - Y up",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(struct vertex_t) * 4,
                    .initial.data = vertices_y_pos,
                  });
  quad.vertices_y_down = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad vertices buffer - Y down",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(struct vertex_t) * 4,
                    .initial.data = vertices_y_neg,
                  });

  // Create two set of indices, one for counter clock wise, and one for clock
  // wise rendering
  static uint32_t indices_ccw[6] = {
    2, 1, 0, //
    0, 3, 2, //
  };
  quad.indices_ccw = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad indices buffer - CCW",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(uint32_t) * 6,
                    .count = (uint32_t)ARRAY_SIZE(indices_ccw),
                    .initial.data = indices_ccw,
                  });
  static uint32_t indices_cw[6] = {
    0, 1, 2, //
    2, 3, 0, //
  };
  quad.indices_cw = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad indices buffer - CW",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(uint32_t) * 6,
                    .count = (uint32_t)ARRAY_SIZE(indices_cw),
                    .initial.data = indices_cw,
                  });
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Bind group CW
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0 : Fragment shader texture view */
        .binding     = 0,
        .textureView = textures.cw.view,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: Fragment shader image sampler */
        .binding = 1,
        .sampler = textures.cw.sampler,
      },
    };
    bind_groups.cw = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Bind group - CW",
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.cw != NULL);
  }

  // Bind group CCW
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0 : Fragment shader texture view */
        .binding     = 0,
        .textureView = textures.ccw.view,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: Fragment shader image sampler */
        .binding = 1,
        .sampler = textures.ccw.sampler,
      },
    };
    bind_groups.ccw = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Bind group - CCW",
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.ccw != NULL);
  }
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : Fragment shader texture view */
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Fragment shader image sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    }
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = "Coordinate system - Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this descriptor set layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = "Coordinate system - Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
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

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline);

  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology = WGPUPrimitiveTopology_TriangleList,
    .frontFace
    = settings.winding_order == 0 ? WGPUFrontFace_CW : WGPUFrontFace_CCW,
    .cullMode = WGPUCullMode_None + settings.cull_mode,
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
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = false,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    quad, sizeof(float) * 5,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    // Attribute location 1: UV
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, sizeof(float) * 3))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "Quad - Vertex shader WGSL",
                      .wgsl_code.source = quad_vertex_shader_wgsl,
                      .entry            = "main",
                    },
                    .buffer_count = 1,
                    .buffers      = &quad_vertex_buffer_layout,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "Quad - Fragment shader WGSL",
                      .wgsl_code.source = quad_fragment_shader_wgsl,
                      .entry            = "main",
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
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label     = "Coordinate system - Render pipeline",
                            .layout    = pipeline_layout,
                            .primitive = primitive_state,
                            .vertex    = vertex_state,
                            .fragment  = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Scene")) {
    imgui_overlay_text("Quad Type");
    static const char* quadtype[2] = {"VK (Y Negative)", "WebGPU (Y Positive)"};
    imgui_overlay_combo_box(context->imgui_overlay, "##quadtype",
                            &settings.quad_type, quadtype, 2);
  }

  if (imgui_overlay_header("Pipeline")) {
    imgui_overlay_text("Winding Order");
    static const char* windingorder[2] = {"Clock Wise", "Counter Clock Wise"};
    if (imgui_overlay_combo_box(context->imgui_overlay, "##windingorder",
                                &settings.winding_order, windingorder, 2)) {
      prepare_pipelines(context->wgpu_context);
    }
    imgui_overlay_text("Cull Mode");
    static const char* cullmode[3] = {"None", "Front Face", "Back Face"};
    if (imgui_overlay_combo_box(context->imgui_overlay, "##cullmode",
                                &settings.cull_mode, cullmode, 3)) {
      prepare_pipelines(context->wgpu_context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Render the quad with clock wise and counter clock wise indices, visibility
     is determined by pipeline settings */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_groups.cw,
                                    0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, quad.indices_cw.buffer, WGPUIndexFormat_Uint32, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       settings.quad_type == 0 ?
                                         quad.vertices_y_down.buffer :
                                         quad.vertices_y_up.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, 6, 1, 0, 0, 0);

  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_groups.ccw,
                                    0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, quad.indices_ccw.buffer, WGPUIndexFormat_Uint32, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, 6, 1, 0, 0, 0);

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

  // Submit command buffer to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    load_assets(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  wgpu_destroy_texture(&textures.cw);
  wgpu_destroy_texture(&textures.ccw);
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.cw)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.ccw)
  WGPU_RELEASE_RESOURCE(Buffer, quad.vertices_y_up.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, quad.vertices_y_down.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, quad.indices_ccw.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, quad.indices_cw.buffer)
}

void example_coordinate_system(int argc, char* argv[])
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
static const char* quad_vertex_shader_wgsl = CODE(
  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>
  };

  @vertex
  fn main(
    @location(0) inPos: vec3<f32>,
    @location(1) inUV: vec2<f32>
  ) -> Output {
    var output: Output;
    output.uv = inUV;
    output.position = vec4<f32>(inPos.xyz, 1.0);
    return output;
  }
);

static const char* quad_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var textureColor: texture_2d<f32>;
  @group(0) @binding(1) var samplerColor: sampler;

  @fragment
  fn main(
    @location(0) inUV : vec2<f32>
  ) -> @location(0) vec4<f32> {
    return textureSample(textureColor, samplerColor, inUV);
  }
);
// clang-format on
