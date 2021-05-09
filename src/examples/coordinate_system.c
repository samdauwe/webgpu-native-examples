#include "example_base.h"
#include "examples.h"

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
static WGPUPipelineLayout pipeline_layout;

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachmentDescriptor rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

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
  WGPUBuffer vertices_y_up;
  WGPUBuffer vertices_y_down;
  WGPUBuffer indices_ccw;
  WGPUBuffer indices_cw;
} quad = {0};

// Other variables
static const char* example_title = "Coordinate System";
static bool prepared             = false;

static void load_assets(wgpu_context_t* wgpu_context)
{
  textures.cw = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/texture_orientation_cw_rgba.ktx");
  textures.ccw = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/texture_orientation_ccw_rgba.ktx");

  // Create two quads with different Y orientations
  struct vertex_t {
    vec3 pos;
    vec2 uv;
  };

  const float ar
    = (float)wgpu_context->surface.height / (float)wgpu_context->surface.width;

  // WebGPU style (y points upwards)
  struct vertex_t vertices_y_pos[4] = {
    {.pos = {-1.0f * ar, -1.0f, 1.0f}, .uv = {0.0f, 1.0f}}, //
    {.pos = {-1.0f * ar, 1.0f, 1.0f}, .uv = {0.0f, 0.0f}},  //
    {.pos = {1.0f * ar, 1.0f, 1.0f}, .uv = {1.0f, 0.0f}},   //
    {.pos = {1.0f * ar, -1.0f, 1.0f}, .uv = {1.0f, 1.0f}},  //
  };

  // Vulkan style (y points downwards)
  struct vertex_t vertices_y_neg[4] = {
    {.pos = {-1.0f * ar, 1.0f, 1.0f}, .uv = {0.0f, 1.0f}},  //
    {.pos = {-1.0f * ar, -1.0f, 1.0f}, .uv = {0.0f, 0.0f}}, //
    {.pos = {1.0f * ar, -1.0f, 1.0f}, .uv = {1.0f, 0.0f}},  //
    {.pos = {1.0f * ar, 1.0f, 1.0f}, .uv = {1.0f, 1.0f}},   //
  };

  quad.vertices_y_up = wgpu_create_buffer_from_data(
    wgpu_context, vertices_y_pos, sizeof(struct vertex_t) * 4,
    WGPUBufferUsage_Vertex);
  quad.vertices_y_down = wgpu_create_buffer_from_data(
    wgpu_context, vertices_y_neg, sizeof(struct vertex_t) * 4,
    WGPUBufferUsage_Vertex);

  // Create two set of indices, one for counter clock wise, and one for clock
  // wise rendering
  static uint32_t indices_ccw[6] = {
    2, 1, 0, //
    0, 3, 2, //
  };
  quad.indices_ccw = wgpu_create_buffer_from_data(
    wgpu_context, indices_ccw, sizeof(uint32_t) * 6, WGPUBufferUsage_Index);
  static uint32_t indices_cw[6] = {
    0, 1, 2, //
    2, 3, 0, //
  };
  quad.indices_cw = wgpu_create_buffer_from_data(
    wgpu_context, indices_cw, sizeof(uint32_t) * 6, WGPUBufferUsage_Index);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Bind group CW
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0 : Fragment shader texture view
        .binding = 0,
        .textureView = textures.cw.view,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment shader image sampler
        .binding = 1,
        .sampler = textures.cw.sampler,
      },
    };
    bind_groups.cw = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.cw != NULL)
  }

  // Bind group CCW
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0 : Fragment shader texture view
        .binding = 0,
        .textureView = textures.ccw.view,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment shader image sampler
        .binding = 1,
        .sampler = textures.ccw.sampler,
      },
    };
    bind_groups.ccw = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.ccw != NULL)
  }
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Texture view
      .binding = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Sampler
      .binding = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type=WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    }
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL)

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this descriptor set layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL)
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachmentDescriptor) {
      .view       = NULL,
      .attachment = NULL,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context);

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline);

  // Rasterization state
  WGPURasterizationStateDescriptor rasterization_state_desc
    = wgpu_create_rasterization_state_descriptor(
      &(create_rasterization_state_desc_t){
        .front_face
        = settings.winding_order == 0 ? WGPUFrontFace_CW : WGPUFrontFace_CCW,
        .cull_mode = WGPUCullMode_None + settings.cull_mode,
      });

  // Color blend state
  WGPUColorStateDescriptor color_state_desc
    = wgpu_create_color_state_descriptor(&(create_color_state_desc_t){
      .format       = wgpu_context->swap_chain.format,
      .enable_blend = false,
    });

  // Depth and stencil state containing depth and stencil compare and test
  // operations
  WGPUDepthStencilStateDescriptor depth_stencil_state_desc
    = wgpu_create_depth_stencil_state_descriptor(
      &(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24PlusStencil8,
        .depth_write_enabled = false,
      });

  // Vertex input binding (=> Input assembly)
  WGPU_VERTSTATE(
    quad, sizeof(float) * 5,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    // Attribute location 1: UV
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, sizeof(float) * 3))

  // Shaders
  // Vertex shader
  wgpu_shader_t vert_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Vertex shader SPIR-V
                    .file = "shaders/coordinate_system/quad.vert.spv",
                  });
  // Fragment shader
  wgpu_shader_t frag_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Fragment shader SPIR-V
                    .file = "shaders/coordinate_system/quad.frag.spv",
                  });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .layout = pipeline_layout,
      // Vertex shader
      .vertexStage = vert_shader.programmable_stage_descriptor,
      // Fragment shader
      .fragmentStage = &frag_shader.programmable_stage_descriptor,
      // Rasterization state
      .rasterizationState     = &rasterization_state_desc,
      .primitiveTopology      = WGPUPrimitiveTopology_TriangleList,
      .colorStateCount        = 1,
      .colorStates            = &color_state_desc,
      .depthStencilState      = &depth_stencil_state_desc,
      .vertexState            = &vert_state_quad,
      .sampleCount            = 1,
      .sampleMask             = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    });

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  wgpu_shader_release(&frag_shader);
  wgpu_shader_release(&vert_shader);
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Scene")) {
    imgui_overlay_text("Quad type");
    static const char* quadtype[2] = {"VK (y negative)", "WebGPU (y positive)"};
    imgui_overlay_combo_box(context->imgui_overlay, "##quadtype",
                            &settings.quad_type, quadtype, 2);
  }

  if (imgui_overlay_header("Pipeline")) {
    imgui_overlay_text("Winding order");
    static const char* windingorder[2] = {"clock wise", "counter clock wise"};
    if (imgui_overlay_combo_box(context->imgui_overlay, "##windingorder",
                                &settings.winding_order, windingorder, 2)) {
      prepare_pipelines(context->wgpu_context);
    }
    imgui_overlay_text("Cull mode");
    static const char* cullmode[3] = {"none", "front face", "back face"};
    if (imgui_overlay_combo_box(context->imgui_overlay, "##cullmode",
                                &settings.cull_mode, cullmode, 3)) {
      prepare_pipelines(context->wgpu_context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Render the quad with clock wise and counter clock wise indices, visibility
  // is determined by pipeline settings
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_groups.cw,
                                    0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(wgpu_context->rpass_enc, quad.indices_cw,
                                      WGPUIndexFormat_Uint32, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    wgpu_context->rpass_enc, 0,
    settings.quad_type == 0 ? quad.vertices_y_down : quad.vertices_y_up, 0, 0);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, 6, 1, 0, 0, 0);

  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_groups.ccw,
                                    0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(wgpu_context->rpass_enc, quad.indices_ccw,
                                      WGPUIndexFormat_Uint32, 0, 0);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, 6, 1, 0, 0, 0);

  // End render pass
  wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
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

  return 0;
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
    return 0;
  }

  return 1;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
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
  WGPU_RELEASE_RESOURCE(Buffer, quad.vertices_y_up)
  WGPU_RELEASE_RESOURCE(Buffer, quad.vertices_y_down)
  WGPU_RELEASE_RESOURCE(Buffer, quad.indices_ccw)
  WGPU_RELEASE_RESOURCE(Buffer, quad.indices_cw)
}

void example_coordinate_system(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
  });
  // clang-format on
}
