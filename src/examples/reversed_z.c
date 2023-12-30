#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Reversed Z
 *
 * This example shows the use of reversed z technique for better utilization of
 * depth buffer precision. The left column uses regular method, while the right
 * one uses reversed z technique. Both are using depth32float as their depth
 * buffer format. A set of red and green planes are positioned very close to
 * each other. Higher sets are placed further from camera (and are scaled for
 * better visual purpose). To use reversed z to render your scene, you will need
 * depth store value to be 0.0, depth compare function to be greater, and remap
 * depth range by multiplying an additional matrix to your projection matrix.
 *
 * Related reading:
 * https://developer.nvidia.com/content/depth-precision-visualized
 * https://web.archive.org/web/20220724174000/https://thxforthefish.com/posts/reverse_z/
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/blob/main/src/sample/reversedZ/main.ts
 * https://developer.nvidia.com/content/depth-precision-visualized
 * -------------------------------------------------------------------------- */

#define DEFAULT_CANVAS_WIDTH 600
#define DEFAULT_CANVAS_HEIGHT 600

#define X_COUNT 1
#define Y_COUNT 5
#define NUM_INSTANCES (X_COUNT * Y_COUNT)
#define MATRIX_FLOAT_COUNT sizeof(mat4)

// Two planes close to each other for depth precision test
static const uint32_t geometry_vertex_size
  = 4 * 8; // Byte size of one geometry vertex.
static const uint32_t geometry_position_offset = 0;
static const uint32_t geometry_color_offset
  = 4 * 4; // Byte offset of geometry vertex color attribute.
static const uint32_t geometry_draw_count = 6 * 2;

static const float d = 0.0001f; // half distance between two planes
static const float o
  = 0.5f; // half x offset to shift planes so they are only partially overlaping

static const uint32_t default_canvas_width  = (uint32_t)DEFAULT_CANVAS_WIDTH;
static const uint32_t default_canvas_height = (uint32_t)DEFAULT_CANVAS_HEIGHT;

static const uint32_t viewport_width = default_canvas_width / 2;

const uint32_t x_count            = (uint32_t)X_COUNT;
const uint32_t y_count            = (uint32_t)Y_COUNT;
const uint32_t num_instances      = (uint32_t)NUM_INSTANCES;
const uint32_t matrix_float_count = (uint32_t)MATRIX_FLOAT_COUNT; // 4x4 matrix
const uint32_t matrix_stride      = 4 * matrix_float_count;

static mat4 model_matrices[NUM_INSTANCES]                          = {0};
static float mvp_matrices_data[NUM_INSTANCES * MATRIX_FLOAT_COUNT] = {0};
static mat4 depth_range_remap_matrix                               = {
  // clang-format off
  {1.0f, 0.0f,  0.0f, 0.0f}, //
  {0.0f, 1.0f,  0.0f, 0.0f}, //
  {0.0f, 0.0f, -1.0f, 0.0f}, //
  {0.0f, 0.0f,  1.0f, 1.0f}, //
  // clang-format on
};
static mat4 tmp_mat4 = GLM_MAT4_IDENTITY_INIT;

static const WGPUTextureFormat depth_buffer_format
  = WGPUTextureFormat_Depth32Float;

// Vertex buffer and attributes
static struct wgpu_buffer_t vertices = {0};

static struct {
  WGPUPipelineLayout depth_prepass_render;
  WGPUPipelineLayout precision_pass_render;
  WGPUPipelineLayout color_pass_render;
  WGPUPipelineLayout texture_quad_pass;
} pipline_layouts = {0};

static struct {
  WGPURenderPipeline depth_pre_pass[2];
  WGPURenderPipeline precision_pass[2];
  WGPURenderPipeline color_pass[2];
  WGPURenderPipeline texture_quad_pass;
} render_pipelines = {0};

static struct {
  texture_t depth;
  texture_t default_depth;
} textures = {0};

static WGPURenderPassDescriptor depth_pre_pass_descriptor             = {0};
static WGPURenderPassDepthStencilAttachment dppd_rp_ds_att_descriptor = {0};

static WGPURenderPassColorAttachment dpd_rp_color_att_descriptors[2][1]  = {0};
static WGPURenderPassDepthStencilAttachment dpd_rp_ds_att_descriptors[2] = {0};
static WGPURenderPassDescriptor draw_pass_descriptors[2]                 = {0};

static WGPURenderPassColorAttachment tqd_rp_color_att_descriptors[2][1] = {0};
static WGPURenderPassDescriptor texture_quad_pass_descriptors[2]        = {0};

static struct {
  WGPUBindGroupLayout depth_texture;
  WGPUBindGroupLayout uniform;
} bind_group_layouts = {0};

static struct {
  WGPUBindGroup depth_texture;
  WGPUBindGroup uniform[2];
} bind_groups = {0};

static struct {
  wgpu_buffer_t uniform;
  wgpu_buffer_t camera_matrix;
  wgpu_buffer_t camera_matrix_reversed_depth;
} uniform_buffers = {0};

static uint32_t uniform_buffer_size = num_instances * matrix_stride;

// Other variables
static const char* example_title = "Reversed Z";
static bool prepared             = false;

typedef enum render_mode_enum {
  RenderMode_Color              = 0,
  RenderMode_Precision_Error    = 1,
  RenderMode_Depth_Texture_Quad = 2,
} render_mode_enum;

static render_mode_enum current_render_mode = RenderMode_Color;

typedef enum depth_buffer_mode_enum {
  DepthBufferMode_Default  = 0,
  DepthBufferMode_Reversed = 1,
} depth_buffer_mode_enum;

static const depth_buffer_mode_enum depth_buffer_modes[2] = {
  DepthBufferMode_Default,  // Default
  DepthBufferMode_Reversed, // Reversed
};

static const WGPUCompareFunction depth_compare_funcs[2] = {
  WGPUCompareFunction_Less,    // Default
  WGPUCompareFunction_Greater, // Reversed
};

static const float depth_clear_values[2] = {
  1.0f, // Default
  0.0f, // Reversed
};

static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  static const float geometry_vertex_array[(4 + 4) * 6 * 2] = {
    // float4 position, float4 color
    -1 - o, -1, d,  1, 1, 0, 0, 1, //
    1 - o,  -1, d,  1, 1, 0, 0, 1, //
    -1 - o, 1,  d,  1, 1, 0, 0, 1, //
    1 - o,  -1, d,  1, 1, 0, 0, 1, //
    1 - o,  1,  d,  1, 1, 0, 0, 1, //
    -1 - o, 1,  d,  1, 1, 0, 0, 1, //

    -1 + o, -1, -d, 1, 0, 1, 0, 1, //
    1 + o,  -1, -d, 1, 0, 1, 0, 1, //
    -1 + o, 1,  -d, 1, 0, 1, 0, 1, //
    1 + o,  -1, -d, 1, 0, 1, 0, 1, //
    1 + o,  1,  -d, 1, 0, 1, 0, 1, //
    -1 + o, 1,  -d, 1, 0, 1, 0, 1, //
  };

  // Create vertex buffer
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(geometry_vertex_array),
                    .count = (uint32_t)ARRAY_SIZE(geometry_vertex_array),
                    .initial.data = geometry_vertex_array,
                  });
}

// depthPrePass is used to render scene to the depth texture
// this is not needed if you just want to use reversed z to render a scene
static void prepare_depth_pre_pass_render_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = depth_buffer_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    depth_pre_pass, geometry_vertex_size,
    /* Attribute descriptions */
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, geometry_position_offset))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label = "Vertex depth pre pass vertex shader",
                  .file  = "shaders/reversed_z/vertexDepthPrePass.wgsl",
                  .entry = "main",
                },
                .buffer_count = 1,
                .buffers      = &depth_pre_pass_vertex_buffer_layout,
              });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // depthPrePass is used to render scene to the depth texture
  // this is not needed if you just want to use reversed z to render a scene
  WGPURenderPipelineDescriptor depth_pre_pass_render_pipeline_descriptor_base
    = (WGPURenderPipelineDescriptor){
      .label        = "depth_pre_pass_render_pipeline",
      .layout       = pipline_layouts.depth_prepass_render,
      .primitive    = primitive_state,
      .vertex       = vertex_state,
      .fragment     = NULL,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

  // we need the depthCompare to fit the depth buffer mode we are using.
  // this is the same for other passes
  /* Default */
  depth_stencil_state.depthCompare
    = depth_compare_funcs[(uint32_t)DepthBufferMode_Default];
  render_pipelines.depth_pre_pass[(uint32_t)DepthBufferMode_Default]
    = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &depth_pre_pass_render_pipeline_descriptor_base);
  /* Reversed */
  depth_stencil_state.depthCompare
    = depth_compare_funcs[(uint32_t)DepthBufferMode_Reversed];
  render_pipelines.depth_pre_pass[(uint32_t)DepthBufferMode_Reversed]
    = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &depth_pre_pass_render_pipeline_descriptor_base);

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
}

// precisionPass is to draw precision error as color of depth value stored in
// depth buffer compared to that directly calcualated in the shader
static void prepare_precision_pass_render_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = depth_buffer_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    precision_error_pass, geometry_vertex_size,
    /* Attribute descriptions */
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, geometry_position_offset))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label = "vertex_precision_error_pass_vertex_shader",
                  .file  = "shaders/reversed_z/vertexPrecisionErrorPass.wgsl",
                  .entry = "main",
                },
                .buffer_count = 1,
                .buffers      = &precision_error_pass_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .label = "fragment_precision_error_pass_fragment_shader",
                  .file  = "shaders/reversed_z/fragmentPrecisionErrorPass.wgsl",
                  .entry = "main",
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

  // precisionPass is to draw precision error as color of depth value stored in
  // depth buffer compared to that directly calcualated in the shader
  WGPURenderPipelineDescriptor precision_pass_render_pipeline_descriptor_base
    = (WGPURenderPipelineDescriptor){
      .label        = "precision_error_pass_render_pipeline",
      .layout       = pipline_layouts.precision_pass_render,
      .primitive    = primitive_state,
      .vertex       = vertex_state,
      .fragment     = &fragment_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

  /* Default */
  depth_stencil_state.depthCompare
    = depth_compare_funcs[(uint32_t)DepthBufferMode_Default];
  render_pipelines.precision_pass[(uint32_t)DepthBufferMode_Default]
    = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &precision_pass_render_pipeline_descriptor_base);
  /* Reversed */
  depth_stencil_state.depthCompare
    = depth_compare_funcs[(uint32_t)DepthBufferMode_Reversed];
  render_pipelines.precision_pass[(uint32_t)DepthBufferMode_Reversed]
    = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &precision_pass_render_pipeline_descriptor_base);

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

// colorPass is the regular render pass to render the scene
static void prepare_color_pass_render_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = depth_buffer_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    color_pass, geometry_vertex_size,
    /* Attribute descriptions */
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, geometry_position_offset),
    // Attribute location 1: Color
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4, geometry_color_offset))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label = "color_pass_vertex_shader",
                  .file  = "shaders/reversed_z/vertex.wgsl",
                  .entry = "main",
                },
                .buffer_count = 1,
                .buffers      = &color_pass_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .label = "color_pass_fragment_shader",
                  .file  = "shaders/reversed_z/fragment.wgsl",
                  .entry = "main",
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

  // colorPass is the regular render pass to render the scene
  WGPURenderPipelineDescriptor color_passRender_pipeline_descriptor_base
    = (WGPURenderPipelineDescriptor){
      .label        = "color_pass_render_pipeline",
      .layout       = pipline_layouts.color_pass_render,
      .primitive    = primitive_state,
      .vertex       = vertex_state,
      .fragment     = &fragment_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    };

  /* Default */
  depth_stencil_state.depthCompare
    = depth_compare_funcs[(uint32_t)DepthBufferMode_Default];
  render_pipelines.color_pass[(uint32_t)DepthBufferMode_Default]
    = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &color_passRender_pipeline_descriptor_base);
  /* Reversed */
  depth_stencil_state.depthCompare
    = depth_compare_funcs[(uint32_t)DepthBufferMode_Reversed];
  render_pipelines.color_pass[(uint32_t)DepthBufferMode_Reversed]
    = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &color_passRender_pipeline_descriptor_base);

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

// textureQuadPass is draw a full screen quad of depth texture
// to see the difference of depth value using reversed z compared to default
// depth buffer usage 0.0 will be the furthest and 1.0 will be the closest
static void
prepare_texture_quad_pass_render_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color blend state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader SPIR-V
              .label = "vertex_texture_quad_vertex_shader",
              .file  = "shaders/reversed_z/vertexTextureQuad.wgsl",
              .entry = "main",
            },
           .buffer_count = 0,
          });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader SPIR-V
              .label = "fragment_texture_quad_fragment_shader",
              .file  = "shaders/reversed_z/fragmentTextureQuad.wgsl",
              .entry = "main",
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

  // textureQuadPass is draw a full screen quad of depth texture
  // to see the difference of depth value using reversed z compared to default
  // depth buffer usage 0.0 will be the furthest and 1.0 will be the closest
  render_pipelines.texture_quad_pass = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "texture_quad_pass_render_pipeline",
                            .layout      = pipline_layouts.texture_quad_pass,
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void prepare_depth_textures(wgpu_context_t* wgpu_context)
{
  // Create the depth texture.
  {
    WGPUTextureDescriptor texture_desc = {
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
      .dimension     = WGPUTextureDimension_2D,
      .format        = depth_buffer_format,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .size          = (WGPUExtent3D)  {
        .width               = wgpu_context->surface.width,
        .height              = wgpu_context->surface.height,
        .depthOrArrayLayers  = 1,
      },
    };
    textures.depth.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    textures.depth.view
      = wgpuTextureCreateView(textures.depth.texture, &texture_view_dec);
  }

  // Create the default depth texture.
  {
    WGPUTextureDescriptor texture_desc = {
      .usage         = WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .format        = depth_buffer_format,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .size          = (WGPUExtent3D)  {
        .width               = wgpu_context->surface.width,
        .height              = wgpu_context->surface.height,
        .depthOrArrayLayers  = 1,
      },
    };
    textures.default_depth.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    textures.default_depth.view = wgpuTextureCreateView(
      textures.default_depth.texture, &texture_view_dec);
  }
}

static void prepare_depth_pre_pass_descriptor(void)
{
  dppd_rp_ds_att_descriptor = (WGPURenderPassDepthStencilAttachment){
    .view            = textures.depth.view,
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  };

  depth_pre_pass_descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 0,
    .colorAttachments       = NULL,
    .depthStencilAttachment = &dppd_rp_ds_att_descriptor,
  };
}

// drawPassDescriptor and drawPassLoadDescriptor are used for drawing
// the scene twice using different depth buffer mode on splitted viewport
// of the same canvas
// see the difference of the loadValue of the colorAttachments
static void prepare_draw_pass_descriptors(void)
{
  // drawPassDescriptor
  {
    // Color attachment
    dpd_rp_color_att_descriptors[0][0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, // view is acquired and set in render loop.
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.5f,
        .a = 1.0f,
      },
    };

    dpd_rp_ds_att_descriptors[0] = (WGPURenderPassDepthStencilAttachment){
      .view            = textures.default_depth.view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
    };

    draw_pass_descriptors[0] = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 1,
      .colorAttachments       = dpd_rp_color_att_descriptors[0],
      .depthStencilAttachment = &dpd_rp_ds_att_descriptors[0],
    };
  }

  // drawPassLoadDescriptor
  {
    dpd_rp_color_att_descriptors[1][0] = (WGPURenderPassColorAttachment){
      .view       = NULL, // view is acquired and set in render loop.
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Load,
      .storeOp    = WGPUStoreOp_Store,
    };

    dpd_rp_ds_att_descriptors[1] = (WGPURenderPassDepthStencilAttachment){
      .view            = textures.default_depth.view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
    };

    draw_pass_descriptors[1] = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 1,
      .colorAttachments       = dpd_rp_color_att_descriptors[1],
      .depthStencilAttachment = &dpd_rp_ds_att_descriptors[1],
    };
  }
}

static void prepare_texture_quad_pass_descriptors(void)
{
  // textureQuadPassDescriptor
  {
    tqd_rp_color_att_descriptors[0][0]
     = (WGPURenderPassColorAttachment) {
      .view       = NULL, // view is acquired and set in render loop.
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.5f,
        .a = 1.0f,
      },
    };

    texture_quad_pass_descriptors[0] = (WGPURenderPassDescriptor){
      .colorAttachmentCount = 1,
      .colorAttachments     = tqd_rp_color_att_descriptors[0],
    };
  }

  // textureQuadPassLoadDescriptor
  {
    tqd_rp_color_att_descriptors[1][0] = (WGPURenderPassColorAttachment){
      .view       = NULL, // attachment is acquired and set in render loop.
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Load,
      .storeOp    = WGPUStoreOp_Store,
    };

    texture_quad_pass_descriptors[1] = (WGPURenderPassDescriptor){
      .colorAttachmentCount = 1,
      .colorAttachments     = tqd_rp_color_att_descriptors[1],
    };
  }
}

static void
prepare_depth_texture_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Texture view
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Depth,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  bind_group_layouts.depth_texture
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(bind_group_layouts.depth_texture != NULL)
}

// Model, view, projection matrices
static void prepare_uniform_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Uniform buffer
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = uniform_buffer_size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Uniform buffer
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(mat4), // 4x4 matrix
      },
      .sampler = {0},
    }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  bind_group_layouts.uniform
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(bind_group_layouts.uniform != NULL)
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Depth Pre-pass render pipeline layout
  {
    WGPUBindGroupLayout bind_group_layout_array[1] = {
      bind_group_layouts.uniform,
    };
    pipline_layouts.depth_prepass_render = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layout_array),
        .bindGroupLayouts     = bind_group_layout_array,
      });
    ASSERT(pipline_layouts.depth_prepass_render != NULL)
  }

  // Precision pass render pipeline layout
  {
    WGPUBindGroupLayout bind_group_layout_array[2] = {
      bind_group_layouts.uniform,       // Group 0
      bind_group_layouts.depth_texture, // Group 1
    };
    pipline_layouts.precision_pass_render = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layout_array),
        .bindGroupLayouts     = bind_group_layout_array,
      });
    ASSERT(pipline_layouts.precision_pass_render != NULL)
  }

  // Color pass render pipeline layout
  {
    WGPUBindGroupLayout bind_group_layout_array[1] = {
      bind_group_layouts.uniform,
    };
    pipline_layouts.color_pass_render = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layout_array),
        .bindGroupLayouts     = bind_group_layout_array,
      });
    ASSERT(pipline_layouts.color_pass_render != NULL)
  }

  // Texture quad pass pipline layout
  {
    WGPUBindGroupLayout bind_group_layout_array[1] = {
      bind_group_layouts.depth_texture,
    };
    pipline_layouts.texture_quad_pass = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layout_array),
        .bindGroupLayouts     = bind_group_layout_array,
      });
    ASSERT(pipline_layouts.texture_quad_pass != NULL)
  }
}

static void prepare_depth_texture_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding     = 0,
      .textureView = textures.depth.view,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .layout     = bind_group_layouts.depth_texture,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  bind_groups.depth_texture
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(bind_groups.depth_texture != NULL)
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  uniform_buffers.uniform = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = uniform_buffer_size,
                  });

  uniform_buffers.camera_matrix = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(mat4), // 4x4 matrix
                  });

  uniform_buffers.camera_matrix_reversed_depth = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(mat4), // 4x4 matrix
                  });
}

static void setup_uniform_bind_groups(wgpu_context_t* wgpu_context)
{
  // 1st uniform bind group
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = uniform_buffers.uniform.buffer,
        .size    = uniform_buffer_size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = uniform_buffers.camera_matrix.buffer,
        .size    = sizeof(mat4), // 4x4 matrix
      }
    };
    bind_groups.uniform[0] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layouts.uniform,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
  }

  // 2nd uniform bind group
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = uniform_buffers.uniform.buffer,
        .size    = uniform_buffer_size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = uniform_buffers.camera_matrix_reversed_depth.buffer,
        .size    = sizeof(mat4), // 4x4 matrix
      }
    };
    bind_groups.uniform[1] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layouts.uniform,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
  }
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  uint32_t m = 0;
  float z = 0.0f, s = 0.0f;
  for (uint32_t x = 0; x < x_count; ++x) {
    for (uint32_t y = 0; y < y_count; ++y) {
      z = -800.0f * m;
      s = 1.0f + 50.0f * m;
      glm_mat4_identity(model_matrices[m]);
      glm_translate(model_matrices[m], //
                    (vec3){
                      x - x_count / 2.0f + 0.5f,                       // x
                      (4.0f - 0.2f * z) * (y - y_count / 2.0f + 1.0f), // y
                      z,                                               // z
                    });
      glm_scale(model_matrices[m], (vec3){s, s, s});
      ++m;
    }
  }

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_translate(view_matrix, (vec3){0.0f, 0.0f, -12.0f});

  const float aspect = (0.5f * (float)wgpu_context->surface.width)
                       / (float)wgpu_context->surface.height;
  mat4 projection_matrix = GLM_MAT4_IDENTITY_INIT;
  float far              = INFINITY;
  perspective_zo(&projection_matrix, PI2 / 5.0f, aspect, 5.0f, &far);

  mat4 view_projection_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_mul(projection_matrix, view_matrix, view_projection_matrix);
  mat4 reversed_range_view_projection_matrix = GLM_MAT4_IDENTITY_INIT;
  // to use 1/z we just multiple depthRangeRemapMatrix to our default camera
  // view projection matrix
  glm_mat4_mul(depth_range_remap_matrix, view_projection_matrix,
               reversed_range_view_projection_matrix);

  wgpu_queue_write_buffer(wgpu_context, uniform_buffers.camera_matrix.buffer, 0,
                          view_projection_matrix, sizeof(mat4));
  wgpu_queue_write_buffer(
    wgpu_context, uniform_buffers.camera_matrix_reversed_depth.buffer, 0,
    reversed_range_view_projection_matrix, sizeof(mat4));
}

static void update_transformation_matrix(wgpu_example_context_t* context)
{
  const float now     = context->frame.timestamp_millis / 1000.0f;
  const float sin_now = sin(now), cos_now = cos(now);

  for (uint32_t i = 0, m = 0; i < num_instances; ++i, m += matrix_float_count) {
    glm_mat4_copy(model_matrices[i], tmp_mat4);
    glm_rotate(tmp_mat4, (PI / 180.f) * 30.0f, (vec3){sin_now, cos_now, 0.0f});
    memcpy(&mvp_matrices_data[m], tmp_mat4, sizeof(mat4));
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  update_transformation_matrix(context);

  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.uniform.buffer,
                          0, &mvp_matrices_data, sizeof(mvp_matrices_data));
}

static int example_initialize(wgpu_example_context_t* context)
{
  UNUSED_VAR(depth_buffer_modes);

  if (context) {
    prepare_vertex_buffer(context->wgpu_context);
    prepare_depth_texture_bind_group_layout(context->wgpu_context);
    prepare_uniform_bind_group_layout(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_depth_pre_pass_render_pipeline(context->wgpu_context);
    prepare_precision_pass_render_pipeline(context->wgpu_context);
    prepare_color_pass_render_pipeline(context->wgpu_context);
    prepare_texture_quad_pass_render_pipeline(context->wgpu_context);
    prepare_depth_textures(context->wgpu_context);
    prepare_depth_pre_pass_descriptor();
    prepare_draw_pass_descriptors();
    prepare_texture_quad_pass_descriptors();
    prepare_depth_texture_bind_group(context->wgpu_context);
    prepare_uniform_buffers(context->wgpu_context);
    setup_uniform_bind_groups(context->wgpu_context);
    init_uniform_buffers(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    static const char* mode[3] = {"color", "precision-error", "depth-texture"};
    int32_t item_index         = (int32_t)current_render_mode;
    if (imgui_overlay_combo_box(context->imgui_overlay, "Mode", &item_index,
                                mode, 3)) {
      current_render_mode = (render_mode_enum)item_index;
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  const WGPUTextureView attachment = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  if (current_render_mode == RenderMode_Color) {
    for (uint32_t m = 0; m < (uint32_t)ARRAY_SIZE(depth_buffer_modes); ++m) {
      dpd_rp_color_att_descriptors[m][0].view      = attachment;
      dpd_rp_ds_att_descriptors[m].depthClearValue = depth_clear_values[m];
      WGPURenderPassEncoder color_pass = wgpuCommandEncoderBeginRenderPass(
        wgpu_context->cmd_enc, &draw_pass_descriptors[m]);
      wgpuRenderPassEncoderSetPipeline(color_pass,
                                       render_pipelines.color_pass[m]);
      wgpuRenderPassEncoderSetBindGroup(color_pass, 0, bind_groups.uniform[m],
                                        0, 0);
      wgpuRenderPassEncoderSetVertexBuffer(color_pass, 0, vertices.buffer, 0,
                                           WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetViewport(color_pass, (viewport_width * m) / 2.0f,
                                       0.0f, viewport_width / 2.0f,
                                       default_canvas_height, 0.0f, 1.0f);
      wgpuRenderPassEncoderDraw(color_pass, geometry_draw_count, num_instances,
                                0, 0);
      wgpuRenderPassEncoderEnd(color_pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, color_pass)
    }
  }
  else if (current_render_mode == RenderMode_Precision_Error) {
    for (uint32_t m = 0; m < (uint32_t)ARRAY_SIZE(depth_buffer_modes); ++m) {
      // depthPrePass
      {
        dppd_rp_ds_att_descriptor.depthClearValue = depth_clear_values[m];
        WGPURenderPassEncoder depth_pre_pass
          = wgpuCommandEncoderBeginRenderPass(wgpu_context->cmd_enc,
                                              &depth_pre_pass_descriptor);
        wgpuRenderPassEncoderSetPipeline(depth_pre_pass,
                                         render_pipelines.depth_pre_pass[m]);
        wgpuRenderPassEncoderSetBindGroup(depth_pre_pass, 0,
                                          bind_groups.uniform[m], 0, 0);
        wgpuRenderPassEncoderSetVertexBuffer(depth_pre_pass, 0, vertices.buffer,
                                             0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetViewport(
          depth_pre_pass, (viewport_width * m) / 2.0f, 0.0f,
          viewport_width / 2.0f, default_canvas_height, 0.0f, 1.0f);
        wgpuRenderPassEncoderDraw(depth_pre_pass, geometry_draw_count,
                                  num_instances, 0, 0);
        wgpuRenderPassEncoderEnd(depth_pre_pass);
        WGPU_RELEASE_RESOURCE(RenderPassEncoder, depth_pre_pass)
      }
      // precisionErrorPass
      {
        dpd_rp_color_att_descriptors[m][0].view      = attachment;
        dpd_rp_ds_att_descriptors[m].depthClearValue = depth_clear_values[m];
        WGPURenderPassEncoder precision_error_pass
          = wgpuCommandEncoderBeginRenderPass(wgpu_context->cmd_enc,
                                              &draw_pass_descriptors[m]);
        wgpuRenderPassEncoderSetPipeline(precision_error_pass,
                                         render_pipelines.precision_pass[m]);
        wgpuRenderPassEncoderSetBindGroup(precision_error_pass, 0,
                                          bind_groups.uniform[m], 0, 0);
        wgpuRenderPassEncoderSetBindGroup(precision_error_pass, 1,
                                          bind_groups.depth_texture, 0, 0);
        wgpuRenderPassEncoderSetVertexBuffer(
          precision_error_pass, 0, vertices.buffer, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetViewport(
          precision_error_pass, (viewport_width * m) / 2.0f, 0.0f,
          viewport_width / 2.0f, default_canvas_height, 0.0f, 1.0f);
        wgpuRenderPassEncoderDraw(precision_error_pass, geometry_draw_count,
                                  num_instances, 0, 0);
        wgpuRenderPassEncoderEnd(precision_error_pass);
        WGPU_RELEASE_RESOURCE(RenderPassEncoder, precision_error_pass)
      }
    }
  }
  else {
    // depth texture quad
    for (uint32_t m = 0; m < (uint32_t)ARRAY_SIZE(depth_buffer_modes); ++m) {
      // depthPrePass
      {
        dppd_rp_ds_att_descriptor.depthClearValue = depth_clear_values[m];
        WGPURenderPassEncoder depth_pre_pass
          = wgpuCommandEncoderBeginRenderPass(wgpu_context->cmd_enc,
                                              &depth_pre_pass_descriptor);
        wgpuRenderPassEncoderSetPipeline(depth_pre_pass,
                                         render_pipelines.depth_pre_pass[m]);
        wgpuRenderPassEncoderSetBindGroup(depth_pre_pass, 0,
                                          bind_groups.uniform[m], 0, 0);
        wgpuRenderPassEncoderSetVertexBuffer(depth_pre_pass, 0, vertices.buffer,
                                             0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetViewport(
          depth_pre_pass, (viewport_width * m) / 2.0f, 0.0f,
          viewport_width / 2.0f, default_canvas_height, 0.0f, 1.0f);
        wgpuRenderPassEncoderDraw(depth_pre_pass, geometry_draw_count,
                                  num_instances, 0, 0);
        wgpuRenderPassEncoderEnd(depth_pre_pass);
        WGPU_RELEASE_RESOURCE(RenderPassEncoder, depth_pre_pass)
      }
      // depthTextureQuadPass
      {
        tqd_rp_color_att_descriptors[m][0].view = attachment;
        WGPURenderPassEncoder depth_texture_quad_pass
          = wgpuCommandEncoderBeginRenderPass(
            wgpu_context->cmd_enc, &texture_quad_pass_descriptors[m]);
        wgpuRenderPassEncoderSetPipeline(depth_texture_quad_pass,
                                         render_pipelines.texture_quad_pass);
        wgpuRenderPassEncoderSetBindGroup(depth_texture_quad_pass, 0,
                                          bind_groups.depth_texture, 0, 0);
        wgpuRenderPassEncoderSetViewport(
          depth_texture_quad_pass, (viewport_width * m) / 2.0f, 0.0f,
          viewport_width / 2.0f, default_canvas_height, 0.0f, 1.0f);
        wgpuRenderPassEncoderDraw(depth_texture_quad_pass, 6, 1, 0, 0);
        wgpuRenderPassEncoderEnd(depth_texture_quad_pass);
        WGPU_RELEASE_RESOURCE(RenderPassEncoder, depth_texture_quad_pass)
      }
    }
  }

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL)
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

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipline_layouts.depth_prepass_render)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipline_layouts.precision_pass_render)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipline_layouts.color_pass_render)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipline_layouts.texture_quad_pass)

  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.uniform.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.camera_matrix.buffer)
  WGPU_RELEASE_RESOURCE(Buffer,
                        uniform_buffers.camera_matrix_reversed_depth.buffer)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.depth_texture)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.uniform)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.depth_texture)
  for (uint32_t i = 0; i < 2; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.uniform[i])
  }

  wgpu_destroy_texture(&textures.depth);
  wgpu_destroy_texture(&textures.default_depth);

  for (uint32_t i = 0; i < 2; ++i) {
    WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.depth_pre_pass[i])
    WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.precision_pass[i])
    WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.color_pass[i])
  }
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.texture_quad_pass)
}

void example_reversed_z(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title   = example_title,
     .overlay = true,
    },
    .example_window_config = (window_config_t){
      .width  = default_canvas_width,
      .height = default_canvas_height,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy
  });
  // clang-format on
}
