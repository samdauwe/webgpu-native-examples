#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

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
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/reversedZ
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;
static const char* vertex_depth_pre_pass_shader_wgsl;
static const char* vertex_texture_quad_shader_wgsl;
static const char* fragment_texture_quad_shader_wgsl;
static const char* vertex_precision_error_pass_shader_wgsl;
static const char* fragment_precision_error_pass_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Reversed Z example
 * -------------------------------------------------------------------------- */

/* Depth buffer modes */
typedef enum depth_buffer_mode_t {
  DEPTH_BUFFER_MODE_DEFAULT  = 0,
  DEPTH_BUFFER_MODE_REVERSED = 1,
  DEPTH_BUFFER_MODE_COUNT    = 2,
} depth_buffer_mode_t;

/* Render modes */
typedef enum render_mode_t {
  RENDER_MODE_COLOR           = 0,
  RENDER_MODE_PRECISION_ERROR = 1,
  RENDER_MODE_DEPTH_TEXTURE   = 2,
  RENDER_MODE_COUNT           = 3,
} render_mode_t;

/* Geometry constants */
#define GEOMETRY_VERTEX_SIZE (4 * 8) /* Byte size of one geometry vertex */
#define GEOMETRY_POSITION_OFFSET (0)
#define GEOMETRY_COLOR_OFFSET (4 * 4) /* Byte offset of vertex color attr */
#define GEOMETRY_DRAW_COUNT (6 * 2)

/* Instance constants */
#define X_COUNT (1)
#define Y_COUNT (5)
#define NUM_INSTANCES (X_COUNT * Y_COUNT)
#define MATRIX_FLOAT_COUNT (16) /* 4x4 matrix */
#define MATRIX_STRIDE (4 * MATRIX_FLOAT_COUNT)

/* Two planes close to each other for depth precision test */
static const float d = 0.0001f; /* half distance between two planes */
static const float o = 0.5f;    /* half x offset to shift planes */

/* State struct */
static struct {
  /* Geometry */
  struct {
    float vertex_array[GEOMETRY_DRAW_COUNT * 8];
    wgpu_buffer_t vertices_buffer;
  } geometry;
  /* Depth textures */
  struct {
    WGPUTexture depth_texture;
    WGPUTextureView depth_texture_view;
    WGPUTexture default_depth_texture;
    WGPUTextureView default_depth_texture_view;
  } textures;
  /* Uniform buffers */
  struct {
    WGPUBuffer uniform_buffer;
    WGPUBuffer camera_matrix_buffer;
    WGPUBuffer camera_matrix_reversed_depth_buffer;
  } uniforms;
  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout depth_texture_bgl;
    WGPUBindGroupLayout uniform_bgl;
  } bind_group_layouts;
  /* Bind groups */
  struct {
    WGPUBindGroup uniform_bind_groups[DEPTH_BUFFER_MODE_COUNT];
    WGPUBindGroup depth_texture_bind_group;
  } bind_groups;
  /* Pipelines */
  struct {
    WGPURenderPipeline depth_pre_pass[DEPTH_BUFFER_MODE_COUNT];
    WGPURenderPipeline precision_pass[DEPTH_BUFFER_MODE_COUNT];
    WGPURenderPipeline color_pass[DEPTH_BUFFER_MODE_COUNT];
    WGPURenderPipeline texture_quad_pass;
  } pipelines;
  /* Pipeline layouts */
  struct {
    WGPUPipelineLayout depth_pre_pass;
    WGPUPipelineLayout precision_pass;
    WGPUPipelineLayout color_pass;
    WGPUPipelineLayout texture_quad_pass;
  } pipeline_layouts;
  /* Model matrices */
  struct {
    mat4 model_matrices[NUM_INSTANCES];
    float mvp_matrices_data[MATRIX_FLOAT_COUNT * NUM_INSTANCES];
  } transforms;
  /* View/projection matrices */
  struct {
    mat4 view_matrix;
    mat4 projection_matrix;
    mat4 view_projection_matrix;
    mat4 reversed_range_view_projection_matrix;
    mat4 depth_range_remap_matrix;
  } camera;
  /* Render pass descriptors */
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassColorAttachment color_attachment_load;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDepthStencilAttachment depth_pre_pass_depth_attachment;
    WGPURenderPassDescriptor depth_pre_pass_desc;
    WGPURenderPassDescriptor draw_pass_desc[DEPTH_BUFFER_MODE_COUNT];
    WGPURenderPassDescriptor texture_quad_pass_desc[DEPTH_BUFFER_MODE_COUNT];
  } render_pass;
  /* GUI settings */
  struct {
    render_mode_t mode;
  } settings;
  const char* render_modes_str[RENDER_MODE_COUNT];
  uint64_t last_frame_time;
  WGPUBool initialized;
} state = {
  .render_pass.color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.5, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass.color_attachment_load = {
    .loadOp     = WGPULoadOp_Load,
    .storeOp    = WGPUStoreOp_Store,
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass.depth_stencil_attachment = {
    .depthLoadOp   = WGPULoadOp_Clear,
    .depthStoreOp  = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  },
  .render_pass.depth_pre_pass_depth_attachment = {
    .depthLoadOp   = WGPULoadOp_Clear,
    .depthStoreOp  = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  },
  .settings = {
    .mode = RENDER_MODE_COLOR,
  },
  .render_modes_str = {
    "Color",
    "Precision Error",
    "Depth Texture",
  },
};

/* Initialize geometry vertex array */
static void init_geometry(void)
{
  /* float4 position, float4 color */
  /* clang-format off */
  float vertices[] = {
    /* Red plane (front) */
    -1.0f - o, -1.0f, d, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
     1.0f - o, -1.0f, d, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    -1.0f - o,  1.0f, d, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
     1.0f - o, -1.0f, d, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
     1.0f - o,  1.0f, d, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    -1.0f - o,  1.0f, d, 1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    /* Green plane (back) */
    -1.0f + o, -1.0f, -d, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,
     1.0f + o, -1.0f, -d, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,
    -1.0f + o,  1.0f, -d, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,
     1.0f + o, -1.0f, -d, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,
     1.0f + o,  1.0f, -d, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,
    -1.0f + o,  1.0f, -d, 1.0f,  0.0f, 1.0f, 0.0f, 1.0f,
  };
  /* clang-format on */
  memcpy(state.geometry.vertex_array, vertices, sizeof(vertices));
}

/* Create vertex buffer from geometry data */
static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  state.geometry.vertices_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label        = "Geometry - Vertices buffer",
                    .usage        = WGPUBufferUsage_Vertex,
                    .size         = sizeof(state.geometry.vertex_array),
                    .initial.data = state.geometry.vertex_array,
                  });
}

/* Initialize depth textures */
static void init_depth_textures(wgpu_context_t* wgpu_context)
{
  /* Cleanup existing textures */
  if (state.textures.depth_texture) {
    wgpuTextureViewRelease(state.textures.depth_texture_view);
    wgpuTextureDestroy(state.textures.depth_texture);
    state.textures.depth_texture      = NULL;
    state.textures.depth_texture_view = NULL;
  }
  if (state.textures.default_depth_texture) {
    wgpuTextureViewRelease(state.textures.default_depth_texture_view);
    wgpuTextureDestroy(state.textures.default_depth_texture);
    state.textures.default_depth_texture      = NULL;
    state.textures.default_depth_texture_view = NULL;
  }

  const uint32_t width  = (uint32_t)wgpu_context->width;
  const uint32_t height = (uint32_t)wgpu_context->height;

  /* Create depth texture with TEXTURE_BINDING usage */
  WGPUTextureDescriptor depth_tex_desc = {
    .label = STRVIEW("Depth texture"),
    .size  = (WGPUExtent3D){
       .width              = width,
       .height             = height,
       .depthOrArrayLayers = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth32Float,
    .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  state.textures.depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_tex_desc);
  ASSERT(state.textures.depth_texture != NULL);

  WGPUTextureViewDescriptor depth_view_desc = {
    .label           = STRVIEW("Depth texture view"),
    .format          = WGPUTextureFormat_Depth32Float,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  state.textures.depth_texture_view
    = wgpuTextureCreateView(state.textures.depth_texture, &depth_view_desc);
  ASSERT(state.textures.depth_texture_view != NULL);

  /* Create default depth texture (render attachment only) */
  WGPUTextureDescriptor default_depth_tex_desc = {
    .label = STRVIEW("Default depth texture"),
    .size  = (WGPUExtent3D){
       .width              = width,
       .height             = height,
       .depthOrArrayLayers = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth32Float,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.textures.default_depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &default_depth_tex_desc);
  ASSERT(state.textures.default_depth_texture != NULL);

  WGPUTextureViewDescriptor default_depth_view_desc = {
    .label           = STRVIEW("Default depth texture view"),
    .format          = WGPUTextureFormat_Depth32Float,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  state.textures.default_depth_texture_view = wgpuTextureCreateView(
    state.textures.default_depth_texture, &default_depth_view_desc);
  ASSERT(state.textures.default_depth_texture_view != NULL);
}

/* Initialize bind group layouts */
static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Depth texture bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout){
          .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Depth texture - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    state.bind_group_layouts.depth_texture_bgl
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(state.bind_group_layouts.depth_texture_bgl != NULL);
  }

  /* Uniform bind group layout (model matrices + camera matrix) */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(state.transforms.mvp_matrices_data),
        },
      },
      [1] = (WGPUBindGroupLayoutEntry){
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(mat4),
        },
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Uniform - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    state.bind_group_layouts.uniform_bgl
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(state.bind_group_layouts.uniform_bgl != NULL);
  }
}

/* Initialize uniform buffers */
static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Model matrices uniform buffer */
  state.uniforms.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Model matrices - Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.transforms.mvp_matrices_data),
    });
  ASSERT(state.uniforms.uniform_buffer != NULL);

  /* Camera matrix uniform buffer */
  state.uniforms.camera_matrix_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Camera matrix - Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4),
    });
  ASSERT(state.uniforms.camera_matrix_buffer != NULL);

  /* Camera matrix reversed depth uniform buffer */
  state.uniforms.camera_matrix_reversed_depth_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Camera matrix reversed - Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4),
    });
  ASSERT(state.uniforms.camera_matrix_reversed_depth_buffer != NULL);
}

/* Initialize model matrices */
static void init_model_matrices(wgpu_context_t* wgpu_context)
{
  int32_t m = 0;
  for (int32_t x = 0; x < X_COUNT; ++x) {
    for (int32_t y = 0; y < Y_COUNT; ++y) {
      const float z = -800.0f * (float)m;
      const float s = 1.0f + 50.0f * (float)m;

      const float tx = (float)x - (float)X_COUNT / 2.0f + 0.5f;
      const float ty
        = (4.0f - 0.2f * z) * ((float)y - (float)Y_COUNT / 2.0f + 1.0f);
      const float tz = z;

      glm_mat4_identity(state.transforms.model_matrices[m]);
      glm_translate(state.transforms.model_matrices[m], (vec3){tx, ty, tz});
      glm_scale(state.transforms.model_matrices[m], (vec3){s, s, s});

      m++;
    }
  }

  /* Setup view matrix */
  glm_mat4_identity(state.camera.view_matrix);
  glm_translate(state.camera.view_matrix, (vec3){0.0f, 0.0f, -12.0f});

  /* Setup projection matrix */
  const float aspect
    = (0.5f * (float)wgpu_context->width) / (float)wgpu_context->height;
  glm_perspective(PI2 / 5.0f, aspect, 5.0f, 9999.0f,
                  state.camera.projection_matrix);

  /* Calculate view-projection matrix */
  glm_mat4_mul(state.camera.projection_matrix, state.camera.view_matrix,
               state.camera.view_projection_matrix);

  /* Setup depth range remap matrix for reversed-z:
   * [ 1  0  0  0 ]
   * [ 0  1  0  0 ]
   * [ 0  0 -1  1 ]
   * [ 0  0  0  1 ]
   */
  glm_mat4_identity(state.camera.depth_range_remap_matrix);
  state.camera.depth_range_remap_matrix[2][2] = -1.0f;
  state.camera.depth_range_remap_matrix[3][2] = 1.0f;

  /* Calculate reversed range view-projection matrix */
  glm_mat4_mul(state.camera.depth_range_remap_matrix,
               state.camera.view_projection_matrix,
               state.camera.reversed_range_view_projection_matrix);

  /* Write camera matrices to GPU */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniforms.camera_matrix_buffer,
                       0, state.camera.view_projection_matrix, sizeof(mat4));
  wgpuQueueWriteBuffer(
    wgpu_context->queue, state.uniforms.camera_matrix_reversed_depth_buffer, 0,
    state.camera.reversed_range_view_projection_matrix, sizeof(mat4));
}

/* Initialize bind groups */
static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Uniform bind group - Default */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniforms.uniform_buffer,
        .size    = sizeof(state.transforms.mvp_matrices_data),
      },
      [1] = (WGPUBindGroupEntry){
        .binding = 1,
        .buffer  = state.uniforms.camera_matrix_buffer,
        .size    = sizeof(mat4),
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Uniform - Default bind group"),
      .layout     = state.bind_group_layouts.uniform_bgl,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.bind_groups.uniform_bind_groups[DEPTH_BUFFER_MODE_DEFAULT]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  }

  /* Uniform bind group - Reversed */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniforms.uniform_buffer,
        .size    = sizeof(state.transforms.mvp_matrices_data),
      },
      [1] = (WGPUBindGroupEntry){
        .binding = 1,
        .buffer  = state.uniforms.camera_matrix_reversed_depth_buffer,
        .size    = sizeof(mat4),
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Uniform - Reversed bind group"),
      .layout     = state.bind_group_layouts.uniform_bgl,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.bind_groups.uniform_bind_groups[DEPTH_BUFFER_MODE_REVERSED]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  }

  /* Depth texture bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding     = 0,
        .textureView = state.textures.depth_texture_view,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Depth texture - Bind group"),
      .layout     = state.bind_group_layouts.depth_texture_bgl,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.bind_groups.depth_texture_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  }
}

/* Initialize pipeline layouts */
static void init_pipeline_layouts(wgpu_context_t* wgpu_context)
{
  /* Depth pre-pass pipeline layout */
  {
    WGPUBindGroupLayout bg_layouts[1] = {state.bind_group_layouts.uniform_bgl};
    WGPUPipelineLayoutDescriptor pl_desc = {
      .label                = STRVIEW("Depth pre-pass - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bg_layouts),
      .bindGroupLayouts     = bg_layouts,
    };
    state.pipeline_layouts.depth_pre_pass
      = wgpuDeviceCreatePipelineLayout(wgpu_context->device, &pl_desc);
    ASSERT(state.pipeline_layouts.depth_pre_pass != NULL);
  }

  /* Precision pass pipeline layout */
  {
    WGPUBindGroupLayout bg_layouts[2] = {
      state.bind_group_layouts.uniform_bgl,
      state.bind_group_layouts.depth_texture_bgl,
    };
    WGPUPipelineLayoutDescriptor pl_desc = {
      .label                = STRVIEW("Precision pass - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bg_layouts),
      .bindGroupLayouts     = bg_layouts,
    };
    state.pipeline_layouts.precision_pass
      = wgpuDeviceCreatePipelineLayout(wgpu_context->device, &pl_desc);
    ASSERT(state.pipeline_layouts.precision_pass != NULL);
  }

  /* Color pass pipeline layout */
  {
    WGPUBindGroupLayout bg_layouts[1] = {state.bind_group_layouts.uniform_bgl};
    WGPUPipelineLayoutDescriptor pl_desc = {
      .label                = STRVIEW("Color pass - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bg_layouts),
      .bindGroupLayouts     = bg_layouts,
    };
    state.pipeline_layouts.color_pass
      = wgpuDeviceCreatePipelineLayout(wgpu_context->device, &pl_desc);
    ASSERT(state.pipeline_layouts.color_pass != NULL);
  }

  /* Texture quad pass pipeline layout */
  {
    WGPUBindGroupLayout bg_layouts[1]
      = {state.bind_group_layouts.depth_texture_bgl};
    WGPUPipelineLayoutDescriptor pl_desc = {
      .label                = STRVIEW("Texture quad pass - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bg_layouts),
      .bindGroupLayouts     = bg_layouts,
    };
    state.pipeline_layouts.texture_quad_pass
      = wgpuDeviceCreatePipelineLayout(wgpu_context->device, &pl_desc);
    ASSERT(state.pipeline_layouts.texture_quad_pass != NULL);
  }
}

/* Create render pipelines */
static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Vertex buffer layout for geometry (position + color) */
  WGPU_VERTEX_BUFFER_LAYOUT(
    geometry, GEOMETRY_VERTEX_SIZE,
    /* Attribute 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, GEOMETRY_POSITION_OFFSET),
    /* Attribute 1: Color */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4, GEOMETRY_COLOR_OFFSET))

  /* Vertex buffer layout for depth pre-pass (position only) */
  WGPU_VERTEX_BUFFER_LAYOUT(
    geometry_pos_only, GEOMETRY_VERTEX_SIZE,
    /* Attribute 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, GEOMETRY_POSITION_OFFSET))

  WGPUCompareFunction depth_compare_funcs[DEPTH_BUFFER_MODE_COUNT] = {
    [DEPTH_BUFFER_MODE_DEFAULT]  = WGPUCompareFunction_Less,
    [DEPTH_BUFFER_MODE_REVERSED] = WGPUCompareFunction_Greater,
  };

  /* Create depth pre-pass pipelines */
  {
    WGPUShaderModule shader_module = wgpu_create_shader_module(
      wgpu_context->device, vertex_depth_pre_pass_shader_wgsl);

    for (uint32_t m = 0; m < DEPTH_BUFFER_MODE_COUNT; ++m) {
      WGPUDepthStencilState depth_stencil_state = {
        .format            = WGPUTextureFormat_Depth32Float,
        .depthWriteEnabled = true,
        .depthCompare      = depth_compare_funcs[m],
      };

      WGPURenderPipelineDescriptor rp_desc = {
        .label  = STRVIEW("Depth pre-pass - Render pipeline"),
        .layout = state.pipeline_layouts.depth_pre_pass,
        .vertex = {
          .module      = shader_module,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = 1,
          .buffers     = &geometry_pos_only_vertex_buffer_layout,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Back,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_stencil_state,
        .multisample  = {
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
      };
      state.pipelines.depth_pre_pass[m]
        = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
      ASSERT(state.pipelines.depth_pre_pass[m] != NULL);
    }
    wgpuShaderModuleRelease(shader_module);
  }

  /* Create precision pass pipelines */
  {
    WGPUShaderModule vert_module = wgpu_create_shader_module(
      wgpu_context->device, vertex_precision_error_pass_shader_wgsl);
    WGPUShaderModule frag_module = wgpu_create_shader_module(
      wgpu_context->device, fragment_precision_error_pass_shader_wgsl);

    for (uint32_t m = 0; m < DEPTH_BUFFER_MODE_COUNT; ++m) {
      WGPUBlendState blend_state = wgpu_create_blend_state(true);

      WGPUDepthStencilState depth_stencil_state = {
        .format            = WGPUTextureFormat_Depth32Float,
        .depthWriteEnabled = true,
        .depthCompare      = depth_compare_funcs[m],
      };

      WGPURenderPipelineDescriptor rp_desc = {
        .label  = STRVIEW("Precision pass - Render pipeline"),
        .layout = state.pipeline_layouts.precision_pass,
        .vertex = {
          .module      = vert_module,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = 1,
          .buffers     = &geometry_pos_only_vertex_buffer_layout,
        },
        .fragment = &(WGPUFragmentState){
          .module      = frag_module,
          .entryPoint  = STRVIEW("main"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = wgpu_context->render_format,
            .blend     = &blend_state,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Back,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_stencil_state,
        .multisample  = {
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
      };
      state.pipelines.precision_pass[m]
        = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
      ASSERT(state.pipelines.precision_pass[m] != NULL);
    }
    wgpuShaderModuleRelease(vert_module);
    wgpuShaderModuleRelease(frag_module);
  }

  /* Create color pass pipelines */
  {
    WGPUShaderModule vert_module
      = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
    WGPUShaderModule frag_module
      = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

    for (uint32_t m = 0; m < DEPTH_BUFFER_MODE_COUNT; ++m) {
      WGPUBlendState blend_state = wgpu_create_blend_state(true);

      WGPUDepthStencilState depth_stencil_state = {
        .format            = WGPUTextureFormat_Depth32Float,
        .depthWriteEnabled = true,
        .depthCompare      = depth_compare_funcs[m],
      };

      WGPURenderPipelineDescriptor rp_desc = {
        .label  = STRVIEW("Color pass - Render pipeline"),
        .layout = state.pipeline_layouts.color_pass,
        .vertex = {
          .module      = vert_module,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = 1,
          .buffers     = &geometry_vertex_buffer_layout,
        },
        .fragment = &(WGPUFragmentState){
          .module      = frag_module,
          .entryPoint  = STRVIEW("main"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = wgpu_context->render_format,
            .blend     = &blend_state,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Back,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_stencil_state,
        .multisample  = {
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
      };
      state.pipelines.color_pass[m]
        = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
      ASSERT(state.pipelines.color_pass[m] != NULL);
    }
    wgpuShaderModuleRelease(vert_module);
    wgpuShaderModuleRelease(frag_module);
  }

  /* Create texture quad pass pipeline */
  {
    WGPUShaderModule vert_module = wgpu_create_shader_module(
      wgpu_context->device, vertex_texture_quad_shader_wgsl);
    WGPUShaderModule frag_module = wgpu_create_shader_module(
      wgpu_context->device, fragment_texture_quad_shader_wgsl);

    WGPUBlendState blend_state = wgpu_create_blend_state(true);

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Texture quad pass - Render pipeline"),
      .layout = state.pipeline_layouts.texture_quad_pass,
      .vertex = {
        .module     = vert_module,
        .entryPoint = STRVIEW("main"),
      },
      .fragment = &(WGPUFragmentState){
        .module      = frag_module,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    };
    state.pipelines.texture_quad_pass
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.pipelines.texture_quad_pass != NULL);

    wgpuShaderModuleRelease(vert_module);
    wgpuShaderModuleRelease(frag_module);
  }
}

/* Initialize render pass descriptors */
static void init_render_pass_descriptors(void)
{
  /* Depth pre-pass descriptor */
  state.render_pass.depth_pre_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount = 0,
    .colorAttachments     = NULL,
    .depthStencilAttachment
    = &state.render_pass.depth_pre_pass_depth_attachment,
  };

  /* Draw pass descriptors */
  state.render_pass.draw_pass_desc[DEPTH_BUFFER_MODE_DEFAULT]
    = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.render_pass.color_attachment,
      .depthStencilAttachment = &state.render_pass.depth_stencil_attachment,
    };

  state.render_pass.draw_pass_desc[DEPTH_BUFFER_MODE_REVERSED]
    = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.render_pass.color_attachment_load,
      .depthStencilAttachment = &state.render_pass.depth_stencil_attachment,
    };

  /* Texture quad pass descriptors */
  state.render_pass.texture_quad_pass_desc[DEPTH_BUFFER_MODE_DEFAULT]
    = (WGPURenderPassDescriptor){
      .colorAttachmentCount = 1,
      .colorAttachments     = &state.render_pass.color_attachment,
    };

  state.render_pass.texture_quad_pass_desc[DEPTH_BUFFER_MODE_REVERSED]
    = (WGPURenderPassDescriptor){
      .colorAttachmentCount = 1,
      .colorAttachments     = &state.render_pass.color_attachment_load,
    };
}

/* Update transformation matrices */
static void update_transformation_matrix(void)
{
  const float now     = (float)stm_sec(stm_now());
  const float sin_now = sinf(now);
  const float cos_now = cosf(now);

  mat4 tmp_mat;
  for (uint32_t i = 0, m = 0; i < NUM_INSTANCES; ++i, m += MATRIX_FLOAT_COUNT) {
    glm_mat4_copy(state.transforms.model_matrices[i], tmp_mat);
    glm_rotate(tmp_mat, TO_RADIANS(30.0f), (vec3){sin_now, cos_now, 0.0f});
    memcpy(&state.transforms.mvp_matrices_data[m], tmp_mat, sizeof(mat4));
  }
}

/* Update uniform buffers */
static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  update_transformation_matrix();
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniforms.uniform_buffer, 0,
                       state.transforms.mvp_matrices_data,
                       sizeof(state.transforms.mvp_matrices_data));
}

/* Init function */
static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_geometry();
    init_vertex_buffer(wgpu_context);
    init_depth_textures(wgpu_context);
    init_bind_group_layouts(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_model_matrices(wgpu_context);
    init_bind_groups(wgpu_context);
    init_pipeline_layouts(wgpu_context);
    init_pipelines(wgpu_context);
    init_render_pass_descriptors();
    imgui_overlay_init(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

/* Render GUI */
static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Reversed Z Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Render mode selector */
  int mode = (int)state.settings.mode;
  if (imgui_overlay_combo_box("Mode", &mode, state.render_modes_str,
                              RENDER_MODE_COUNT)) {
    state.settings.mode = (render_mode_t)mode;
  }

  /* Info text */
  igSeparator();
  igTextWrapped("Left: Default depth buffer");
  igTextWrapped("Right: Reversed Z depth buffer");

  igEnd();
}

/* Input event callback */
static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    /* Recreate depth textures on resize */
    init_depth_textures(wgpu_context);

    /* Recreate depth texture bind group */
    WGPU_RELEASE_RESOURCE(BindGroup,
                          state.bind_groups.depth_texture_bind_group);
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding     = 0,
        .textureView = state.textures.depth_texture_view,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Depth texture - Bind group"),
      .layout     = state.bind_group_layouts.depth_texture_bgl,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.bind_groups.depth_texture_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  }
}

/* Frame function */
static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update uniforms */
  update_uniform_buffers(wgpu_context);

  /* Calculate delta time for ImGui */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUTextureView attachment = wgpu_context->swapchain_view;
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  const uint32_t width  = (uint32_t)wgpu_context->width;
  const uint32_t height = (uint32_t)wgpu_context->height;

  float depth_clear_values[DEPTH_BUFFER_MODE_COUNT] = {
    [DEPTH_BUFFER_MODE_DEFAULT]  = 1.0f,
    [DEPTH_BUFFER_MODE_REVERSED] = 0.0f,
  };

  if (state.settings.mode == RENDER_MODE_COLOR) {
    /* Color mode: render scene directly */
    for (uint32_t m = 0; m < DEPTH_BUFFER_MODE_COUNT; ++m) {
      state.render_pass.color_attachment.view      = attachment;
      state.render_pass.color_attachment_load.view = attachment;
      state.render_pass.depth_stencil_attachment.view
        = state.textures.default_depth_texture_view;
      state.render_pass.depth_stencil_attachment.depthClearValue
        = depth_clear_values[m];

      WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
        cmd_enc, &state.render_pass.draw_pass_desc[m]);

      wgpuRenderPassEncoderSetPipeline(rpass_enc,
                                       state.pipelines.color_pass[m]);
      wgpuRenderPassEncoderSetBindGroup(
        rpass_enc, 0, state.bind_groups.uniform_bind_groups[m], 0, NULL);
      wgpuRenderPassEncoderSetVertexBuffer(
        rpass_enc, 0, state.geometry.vertices_buffer.buffer, 0,
        WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetViewport(rpass_enc, (float)(width * m) / 2.0f,
                                       0.0f, (float)width / 2.0f, (float)height,
                                       0.0f, 1.0f);
      wgpuRenderPassEncoderDraw(rpass_enc, GEOMETRY_DRAW_COUNT, NUM_INSTANCES,
                                0, 0);
      wgpuRenderPassEncoderEnd(rpass_enc);
      wgpuRenderPassEncoderRelease(rpass_enc);
    }
  }
  else if (state.settings.mode == RENDER_MODE_PRECISION_ERROR) {
    /* Precision error mode */
    for (uint32_t m = 0; m < DEPTH_BUFFER_MODE_COUNT; ++m) {
      /* Depth pre-pass */
      {
        state.render_pass.depth_pre_pass_depth_attachment.view
          = state.textures.depth_texture_view;
        state.render_pass.depth_pre_pass_depth_attachment.depthClearValue
          = depth_clear_values[m];

        WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
          cmd_enc, &state.render_pass.depth_pre_pass_desc);

        wgpuRenderPassEncoderSetPipeline(rpass_enc,
                                         state.pipelines.depth_pre_pass[m]);
        wgpuRenderPassEncoderSetBindGroup(
          rpass_enc, 0, state.bind_groups.uniform_bind_groups[m], 0, NULL);
        wgpuRenderPassEncoderSetVertexBuffer(
          rpass_enc, 0, state.geometry.vertices_buffer.buffer, 0,
          WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetViewport(rpass_enc, (float)(width * m) / 2.0f,
                                         0.0f, (float)width / 2.0f,
                                         (float)height, 0.0f, 1.0f);
        wgpuRenderPassEncoderDraw(rpass_enc, GEOMETRY_DRAW_COUNT, NUM_INSTANCES,
                                  0, 0);
        wgpuRenderPassEncoderEnd(rpass_enc);
        wgpuRenderPassEncoderRelease(rpass_enc);
      }

      /* Precision error pass */
      {
        state.render_pass.color_attachment.view      = attachment;
        state.render_pass.color_attachment_load.view = attachment;
        state.render_pass.depth_stencil_attachment.view
          = state.textures.default_depth_texture_view;
        state.render_pass.depth_stencil_attachment.depthClearValue
          = depth_clear_values[m];

        WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
          cmd_enc, &state.render_pass.draw_pass_desc[m]);

        wgpuRenderPassEncoderSetPipeline(rpass_enc,
                                         state.pipelines.precision_pass[m]);
        wgpuRenderPassEncoderSetBindGroup(
          rpass_enc, 0, state.bind_groups.uniform_bind_groups[m], 0, NULL);
        wgpuRenderPassEncoderSetBindGroup(
          rpass_enc, 1, state.bind_groups.depth_texture_bind_group, 0, NULL);
        wgpuRenderPassEncoderSetVertexBuffer(
          rpass_enc, 0, state.geometry.vertices_buffer.buffer, 0,
          WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetViewport(rpass_enc, (float)(width * m) / 2.0f,
                                         0.0f, (float)width / 2.0f,
                                         (float)height, 0.0f, 1.0f);
        wgpuRenderPassEncoderDraw(rpass_enc, GEOMETRY_DRAW_COUNT, NUM_INSTANCES,
                                  0, 0);
        wgpuRenderPassEncoderEnd(rpass_enc);
        wgpuRenderPassEncoderRelease(rpass_enc);
      }
    }
  }
  else {
    /* Depth texture mode */
    for (uint32_t m = 0; m < DEPTH_BUFFER_MODE_COUNT; ++m) {
      /* Depth pre-pass */
      {
        state.render_pass.depth_pre_pass_depth_attachment.view
          = state.textures.depth_texture_view;
        state.render_pass.depth_pre_pass_depth_attachment.depthClearValue
          = depth_clear_values[m];

        WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
          cmd_enc, &state.render_pass.depth_pre_pass_desc);

        wgpuRenderPassEncoderSetPipeline(rpass_enc,
                                         state.pipelines.depth_pre_pass[m]);
        wgpuRenderPassEncoderSetBindGroup(
          rpass_enc, 0, state.bind_groups.uniform_bind_groups[m], 0, NULL);
        wgpuRenderPassEncoderSetVertexBuffer(
          rpass_enc, 0, state.geometry.vertices_buffer.buffer, 0,
          WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetViewport(rpass_enc, (float)(width * m) / 2.0f,
                                         0.0f, (float)width / 2.0f,
                                         (float)height, 0.0f, 1.0f);
        wgpuRenderPassEncoderDraw(rpass_enc, GEOMETRY_DRAW_COUNT, NUM_INSTANCES,
                                  0, 0);
        wgpuRenderPassEncoderEnd(rpass_enc);
        wgpuRenderPassEncoderRelease(rpass_enc);
      }

      /* Texture quad pass */
      {
        state.render_pass.color_attachment.view      = attachment;
        state.render_pass.color_attachment_load.view = attachment;

        WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
          cmd_enc, &state.render_pass.texture_quad_pass_desc[m]);

        wgpuRenderPassEncoderSetPipeline(rpass_enc,
                                         state.pipelines.texture_quad_pass);
        wgpuRenderPassEncoderSetBindGroup(
          rpass_enc, 0, state.bind_groups.depth_texture_bind_group, 0, NULL);
        wgpuRenderPassEncoderSetViewport(rpass_enc, (float)(width * m) / 2.0f,
                                         0.0f, (float)width / 2.0f,
                                         (float)height, 0.0f, 1.0f);
        wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 0);
        wgpuRenderPassEncoderEnd(rpass_enc);
        wgpuRenderPassEncoderRelease(rpass_enc);
      }
    }
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* Shutdown function */
static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Release geometry buffers */
  wgpu_destroy_buffer(&state.geometry.vertices_buffer);

  /* Release depth textures */
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.depth_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, state.textures.depth_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.default_depth_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, state.textures.default_depth_texture)

  /* Release uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.camera_matrix_buffer)
  WGPU_RELEASE_RESOURCE(Buffer,
                        state.uniforms.camera_matrix_reversed_depth_buffer)

  /* Release bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.bind_group_layouts.depth_texture_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.uniform_bgl)

  /* Release bind groups */
  for (uint32_t i = 0; i < DEPTH_BUFFER_MODE_COUNT; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.uniform_bind_groups[i])
  }
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.depth_texture_bind_group)

  /* Release pipelines */
  for (uint32_t i = 0; i < DEPTH_BUFFER_MODE_COUNT; ++i) {
    WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.depth_pre_pass[i])
    WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.precision_pass[i])
    WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.color_pass[i])
  }
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.texture_quad_pass)

  /* Release pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.depth_pre_pass)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.precision_pass)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.color_pass)
  WGPU_RELEASE_RESOURCE(PipelineLayout,
                        state.pipeline_layouts.texture_quad_pass)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Reversed Z",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* vertex_shader_wgsl = CODE(
  struct Uniforms {
    modelMatrix : array<mat4x4f, 5>,
  }
  struct Camera {
    viewProjectionMatrix : mat4x4f,
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<uniform> camera : Camera;

  struct VertexOutput {
    @builtin(position) Position : vec4f,
    @location(0) fragColor : vec4f,
  }

  @vertex
  fn main(
    @builtin(instance_index) instanceIdx : u32,
    @location(0) position : vec4f,
    @location(1) color : vec4f
  ) -> VertexOutput {
    var output : VertexOutput;
    output.Position = camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
    output.fragColor = color;
    return output;
  }
);

static const char* fragment_shader_wgsl = CODE(
  @fragment
  fn main(
    @location(0) fragColor: vec4f
  ) -> @location(0) vec4f {
    return fragColor;
  }
);

static const char* vertex_depth_pre_pass_shader_wgsl = CODE(
  struct Uniforms {
    modelMatrix : array<mat4x4f, 5>,
  }
  struct Camera {
    viewProjectionMatrix : mat4x4f,
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<uniform> camera : Camera;

  @vertex
  fn main(
    @builtin(instance_index) instanceIdx : u32,
    @location(0) position : vec4f
  ) -> @builtin(position) vec4f {
    return camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
  }
);

static const char* vertex_texture_quad_shader_wgsl = CODE(
  @vertex
  fn main(
    @builtin(vertex_index) VertexIndex : u32
  ) -> @builtin(position) vec4f {
    const pos = array(
      vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
      vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    );

    return vec4(pos[VertexIndex], 0.0, 1.0);
  }
);

static const char* fragment_texture_quad_shader_wgsl = CODE(
  @group(0) @binding(0) var depthTexture: texture_2d<f32>;

  @fragment
  fn main(
    @builtin(position) coord : vec4f
  ) -> @location(0) vec4f {
    let depthValue = textureLoad(depthTexture, vec2i(floor(coord.xy)), 0).x;
    return vec4f(depthValue, depthValue, depthValue, 1.0);
  }
);

static const char* vertex_precision_error_pass_shader_wgsl = CODE(
  struct Uniforms {
    modelMatrix : array<mat4x4f, 5>,
  }
  struct Camera {
    viewProjectionMatrix : mat4x4f,
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<uniform> camera : Camera;

  struct VertexOutput {
    @builtin(position) Position : vec4f,
    @location(0) clipPos : vec4f,
  }

  @vertex
  fn main(
    @builtin(instance_index) instanceIdx : u32,
    @location(0) position : vec4f
  ) -> VertexOutput {
    var output : VertexOutput;
    output.Position = camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
    output.clipPos = output.Position;
    return output;
  }
);

static const char* fragment_precision_error_pass_shader_wgsl = CODE(
  @group(1) @binding(0) var depthTexture: texture_2d<f32>;

  @fragment
  fn main(
    @builtin(position) coord: vec4f,
    @location(0) clipPos: vec4f
  ) -> @location(0) vec4f {
    let depthValue = textureLoad(depthTexture, vec2i(floor(coord.xy)), 0).x;
    let v : f32 = abs(clipPos.z / clipPos.w - depthValue) * 2000000.0;
    return vec4f(v, v, v, 1.0);
  }
);
// clang-format on
