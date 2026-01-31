#include "meshes.h"
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <string.h>

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
 * WebGPU Example - Wireframe
 *
 * This example demonstrates drawing wireframes from triangles in two ways:
 * 1. Using line-list primitive topology with indexed vertex pulling
 * 2. Using barycentric coordinates in the fragment shader
 *
 * The second method creates smoother lines with adjustable thickness and
 * alpha threshold for anti-aliased edges.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/wireframe
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* solid_color_lit_shader_wgsl;
static const char* wireframe_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define NUM_OBJECTS 200
#define DEPTH_FORMAT WGPUTextureFormat_Depth24Plus

/* -------------------------------------------------------------------------- *
 * Model data structures
 * -------------------------------------------------------------------------- */

typedef struct model_t {
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  uint32_t vertex_count;
  uint32_t stride; /* Vertex stride in number of floats (6 = pos + normal) */
} model_t;

typedef struct object_info_t {
  mat4 world_view_projection_matrix;
  mat4 world_matrix;
  float color[4];
  model_t* model;
  WGPUBuffer uniform_buffer;
  WGPUBuffer line_uniform_buffer;
  WGPUBindGroup lit_bind_group;
  WGPUBindGroup wireframe_bind_group;
  WGPUBindGroup barycentric_wireframe_bind_group;
} object_info_t;

/* Uniform buffer structures (must match shader layout) */
typedef struct uniforms_t {
  mat4 world_view_projection_matrix;
  mat4 world_matrix;
  float color[4];
} uniforms_t;

typedef struct line_uniforms_t {
  uint32_t stride;
  float thickness;
  float alpha_threshold;
  uint32_t padding;
} line_uniforms_t;

/* -------------------------------------------------------------------------- *
 * State struct
 * -------------------------------------------------------------------------- */

static struct {
  /* Models */
  struct {
    model_t teapot;
    model_t sphere;
    model_t jewel;
    model_t rock;
    model_t* all[4];
  } models;
  /* Mesh data */
  utah_teapot_mesh_t teapot_mesh;
  sphere_mesh_t sphere_mesh;
  primitive_vertex_data_t jewel_data;
  primitive_vertex_data_t rock_data;
  /* Objects */
  object_info_t objects[NUM_OBJECTS];
  /* Pipelines */
  WGPURenderPipeline lit_pipeline;
  WGPURenderPipeline wireframe_pipeline;
  WGPURenderPipeline barycentric_wireframe_pipeline;
  /* Bind group layouts */
  WGPUBindGroupLayout lit_bind_group_layout;
  WGPUBindGroupLayout wireframe_bind_group_layout;
  /* Depth texture */
  wgpu_texture_t depth_texture;
  /* View matrices */
  mat4 projection_matrix;
  mat4 view_matrix;
  mat4 view_projection_matrix;
  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* Settings */
  struct {
    bool barycentric_coordinates_based;
    float thickness;
    float alpha_threshold;
    bool animate;
    bool lines;
    int32_t depth_bias;
    float depth_bias_slope_scale;
    bool show_models;
  } settings;
  /* Timing */
  float time;
  uint64_t last_frame_time;
  /* State */
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.3f, 0.3f, 0.3f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .settings = {
    .barycentric_coordinates_based = false,
    .thickness                     = 2.0f,
    .alpha_threshold               = 0.5f,
    .animate                       = true,
    .lines                         = true,
    .depth_bias                    = 1,
    .depth_bias_slope_scale        = 0.5f,
    .show_models                   = true,
  },
};

/* -------------------------------------------------------------------------- *
 * Random helper functions
 * -------------------------------------------------------------------------- */

static float rand_float(void)
{
  return (float)rand() / (float)RAND_MAX;
}

static void rand_color(float color[4])
{
  color[0] = rand_float();
  color[1] = rand_float();
  color[2] = rand_float();
  color[3] = 1.0f;
}

static model_t* rand_model(void)
{
  return state.models.all[rand() % 4];
}

/* -------------------------------------------------------------------------- *
 * Model creation functions
 * -------------------------------------------------------------------------- */

static void create_model_from_mesh(wgpu_context_t* wgpu_context,
                                   const float* vertices,
                                   uint64_t vertices_byte_size,
                                   const uint32_t* indices,
                                   uint64_t indices_byte_size, model_t* model)
{
  model->vertex_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Model - Vertex buffer"),
      .size  = vertices_byte_size,
      .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage
               | WGPUBufferUsage_CopyDst,
    });
  wgpuQueueWriteBuffer(wgpu_context->queue, model->vertex_buffer, 0, vertices,
                       vertices_byte_size);

  model->index_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Model - Index buffer"),
      .size  = indices_byte_size,
      .usage = WGPUBufferUsage_Index | WGPUBufferUsage_Storage
               | WGPUBufferUsage_CopyDst,
    });
  wgpuQueueWriteBuffer(wgpu_context->queue, model->index_buffer, 0, indices,
                       indices_byte_size);

  model->vertex_count = (uint32_t)(indices_byte_size / sizeof(uint32_t));
  model->stride       = 6; /* position (3) + normal (3) */
}

static void create_teapot_model(wgpu_context_t* wgpu_context)
{
  /* Load teapot mesh from JSON */
  const char* teapot_json_file = "assets/meshes/teapot.json";
  FILE* fp                     = fopen(teapot_json_file, "r");
  if (!fp) {
    printf("Failed to open teapot.json\n");
    return;
  }

  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char* json_data = malloc(file_size + 1);
  if (fread(json_data, 1, file_size, fp) != (size_t)file_size) {
    printf("Failed to read teapot.json\n");
    free(json_data);
    fclose(fp);
    return;
  }
  json_data[file_size] = '\0';
  fclose(fp);

  utah_teapot_mesh_init(&state.teapot_mesh, json_data);
  free(json_data);

  utah_teapot_mesh_compute_normals(&state.teapot_mesh);

  /* Create vertex buffer with positions and normals interleaved */
  uint64_t vertex_count = state.teapot_mesh.positions.count;
  uint64_t vertex_size  = vertex_count * 6 * sizeof(float);
  float* vertices       = malloc(vertex_size);
  const float scale     = 1.5f;

  for (uint64_t i = 0; i < vertex_count; ++i) {
    vertices[i * 6 + 0] = state.teapot_mesh.positions.data[i][0] * scale;
    vertices[i * 6 + 1] = state.teapot_mesh.positions.data[i][1] * scale;
    vertices[i * 6 + 2] = state.teapot_mesh.positions.data[i][2] * scale;
    vertices[i * 6 + 3] = state.teapot_mesh.normals.data[i][0];
    vertices[i * 6 + 4] = state.teapot_mesh.normals.data[i][1];
    vertices[i * 6 + 5] = state.teapot_mesh.normals.data[i][2];
  }

  /* Convert indices to uint32 */
  uint64_t index_count = state.teapot_mesh.triangles.count * 3;
  uint32_t* indices    = malloc(index_count * sizeof(uint32_t));

  for (uint64_t i = 0; i < state.teapot_mesh.triangles.count; ++i) {
    indices[i * 3 + 0] = state.teapot_mesh.triangles.data[i][0];
    indices[i * 3 + 1] = state.teapot_mesh.triangles.data[i][1];
    indices[i * 3 + 2] = state.teapot_mesh.triangles.data[i][2];
  }

  create_model_from_mesh(wgpu_context, vertices, vertex_size, indices,
                         index_count * sizeof(uint32_t), &state.models.teapot);

  free(vertices);
  free(indices);
}

static void create_sphere_model(wgpu_context_t* wgpu_context)
{
  sphere_mesh_init(&state.sphere_mesh, 20.0f, 32, 16, 0.0f);

  /* Create vertex buffer - sphere mesh has 8 floats per vertex but we need 6 */
  uint64_t src_vertex_count = state.sphere_mesh.vertices.length / 8;
  uint64_t vertex_size      = src_vertex_count * 6 * sizeof(float);
  float* vertices           = malloc(vertex_size);

  for (uint64_t i = 0; i < src_vertex_count; ++i) {
    /* Copy position (3 floats) and normal (3 floats), skip UV */
    vertices[i * 6 + 0] = state.sphere_mesh.vertices.data[i * 8 + 0];
    vertices[i * 6 + 1] = state.sphere_mesh.vertices.data[i * 8 + 1];
    vertices[i * 6 + 2] = state.sphere_mesh.vertices.data[i * 8 + 2];
    vertices[i * 6 + 3] = state.sphere_mesh.vertices.data[i * 8 + 3];
    vertices[i * 6 + 4] = state.sphere_mesh.vertices.data[i * 8 + 4];
    vertices[i * 6 + 5] = state.sphere_mesh.vertices.data[i * 8 + 5];
  }

  /* Convert indices to uint32 */
  uint64_t index_count = state.sphere_mesh.indices.length;
  uint32_t* indices    = malloc(index_count * sizeof(uint32_t));
  for (uint64_t i = 0; i < index_count; ++i) {
    indices[i] = state.sphere_mesh.indices.data[i];
  }

  create_model_from_mesh(wgpu_context, vertices, vertex_size, indices,
                         index_count * sizeof(uint32_t), &state.models.sphere);

  free(vertices);
  free(indices);
}

static void create_jewel_model(wgpu_context_t* wgpu_context)
{
  /* Create a low-poly sphere and flatten normals */
  primitive_vertex_data_t sphere_data = {0};
  primitive_create_sphere(
    &(primitive_sphere_options_t){
      .radius              = 20.0f,
      .subdivisions_axis   = 5,
      .subdivisions_height = 3,
    },
    &sphere_data);

  /* Flatten normals (faceted look) */
  primitive_facet(&sphere_data, &state.jewel_data);
  primitive_vertex_data_destroy(&sphere_data);

  /* Create vertex buffer - primitive has 8 floats per vertex, we need 6 */
  uint64_t vertex_count = state.jewel_data.vertex_count;
  uint64_t vertex_size  = vertex_count * 6 * sizeof(float);
  float* vertices       = malloc(vertex_size);

  for (uint64_t i = 0; i < vertex_count; ++i) {
    /* position + normal, skip UV */
    vertices[i * 6 + 0] = state.jewel_data.vertices[i * 8 + 0];
    vertices[i * 6 + 1] = state.jewel_data.vertices[i * 8 + 1];
    vertices[i * 6 + 2] = state.jewel_data.vertices[i * 8 + 2];
    vertices[i * 6 + 3] = state.jewel_data.vertices[i * 8 + 3];
    vertices[i * 6 + 4] = state.jewel_data.vertices[i * 8 + 4];
    vertices[i * 6 + 5] = state.jewel_data.vertices[i * 8 + 5];
  }

  /* Create indices (after facet, indices are sequential) */
  uint32_t* indices = malloc(vertex_count * sizeof(uint32_t));
  for (uint64_t i = 0; i < vertex_count; ++i) {
    indices[i] = (uint32_t)i;
  }

  create_model_from_mesh(wgpu_context, vertices, vertex_size, indices,
                         vertex_count * sizeof(uint32_t), &state.models.jewel);

  free(vertices);
  free(indices);
}

static void create_rock_model(wgpu_context_t* wgpu_context)
{
  /* Create a sphere with randomness and flatten normals */
  primitive_vertex_data_t sphere_data = {0};

  /* First create a normal sphere */
  primitive_create_sphere(
    &(primitive_sphere_options_t){
      .radius              = 20.0f,
      .subdivisions_axis   = 32,
      .subdivisions_height = 16,
    },
    &sphere_data);

  /* Add randomness to positions */
  for (uint64_t i = 0; i < sphere_data.vertex_count; ++i) {
    float* pos   = &sphere_data.vertices[i * 8];
    float offset = 0.1f * 20.0f * (rand_float() - 0.5f);
    /* Apply offset in radial direction */
    float len = sqrtf(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
    if (len > 0.0f) {
      pos[0] += (pos[0] / len) * offset;
      pos[1] += (pos[1] / len) * offset;
      pos[2] += (pos[2] / len) * offset;
    }
  }

  /* Flatten normals */
  primitive_facet(&sphere_data, &state.rock_data);
  primitive_vertex_data_destroy(&sphere_data);

  /* Create vertex buffer */
  uint64_t vertex_count = state.rock_data.vertex_count;
  uint64_t vertex_size  = vertex_count * 6 * sizeof(float);
  float* vertices       = malloc(vertex_size);

  for (uint64_t i = 0; i < vertex_count; ++i) {
    vertices[i * 6 + 0] = state.rock_data.vertices[i * 8 + 0];
    vertices[i * 6 + 1] = state.rock_data.vertices[i * 8 + 1];
    vertices[i * 6 + 2] = state.rock_data.vertices[i * 8 + 2];
    vertices[i * 6 + 3] = state.rock_data.vertices[i * 8 + 3];
    vertices[i * 6 + 4] = state.rock_data.vertices[i * 8 + 4];
    vertices[i * 6 + 5] = state.rock_data.vertices[i * 8 + 5];
  }

  uint32_t* indices = malloc(vertex_count * sizeof(uint32_t));
  for (uint64_t i = 0; i < vertex_count; ++i) {
    indices[i] = (uint32_t)i;
  }

  create_model_from_mesh(wgpu_context, vertices, vertex_size, indices,
                         vertex_count * sizeof(uint32_t), &state.models.rock);

  free(vertices);
  free(indices);
}

static void init_models(wgpu_context_t* wgpu_context)
{
  create_teapot_model(wgpu_context);
  create_sphere_model(wgpu_context);
  create_jewel_model(wgpu_context);
  create_rock_model(wgpu_context);

  state.models.all[0] = &state.models.teapot;
  state.models.all[1] = &state.models.sphere;
  state.models.all[2] = &state.models.jewel;
  state.models.all[3] = &state.models.rock;
}

/* -------------------------------------------------------------------------- *
 * Depth texture
 * -------------------------------------------------------------------------- */

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_texture(&state.depth_texture);

  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->width,
    .height             = wgpu_context->height,
    .depthOrArrayLayers = 1,
  };

  state.depth_texture.handle = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = STRVIEW("Depth texture"),
                            .size          = texture_extent,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                            .dimension     = WGPUTextureDimension_2D,
                            .format        = DEPTH_FORMAT,
                            .usage         = WGPUTextureUsage_RenderAttachment,
                          });

  state.depth_texture.view = wgpuTextureCreateView(
    state.depth_texture.handle, &(WGPUTextureViewDescriptor){
                                  .label        = STRVIEW("Depth texture view"),
                                  .dimension    = WGPUTextureViewDimension_2D,
                                  .format       = DEPTH_FORMAT,
                                  .baseMipLevel = 0,
                                  .mipLevelCount   = 1,
                                  .baseArrayLayer  = 0,
                                  .arrayLayerCount = 1,
                                });
}

/* -------------------------------------------------------------------------- *
 * Pipeline creation
 * -------------------------------------------------------------------------- */

static void init_lit_bind_group_layout(wgpu_context_t* wgpu_context)
{
  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = {
        .type = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uniforms_t),
      },
    },
  };

  state.lit_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Lit - Bind group layout"),
                            .entryCount = ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
}

static void rebuild_lit_pipeline(wgpu_context_t* wgpu_context)
{
  /* Release existing pipeline if any */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.lit_pipeline)

  /* Create pipeline layout */
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Lit - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts = &state.lit_bind_group_layout,
                          });

  /* Create shader module */
  WGPUShaderModule shader_module = wgpu_create_shader_module(
    wgpu_context->device, solid_color_lit_shader_wgsl);

  /* Vertex buffer layout */
  WGPUVertexAttribute vert_attrs[2] = {
    [0]
    = {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
    [1] = {.format         = WGPUVertexFormat_Float32x3,
           .offset         = 3 * sizeof(float),
           .shaderLocation = 1},
  };

  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = 6 * sizeof(float),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = ARRAY_SIZE(vert_attrs),
    .attributes     = vert_attrs,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state = {
    .format              = DEPTH_FORMAT,
    .depthWriteEnabled   = true,
    .depthCompare        = WGPUCompareFunction_Less,
    .depthBias           = state.settings.depth_bias,
    .depthBiasSlopeScale = state.settings.depth_bias_slope_scale,
    .stencilFront        = {.compare = WGPUCompareFunction_Always},
    .stencilBack         = {.compare = WGPUCompareFunction_Always},
  };

  /* Blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Render pipeline descriptor */
  WGPURenderPipelineDescriptor pipeline_desc = {
    .label  = STRVIEW("Lit - Render pipeline"),
    .layout = pipeline_layout,
    .vertex = {
      .module      = shader_module,
      .entryPoint  = STRVIEW("vs"),
      .bufferCount = 1,
      .buffers     = &vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState){
      .module      = shader_module,
      .entryPoint  = STRVIEW("fs"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
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
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.lit_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);

  wgpuShaderModuleRelease(shader_module);
  wgpuPipelineLayoutRelease(pipeline_layout);
}

static void init_wireframe_pipelines(wgpu_context_t* wgpu_context)
{
  /* Create wireframe bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uniforms_t),
      },
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = {
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = 0,
      },
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = {
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = 0,
      },
    },
    [3] = {
      .binding    = 3,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(line_uniforms_t),
      },
    },
  };

  state.wireframe_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Wireframe - Bind group layout"),
                            .entryCount = ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });

  /* Create pipeline layout */
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Wireframe - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.wireframe_bind_group_layout,
    });

  /* Create shader module */
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, wireframe_shader_wgsl);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state = {
    .format            = DEPTH_FORMAT,
    .depthWriteEnabled = true,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
  };

  /* Blend state for barycentric method (alpha blending) */
  WGPUBlendState alpha_blend_state = {
    .color = {
      .srcFactor = WGPUBlendFactor_One,
      .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
      .operation = WGPUBlendOperation_Add,
    },
    .alpha = {
      .srcFactor = WGPUBlendFactor_One,
      .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
      .operation = WGPUBlendOperation_Add,
    },
  };

  /* Line-list wireframe pipeline */
  WGPURenderPipelineDescriptor wireframe_desc = {
    .label  = STRVIEW("Wireframe - Line list pipeline"),
    .layout = pipeline_layout,
    .vertex = {
      .module     = shader_module,
      .entryPoint = STRVIEW("vsIndexedU32"),
    },
    .fragment = &(WGPUFragmentState){
      .module      = shader_module,
      .entryPoint  = STRVIEW("fs"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology = WGPUPrimitiveTopology_LineList,
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.wireframe_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &wireframe_desc);

  /* Barycentric coordinates based wireframe pipeline */
  WGPURenderPipelineDescriptor barycentric_desc = {
    .label  = STRVIEW("Wireframe - Barycentric pipeline"),
    .layout = pipeline_layout,
    .vertex = {
      .module     = shader_module,
      .entryPoint = STRVIEW("vsIndexedU32BarycentricCoordinateBasedLines"),
    },
    .fragment = &(WGPUFragmentState){
      .module      = shader_module,
      .entryPoint  = STRVIEW("fsBarycentricCoordinateBasedLines"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &alpha_blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology = WGPUPrimitiveTopology_TriangleList,
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.barycentric_wireframe_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &barycentric_desc);

  wgpuShaderModuleRelease(shader_module);
  wgpuPipelineLayoutRelease(pipeline_layout);
}

/* -------------------------------------------------------------------------- *
 * Object initialization
 * -------------------------------------------------------------------------- */

static void init_objects(wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < NUM_OBJECTS; ++i) {
    object_info_t* obj = &state.objects[i];

    /* Assign random color and model */
    rand_color(obj->color);
    obj->model = rand_model();

    /* Create uniform buffer */
    obj->uniform_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label = STRVIEW("Object - Uniform buffer"),
        .size  = sizeof(uniforms_t),
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      });

    /* Create line uniform buffer */
    line_uniforms_t line_uniforms = {
      .stride          = obj->model->stride,
      .thickness       = state.settings.thickness,
      .alpha_threshold = state.settings.alpha_threshold,
    };
    obj->line_uniform_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label = STRVIEW("Line - Uniform buffer"),
        .size  = sizeof(line_uniforms_t),
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      });
    wgpuQueueWriteBuffer(wgpu_context->queue, obj->line_uniform_buffer, 0,
                         &line_uniforms, sizeof(line_uniforms));

    /* Create lit bind group */
    obj->lit_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label   = STRVIEW("Lit - Bind group"),
                              .layout  = state.lit_bind_group_layout,
                              .entryCount = 1,
                              .entries = &(WGPUBindGroupEntry){
                                .binding = 0,
                                .buffer  = obj->uniform_buffer,
                                .size    = sizeof(uniforms_t),
                              },
                            });

    /* Create wireframe bind groups */
    WGPUBindGroupEntry wf_entries[4] = {
      [0] = {.binding = 0,
             .buffer  = obj->uniform_buffer,
             .size    = sizeof(uniforms_t)},
      [1] = {.binding = 1,
             .buffer  = obj->model->vertex_buffer,
             .size    = WGPU_WHOLE_SIZE},
      [2] = {.binding = 2,
             .buffer  = obj->model->index_buffer,
             .size    = WGPU_WHOLE_SIZE},
      [3] = {.binding = 3,
             .buffer  = obj->line_uniform_buffer,
             .size    = sizeof(line_uniforms_t)},
    };

    obj->wireframe_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Wireframe - Bind group"),
                              .layout     = state.wireframe_bind_group_layout,
                              .entryCount = ARRAY_SIZE(wf_entries),
                              .entries    = wf_entries,
                            });

    obj->barycentric_wireframe_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Barycentric wireframe - Bind group"),
        .layout     = state.wireframe_bind_group_layout,
        .entryCount = ARRAY_SIZE(wf_entries),
        .entries    = wf_entries,
      });
  }
}

/* -------------------------------------------------------------------------- *
 * View matrices
 * -------------------------------------------------------------------------- */

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  const float fov    = TO_RADIANS(60.0f);

  glm_perspective(fov, aspect, 0.1f, 1000.0f, state.projection_matrix);
  glm_lookat((vec3){-300.0f, 0.0f, 300.0f}, (vec3){0.0f, 0.0f, 0.0f},
             (vec3){0.0f, 1.0f, 0.0f}, state.view_matrix);
  glm_mat4_mul(state.projection_matrix, state.view_matrix,
               state.view_projection_matrix);
}

/* -------------------------------------------------------------------------- *
 * Uniform updates
 * -------------------------------------------------------------------------- */

static void update_line_uniforms(wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < NUM_OBJECTS; ++i) {
    line_uniforms_t line_uniforms = {
      .stride          = state.objects[i].model->stride,
      .thickness       = state.settings.thickness,
      .alpha_threshold = state.settings.alpha_threshold,
    };
    wgpuQueueWriteBuffer(wgpu_context->queue,
                         state.objects[i].line_uniform_buffer, 0,
                         &line_uniforms, sizeof(line_uniforms));
  }
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < NUM_OBJECTS; ++i) {
    object_info_t* obj = &state.objects[i];

    mat4 world;
    glm_mat4_identity(world);
    glm_translate(
      world,
      (vec3){0.0f, 0.0f, sinf((float)i * 3.721f + state.time * 0.1f) * 200.0f});
    glm_rotate_x(world, (float)i * 4.567f, world);
    glm_rotate_y(world, (float)i * 2.967f, world);
    glm_translate(
      world,
      (vec3){0.0f, 0.0f, sinf((float)i * 9.721f + state.time * 0.1f) * 200.0f});
    glm_rotate_x(world, state.time * 0.53f + (float)i, world);

    glm_mat4_mul(state.view_projection_matrix, world,
                 obj->world_view_projection_matrix);
    glm_mat4_copy(world, obj->world_matrix);

    /* Pack uniforms */
    uniforms_t uniforms;
    glm_mat4_copy(obj->world_view_projection_matrix,
                  uniforms.world_view_projection_matrix);
    glm_mat4_copy(obj->world_matrix, uniforms.world_matrix);
    memcpy(uniforms.color, obj->color, sizeof(float) * 4);

    wgpuQueueWriteBuffer(wgpu_context->queue, obj->uniform_buffer, 0, &uniforms,
                         sizeof(uniforms));
  }
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){320.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Wireframe Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Wireframe method */
  if (igCheckbox("Barycentric Coordinates Based",
                 &state.settings.barycentric_coordinates_based)) {
    /* Method changed - nothing else to do */
  }

  igCheckbox("Lines", &state.settings.lines);
  igCheckbox("Models", &state.settings.show_models);
  igCheckbox("Animate", &state.settings.animate);

  igSeparator();

  if (state.settings.barycentric_coordinates_based) {
    /* Barycentric method settings */
    if (imgui_overlay_slider_float("Thickness", &state.settings.thickness, 0.0f,
                                   10.0f, "%.1f")) {
      update_line_uniforms(wgpu_context);
    }
    if (imgui_overlay_slider_float("Alpha Threshold",
                                   &state.settings.alpha_threshold, 0.0f, 1.0f,
                                   "%.2f")) {
      update_line_uniforms(wgpu_context);
    }
  }
  else {
    /* Line-list method settings (depth bias affects lit models) */
    if (imgui_overlay_slider_int("Depth Bias", &state.settings.depth_bias, -3,
                                 3)) {
      rebuild_lit_pipeline(wgpu_context);
    }
    if (imgui_overlay_slider_float("Depth Bias Slope Scale",
                                   &state.settings.depth_bias_slope_scale,
                                   -1.0f, 1.0f, "%.2f")) {
      rebuild_lit_pipeline(wgpu_context);
    }
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
    init_view_matrices(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Initialization
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    srand((unsigned int)stm_now());

    init_models(wgpu_context);
    init_depth_texture(wgpu_context);
    init_view_matrices(wgpu_context);
    init_lit_bind_group_layout(wgpu_context);
    rebuild_lit_pipeline(wgpu_context);
    init_wireframe_pipelines(wgpu_context);
    init_objects(wgpu_context);
    imgui_overlay_init(wgpu_context);

    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

/* -------------------------------------------------------------------------- *
 * Frame rendering
 * -------------------------------------------------------------------------- */

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update time */
  if (state.settings.animate) {
    state.time = (float)stm_sec(stm_now());
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

  /* Setup render pass */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth_texture.view;

  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Render lit models */
  if (state.settings.show_models) {
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.lit_pipeline);

    for (uint32_t i = 0; i < NUM_OBJECTS; ++i) {
      object_info_t* obj = &state.objects[i];
      wgpuRenderPassEncoderSetVertexBuffer(
        rpass_enc, 0, obj->model->vertex_buffer, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, obj->model->index_buffer,
                                          WGPUIndexFormat_Uint32, 0,
                                          WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, obj->lit_bind_group, 0,
                                        0);
      wgpuRenderPassEncoderDrawIndexed(rpass_enc, obj->model->vertex_count, 1,
                                       0, 0, 0);
    }
  }

  /* Render wireframe lines */
  if (state.settings.lines) {
    if (state.settings.barycentric_coordinates_based) {
      wgpuRenderPassEncoderSetPipeline(rpass_enc,
                                       state.barycentric_wireframe_pipeline);
      for (uint32_t i = 0; i < NUM_OBJECTS; ++i) {
        object_info_t* obj = &state.objects[i];
        wgpuRenderPassEncoderSetBindGroup(
          rpass_enc, 0, obj->barycentric_wireframe_bind_group, 0, 0);
        wgpuRenderPassEncoderDraw(rpass_enc, obj->model->vertex_count, 1, 0, 0);
      }
    }
    else {
      wgpuRenderPassEncoderSetPipeline(rpass_enc, state.wireframe_pipeline);
      for (uint32_t i = 0; i < NUM_OBJECTS; ++i) {
        object_info_t* obj = &state.objects[i];
        wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                          obj->wireframe_bind_group, 0, 0);
        /* Draw 6 vertices per triangle (2 per edge * 3 edges) */
        wgpuRenderPassEncoderDraw(rpass_enc, obj->model->vertex_count * 2, 1, 0,
                                  0);
      }
    }
  }

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Shutdown
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Destroy objects */
  for (uint32_t i = 0; i < NUM_OBJECTS; ++i) {
    WGPU_RELEASE_RESOURCE(Buffer, state.objects[i].uniform_buffer)
    WGPU_RELEASE_RESOURCE(Buffer, state.objects[i].line_uniform_buffer)
    WGPU_RELEASE_RESOURCE(BindGroup, state.objects[i].lit_bind_group)
    WGPU_RELEASE_RESOURCE(BindGroup, state.objects[i].wireframe_bind_group)
    WGPU_RELEASE_RESOURCE(BindGroup,
                          state.objects[i].barycentric_wireframe_bind_group)
  }

  /* Destroy models */
  WGPU_RELEASE_RESOURCE(Buffer, state.models.teapot.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.models.teapot.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.models.sphere.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.models.sphere.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.models.jewel.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.models.jewel.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.models.rock.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.models.rock.index_buffer)

  /* Destroy mesh data */
  sphere_mesh_destroy(&state.sphere_mesh);
  primitive_vertex_data_destroy(&state.jewel_data);
  primitive_vertex_data_destroy(&state.rock_data);

  /* Destroy pipelines and layouts */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.lit_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.wireframe_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.barycentric_wireframe_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.lit_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.wireframe_bind_group_layout)

  /* Destroy depth texture */
  wgpu_destroy_texture(&state.depth_texture);
}

/* -------------------------------------------------------------------------- *
 * Main entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Wireframe",
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
static const char* solid_color_lit_shader_wgsl = CODE(
  struct Uniforms {
    worldViewProjectionMatrix: mat4x4f,
    worldMatrix: mat4x4f,
    color: vec4f,
  };

  struct Vertex {
    @location(0) position: vec4f,
    @location(1) normal: vec3f,
  };

  struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
  };

  @group(0) @binding(0) var<uniform> uni: Uniforms;

  @vertex fn vs(vin: Vertex) -> VSOut {
    var vOut: VSOut;
    vOut.position = uni.worldViewProjectionMatrix * vin.position;
    vOut.normal = (uni.worldMatrix * vec4f(vin.normal, 0)).xyz;
    return vOut;
  }

  @fragment fn fs(vin: VSOut) -> @location(0) vec4f {
    let lightDirection = normalize(vec3f(4, 10, 6));
    let light = dot(normalize(vin.normal), lightDirection) * 0.5 + 0.5;
    return vec4f(uni.color.rgb * light, uni.color.a);
  }
);

static const char* wireframe_shader_wgsl = CODE(
  struct Uniforms {
    worldViewProjectionMatrix: mat4x4f,
    worldMatrix: mat4x4f,
    color: vec4f,
  };

  struct LineUniforms {
    stride: u32,
    thickness: f32,
    alphaThreshold: f32,
  };

  struct VSOut {
    @builtin(position) position: vec4f,
  };

  @group(0) @binding(0) var<uniform> uni: Uniforms;
  @group(0) @binding(1) var<storage, read> positions: array<f32>;
  @group(0) @binding(2) var<storage, read> indices: array<u32>;
  @group(0) @binding(3) var<uniform> line: LineUniforms;

  @vertex fn vsIndexedU32(@builtin(vertex_index) vNdx: u32) -> VSOut {
    // indices make a triangle so for every 3 indices we need to output
    // 6 values
    let triNdx = vNdx / 6;
    // 0 1 0 1 0 1  0 1 0 1 0 1  vNdx % 2
    // 0 0 1 1 2 2  3 3 4 4 5 5  vNdx / 2
    // 0 1 1 2 2 3  3 4 4 5 5 6  vNdx % 2 + vNdx / 2
    // 0 1 1 2 2 0  0 1 1 2 2 0  (vNdx % 2 + vNdx / 2) % 3
    let vertNdx = (vNdx % 2 + vNdx / 2) % 3;
    let index = indices[triNdx * 3 + vertNdx];

    let pNdx = index * line.stride;
    let position = vec4f(positions[pNdx], positions[pNdx + 1], positions[pNdx + 2], 1);

    var vOut: VSOut;
    vOut.position = uni.worldViewProjectionMatrix * position;
    return vOut;
  }

  @fragment fn fs() -> @location(0) vec4f {
    return uni.color + vec4f(0.5);
  }

  struct BarycentricCoordinateBasedVSOutput {
    @builtin(position) position: vec4f,
    @location(0) barycenticCoord: vec3f,
  };

  @vertex fn vsIndexedU32BarycentricCoordinateBasedLines(
    @builtin(vertex_index) vNdx: u32
  ) -> BarycentricCoordinateBasedVSOutput {
    let vertNdx = vNdx % 3;
    let index = indices[vNdx];

    let pNdx = index * line.stride;
    let position = vec4f(positions[pNdx], positions[pNdx + 1], positions[pNdx + 2], 1);

    var vsOut: BarycentricCoordinateBasedVSOutput;
    vsOut.position = uni.worldViewProjectionMatrix * position;

    // emit a barycentric coordinate
    vsOut.barycenticCoord = vec3f(0);
    vsOut.barycenticCoord[vertNdx] = 1.0;
    return vsOut;
  }

  fn edgeFactor(bary: vec3f) -> f32 {
    let d = fwidth(bary);
    let a3 = smoothstep(vec3f(0.0), d * line.thickness, bary);
    return min(min(a3.x, a3.y), a3.z);
  }

  @fragment fn fsBarycentricCoordinateBasedLines(
    v: BarycentricCoordinateBasedVSOutput
  ) -> @location(0) vec4f {
    let a = 1.0 - edgeFactor(v.barycenticCoord);
    if (a < line.alphaThreshold) {
      discard;
    }

    return vec4((uni.color.rgb + 0.5) * a, a);
  }
);
// clang-format on
