#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Blinn-Phong Lighting example
 *
 * Ref:
 * https://github.com/jack1232/ebook-webgpu-lighting/tree/main/src/examples/ch04
 *
 * Note:
 * https://learnopengl.com/Advanced-Lighting/Advanced-Lighting
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * webgpu-simplified - Enums
 * -------------------------------------------------------------------------- */

/** The enumeration for specifying the type of a GPU buffer. */
typedef enum bufer_type_enum {
  BufferType_Uniform          = 0,
  BufferType_Vertex           = 1,
  BufferType_Index            = 2,
  BufferType_Storage          = 3,
  BufferType_Vertex_Storage   = 4,
  BufferType_Index_Storage    = 5,
  BufferType_Indirect         = 6,
  BufferType_Indirect_Storage = 7,
  BufferType_Read             = 8,
  BufferType_Write            = 9,
} bufer_type_enum;

/** The enumeration for specifying the type of input data. */
typedef enum array_data_type_enum {
  DataType_Float32Array = 0,
  DataType_Float64Array = 1,
  DataType_Uint16Array  = 2,
  DataType_Uint32Array  = 3,
} array_data_type_enum;

/* -------------------------------------------------------------------------- *
 * webgpu-simplified - Data types
 * -------------------------------------------------------------------------- */

/**
 * @brief Interface as output of the `initWebGPU` function.
 */
typedef struct iweb_gpu_init_t {
  /** The GPU device */
  WGPUDevice device;
  /** The WGPU context */
  wgpu_context_t* wgpu_context;
  /** The GPU texture format */
  WGPUTextureFormat format;
  /** The canvas size */
  WGPUExtent2D size;
  /** The background color for the scene */
  WGPUColor background;
  /** MSAA count (1 or 4) */
  uint32_t msaa_count;
} iweb_gpu_init_t;

#define PIPELINE_COUNT 4
#define VERTEX_BUFFER_COUNT 4
#define UNIFORM_BUFFER_COUNT 4
#define UNIFORM_BIND_GROUP_COUNT 4
#define GPU_TEXTURE_COUNT 1
#define DEPTH_TEXTURE_COUNT 1

typedef struct ipipeline_t {
  /** The render pipelines */
  WGPURenderPipeline pipelines[PIPELINE_COUNT];
  /** The vertex buffer array */
  WGPUBuffer vertex_buffers[VERTEX_BUFFER_COUNT];
  /** The uniform buffer array */
  WGPUBuffer uniform_buffers[UNIFORM_BUFFER_COUNT];
  /** The uniform bind group array */
  WGPUBindGroup uniform_bind_groups[UNIFORM_BIND_GROUP_COUNT];
  /** The GPU texture array */
  WGPUTexture gpu_textures[GPU_TEXTURE_COUNT];
  WGPUTextureView gpu_texture_views[GPU_TEXTURE_COUNT];
  /** The depth texture array */
  WGPUTexture depth_textures[DEPTH_TEXTURE_COUNT];
  WGPUTextureView depth_texture_views[DEPTH_TEXTURE_COUNT];
  /** The render pass */
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } render_pass;
} ipipeline_t;

/** Interface as input of the `createRenderPipelineDescriptor` function. */
typedef struct irender_pipeline_input {
  /** The iweb_gpu_init_t interface */
  iweb_gpu_init_t* init;
  /** The GPU primative topology with default */
  WGPUPrimitiveTopology primitive_type;
  /** The GPU cull mode - defines which polygon orientation will be culled */
  WGPUCullMode cull_mode;
  /** The boolean variable - indicates whether the render pipeline should
   * include a depth stencial state or not */
  bool is_depth_stencil;
  /** The `buffers` attribute of the vertex state in a render pipeline
   * descriptor */
  size_t buffer_count;
  WGPUVertexBufferLayout const* buffers;
  /** The WGSL vertex shader */
  const char* vs_shader;
  /** The WGSL fragment shader */
  const char* fs_shader;
  /** The entry point for the vertex shader. Default `'vs_main'`  */
  const char* vs_entry;
  /** The entry point for the fragment shader. Default `'fs_main'`  */
  const char* fs_entry;
} irender_pipeline_input;

typedef struct range_t {
  const void* ptr;
  size_t size;
} range_t;

typedef struct ivertex_data_t {
  range_t positions;
  range_t colors;
  range_t normals;
  range_t uvs;
  range_t indices;
  range_t indices2;
  uint32_t indices_count;
  uint32_t indices2_count;
} ivertex_data_t;

/* -------------------------------------------------------------------------- *
 * webgpu-simplified - Functions
 * -------------------------------------------------------------------------- */

/**
 * @brief This function is used to initialize the WebGPU apps. It returns the
 * iweb_gpu_init_t interface.
 *
 * @param wgpu_context the WebGPU context
 * @param msaa_count the MSAA count
 * @param iweb_gpu_init_t object
 */
static iweb_gpu_init_t init_web_gpu(wgpu_context_t* wgpu_context,
                                    uint32_t msaa_count)
{
  return (iweb_gpu_init_t) {
    .device = wgpu_context->device,
    .format = wgpu_context->swap_chain.format,
      .size = (WGPUExtent2D) {
      .width = wgpu_context->surface.width,
      .height = wgpu_context->surface.height,
    },
    .background = (WGPUColor) {
      .r = 0.009f,
      .g = 0.0125,
      .b = 0.0164f,
      .a = 1.0f
    },
    .msaa_count = msaa_count,
  };
}

static WGPUBufferUsageFlags
get_buffer_usage_flags_from_buffer_type(bufer_type_enum buffer_type)
{
  WGPUBufferUsageFlags common_flags
    = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
  WGPUBufferUsageFlags flag = WGPUBufferUsage_Uniform | common_flags;
  if (buffer_type == BufferType_Vertex) {
    flag = WGPUBufferUsage_Vertex | common_flags;
  }
  else if (buffer_type == BufferType_Index) {
    flag = WGPUBufferUsage_Index | common_flags;
  }
  else if (buffer_type == BufferType_Storage) {
    flag = WGPUBufferUsage_Storage | common_flags;
  }
  else if (buffer_type == BufferType_Vertex_Storage) {
    flag = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage | common_flags;
  }
  else if (buffer_type == BufferType_Index_Storage) {
    flag = WGPUBufferUsage_Index | WGPUBufferUsage_Storage | common_flags;
  }
  else if (buffer_type == BufferType_Indirect) {
    flag = WGPUBufferUsage_Indirect | common_flags;
  }
  else if (buffer_type == BufferType_Indirect_Storage) {
    flag = WGPUBufferUsage_Indirect | WGPUBufferUsage_Storage | common_flags;
  }
  else if (buffer_type == BufferType_Read) {
    flag = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  }
  else if (buffer_type == BufferType_Write) {
    flag = WGPUBufferUsage_MapWrite | common_flags;
  }
  return flag;
}

static WGPURenderPipeline create_render_pipeline(irender_pipeline_input* input)
{
  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    input->init->wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      .wgsl_code.source = input->vs_shader,
                      .entry            = input->vs_entry,
                    },
                    .buffer_count = input->buffer_count,
                    .buffers      = input->buffers,
                  });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    input->init->wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      .wgsl_code.source = input->fs_shader,
                      .entry            = input->fs_entry,
                    },
                    .target_count = 1,
                    .targets = &(WGPUColorTargetState){
                      .format = input->init->format,
                    }
                  });

  /* Pipeline descriptor */
  WGPURenderPipelineDescriptor descriptor = {
    .primitive = {
      .topology  = input->primitive_type,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode = input->cull_mode,
    },
    .multisample = {
      .count                  = input->init->msaa_count,
      .mask                   = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    },
  };
  if (input->is_depth_stencil) {
    descriptor.depthStencil = &(WGPUDepthStencilState){
      .format            = WGPUTextureFormat_Depth24Plus,
      .depthWriteEnabled = true,
      .depthCompare      = WGPUCompareFunction_Less,
    };
  };

  /* Create pipeline */
  WGPURenderPipeline pipeline
    = wgpuDeviceCreateRenderPipeline(input->init->device, &descriptor);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);

  return pipeline;
}

/**
 * @brief This function can be used to create vertex, uniform, or storage GPU
 * buffer. The default is a uniform buffer.
 * @param device GPU device
 * @param bufferSize Buffer size.
 * @param buffer_type Of the `buffer_type` enum.
 */
static WGPUBuffer create_buffer(WGPUDevice device, size_t buffer_size,
                                bufer_type_enum buffer_type)
{
  return wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .usage = get_buffer_usage_flags_from_buffer_type(buffer_type),
              .size  = buffer_size,
              .mappedAtCreation = false,
            });
}

/**
 * @brief This function creats a GPU buffer with data to initialize it. If the
 * input data is a type of `Float32Array` or `Float64Array`, it returns a
 * vertex, uniform, or storage buffer specified by the enum `bufferType`.
 * Otherwise, if the input data has a `Uint16Array` or `Uint32Array`, this
 * function will return an index buffer.
 * @param device GPU device
 * @param data Input data that should be one of four data types: `Float32Array`,
 * `Float64Array`, `Uint16Array`, and `Uint32Array`
 * @param data_byte_length the data size in bytes
 * @param array_data_type the data type which should be f four data types:
 * `Float32Array`, `Float64Array`, `Uint16Array`, and `Uint32Array`
 * @param bufferType Type of enum `bufer_type_enum`. It is used to specify the
 * type of the returned buffer. The default is vertex buffer
 */
static WGPUBuffer create_buffer_with_data(WGPUDevice device, const void* data,
                                          size_t data_byte_length,
                                          array_data_type_enum array_data_type,
                                          bufer_type_enum buffer_type)
{
  WGPUBufferUsageFlags flag
    = get_buffer_usage_flags_from_buffer_type(buffer_type);
  if (buffer_type == BufferType_Vertex
      && (array_data_type == DataType_Uint16Array
          || array_data_type == DataType_Uint32Array)) {
    flag = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst
           | WGPUBufferUsage_CopySrc;
  }
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){
                                                       .usage = flag,
                                                       .size = data_byte_length,
                                                       .mappedAtCreation = true,
                                                     });
  ASSERT(buffer != NULL);
  if (array_data_type == DataType_Uint32Array) {
    uint32_t* mapping
      = (uint32_t*)wgpuBufferGetMappedRange(buffer, 0, data_byte_length);
    memcpy(mapping, data, data_byte_length);
  }
  else if (array_data_type == DataType_Uint16Array) {
    uint16_t* mapping
      = (uint16_t*)wgpuBufferGetMappedRange(buffer, 0, data_byte_length);
    memcpy(mapping, data, data_byte_length);
  }
  else if (array_data_type == DataType_Float64Array) {
    double* mapping
      = (double*)wgpuBufferGetMappedRange(buffer, 0, data_byte_length);
    memcpy(mapping, data, data_byte_length);
  }
  else {
    float* mapping
      = (float*)wgpuBufferGetMappedRange(buffer, 0, data_byte_length);
    memcpy(mapping, data, data_byte_length);
  }

  wgpuBufferUnmap(buffer);
  return buffer;
}

/**
 * @brief This function is used to create a GPU bind group that defines a set of
 * resources to be bound together in a group and how the resources are used in
 * shader stages. It accepts GPU device, GPU bind group layout, uniform buffer
 * array, and the other GPU binding resource array as its input arguments. If
 * both the buffer and other resource arrays have none zero elements, you need
 * to place the buffer array ahead of the other resource array. Make sure that
 * the order of buffers and other resources is consistent with the `@group
 * @binding` attributes defined in the shader code.
 * @param device GPU device
 * @param layout GPU bind group layout that defines the interface between a set
 * of resources bound in a GPU bind group and their accessibility in shader
 * stages.
 * @param buffers The uniform buffer array
 * @param buffers_len The number of buffers.
 */
static WGPUBindGroup create_bind_group(WGPUDevice device,
                                       WGPUBindGroupLayout layout,
                                       WGPUBuffer* buffers,
                                       uint32_t buffers_len)
{
#define MAX_BIND_GROUP_COUNT 32

  WGPUBindGroupEntry entries[MAX_BIND_GROUP_COUNT] = {0};
  uint32_t entry_count                             = 0;
  for (uint32_t i = 0; i < buffers_len && i < MAX_BIND_GROUP_COUNT; ++i) {
    entries[i] = (WGPUBindGroupEntry){
      .binding = i,
      .buffer  = buffers[i],
    };
    ++entry_count;
  }
  return wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                             .layout     = layout,
                                             .entryCount = entry_count,
                                             .entries    = entries,
                                           });
}

/**
 * @brief This function creates a GPU texture.
 * @param init The `iweb_gpu_init_t` interface
 * @param format The GPU texture format
 */
static WGPUTexture create_texture(iweb_gpu_init_t* init,
                                  WGPUTextureFormat format)
{
  return wgpuDeviceCreateTexture(init->device, &(WGPUTextureDescriptor) {
     .usage         = WGPUTextureUsage_RenderAttachment,
     .dimension     = WGPUTextureDimension_2D,
     .format        = format,
     .mipLevelCount = 1,
     .sampleCount   = init->msaa_count,
     .size          = (WGPUExtent3D)  {
      .width               = init->size.width,
      .height              = init->size.height,
      .depthOrArrayLayers  = 1,
     },
   });
}

/**
 * @brief This function creates a GPU texture view.
 * @param texture The texture to create the view from
 * @param format The GPU texture format
 */
static WGPUTextureView create_texture_view(WGPUTexture texture,
                                           WGPUTextureFormat format)
{
  return wgpuTextureCreateView(texture,
                               &(WGPUTextureViewDescriptor){
                                 .dimension       = WGPUTextureViewDimension_2D,
                                 .format          = format,
                                 .mipLevelCount   = 1,
                                 .arrayLayerCount = 1,
                               });
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */
static const char* blinn_phong_lighting_vertex_shader_wgsl;
static const char* blinn_phong_lighting_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Blinn-Phong Lighting example
 * -------------------------------------------------------------------------- */

static struct {
  mat4 view_project_mat;
  mat4 model_mat;
  mat4 normal_mat;
} view_uniforms = {0};

static struct {
  vec4 light_position;
  vec4 eye_position;
  vec4 color;
  vec4 specular_color;
} light_uniforms = {0};

static struct {
  float ambient;
  float diffuse;
  float specular;
  float shininess;
} material_uniforms = {0};

typedef enum plot_type_enum {
  PLOT_TYPE_WIRE_FRAME_ONLY = 0,
  PLOT_TYPE_SHAPE_ONLY      = 1,
} plot_type_enum;

static ipipeline_t prepare_render_pipelines(iweb_gpu_init_t* init,
                                            ivertex_data_t* data)
{
  /* The pipeline input */
  irender_pipeline_input render_pipeline_input = {
    .init             = init,
    .primitive_type   = WGPUPrimitiveTopology_TriangleList,
    .cull_mode        = WGPUCullMode_None,
    .is_depth_stencil = true,
    .buffer_count     = 1,
    .buffers          = &(WGPUVertexBufferLayout) {
      .arrayStride    = sizeof(float) * 3 * 2,
      .attributeCount = 2,
      .attributes     = (WGPUVertexAttribute[2]){
        {
          .format         = WGPUVertexFormat_Float32x3,
          .offset         = 0,
          .shaderLocation = 0,
        },
        {
          .format         = WGPUVertexFormat_Float32x3,
          .offset         = sizeof(float) * 3,
          .shaderLocation = 1,
        },
      },
      .stepMode = WGPUVertexStepMode_Vertex,
    },
    .vs_shader        = blinn_phong_lighting_vertex_shader_wgsl,
    .fs_shader        = blinn_phong_lighting_fragment_shader_wgsl,
    .vs_entry         = "vs_main",
    .fs_entry         = "fs_main",
  };

  /* pipeline for shape */
  WGPURenderPipeline shape_render_pipeline
    = create_render_pipeline(&render_pipeline_input);

  /* render pipeline for wireframe */
  WGPURenderPipeline wireframe_render_pipeline
    = create_render_pipeline(&render_pipeline_input);

  /* create vertex and index buffers */
  WGPUBuffer position_buffer = create_buffer_with_data(
    init->device, data->positions.ptr, data->positions.size,
    DataType_Float32Array, BufferType_Vertex);
  WGPUBuffer normal_buffer = create_buffer_with_data(
    init->device, data->normals.ptr, data->normals.size, DataType_Float32Array,
    BufferType_Vertex);
  WGPUBuffer index_buffer = create_buffer_with_data(
    init->device, data->indices.ptr, data->indices.size, DataType_Uint32Array,
    BufferType_Vertex);
  WGPUBuffer index_buffer_2 = create_buffer_with_data(
    init->device, data->indices2.ptr, data->indices2.size, DataType_Uint32Array,
    BufferType_Vertex);

  /* uniform buffer for model-matrix, vp-matrix, and normal-matrix */
  WGPUBuffer view_uniform_buffer
    = create_buffer(init->device, sizeof(view_uniforms), BufferType_Uniform);

  /* light uniform buffers for shape and wireframe */
  WGPUBuffer light_uniform_buffer
    = create_buffer(init->device, sizeof(light_uniforms), BufferType_Uniform);
  WGPUBuffer light_uniform_buffer_2
    = create_buffer(init->device, sizeof(light_uniforms), BufferType_Uniform);

  /* uniform buffer for material */
  WGPUBuffer material_uniform_buffer = create_buffer(
    init->device, sizeof(material_uniforms), BufferType_Uniform);

  /* uniform bind group for vertex shader */
  WGPUBindGroup vert_bind_group = create_bind_group(
    init->device,
    wgpuRenderPipelineGetBindGroupLayout(shape_render_pipeline, 0),
    &view_uniform_buffer, 1);
  WGPUBindGroup vert_bind_group_2 = create_bind_group(
    init->device,
    wgpuRenderPipelineGetBindGroupLayout(wireframe_render_pipeline, 0),
    &view_uniform_buffer, 1);

  /* uniform bind group for fragment shader */
  WGPUBuffer shape_frag_ubos[2]
    = {light_uniform_buffer, material_uniform_buffer};
  WGPUBindGroup frag_bind_group = create_bind_group(
    init->device,
    wgpuRenderPipelineGetBindGroupLayout(shape_render_pipeline, 1),
    shape_frag_ubos, 2);
  WGPUBuffer wireframe_frag_ubos[2]
    = {light_uniform_buffer_2, material_uniform_buffer};
  WGPUBindGroup frag_bind_group_2 = create_bind_group(
    init->device,
    wgpuRenderPipelineGetBindGroupLayout(wireframe_render_pipeline, 1),
    wireframe_frag_ubos, 2);

  /* create depth view */
  WGPUTextureFormat depth_texture_format = WGPUTextureFormat_Depth24Plus;
  WGPUTexture depth_texture = create_texture(init, depth_texture_format);
  WGPUTextureView depth_texture_view
    = create_texture_view(depth_texture, depth_texture_format);

  /* create texture view for MSAA */
  WGPUTextureFormat texture_format = init->format;
  WGPUTexture msaa_texture         = create_texture(init, texture_format);
  WGPUTextureView msaa_texture_view
    = create_texture_view(msaa_texture, texture_format);

  return (ipipeline_t){
    .vertex_buffers
    = {position_buffer, normal_buffer, index_buffer, index_buffer_2},
    .uniform_buffers = {view_uniform_buffer, light_uniform_buffer,
                        material_uniform_buffer, light_uniform_buffer_2},
    .uniform_bind_groups
    = {vert_bind_group, frag_bind_group, vert_bind_group_2, frag_bind_group_2},
    .gpu_textures        = {msaa_texture},
    .gpu_texture_views   = {msaa_texture_view},
    .depth_textures      = {depth_texture},
    .depth_texture_views = {depth_texture_view},
  };
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

/* draw shape */
static void draw_shape(WGPURenderPassEncoder render_Pass, ipipeline_t* p,
                       ivertex_data_t* data)
{
  wgpuRenderPassEncoderSetPipeline(render_Pass, p->pipelines[0]);
  wgpuRenderPassEncoderSetVertexBuffer(render_Pass, 0, p->vertex_buffers[0], 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(render_Pass, 1, p->vertex_buffers[1], 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(render_Pass, 0, p->uniform_bind_groups[0],
                                    0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_Pass, 1, p->uniform_bind_groups[1],
                                    0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(render_Pass, p->vertex_buffers[2],
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(render_Pass, data->indices_count, 1, 0, 0,
                                   0);
}

/* draw wireframe */
static void draw_wireframe(WGPURenderPassEncoder render_Pass, ipipeline_t* p,
                           ivertex_data_t* data)
{
  wgpuRenderPassEncoderSetPipeline(render_Pass, p->pipelines[1]);
  wgpuRenderPassEncoderSetVertexBuffer(render_Pass, 0, p->vertex_buffers[0], 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(render_Pass, 1, p->vertex_buffers[1], 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(render_Pass, 0, p->uniform_bind_groups[2],
                                    0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_Pass, 1, p->uniform_bind_groups[3],
                                    0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(render_Pass, p->vertex_buffers[3],
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(render_Pass, data->indices2_count, 1, 0, 0,
                                   0);
}

static WGPUCommandBuffer draw(iweb_gpu_init_t* init, ipipeline_t* p,
                              plot_type_enum plot_type, ivertex_data_t* data)
{
  wgpu_context_t* wgpu_context = init->wgpu_context;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder render_Pass = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &p->render_pass.descriptor);

  if (plot_type == PLOT_TYPE_WIRE_FRAME_ONLY) {
    draw_wireframe(render_Pass, p, data);
  }
  else if (PLOT_TYPE_SHAPE_ONLY) {
    draw_shape(render_Pass, p, data);
  }
  else {
    draw_shape(render_Pass, p, data);
    draw_wireframe(render_Pass, p, data);
  }

  wgpuRenderPassEncoderEnd(render_Pass);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_Pass)

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static void set_light_eye_positions(WGPUQueue queue, ipipeline_t* p,
                                    vec4 light_position, vec4 eye_position)
{
  wgpuQueueWriteBuffer(queue, p->uniform_buffers[1], 0, light_position,
                       sizeof(vec4));
  wgpuQueueWriteBuffer(queue, p->uniform_buffers[1], 16, eye_position,
                       sizeof(vec4));
  wgpuQueueWriteBuffer(queue, p->uniform_buffers[3], 0, light_position,
                       sizeof(vec4));
  wgpuQueueWriteBuffer(queue, p->uniform_buffers[3], 16, eye_position,
                       sizeof(vec4));
}

static void update_view_projection(WGPUQueue queue, ipipeline_t* p,
                                   mat4 vp_matrix, vec4 light_position,
                                   vec4 eye_position)
{
  wgpuQueueWriteBuffer(queue, p->uniform_buffers[0], 0, vp_matrix,
                       sizeof(mat4));
  wgpuQueueWriteBuffer(queue, p->uniform_buffers[1], 0, light_position,
                       sizeof(vec4));
  wgpuQueueWriteBuffer(queue, p->uniform_buffers[1], 16, eye_position,
                       sizeof(vec4));
  wgpuQueueWriteBuffer(queue, p->uniform_buffers[3], 0, light_position,
                       sizeof(vec4));
  wgpuQueueWriteBuffer(queue, p->uniform_buffers[3], 16, eye_position,
                       sizeof(vec4));
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* blinn_phong_lighting_vertex_shader_wgsl = CODE(
  // vertex shader
  struct Uniforms {
    viewProjectMat : mat4x4f,
    modelMat : mat4x4f,
    normalMat : mat4x4f,
  };
  @binding(0) @group(0) var<uniform> uniforms : Uniforms;

  struct Output {
    @builtin(position) position : vec4f,
    @location(0) vPosition : vec4f,
    @location(1) vNormal : vec4f,
  };

  @vertex
  fn vs_main(@location(0) pos: vec3f, @location(1) normal: vec3f) -> Output {
    var output: Output;
    let mPosition = uniforms.modelMat * vec4(pos, 1.0);
    output.vPosition = mPosition;
    output.vNormal =  uniforms.normalMat * vec4(normal, 1.0);
    output.position = uniforms.viewProjectMat * mPosition;
    return output;
  }
);

static const char* blinn_phong_lighting_fragment_shader_wgsl = CODE(
  // fragment shader
  struct LightUniforms {
    lightPosition: vec4f,
    eyePosition: vec4f,
    color: vec4f,
    specularColor: vec4f,
  };
  @group(1) @binding(0) var<uniform> light : LightUniforms;

  struct MaterialUniforms {
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
  };
  @group(1) @binding(1) var<uniform> material : MaterialUniforms;

  fn blinnPhong(N:vec3f, L:vec3f, V:vec3f) -> vec2f{
    let H = normalize(L + V);
    var diffuse = material.diffuse * max(dot(N, L), 0.0);
    diffuse += material.diffuse * max(dot(-N, L), 0.0);
    var specular = material.specular * pow(max(dot(N, H), 0.0), material.shininess);
    specular += material.specular * pow(max(dot(-N, H),0.0), material.shininess);
    return vec2(diffuse, specular);
  }

  @fragment
  fn fs_main(@location(0) vPosition:vec4f, @location(1) vNormal:vec4f) ->  @location(0) vec4f {
    var N = normalize(vNormal.xyz);
    let L = normalize(light.lightPosition.xyz - vPosition.xyz);
    let V = normalize(light.eyePosition.xyz - vPosition.xyz);

    let bp = blinnPhong(N, L, V);

    let finalColor = light.color*(material.ambient + bp[0]) + light.specularColor * bp[1];
    return vec4(finalColor.rgb, 1.0);
  }
);
// clang-format on
