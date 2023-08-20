#include "example_base.h"

#include <string.h>

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

typedef struct iPipeline_t {
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
} iPipeline_t;

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
  uint32_t i                                       = 0;
  for (i = 0; i < buffers_len && i < MAX_BIND_GROUP_COUNT; ++i) {
    entries[i] = (WGPUBindGroupEntry){
      .binding = i,
      .buffer  = buffers[i],
    };
  }
  return wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                             .layout     = layout,
                                             .entryCount = i,
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
 * Blinn-Phong Lighting example
 * -------------------------------------------------------------------------- */

static iPipeline_t prepare_render_pipelines(iweb_gpu_init_t* init,
                                            ivertex_data_t* data)
{
  /* pipeline for shape */
  WGPURenderPipeline shape_render_pipeline = NULL;

  /* render pipeline for wireframe */
  WGPURenderPipeline wireframe_render_pipeline = NULL;

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
    = create_buffer(init->device, 192, BufferType_Uniform);

  /* light uniform buffers for shape and wireframe */
  WGPUBuffer light_uniform_buffer
    = create_buffer(init->device, 64, BufferType_Uniform);
  WGPUBuffer light_uniform_buffer_2
    = create_buffer(init->device, 64, BufferType_Uniform);

  /* uniform buffer for material */
  WGPUBuffer material_uniform_buffer
    = create_buffer(init->device, 16, BufferType_Uniform);

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

  return (iPipeline_t){
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
