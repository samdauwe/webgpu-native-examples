#include "example_base.h"

#include <cJSON.h>
#include <string.h>

#include "../core/log.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Normal Mapping example
 *
 * This example demonstrates how to achieve normal mapping in WebGPU. A normal
 * map uses RGB information that corresponds directly with the X, Y and Z axis
 * in 3D space. This RGB information tells the 3D application the exact
 * direction of the surface normals are oriented in for each and every polygon.
 *
 * Ref:
 * https://github.com/Konstantin84UKR/webgpu_examples/tree/master/normalMap
 *
 * Note:
 * http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Vertex data - Torus Knot Mesh / Plane Mesh
 * -------------------------------------------------------------------------- */

typedef enum mesh_type_t {
  MESH_TYPE_TORUS_KNOT = 0,
  MESH_TYPE_PLANE      = 1,
  MESH_TYPE_COUNT      = 2,
} mesh_type_t;

#define TORUS_KNOT_VERTEX_COUNT 7893
#define TORUS_KNOT_FACES_COUNT 3000
#define TORUS_KNOT_INDEX_COUNT (TORUS_KNOT_FACES_COUNT * 3)
#define TORUS_KNOT_UV_COUNT 5262
#define TORUS_KNOT_NORMAL_COUNT 7893
#define TORUS_KNOT_TANGENTS_COUNT 7893
#define TORUS_KNOT_BITANGENTS_COUNT 7893

#define PLANE_VERTEX_COUNT 75
#define PLANE_FACES_COUNT 32
#define PLANE_INDEX_COUNT (TORUS_KNOT_FACES_COUNT * 3)
#define PLANE_UV_COUNT 50
#define PLANE_NORMAL_COUNT 75
#define PLANE_TANGENTS_COUNT 75
#define PLANE_BITANGENTS_COUNT 75

static struct torus_knot_mesh {
  float vertices[TORUS_KNOT_VERTEX_COUNT];
  uint32_t indices[TORUS_KNOT_INDEX_COUNT];
  float uvs[TORUS_KNOT_UV_COUNT];
  float normals[TORUS_KNOT_NORMAL_COUNT];
  float tangents[TORUS_KNOT_TANGENTS_COUNT];
  float bitangents[TORUS_KNOT_BITANGENTS_COUNT];
} torus_knot_mesh = {0};

static struct plane_mesh {
  float vertices[PLANE_VERTEX_COUNT];
  uint32_t indices[PLANE_INDEX_COUNT];
  float uvs[PLANE_UV_COUNT];
  float normals[PLANE_NORMAL_COUNT];
  float tangents[PLANE_TANGENTS_COUNT];
  float bitangents[PLANE_BITANGENTS_COUNT];
} plane_mesh = {0};

int32_t prepare_meshes(mesh_type_t mesh_type)
{
  int32_t res = EXIT_FAILURE;

  file_read_result_t file_read_result = {0};
  read_file("meshes/model.json", &file_read_result, true);
  const char* const json_data = (const char* const)file_read_result.data;

  cJSON* model_json = cJSON_Parse(json_data);
  if (model_json == NULL) {
    const char* error_ptr = cJSON_GetErrorPtr();
    if (error_ptr != NULL) {
      log_error("Error before: %s", error_ptr);
    }
    goto load_json_end;
  }

  if (!cJSON_IsObject(model_json)
      || !cJSON_HasObjectItem(model_json, "meshes")) {
    log_error("Invalid mesh file, does not contain 'meshes' array");
    goto load_json_end;
  }

  /* Get meshes */
  for (int32_t mesh_id = 0; mesh_id < (int32_t)MESH_TYPE_COUNT; ++mesh_id) {
    const cJSON* meshes_array
      = cJSON_GetObjectItemCaseSensitive(model_json, "meshes");
    if (!cJSON_IsArray(meshes_array)) {
      log_error("'meshes' object item is not an array");
      goto load_json_end;
    }
    if (!(cJSON_GetArraySize(meshes_array) > mesh_id)) {
      log_error(
        "'meshes' array does not contain any mesh object for mesh type %d",
        mesh_id);
      goto load_json_end;
    }
    const cJSON* meshes_item = cJSON_GetArrayItem(meshes_array, mesh_id);

    if (!cJSON_IsObject(meshes_item)
        || !cJSON_HasObjectItem(meshes_item, "vertices")
        || !cJSON_HasObjectItem(meshes_item, "faces")
        || !cJSON_HasObjectItem(meshes_item, "texturecoords")
        || !cJSON_HasObjectItem(meshes_item, "normals")
        || !cJSON_HasObjectItem(meshes_item, "tangents")
        || !cJSON_HasObjectItem(meshes_item, "bitangents")) {
      log_error(
        "Invalid mesh object, does not contain 'vertices', 'faces', "
        "'texturecoords', 'normals', 'tangents', 'bitangents' array");
      goto load_json_end;
    }

    /* Parse vertices */
    {
      const cJSON* vertex_array = NULL;
      const cJSON* vertex_item  = NULL;

      vertex_array = cJSON_GetObjectItemCaseSensitive(meshes_item, "vertices");
      if (!cJSON_IsArray(vertex_array)) {
        log_error("vertices object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_VERTEX_COUNT :
                           PLANE_VERTEX_COUNT;
      ASSERT(cJSON_GetArraySize(vertex_array) == expectedSize);

      float* mesh_vertices = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                               torus_knot_mesh.vertices :
                               plane_mesh.vertices;
      uint32_t c           = 0;
      cJSON_ArrayForEach(vertex_item, vertex_array)
      {
        mesh_vertices[c++] = (float)vertex_item->valuedouble;
      }
    }

    /* Parse indices */
    {
      const cJSON* faces_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "faces");
      if (!cJSON_IsArray(faces_array)) {
        log_error("'faces' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_FACES_COUNT :
                           PLANE_FACES_COUNT;
      ASSERT(cJSON_GetArraySize(faces_array) == expectedSize);

      uint32_t* mesh_indices = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                                 torus_knot_mesh.indices :
                                 plane_mesh.indices;
      const cJSON* face_item = NULL;
      uint32_t c             = 0;
      cJSON_ArrayForEach(face_item, faces_array)
      {
        if (!(cJSON_GetArraySize(face_item) == 3)) {
          log_error("'face' item is not an array of size 3");
          goto load_json_end;
        }
        for (uint32_t i = 0; i < 3; ++i) {
          mesh_indices[c++]
            = (uint32_t)cJSON_GetArrayItem(face_item, i)->valueint;
        }
      }
    }

    /* Parse uvs */
    {
      const cJSON* texturecoord_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "texturecoords");
      if (!(cJSON_GetArraySize(texturecoord_array) > 0)) {
        log_error("'texturecoords' array does not contain any object");
        goto load_json_end;
      }
      const cJSON* texturecoords_item
        = cJSON_GetArrayItem(texturecoord_array, 0);
      if (!cJSON_IsArray(texturecoords_item)) {
        log_error("'texturecoords' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_UV_COUNT :
                           PLANE_UV_COUNT;
      ASSERT(cJSON_GetArraySize(texturecoords_item) == expectedSize);

      float* mesh_uvs                = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                                         torus_knot_mesh.uvs :
                                         plane_mesh.uvs;
      const cJSON* texturecoord_item = NULL;
      uint32_t c                     = 0;
      cJSON_ArrayForEach(texturecoord_item, texturecoords_item)
      {
        mesh_uvs[c++] = (float)texturecoord_item->valuedouble;
      }
    }

    /* Parse normals */
    {
      const cJSON* normal_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "normals");
      if (!cJSON_IsArray(normal_array)) {
        log_error("'normals' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_NORMAL_COUNT :
                           PLANE_NORMAL_COUNT;
      ASSERT(cJSON_GetArraySize(normal_array) == expectedSize);

      float* mesh_normals      = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                                   torus_knot_mesh.normals :
                                   plane_mesh.normals;
      const cJSON* normal_item = NULL;
      uint32_t c               = 0;
      cJSON_ArrayForEach(normal_item, normal_array)
      {
        mesh_normals[c++] = (float)normal_item->valuedouble;
      }
    }

    /* Parse tangents */
    {
      const cJSON* tangent_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "tangents");
      if (!cJSON_IsArray(tangent_array)) {
        log_error("'tangents' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_TANGENTS_COUNT :
                           PLANE_TANGENTS_COUNT;
      ASSERT(cJSON_GetArraySize(tangent_array) == expectedSize);

      float* mesh_tangents      = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                                    torus_knot_mesh.tangents :
                                    plane_mesh.tangents;
      const cJSON* tangent_item = NULL;
      uint32_t c                = 0;
      cJSON_ArrayForEach(tangent_item, tangent_array)
      {
        mesh_tangents[c++] = (float)tangent_array->valuedouble;
      }
    }

    /* Parse bitangents */
    {
      const cJSON* bitangent_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "bitangents");
      if (!cJSON_IsArray(bitangent_array)) {
        log_error("'bitangents' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_BITANGENTS_COUNT :
                           PLANE_BITANGENTS_COUNT;
      ASSERT(cJSON_GetArraySize(bitangent_array) == expectedSize);

      float* mesh_bitangents      = (mesh_type == MESH_TYPE_TORUS_KNOT) ?
                                      torus_knot_mesh.bitangents :
                                      plane_mesh.bitangents;
      const cJSON* bitangent_item = NULL;
      uint32_t c                  = 0;
      cJSON_ArrayForEach(bitangent_item, bitangent_array)
      {
        mesh_bitangents[c++] = (float)bitangent_array->valuedouble;
      }
    }
  }

  res = EXIT_SUCCESS;

load_json_end:
  cJSON_Delete(model_json);
  free(file_read_result.data);

  return res;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */
static const char* shadow_vertex_shader_wgsl;
static const char* normal_map_vertex_shader_wgsl;
static const char* normal_map_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Normal Mapping example
 * -------------------------------------------------------------------------- */

/* Buffers */
static struct {
  struct {
    wgpu_buffer_t vertex;
    wgpu_buffer_t index;
    wgpu_buffer_t uv;
    wgpu_buffer_t normal;
    wgpu_buffer_t tangent;
    wgpu_buffer_t bitangent;
  } torus_knot;
  wgpu_buffer_t normal_map_vs_uniform_buffer;
  wgpu_buffer_t normal_map_fs_uniform_buffer_0;
  wgpu_buffer_t normal_map_fs_uniform_buffer_1;
  wgpu_buffer_t uniform_buffer_shadow;
} buffers = {0};

/* Textures and samplers */
static struct {
  texture_t diffuse;
  texture_t normal;
  texture_t specular;
  texture_t shadow_depth;
  texture_t depth;
} textures = {0};

static struct {
  WGPUSampler normal_map;
  WGPUSampler shadow_depth;
} samplers = {0};

/* Uniform bind groups and render pipelines (and layout) */
static struct {
  WGPUBindGroup shadow;
  WGPUBindGroup normal_map;
  WGPUBindGroup shadow_Depth;
} bind_groups = {0};

static struct {
  WGPURenderPipeline shadow;
  WGPURenderPipeline normal_map;
} pipelines = {0};

// Other variables
static const char* example_title = "Normal Mapping example";
static bool prepared             = false;

static void prepare_buffers(wgpu_context_t* wgpu_context)
{
  //******************************* Torus Knot *******************************//

  /* Vertex buffer */
  buffers.torus_knot.vertex = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.vertices),
                    .initial.data = torus_knot_mesh.vertices,
                  });

  /* Index buffer */
  buffers.torus_knot.index = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(torus_knot_mesh.indices),
                    .initial.data = torus_knot_mesh.indices,
                  });

  /* UV buffer */
  buffers.torus_knot.uv = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "UV buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.uvs),
                    .initial.data = torus_knot_mesh.uvs,
                  });

  /* Normal buffer */
  buffers.torus_knot.normal = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Normal buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.normals),
                    .initial.data = torus_knot_mesh.normals,
                  });

  /* Tangent buffer */
  buffers.torus_knot.tangent = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Tangents buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.tangents),
                    .initial.data = torus_knot_mesh.tangents,
                  });

  /* Bitangent buffer */
  buffers.torus_knot.bitangent = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Bitangents buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.bitangents),
                    .initial.data = torus_knot_mesh.bitangents,
                  });
}

static void prepare_textures(wgpu_context_t* wgpu_context)
{
  /* Diffuse texture*/
  {
    const char* file = "textures/brics_diffuse.jpg";
    textures.diffuse = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  /* Normal texture*/
  {
    const char* file = "textures/brics_normal.jpg";
    textures.normal  = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  /* Specular texture*/
  {
    const char* file  = "textures/brics_specular.jpg";
    textures.specular = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  /* Samplers */
  {
    samplers.normal_map = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = "Normal map texture sampler",
                              .addressModeU  = WGPUAddressMode_Repeat,
                              .addressModeV  = WGPUAddressMode_Repeat,
                              .addressModeW  = WGPUAddressMode_Repeat,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(samplers.normal_map);

    samplers.shadow_depth = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = "Shadow depth texture sampler",
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Nearest,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                              .compare       = WGPUCompareFunction_Less,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(samplers.shadow_depth != NULL);
  }

  /* Shadow depth texture */
  {
    textures.shadow_depth.texture =  wgpuDeviceCreateTexture(wgpu_context->device,
      &(WGPUTextureDescriptor) {
        .label         = "Shadow depth texture",
        .usage         = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
        .dimension     = WGPUTextureDimension_2D,
        .format        = WGPUTextureFormat_Depth24Plus,
        .mipLevelCount = 1,
        .sampleCount   = 1,
        .size          = (WGPUExtent3D)  {
          .width               = wgpu_context->surface.width,
          .height              = wgpu_context->surface.height,
          .depthOrArrayLayers  = 1,
        },
      });

    textures.shadow_depth.view = wgpuTextureCreateView(
      textures.shadow_depth.texture, &(WGPUTextureViewDescriptor){
                                       .label     = "Shadow depth texture view",
                                       .dimension = WGPUTextureViewDimension_2D,
                                       .format = WGPUTextureFormat_Depth24Plus,
                                       .mipLevelCount   = 1,
                                       .arrayLayerCount = 1,
                                     });
  }

  /* Depth texture */
  {
    textures.depth.texture =  wgpuDeviceCreateTexture(wgpu_context->device,
      &(WGPUTextureDescriptor) {
        .label         = "Depth texture",
        .usage         = WGPUTextureUsage_RenderAttachment,
        .dimension     = WGPUTextureDimension_2D,
        .format        = WGPUTextureFormat_Depth24Plus,
        .mipLevelCount = 1,
        .sampleCount   = 1,
        .size          = (WGPUExtent3D)  {
          .width               = wgpu_context->surface.width,
          .height              = wgpu_context->surface.height,
          .depthOrArrayLayers  = 1,
        },
      });

    textures.depth.view = wgpuTextureCreateView(
      textures.depth.texture, &(WGPUTextureViewDescriptor){
                                .label         = "Depth texture view",
                                .dimension     = WGPUTextureViewDimension_2D,
                                .format        = WGPUTextureFormat_Depth24Plus,
                                .mipLevelCount = 1,
                                .arrayLayerCount = 1,
                              });
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Shadow bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = buffers.uniform_buffer_shadow.buffer,
        .offset  = 0,
        .size    = buffers.uniform_buffer_shadow.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Bind group for shadow pass",
      .layout     = wgpuRenderPipelineGetBindGroupLayout(pipelines.shadow, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.shadow
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.shadow != NULL);
  }

  /* Normal map */
  {
    WGPUBindGroupEntry bg_entries[7] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = buffers.normal_map_vs_uniform_buffer.buffer,
        .offset  = 0,
        .size    = buffers.normal_map_vs_uniform_buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .sampler = samplers.normal_map,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding     = 2,
        .textureView = textures.diffuse.view,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .buffer  = buffers.normal_map_fs_uniform_buffer_0.buffer,
        .offset  = 0,
        .size    = buffers.normal_map_fs_uniform_buffer_0.size,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding = 4,
        .buffer  = buffers.uniform_buffer_shadow.buffer,
        .offset  = 0,
        .size    = buffers.uniform_buffer_shadow.size,
      },
      [5] = (WGPUBindGroupEntry) {
        .binding     = 5,
        .textureView = textures.normal.view,
      },
      [6] = (WGPUBindGroupEntry) {
        .binding     = 6,
        .textureView = textures.specular.view,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label  = "Bind group for normal map pass",
      .layout = wgpuRenderPipelineGetBindGroupLayout(pipelines.normal_map, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.normal_map
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.normal_map != NULL);
  }

  /* Shadow depth */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = textures.shadow_depth.view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .sampler = samplers.shadow_depth,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = buffers.normal_map_fs_uniform_buffer_1.buffer,
        .offset  = 0,
        .size    = buffers.normal_map_fs_uniform_buffer_1.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label  = "Bind group for shadow depth pass",
      .layout = wgpuRenderPipelineGetBindGroupLayout(pipelines.normal_map, 1),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.shadow_Depth
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.shadow_Depth != NULL);
  }
}

static void prepare_shadow_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPUVertexBufferLayout textured_torus_knot_vertex_buffer_layouts[3] = {0};
  {
    WGPUVertexAttribute attribute = {
      // Shader location 0 : position attribute
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    textured_torus_knot_vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 1 : uv attribute
      .shaderLocation = 1,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x2,
    };
    textured_torus_knot_vertex_buffer_layouts[1] = (WGPUVertexBufferLayout){
      .arrayStride    = 2 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 2 : Normal attribute
      .shaderLocation = 2,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    textured_torus_knot_vertex_buffer_layouts[2] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "shadow_vertex_shader_wgsl",
                      .wgsl_code.source = shadow_vertex_shader_wgsl,
                      .entry            = "main",
                    },
                    .buffer_count = (uint32_t)ARRAY_SIZE(textured_torus_knot_vertex_buffer_layouts),
                    .buffers = textured_torus_knot_vertex_buffer_layouts,
                  });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  pipelines.shadow = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "shadow_render_pipeline",
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipelines.shadow != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
}

static void prepare_normal_map_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  // Enable depth testing so that the fragment closest to the camera is rendered
  // in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPUVertexBufferLayout normal_map_vertex_buffer_layouts[5] = {0};
  {
    WGPUVertexAttribute attribute = {
      // Shader location 0 : position attribute
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x4,
    };
    normal_map_vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
      .arrayStride    = 4 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 1 : uv attribute
      .shaderLocation = 1,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x2,
    };
    normal_map_vertex_buffer_layouts[1] = (WGPUVertexBufferLayout){
      .arrayStride    = 2 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 2 : normal attribute
      .shaderLocation = 2,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    normal_map_vertex_buffer_layouts[2] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 3 : tangent attribute
      .shaderLocation = 3,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    normal_map_vertex_buffer_layouts[3] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 4 : bitangent attribute
      .shaderLocation = 4,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    normal_map_vertex_buffer_layouts[4] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "normal_map_vertex_shader_wgsl",
                      .wgsl_code.source = normal_map_vertex_shader_wgsl,
                      .entry            = "main",
                    },
                    .buffer_count = (uint32_t)ARRAY_SIZE(normal_map_vertex_buffer_layouts),
                    .buffers      = normal_map_vertex_buffer_layouts,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "normal_map_fragment_shader_wgsl",
                      .wgsl_code.source = normal_map_fragment_shader_wgsl,
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
  pipelines.normal_map = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "normal_map_render_pipeline",
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipelines.normal_map != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* shadow_vertex_shader_wgsl = CODE(
  struct Uniform {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };

  @group(0) @binding(0) var<uniform> uniforms : Uniform;

  @vertex
  fn main(@location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>
  ) -> @builtin(position) vec4<f32> {
    return uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
  }
);

static const char* normal_map_vertex_shader_wgsl = CODE(
  struct Uniform {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @group(0) @binding(0) var<uniform> uniforms : Uniform;

  struct UniformLight {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @group(0) @binding(4) var<uniform> uniformsLight : UniformLight;

  struct Output {
     @builtin(position) Position : vec4<f32>,
     @location(0) fragPosition : vec3<f32>,
     @location(1) fragUV : vec2<f32>,
     // @location(2) fragNormal : vec3<f32>,
     @location(3) shadowPos : vec3<f32>,
     @location(4) fragNor : vec3<f32>,
     @location(5) fragTangent : vec3<f32>,
     @location(6) fragBitangent : vec3<f32>
  };


  @vertex
  fn main(@location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>
  ) -> Output {
    var output: Output;
    output.Position = uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
    output.fragPosition = (uniforms.mMatrix * pos).xyz;
    output.fragUV = uv;
    //output.fragNormal  = (uniforms.mMatrix * vec4<f32>(normal,1.0)).xyz;

    // -----NORMAL --------------------------------

    var nMatrix : mat4x4<f32> = uniforms.mMatrix;
    nMatrix[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    let norm : vec3<f32>  = normalize((nMatrix * vec4<f32>(normal,1.0)).xyz);
    let tang : vec3<f32> = normalize((nMatrix * vec4<f32>(tangent,1.0)).xyz);
    let binormal : vec3<f32> = normalize((nMatrix * vec4<f32>(bitangent,1.0)).xyz);

    output.fragNor  = norm;
    output.fragTangent  = tang;
    output.fragBitangent  = binormal;


    let posFromLight: vec4<f32> = uniformsLight.pMatrix * uniformsLight.vMatrix * uniformsLight.mMatrix * pos;
    // Convert shadowPos XY to (0, 1) to fit texture UV
    output.shadowPos = vec3<f32>(posFromLight.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5), posFromLight.z);

    return output;
   }
);

static const char* normal_map_fragment_shader_wgsl = CODE(
  @binding(1) @group(0) var textureSampler : sampler;
  @binding(2) @group(0) var textureData : texture_2d<f32>;
  @binding(5) @group(0) var textureDataNormal : texture_2d<f32>;
  @binding(6) @group(0) var textureDataSpecular : texture_2d<f32>;

  struct Uniforms {
    eyePosition : vec4<f32>,
    lightPosition : vec4<f32>,
  };
  @binding(3) @group(0) var<uniform> uniforms : Uniforms;

  @binding(0) @group(1) var shadowMap : texture_depth_2d;
  @binding(1) @group(1) var shadowSampler : sampler_comparison;
  @binding(2) @group(1) var<uniform> test : vec3<f32>;


  @fragment
  fn main(@location(0) fragPosition: vec3<f32>,
    @location(1) fragUV: vec2<f32>,
    //@location(2) fragNormal: vec3<f32>,
    @location(3) shadowPos: vec3<f32>,
    @location(4) fragNor: vec3<f32>,
    @location(5) fragTangent: vec3<f32>,
    @location(6) fragBitangent: vec3<f32>
  ) -> @location(0) vec4<f32> {
    let specularColor: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    let i = 4.0f;
    let textureColor: vec3<f32> = (textureSample(textureData, textureSampler, fragUV * i)).rgb;
    let texturSpecular: vec3<f32> = (textureSample(textureDataSpecular, textureSampler, fragUV * i)).rgb;

    var textureNormal: vec3<f32> = normalize(2.0 * (textureSample(textureDataNormal, textureSampler, fragUV * i)).rgb - 1.0);
    var colorNormal = normalize(vec3<f32>(textureNormal.x, textureNormal.y, textureNormal.z));
    colorNormal.y *= -1;

    var tbnMatrix : mat3x3<f32> = mat3x3<f32>(
      normalize(fragTangent),
      normalize(fragBitangent),
      normalize(fragNor)
    );

    colorNormal = normalize(tbnMatrix * colorNormal);

    var shadow : f32 = 0.0;
    // apply Percentage-closer filtering (PCF)
    // sample nearest 9 texels to smooth result
    let size = f32(textureDimensions(shadowMap).x);
    for (var y : i32 = -1 ; y <= 1 ; y = y + 1) {
      for (var x : i32 = -1 ; x <= 1 ; x = x + 1) {
        let offset = vec2<f32>(f32(x) / size, f32(y) / size);
        shadow = shadow + textureSampleCompare(
            shadowMap,
            shadowSampler,
            shadowPos.xy + offset,
            shadowPos.z - 0.005  // apply a small bias to avoid acne
        );
      }
    }
    shadow = shadow / 9.0;

    let N: vec3<f32> = normalize(colorNormal.xyz);
    let L: vec3<f32> = normalize((uniforms.lightPosition).xyz - fragPosition.xyz);
    let V: vec3<f32> = normalize((uniforms.eyePosition).xyz - fragPosition.xyz);
    let H: vec3<f32> = normalize(L + V);

    let diffuse: f32 = 0.8 * max(dot(N, L), 0.0);
    let specular = pow(max(dot(N, H),0.0),100.0);
    let ambient: vec3<f32> = vec3<f32>(test.x + 0.2, 0.4, 0.5);

    let finalColor: vec3<f32> =  textureColor * ( shadow * diffuse + ambient) + (texturSpecular * specular * shadow);
    // let finalColor:vec3<f32> =  colorNormal * 0.5 + 0.5;  //let color = N * 0.5 + 0.5;
    // let finalColor:vec3<f32> =  texturSpecular ;  //let color = N * 0.5 + 0.5;

    return vec4<f32>(finalColor, 1.0);
);
// clang-format on
