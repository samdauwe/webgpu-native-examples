#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cJSON.h>
#include <cglm/cglm.h>
#include <string.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

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

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Blinn-Phong Lighting
 *
 * This example demonstrates how to render a torus knot mesh with Blinn-Phong
 * lighting model. A small sphere represents the light source position which
 * orbits the torus knot. The scene includes diffuse texturing, ambient
 * lighting, and specular highlights using the Blinn-Phong BRDF.
 *
 * Ref:
 * https://github.com/Konstantin84UKR/webgpu_examples/tree/master/phong
 * https://github.com/jack1232/ebook-webgpu-lighting/tree/main/src/examples/ch04
 *
 * Note:
 * https://learnopengl.com/Advanced-Lighting/Advanced-Lighting
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* torus_knot_vertex_shader_wgsl;
static const char* torus_knot_fragment_shader_wgsl;
static const char* sphere_vertex_shader_wgsl;
static const char* sphere_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Vertex data constants
 * -------------------------------------------------------------------------- */

#define TORUS_KNOT_VERTEX_COUNT 7893u
#define TORUS_KNOT_FACES_COUNT 3000u
#define TORUS_KNOT_INDEX_COUNT (TORUS_KNOT_FACES_COUNT * 3u)
#define TORUS_KNOT_UV_COUNT 5262u
#define TORUS_KNOT_NORMAL_COUNT 7893u

/* -------------------------------------------------------------------------- *
 * Blinn-Phong Lighting example
 * -------------------------------------------------------------------------- */

/* Torus knot mesh data */
typedef struct torus_knot_mesh_t {
  float vertices[TORUS_KNOT_VERTEX_COUNT];
  uint32_t indices[TORUS_KNOT_INDEX_COUNT];
  float uvs[TORUS_KNOT_UV_COUNT];
  float normals[TORUS_KNOT_NORMAL_COUNT];
  bool loaded;
} torus_knot_mesh_t;

/* Sphere geometry (light indicator) */
typedef struct sphere_mesh_t {
  float* vertices;
  uint32_t* indices;
  uint32_t vertex_count;
  uint32_t index_count;
} sphere_mesh_t;

/* View matrices for uniforms */
typedef struct view_matrices_t {
  mat4 projection;
  mat4 view;
  mat4 model;
} view_matrices_t;

/* Fragment uniform: lighting parameters */
typedef struct light_uniforms_t {
  vec4 eye_position;
  vec4 light_position;
  vec4 params;  /* x: shininess, y: flux, z: unused, w: unused */
  vec4 ambient; /* rgb: ambient color, a: unused */
} light_uniforms_t;

/* State struct */
static struct {
  /* Mesh data */
  torus_knot_mesh_t torus_knot_mesh;
  sphere_mesh_t sphere_mesh;
  /* Async loading buffers */
  uint8_t mesh_file_buffer[768 * 1024];
  uint8_t texture_file_buffer[512 * 512 * 4];
  /* GPU Buffers */
  struct {
    struct {
      wgpu_buffer_t vertex;
      wgpu_buffer_t index;
      wgpu_buffer_t uv;
      wgpu_buffer_t normal;
      wgpu_buffer_t vs_uniform;
      wgpu_buffer_t fs_uniform;
    } torus_knot;
    struct {
      wgpu_buffer_t vertex;
      wgpu_buffer_t index;
      wgpu_buffer_t vs_uniform;
    } sphere;
  } buffers;
  /* Textures */
  struct {
    wgpu_texture_t face;
    WGPUTexture depth;
    WGPUTextureView depth_view;
  } textures;
  /* Bind groups */
  struct {
    WGPUBindGroup torus_knot;
    WGPUBindGroup sphere;
  } bind_groups;
  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout torus_knot;
    WGPUBindGroupLayout sphere;
  } bind_group_layouts;
  /* Pipeline layouts */
  struct {
    WGPUPipelineLayout torus_knot;
    WGPUPipelineLayout sphere;
  } pipeline_layouts;
  /* Render pipelines */
  struct {
    WGPURenderPipeline torus_knot;
    WGPURenderPipeline sphere;
  } pipelines;
  /* View matrices */
  view_matrices_t torus_knot_matrices;
  view_matrices_t sphere_matrices;
  /* Light uniforms */
  light_uniforms_t light_uniforms;
  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* GUI settings */
  struct {
    bool paused;
    float shininess;
    float light_flux;
    float ambient_r;
    float ambient_g;
    float ambient_b;
  } settings;
  /* Timing */
  uint64_t last_frame_time;
  float time_elapsed;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1, 0.2, 0.3, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
    .stencilLoadOp   = WGPULoadOp_Undefined,
    .stencilStoreOp  = WGPUStoreOp_Undefined,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .light_uniforms = {
    .eye_position   = {0.0f, 1.0f, 8.0f, 1.0f},
    .light_position = {0.0f, 0.0f, 1.0f, 1.0f},
    .params         = {100.0f, 10.0f, 0.0f, 0.0f},
    .ambient        = {0.1f, 0.1f, 0.15f, 1.0f},
  },
  .settings = {
    .paused     = false,
    .shininess  = 100.0f,
    .light_flux = 10.0f,
    .ambient_r  = 0.1f,
    .ambient_g  = 0.1f,
    .ambient_b  = 0.15f,
  },
};

/* -------------------------------------------------------------------------- *
 * Sphere geometry generation
 * -------------------------------------------------------------------------- */

static void generate_sphere_mesh(sphere_mesh_t* mesh, float radius,
                                 uint32_t width_segments,
                                 uint32_t height_segments)
{
  const uint32_t vert_count = (width_segments + 1) * (height_segments + 1);
  const uint32_t idx_count  = width_segments * height_segments * 6;

  mesh->vertices     = (float*)malloc(vert_count * 3 * sizeof(float));
  mesh->indices      = (uint32_t*)malloc(idx_count * sizeof(uint32_t));
  mesh->vertex_count = 0;
  mesh->index_count  = 0;

  for (uint32_t iy = 0; iy <= height_segments; iy++) {
    const float v     = (float)iy / (float)height_segments;
    const float theta = v * PI;

    for (uint32_t ix = 0; ix <= width_segments; ix++) {
      const float u   = (float)ix / (float)width_segments;
      const float phi = u * PI2;

      const float x = -radius * cosf(phi) * sinf(theta);
      const float y = radius * cosf(theta);
      const float z = radius * sinf(phi) * sinf(theta);

      mesh->vertices[mesh->vertex_count * 3 + 0] = x;
      mesh->vertices[mesh->vertex_count * 3 + 1] = y;
      mesh->vertices[mesh->vertex_count * 3 + 2] = z;
      mesh->vertex_count++;

      if (iy < height_segments && ix < width_segments) {
        const uint32_t cur = ix + iy * (width_segments + 1);
        const uint32_t nx  = cur + 1;
        const uint32_t ny  = cur + width_segments + 1;
        const uint32_t nxy = ny + 1;

        mesh->indices[mesh->index_count++] = cur;
        mesh->indices[mesh->index_count++] = ny;
        mesh->indices[mesh->index_count++] = nx;
        mesh->indices[mesh->index_count++] = ny;
        mesh->indices[mesh->index_count++] = nxy;
        mesh->indices[mesh->index_count++] = nx;
      }
    }
  }
}

static void destroy_sphere_mesh(sphere_mesh_t* mesh)
{
  if (mesh->vertices) {
    free(mesh->vertices);
    mesh->vertices = NULL;
  }
  if (mesh->indices) {
    free(mesh->indices);
    mesh->indices = NULL;
  }
  mesh->vertex_count = 0;
  mesh->index_count  = 0;
}

/* -------------------------------------------------------------------------- *
 * Mesh JSON parsing (asynchronous callback)
 * -------------------------------------------------------------------------- */

static void parse_torus_knot_mesh(const void* data, size_t size)
{
  torus_knot_mesh_t* mesh = &state.torus_knot_mesh;
  cJSON* model_json       = cJSON_ParseWithLength((const char*)data, size);
  if (!model_json) {
    printf("Error: Failed to parse model.json\n");
    return;
  }

  cJSON* meshes_array = cJSON_GetObjectItemCaseSensitive(model_json, "meshes");
  if (!cJSON_IsArray(meshes_array) || cJSON_GetArraySize(meshes_array) == 0) {
    printf("Error: Invalid meshes array\n");
    cJSON_Delete(model_json);
    return;
  }

  cJSON* mesh_item = cJSON_GetArrayItem(meshes_array, 0);

  /* Parse vertices */
  {
    cJSON* arr = cJSON_GetObjectItemCaseSensitive(mesh_item, "vertices");
    uint32_t c = 0;
    cJSON* item;
    cJSON_ArrayForEach(item, arr)
    {
      if (c < TORUS_KNOT_VERTEX_COUNT) {
        mesh->vertices[c++] = (float)item->valuedouble;
      }
    }
  }

  /* Parse faces -> indices */
  {
    cJSON* arr = cJSON_GetObjectItemCaseSensitive(mesh_item, "faces");
    uint32_t c = 0;
    cJSON* face;
    cJSON_ArrayForEach(face, arr)
    {
      for (int i = 0; i < cJSON_GetArraySize(face) && i < 3; ++i) {
        if (c < TORUS_KNOT_INDEX_COUNT) {
          mesh->indices[c++] = (uint32_t)cJSON_GetArrayItem(face, i)->valueint;
        }
      }
    }
  }

  /* Parse UVs */
  {
    cJSON* tc_arr
      = cJSON_GetObjectItemCaseSensitive(mesh_item, "texturecoords");
    if (cJSON_IsArray(tc_arr) && cJSON_GetArraySize(tc_arr) > 0) {
      cJSON* uv_arr = cJSON_GetArrayItem(tc_arr, 0);
      uint32_t c    = 0;
      cJSON* item;
      cJSON_ArrayForEach(item, uv_arr)
      {
        if (c < TORUS_KNOT_UV_COUNT) {
          mesh->uvs[c++] = (float)item->valuedouble;
        }
      }
    }
  }

  /* Parse normals */
  {
    cJSON* arr = cJSON_GetObjectItemCaseSensitive(mesh_item, "normals");
    uint32_t c = 0;
    cJSON* item;
    cJSON_ArrayForEach(item, arr)
    {
      if (c < TORUS_KNOT_NORMAL_COUNT) {
        mesh->normals[c++] = (float)item->valuedouble;
      }
    }
  }

  mesh->loaded = true;
  cJSON_Delete(model_json);
}

/* -------------------------------------------------------------------------- *
 * Async file loading callbacks
 * -------------------------------------------------------------------------- */

static void mesh_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Error: mesh fetch failed (error: %d)\n", response->error_code);
    return;
  }
  parse_torus_knot_mesh(response->data.ptr, response->data.size);
}

static void texture_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Error: texture fetch failed (error: %d)\n", response->error_code);
    return;
  }

  int img_width, img_height, num_channels;
  stbi_uc* pixels
    = stbi_load_from_memory(response->data.ptr, (int)response->data.size,
                            &img_width, &img_height, &num_channels, 4);
  if (pixels) {
    wgpu_texture_t* texture = &state.textures.face;
    texture->desc           = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = (uint32_t)img_width,
        .height             = (uint32_t)img_height,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels = {
        .ptr  = pixels,
        .size = (size_t)(img_width * img_height * 4),
      },
    };
    texture->desc.is_dirty = true;
  }
}

/* -------------------------------------------------------------------------- *
 * Depth texture
 * -------------------------------------------------------------------------- */

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  /* Release old resources */
  if (state.textures.depth_view) {
    wgpuTextureViewRelease(state.textures.depth_view);
    state.textures.depth_view = NULL;
  }
  if (state.textures.depth) {
    wgpuTextureRelease(state.textures.depth);
    state.textures.depth = NULL;
  }

  state.textures.depth = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label         = STRVIEW("Depth texture"),
      .usage         = WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24Plus,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .size          = (WGPUExtent3D){
        .width              = (uint32_t)wgpu_context->width,
        .height             = (uint32_t)wgpu_context->height,
        .depthOrArrayLayers = 1,
      },
    });

  state.textures.depth_view = wgpuTextureCreateView(
    state.textures.depth, &(WGPUTextureViewDescriptor){
                            .label           = STRVIEW("Depth texture view"),
                            .dimension       = WGPUTextureViewDimension_2D,
                            .format          = WGPUTextureFormat_Depth24Plus,
                            .mipLevelCount   = 1,
                            .arrayLayerCount = 1,
                          });
}

/* -------------------------------------------------------------------------- *
 * Buffer initialization
 * -------------------------------------------------------------------------- */

static void init_buffers(wgpu_context_t* wgpu_context)
{
  torus_knot_mesh_t* mesh = &state.torus_knot_mesh;

  /* Torus knot buffers */
  state.buffers.torus_knot.vertex = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(mesh->vertices),
                    .initial.data = mesh->vertices,
                  });

  state.buffers.torus_knot.index = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(mesh->indices),
                    .initial.data = mesh->indices,
                  });

  state.buffers.torus_knot.uv = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - UV buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(mesh->uvs),
                    .initial.data = mesh->uvs,
                  });

  state.buffers.torus_knot.normal = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - Normal buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(mesh->normals),
                    .initial.data = mesh->normals,
                  });

  state.buffers.torus_knot.vs_uniform = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot VS - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(view_matrices_t),
                  });

  state.buffers.torus_knot.fs_uniform = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot FS - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(light_uniforms_t),
                  });

  /* Sphere buffers */
  sphere_mesh_t* sphere = &state.sphere_mesh;

  state.buffers.sphere.vertex = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sphere - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sphere->vertex_count * 3 * (uint32_t)sizeof(float),
                    .initial.data = sphere->vertices,
                  });

  state.buffers.sphere.index = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sphere - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sphere->index_count * (uint32_t)sizeof(uint32_t),
                    .initial.data = sphere->indices,
                    .count        = sphere->index_count,
                  });

  state.buffers.sphere.vs_uniform = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sphere VS - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(view_matrices_t),
                  });
}

/* -------------------------------------------------------------------------- *
 * Uniform data initialization
 * -------------------------------------------------------------------------- */

static void init_uniform_data(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;
  const float fovy = 40.0f * PI / 180.0f;

  /* Torus knot matrices */
  glm_lookat((vec3){state.light_uniforms.eye_position[0],
                    state.light_uniforms.eye_position[1],
                    state.light_uniforms.eye_position[2]},
             (vec3){0.0f, 0.0f, 0.0f}, (vec3){0.0f, 1.0f, 0.0f},
             state.torus_knot_matrices.view);
  glm_perspective(fovy, aspect_ratio, 1.0f, 25.0f,
                  state.torus_knot_matrices.projection);

  /* Sphere matrices (share view/projection with torus knot) */
  glm_mat4_copy(state.torus_knot_matrices.projection,
                state.sphere_matrices.projection);
  glm_mat4_copy(state.torus_knot_matrices.view, state.sphere_matrices.view);
  glm_mat4_identity(state.sphere_matrices.model);
  glm_translate(state.sphere_matrices.model,
                (vec3){state.light_uniforms.light_position[0],
                       state.light_uniforms.light_position[1],
                       state.light_uniforms.light_position[2]});
}

/* -------------------------------------------------------------------------- *
 * Texture initialization
 * -------------------------------------------------------------------------- */

static void init_texture(wgpu_context_t* wgpu_context)
{
  /* Create placeholder (color bars) texture */
  state.textures.face = wgpu_create_color_bars_texture(wgpu_context, NULL);

  /* Start async fetch of the actual texture */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/uv.jpg",
    .callback = texture_fetch_callback,
    .buffer   = SFETCH_RANGE(state.texture_file_buffer),
  });
}

/* -------------------------------------------------------------------------- *
 * Bind group layout and pipeline layout
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Torus knot bind group layout */
  {
    WGPUBindGroupLayoutEntry entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(view_matrices_t),
        },
      },
      [1] = (WGPUBindGroupLayoutEntry){
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [2] = (WGPUBindGroupLayoutEntry){
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout){
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [3] = (WGPUBindGroupLayoutEntry){
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(light_uniforms_t),
        },
      },
    };
    state.bind_group_layouts.torus_knot = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Torus knot - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(entries),
        .entries    = entries,
      });
    ASSERT(state.bind_group_layouts.torus_knot != NULL);
  }

  /* Sphere bind group layout */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(view_matrices_t),
        },
      },
    };
    state.bind_group_layouts.sphere = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("Sphere - Bind group layout"),
                              .entryCount = (uint32_t)ARRAY_SIZE(entries),
                              .entries    = entries,
                            });
    ASSERT(state.bind_group_layouts.sphere != NULL);
  }

  /* Pipeline layouts */
  state.pipeline_layouts.torus_knot = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Torus knot - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.bind_group_layouts.torus_knot,
    });
  ASSERT(state.pipeline_layouts.torus_knot != NULL);

  state.pipeline_layouts.sphere = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Sphere - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.bind_group_layouts.sphere,
    });
  ASSERT(state.pipeline_layouts.sphere != NULL);
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Release old bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.torus_knot)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.sphere)

  /* Torus knot bind group */
  {
    WGPUBindGroupEntry entries[4] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.buffers.torus_knot.vs_uniform.buffer,
        .offset  = 0,
        .size    = state.buffers.torus_knot.vs_uniform.size,
      },
      [1] = (WGPUBindGroupEntry){
        .binding = 1,
        .sampler = state.textures.face.sampler,
      },
      [2] = (WGPUBindGroupEntry){
        .binding     = 2,
        .textureView = state.textures.face.view,
      },
      [3] = (WGPUBindGroupEntry){
        .binding = 3,
        .buffer  = state.buffers.torus_knot.fs_uniform.buffer,
        .offset  = 0,
        .size    = state.buffers.torus_knot.fs_uniform.size,
      },
    };
    state.bind_groups.torus_knot = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Torus knot - Bind group"),
                              .layout     = state.bind_group_layouts.torus_knot,
                              .entryCount = (uint32_t)ARRAY_SIZE(entries),
                              .entries    = entries,
                            });
    ASSERT(state.bind_groups.torus_knot != NULL);
  }

  /* Sphere bind group */
  {
    WGPUBindGroupEntry entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.buffers.sphere.vs_uniform.buffer,
        .offset  = 0,
        .size    = state.buffers.sphere.vs_uniform.size,
      },
    };
    state.bind_groups.sphere = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Sphere - Bind group"),
                              .layout     = state.bind_group_layouts.sphere,
                              .entryCount = (uint32_t)ARRAY_SIZE(entries),
                              .entries    = entries,
                            });
    ASSERT(state.bind_groups.sphere != NULL);
  }
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* ---------- Torus knot pipeline ---------- */
  {
    WGPUShaderModule vert_module = wgpu_create_shader_module(
      wgpu_context->device, torus_knot_vertex_shader_wgsl);
    WGPUShaderModule frag_module = wgpu_create_shader_module(
      wgpu_context->device, torus_knot_fragment_shader_wgsl);

    /* Vertex buffer layouts: position, uv, normal (separate buffers) */
    WGPUVertexAttribute pos_attr = {
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    WGPUVertexAttribute uv_attr = {
      .shaderLocation = 1,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x2,
    };
    WGPUVertexAttribute normal_attr = {
      .shaderLocation = 2,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    WGPUVertexBufferLayout vert_buf_layouts[3] = {
      {
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &pos_attr,
      },
      {
        .arrayStride    = 2 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &uv_attr,
      },
      {
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &normal_attr,
      },
    };

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Torus knot - Render pipeline"),
      .layout = state.pipeline_layouts.torus_knot,
      .vertex = {
        .module      = vert_module,
        .entryPoint  = STRVIEW("main"),
        .bufferCount = 3,
        .buffers     = vert_buf_layouts,
      },
      .fragment = &(WGPUFragmentState){
        .module      = frag_module,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .depthStencil = &depth_stencil_state,
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff,
      },
    };

    state.pipelines.torus_knot
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.pipelines.torus_knot != NULL);

    wgpuShaderModuleRelease(vert_module);
    wgpuShaderModuleRelease(frag_module);
  }

  /* ---------- Sphere pipeline ---------- */
  {
    WGPUShaderModule vert_module = wgpu_create_shader_module(
      wgpu_context->device, sphere_vertex_shader_wgsl);
    WGPUShaderModule frag_module = wgpu_create_shader_module(
      wgpu_context->device, sphere_fragment_shader_wgsl);

    WGPUVertexAttribute pos_attr = {
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    WGPUVertexBufferLayout vert_buf_layout = {
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &pos_attr,
    };

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Sphere - Render pipeline"),
      .layout = state.pipeline_layouts.sphere,
      .vertex = {
        .module      = vert_module,
        .entryPoint  = STRVIEW("main"),
        .bufferCount = 1,
        .buffers     = &vert_buf_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = frag_module,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .depthStencil = &depth_stencil_state,
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff,
      },
    };

    state.pipelines.sphere
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.pipelines.sphere != NULL);

    wgpuShaderModuleRelease(vert_module);
    wgpuShaderModuleRelease(frag_module);
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer updates
 * -------------------------------------------------------------------------- */

static void update_uniform_buffers(wgpu_context_t* wgpu_context, float dt)
{
  if (state.settings.paused) {
    return;
  }

  state.time_elapsed += dt;

  /* Rotate torus knot */
  glm_rotate_x(state.torus_knot_matrices.model, dt * 0.2f,
               state.torus_knot_matrices.model);
  glm_rotate_y(state.torus_knot_matrices.model, dt * 0.2f,
               state.torus_knot_matrices.model);
  glm_rotate_z(state.torus_knot_matrices.model, dt * 0.2f,
               state.torus_knot_matrices.model);

  /* Orbit light position */
  state.light_uniforms.light_position[0] = sinf(state.time_elapsed) * 4.0f;

  /* Update sphere model matrix to follow light */
  glm_mat4_identity(state.sphere_matrices.model);
  glm_translate(state.sphere_matrices.model,
                (vec3){state.light_uniforms.light_position[0],
                       state.light_uniforms.light_position[1],
                       state.light_uniforms.light_position[2]});

  /* Copy GUI settings into uniforms */
  state.light_uniforms.params[0]  = state.settings.shininess;
  state.light_uniforms.params[1]  = state.settings.light_flux;
  state.light_uniforms.ambient[0] = state.settings.ambient_r;
  state.light_uniforms.ambient[1] = state.settings.ambient_g;
  state.light_uniforms.ambient[2] = state.settings.ambient_b;
  state.light_uniforms.ambient[3] = 1.0f;

  /* Write torus knot uniforms */
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.buffers.torus_knot.vs_uniform.buffer, 0,
                       &state.torus_knot_matrices, sizeof(view_matrices_t));
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.buffers.torus_knot.fs_uniform.buffer, 0,
                       &state.light_uniforms, sizeof(light_uniforms_t));

  /* Write sphere uniforms */
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.buffers.sphere.vs_uniform.buffer, 0,
                       &state.sphere_matrices, sizeof(view_matrices_t));
}

/* -------------------------------------------------------------------------- *
 * GUI rendering
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Blinn-Phong Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  imgui_overlay_checkbox("Paused", &state.settings.paused);

  if (igCollapsingHeaderBoolPtr("Lighting", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_slider_float("Shininess", &state.settings.shininess, 1.0f,
                               256.0f, "%.0f");
    imgui_overlay_slider_float("Light Flux", &state.settings.light_flux, 1.0f,
                               50.0f, "%.1f");
  }

  if (igCollapsingHeaderBoolPtr("Ambient", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_slider_float("R", &state.settings.ambient_r, 0.0f, 1.0f,
                               "%.2f");
    imgui_overlay_slider_float("G", &state.settings.ambient_g, 0.0f, 1.0f,
                               "%.2f");
    imgui_overlay_slider_float("B", &state.settings.ambient_b, 0.0f, 1.0f,
                               "%.2f");
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input event handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
    init_uniform_data(wgpu_context);
    /* Write updated uniforms after resize */
    wgpuQueueWriteBuffer(wgpu_context->queue,
                         state.buffers.torus_knot.vs_uniform.buffer, 0,
                         &state.torus_knot_matrices, sizeof(view_matrices_t));
    wgpuQueueWriteBuffer(wgpu_context->queue,
                         state.buffers.sphere.vs_uniform.buffer, 0,
                         &state.sphere_matrices, sizeof(view_matrices_t));
  }
  else if (input_event->type == INPUT_EVENT_TYPE_CHAR
           && input_event->char_code == (uint32_t)'p') {
    state.settings.paused = !state.settings.paused;
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 2,
    .num_channels = 1,
    .num_lanes    = 1,
    .logger.func  = slog_func,
  });

  /* Generate sphere mesh for light indicator */
  generate_sphere_mesh(&state.sphere_mesh, 0.1f, 16, 8);

  /* Initialize model matrix for torus knot */
  glm_mat4_identity(state.torus_knot_matrices.model);

  /* Start async mesh loading */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/meshes/model.json",
    .callback = mesh_fetch_callback,
    .buffer   = SFETCH_RANGE(state.mesh_file_buffer),
  });

  /* Initialize uniform data */
  init_uniform_data(wgpu_context);

  /* Create buffers (initially with zeroed torus knot data) */
  init_buffers(wgpu_context);

  /* Initialize texture with async loading */
  init_texture(wgpu_context);

  /* Create depth texture */
  init_depth_texture(wgpu_context);

  /* Create bind group layouts and pipeline layouts */
  init_bind_group_layouts(wgpu_context);

  /* Create bind groups */
  init_bind_groups(wgpu_context);

  /* Create render pipelines */
  init_pipelines(wgpu_context);

  /* Write initial uniforms */
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.buffers.torus_knot.vs_uniform.buffer, 0,
                       &state.torus_knot_matrices, sizeof(view_matrices_t));
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.buffers.torus_knot.fs_uniform.buffer, 0,
                       &state.light_uniforms, sizeof(light_uniforms_t));
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.buffers.sphere.vs_uniform.buffer, 0,
                       &state.sphere_matrices, sizeof(view_matrices_t));

  /* Initialize ImGui overlay */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async fetches */
  sfetch_dowork();

  /* Handle texture reload */
  if (state.textures.face.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.textures.face);
    FREE_TEXTURE_PIXELS(state.textures.face);
    init_bind_groups(wgpu_context);
  }

  /* Handle mesh data becoming available */
  if (state.torus_knot_mesh.loaded) {
    /* Update vertex buffers with loaded mesh data */
    wgpuQueueWriteBuffer(
      wgpu_context->queue, state.buffers.torus_knot.vertex.buffer, 0,
      state.torus_knot_mesh.vertices, sizeof(state.torus_knot_mesh.vertices));
    wgpuQueueWriteBuffer(
      wgpu_context->queue, state.buffers.torus_knot.index.buffer, 0,
      state.torus_knot_mesh.indices, sizeof(state.torus_knot_mesh.indices));
    wgpuQueueWriteBuffer(
      wgpu_context->queue, state.buffers.torus_knot.uv.buffer, 0,
      state.torus_knot_mesh.uvs, sizeof(state.torus_knot_mesh.uvs));
    wgpuQueueWriteBuffer(
      wgpu_context->queue, state.buffers.torus_knot.normal.buffer, 0,
      state.torus_knot_mesh.normals, sizeof(state.torus_knot_mesh.normals));
    state.torus_knot_mesh.loaded = false; /* Only upload once */
  }

  /* Calculate delta time */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Update uniforms */
  update_uniform_buffers(wgpu_context, delta_time);

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* Set render pass attachments */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.textures.depth_view;

  /* Create command encoder and begin render pass */
  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Draw torus knot */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipelines.torus_knot);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 0, state.buffers.torus_knot.vertex.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 1, state.buffers.torus_knot.uv.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 2, state.buffers.torus_knot.normal.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rpass_enc, state.buffers.torus_knot.index.buffer, WGPUIndexFormat_Uint32, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_groups.torus_knot,
                                    0, 0);
  wgpuRenderPassEncoderDrawIndexed(rpass_enc, TORUS_KNOT_INDEX_COUNT, 1, 0, 0,
                                   0);

  /* Draw sphere (light indicator) */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipelines.sphere);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 0, state.buffers.sphere.vertex.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rpass_enc, state.buffers.sphere.index.buffer, WGPUIndexFormat_Uint32, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_groups.sphere, 0,
                                    0);
  wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.sphere_mesh.index_count, 1,
                                   0, 0, 0);

  /* End render pass */
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit */
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  destroy_sphere_mesh(&state.sphere_mesh);

  wgpu_destroy_buffer(&state.buffers.torus_knot.vertex);
  wgpu_destroy_buffer(&state.buffers.torus_knot.index);
  wgpu_destroy_buffer(&state.buffers.torus_knot.uv);
  wgpu_destroy_buffer(&state.buffers.torus_knot.normal);
  wgpu_destroy_buffer(&state.buffers.torus_knot.vs_uniform);
  wgpu_destroy_buffer(&state.buffers.torus_knot.fs_uniform);
  wgpu_destroy_buffer(&state.buffers.sphere.vertex);
  wgpu_destroy_buffer(&state.buffers.sphere.index);
  wgpu_destroy_buffer(&state.buffers.sphere.vs_uniform);

  wgpu_destroy_texture(&state.textures.face);

  if (state.textures.depth_view) {
    wgpuTextureViewRelease(state.textures.depth_view);
  }
  if (state.textures.depth) {
    wgpuTextureRelease(state.textures.depth);
  }

  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.torus_knot)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.sphere)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.torus_knot)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.sphere)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.torus_knot)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.sphere)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.torus_knot)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.sphere)
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Blinn-Phong Lighting",
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
static const char* torus_knot_vertex_shader_wgsl = CODE(
  struct Uniform {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @binding(0) @group(0) var<uniform> uniforms : Uniform;

  struct Output {
    @builtin(position) Position : vec4<f32>,
    @location(0) vPosition : vec4<f32>,
    @location(1) vUV : vec2<f32>,
    @location(2) vNormal : vec4<f32>,
  };

  @vertex
  fn main(
    @location(0) pos : vec4<f32>,
    @location(1) uv : vec2<f32>,
    @location(2) normal : vec3<f32>
  ) -> Output {
    var output : Output;
    output.Position = uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
    output.vPosition = uniforms.mMatrix * pos;
    output.vUV = uv;
    output.vNormal = uniforms.mMatrix * vec4<f32>(normal, 1.0);
    return output;
  }
);

static const char* torus_knot_fragment_shader_wgsl = CODE(
  @binding(1) @group(0) var textureSampler : sampler;
  @binding(2) @group(0) var textureData : texture_2d<f32>;

  const PI : f32 = 3.1415926535897932384626433832795;

  struct Uniforms {
    eyePosition : vec4<f32>,
    lightPosition : vec4<f32>,
    params : vec4<f32>,
    ambient : vec4<f32>,
  };
  @binding(3) @group(0) var<uniform> uniforms : Uniforms;

  fn lin2rgb(lin : vec3<f32>) -> vec3<f32> {
    return pow(lin, vec3<f32>(1.0 / 2.2));
  }

  fn rgb2lin(rgb : vec3<f32>) -> vec3<f32> {
    return pow(rgb, vec3<f32>(2.2));
  }

  fn brdfPhong(
    lighDir : vec3<f32>,
    viewDir : vec3<f32>,
    halfDir : vec3<f32>,
    normal : vec3<f32>,
    phongDiffuseColor : vec3<f32>,
    phongSpecularColor : vec3<f32>,
    phongShininess : f32
  ) -> vec3<f32> {
    var color : vec3<f32> = phongDiffuseColor;
    let specDot : f32 = max(dot(normal, halfDir), 0.0);
    color += pow(specDot, phongShininess) * phongSpecularColor;
    return color;
  }

  @fragment
  fn main(
    @location(0) vPosition : vec4<f32>,
    @location(1) vUV : vec2<f32>,
    @location(2) vNormal : vec4<f32>
  ) -> @location(0) vec4<f32> {
    let specularColor : vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    let diffuseColor : vec3<f32> = vec3<f32>(0.25, 0.5, 0.2);
    let lightColor : vec3<f32> = vec3<f32>(0.6, 0.7, 0.8);
    let ambientColor : vec3<f32> = uniforms.ambient.rgb;
    let shininess = uniforms.params.x;
    let flux = uniforms.params.y;

    let textureColor : vec3<f32> = (textureSample(textureData, textureSampler, vUV)).rgb;

    let N : vec3<f32> = normalize(vNormal.xyz);
    let L : vec3<f32> = normalize((uniforms.lightPosition).xyz - vPosition.xyz);
    let V : vec3<f32> = normalize((uniforms.eyePosition).xyz - vPosition.xyz);
    let H : vec3<f32> = normalize(L + V);

    let distlight = distance((uniforms.lightPosition).xyz, vPosition.xyz);

    let ambient : vec3<f32> = rgb2lin(textureColor.rgb) * rgb2lin(ambientColor.rgb);

    let irradiance = flux / (4.0 * PI * distlight * distlight) * max(dot(N, L), 0.0);

    let brdf = brdfPhong(L, V, H, N, rgb2lin(textureColor), rgb2lin(specularColor), shininess);

    var radiance = brdf * irradiance * rgb2lin(lightColor.rgb) + ambient;

    return vec4<f32>(lin2rgb(radiance), 1.0);
  }
);

static const char* sphere_vertex_shader_wgsl = CODE(
  struct Uniform {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @binding(0) @group(0) var<uniform> uniforms : Uniform;

  struct Output {
    @builtin(position) Position : vec4<f32>,
  };

  @vertex
  fn main(@location(0) pos : vec4<f32>) -> Output {
    var output : Output;
    output.Position = uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
    return output;
  }
);

static const char* sphere_fragment_shader_wgsl = CODE(
  @fragment
  fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.9, 0.9, 0.9, 1.0);
  }
);
// clang-format on
