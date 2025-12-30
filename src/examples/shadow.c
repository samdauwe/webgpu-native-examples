#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Shadow Mapping
 *
 * This example demonstrates shadow mapping using a depth texture array.
 * Multiple lights cast shadows on a scene with a plane and rotating cubes.
 *
 * Ref:
 * https://github.com/gfx-rs/wgpu/blob/trunk/examples/features/src/shadow
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* shadow_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Shadow Mapping Example
 * -------------------------------------------------------------------------- */

#define MAX_LIGHTS 10
#define SHADOW_SIZE 512
#define NUM_CUBES 4

/* Vertex structure */
typedef struct {
  int8_t pos[4];
  int8_t normal[4];
} vertex_t;

/* Entity uniforms */
typedef struct {
  mat4 model;
  vec4 color;
} entity_uniforms_t;

/* Global uniforms */
typedef struct {
  mat4 view_proj;
  uint32_t num_lights[4];
} global_uniforms_t;

/* Light structure (raw data for GPU) */
typedef struct {
  mat4 proj;
  vec4 pos;
  vec4 color;
} light_raw_t;

/* Light structure (CPU side) */
typedef struct {
  vec3 pos;
  vec4 color;
  float fov;
  float depth_near;
  float depth_far;
  WGPUTextureView target_view;
} light_t;

/* Entity structure */
typedef struct {
  mat4 mx_world;
  float rotation_speed;
  vec4 color;
  WGPUBuffer vertex_buf;
  WGPUBuffer index_buf;
  uint32_t index_count;
  uint32_t uniform_offset;
} entity_t;

/* Render pass structure */
typedef struct {
  WGPURenderPipeline pipeline;
  WGPUBindGroup bind_group;
  WGPUBuffer uniform_buf;
} pass_t;

/* Example state */
static struct {
  /* Entities */
  entity_t entities[NUM_CUBES + 1]; // 1 plane + 4 cubes
  uint32_t entity_count;

  /* Lights */
  light_t lights[2];
  uint32_t light_count;
  WGPUBool lights_are_dirty;

  /* Buffers */
  WGPUBuffer entity_uniform_buf;
  WGPUBuffer light_storage_buf;

  /* Passes */
  pass_t shadow_pass;
  pass_t forward_pass;

  /* Bind groups */
  WGPUBindGroup entity_bind_group;
  WGPUBindGroupLayout local_bind_group_layout;

  /* Depth and shadow textures */
  WGPUTextureView forward_depth;
  WGPUTexture shadow_texture;
  WGPUTextureView shadow_view;
  WGPUSampler shadow_sampler;

  /* Geometry buffers */
  WGPUBuffer cube_vertex_buf;
  WGPUBuffer cube_index_buf;
  uint32_t cube_index_count;
  WGPUBuffer plane_vertex_buf;
  WGPUBuffer plane_index_buf;
  uint32_t plane_index_count;

  /* Uniform alignment */
  uint32_t uniform_alignment;

  /* Render pass */
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } render_pass;

  WGPUBool initialized;
  WGPUBool supports_storage_resources;
} state = {
  .entity_count = 0,
  .light_count = 2,
  .lights_are_dirty = true,
  .render_pass = {
    .color_attachment = {
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.1, 0.2, 0.3, 1.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    },
    .depth_stencil_attachment = {
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Discard,
      .depthClearValue   = 1.0f,
    },
    .descriptor = {
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.render_pass.color_attachment,
      .depthStencilAttachment = &state.render_pass.depth_stencil_attachment,
    },
  }
};

/* Helper: create vertex */
static vertex_t vertex(int8_t x, int8_t y, int8_t z, int8_t nx, int8_t ny,
                       int8_t nz)
{
  vertex_t v;
  v.pos[0]    = x;
  v.pos[1]    = y;
  v.pos[2]    = z;
  v.pos[3]    = 1;
  v.normal[0] = nx;
  v.normal[1] = ny;
  v.normal[2] = nz;
  v.normal[3] = 0;
  return v;
}

/* Create cube geometry */
static void create_cube(vertex_t** vertices, uint32_t* vertex_count,
                        uint16_t** indices, uint32_t* index_count)
{
  static vertex_t cube_vertices[24] = {0};
  static uint16_t cube_indices[36]  = {0};
  static WGPUBool initialized       = false;

  if (!initialized) {
    /* Top (0, 0, 1) */
    cube_vertices[0] = vertex(-1, -1, 1, 0, 0, 1);
    cube_vertices[1] = vertex(1, -1, 1, 0, 0, 1);
    cube_vertices[2] = vertex(1, 1, 1, 0, 0, 1);
    cube_vertices[3] = vertex(-1, 1, 1, 0, 0, 1);
    /* Bottom (0, 0, -1) */
    cube_vertices[4] = vertex(-1, 1, -1, 0, 0, -1);
    cube_vertices[5] = vertex(1, 1, -1, 0, 0, -1);
    cube_vertices[6] = vertex(1, -1, -1, 0, 0, -1);
    cube_vertices[7] = vertex(-1, -1, -1, 0, 0, -1);
    /* Right (1, 0, 0) */
    cube_vertices[8]  = vertex(1, -1, -1, 1, 0, 0);
    cube_vertices[9]  = vertex(1, 1, -1, 1, 0, 0);
    cube_vertices[10] = vertex(1, 1, 1, 1, 0, 0);
    cube_vertices[11] = vertex(1, -1, 1, 1, 0, 0);
    /* Left (-1, 0, 0) */
    cube_vertices[12] = vertex(-1, -1, 1, -1, 0, 0);
    cube_vertices[13] = vertex(-1, 1, 1, -1, 0, 0);
    cube_vertices[14] = vertex(-1, 1, -1, -1, 0, 0);
    cube_vertices[15] = vertex(-1, -1, -1, -1, 0, 0);
    /* Front (0, 1, 0) */
    cube_vertices[16] = vertex(1, 1, -1, 0, 1, 0);
    cube_vertices[17] = vertex(-1, 1, -1, 0, 1, 0);
    cube_vertices[18] = vertex(-1, 1, 1, 0, 1, 0);
    cube_vertices[19] = vertex(1, 1, 1, 0, 1, 0);
    /* Back (0, -1, 0) */
    cube_vertices[20] = vertex(1, -1, 1, 0, -1, 0);
    cube_vertices[21] = vertex(-1, -1, 1, 0, -1, 0);
    cube_vertices[22] = vertex(-1, -1, -1, 0, -1, 0);
    cube_vertices[23] = vertex(1, -1, -1, 0, -1, 0);

    /* Indices */
    uint16_t idx_data[36] = {
      0,  1,  2,  2,  3,  0,  /* top */
      4,  5,  6,  6,  7,  4,  /* bottom */
      8,  9,  10, 10, 11, 8,  /* right */
      12, 13, 14, 14, 15, 12, /* left */
      16, 17, 18, 18, 19, 16, /* front */
      20, 21, 22, 22, 23, 20  /* back */
    };
    memcpy(cube_indices, idx_data, sizeof(idx_data));
    initialized = true;
  }

  *vertices     = cube_vertices;
  *vertex_count = 24;
  *indices      = cube_indices;
  *index_count  = 36;
}

/* Create plane geometry */
static void create_plane(int8_t size, vertex_t** vertices,
                         uint32_t* vertex_count, uint16_t** indices,
                         uint32_t* index_count)
{
  static vertex_t plane_vertices[4] = {0};
  static uint16_t plane_indices[6]  = {0};
  static WGPUBool initialized       = false;

  if (!initialized || plane_vertices[0].pos[0] != size) {
    plane_vertices[0] = vertex(size, -size, 0, 0, 0, 1);
    plane_vertices[1] = vertex(size, size, 0, 0, 0, 1);
    plane_vertices[2] = vertex(-size, -size, 0, 0, 0, 1);
    plane_vertices[3] = vertex(-size, size, 0, 0, 0, 1);

    plane_indices[0] = 0;
    plane_indices[1] = 1;
    plane_indices[2] = 2;
    plane_indices[3] = 2;
    plane_indices[4] = 1;
    plane_indices[5] = 3;
    initialized      = true;
  }

  *vertices     = plane_vertices;
  *vertex_count = 4;
  *indices      = plane_indices;
  *index_count  = 6;
}

/* Generate view-projection matrix */
static void generate_matrix(float aspect_ratio, mat4 dest)
{
  mat4 projection, view;
  glm_perspective(GLM_PI_4f, aspect_ratio, 1.0f, 20.0f, projection);
  glm_lookat((vec3){3.0f, -10.0f, 6.0f}, (vec3){0.0f, 0.0f, 0.0f},
             (vec3){0.0f, 0.0f, 1.0f}, view);
  glm_mat4_mul(projection, view, dest);
}

/* Create depth texture */
static WGPUTextureView create_depth_texture(wgpu_context_t* wgpu_context)
{
  WGPUTexture depth_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Shadow - Depth texture"),
      .size = (WGPUExtent3D){
        .width = wgpu_context->width,
        .height = wgpu_context->height,
        .depthOrArrayLayers = 1,
      },
      .mipLevelCount = 1,
      .sampleCount = 1,
      .dimension = WGPUTextureDimension_2D,
      .format = WGPUTextureFormat_Depth32Float,
      .usage = WGPUTextureUsage_RenderAttachment,
    });

  WGPUTextureView view = wgpuTextureCreateView(
    depth_texture, &(WGPUTextureViewDescriptor){
                     .format          = WGPUTextureFormat_Depth32Float,
                     .dimension       = WGPUTextureViewDimension_2D,
                     .baseMipLevel    = 0,
                     .mipLevelCount   = 1,
                     .baseArrayLayer  = 0,
                     .arrayLayerCount = 1,
                     .aspect          = WGPUTextureAspect_All,
                   });

  wgpuTextureRelease(depth_texture);
  return view;
}

/* Convert light to raw format for GPU */
static void light_to_raw(const light_t* light, light_raw_t* raw)
{
  mat4 view, projection, view_proj;
  vec3 light_pos = {light->pos[0], light->pos[1], light->pos[2]};

  glm_lookat(light_pos, (vec3){0.0f, 0.0f, 0.0f}, (vec3){0.0f, 0.0f, 1.0f},
             view);
  glm_perspective(glm_rad(light->fov), 1.0f, light->depth_near,
                  light->depth_far, projection);
  glm_mat4_mul(projection, view, view_proj);

  glm_mat4_copy(view_proj, raw->proj);
  raw->pos[0]   = light->pos[0];
  raw->pos[1]   = light->pos[1];
  raw->pos[2]   = light->pos[2];
  raw->pos[3]   = 1.0f;
  raw->color[0] = light->color[0];
  raw->color[1] = light->color[1];
  raw->color[2] = light->color[2];
  raw->color[3] = 1.0f;
}

/* Initialize geometry buffers */
static void init_geometry_buffers(wgpu_context_t* wgpu_context)
{
  /* Create cube buffers */
  vertex_t* cube_vertices;
  uint16_t* cube_indices;
  uint32_t cube_vertex_count;
  create_cube(&cube_vertices, &cube_vertex_count, &cube_indices,
              &state.cube_index_count);

  state.cube_vertex_buf = wgpu_create_buffer_from_data(
    wgpu_context, cube_vertices, cube_vertex_count * sizeof(vertex_t),
    WGPUBufferUsage_Vertex);

  state.cube_index_buf = wgpu_create_buffer_from_data(
    wgpu_context, cube_indices, state.cube_index_count * sizeof(uint16_t),
    WGPUBufferUsage_Index);

  /* Create plane buffers */
  vertex_t* plane_vertices;
  uint16_t* plane_indices;
  uint32_t plane_vertex_count;
  create_plane(7, &plane_vertices, &plane_vertex_count, &plane_indices,
               &state.plane_index_count);

  state.plane_vertex_buf = wgpu_create_buffer_from_data(
    wgpu_context, plane_vertices, plane_vertex_count * sizeof(vertex_t),
    WGPUBufferUsage_Vertex);

  state.plane_index_buf = wgpu_create_buffer_from_data(
    wgpu_context, plane_indices, state.plane_index_count * sizeof(uint16_t),
    WGPUBufferUsage_Index);
}

/* Initialize entities */
static void init_entities(wgpu_context_t* wgpu_context)
{
  /* Calculate uniform alignment */
  /* Get device limits (not adapter limits) for
   * min_uniform_buffer_offset_alignment */
  WGPULimits limits = {0};
  wgpuDeviceGetLimits(wgpu_context->device, &limits);

  uint32_t entity_uniform_size = sizeof(entity_uniforms_t);
  uint32_t min_alignment       = limits.minUniformBufferOffsetAlignment;

  /* Align entity uniform size to minUniformBufferOffsetAlignment */
  /* Each uniform must be aligned to minUniformBufferOffsetAlignment boundary */
  /* Calculate: round up entity_uniform_size to next multiple of min_alignment
   */
  if (entity_uniform_size > min_alignment) {
    state.uniform_alignment
      = ((entity_uniform_size + min_alignment - 1) / min_alignment)
        * min_alignment;
  }
  else {
    state.uniform_alignment = min_alignment;
  }

  /* Create entity uniform buffer */
  uint32_t num_entities    = NUM_CUBES + 1;
  state.entity_uniform_buf = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Shadow - Entity uniform buffer"),
      .size             = num_entities * state.uniform_alignment,
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .mappedAtCreation = false,
    });

  /* Plane entity */
  state.entities[0] = (entity_t){
    .rotation_speed = 0.0f,
    .color          = {1.0f, 1.0f, 1.0f, 1.0f},
    .vertex_buf     = state.plane_vertex_buf,
    .index_buf      = state.plane_index_buf,
    .index_count    = state.plane_index_count,
    .uniform_offset = 0,
  };
  glm_mat4_identity(state.entities[0].mx_world);
  state.entity_count = 1;

  /* Cube entities */
  struct {
    vec3 offset;
    float angle;
    float scale;
    float rotation;
  } cube_descs[NUM_CUBES] = {
    {{-2.0f, -2.0f, 2.0f}, 10.0f, 0.7f, 0.1f},
    {{2.0f, -2.0f, 2.0f}, 50.0f, 1.3f, 0.2f},
    {{-2.0f, 2.0f, 2.0f}, 140.0f, 1.1f, 0.3f},
    {{2.0f, 2.0f, 2.0f}, 210.0f, 0.9f, 0.4f},
  };

  for (uint32_t i = 0; i < NUM_CUBES; ++i) {
    entity_t* e = &state.entities[state.entity_count];

    vec3 axis;
    glm_vec3_normalize_to(cube_descs[i].offset, axis);

    mat4 scale_mat, rot_mat, trans_mat;
    glm_scale_make(scale_mat, (vec3){cube_descs[i].scale, cube_descs[i].scale,
                                     cube_descs[i].scale});
    glm_rotate_make(rot_mat, glm_rad(cube_descs[i].angle), axis);
    glm_translate_make(trans_mat, cube_descs[i].offset);

    glm_mat4_mul(rot_mat, scale_mat, e->mx_world);
    glm_mat4_mul(trans_mat, e->mx_world, e->mx_world);

    e->rotation_speed = cube_descs[i].rotation;
    e->color[0]       = 0.0f;
    e->color[1]       = 1.0f;
    e->color[2]       = 0.0f;
    e->color[3]       = 1.0f;
    e->vertex_buf     = state.cube_vertex_buf;
    e->index_buf      = state.cube_index_buf;
    e->index_count    = state.cube_index_count;
    e->uniform_offset = state.entity_count * state.uniform_alignment;

    state.entity_count++;
  }
}

/* Initialize shadow texture and sampler */
static void init_shadow_resources(wgpu_context_t* wgpu_context)
{
  /* Shadow texture */
  state.shadow_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Shadow - Shadow texture array"),
      .size = (WGPUExtent3D){
        .width = SHADOW_SIZE,
        .height = SHADOW_SIZE,
        .depthOrArrayLayers = MAX_LIGHTS,
      },
      .mipLevelCount = 1,
      .sampleCount = 1,
      .dimension = WGPUTextureDimension_2D,
      .format = WGPUTextureFormat_Depth32Float,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    });

  state.shadow_view = wgpuTextureCreateView(
    state.shadow_texture, &(WGPUTextureViewDescriptor){
                            .format          = WGPUTextureFormat_Depth32Float,
                            .dimension       = WGPUTextureViewDimension_2DArray,
                            .baseMipLevel    = 0,
                            .mipLevelCount   = 1,
                            .baseArrayLayer  = 0,
                            .arrayLayerCount = MAX_LIGHTS,
                            .aspect          = WGPUTextureAspect_All,
                          });

  /* Shadow sampler */
  state.shadow_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Shadow - Shadow sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                            .compare       = WGPUCompareFunction_LessEqual,
                            .maxAnisotropy = 1,
                          });

  /* Initialize lights */
  for (uint32_t i = 0; i < state.light_count; ++i) {
    state.lights[i].target_view = wgpuTextureCreateView(
      state.shadow_texture, &(WGPUTextureViewDescriptor){
                              .format          = WGPUTextureFormat_Depth32Float,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = i,
                              .arrayLayerCount = 1,
                              .aspect          = WGPUTextureAspect_All,
                            });
  }

  /* Light 0 */
  state.lights[0].pos[0]     = 7.0f;
  state.lights[0].pos[1]     = -5.0f;
  state.lights[0].pos[2]     = 10.0f;
  state.lights[0].color[0]   = 0.5f;
  state.lights[0].color[1]   = 1.0f;
  state.lights[0].color[2]   = 0.5f;
  state.lights[0].color[3]   = 1.0f;
  state.lights[0].fov        = 60.0f;
  state.lights[0].depth_near = 1.0f;
  state.lights[0].depth_far  = 20.0f;

  /* Light 1 */
  state.lights[1].pos[0]     = -5.0f;
  state.lights[1].pos[1]     = 7.0f;
  state.lights[1].pos[2]     = 10.0f;
  state.lights[1].color[0]   = 1.0f;
  state.lights[1].color[1]   = 0.5f;
  state.lights[1].color[2]   = 0.5f;
  state.lights[1].color[3]   = 1.0f;
  state.lights[1].fov        = 45.0f;
  state.lights[1].depth_near = 1.0f;
  state.lights[1].depth_far  = 20.0f;

  /* Light storage buffer */
  uint32_t light_storage_size = MAX_LIGHTS * sizeof(light_raw_t);
  state.light_storage_buf     = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
          .label = STRVIEW("Shadow - Light storage buffer"),
          .size  = light_storage_size,
          .usage = (state.supports_storage_resources ? WGPUBufferUsage_Storage :
                                                       WGPUBufferUsage_Uniform)
               | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
          .mappedAtCreation = false,
    });
}

/* Initialize shadow pass */
static void init_shadow_pass(wgpu_context_t* wgpu_context,
                             WGPUShaderModule shader)
{
  /* Create bind group layout */
  WGPUBindGroupLayout bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label = STRVIEW("Shadow - Shadow pass bind group layout"),
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry){
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = sizeof(global_uniforms_t),
        },
      },
    });

  /* Pipeline layout */
  WGPUBindGroupLayout layouts[2]
    = {bind_group_layout, state.local_bind_group_layout};
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Shadow - Shadow pass pipeline layout"),
      .bindGroupLayoutCount = 2,
      .bindGroupLayouts     = layouts,
    });

  /* Uniform buffer */
  state.shadow_pass.uniform_buf = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Shadow - Shadow pass uniform buffer"),
      .size             = sizeof(global_uniforms_t),
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .mappedAtCreation = false,
    });

  /* Bind group */
  state.shadow_pass.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label = STRVIEW("Shadow - Shadow pass bind group"),
      .layout = bind_group_layout,
      .entryCount = 1,
      .entries = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer = state.shadow_pass.uniform_buf,
        .offset = 0,
        .size = sizeof(global_uniforms_t),
      },
    });

  /* Vertex buffer layout */
  WGPUVertexAttribute vertex_attrs[2] = {
    {
      .format         = WGPUVertexFormat_Sint8x4,
      .offset         = 0,
      .shaderLocation = 0,
    },
    {
      .format         = WGPUVertexFormat_Sint8x4,
      .offset         = 4,
      .shaderLocation = 1,
    },
  };
  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = sizeof(vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 2,
    .attributes     = vertex_attrs,
  };

  /* Render pipeline */
  state.shadow_pass.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("Shadow - Shadow pass pipeline"),
      .layout = pipeline_layout,
      .vertex = (WGPUVertexState){
        .module = shader,
        .entryPoint = STRVIEW("vs_bake"),
        .bufferCount = 1,
        .buffers = &vertex_buffer_layout,
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode = WGPUCullMode_Back,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = WGPUTextureFormat_Depth32Float,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_LessEqual,
        .stencilFront = (WGPUStencilFaceState){
          .compare = WGPUCompareFunction_Always,
          .failOp = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp = WGPUStencilOperation_Keep,
        },
        .stencilBack = (WGPUStencilFaceState){
          .compare = WGPUCompareFunction_Always,
          .failOp = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp = WGPUStencilOperation_Keep,
        },
        .depthBias = 2,
        .depthBiasSlopeScale = 2.0f,
        .depthBiasClamp = 0.0f,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = 0xFFFFFFFF,
        .alphaToCoverageEnabled = false,
      },
      .fragment = NULL,
    });

  wgpuPipelineLayoutRelease(pipeline_layout);
  wgpuBindGroupLayoutRelease(bind_group_layout);
}

/* Initialize forward pass */
static void init_forward_pass(wgpu_context_t* wgpu_context,
                              WGPUShaderModule shader)
{
  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    {
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout){
        .type = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize = sizeof(global_uniforms_t),
      },
    },
    {
      .binding = 1,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout){
        .type = state.supports_storage_resources ? WGPUBufferBindingType_ReadOnlyStorage : WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize = MAX_LIGHTS * sizeof(light_raw_t),
      },
    },
    {
      .binding = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType = WGPUTextureSampleType_Depth,
        .viewDimension = WGPUTextureViewDimension_2DArray,
        .multisampled = false,
      },
    },
    {
      .binding = 3,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Comparison,
      },
    },
  };

  WGPUBindGroupLayout bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Shadow - Forward pass bind group layout"),
      .entryCount = 4,
      .entries    = bgl_entries,
    });

  /* Pipeline layout */
  WGPUBindGroupLayout layouts[2]
    = {bind_group_layout, state.local_bind_group_layout};
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Shadow - Forward pass pipeline layout"),
      .bindGroupLayoutCount = 2,
      .bindGroupLayouts     = layouts,
    });

  /* Uniform buffer */
  mat4 mx_total;
  generate_matrix((float)wgpu_context->width / (float)wgpu_context->height,
                  mx_total);
  global_uniforms_t forward_uniforms = {0};
  glm_mat4_copy(mx_total, forward_uniforms.view_proj);
  forward_uniforms.num_lights[0] = state.light_count;

  state.forward_pass.uniform_buf = wgpu_create_buffer_from_data(
    wgpu_context, &forward_uniforms, sizeof(global_uniforms_t),
    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[4] = {
    {
      .binding = 0,
      .buffer  = state.forward_pass.uniform_buf,
      .offset  = 0,
      .size    = sizeof(global_uniforms_t),
    },
    {
      .binding = 1,
      .buffer  = state.light_storage_buf,
      .offset  = 0,
      .size    = MAX_LIGHTS * sizeof(light_raw_t),
    },
    {
      .binding     = 2,
      .textureView = state.shadow_view,
    },
    {
      .binding = 3,
      .sampler = state.shadow_sampler,
    },
  };

  state.forward_pass.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Shadow - Forward pass bind group"),
      .layout     = bind_group_layout,
      .entryCount = 4,
      .entries    = bg_entries,
    });

  /* Vertex buffer layout */
  WGPUVertexAttribute vertex_attrs[2] = {
    {
      .format         = WGPUVertexFormat_Sint8x4,
      .offset         = 0,
      .shaderLocation = 0,
    },
    {
      .format         = WGPUVertexFormat_Sint8x4,
      .offset         = 4,
      .shaderLocation = 1,
    },
  };
  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = sizeof(vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 2,
    .attributes     = vertex_attrs,
  };

  /* Render pipeline */
  const char* fs_entry
    = state.supports_storage_resources ? "fs_main" : "fs_main_without_storage";

  state.forward_pass.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label = STRVIEW("Shadow - Forward pass pipeline"),
      .layout = pipeline_layout,
      .vertex = (WGPUVertexState){
        .module = shader,
        .entryPoint = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers = &vertex_buffer_layout,
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode = WGPUCullMode_Back,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format = WGPUTextureFormat_Depth32Float,
        .depthWriteEnabled = true,
        .depthCompare = WGPUCompareFunction_Less,
        .stencilFront = (WGPUStencilFaceState){
          .compare = WGPUCompareFunction_Always,
          .failOp = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp = WGPUStencilOperation_Keep,
        },
        .stencilBack = (WGPUStencilFaceState){
          .compare = WGPUCompareFunction_Always,
          .failOp = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp = WGPUStencilOperation_Keep,
        },
        .depthBias = 0,
        .depthBiasSlopeScale = 0.0f,
        .depthBiasClamp = 0.0f,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask = 0xFFFFFFFF,
        .alphaToCoverageEnabled = false,
      },
      .fragment = &(WGPUFragmentState){
        .module = shader,
        .entryPoint = STRVIEW(fs_entry),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format = wgpu_context->render_format,
          .blend = &(WGPUBlendState){
            .color = (WGPUBlendComponent){
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_Zero,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = (WGPUBlendComponent){
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_Zero,
              .operation = WGPUBlendOperation_Add,
            },
          },
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    });

  wgpuPipelineLayoutRelease(pipeline_layout);
  wgpuBindGroupLayoutRelease(bind_group_layout);
}

/* Initialize bind groups */
static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Local bind group layout (for entity uniforms) */
  state.local_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label = STRVIEW("Shadow - Local bind group layout"),
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry){
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize = sizeof(entity_uniforms_t),
        },
      },
    });

  /* Entity bind group */
  state.entity_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label = STRVIEW("Shadow - Entity bind group"),
      .layout = state.local_bind_group_layout,
      .entryCount = 1,
      .entries = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer = state.entity_uniform_buf,
        .offset = 0,
        .size = sizeof(entity_uniforms_t),
      },
    });
}

/* Initialization */
static int init(wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  /* Check for storage buffer support */
  WGPULimits limits = {0};
  wgpuDeviceGetLimits(wgpu_context->device, &limits);
  state.supports_storage_resources = limits.maxStorageBuffersPerShaderStage > 0;

  /* Initialize geometry */
  init_geometry_buffers(wgpu_context);
  init_entities(wgpu_context);

  /* Initialize shadow resources */
  init_shadow_resources(wgpu_context);

  /* Create depth texture */
  state.forward_depth = create_depth_texture(wgpu_context);

  /* Create shader module */
  WGPUShaderModule shader
    = wgpu_create_shader_module(wgpu_context->device, shadow_shader_wgsl);

  /* Initialize bind groups */
  init_bind_groups(wgpu_context);

  /* Initialize passes */
  init_shadow_pass(wgpu_context, shader);
  init_forward_pass(wgpu_context, shader);

  wgpuShaderModuleRelease(shader);

  state.initialized = true;
  return EXIT_SUCCESS;
}

/* Resize handler */
static void resize(wgpu_context_t* wgpu_context)
{
  /* Update view-projection matrix */
  mat4 mx_total;
  generate_matrix((float)wgpu_context->width / (float)wgpu_context->height,
                  mx_total);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.forward_pass.uniform_buf, 0,
                       mx_total, sizeof(mat4));

  /* Recreate forward depth texture */
  if (state.forward_depth) {
    wgpuTextureViewRelease(state.forward_depth);
  }
  state.forward_depth = create_depth_texture(wgpu_context);
}

/* Input event callback */
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    resize(wgpu_context);
  }
}

/* Frame rendering */
static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update entity uniforms */
  for (uint32_t i = 0; i < state.entity_count; ++i) {
    entity_t* entity = &state.entities[i];

    if (entity->rotation_speed != 0.0f) {
      mat4 rotation;
      glm_rotate_make(rotation, glm_rad(entity->rotation_speed),
                      (vec3){1.0f, 0.0f, 0.0f});
      glm_mat4_mul(entity->mx_world, rotation, entity->mx_world);
    }

    entity_uniforms_t data;
    glm_mat4_copy(entity->mx_world, data.model);
    glm_vec4_copy(entity->color, data.color);

    wgpuQueueWriteBuffer(wgpu_context->queue, state.entity_uniform_buf,
                         entity->uniform_offset, &data,
                         sizeof(entity_uniforms_t));
  }

  /* Update light uniforms */
  if (state.lights_are_dirty) {
    state.lights_are_dirty = false;
    for (uint32_t i = 0; i < state.light_count; ++i) {
      light_raw_t light_raw;
      light_to_raw(&state.lights[i], &light_raw);
      wgpuQueueWriteBuffer(wgpu_context->queue, state.light_storage_buf,
                           i * sizeof(light_raw_t), &light_raw,
                           sizeof(light_raw_t));
    }
  }

  /* Create command encoder */
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("Shadow - Command encoder"),
                          });

  /* Shadow passes */
  for (uint32_t i = 0; i < state.light_count; ++i) {
    /* Copy light projection to shadow uniform buffer */
    wgpuCommandEncoderCopyBufferToBuffer(encoder, state.light_storage_buf,
                                         i * sizeof(light_raw_t),
                                         state.shadow_pass.uniform_buf, 0, 64);

    /* Render pass */
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      encoder,
      &(WGPURenderPassDescriptor){
        .label = STRVIEW("Shadow - Shadow pass"),
        .colorAttachmentCount = 0,
        .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
          .view = state.lights[i].target_view,
          .depthLoadOp = WGPULoadOp_Clear,
          .depthStoreOp = WGPUStoreOp_Store,
          .depthClearValue = 1.0f,
        },
      });

    wgpuRenderPassEncoderSetPipeline(pass, state.shadow_pass.pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.shadow_pass.bind_group, 0,
                                      NULL);

    for (uint32_t j = 0; j < state.entity_count; ++j) {
      entity_t* entity = &state.entities[j];
      wgpuRenderPassEncoderSetBindGroup(pass, 1, state.entity_bind_group, 1,
                                        &entity->uniform_offset);
      wgpuRenderPassEncoderSetVertexBuffer(pass, 0, entity->vertex_buf, 0,
                                           WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetIndexBuffer(
        pass, entity->index_buf, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDrawIndexed(pass, entity->index_count, 1, 0, 0, 0);
    }

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* Forward pass */
  state.render_pass.color_attachment.view = wgpu_context->swapchain_view;
  state.render_pass.depth_stencil_attachment.view = state.forward_depth;

  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(encoder, &state.render_pass.descriptor);

  wgpuRenderPassEncoderSetPipeline(pass, state.forward_pass.pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.forward_pass.bind_group, 0,
                                    NULL);

  for (uint32_t i = 0; i < state.entity_count; ++i) {
    entity_t* entity = &state.entities[i];
    wgpuRenderPassEncoderSetBindGroup(pass, 1, state.entity_bind_group, 1,
                                      &entity->uniform_offset);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, entity->vertex_buf, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, entity->index_buf, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(pass, entity->index_count, 1, 0, 0, 0);
  }

  wgpuRenderPassEncoderEnd(pass);
  wgpuRenderPassEncoderRelease(pass);

  /* Submit */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(encoder);

  return EXIT_SUCCESS;
}

/* Cleanup */
static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  WGPU_RELEASE_RESOURCE(Buffer, state.cube_vertex_buf)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_index_buf)
  WGPU_RELEASE_RESOURCE(Buffer, state.plane_vertex_buf)
  WGPU_RELEASE_RESOURCE(Buffer, state.plane_index_buf)
  WGPU_RELEASE_RESOURCE(Buffer, state.entity_uniform_buf)
  WGPU_RELEASE_RESOURCE(Buffer, state.light_storage_buf)
  WGPU_RELEASE_RESOURCE(Buffer, state.shadow_pass.uniform_buf)
  WGPU_RELEASE_RESOURCE(Buffer, state.forward_pass.uniform_buf)
  WGPU_RELEASE_RESOURCE(BindGroup, state.shadow_pass.bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.forward_pass.bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.entity_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.local_bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.shadow_pass.pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.forward_pass.pipeline)
  WGPU_RELEASE_RESOURCE(Texture, state.shadow_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.shadow_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.forward_depth)
  WGPU_RELEASE_RESOURCE(Sampler, state.shadow_sampler)

  for (uint32_t i = 0; i < state.light_count; ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, state.lights[i].target_view)
  }
}

/* Main entry point */
int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Shadow Mapping",
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
static const char* shadow_shader_wgsl = CODE(
  struct Globals {
    view_proj: mat4x4<f32>,
    num_lights: vec4<u32>,
  };

  @group(0)
  @binding(0)
  var<uniform> u_globals: Globals;

  struct Entity {
    world: mat4x4<f32>,
    color: vec4<f32>,
  };

  @group(1)
  @binding(0)
  var<uniform> u_entity: Entity;

  @vertex
  fn vs_bake(@location(0) position: vec4<i32>) -> @builtin(position) vec4<f32> {
    return u_globals.view_proj * u_entity.world * vec4<f32>(position);
  }

  struct VertexOutput {
    @builtin(position) proj_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec4<f32>
  };

  @vertex
  fn vs_main(
    @location(0) position: vec4<i32>,
    @location(1) normal: vec4<i32>,
  ) -> VertexOutput {
    let w = u_entity.world;
    let world_pos = u_entity.world * vec4<f32>(position);
    var result: VertexOutput;
    result.world_normal = mat3x3<f32>(w[0].xyz, w[1].xyz, w[2].xyz) * vec3<f32>(normal.xyz);
    result.world_position = world_pos;
    result.proj_position = u_globals.view_proj * world_pos;
    return result;
  }

  // fragment shader

  struct Light {
    proj: mat4x4<f32>,
    pos: vec4<f32>,
    color: vec4<f32>,
  };

  @group(0)
  @binding(1)
  var<storage, read> s_lights: array<Light>;
  @group(0)
  @binding(1)
  var<uniform> u_lights: array<Light, 10>; // Used when storage types are not supported
  @group(0)
  @binding(2)
  var t_shadow: texture_depth_2d_array;
  @group(0)
  @binding(3)
  var sampler_shadow: sampler_comparison;

  fn fetch_shadow(light_id: u32, homogeneous_coords: vec4<f32>) -> f32 {
    if (homogeneous_coords.w <= 0.0) {
      return 1.0;
    }
    // compensate for the Y-flip difference between the NDC and texture coordinates
    let flip_correction = vec2<f32>(0.5, -0.5);
    // compute texture coordinates for shadow lookup
    let proj_correction = 1.0 / homogeneous_coords.w;
    let light_local = homogeneous_coords.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
    // do the lookup, using HW PCF and comparison
    return textureSampleCompareLevel(t_shadow, sampler_shadow, light_local, i32(light_id), homogeneous_coords.z * proj_correction);
  }

  const c_ambient: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
  const c_max_lights: u32 = 10u;

  @fragment
  fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(vertex.world_normal);
    // accumulate color
    var color: vec3<f32> = c_ambient;
    for(var i = 0u; i < min(u_globals.num_lights.x, c_max_lights); i += 1u) {
      let light = s_lights[i];
      // project into the light space
      let shadow = fetch_shadow(i, light.proj * vertex.world_position);
      // compute Lambertian diffuse term
      let light_dir = normalize(light.pos.xyz - vertex.world_position.xyz);
      let diffuse = max(0.0, dot(normal, light_dir));
      // add light contribution
      color += shadow * diffuse * light.color.xyz;
    }
    // multiply the light by material color
    return vec4<f32>(color, 1.0) * u_entity.color;
  }

  // The fragment entrypoint used when storage buffers are not available for the lights
  @fragment
  fn fs_main_without_storage(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(vertex.world_normal);
    var color: vec3<f32> = c_ambient;
    for(var i = 0u; i < min(u_globals.num_lights.x, c_max_lights); i += 1u) {
      // This line is the only difference from the entrypoint above. It uses the lights
      // uniform instead of the lights storage buffer
      let light = u_lights[i];
      let shadow = fetch_shadow(i, light.proj * vertex.world_position);
      let light_dir = normalize(light.pos.xyz - vertex.world_position.xyz);
      let diffuse = max(0.0, dot(normal, light_dir));
      color += shadow * diffuse * light.color.xyz;
    }
    return vec4<f32>(color, 1.0) * u_entity.color;
  }
);
// clang-format on
