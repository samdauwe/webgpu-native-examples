#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <string.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Stencil Mask
 *
 * This example demonstrates using the stencil buffer for masking. It draws
 * the 6 faces of a rotating cube into the stencil buffer, each with a
 * different stencil value. Then it draws different scenes of animated objects
 * where the stencil value matches, creating a cube-shaped window into
 * different worlds.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/stencilMask
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* simple_lighting_shader_wgsl = CODE(
  struct Uniforms {
    world: mat4x4f,
    color: vec4f,
  };

  struct SharedUniforms {
    viewProjection: mat4x4f,
    lightDirection: vec3f,
  };

  @group(0) @binding(0) var<uniform> uni: Uniforms;
  @group(0) @binding(1) var<uniform> sharedUni: SharedUniforms;

  struct MyVSInput {
    @location(0) position: vec4f,
    @location(1) normal: vec3f,
    @location(2) texcoord: vec2f,
  };

  struct MyVSOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) texcoord: vec2f,
  };

  @vertex
  fn myVSMain(v: MyVSInput) -> MyVSOutput {
    var vsOut: MyVSOutput;
    vsOut.position = sharedUni.viewProjection * uni.world * v.position;
    vsOut.normal = (uni.world * vec4f(v.normal, 0.0)).xyz;
    vsOut.texcoord = v.texcoord;
    return vsOut;
  }

  @fragment
  fn myFSMain(v: MyVSOutput) -> @location(0) vec4f {
    let diffuseColor = uni.color;
    let a_normal = normalize(v.normal);
    let l = dot(a_normal, sharedUni.lightDirection) * 0.5 + 0.5;
    return vec4f(diffuseColor.rgb * l, diffuseColor.a);
  }
);
// clang-format on

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define NUM_MASK_SCENES (6)
#define NUM_OBJECT_SCENES (7)
#define NUM_INSTANCES_PER_SCENE (100)
#define DEPTH_FORMAT WGPUTextureFormat_Depth24PlusStencil8

/* -------------------------------------------------------------------------- *
 * Types
 * -------------------------------------------------------------------------- */

/* Represents geometry like a cube, a sphere, a torus */
typedef struct geometry_t {
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  WGPUIndexFormat index_format;
  uint32_t num_vertices;
} geometry_t;

/* Per object data */
typedef struct object_info_t {
  float uniform_values[16 + 4]; /* mat4x4f + vec4f */
  WGPUBuffer uniform_buffer;
  WGPUBindGroup bind_group;
  uint32_t geometry_index;
} object_info_t;

/* Per scene data */
typedef struct scene_t {
  object_info_t* object_infos;
  uint32_t num_objects;
  WGPUBuffer shared_uniform_buffer;
  float shared_uniform_values[16 + 4]; /* mat4x4f + vec3f + padding */
} scene_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Mouse position (-1 to 1 normalized) */
  struct {
    float x;
    float y;
  } mouse_pos;

  /* Geometries */
  geometry_t plane_geo;
  geometry_t sphere_geo;
  geometry_t torus_geo;
  geometry_t cube_geo;
  geometry_t cone_geo;
  geometry_t cylinder_geo;
  geometry_t jem_geo;  /* Faceted sphere */
  geometry_t dice_geo; /* Faceted torus */

  /* All geometries for easy access */
  geometry_t* geometries[8];

  /* Scenes */
  scene_t mask_scenes[NUM_MASK_SCENES];
  scene_t object_scenes[NUM_OBJECT_SCENES];

  /* Pipelines */
  WGPURenderPipeline stencil_set_pipeline;
  WGPURenderPipeline stencil_mask_pipeline;
  WGPUBindGroupLayout bind_group_layout;
  WGPUPipelineLayout pipeline_layout;

  /* Depth/stencil texture */
  WGPUTexture depth_texture;
  WGPUTextureView depth_texture_view;
  uint32_t depth_texture_width;
  uint32_t depth_texture_height;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment clear_color_attachment;
  WGPURenderPassDepthStencilAttachment clear_depth_attachment;
  WGPURenderPassDescriptor clear_pass_desc;

  WGPURenderPassColorAttachment load_color_attachment;
  WGPURenderPassDepthStencilAttachment load_depth_attachment;
  WGPURenderPassDescriptor load_pass_desc;

  /* Timing */
  uint64_t last_time;

  WGPUBool initialized;
} state = {
  .mouse_pos = {0.0f, 0.0f},
  .clear_color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.2, 0.2, 0.2, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .clear_depth_attachment = {
    .depthClearValue   = 1.0f,
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .stencilClearValue = 0,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
  },
  .load_color_attachment = {
    .loadOp     = WGPULoadOp_Load,
    .storeOp    = WGPUStoreOp_Store,
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .load_depth_attachment = {
    .depthClearValue   = 1.0f,
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .stencilLoadOp     = WGPULoadOp_Load,
    .stencilStoreOp    = WGPUStoreOp_Store,
  },
};

/* -------------------------------------------------------------------------- *
 * Helper functions
 * -------------------------------------------------------------------------- */

static float randf(float min_val, float max_val)
{
  return min_val + ((float)rand() / (float)RAND_MAX) * (max_val - min_val);
}

static uint32_t rand_elem(uint32_t count)
{
  return (uint32_t)(randf(0.0f, (float)count));
}

/* Convert HSL to RGBA */
static void hsl_to_rgba(float h, float s, float l, float* rgba)
{
  /* Normalize hue to 0-1 range */
  h = fmodf(h, 1.0f);
  if (h < 0.0f)
    h += 1.0f;

  float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
  float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
  float m = l - c / 2.0f;

  float r, g, b;
  if (h < 1.0f / 6.0f) {
    r = c;
    g = x;
    b = 0;
  }
  else if (h < 2.0f / 6.0f) {
    r = x;
    g = c;
    b = 0;
  }
  else if (h < 3.0f / 6.0f) {
    r = 0;
    g = c;
    b = x;
  }
  else if (h < 4.0f / 6.0f) {
    r = 0;
    g = x;
    b = c;
  }
  else if (h < 5.0f / 6.0f) {
    r = x;
    g = 0;
    b = c;
  }
  else {
    r = c;
    g = 0;
    b = x;
  }

  rgba[0] = r + m;
  rgba[1] = g + m;
  rgba[2] = b + m;
  rgba[3] = 1.0f;
}

/* Create a buffer with data */
static WGPUBuffer create_buffer_with_data(wgpu_context_t* wgpu_context,
                                          const void* data, size_t size,
                                          WGPUBufferUsage usage,
                                          const char* label)
{
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device, &(WGPUBufferDescriptor){
                            .label = STRVIEW(label),
                            .size  = size,
                            .usage = usage | WGPUBufferUsage_CopyDst,
                          });
  wgpuQueueWriteBuffer(wgpu_context->queue, buffer, 0, data, size);
  return buffer;
}

/* Create geometry from primitive vertex data */
static void create_geometry(wgpu_context_t* wgpu_context,
                            primitive_vertex_data_t* data, geometry_t* geo,
                            const char* label)
{
  char vertex_label[64], index_label[64];
  snprintf(vertex_label, sizeof(vertex_label), "%s - Vertex buffer", label);
  snprintf(index_label, sizeof(index_label), "%s - Index buffer", label);

  geo->vertex_buffer = create_buffer_with_data(
    wgpu_context, data->vertices, data->vertex_count * PRIMITIVE_VERTEX_STRIDE,
    WGPUBufferUsage_Vertex, vertex_label);

  geo->index_buffer = create_buffer_with_data(
    wgpu_context, data->indices, data->index_count * sizeof(uint16_t),
    WGPUBufferUsage_Index, index_label);

  geo->index_format = WGPUIndexFormat_Uint16;
  geo->num_vertices = (uint32_t)data->index_count;
}

/* -------------------------------------------------------------------------- *
 * Initialize geometries
 * -------------------------------------------------------------------------- */

static void init_geometries(wgpu_context_t* wgpu_context)
{
  primitive_vertex_data_t data    = {0};
  primitive_vertex_data_t faceted = {0};

  /* Create plane (reoriented to be at y=0.5) */
  primitive_create_plane(NULL, &data);
  /* Translate plane vertices by 0.5 in Y */
  for (uint64_t i = 0; i < data.vertex_count; i++) {
    data.vertices[i * PRIMITIVE_VERTEX_SIZE + 1] += 0.5f;
  }
  create_geometry(wgpu_context, &data, &state.plane_geo, "Plane");
  primitive_vertex_data_destroy(&data);

  /* Create sphere */
  primitive_create_sphere(NULL, &data);
  create_geometry(wgpu_context, &data, &state.sphere_geo, "Sphere");
  primitive_vertex_data_destroy(&data);

  /* Create torus with thickness 0.5 (other options use defaults via macros) */
  primitive_create_torus(
    &(primitive_torus_options_t){
      .thickness = 0.5f,
    },
    &data);
  create_geometry(wgpu_context, &data, &state.torus_geo, "Torus");
  primitive_vertex_data_destroy(&data);

  /* Create cube */
  primitive_create_cube(NULL, &data);
  create_geometry(wgpu_context, &data, &state.cube_geo, "Cube");
  primitive_vertex_data_destroy(&data);

  /* Create truncated cone */
  primitive_create_truncated_cone(NULL, &data);
  create_geometry(wgpu_context, &data, &state.cone_geo, "Cone");
  primitive_vertex_data_destroy(&data);

  /* Create cylinder */
  primitive_create_cylinder(NULL, &data);
  create_geometry(wgpu_context, &data, &state.cylinder_geo, "Cylinder");
  primitive_vertex_data_destroy(&data);

  /* Create jem (faceted sphere with low subdivision) */
  primitive_create_sphere(
    &(primitive_sphere_options_t){
      .subdivisions_axis   = 6,
      .subdivisions_height = 5,
    },
    &data);
  primitive_facet(&data, &faceted);
  primitive_vertex_data_destroy(&data);
  create_geometry(wgpu_context, &faceted, &state.jem_geo, "Jem");
  primitive_vertex_data_destroy(&faceted);

  /* Create dice (faceted torus with low subdivision) */
  primitive_create_torus(
    &(primitive_torus_options_t){
      .thickness           = 0.5f,
      .radial_subdivisions = 8,
      .body_subdivisions   = 8,
    },
    &data);
  primitive_facet(&data, &faceted);
  primitive_vertex_data_destroy(&data);
  create_geometry(wgpu_context, &faceted, &state.dice_geo, "Dice");
  primitive_vertex_data_destroy(&faceted);

  /* Store geometry pointers for random selection */
  state.geometries[0] = &state.plane_geo;
  state.geometries[1] = &state.sphere_geo;
  state.geometries[2] = &state.torus_geo;
  state.geometries[3] = &state.cube_geo;
  state.geometries[4] = &state.cone_geo;
  state.geometries[5] = &state.cylinder_geo;
  state.geometries[6] = &state.jem_geo;
  state.geometries[7] = &state.dice_geo;
}

/* -------------------------------------------------------------------------- *
 * Initialize scene
 * -------------------------------------------------------------------------- */

static void init_scene(wgpu_context_t* wgpu_context, scene_t* scene,
                       uint32_t num_instances, float hue,
                       uint32_t geometry_index_start,
                       uint32_t geometry_index_count)
{
  scene->num_objects = num_instances;
  scene->object_infos
    = (object_info_t*)calloc(num_instances, sizeof(object_info_t));

  /* Create shared uniform buffer */
  scene->shared_uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Shared uniform buffer"),
      .size  = sizeof(scene->shared_uniform_values),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    });

  /* Create per-object uniform buffers and bind groups */
  for (uint32_t i = 0; i < num_instances; i++) {
    object_info_t* obj = &scene->object_infos[i];

    /* Initialize color with random hue variation */
    float rgba[4];
    hsl_to_rgba(hue + randf(0.0f, 0.2f), randf(0.7f, 1.0f), randf(0.5f, 0.8f),
                rgba);

    /* Set color in uniform values (after world matrix) */
    obj->uniform_values[16] = rgba[0];
    obj->uniform_values[17] = rgba[1];
    obj->uniform_values[18] = rgba[2];
    obj->uniform_values[19] = rgba[3];

    /* Create uniform buffer */
    obj->uniform_buffer = create_buffer_with_data(
      wgpu_context, obj->uniform_values, sizeof(obj->uniform_values),
      WGPUBufferUsage_Uniform, "Object uniform buffer");

    /* Create bind group */
    obj->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Object bind group"),
        .layout     = state.bind_group_layout,
        .entryCount = 2,
        .entries    = (WGPUBindGroupEntry[]){
          {
            .binding = 0,
            .buffer  = obj->uniform_buffer,
            .size    = sizeof(obj->uniform_values),
          },
          {
            .binding = 1,
            .buffer  = scene->shared_uniform_buffer,
            .size    = sizeof(scene->shared_uniform_values),
          },
        },
      });

    /* Assign random geometry from allowed range */
    obj->geometry_index
      = geometry_index_start + rand_elem(geometry_index_count);
  }
}

/* -------------------------------------------------------------------------- *
 * Initialize pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Create bind group layout */
  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Bind group layout"),
      .entryCount = 2,
      .entries    = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
          .buffer     = {.type = WGPUBufferBindingType_Uniform},
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
          .buffer     = {.type = WGPUBufferBindingType_Uniform},
        },
      },
    });

  /* Create pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = STRVIEW("Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });

  /* Create shader module */
  WGPUShaderModule shader_module = wgpu_create_shader_module(
    wgpu_context->device, simple_lighting_shader_wgsl);

  /* Vertex buffer layout */
  WGPUVertexAttribute vertex_attributes[3] = {
    {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
    {.format = WGPUVertexFormat_Float32x3, .offset = 12, .shaderLocation = 1},
    {.format = WGPUVertexFormat_Float32x2, .offset = 24, .shaderLocation = 2},
  };

  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = 32,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 3,
    .attributes     = vertex_attributes,
  };

  /* Create stencil set pipeline */
  state.stencil_set_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Stencil set pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("myVSMain"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("myFSMain"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_Back,
        .frontFace = WGPUFrontFace_CCW,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format            = DEPTH_FORMAT,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
        .stencilFront      = {
          .compare     = WGPUCompareFunction_Always,
          .failOp      = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp      = WGPUStencilOperation_Replace,
        },
        .stencilBack = {
          .compare     = WGPUCompareFunction_Always,
          .failOp      = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp      = WGPUStencilOperation_Keep,
        },
        .stencilReadMask  = 0xFF,
        .stencilWriteMask = 0xFF,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });

  /* Create stencil mask pipeline */
  state.stencil_mask_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Stencil mask pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("myVSMain"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("myFSMain"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_Back,
        .frontFace = WGPUFrontFace_CCW,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format            = DEPTH_FORMAT,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
        .stencilFront      = {
          .compare     = WGPUCompareFunction_Equal,
          .failOp      = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp      = WGPUStencilOperation_Keep,
        },
        .stencilBack = {
          .compare     = WGPUCompareFunction_Equal,
          .failOp      = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp      = WGPUStencilOperation_Keep,
        },
        .stencilReadMask  = 0xFF,
        .stencilWriteMask = 0xFF,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });

  wgpuShaderModuleRelease(shader_module);
}

/* -------------------------------------------------------------------------- *
 * Initialize scenes
 * -------------------------------------------------------------------------- */

static void init_scenes(wgpu_context_t* wgpu_context)
{
  /* Mask scenes - each with a single plane */
  for (uint32_t i = 0; i < NUM_MASK_SCENES; i++) {
    float hue = (float)i / 6.0f + 0.5f;
    init_scene(wgpu_context, &state.mask_scenes[i], 1, hue, 0,
               1); /* plane only */
  }

  /* Object scenes with different geometry types */
  init_scene(wgpu_context, &state.object_scenes[0], NUM_INSTANCES_PER_SCENE,
             0.0f / 7.0f, 1, 1); /* sphere */
  init_scene(wgpu_context, &state.object_scenes[1], NUM_INSTANCES_PER_SCENE,
             1.0f / 7.0f, 3, 1); /* cube */
  init_scene(wgpu_context, &state.object_scenes[2], NUM_INSTANCES_PER_SCENE,
             2.0f / 7.0f, 2, 1); /* torus */
  init_scene(wgpu_context, &state.object_scenes[3], NUM_INSTANCES_PER_SCENE,
             3.0f / 7.0f, 4, 1); /* cone */
  init_scene(wgpu_context, &state.object_scenes[4], NUM_INSTANCES_PER_SCENE,
             4.0f / 7.0f, 5, 1); /* cylinder */
  init_scene(wgpu_context, &state.object_scenes[5], NUM_INSTANCES_PER_SCENE,
             5.0f / 7.0f, 6, 1); /* jem */
  init_scene(wgpu_context, &state.object_scenes[6], NUM_INSTANCES_PER_SCENE,
             6.0f / 7.0f, 7, 1); /* dice */
}

/* -------------------------------------------------------------------------- *
 * Update depth texture
 * -------------------------------------------------------------------------- */

static void update_depth_texture(wgpu_context_t* wgpu_context)
{
  if (state.depth_texture == NULL
      || state.depth_texture_width != (uint32_t)wgpu_context->width
      || state.depth_texture_height != (uint32_t)wgpu_context->height) {

    if (state.depth_texture_view) {
      wgpuTextureViewRelease(state.depth_texture_view);
    }
    if (state.depth_texture) {
      wgpuTextureDestroy(state.depth_texture);
      wgpuTextureRelease(state.depth_texture);
    }

    state.depth_texture_width  = wgpu_context->width;
    state.depth_texture_height = wgpu_context->height;

    state.depth_texture = wgpuDeviceCreateTexture(
      wgpu_context->device,
      &(WGPUTextureDescriptor){
        .label = STRVIEW("Depth stencil texture"),
        .size  = {state.depth_texture_width, state.depth_texture_height, 1},
        .mipLevelCount = 1,
        .sampleCount   = 1,
        .dimension     = WGPUTextureDimension_2D,
        .format        = DEPTH_FORMAT,
        .usage         = WGPUTextureUsage_RenderAttachment,
      });

    state.depth_texture_view = wgpuTextureCreateView(
      state.depth_texture, &(WGPUTextureViewDescriptor){
                             .label           = STRVIEW("Depth stencil view"),
                             .format          = DEPTH_FORMAT,
                             .dimension       = WGPUTextureViewDimension_2D,
                             .mipLevelCount   = 1,
                             .arrayLayerCount = 1,
                           });
  }
}

/* -------------------------------------------------------------------------- *
 * Update mask scene
 * -------------------------------------------------------------------------- */

static void update_mask(wgpu_context_t* wgpu_context, float time,
                        scene_t* scene, float rotation[3])
{
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Create view projection matrix */
  mat4 projection, view, view_projection;
  glm_perspective(GLM_PI / 6.0f, aspect, 0.5f, 100.0f, projection);

  vec3 eye    = {0.0f, 0.0f, 45.0f};
  vec3 target = {0.0f, 0.0f, 0.0f};
  vec3 up     = {0.0f, 1.0f, 0.0f};
  glm_lookat(eye, target, up, view);

  glm_mat4_mul(projection, view, view_projection);

  /* Copy to shared uniforms */
  memcpy(scene->shared_uniform_values, view_projection, sizeof(mat4));

  /* Light direction */
  vec3 light_dir = {1.0f, 8.0f, 10.0f};
  glm_vec3_normalize(light_dir);
  scene->shared_uniform_values[16] = light_dir[0];
  scene->shared_uniform_values[17] = light_dir[1];
  scene->shared_uniform_values[18] = light_dir[2];

  wgpuQueueWriteBuffer(wgpu_context->queue, scene->shared_uniform_buffer, 0,
                       scene->shared_uniform_values,
                       sizeof(scene->shared_uniform_values));

  /* Update each object */
  for (uint32_t i = 0; i < scene->num_objects; i++) {
    object_info_t* obj = &scene->object_infos[i];

    mat4 world;
    glm_mat4_identity(world);

    /* Translate based on mouse position */
    float world_x = state.mouse_pos.x * 10.0f;
    float world_y = state.mouse_pos.y * 10.0f;
    glm_translate(world, (vec3){world_x, world_y, 0.0f});

    /* Apply time-based rotation */
    glm_rotate_x(world, time * 0.25f, world);
    glm_rotate_y(world, time * 0.15f, world);

    /* Apply face rotation */
    glm_rotate_x(world, rotation[0] * GLM_PI, world);
    glm_rotate_z(world, rotation[2] * GLM_PI, world);

    /* Scale */
    glm_scale(world, (vec3){10.0f, 10.0f, 10.0f});

    memcpy(obj->uniform_values, world, sizeof(mat4));
    /* Only write world matrix (64 bytes), color is static */
    wgpuQueueWriteBuffer(wgpu_context->queue, obj->uniform_buffer, 0,
                         obj->uniform_values, sizeof(mat4));
  }
}

/* -------------------------------------------------------------------------- *
 * Update scene0 - fixed camera, orbiting objects
 * -------------------------------------------------------------------------- */

static void update_scene0(wgpu_context_t* wgpu_context, float time,
                          scene_t* scene)
{
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  mat4 projection, view, view_projection;
  glm_perspective(GLM_PI / 6.0f, aspect, 0.5f, 100.0f, projection);

  vec3 eye    = {0.0f, 0.0f, 35.0f};
  vec3 target = {0.0f, 0.0f, 0.0f};
  vec3 up     = {0.0f, 1.0f, 0.0f};
  glm_lookat(eye, target, up, view);

  glm_mat4_mul(projection, view, view_projection);

  memcpy(scene->shared_uniform_values, view_projection, sizeof(mat4));

  vec3 light_dir = {1.0f, 8.0f, 10.0f};
  glm_vec3_normalize(light_dir);
  scene->shared_uniform_values[16] = light_dir[0];
  scene->shared_uniform_values[17] = light_dir[1];
  scene->shared_uniform_values[18] = light_dir[2];

  wgpuQueueWriteBuffer(wgpu_context->queue, scene->shared_uniform_buffer, 0,
                       scene->shared_uniform_values,
                       sizeof(scene->shared_uniform_values));

  for (uint32_t i = 0; i < scene->num_objects; i++) {
    object_info_t* obj = &scene->object_infos[i];

    mat4 world;
    glm_mat4_identity(world);

    glm_translate(world,
                  (vec3){0.0f, 0.0f, sinf(i * 3.721f + time * 0.1f) * 10.0f});
    glm_rotate_x(world, i * 4.567f, world);
    glm_rotate_y(world, i * 2.967f, world);
    glm_translate(world,
                  (vec3){0.0f, 0.0f, sinf(i * 9.721f + time * 0.1f) * 10.0f});
    glm_rotate_x(world, time * 0.53f + i, world);

    memcpy(obj->uniform_values, world, sizeof(mat4));
    /* Only write world matrix (64 bytes), color is static */
    wgpuQueueWriteBuffer(wgpu_context->queue, obj->uniform_buffer, 0,
                         obj->uniform_values, sizeof(mat4));
  }
}

/* -------------------------------------------------------------------------- *
 * Update scene1 - orbiting camera, orbiting objects
 * -------------------------------------------------------------------------- */

static void update_scene1(wgpu_context_t* wgpu_context, float time,
                          scene_t* scene)
{
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  mat4 projection, view, view_projection;
  glm_perspective(GLM_PI / 6.0f, aspect, 0.5f, 100.0f, projection);

  float radius = 35.0f;
  float t      = time * 0.1f;
  vec3 eye     = {cosf(t) * radius, 4.0f, sinf(t) * radius};
  vec3 target  = {0.0f, 0.0f, 0.0f};
  vec3 up      = {0.0f, 1.0f, 0.0f};
  glm_lookat(eye, target, up, view);

  glm_mat4_mul(projection, view, view_projection);

  memcpy(scene->shared_uniform_values, view_projection, sizeof(mat4));

  vec3 light_dir = {1.0f, 8.0f, 10.0f};
  glm_vec3_normalize(light_dir);
  scene->shared_uniform_values[16] = light_dir[0];
  scene->shared_uniform_values[17] = light_dir[1];
  scene->shared_uniform_values[18] = light_dir[2];

  wgpuQueueWriteBuffer(wgpu_context->queue, scene->shared_uniform_buffer, 0,
                       scene->shared_uniform_values,
                       sizeof(scene->shared_uniform_values));

  for (uint32_t i = 0; i < scene->num_objects; i++) {
    object_info_t* obj = &scene->object_infos[i];

    mat4 world;
    glm_mat4_identity(world);

    glm_translate(world,
                  (vec3){0.0f, 0.0f, sinf(i * 3.721f + time * 0.1f) * 10.0f});
    glm_rotate_x(world, i * 4.567f, world);
    glm_rotate_y(world, i * 2.967f, world);
    glm_translate(world,
                  (vec3){0.0f, 0.0f, sinf(i * 9.721f + time * 0.1f) * 10.0f});
    glm_rotate_x(world, time * 1.53f + i, world);

    memcpy(obj->uniform_values, world, sizeof(mat4));
    /* Only write world matrix (64 bytes), color is static */
    wgpuQueueWriteBuffer(wgpu_context->queue, obj->uniform_buffer, 0,
                         obj->uniform_values, sizeof(mat4));
  }
}

/* -------------------------------------------------------------------------- *
 * Draw scene
 * -------------------------------------------------------------------------- */

static void draw_scene(WGPUCommandEncoder encoder,
                       WGPURenderPassDescriptor* pass_desc,
                       WGPURenderPipeline pipeline, scene_t* scene,
                       uint32_t stencil_ref)
{
  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(encoder, pass_desc);
  wgpuRenderPassEncoderSetPipeline(pass, pipeline);
  wgpuRenderPassEncoderSetStencilReference(pass, stencil_ref);

  for (uint32_t i = 0; i < scene->num_objects; i++) {
    object_info_t* obj = &scene->object_infos[i];
    geometry_t* geo    = state.geometries[obj->geometry_index];

    wgpuRenderPassEncoderSetBindGroup(pass, 0, obj->bind_group, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, geo->vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(pass, geo->index_buffer,
                                        geo->index_format, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(pass, geo->num_vertices, 1, 0, 0, 0);
  }

  wgpuRenderPassEncoderEnd(pass);
  wgpuRenderPassEncoderRelease(pass);
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    /* Normalize mouse position to -1 to 1 */
    state.mouse_pos.x
      = (input_event->mouse_x / (float)wgpu_context->width) * 2.0f - 1.0f;
    state.mouse_pos.y
      = -((input_event->mouse_y / (float)wgpu_context->height) * 2.0f - 1.0f);
  }
}

/* -------------------------------------------------------------------------- *
 * Initialize example
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Initialize random seed */
  srand(12345);

  /* Initialize timing */
  stm_setup();
  state.last_time = stm_now();

  /* Initialize pipelines first (need bind group layout) */
  init_pipelines(wgpu_context);

  /* Initialize geometries */
  init_geometries(wgpu_context);

  /* Initialize scenes */
  init_scenes(wgpu_context);

  /* Setup render pass descriptors */
  state.clear_pass_desc.colorAttachmentCount   = 1;
  state.clear_pass_desc.colorAttachments       = &state.clear_color_attachment;
  state.clear_pass_desc.depthStencilAttachment = &state.clear_depth_attachment;

  state.load_pass_desc.colorAttachmentCount   = 1;
  state.load_pass_desc.colorAttachments       = &state.load_color_attachment;
  state.load_pass_desc.depthStencilAttachment = &state.load_depth_attachment;

  state.initialized = true;

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Frame
 * -------------------------------------------------------------------------- */

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  float time = (float)stm_sec(stm_since(0));

  /* Update depth texture if needed */
  update_depth_texture(wgpu_context);

  /* Update mask scenes */
  float rotations[6][3] = {
    {0.0f, 0.0f, 0.0f},  /* front */
    {1.0f, 0.0f, 0.0f},  /* back */
    {0.0f, 0.0f, 0.5f},  /* left */
    {0.0f, 0.0f, -0.5f}, /* right */
    {-0.5f, 0.0f, 0.0f}, /* top */
    {0.5f, 0.0f, 0.0f},  /* bottom */
  };

  for (uint32_t i = 0; i < NUM_MASK_SCENES; i++) {
    update_mask(wgpu_context, time, &state.mask_scenes[i], rotations[i]);
  }

  /* Update object scenes - alternating update patterns */
  update_scene0(wgpu_context, time, &state.object_scenes[0]);
  update_scene1(wgpu_context, time, &state.object_scenes[1]);
  update_scene0(wgpu_context, time, &state.object_scenes[2]);
  update_scene1(wgpu_context, time, &state.object_scenes[3]);
  update_scene0(wgpu_context, time, &state.object_scenes[4]);
  update_scene1(wgpu_context, time, &state.object_scenes[5]);
  update_scene0(wgpu_context, time, &state.object_scenes[6]);

  /* Set up render pass attachments */
  state.clear_color_attachment.view = wgpu_context->swapchain_view;
  state.clear_depth_attachment.view = state.depth_texture_view;
  state.load_color_attachment.view  = wgpu_context->swapchain_view;
  state.load_depth_attachment.view  = state.depth_texture_view;

  /* Create command encoder */
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Draw the 6 faces of a cube into the stencil buffer */
  draw_scene(encoder, &state.clear_pass_desc, state.stencil_set_pipeline,
             &state.mask_scenes[0], 1);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_set_pipeline,
             &state.mask_scenes[1], 2);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_set_pipeline,
             &state.mask_scenes[2], 3);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_set_pipeline,
             &state.mask_scenes[3], 4);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_set_pipeline,
             &state.mask_scenes[4], 5);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_set_pipeline,
             &state.mask_scenes[5], 6);

  /* Draw each scene where stencil matches */
  draw_scene(encoder, &state.load_pass_desc, state.stencil_mask_pipeline,
             &state.object_scenes[0], 0);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_mask_pipeline,
             &state.object_scenes[1], 1);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_mask_pipeline,
             &state.object_scenes[2], 2);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_mask_pipeline,
             &state.object_scenes[3], 3);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_mask_pipeline,
             &state.object_scenes[4], 4);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_mask_pipeline,
             &state.object_scenes[5], 5);
  draw_scene(encoder, &state.load_pass_desc, state.stencil_mask_pipeline,
             &state.object_scenes[6], 6);

  /* Submit */
  WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  wgpuCommandBufferRelease(command_buffer);
  wgpuCommandEncoderRelease(encoder);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Cleanup
 * -------------------------------------------------------------------------- */

static void destroy_geometry(geometry_t* geo)
{
  if (geo->vertex_buffer) {
    wgpuBufferDestroy(geo->vertex_buffer);
    wgpuBufferRelease(geo->vertex_buffer);
  }
  if (geo->index_buffer) {
    wgpuBufferDestroy(geo->index_buffer);
    wgpuBufferRelease(geo->index_buffer);
  }
}

static void destroy_scene(scene_t* scene)
{
  for (uint32_t i = 0; i < scene->num_objects; i++) {
    object_info_t* obj = &scene->object_infos[i];
    if (obj->uniform_buffer) {
      wgpuBufferDestroy(obj->uniform_buffer);
      wgpuBufferRelease(obj->uniform_buffer);
    }
    if (obj->bind_group) {
      wgpuBindGroupRelease(obj->bind_group);
    }
  }
  if (scene->shared_uniform_buffer) {
    wgpuBufferDestroy(scene->shared_uniform_buffer);
    wgpuBufferRelease(scene->shared_uniform_buffer);
  }
  free(scene->object_infos);
  scene->object_infos = NULL;
}

static void cleanup(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Destroy geometries */
  destroy_geometry(&state.plane_geo);
  destroy_geometry(&state.sphere_geo);
  destroy_geometry(&state.torus_geo);
  destroy_geometry(&state.cube_geo);
  destroy_geometry(&state.cone_geo);
  destroy_geometry(&state.cylinder_geo);
  destroy_geometry(&state.jem_geo);
  destroy_geometry(&state.dice_geo);

  /* Destroy scenes */
  for (uint32_t i = 0; i < NUM_MASK_SCENES; i++) {
    destroy_scene(&state.mask_scenes[i]);
  }
  for (uint32_t i = 0; i < NUM_OBJECT_SCENES; i++) {
    destroy_scene(&state.object_scenes[i]);
  }

  /* Destroy depth texture */
  if (state.depth_texture_view) {
    wgpuTextureViewRelease(state.depth_texture_view);
  }
  if (state.depth_texture) {
    wgpuTextureDestroy(state.depth_texture);
    wgpuTextureRelease(state.depth_texture);
  }

  /* Destroy pipelines */
  if (state.stencil_set_pipeline) {
    wgpuRenderPipelineRelease(state.stencil_set_pipeline);
  }
  if (state.stencil_mask_pipeline) {
    wgpuRenderPipelineRelease(state.stencil_mask_pipeline);
  }
  if (state.pipeline_layout) {
    wgpuPipelineLayoutRelease(state.pipeline_layout);
  }
  if (state.bind_group_layout) {
    wgpuBindGroupLayoutRelease(state.bind_group_layout);
  }
}

/* -------------------------------------------------------------------------- *
 * Main entry point
 * -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  wgpu_start(&(wgpu_desc_t){
    .title          = "Stencil Mask",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = cleanup,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}
