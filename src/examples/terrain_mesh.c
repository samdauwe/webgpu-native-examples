#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Terrain Mesh
 *
 * This example shows how to render an infinite landscape for the camera to
 * meander around in. The terrain consists of a tiled planar mesh that is
 * displaced with a heightmap.
 *
 * The example demonstrates the following:
 *  * texture creation and sampling
 *  * displacement mapping in GLSL
 *  * bind groups for efficient resource binding
 *  * indexed and instanced draw calls
 *
 * Ref:
 * https://metalbyexample.com/webgpu-part-one/
 * https://metalbyexample.com/webgpu-part-two/
 * https://blogs.igalia.com/itoral/2016/10/13/opengl-terrain-renderer-rendering-the-terrain-mesh/
 * -------------------------------------------------------------------------- */

// Terrain patch parameters
#define PATCH_SIZE 50
#define PATCH_SEGMENT_COUNT 40
#define PATCH_INDEX_COUNT PATCH_SEGMENT_COUNT* PATCH_SEGMENT_COUNT * 6
#define PATCH_VERTEX_COUNT (PATCH_SEGMENT_COUNT + 1) * (PATCH_SEGMENT_COUNT + 1)
#define PATCH_FLOATS_PER_VERTEX 6

// Camera parameters
static const float fov_y  = TO_RADIANS(60.0f);
static const float near_z = 0.1f, far_z = 150.0f;
static vec3 camera_position                     = {0.0f, 5.0f, 0.0f};
static float camera_heading                     = PI / 2.0f; // radians
static float camera_target_heading              = PI / 2.0f; // radians
static const float camera_angular_easing_factor = 0.005f;
static const float camera_speed                 = 8.0f; // meters per second

// Used to calculate view and projection matrices
static float rot_y[16], trans[16], view_matrix[16], projection_matrix[16];

// Camera matrices
static float model_matrix[16], model_view_matrix[16];
static float model_view_projection_matrix[16];

// Nine terrain patches
static vec3 patch_centers[9];

// Time-related state
static float last_frame_time            = -1.0f;
static float direction_change_countdown = 6.0f; // seconds

// Internal constants
static const uint32_t instance_length
  = 16 * 2; // Length of the data associated with a single instance
static const uint32_t max_instance_count = 9;
static const uint64_t instance_buffer_length
  = 4 * instance_length * max_instance_count; // in bytes
static float* instance_data    = NULL;
static uint32_t instance_count = 1;

// Vertex buffer
static wgpu_buffer_t vertices = {0};

// Index buffer
static wgpu_buffer_t indices = {0};

// Instance buffer
static wgpu_buffer_t instance_buffer = {0};

// Textures
static struct {
  texture_t color;
  texture_t heightmap;
} textures;
static WGPUSampler linear_sampler = {0};

// Render pipeline + layout
static WGPURenderPipeline render_pipeline = {0};
static WGPUPipelineLayout pipeline_layout = {0};

// Bind group layouts
static struct {
  WGPUBindGroupLayout frame_constants;
  WGPUBindGroupLayout instance_buffer;
} bind_group_layouts = {0};

// Bind groups
static struct {
  WGPUBindGroup frame_constants;
  WGPUBindGroup instance_buffer;
} bind_groups = {0};

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

// Other variables
static const char* example_title = "Terrain Mesh";
static bool prepared             = false;

/* -------------------------------------------------------------------------- *
 * Custom math
 * -------------------------------------------------------------------------- */

static void mat4_mul(float (*a)[16], float (*b)[16], float (*m)[16])
{
  memset(m, 0, sizeof(*m));
  // clang-format off
  (*m)[0]  = (*a)[0] * (*b)[0]  + (*a)[4] * (*b)[1]  + (*a)[8]  * (*b)[2]  + (*a)[12] * (*b)[3];
  (*m)[1]  = (*a)[1] * (*b)[0]  + (*a)[5] * (*b)[1]  + (*a)[9]  * (*b)[2]  + (*a)[13] * (*b)[3];
  (*m)[2]  = (*a)[2] * (*b)[0]  + (*a)[6] * (*b)[1]  + (*a)[10] * (*b)[2]  + (*a)[14] * (*b)[3];
  (*m)[3]  = (*a)[3] * (*b)[0]  + (*a)[7] * (*b)[1]  + (*a)[11] * (*b)[2]  + (*a)[15] * (*b)[3];
  (*m)[4]  = (*a)[0] * (*b)[4]  + (*a)[4] * (*b)[5]  + (*a)[8]  * (*b)[6]  + (*a)[12] * (*b)[7];
  (*m)[5]  = (*a)[1] * (*b)[4]  + (*a)[5] * (*b)[5]  + (*a)[9]  * (*b)[6]  + (*a)[13] * (*b)[7];
  (*m)[6]  = (*a)[2] * (*b)[4]  + (*a)[6] * (*b)[5]  + (*a)[10] * (*b)[6]  + (*a)[14] * (*b)[7];
  (*m)[7]  = (*a)[3] * (*b)[4]  + (*a)[7] * (*b)[5]  + (*a)[11] * (*b)[6]  + (*a)[15] * (*b)[7];
  (*m)[8]  = (*a)[0] * (*b)[8]  + (*a)[4] * (*b)[9]  + (*a)[8]  * (*b)[10] + (*a)[12] * (*b)[11];
  (*m)[9]  = (*a)[1] * (*b)[8]  + (*a)[5] * (*b)[9]  + (*a)[9]  * (*b)[10] + (*a)[13] * (*b)[11];
  (*m)[10] = (*a)[2] * (*b)[8]  + (*a)[6] * (*b)[9]  + (*a)[10] * (*b)[10] + (*a)[14] * (*b)[11];
  (*m)[11] = (*a)[3] * (*b)[8]  + (*a)[7] * (*b)[9]  + (*a)[11] * (*b)[10] + (*a)[15] * (*b)[11];
  (*m)[12] = (*a)[0] * (*b)[12] + (*a)[4] * (*b)[13] + (*a)[8]  * (*b)[14] + (*a)[12] * (*b)[15];
  (*m)[13] = (*a)[1] * (*b)[12] + (*a)[5] * (*b)[13] + (*a)[9]  * (*b)[14] + (*a)[13] * (*b)[15];
  (*m)[14] = (*a)[2] * (*b)[12] + (*a)[6] * (*b)[13] + (*a)[10] * (*b)[14] + (*a)[14] * (*b)[15];
  (*m)[15] = (*a)[3] * (*b)[12] + (*a)[7] * (*b)[13] + (*a)[11] * (*b)[14] + (*a)[15] * (*b)[15];
  // clang-format on
}

static void mat4_translation(float (*m)[16], vec3 t)
{
  memset(m, 0, sizeof(*m));
  (*m)[0]  = 1.0f;
  (*m)[5]  = 1.0f;
  (*m)[10] = 1.0f;
  (*m)[12] = t[0];
  (*m)[13] = t[1];
  (*m)[14] = t[2];
  (*m)[15] = 1.0f;
}

static void mat4_rotation_y(float (*m)[16], float angle)
{
  memset(m, 0, sizeof(*m));
  const float c = cos(angle);
  const float s = sin(angle);
  (*m)[0]       = c;
  (*m)[2]       = -s;
  (*m)[5]       = 1.0f;
  (*m)[8]       = s;
  (*m)[10]      = c;
  (*m)[15]      = 1.0f;
}

/*
 * Calculates a perspective projection matrix that maps from right-handed view
 * space to left-handed clip space with z on [0, 1]
 */
static void mat4_perspective_fov(float fovY, float aspect, float near,
                                 float far, float (*m)[16])
{
  memset(m, 0, sizeof(*m));
  const float sy = 1.0f / tan(fovY * 0.5f);
  const float nf = 1.0f / (near - far);
  (*m)[0]        = sy / aspect;
  (*m)[5]        = sy;
  (*m)[10]       = far * nf;
  (*m)[11]       = -1.0f;
  (*m)[14]       = far * near * nf;
}

/* -------------------------------------------------------------------------- *
 * Terrain Mesh example
 * -------------------------------------------------------------------------- */

static void prepare_patch_mesh(wgpu_context_t* wgpu_context)
{
  float vertices_data[PATCH_VERTEX_COUNT * PATCH_FLOATS_PER_VERTEX] = {0};
  uint32_t indices_data[PATCH_INDEX_COUNT]                          = {0};

  const uint32_t patch_size          = (uint32_t)PATCH_SIZE;
  const uint32_t patch_segment_count = (uint32_t)PATCH_SEGMENT_COUNT;
  const uint32_t floats_per_vertex   = (uint32_t)PATCH_FLOATS_PER_VERTEX;

  for (uint32_t zi = 0, v = 0; zi < patch_segment_count + 1; ++zi) {
    for (uint32_t xi = 0; xi < patch_segment_count + 1; ++xi) {
      float s               = xi / (float)patch_segment_count;
      float t               = zi / (float)patch_segment_count;
      uint64_t vi           = v * floats_per_vertex;
      vertices_data[vi + 0] = (s * patch_size) - (patch_size * 0.5f); /* x */
      vertices_data[vi + 1] = 0.0f;                                   /* y */
      vertices_data[vi + 2] = (t * patch_size) - (patch_size * 0.5f); /* z */
      vertices_data[vi + 3] = 1.0f;                                   /* w */
      vertices_data[vi + 4] = s;
      vertices_data[vi + 5] = t;
      ++v;
    }
  }

  for (uint32_t zi = 0, ii = 0; zi < patch_segment_count; ++zi) {
    for (uint32_t xi = 0; xi < patch_segment_count; ++xi) {
      const uint32_t bi    = zi * (patch_segment_count + 1);
      indices_data[ii + 0] = bi + xi;
      indices_data[ii + 1] = bi + xi + (patch_segment_count + 1);
      indices_data[ii + 2] = bi + xi + (patch_segment_count + 1) + 1;
      indices_data[ii + 3] = bi + xi + (patch_segment_count + 1) + 1;
      indices_data[ii + 4] = bi + xi + 1;
      indices_data[ii + 5] = bi + xi;
      ii += 6;
    }
  }

  /* Create vertex buffer */
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Terrain mesh - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertices_data),
                    .count = (uint32_t)ARRAY_SIZE(vertices_data),
                    .initial.data = vertices_data,
                  });

  /* Create index buffer */
  indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Terrain mesh - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(indices_data),
                    .count = (uint32_t)ARRAY_SIZE(indices_data),
                    .initial.data = indices_data,
                  });
}

static void prepare_textures(wgpu_context_t* wgpu_context)
{
  /* Color texture */
  {
    const char* file = "textures/color.png";
    textures.color   = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  /* Heightmap texture */
  {
    const char* file = "textures/heightmap.png";
    textures.heightmap
      = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  /* Linear sampler */
  WGPUSamplerDescriptor sampler_desc = {
    .label         = "Color texture - Linear sampler",
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
    .addressModeW  = WGPUAddressMode_Repeat,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Nearest,
    .mipmapFilter  = WGPUMipmapFilterMode_Linear,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  linear_sampler = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.812f,
        .g = 0.914f,
        .b = 1.0f,
        .a = 1.0f,
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context,
                          &(struct deph_stencil_texture_creation_options_t){
                            .format = WGPUTextureFormat_Depth32Float,
                          });

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static float float_random(float min, float max)
{
  const float scale = rand() / (float)RAND_MAX; /* [0, 1.0]   */
  return min + scale * (max - min);             /* [min, max] */
}

static void update_camera_pose(float dt)
{
  /* Update camera position */
  const float dx = -sin(camera_heading) * camera_speed * dt;
  const float dz = -cos(camera_heading) * camera_speed * dt;
  camera_position[0] += dx;
  camera_position[2] += dz;

  /* Update camera direction, choosing a new direction if needed */
  camera_heading
    += (camera_target_heading - camera_heading) * camera_angular_easing_factor;
  if (direction_change_countdown < 0.0f) {
    camera_target_heading      = (float_random(0.0f, 1.0f) * PI * 2.0f) - PI;
    direction_change_countdown = 6.0f;
  }
  direction_change_countdown -= dt;
}

static void update_uniforms(wgpu_example_context_t* context)
{
  const float frame_timestamp_millis = context->frame.timestamp_millis;
  const float dt  = (frame_timestamp_millis - last_frame_time) * 0.001; // s
  last_frame_time = frame_timestamp_millis;

  update_camera_pose(dt);

  const float patch_size = (float)PATCH_SIZE;

  // Determine the nearest nine terrain patches and calculate their positions
  const vec3 nearest_patch_center = {
    round(camera_position[0] / patch_size) * patch_size, /* x */
    0.0f,                                                /* y */
    round(camera_position[2] / patch_size) * patch_size  /* z */
  };
  uint32_t patch_index = 0;
  for (int8_t pz = -1; pz <= 1; ++pz) {
    for (int8_t px = -1; px <= 1; ++px) {
      glm_vec3_copy(
        (vec3){
          nearest_patch_center[0] + patch_size * px, /* x */
          nearest_patch_center[1],                   /* y */
          nearest_patch_center[2] + patch_size * pz  /* z */
        },
        patch_centers[patch_index]);
      ++patch_index;
    }
  }

  // Calculate view and projection matrices
  mat4_rotation_y(&rot_y, -camera_heading);
  mat4_translation(&trans, (vec3){-camera_position[0], -camera_position[1],
                                  -camera_position[2]});
  mat4_mul(&rot_y, &trans, &view_matrix);
  const float aspect_ratio = context->window_size.aspect_ratio;
  mat4_perspective_fov(fov_y, aspect_ratio, near_z, far_z, &projection_matrix);

  // Calculate the per-instance matrices
  if (instance_data == NULL) {
    instance_data = calloc(instance_buffer_length / 4, sizeof(float));
  }
  instance_count = MIN(ARRAY_SIZE(patch_centers), max_instance_count);
  for (uint32_t i = 0; i < instance_count; ++i) {
    mat4_translation(&model_matrix, patch_centers[i]);
    mat4_mul(&view_matrix, &model_matrix, &model_view_matrix);
    mat4_mul(&projection_matrix, &model_view_matrix,
             &model_view_projection_matrix);
    memcpy(instance_data + (i * instance_length), model_view_matrix,
           sizeof(model_view_matrix));
    memcpy(instance_data + (i * instance_length + 16),
           model_view_projection_matrix, sizeof(model_view_projection_matrix));
  }

  // Write the instance data to the instance buffer
  wgpu_queue_write_buffer(context->wgpu_context, instance_buffer.buffer, 0,
                          instance_data,
                          (instance_buffer_length / 4) * sizeof(float));
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  instance_buffer = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Instance - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = instance_buffer_length,
    });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Frame constants bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Sampler */
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Texture view */
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Texture view */
        .binding    = 2,
        .visibility = WGPUShaderStage_Vertex,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      }
    };
    bind_group_layouts.frame_constants = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = "Frame constants - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.frame_constants != NULL)
  }

  /* Instance buffer bind group */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Transform */
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = instance_buffer_length,
        },
        .sampler = {0},
      },
    };
    bind_group_layouts.instance_buffer = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = "Instance buffer - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.instance_buffer != NULL)
  }

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  WGPUBindGroupLayout bindGroupLayouts[2] = {
    bind_group_layouts.frame_constants, /* Set 0 */
    bind_group_layouts.instance_buffer, /* Set 1 */
  };
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bindGroupLayouts),
      .bindGroupLayouts     = bindGroupLayouts,
    });
  ASSERT(pipeline_layout != NULL)
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Frame constants bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .sampler = linear_sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = textures.color.view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding     = 2,
        .textureView = textures.heightmap.view,
      }
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Frame constants - Bind group",
      .layout     = bind_group_layouts.frame_constants,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.frame_constants
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.frame_constants != NULL)
  }

  /* Instance buffer bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = instance_buffer.buffer,
        .offset  = 0,
        .size    = instance_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Instance buffer - Bind group",
      .layout     = bind_group_layouts.instance_buffer,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.instance_buffer
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.instance_buffer != NULL)
  }
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth32Float,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    terrain_mesh, 24,
    /* Attribute descriptions */
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, 0),
    // Attribute location 1: Texture coordinates
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 16))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader SPIR-V
                      .label = "Terrain mesh - Vertex shader",
                      .file  = "shaders/terrain_mesh/shader.vert.spv",
                    },
                    .buffer_count = 1,
                    .buffers      = &terrain_mesh_vertex_buffer_layout,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader SPIR-V
                      .label = "Terrain mesh - Fragment shader",
                      .file  = "shaders/terrain_mesh/shader.frag.spv",
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
  render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Terrain mesh - Render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_patch_mesh(context->wgpu_context);
    prepare_uniform_buffers(context);
    prepare_textures(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
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

  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, render_pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.frame_constants, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 1,
                                    bind_groups.instance_buffer, 0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(wgpu_context->rpass_enc, indices.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                   (uint32_t)PATCH_INDEX_COUNT, instance_count,
                                   0, 0, 0);

  // Create command buffer and cleanup
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

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

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  update_uniforms(context);
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  if (instance_data != NULL) {
    free(instance_data);
  }

  wgpu_destroy_texture(&textures.color);
  wgpu_destroy_texture(&textures.heightmap);
  WGPU_RELEASE_RESOURCE(Sampler, linear_sampler)

  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, instance_buffer.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.frame_constants)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.instance_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.frame_constants)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.instance_buffer)
}

void example_terrain_mesh(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .vsync = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
