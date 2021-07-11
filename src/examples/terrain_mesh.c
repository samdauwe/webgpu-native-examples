#include "example_base.h"
#include "examples.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Terrain Mesh
 *
 * This example shows how to render an infinite landscape for the camera to
 * meander around in. The terrain will consist of a tiled planar mesh that is
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

#define RADIANS_PER_DEGREE (PI / 180.0f)

// Camera parameters
static const float fov_y  = 60.0f * RADIANS_PER_DEGREE;
static const float near_z = 0.1f, far_z = 150.0f;
static vec3 camera_position               = {0.0f, 5.0f, 0.0f};
static float camera_heading               = PI / 2.0f; // radians
static float camera_target_heading        = PI / 2.0f; // radians
static float camera_angular_easing_factor = 0.01f;
static float camera_speed                 = 8.0f; // meters per second

// Terrain patch parameters
#define PATCH_SIZE 50
#define PATCH_SEGMENT_COUNT 40
#define PATCH_INDEX_COUNT PATCH_SEGMENT_COUNT* PATCH_SEGMENT_COUNT * 6
#define PATCH_VERTEX_COUNT (PATCH_SEGMENT_COUNT + 1) * (PATCH_SEGMENT_COUNT + 1)
#define PATCH_FLOATS_PER_VERTEX 6

// Nine terrain patches
static vec3 patch_centers[9];

// Time-related state
static float last_frame_time            = -1.0f;
static float direction_change_countdown = 6.0f; // seconds

// Internal constants
static const uint32_t instance_length
  = sizeof(mat4) * 2; // Length of the data associated with a single instance
static const uint32_t max_instance_count = 9;
static const uint64_t instance_buffer_length
  = 4 * instance_length * max_instance_count; // in bytes
static float* instance_data    = NULL;
static uint32_t instance_count = 1;

// Vertex buffer
static struct vertices_t {
  WGPUBuffer buffer;
  uint32_t count;
} vertices = {0};

// Index buffer
static struct indices_t {
  WGPUBuffer buffer;
  uint32_t count;
} indices = {0};

// Instance buffer
static WGPUBuffer instance_buffer;

// Textures
static struct textures_t {
  texture_t color;
  texture_t heightmap;
} textures;
static WGPUSampler linear_sampler;

// Render pipeline + layout
static WGPURenderPipeline render_pipeline;
static WGPUPipelineLayout pipeline_layout;

// Bind group layouts
static struct bind_group_layouts_t {
  WGPUBindGroupLayout frame_constants;
  WGPUBindGroupLayout instance_buffer;
} bind_group_layouts;

// Bind groups
static struct bind_groups_t {
  WGPUBindGroup frame_constants;
  WGPUBindGroup instance_buffer;
} bind_groups;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachmentDescriptor rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

// Other variables
static const char* example_title = "Terrain Mesh";
static bool prepared             = false;

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
      vertices_data[vi + 0] = (s * patch_size) - (patch_size * 0.5f); // x
      vertices_data[vi + 1] = 0.0f;                                   // y
      vertices_data[vi + 2] = (t * patch_size) - (patch_size * 0.5f); // z
      vertices_data[vi + 3] = 1.0f;                                   // w
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

  // Create vertex buffer
  vertices.count              = (uint32_t)ARRAY_SIZE(vertices_data);
  uint32_t vertex_buffer_size = vertices.count * sizeof(float);
  vertices.buffer             = wgpu_create_buffer_from_data(
    wgpu_context, vertices_data, vertex_buffer_size, WGPUBufferUsage_Vertex);

  // Create index buffer
  indices.count              = (uint32_t)ARRAY_SIZE(indices_data);
  uint32_t index_buffer_size = indices.count * sizeof(uint32_t);
  indices.buffer             = wgpu_create_buffer_from_data(
    wgpu_context, indices_data, index_buffer_size, WGPUBufferUsage_Index);
}

static void prepare_textures(wgpu_context_t* wgpu_context)
{
  // Color texture
  {
    const char* file = "textures/color.png";
    textures.color   = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  // Heightmap texture
  {
    const char* file = "textures/heightmap.png";
    textures.heightmap
      = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  // Linear sampler
  WGPUSamplerDescriptor sampler_desc = {
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
    .addressModeW  = WGPUAddressMode_Repeat,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Nearest,
    .mipmapFilter  = WGPUFilterMode_Linear,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  linear_sampler = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachmentDescriptor) {
      .view       = NULL,
      .attachment = NULL,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.812f,
        .g = 0.914f,
        .b = 1.0f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context,
                          &(struct deph_stencil_texture_creation_options_t){
                            .format = WGPUTextureFormat_Depth32Float,
                          });

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static float float_random(float min, float max)
{
  const float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
  return min + scale * (max - min);             /* [min, max] */
}

static void update_camera_pose(float dt)
{
  // Update camera position
  const float dx = -sin(camera_heading) * camera_speed * dt;
  const float dz = -cos(camera_heading) * camera_speed * dt;
  camera_position[0] += dx;
  camera_position[2] += dz;

  // Update camera direction, choosing a new direction if needed
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
  float frame_timestamp_millis = context->frame.timestamp_millis;

  const float dt  = (last_frame_time < 0.0f) ?
                      (frame_timestamp_millis - last_frame_time) * 0.001 :
                      0.0f; // s
  last_frame_time = frame_timestamp_millis;

  update_camera_pose(dt);

  const float patch_size = (float)PATCH_SIZE;

  // Determine the nearest nine terrain patches and calculate their positions
  const vec3 nearest_patch_center = {
    round(camera_position[0] / patch_size) * patch_size, // x
    0.0f,                                                // y
    round(camera_position[2] / patch_size) * patch_size  // z
  };
  uint32_t patch_index = 0;
  for (int8_t pz = -1; pz <= 1; ++pz) {
    for (int8_t px = -1; px <= 1; ++px) {
      glm_vec3_copy(
        (vec3){
          nearest_patch_center[0] + patch_size * px, // x
          nearest_patch_center[1],                   // y
          nearest_patch_center[2] + patch_size * pz  // z
        },
        patch_centers[patch_index]);
      ++patch_index;
    }
  }

  // Calculate view and projection matrices
  mat4 rot_y_src = GLM_MAT4_IDENTITY_INIT, rot_y_dst = GLM_MAT4_ZERO_INIT;
  glm_rotate_y(rot_y_src, -camera_heading, rot_y_dst);

  mat4 trans = GLM_MAT4_IDENTITY_INIT;
  glm_translate(trans, (vec3){-camera_position[0], -camera_position[1],
                              -camera_position[2]});
  mat4 view_matrix = GLM_MAT4_ZERO_INIT;
  glm_mat4_mul(rot_y_dst, trans, view_matrix);
  const float aspect_ratio = context->window_size.aspect_ratio;
  mat4 projection_matrix   = GLM_MAT4_IDENTITY_INIT;
  glm_perspective(fov_y, aspect_ratio, near_z, far_z, projection_matrix);

  // Calculate the per-instance matrices
  if (instance_data == NULL) {
    instance_data = calloc(instance_buffer_length / 4, sizeof(float));
  }
  instance_count         = MIN(ARRAY_SIZE(patch_centers), max_instance_count);
  mat4 model_matrix      = GLM_MAT4_ZERO_INIT;
  mat4 model_view_matrix = GLM_MAT4_ZERO_INIT;
  mat4 model_view_projection_matrix = GLM_MAT4_ZERO_INIT;
  for (uint32_t i = 0; i < instance_count; ++i) {
    glm_mat4_identity(model_matrix);
    glm_translate(model_matrix, patch_centers[i]);
    glm_mat4_mul(view_matrix, model_matrix, model_view_matrix);
    glm_mat4_mul(projection_matrix, model_view_matrix,
                 model_view_projection_matrix);
    memcpy(instance_data + (i * 32), model_view_matrix,
           sizeof(model_view_matrix));
    memcpy(instance_data + (i * 32 + 16), model_view_projection_matrix,
           sizeof(model_view_projection_matrix));
  }

  // Write the instance data to the instance buffer
  wgpu_queue_write_buffer(context->wgpu_context, instance_buffer, 0,
                          &instance_data, sizeof(instance_data));
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  WGPUBufferDescriptor ubo_desc = {
    .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
    .size             = instance_buffer_length,
    .mappedAtCreation = false,
  };
  instance_buffer
    = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_desc);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Frame constants bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Sampler
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Texture view
        .binding = 1,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Texture view
        .binding = 2,
        .visibility = WGPUShaderStage_Vertex,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      }
    };
    bind_group_layouts.frame_constants = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.frame_constants != NULL)
  }

  // Instance buffer bind group
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Transform
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = instance_buffer_length,
        },
        .sampler = {0},
      },
    };
    bind_group_layouts.instance_buffer = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.instance_buffer != NULL)
  }

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  WGPUBindGroupLayout bindGroupLayouts[2] = {
    bind_group_layouts.frame_constants, // set 0
    bind_group_layouts.instance_buffer, // set 1
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
  // Frame constants bind group
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .sampler = linear_sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .textureView = textures.color.view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .textureView = textures.heightmap.view,
      }
    };
    WGPUBindGroupDescriptor bg_desc = {
      .layout     = bind_group_layouts.frame_constants,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.frame_constants
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.frame_constants != NULL)
  }

  // Instance buffer bind group
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = instance_buffer,
        .offset = 0,
        .size = instance_buffer_length,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
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
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth32Float,
      .depth_write_enabled = true,
    });
  depth_stencil_state_desc.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    terrain_mesh, 24,
    /* Attribute descriptions */
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, 0),
    // Attribute location 1: Texture coordinates
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 16))

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader SPIR-V
                      .file = "shaders/terrain_mesh/shader.vert.spv",
                    },
                    .buffer_count = 1,
                    .buffers = &terrain_mesh_vertex_buffer_layout,
                  });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader SPIR-V
                      .file = "shaders/terrain_mesh/shader.frag.spv",
                    },
                    .target_count = 1,
                    .targets = &color_target_state_desc,
                  });

  // Multisample state
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "terrain_mesh_render_pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state_desc,
                            .vertex       = vertex_state_desc,
                            .fragment     = &fragment_state_desc,
                            .depthStencil = &depth_stencil_state_desc,
                            .multisample  = multisample_state_desc,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
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
    return 0;
  }

  return 1;
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
                                       vertices.buffer, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.frame_constants, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 1,
                                    bind_groups.instance_buffer, 0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(wgpu_context->rpass_enc, indices.buffer,
                                      WGPUIndexFormat_Uint32, 0, 0);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                   (uint32_t)PATCH_INDEX_COUNT, instance_count,
                                   0, 0, 0);

  // Create command buffer and cleanup
  wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
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

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
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
  WGPU_RELEASE_RESOURCE(Buffer, instance_buffer)
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
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
  });
  // clang-format on
}
