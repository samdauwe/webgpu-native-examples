#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include "sokol_fetch.h"

#define SOKOL_LOG_IMPL
#include "sokol_log.h"

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

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

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* terrain_mesh_vertex_shader_wgsl;
static const char* terrain_mesh_fragment_shader_wgsl;

/* Terrain patch parameters */
// clang-format off
#define PATCH_SIZE (50)
#define PATCH_SEGMENT_COUNT (40)
#define PATCH_INDEX_COUNT (PATCH_SEGMENT_COUNT* PATCH_SEGMENT_COUNT * 6)
#define PATCH_VERTEX_COUNT ((PATCH_SEGMENT_COUNT + 1) * (PATCH_SEGMENT_COUNT + 1))
#define PATCH_FLOATS_PER_VERTEX (6)
// clang-format on

/* State struct */
static struct {
  /* Camera parameters */
  struct {
    float fov_y;
    float near_z;
    float far_z;
    vec3 position;
    float heading;
    float target_heading;
    float angular_easing_factor;
    float speed;
  } camera;
  /* Camera matrices */
  struct {
    float model[16];
    float model_view[16];
    float model_view_projection[16];
  } camera_mtx;
  /* Used to calculate view and projection matrices */
  float rot_y[16];
  float trans[16];
  float view_matrix[16];
  float projection_matrix[16];
  /* Nine terrain patches */
  vec3 patch_centers[9];
  /* Time-related state */
  float last_frame_time;
  float direction_change_countdown;
  /* Internal constants */
  uint32_t
    instance_length; /* Length of the data associated with a single instance */
  uint32_t max_instance_count;
  uint64_t instance_buffer_length; /* In bytes */
  float* instance_data;
  uint32_t instance_count;
  /* Vertex buffer */
  wgpu_buffer_t vertices;
  /* Index buffer */
  wgpu_buffer_t indices;
  /* Instance buffer */
  wgpu_buffer_t instance_buffer;
  /* Textures */
  struct {
    wgpu_texture_t color;
    wgpu_texture_t heightmap;
  } textures;
  WGPUSampler linear_sampler;
  uint8_t file_buffers[2][512 * 512 * 4];
  /* Render pipeline + layout */
  WGPURenderPipeline render_pipeline;
  WGPUPipelineLayout pipeline_layout;
  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout frame_constants;
    WGPUBindGroupLayout instance_buffer;
  } bind_group_layouts;
  /* Bind groups */
  struct {
    WGPUBindGroup frame_constants;
    WGPUBindGroup instance_buffer;
  } bind_groups;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  int8_t initialized;
} state = {
  .camera = {
    .fov_y                 = TO_RADIANS(60.0f),
    .near_z                = 0.1f,
    .far_z                 = 150.0f,
    .position              = {0.0f, 5.0f, 0.0f},
    .heading               = PI / 2.0f, // radians */
    .target_heading        = PI / 2.0f, /* radians */
    .angular_easing_factor = 0.005f,
    .speed                 = 8.0f,      /* meters per second */
  },
  .last_frame_time            = -1.0f,
  .direction_change_countdown =  6.0f, /* seconds */
  .instance_length            =  16 * 2,
  .max_instance_count         =  9,
  .instance_count             =  1,
  .color_attachment = {
     .loadOp     = WGPULoadOp_Clear,
     .storeOp    = WGPUStoreOp_Store,
     .clearValue = {0.812, 0.914, 1.0, 1.0},
     .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_dscriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  }
};

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

static void init_patch_mesh(wgpu_context_t* wgpu_context)
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
  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Terrain mesh - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertices_data),
                    .count = (uint32_t)ARRAY_SIZE(vertices_data),
                    .initial.data = vertices_data,
                  });

  /* Create index buffer */
  state.indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Terrain mesh - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(indices_data),
                    .count = (uint32_t)ARRAY_SIZE(indices_data),
                    .initial.data = indices_data,
                  });
}

static void init_textures(wgpu_context_t* wgpu_context)
{
  /* Color texture */
  state.textures.color = wgpu_create_color_bars_texture(wgpu_context, 16, 16);

  /* Heightmap texture */
  state.textures.heightmap
    = wgpu_create_color_bars_texture(wgpu_context, 16, 16);

  /* Linear sampler */
  WGPUSamplerDescriptor sampler_desc = {
    .label         = STRVIEW("Color texture - Linear sampler"),
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
  state.linear_sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
}

/**
 * @brief The fetch-callback is called by sokol_fetch.h when the data is loaded,
 * or when an error has occurred.
 */
static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("File fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* The file data has been fetched, since we provided a big-enough buffer we
   * can be sure that all data has been loaded here */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc = (wgpu_texture_desc_t){
        .extent = (WGPUExtent3D) {
          .width              = img_width,
          .height             = img_height,
          .depthOrArrayLayers = 4,
        },
        .format = WGPUTextureFormat_RGBA8Unorm,
        .pixels = {
          .ptr  = pixels,
          .size = img_width * img_height * 4,
        },
      };
    texture->desc.is_dirty = true;
  }
}

static void fetch_textures(void)
{
  const char* tex_paths[2] = {
    "assets/textures/color.png",     /* Color texture */
    "assets/textures/heightmap.png", /* Heightmap texture */
  };

  for (uint32_t i = 0; i < ARRAY_SIZE(tex_paths); ++i) {
    wgpu_texture_t* texture
      = (i == 0) ? &state.textures.color : &state.textures.heightmap;
    /* Start loading the image file */
    sfetch_send(&(sfetch_request_t){
      .path      = tex_paths[i],
      .callback  = fetch_callback,
      .buffer    = SFETCH_RANGE(state.file_buffers[i]),
      .user_data = {
        .ptr = &texture,
        .size = sizeof(wgpu_texture_t*),
      },
    });
  }
}

static void update_camera_pose(float dt)
{
  /* Update camera position */
  const float dx = -sin(state.camera.heading) * state.camera.speed * dt;
  const float dz = -cos(state.camera.heading) * state.camera.speed * dt;
  state.camera.position[0] += dx;
  state.camera.position[2] += dz;

  /* Update camera direction, choosing a new direction if needed */
  state.camera.heading += (state.camera.target_heading - state.camera.heading)
                          * state.camera.angular_easing_factor;
  if (state.direction_change_countdown < 0.0f) {
    state.camera.target_heading
      = (random_float_min_max(0.0f, 1.0f) * PI * 2.0f) - PI;
    state.direction_change_countdown = 6.0f;
  }
  state.direction_change_countdown -= dt;
}

static void update_uniform_buffer(wgpu_context_t* wgpu_context)
{
  const float frame_timestamp_millis = stm_ms(stm_now());
  const float dt
    = (frame_timestamp_millis - state.last_frame_time) * 0.001; // s
  state.last_frame_time = frame_timestamp_millis;

  update_camera_pose(dt);

  const float patch_size = (float)PATCH_SIZE;

  // Determine the nearest nine terrain patches and calculate their positions
  const vec3 nearest_patch_center = {
    round(state.camera.position[0] / patch_size) * patch_size, /* x */
    0.0f,                                                      /* y */
    round(state.camera.position[2] / patch_size) * patch_size  /* z */
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
        state.patch_centers[patch_index]);
      ++patch_index;
    }
  }

  // Calculate view and projection matrices
  mat4_rotation_y(&state.rot_y, -state.camera.heading);
  mat4_translation(&state.trans,
                   (vec3){-state.camera.position[0], -state.camera.position[1],
                          -state.camera.position[2]});
  mat4_mul(&state.rot_y, &state.trans, &state.view_matrix);
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;
  mat4_perspective_fov(state.camera.fov_y, aspect_ratio, state.camera.near_z,
                       state.camera.far_z, &state.projection_matrix);

  // Calculate the per-instance matrices
  if (state.instance_data == NULL) {
    state.instance_data
      = calloc(state.instance_buffer_length / 4, sizeof(float));
  }
  state.instance_count
    = MIN(ARRAY_SIZE(state.patch_centers), state.max_instance_count);
  for (uint32_t i = 0; i < state.instance_count; ++i) {
    mat4_translation(&state.camera_mtx.model, state.patch_centers[i]);
    mat4_mul(&state.view_matrix, &state.camera_mtx.model,
             &state.camera_mtx.model_view);
    mat4_mul(&state.projection_matrix, &state.camera_mtx.model_view,
             &state.camera_mtx.model_view_projection);
    memcpy(state.instance_data + (i * state.instance_length),
           state.camera_mtx.model_view, sizeof(state.camera_mtx.model_view));
    memcpy(state.instance_data + (i * state.instance_length + 16),
           state.camera_mtx.model_view_projection,
           sizeof(state.camera_mtx.model_view_projection));
  }

  /* Write the instance data to the instance buffer */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.instance_buffer.buffer, 0,
                       state.instance_data,
                       (state.instance_buffer_length / 4) * sizeof(float));
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.instance_buffer_length
    = 4 * state.instance_length * state.max_instance_count;
  state.instance_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Instance - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = state.instance_buffer_length,
                  });
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
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
    state.bind_group_layouts.frame_constants = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Frame constants - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(state.bind_group_layouts.frame_constants != NULL)
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
          .minBindingSize   = state.instance_buffer_length,
        },
        .sampler = {0},
      },
    };
    state.bind_group_layouts.instance_buffer = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Instance buffer - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(state.bind_group_layouts.instance_buffer != NULL)
  }

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  WGPUBindGroupLayout bindGroupLayouts[2] = {
    state.bind_group_layouts.frame_constants, /* Set 0 */
    state.bind_group_layouts.instance_buffer, /* Set 1 */
  };
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bindGroupLayouts),
      .bindGroupLayouts     = bindGroupLayouts,
    });
  ASSERT(state.pipeline_layout != NULL)
}

static void init_frame_constants_bind_group(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.frame_constants)

  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .sampler = state.linear_sampler,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding     = 1,
      .textureView = state.textures.color.view,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = state.textures.heightmap.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Frame constants - Bind group"),
    .layout     = state.bind_group_layouts.frame_constants,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.bind_groups.frame_constants
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.bind_groups.frame_constants != NULL)
}

static void init_instance_buffer_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = state.instance_buffer.buffer,
      .offset  = 0,
      .size    = state.instance_buffer.size,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Instance buffer - Bind group"),
    .layout     = state.bind_group_layouts.instance_buffer,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.bind_groups.instance_buffer
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.bind_groups.instance_buffer != NULL)
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  init_frame_constants_bind_group(wgpu_context);
  init_instance_buffer_bind_group(wgpu_context);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, terrain_mesh_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, terrain_mesh_fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
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

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Terrain mesh - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &terrain_mesh_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_Back,
      .frontFace = WGPUFrontFace_CCW
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.render_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.render_pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 2,
      .num_channels = 1,
      .num_lanes    = 1,
      .logger.func  = slog_func,
    });
    init_patch_mesh(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_textures(wgpu_context);
    fetch_textures();
    init_pipeline_layout(wgpu_context);
    init_pipeline(wgpu_context);
    init_bind_groups(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void update_textures(struct wgpu_context_t* wgpu_context)
{
  if (!state.textures.color.desc.is_dirty
      || !state.textures.heightmap.desc.is_dirty) {
    return;
  }

  /* Recreate color texture */
  wgpu_recreate_texture(wgpu_context, &state.textures.color);
  FREE_TEXTURE_PIXELS(state.textures.color);

  /* Recreate heightmap texture */
  wgpu_recreate_texture(wgpu_context, &state.textures.heightmap);
  FREE_TEXTURE_PIXELS(state.textures.heightmap);

  /* Upddate the bind group */
  init_frame_constants_bind_group(wgpu_context);
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Update texture when pixel data loaded */
  update_textures(wgpu_context);

  /* Update matrix data */
  update_uniform_buffer(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_dscriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.render_pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                    state.bind_groups.frame_constants, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 1,
                                    state.bind_groups.instance_buffer, 0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.indices.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rpass_enc, (uint32_t)PATCH_INDEX_COUNT,
                                   state.instance_count, 0, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  if (state.instance_data != NULL) {
    free(state.instance_data);
    state.instance_data = NULL;
  }

  wgpu_destroy_texture(&state.textures.color);
  wgpu_destroy_texture(&state.textures.heightmap);
  WGPU_RELEASE_RESOURCE(Sampler, state.linear_sampler)

  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.instance_buffer.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.bind_group_layouts.frame_constants)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.bind_group_layouts.instance_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.frame_constants)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.instance_buffer)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Terrain Mesh",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* terrain_mesh_vertex_shader_wgsl = CODE(
  const NUM_INSTANCES: u32 = 9;

  struct Instance {
    modelViewMatrix: mat4x4<f32>,
    modelViewProjectionMatrix: mat4x4<f32>,
  };

  struct Uniforms {
    instances: array<Instance, NUM_INSTANCES>,
  };

  @group(0) @binding(0) var linearSampler: sampler;
  @group(0) @binding(1) var colorTexture: texture_2d<f32>;
  @group(0) @binding(2) var heightmap: texture_2d<f32>;
  @group(1) @binding(0) var<uniform> uniforms: Uniforms;

  struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) inTexCoords: vec2<f32>,
    @builtin(instance_index) instanceIndex: u32,
  };

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) eyePosition: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) outTexCoords: vec2<f32>,
  };

  @vertex
  fn main(input: VertexInput) -> VertexOutput {
    // Displacement mapping constants
    let patchSize = 50.0;
    let heightScale = 8.0;
    let d = vec3<f32>(1.0 / 256.0, 1.0 / 256.0, 0.0);
    let dydy = heightScale / patchSize;

    // Calculate displacement and differentials (for normal calculation)
    let height = textureSampleLevel(heightmap, linearSampler, input.inTexCoords, 0.0f).x;
    let dydx = height - textureSampleLevel(heightmap, linearSampler, input.inTexCoords + d.xz, 0.0f).x;
    let dydz = height - textureSampleLevel(heightmap, linearSampler, input.inTexCoords + d.zy, 0.0f).x;

    // Calculate model-space vertex position and normal
    let modelPosition = vec4<f32>(input.position.x, input.position.y + height * heightScale, input.position.z, 1.0);
    let modelNormal = vec4<f32>(normalize(vec3<f32>(dydx, dydy, dydz)), 0.0);

    // Retrieve MV and MVP matrices from instance data
    let modelViewMatrix = uniforms.instances[input.instanceIndex].modelViewMatrix;
    let modelViewProjectionMatrix = uniforms.instances[input.instanceIndex].modelViewProjectionMatrix;

    var output: VertexOutput;
    output.position = modelViewProjectionMatrix * modelPosition;
    output.eyePosition = modelViewMatrix * modelPosition;
    output.normal = (modelViewMatrix * modelNormal).xyz;
    output.outTexCoords = input.inTexCoords;
    return output;
  }
);

static const char* terrain_mesh_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var linearSampler: sampler;
  @group(0) @binding(1) var colorTexture: texture_2d<f32>;

  struct FragmentInput {
    @location(0) eyePosition: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoords: vec2<f32>,
  };

  @fragment
  fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Lighting constants
    let lightColor = 2.0 * vec3<f32>(0.812, 0.914, 1.0);
    let L = normalize(vec3<f32>(1.0, 1.0, 1.0));

    // Determine diffuse lighting contribution
    let N = normalize(input.normal);
    let diffuseFactor = clamp(dot(N, L), 0.0, 1.0);

    let texCoordScale = 4.0;
    let baseColor = textureSample(colorTexture, linearSampler,
                                  input.texCoords * texCoordScale).rgb;

    let litColor = diffuseFactor * lightColor * baseColor;

    // Fog constants
    let fogColor = vec3<f32>(0.812, 0.914, 1.0);
    let fogStart = 3.0;
    let fogEnd = 50.0;

    // Calculate fog factor from eye space distance
    let fogDist = length(input.eyePosition.xyz);
    let fogFactor = clamp((fogEnd - fogDist) / (fogEnd - fogStart), 0.0, 1.0);

    // Blend lit color and fog color to get fragment color
    let finalColor = (fogColor * (1.0 - fogFactor)) + (litColor * fogFactor);
    return vec4<f32>(finalColor, 1.0);
  }
);
// clang-format on
