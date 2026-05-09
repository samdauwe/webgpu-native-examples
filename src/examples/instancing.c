#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include "core/camera.h"
#include "core/gltf_model.h"
#include "core/image_loader.h"

#include <cglm/cglm.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Instanced Mesh Rendering
 *
 * Renders thousands of asteroid rocks orbiting a lava planet using GPU
 * instancing. Each rock instance uses a separate per-instance vertex buffer
 * containing position, rotation, scale, and texture array layer index.
 * Three render pipelines handle the star field backdrop (procedural vertex
 * shader), the planet (single 2D texture), and the instanced rocks (2D
 * texture array).
 *
 * Ported from Sascha Willems' Vulkan example "instancing"
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/instancing
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (declared here, defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* instancing_rocks_shader_wgsl;
static const char* instancing_planet_shader_wgsl;
static const char* instancing_starfield_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define INSTANCE_COUNT (8192u)

/* Rock texture atlas: 5 layers of 512×512 RGBA stacked vertically */
#define ROCKS_LAYER_SIZE (512u)
#define ROCKS_LAYER_COUNT (5u)

/* Planet texture: single 512×512 RGBA */
#define PLANET_TEXTURE_SIZE (512u)

/* File buffer sizes (for PNG-compressed data) */
#define PLANET_FILE_BUFFER_SIZE (640u * 1024u)      /* ~640 KB  */
#define ROCKS_FILE_BUFFER_SIZE (3u * 1024u * 1024u) /* ~3 MB    */

// clang-format off
static const char* rock_model_path   = "assets/models/rock01.gltf";
static const char* planet_model_path = "assets/models/lavaplanet.gltf";
static const char* rocks_tex_path    = "assets/textures/texturearray_rocks_rgba.png";
static const char* planet_tex_path   = "assets/textures/lavaplanet_rgba.png";
// clang-format on

/* -------------------------------------------------------------------------- *
 * Data layouts
 * -------------------------------------------------------------------------- */

/* Per-instance data placed in its own vertex buffer (binding 1) */
typedef struct {
  vec3 pos;           /* World-space position for this instance         */
  vec3 rot;           /* Rotation seed angles (x, y, z)                 */
  float scale;        /* Uniform scale                                  */
  uint32_t tex_index; /* Index into the rocks texture array (0 .. 4)    */
} instance_data_t;    /* 32 bytes total                                 */

/* Uniform buffer – must match WGSL struct layout exactly.
 * Offsets (WGSL std140 / WebGPU alignment):
 *   projection  @ 0    (64 bytes)
 *   view        @ 64   (64 bytes)
 *   lightPos    @ 128  (16 bytes)
 *   locSpeed    @ 144  (4 bytes)
 *   globSpeed   @ 148  (4 bytes)
 *   _pad        @ 152  (8 bytes) → struct size = 160 bytes
 */
typedef struct {
  mat4 projection;
  mat4 view;
  vec4 light_pos;
  float loc_speed;
  float glob_speed;
  float _pad[2];
} uniform_data_t; /* 160 bytes */

/* -------------------------------------------------------------------------- *
 * Global state
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* ---- Models ---------------------------------------------------------- */
  struct {
    gltf_model_t rock;
    gltf_model_t planet;
    bool rock_loaded;
    bool planet_loaded;
  } models;

  /* GPU geometry buffers (one vertex + one index buffer per model) */
  struct {
    struct {
      WGPUBuffer vertex;
      WGPUBuffer index;
    } rock;
    struct {
      WGPUBuffer vertex;
      WGPUBuffer index;
    } planet;
  } model_buffers;

  /* ---- Instance data ---------------------------------------------------- */
  WGPUBuffer instance_buffer;

  /* ---- Textures --------------------------------------------------------- */
  wgpu_texture_t rocks_texture;  /* texture_2d_array, 5 layers               */
  wgpu_texture_t planet_texture; /* texture_2d, single layer                 */

  /* Async-load staging data */
  uint8_t* rocks_file_buffer;
  uint8_t* planet_file_buffer;

  bool rocks_texture_loaded;
  bool planet_texture_loaded;

  /* ---- Uniform buffer --------------------------------------------------- */
  WGPUBuffer uniform_buffer;
  uniform_data_t ubo;

  /* ---- Bind group layouts ----------------------------------------------- */
  WGPUBindGroupLayout bgl_ubo_only; /* starfield: binding 0 = UBO only       */
  WGPUBindGroupLayout bgl_static;   /* planet: binding 0=UBO, 1=sampler,     */
                                    /*         binding 2=texture_2d          */
  WGPUBindGroupLayout bgl_rocks;    /* rocks:  binding 0=UBO, 1=sampler,     */
                                    /*         binding 2=texture_2d_array    */

  /* ---- Bind groups ------------------------------------------------------ */
  WGPUBindGroup bg_starfield; /* UBO only                                    */
  WGPUBindGroup bg_planet;    /* UBO + sampler + planet texture              */
  WGPUBindGroup bg_rocks;     /* UBO + sampler + rocks texture array         */

  /* ---- Pipeline layouts ------------------------------------------------- */
  WGPUPipelineLayout pl_starfield;
  WGPUPipelineLayout pl_static;
  WGPUPipelineLayout pl_rocks;

  /* ---- Pipelines -------------------------------------------------------- */
  WGPURenderPipeline pipeline_starfield;
  WGPURenderPipeline pipeline_planet;
  WGPURenderPipeline pipeline_rocks;

  /* ---- Render pass ------------------------------------------------------ */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* ---- Timing ----------------------------------------------------------- */
  uint64_t last_frame_time;

  bool paused;
  WGPUBool initialized;
} state = {
  .ubo = {
    .light_pos  = {0.0f, 5.0f, 0.0f, 1.0f}, /* WebGPU Y-up: light above    */
    .loc_speed  = 0.0f,
    .glob_speed = 0.0f,
  },
  .color_attachment = {
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.2f, 1.0f},
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
};

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_models(void)
{
  gltf_model_desc_t desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
  };

  state.models.rock_loaded = gltf_model_load_from_file_ext(
    &state.models.rock, rock_model_path, 1.0f, &desc);
  if (!state.models.rock_loaded) {
    printf("[instancing] Failed to load rock model: %s\n", rock_model_path);
  }

  state.models.planet_loaded = gltf_model_load_from_file_ext(
    &state.models.planet, planet_model_path, 1.0f, &desc);
  if (!state.models.planet_loaded) {
    printf("[instancing] Failed to load planet model: %s\n", planet_model_path);
  }
}

static void create_model_buffers(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ---- Rock model buffers ---------------------------------------------- */
  if (state.models.rock_loaded) {
    uint32_t vb_size
      = state.models.rock.vertex_count * (uint32_t)sizeof(gltf_vertex_t);
    state.model_buffers.rock.vertex = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Rock vertex buffer"),
                .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                .size  = vb_size,
                .mappedAtCreation = false,
              });
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model_buffers.rock.vertex,
                         0, state.models.rock.vertices, vb_size);

    uint32_t ib_size
      = state.models.rock.index_count * (uint32_t)sizeof(uint32_t);
    state.model_buffers.rock.index = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Rock index buffer"),
                .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                .size  = ib_size,
                .mappedAtCreation = false,
              });
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model_buffers.rock.index, 0,
                         state.models.rock.indices, ib_size);
  }

  /* ---- Planet model buffers -------------------------------------------- */
  if (state.models.planet_loaded) {
    uint32_t vb_size
      = state.models.planet.vertex_count * (uint32_t)sizeof(gltf_vertex_t);
    state.model_buffers.planet.vertex = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Planet vertex buffer"),
                .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                .size  = vb_size,
                .mappedAtCreation = false,
              });
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model_buffers.planet.vertex,
                         0, state.models.planet.vertices, vb_size);

    uint32_t ib_size
      = state.models.planet.index_count * (uint32_t)sizeof(uint32_t);
    state.model_buffers.planet.index = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Planet index buffer"),
                .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                .size  = ib_size,
                .mappedAtCreation = false,
              });
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model_buffers.planet.index,
                         0, state.models.planet.indices, ib_size);
  }
}

/* -------------------------------------------------------------------------- *
 * Instance data
 * -------------------------------------------------------------------------- */

static float uniform_rand(void)
{
  return (float)rand() / ((float)RAND_MAX + 1.0f);
}

static void prepare_instance_data(wgpu_context_t* wgpu_context)
{
  srand((unsigned int)time(NULL));

  instance_data_t* instances
    = (instance_data_t*)malloc(INSTANCE_COUNT * sizeof(instance_data_t));
  if (!instances) {
    printf("[instancing] Failed to allocate instance data\n");
    return;
  }

  const float pi = 3.14159265358979323846f;

  /* Distribute rocks on two concentric rings in the XZ plane */
  const float ring0_min = 7.0f, ring0_max = 11.0f;
  const float ring1_min = 14.0f, ring1_max = 18.0f;

  for (uint32_t i = 0; i < INSTANCE_COUNT / 2; ++i) {
    float rho, theta;

    /* Inner ring */
    rho = sqrtf((ring0_max * ring0_max - ring0_min * ring0_min) * uniform_rand()
                + ring0_min * ring0_min);
    theta                  = 2.0f * pi * uniform_rand();
    instances[i].pos[0]    = rho * cosf(theta);
    instances[i].pos[1]    = uniform_rand() * 0.5f - 0.25f;
    instances[i].pos[2]    = rho * sinf(theta);
    instances[i].rot[0]    = pi * uniform_rand();
    instances[i].rot[1]    = pi * uniform_rand();
    instances[i].rot[2]    = pi * uniform_rand();
    instances[i].scale     = (1.5f + uniform_rand() - uniform_rand()) * 0.75f;
    instances[i].tex_index = (uint32_t)(uniform_rand() * ROCKS_LAYER_COUNT);
    if (instances[i].tex_index >= ROCKS_LAYER_COUNT) {
      instances[i].tex_index = ROCKS_LAYER_COUNT - 1;
    }

    /* Outer ring */
    rho = sqrtf((ring1_max * ring1_max - ring1_min * ring1_min) * uniform_rand()
                + ring1_min * ring1_min);
    theta                  = 2.0f * pi * uniform_rand();
    uint32_t j             = i + INSTANCE_COUNT / 2;
    instances[j].pos[0]    = rho * cosf(theta);
    instances[j].pos[1]    = uniform_rand() * 0.5f - 0.25f;
    instances[j].pos[2]    = rho * sinf(theta);
    instances[j].rot[0]    = pi * uniform_rand();
    instances[j].rot[1]    = pi * uniform_rand();
    instances[j].rot[2]    = pi * uniform_rand();
    instances[j].scale     = (1.5f + uniform_rand() - uniform_rand()) * 0.75f;
    instances[j].tex_index = (uint32_t)(uniform_rand() * ROCKS_LAYER_COUNT);
    if (instances[j].tex_index >= ROCKS_LAYER_COUNT) {
      instances[j].tex_index = ROCKS_LAYER_COUNT - 1;
    }
  }

  /* Upload to GPU */
  uint64_t buf_size     = INSTANCE_COUNT * sizeof(instance_data_t);
  state.instance_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Instance data buffer"),
      .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size             = buf_size,
      .mappedAtCreation = false,
    });
  wgpuQueueWriteBuffer(wgpu_context->queue, state.instance_buffer, 0, instances,
                       buf_size);

  free(instances);
}

/* -------------------------------------------------------------------------- *
 * Texture loading (async via sokol_fetch)
 * -------------------------------------------------------------------------- */

/* Forward declarations */
static void init_bind_groups(wgpu_context_t* wgpu_context);
static void update_bind_groups(wgpu_context_t* wgpu_context);

static void fetch_rocks_texture_cb(const sfetch_response_t* resp)
{
  if (!resp->fetched) {
    printf("[instancing] Rocks texture fetch failed, error: %d\n",
           resp->error_code);
    return;
  }

  int w, h, ch;
  uint8_t* pixels = image_pixels_from_memory(
    resp->data.ptr, (int)resp->data.size, &w, &h, &ch, 4);

  if (!pixels) {
    printf("[instancing] Failed to decode rocks texture\n");
    return;
  }

  const int exp_w = (int)ROCKS_LAYER_SIZE;
  const int exp_h = (int)(ROCKS_LAYER_SIZE * ROCKS_LAYER_COUNT);
  if (w != exp_w || h != exp_h) {
    printf("[instancing] Rocks texture size mismatch: %dx%d (expected %dx%d)\n",
           w, h, exp_w, exp_h);
    image_free(pixels);
    return;
  }

  state.rocks_texture.desc = (wgpu_texture_desc_t){
    .extent = (WGPUExtent3D){
      .width              = (uint32_t)ROCKS_LAYER_SIZE,
      .height             = (uint32_t)ROCKS_LAYER_SIZE,
      .depthOrArrayLayers = (uint32_t)ROCKS_LAYER_COUNT,
    },
    .format = WGPUTextureFormat_RGBA8Unorm,
    .pixels = {
      .ptr  = pixels,
      .size = (size_t)ROCKS_LAYER_SIZE * ROCKS_LAYER_SIZE * ROCKS_LAYER_COUNT * 4,
    },
    .generate_mipmaps      = 1,
    .mipmap_view_dimension = WGPU_MIPMAP_VIEW_2D_ARRAY,
  };
  state.rocks_texture.desc.is_dirty = true;
}

static void fetch_planet_texture_cb(const sfetch_response_t* resp)
{
  if (!resp->fetched) {
    printf("[instancing] Planet texture fetch failed, error: %d\n",
           resp->error_code);
    return;
  }

  int w, h, ch;
  uint8_t* pixels = image_pixels_from_memory(
    resp->data.ptr, (int)resp->data.size, &w, &h, &ch, 4);

  if (!pixels) {
    printf("[instancing] Failed to decode planet texture\n");
    return;
  }

  if (w != (int)PLANET_TEXTURE_SIZE || h != (int)PLANET_TEXTURE_SIZE) {
    printf("[instancing] Planet texture size mismatch: %dx%d\n", w, h);
    image_free(pixels);
    return;
  }

  state.planet_texture.desc = (wgpu_texture_desc_t){
    .extent = (WGPUExtent3D){
      .width              = (uint32_t)PLANET_TEXTURE_SIZE,
      .height             = (uint32_t)PLANET_TEXTURE_SIZE,
      .depthOrArrayLayers = 1,
    },
    .format = WGPUTextureFormat_RGBA8Unorm,
    .pixels = {
      .ptr  = pixels,
      .size = (size_t)PLANET_TEXTURE_SIZE * PLANET_TEXTURE_SIZE * 4,
    },
    .generate_mipmaps      = 1,
    .mipmap_view_dimension = WGPU_MIPMAP_VIEW_2D,
  };
  state.planet_texture.desc.is_dirty = true;
}

static void init_textures(wgpu_context_t* wgpu_context)
{
  /* ---- Rocks texture array placeholder (1×1 per layer) ----------------- */
  {
    uint8_t placeholder[4 * ROCKS_LAYER_COUNT];
    memset(placeholder, 64, sizeof(placeholder));
    state.rocks_texture = wgpu_create_texture(
      wgpu_context,
      &(wgpu_texture_desc_t){
        .extent = {1, 1, ROCKS_LAYER_COUNT},
        .format = WGPUTextureFormat_RGBA8Unorm,
        .pixels = {.ptr = placeholder, .size = sizeof(placeholder)},
        .mipmap_view_dimension = WGPU_MIPMAP_VIEW_2D_ARRAY,
      });

    sfetch_send(&(sfetch_request_t){
      .path     = rocks_tex_path,
      .callback = fetch_rocks_texture_cb,
      .buffer
      = {.ptr = state.rocks_file_buffer, .size = ROCKS_FILE_BUFFER_SIZE},
    });
  }

  /* ---- Planet texture placeholder (1×1) -------------------------------- */
  {
    uint8_t placeholder[4] = {64, 64, 64, 255};
    state.planet_texture   = wgpu_create_texture(
      wgpu_context,
      &(wgpu_texture_desc_t){
          .extent = {1, 1, 1},
          .format = WGPUTextureFormat_RGBA8Unorm,
          .pixels = {.ptr = placeholder, .size = sizeof(placeholder)},
          .mipmap_view_dimension = WGPU_MIPMAP_VIEW_2D,
      });

    sfetch_send(&(sfetch_request_t){
      .path     = planet_tex_path,
      .callback = fetch_planet_texture_cb,
      .buffer
      = {.ptr = state.planet_file_buffer, .size = PLANET_FILE_BUFFER_SIZE},
    });
  }
}

static void update_textures(wgpu_context_t* wgpu_context)
{
  /* ---- Update rocks texture if new data arrived ------------------------ */
  if (state.rocks_texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.rocks_texture);

    if (state.rocks_texture.desc.pixels.ptr) {
      image_free((void*)state.rocks_texture.desc.pixels.ptr);
      state.rocks_texture.desc.pixels.ptr  = NULL;
      state.rocks_texture.desc.pixels.size = 0;
    }

    /* Rebind */
    if (state.bg_rocks) {
      wgpuBindGroupRelease(state.bg_rocks);
      state.bg_rocks = NULL;
    }
    state.rocks_texture_loaded = true;
    update_bind_groups(wgpu_context);
  }

  /* ---- Update planet texture if new data arrived ----------------------- */
  if (state.planet_texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.planet_texture);

    if (state.planet_texture.desc.pixels.ptr) {
      image_free((void*)state.planet_texture.desc.pixels.ptr);
      state.planet_texture.desc.pixels.ptr  = NULL;
      state.planet_texture.desc.pixels.size = 0;
    }

    /* Rebind */
    if (state.bg_planet) {
      wgpuBindGroupRelease(state.bg_planet);
      state.bg_planet = NULL;
    }
    state.planet_texture_loaded = true;
    update_bind_groups(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Instancing uniform buffer"),
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = sizeof(uniform_data_t),
      .mappedAtCreation = false,
    });
}

static void update_uniform_buffer(wgpu_context_t* wgpu_context, float dt)
{
  camera_update(&state.camera, dt);

  glm_mat4_copy(state.camera.matrices.perspective, state.ubo.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo.view);

  if (!state.paused) {
    state.ubo.loc_speed += dt * 0.35f;
    state.ubo.glob_speed += dt * 0.01f;
  }

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0, &state.ubo,
                       sizeof(uniform_data_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ---- Starfield: UBO only (binding 0) --------------------------------- */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniform_data_t),
        },
      },
    };
    state.bgl_ubo_only = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("BGL - UBO only"),
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* ---- Planet/static: UBO + sampler + texture_2d ----------------------- */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer     = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniform_data_t),
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = (WGPUTextureBindingLayout){
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
    };
    state.bgl_static = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("BGL - static (planet)"),
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* ---- Rocks: UBO + sampler + texture_2d_array ------------------------- */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer     = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniform_data_t),
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = (WGPUTextureBindingLayout){
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2DArray,
          .multisampled  = false,
        },
      },
    };
    state.bgl_rocks = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("BGL - rocks (texture array)"),
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ---- Starfield bind group: UBO only ---------------------------------- */
  {
    WGPUBindGroupEntry entries[1] = {
      [0] = {
        .binding = 0,
        .buffer  = state.uniform_buffer,
        .offset  = 0,
        .size    = sizeof(uniform_data_t),
      },
    };
    state.bg_starfield = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("BG - starfield"),
                .layout     = state.bgl_ubo_only,
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* ---- Planet bind group: UBO + sampler + planet texture --------------- */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.uniform_buffer,
        .offset  = 0,
        .size    = sizeof(uniform_data_t),
      },
      [1] = {
        .binding = 1,
        .sampler = state.planet_texture.sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.planet_texture.view,
      },
    };
    state.bg_planet = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("BG - planet"),
                .layout     = state.bgl_static,
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* ---- Rocks bind group: UBO + sampler + rocks texture array ----------- */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.uniform_buffer,
        .offset  = 0,
        .size    = sizeof(uniform_data_t),
      },
      [1] = {
        .binding = 1,
        .sampler = state.rocks_texture.sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.rocks_texture.view,
      },
    };
    state.bg_rocks = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("BG - rocks"),
                .layout     = state.bgl_rocks,
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

/* Called when a texture is (re)loaded to recreate the affected bind group */
static void update_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Recreate planet bind group when planet texture changes */
  if (!state.bg_planet && state.planet_texture.view) {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.uniform_buffer,
        .offset  = 0,
        .size    = sizeof(uniform_data_t),
      },
      [1] = {
        .binding = 1,
        .sampler = state.planet_texture.sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.planet_texture.view,
      },
    };
    state.bg_planet = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("BG - planet"),
                .layout     = state.bgl_static,
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Recreate rocks bind group when rocks texture changes */
  if (!state.bg_rocks && state.rocks_texture.view) {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.uniform_buffer,
        .offset  = 0,
        .size    = sizeof(uniform_data_t),
      },
      [1] = {
        .binding = 1,
        .sampler = state.rocks_texture.sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.rocks_texture.view,
      },
    };
    state.bg_rocks = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("BG - rocks"),
                .layout     = state.bgl_rocks,
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Use the framework-managed depth format so it always matches the
   * swapchain depth view (wgpu_context->depth_stencil_view). */
  const WGPUTextureFormat depth_fmt = wgpu_context->depth_stencil_format;

  /* Shared depth-stencil for opaque (planet + rocks) */
  WGPUDepthStencilState depth_opaque = {
    .format            = depth_fmt,
    .depthWriteEnabled = WGPUOptionalBool_True,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
  };

  /* Depth-stencil for starfield (background – no depth write) */
  WGPUDepthStencilState depth_starfield = {
    .format            = depth_fmt,
    .depthWriteEnabled = WGPUOptionalBool_False,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
  };

  WGPUBlendState blend        = wgpu_create_blend_state(false);
  WGPUColorTargetState target = {
    .format    = wgpu_context->render_format,
    .blend     = &blend,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* ================================================================
   * 1. Starfield pipeline (no vertex buffers, no bind groups)
   * ================================================================ */
  {
    state.pl_starfield = wgpuDeviceCreatePipelineLayout(
      device, &(WGPUPipelineLayoutDescriptor){
                .label                = STRVIEW("Starfield pipeline layout"),
                .bindGroupLayoutCount = 1,
                .bindGroupLayouts     = &state.bgl_ubo_only,
              });

    WGPUShaderModule sf_shader
      = wgpu_create_shader_module(device, instancing_starfield_shader_wgsl);

    state.pipeline_starfield = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Starfield pipeline"),
        .layout = state.pl_starfield,
        .vertex = (WGPUVertexState){
          .module      = sf_shader,
          .entryPoint  = STRVIEW("vs_starfield"),
          .bufferCount = 0,
          .buffers     = NULL,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_None,
        },
        .depthStencil = &depth_starfield,
        .multisample  = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = sf_shader,
          .entryPoint  = STRVIEW("fs_starfield"),
          .targetCount = 1,
          .targets     = &target,
        },
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, sf_shader);
  }

  /* ================================================================
   * 2. Planet pipeline (non-instanced, single 2D texture)
   * ================================================================ */
  {
    state.pl_static = wgpuDeviceCreatePipelineLayout(
      device, &(WGPUPipelineLayoutDescriptor){
                .label                = STRVIEW("Planet pipeline layout"),
                .bindGroupLayoutCount = 1,
                .bindGroupLayouts     = &state.bgl_static,
              });

    WGPUShaderModule pl_shader
      = wgpu_create_shader_module(device, instancing_planet_shader_wgsl);

    /* Vertex attributes for gltf_vertex_t */
    WGPUVertexAttribute planet_attrs[] = {
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal)},
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32x2,
       .offset         = offsetof(gltf_vertex_t, uv0)},
      {.shaderLocation = 3,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, color)},
    };

    WGPUVertexBufferLayout planet_vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = (uint32_t)ARRAY_SIZE(planet_attrs),
      .attributes     = planet_attrs,
    };

    state.pipeline_planet = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Planet pipeline"),
        .layout = state.pl_static,
        .vertex = (WGPUVertexState){
          .module      = pl_shader,
          .entryPoint  = STRVIEW("vs_planet"),
          .bufferCount = 1,
          .buffers     = &planet_vb_layout,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Back,
        },
        .depthStencil = &depth_opaque,
        .multisample  = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = pl_shader,
          .entryPoint  = STRVIEW("fs_planet"),
          .targetCount = 1,
          .targets     = &target,
        },
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, pl_shader);
  }

  /* ================================================================
   * 3. Instanced rocks pipeline (two vertex buffers + texture_2d_array)
   * ================================================================ */
  {
    state.pl_rocks = wgpuDeviceCreatePipelineLayout(
      device, &(WGPUPipelineLayoutDescriptor){
                .label                = STRVIEW("Rocks pipeline layout"),
                .bindGroupLayoutCount = 1,
                .bindGroupLayouts     = &state.bgl_rocks,
              });

    WGPUShaderModule rock_shader
      = wgpu_create_shader_module(device, instancing_rocks_shader_wgsl);

    /* Binding 0 – per-vertex attributes from gltf_vertex_t */
    WGPUVertexAttribute vertex_attrs[] = {
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal)},
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32x2,
       .offset         = offsetof(gltf_vertex_t, uv0)},
      {.shaderLocation = 3,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, color)},
    };

    /* Binding 1 – per-instance attributes from instance_data_t */
    WGPUVertexAttribute instance_attrs[] = {
      {.shaderLocation = 4,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(instance_data_t, pos)},
      {.shaderLocation = 5,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(instance_data_t, rot)},
      {.shaderLocation = 6,
       .format         = WGPUVertexFormat_Float32,
       .offset         = offsetof(instance_data_t, scale)},
      {.shaderLocation = 7,
       .format         = WGPUVertexFormat_Uint32,
       .offset         = offsetof(instance_data_t, tex_index)},
    };

    WGPUVertexBufferLayout vb_layouts[2] = {
      [0] = {
        .arrayStride    = sizeof(gltf_vertex_t),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = (uint32_t)ARRAY_SIZE(vertex_attrs),
        .attributes     = vertex_attrs,
      },
      [1] = {
        .arrayStride    = sizeof(instance_data_t),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = (uint32_t)ARRAY_SIZE(instance_attrs),
        .attributes     = instance_attrs,
      },
    };

    state.pipeline_rocks = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Rocks pipeline"),
        .layout = state.pl_rocks,
        .vertex = (WGPUVertexState){
          .module      = rock_shader,
          .entryPoint  = STRVIEW("vs_rocks"),
          .bufferCount = 2,
          .buffers     = vb_layouts,
        },
        .primitive = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Back,
        },
        .depthStencil = &depth_opaque,
        .multisample  = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = rock_shader,
          .entryPoint  = STRVIEW("fs_rocks"),
          .targetCount = 1,
          .targets     = &target,
        },
      });

    WGPU_RELEASE_RESOURCE(ShaderModule, rock_shader);
  }
}

/* -------------------------------------------------------------------------- *
 * Model drawing helpers
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass, gltf_model_t* mdl,
                       WGPUBuffer vb, WGPUBuffer ib)
{
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vb, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  for (uint32_t n = 0; n < mdl->linear_node_count; ++n) {
    gltf_node_t* node = mdl->linear_nodes[n];
    if (!node->mesh)
      continue;
    for (uint32_t p = 0; p < node->mesh->primitive_count; ++p) {
      gltf_primitive_t* prim = &node->mesh->primitives[p];
      if (prim->has_indices && prim->index_count > 0) {
        wgpuRenderPassEncoderDrawIndexed(pass, prim->index_count, 1,
                                         prim->first_index, 0, 0);
      }
    }
  }
}

static void draw_model_instanced(WGPURenderPassEncoder pass, gltf_model_t* mdl,
                                 WGPUBuffer vb, WGPUBuffer ib,
                                 WGPUBuffer inst_buf, uint32_t inst_count)
{
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vb, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 1, inst_buf, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  for (uint32_t n = 0; n < mdl->linear_node_count; ++n) {
    gltf_node_t* node = mdl->linear_nodes[n];
    if (!node->mesh)
      continue;
    for (uint32_t p = 0; p < node->mesh->primitive_count; ++p) {
      gltf_primitive_t* prim = &node->mesh->primitives[p];
      if (prim->has_indices && prim->index_count > 0) {
        wgpuRenderPassEncoderDrawIndexed(pass, prim->index_count, inst_count,
                                         prim->first_index, 0, 0);
      }
    }
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
  igSetNextWindowSize((ImVec2){200.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (imgui_overlay_header("Statistics")) {
    imgui_overlay_text("Instances: %u", INSTANCE_COUNT);
  }

  if (imgui_overlay_header("Settings")) {
    if (igButton(state.paused ? "Resume" : "Pause", (ImVec2){0.0f, 0.0f})) {
      state.paused = !state.paused;
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
    /* The framework recreates wgpu_context->depth_stencil_view in
     * wgpu_swapchain_resized() – no manual depth texture update needed. */
    camera_set_perspective(
      &state.camera, 60.0f,
      (float)wgpu_context->width / (float)wgpu_context->height, 1.0f, 256.0f);
    return;
  }

  if (imgui_overlay_want_capture_mouse()) {
    return;
  }

  camera_on_input_event(&state.camera, input_event);
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  stm_setup();

  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 2,
    .num_channels = 2,
    .num_lanes    = 1,
    .logger.func  = slog_func,
  });

  /* Allocate file fetch buffers dynamically */
  state.rocks_file_buffer  = (uint8_t*)malloc(ROCKS_FILE_BUFFER_SIZE);
  state.planet_file_buffer = (uint8_t*)malloc(PLANET_FILE_BUFFER_SIZE);
  if (!state.rocks_file_buffer || !state.planet_file_buffer) {
    printf("[instancing] Failed to allocate file buffers\n");
    return EXIT_FAILURE;
  }

  /* ---- Camera ---------------------------------------------------------- */
  camera_init(&state.camera);
  state.camera.type           = CameraType_LookAt;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;
  state.camera.rotation_speed = 0.5f;
  state.camera.movement_speed = 5.0f;

  /* Vulkan camera position: (5.5, -1.85, -18.5).
   * camera_set_position() negates Y internally, so pass the Vulkan values
   * directly: stored position → (5.5, 1.85, -18.5) in WebGPU Y-up space. */
  camera_set_position(&state.camera, (vec3){5.5f, -1.85f, -18.5f});

  /* Vulkan rotation: (-17.2, -4.7, 0).
   * VKY_TO_WGPU_CAM_ROT negates pitch (x): stored → (17.2, -4.7, 0). */
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-17.2f, -4.7f, 0.0f));

  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 1.0f, 256.0f);

  /* ---- Load models synchronously --------------------------------------- */
  load_models();
  create_model_buffers(wgpu_context);

  /* ---- Instance data --------------------------------------------------- */
  prepare_instance_data(wgpu_context);

  /* ---- GPU resources --------------------------------------------------- */
  init_uniform_buffer(wgpu_context);
  init_bind_group_layouts(wgpu_context);
  init_textures(wgpu_context); /* starts async loads + creates placeholders */
  init_bind_groups(wgpu_context);
  init_pipelines(wgpu_context);

  /* ---- ImGui ----------------------------------------------------------- */
  imgui_overlay_init(wgpu_context);

  state.last_frame_time = stm_now();
  state.initialized     = true;

  return EXIT_SUCCESS;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized || !state.models.rock_loaded
      || !state.models.planet_loaded) {
    return EXIT_SUCCESS;
  }

  sfetch_dowork();

  /* Update textures when async loads complete */
  update_textures(wgpu_context);

  /* Timing */
  uint64_t now          = stm_now();
  float dt              = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Update uniforms */
  update_uniform_buffer(wgpu_context, dt);

  /* ImGui frame */
  imgui_overlay_new_frame(wgpu_context, dt);
  render_gui(wgpu_context);

  /* ---- Begin render pass ----------------------------------------------- */
  state.color_attachment.view = wgpu_context->swapchain_view;
  /* Use the framework-managed depth view – always correctly sized after
   * any window resize because wgpu_swapchain_resized() recreates it. */
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder rpass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* ---- 1. Starfield (background) --------------------------------------- */
  wgpuRenderPassEncoderSetPipeline(rpass, state.pipeline_starfield);
  wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.bg_starfield, 0, NULL);
  wgpuRenderPassEncoderDraw(rpass, 3, 1, 0, 0);

  /* ---- 2. Planet -------------------------------------------------------- */
  if (state.models.planet_loaded && state.bg_planet) {
    wgpuRenderPassEncoderSetPipeline(rpass, state.pipeline_planet);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.bg_planet, 0, NULL);
    draw_model(rpass, &state.models.planet, state.model_buffers.planet.vertex,
               state.model_buffers.planet.index);
  }

  /* ---- 3. Instanced rocks ---------------------------------------------- */
  if (state.models.rock_loaded && state.bg_rocks) {
    wgpuRenderPassEncoderSetPipeline(rpass, state.pipeline_rocks);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.bg_rocks, 0, NULL);
    draw_model_instanced(
      rpass, &state.models.rock, state.model_buffers.rock.vertex,
      state.model_buffers.rock.index, state.instance_buffer, INSTANCE_COUNT);
  }

  wgpuRenderPassEncoderEnd(rpass);

  WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buf);

  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buf);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc);

  /* ImGui overlay (rendered in its own pass internally) */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Textures */
  wgpu_destroy_texture(&state.rocks_texture);
  wgpu_destroy_texture(&state.planet_texture);

  /* Buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.instance_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.model_buffers.rock.vertex);
  WGPU_RELEASE_RESOURCE(Buffer, state.model_buffers.rock.index);
  WGPU_RELEASE_RESOURCE(Buffer, state.model_buffers.planet.vertex);
  WGPU_RELEASE_RESOURCE(Buffer, state.model_buffers.planet.index);

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_starfield);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_planet);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_rocks);

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_ubo_only);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_static);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_rocks);

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pl_starfield);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pl_static);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pl_rocks);

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline_starfield);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline_planet);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline_rocks);

  /* Models */
  gltf_model_destroy(&state.models.rock);
  gltf_model_destroy(&state.models.planet);

  /* File buffers */
  free(state.rocks_file_buffer);
  state.rocks_file_buffer = NULL;
  free(state.planet_file_buffer);
  state.planet_file_buffer = NULL;
}

/* -------------------------------------------------------------------------- *
 * Main entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Instanced Mesh Rendering",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* ========================================================================== *
 * WGSL Shaders
 * ========================================================================== */

/* -------------------------------------------------------------------------- *
 * Rocks instancing shader (vertex + fragment)
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* instancing_rocks_shader_wgsl = CODE(

struct Uniforms {
  projection : mat4x4f,
  view       : mat4x4f,
  lightPos   : vec4f,
  locSpeed   : f32,
  globSpeed  : f32,
  _pad       : vec2f,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var rockSampler       : sampler;
@group(0) @binding(2) var rockTexArray      : texture_2d_array<f32>;

struct RockVIn {
  // Per-vertex
  @location(0) position     : vec3f,
  @location(1) normal       : vec3f,
  @location(2) uv           : vec2f,
  @location(3) color        : vec4f,
  // Per-instance
  @location(4) instancePos  : vec3f,
  @location(5) instanceRot  : vec3f,
  @location(6) instanceScale: f32,
  @location(7) instanceTex  : u32,
}

struct RockVOut {
  @builtin(position)               pos      : vec4f,
  @location(0)                     normal   : vec3f,
  @location(1)                     color    : vec3f,
  @location(2)                     uv       : vec2f,
  @location(3)                     viewVec  : vec3f,
  @location(4)                     lightVec : vec3f,
  @location(5) @interpolate(flat)  texIndex : u32,
}

@vertex
fn vs_rocks(in: RockVIn) -> RockVOut {
  var out: RockVOut;
  out.color    = in.color.rgb;
  out.uv       = in.uv;
  out.texIndex = in.instanceTex;

  // ---- Local rotation (same math as the Vulkan GLSL shader) ----

  // "mx" - Z-axis rotation using instanceRot.x + locSpeed
  let sx  = sin(in.instanceRot.x + uniforms.locSpeed);
  let cx  = cos(in.instanceRot.x + uniforms.locSpeed);
  let mx  = mat3x3f(
    vec3f( cx, sx, 0.0),
    vec3f(-sx, cx, 0.0),
    vec3f(0.0, 0.0, 1.0)
  );

  // "my" - Y-axis rotation using instanceRot.y + locSpeed
  let sy  = sin(in.instanceRot.y + uniforms.locSpeed);
  let cy  = cos(in.instanceRot.y + uniforms.locSpeed);
  let my  = mat3x3f(
    vec3f(cy, 0.0, sy),
    vec3f(0.0, 1.0, 0.0),
    vec3f(-sy, 0.0, cy)
  );

  // "mz" - X-axis rotation using instanceRot.z + locSpeed
  let sz  = sin(in.instanceRot.z + uniforms.locSpeed);
  let cz  = cos(in.instanceRot.z + uniforms.locSpeed);
  let mz  = mat3x3f(
    vec3f(1.0, 0.0, 0.0),
    vec3f(0.0,  cz, sz),
    vec3f(0.0, -sz, cz)
  );

  let rotMat = mz * my * mx;

  // ---- Global orbital rotation (Y-axis, using globSpeed) ----
  let sg      = sin(in.instanceRot.y + uniforms.globSpeed);
  let cg      = cos(in.instanceRot.y + uniforms.globSpeed);
  let gRotMat = mat4x4f(
    vec4f( cg, 0.0, sg, 0.0),
    vec4f(0.0, 1.0, 0.0, 0.0),
    vec4f(-sg, 0.0, cg, 0.0),
    vec4f(0.0, 0.0, 0.0, 1.0)
  );

  // Apply local rotation, scale, and instance offset
  let locPos = vec4f(in.position * rotMat, 1.0);
  let worldPos = vec4f(locPos.xyz * in.instanceScale + in.instancePos, 1.0);

  out.pos = uniforms.projection * uniforms.view * gRotMat * worldPos;

  // ---- Normal in view space ----
  // outNormal = mat3(view * gRotMat) * transpose(rotMat) * normal
  let viewGrot  = uniforms.view * gRotMat;
  let viewGrot3 = mat3x3f(viewGrot[0].xyz, viewGrot[1].xyz, viewGrot[2].xyz);
  out.normal    = viewGrot3 * transpose(rotMat) * in.normal;

  // ---- Light and view vectors ----
  // Use the un-rotated instance position for the light/view vector calc
  // (matches Vulkan: pos = modelview * (inPos + instancePos))
  let posView   = uniforms.view * vec4f(in.position + in.instancePos, 1.0);
  let view3     = mat3x3f(
    uniforms.view[0].xyz,
    uniforms.view[1].xyz,
    uniforms.view[2].xyz
  );
  let lPos      = view3 * uniforms.lightPos.xyz;
  out.lightVec  = lPos - posView.xyz;
  out.viewVec   = -posView.xyz;

  return out;
}

@fragment
fn fs_rocks(in: RockVOut) -> @location(0) vec4f {
  let color   = textureSample(rockTexArray, rockSampler, in.uv, i32(in.texIndex))
                * vec4f(in.color, 1.0);
  let N       = normalize(in.normal);
  let L       = normalize(in.lightVec);
  let V       = normalize(in.viewVec);
  let R       = reflect(-L, N);
  let diffuse = max(dot(N, L), 0.1) * in.color;
  let nDotL   = dot(N, L);
  let spec    = select(0.0, pow(max(dot(R, V), 0.0), 16.0), nDotL > 0.0);
  let specular = vec3f(spec * 0.75 * color.r);
  return vec4f(diffuse * color.rgb + specular, 1.0);
}

); // clang-format on

/* -------------------------------------------------------------------------- *
 * Planet shader (vertex + fragment)
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* instancing_planet_shader_wgsl = CODE(

struct Uniforms {
  projection : mat4x4f,
  view       : mat4x4f,
  lightPos   : vec4f,
  locSpeed   : f32,
  globSpeed  : f32,
  _pad       : vec2f,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var planetSampler     : sampler;
@group(0) @binding(2) var planetTex         : texture_2d<f32>;

struct PlanetVIn {
  @location(0) position : vec3f,
  @location(1) normal   : vec3f,
  @location(2) uv       : vec2f,
  @location(3) color    : vec4f,
}

struct PlanetVOut {
  @builtin(position) pos      : vec4f,
  @location(0)       normal   : vec3f,
  @location(1)       color    : vec3f,
  @location(2)       uv       : vec2f,
  @location(3)       viewVec  : vec3f,
  @location(4)       lightVec : vec3f,
}

@vertex
fn vs_planet(in: PlanetVIn) -> PlanetVOut {
  var out: PlanetVOut;
  out.color = in.color.rgb;
  out.uv    = in.uv;
  out.pos   = uniforms.projection * uniforms.view * vec4f(in.position, 1.0);

  let posView = uniforms.view * vec4f(in.position, 1.0);
  let view3   = mat3x3f(
    uniforms.view[0].xyz,
    uniforms.view[1].xyz,
    uniforms.view[2].xyz
  );
  out.normal   = view3 * in.normal;
  let lPos     = view3 * uniforms.lightPos.xyz;
  out.lightVec = lPos - posView.xyz;
  out.viewVec  = -posView.xyz;

  return out;
}

@fragment
fn fs_planet(in: PlanetVOut) -> @location(0) vec4f {
  let color   = textureSample(planetTex, planetSampler, in.uv)
                * vec4f(in.color, 1.0) * 1.5;
  let N       = normalize(in.normal);
  let L       = normalize(in.lightVec);
  let V       = normalize(in.viewVec);
  let R       = reflect(-L, N);
  let diffuse = max(dot(N, L), 0.0) * in.color;
  let specular = pow(max(dot(R, V), 0.0), 4.0) * vec3f(0.5) * color.r;
  return vec4f(diffuse * color.rgb + specular, 1.0);
}

); // clang-format on

/* -------------------------------------------------------------------------- *
 * Starfield shader (vertex + fragment)
 * Generates a fullscreen triangle from vertex_index; no vertex buffers used.
 * The UBO is bound (binding 0) but not actively used in this shader – it is
 * included so the pipeline layout is compatible with the shared bind group.
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* instancing_starfield_shader_wgsl = CODE(

struct Uniforms {
  projection : mat4x4f,
  view       : mat4x4f,
  lightPos   : vec4f,
  locSpeed   : f32,
  globSpeed  : f32,
  _pad       : vec2f,
}
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct StarVOut {
  @builtin(position) pos : vec4f,
  @location(0)       uvw : vec3f,
}

@vertex
fn vs_starfield(@builtin(vertex_index) vi: u32) -> StarVOut {
  var out: StarVOut;
  out.uvw = vec3f(
    f32((vi << 1u) & 2u),
    f32(vi & 2u),
    f32(vi & 2u)
  );
  // Fullscreen triangle; z = 0 (near plane), w = 1
  out.pos = vec4f(out.uvw.xy * 2.0 - 1.0, 0.0, 1.0);
  return out;
}

// Dave Hoskins hash (https://www.shadertoy.com/view/4djSRW)
const HASHSCALE3 = vec3f(443.897, 441.423, 437.195);
const STARFREQ: f32 = 0.01;

fn hash33(p3in: vec3f) -> f32 {
  var p = fract(p3in * HASHSCALE3);
  p += dot(p, p.yxz + vec3f(19.19));
  return fract((p.x + p.y) * p.z + (p.x + p.z) * p.y + (p.y + p.z) * p.x);
}

fn starField(uvw: vec3f) -> vec3f {
  var color = vec3f(0.0);
  let threshold = 1.0 - STARFREQ;
  let rnd = hash33(uvw);
  if rnd >= threshold {
    let starCol = pow((rnd - threshold) / (1.0 - threshold), 16.0);
    color = vec3f(starCol);
  }
  return color;
}

@fragment
fn fs_starfield(in: StarVOut) -> @location(0) vec4f {
  return vec4f(starField(in.uvw), 1.0);
}

); // clang-format on
