/*
 * Order Independent Transparency (OIT) using linked lists
 *
 * Ported from the Vulkan example by Sascha Willems / Daemyung Jang.
 *
 * The algorithm builds per-pixel linked lists in a geometry pass using atomic
 * operations on storage buffers, then sorts and composites them in a
 * fullscreen color pass to produce correct transparency regardless of draw
 * order.
 *
 * Two render passes:
 *   1. Geometry pass – no color output; fragments are inserted into per-pixel
 *      linked lists stored in storage buffers via atomics.
 *   2. Color pass – a fullscreen triangle reads the linked lists, sorts
 *      fragments back-to-front by depth, and blends them together.
 *
 * Ref: https://github.com/SaschaWillems/Vulkan
 */

#include "core/camera.h"
#include "core/gltf_model.h"
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

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
 * Constants
 * -------------------------------------------------------------------------- */

/* Maximum per-pixel fragment count stored in linked lists */
#define NODE_COUNT (20u)

/* Node struct (must match WGSL layout):
 *   color : vec4f  (16 bytes)
 *   depth : f32    (4  bytes)
 *   next  : u32    (4  bytes)
 *   _pad  :        (8  bytes)  — padding to 32 bytes for array alignment
 * Total per node = 32 bytes
 */
#define NODE_STRIDE (32u)

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

/* Geometry pass – vertex shader (MVP transform) */
// clang-format off
static const char* geometry_vertex_shader_wgsl = CODE(
  struct RenderPassUBO {
    projection : mat4x4f,
    view       : mat4x4f,
  }

  struct ObjectData {
    model : mat4x4f,
    color : vec4f,
  }

  @group(0) @binding(0) var<uniform> ubo : RenderPassUBO;
  @group(1) @binding(0) var<uniform> obj : ObjectData;

  struct VertexInput {
    @location(0) position : vec3f,
  }

  struct VertexOutput {
    @builtin(position) position : vec4f,
  }

  @vertex
  fn main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    let pvm = ubo.projection * ubo.view * obj.model;
    output.position = pvm * vec4f(input.position, 1.0);
    return output;
  }
);

/* Geometry pass – fragment shader (linked-list building) */
static const char* geometry_fragment_shader_wgsl = CODE(
  struct GeometrySBO {
    count        : atomic<u32>,
    maxNodeCount : u32,
  }

  struct Node {
    color : vec4f,
    depth : f32,
    next  : u32,
    _pad0 : u32,
    _pad1 : u32,
  }

  struct ObjectData {
    model : mat4x4f,
    color : vec4f,
  }

  @group(0) @binding(1) var<storage, read_write> geometrySBO  : GeometrySBO;
  @group(0) @binding(2) var<storage, read_write> headIndices   : array<atomic<u32>>;
  @group(0) @binding(3) var<storage, read_write> linkedList    : array<Node>;

  @group(1) @binding(0) var<uniform> obj : ObjectData;

  struct ScreenDim {
    width  : u32,
    height : u32,
  }
  @group(0) @binding(4) var<uniform> screenDim : ScreenDim;

  @fragment
  fn main(@builtin(position) fragCoord : vec4f) {
    /* Allocate a node in the linked list */
    let nodeIdx = atomicAdd(&geometrySBO.count, 1u);

    /* Check if the linked list is full */
    if (nodeIdx >= geometrySBO.maxNodeCount) {
      return;
    }

    /* Compute 1-D pixel index */
    let pixelIdx = u32(fragCoord.y) * screenDim.width + u32(fragCoord.x);

    /* Exchange new head and get previous head */
    let prevHeadIdx = atomicExchange(&headIndices[pixelIdx], nodeIdx);

    /* Store node data */
    linkedList[nodeIdx].color = obj.color;
    linkedList[nodeIdx].depth = fragCoord.z;
    linkedList[nodeIdx].next  = prevHeadIdx;
  }
);

/* Color pass – vertex shader (fullscreen triangle) */
static const char* color_vertex_shader_wgsl = CODE(
  @vertex
  fn main(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {
    /* Generate a fullscreen triangle from vertex index */
    let u = f32((vertexIndex << 1u) & 2u);
    let v = f32(vertexIndex & 2u);
    return vec4f(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0);
  }
);

/* Color pass – fragment shader (linked-list sorting & compositing) */
static const char* color_fragment_shader_wgsl = CODE(
  const MAX_FRAGMENT_COUNT = 128;

  struct Node {
    color : vec4f,
    depth : f32,
    next  : u32,
    _pad0 : u32,
    _pad1 : u32,
  }

  struct ScreenDim {
    width  : u32,
    height : u32,
  }

  @group(0) @binding(0) var<storage, read> headIndices : array<u32>;
  @group(0) @binding(1) var<storage, read> linkedList  : array<Node>;
  @group(0) @binding(2) var<uniform> screenDim : ScreenDim;

  @fragment
  fn main(@builtin(position) fragCoord : vec4f) -> @location(0) vec4f {
    var fragments : array<Node, MAX_FRAGMENT_COUNT>;
    var count = 0;

    let pixelIdx = u32(fragCoord.y) * screenDim.width + u32(fragCoord.x);
    var nodeIdx  = headIndices[pixelIdx];

    /* Walk the linked list for this pixel */
    while (nodeIdx != 0xffffffffu && count < MAX_FRAGMENT_COUNT) {
      fragments[count] = linkedList[nodeIdx];
      nodeIdx = fragments[count].next;
      count = count + 1;
    }

    /* Insertion sort by depth (back-to-front, farthest first) */
    for (var i = 1; i < count; i = i + 1) {
      let insert = fragments[i];
      var j = i;
      while (j > 0 && insert.depth > fragments[j - 1].depth) {
        fragments[j] = fragments[j - 1];
        j = j - 1;
      }
      fragments[j] = insert;
    }

    /* Blend back-to-front */
    var color = vec4f(0.025, 0.025, 0.025, 1.0);
    for (var i = 0; i < count; i = i + 1) {
      color = mix(color, fragments[i].color, fragments[i].color.a);
    }

    return color;
  }
);

/* Compute shader to clear head index buffer to 0xFFFFFFFF */
static const char* clear_head_index_shader_wgsl = CODE(
  @group(0) @binding(0) var<storage, read_write> headIndices : array<u32>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    if (idx < arrayLength(&headIndices)) {
      headIndices[idx] = 0xffffffffu;
    }
  }
);
// clang-format on

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Models */
  struct {
    gltf_model_t sphere;
    gltf_model_t cube;
  } models;
  bool models_loaded;

  /* GPU vertex / index buffers for models */
  struct {
    WGPUBuffer vertex_buffer;
    WGPUBuffer index_buffer;
  } sphere_buffers, cube_buffers;

  /* Camera */
  camera_t camera;

  /* Geometry pass resources */
  struct {
    WGPUBuffer geometry_sbo;   /* { count: u32, maxNodeCount: u32 } */
    WGPUBuffer head_index_buf; /* Per-pixel head indices (storage) */
    WGPUBuffer linked_list;    /* Node array (storage) */
    WGPUBuffer screen_dim_buf; /* { width: u32, height: u32 } */
    uint32_t max_node_count;   /* NODE_COUNT * width * height */
  } geometry_pass;

  /* Per-frame uniform buffer (projection + view) */
  WGPUBuffer render_pass_ubo;

  /* Per-object uniform buffers (model + color) – dynamic offset approach */
  struct {
    WGPUBuffer buffer;
    uint32_t stride; /* 256-byte aligned stride */
    uint32_t count;  /* total object count */
  } object_ubo;

  /* Descriptor / bind group resources */
  struct {
    WGPUBindGroupLayout geometry_shared; /* group 0 for geometry pass */
    WGPUBindGroupLayout geometry_object; /* group 1 for geometry pass */
    WGPUBindGroupLayout color;           /* group 0 for color pass */
  } bg_layouts;

  struct {
    WGPUBindGroup geometry_shared;
    WGPUBindGroup geometry_object;
    WGPUBindGroup color;
  } bind_groups;

  /* Pipeline layouts and pipelines */
  struct {
    WGPUPipelineLayout geometry;
    WGPUPipelineLayout color;
  } pipeline_layouts;

  struct {
    WGPURenderPipeline geometry;
    WGPURenderPipeline color;
  } pipelines;

  /* Compute pipeline for clearing head index buffer */
  struct {
    WGPUComputePipeline pipeline;
    WGPUPipelineLayout pipeline_layout;
    WGPUBindGroupLayout bg_layout;
    WGPUBindGroup bind_group;
  } clear_pass;

  /* Depth texture (required as dummy attachment for geometry pass) */
  wgpu_texture_t depth_texture;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor color_render_pass_desc;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  // clang-format off
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.025, 0.025, 0.025, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .color_render_pass_desc = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = NULL,
  },
  // clang-format on
};

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_models(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  const gltf_model_desc_t desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
    /* No FlipY for WebGPU */
  };

  bool sphere_ok = gltf_model_load_from_file_ext(
    &state.models.sphere, "assets/models/sphere.gltf", 1.0f, &desc);
  bool cube_ok = gltf_model_load_from_file_ext(
    &state.models.cube, "assets/models/cube.gltf", 1.0f, &desc);

  if (!sphere_ok || !cube_ok) {
    printf("ERROR: Failed to load sphere/cube models\n");
    return;
  }

  state.models_loaded = true;

  /* Create GPU vertex/index buffers for each model */
  struct {
    gltf_model_t* model;
    WGPUBuffer* vb;
    WGPUBuffer* ib;
    const char* vb_label;
    const char* ib_label;
  } items[2] = {
    {&state.models.sphere, &state.sphere_buffers.vertex_buffer,
     &state.sphere_buffers.index_buffer, "Sphere VB", "Sphere IB"},
    {&state.models.cube, &state.cube_buffers.vertex_buffer,
     &state.cube_buffers.index_buffer, "Cube VB", "Cube IB"},
  };

  for (int i = 0; i < 2; i++) {
    gltf_model_t* m = items[i].model;
    size_t vb_size  = m->vertex_count * sizeof(gltf_vertex_t);

    *items[i].vb = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW(items[i].vb_label),
                .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                .size  = vb_size,
                .mappedAtCreation = true,
              });
    void* vdata = wgpuBufferGetMappedRange(*items[i].vb, 0, vb_size);
    memcpy(vdata, m->vertices, vb_size);
    wgpuBufferUnmap(*items[i].vb);

    if (m->index_count > 0) {
      size_t ib_size = m->index_count * sizeof(uint32_t);
      *items[i].ib   = wgpuDeviceCreateBuffer(
        device, &(WGPUBufferDescriptor){
                    .label = STRVIEW(items[i].ib_label),
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                    .size  = ib_size,
                    .mappedAtCreation = true,
                });
      void* idata = wgpuBufferGetMappedRange(*items[i].ib, 0, ib_size);
      memcpy(idata, m->indices, ib_size);
      wgpuBufferUnmap(*items[i].ib);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Camera
 * -------------------------------------------------------------------------- */

static void init_camera(struct wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type           = CameraType_LookAt;
  state.camera.rotation_speed = 0.25f;
  state.camera.movement_speed = 0.1f;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;

  /* Vulkan original: position = (0, 0, -6), rotation = (0, 0, 0)
   * For WebGPU: negate Y for position, negate X for rotation.
   * Since both are 0, no change needed. */
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -6.0f});
  camera_set_rotation(&state.camera, (vec3){0.0f, 0.0f, 0.0f});

  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  camera_set_perspective(&state.camera, 60.0f, aspect, 0.1f, 256.0f);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Render pass UBO: projection + view (2 × mat4 = 128 bytes) */
  state.render_pass_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Render Pass UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = 2 * sizeof(mat4),
            });

  /* Per-object UBO with dynamic offsets:
   * 5×5×5 = 125 spheres + 2 cubes = 127 objects
   * Each object has: model (mat4 = 64 bytes) + color (vec4 = 16 bytes) = 80
   * bytes Padded to 256-byte alignment for dynamic offsets */
  state.object_ubo.count  = 5 * 5 * 5 + 2;
  state.object_ubo.stride = 256; /* WebGPU minimum uniform buffer alignment */

  state.object_ubo.buffer = wgpuDeviceCreateBuffer(
    device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Object Dynamic UBO"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = (uint64_t)state.object_ubo.count * state.object_ubo.stride,
    });
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUQueue queue = wgpu_context->queue;

  /* Update projection + view */
  struct {
    mat4 projection;
    mat4 view;
  } ubo_data;
  glm_mat4_copy(state.camera.matrices.perspective, ubo_data.projection);
  glm_mat4_copy(state.camera.matrices.view, ubo_data.view);
  wgpuQueueWriteBuffer(queue, state.render_pass_ubo, 0, &ubo_data,
                       sizeof(ubo_data));

  /* Build per-object data: 125 spheres + 2 cubes */
  uint8_t* obj_buf
    = (uint8_t*)state.object_ubo.buffer; /* temp pointer, unused – see below */
  (void)obj_buf;

  uint32_t obj_idx = 0;
  struct {
    mat4 model;
    vec4 color;
  } obj_data;

  /* Red spheres: 5×5×5 grid, each at integer offsets -2..+2, scaled 0.3 */
  for (int32_t x = 0; x < 5; x++) {
    for (int32_t y = 0; y < 5; y++) {
      for (int32_t z = 0; z < 5; z++) {
        glm_mat4_identity(obj_data.model);
        glm_translate(obj_data.model,
                      (vec3){(float)(x - 2), (float)(y - 2), (float)(z - 2)});
        glm_scale(obj_data.model, (vec3){0.3f, 0.3f, 0.3f});
        glm_vec4_copy((vec4){1.0f, 0.0f, 0.0f, 0.5f}, obj_data.color);

        wgpuQueueWriteBuffer(queue, state.object_ubo.buffer,
                             (uint64_t)obj_idx * state.object_ubo.stride,
                             &obj_data, sizeof(obj_data));
        obj_idx++;
      }
    }
  }

  /* Blue cubes: 2 cubes at x=-1.5 and x=+1.5, scaled 0.2 */
  for (uint32_t x = 0; x < 2; x++) {
    glm_mat4_identity(obj_data.model);
    glm_translate(obj_data.model, (vec3){3.0f * (float)x - 1.5f, 0.0f, 0.0f});
    glm_scale(obj_data.model, (vec3){0.2f, 0.2f, 0.2f});
    glm_vec4_copy((vec4){0.0f, 0.0f, 1.0f, 0.5f}, obj_data.color);

    wgpuQueueWriteBuffer(queue, state.object_ubo.buffer,
                         (uint64_t)obj_idx * state.object_ubo.stride, &obj_data,
                         sizeof(obj_data));
    obj_idx++;
  }
}

/* -------------------------------------------------------------------------- *
 * Geometry pass resources (storage buffers, head-index buffer)
 * -------------------------------------------------------------------------- */

static void destroy_geometry_pass_resources(void)
{
  if (state.geometry_pass.geometry_sbo) {
    wgpuBufferRelease(state.geometry_pass.geometry_sbo);
    state.geometry_pass.geometry_sbo = NULL;
  }
  if (state.geometry_pass.head_index_buf) {
    wgpuBufferRelease(state.geometry_pass.head_index_buf);
    state.geometry_pass.head_index_buf = NULL;
  }
  if (state.geometry_pass.linked_list) {
    wgpuBufferRelease(state.geometry_pass.linked_list);
    state.geometry_pass.linked_list = NULL;
  }
  if (state.geometry_pass.screen_dim_buf) {
    wgpuBufferRelease(state.geometry_pass.screen_dim_buf);
    state.geometry_pass.screen_dim_buf = NULL;
  }
}

static void init_geometry_pass_resources(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  uint32_t w        = (uint32_t)wgpu_context->width;
  uint32_t h        = (uint32_t)wgpu_context->height;

  state.geometry_pass.max_node_count = NODE_COUNT * w * h;

  /* GeometrySBO: { count: u32 (atomic), maxNodeCount: u32 } = 8 bytes
   * We need: Storage | CopyDst (to clear count each frame) */
  {
    uint32_t init_data[2]            = {0, state.geometry_pass.max_node_count};
    state.geometry_pass.geometry_sbo = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("GeometrySBO"),
                .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                .size  = sizeof(init_data),
                .mappedAtCreation = true,
              });
    void* ptr = wgpuBufferGetMappedRange(state.geometry_pass.geometry_sbo, 0,
                                         sizeof(init_data));
    memcpy(ptr, init_data, sizeof(init_data));
    wgpuBufferUnmap(state.geometry_pass.geometry_sbo);
  }

  /* Head-index buffer: one u32 per pixel (atomic storage) */
  {
    uint32_t buf_size                  = w * h * sizeof(uint32_t);
    state.geometry_pass.head_index_buf = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Head Index Buffer"),
                .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                .size  = buf_size,
              });
  }

  /* Linked-list buffer: NODE_COUNT * w * h nodes of NODE_STRIDE bytes each */
  {
    uint64_t buf_size
      = (uint64_t)state.geometry_pass.max_node_count * NODE_STRIDE;
    state.geometry_pass.linked_list = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Linked List Buffer"),
                .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
                .size  = buf_size,
              });
  }

  /* Screen dimensions uniform buffer */
  {
    uint32_t dim[2]                    = {w, h};
    state.geometry_pass.screen_dim_buf = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Screen Dim UBO"),
                .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                .size  = sizeof(dim),
                .mappedAtCreation = true,
              });
    void* ptr = wgpuBufferGetMappedRange(state.geometry_pass.screen_dim_buf, 0,
                                         sizeof(dim));
    memcpy(ptr, dim, sizeof(dim));
    wgpuBufferUnmap(state.geometry_pass.screen_dim_buf);
  }
}

/* -------------------------------------------------------------------------- *
 * Depth texture (needed as dummy attachment for geometry pass)
 * -------------------------------------------------------------------------- */

static void init_depth_texture(struct wgpu_context_t* wgpu_context)
{
  wgpu_destroy_texture(&state.depth_texture);

  WGPUTexture tex = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = STRVIEW("OIT Depth Texture"),
                            .usage         = WGPUTextureUsage_RenderAttachment,
                            .dimension     = WGPUTextureDimension_2D,
                            .size          = {(uint32_t)wgpu_context->width,
                                              (uint32_t)wgpu_context->height, 1},
                            .format        = WGPUTextureFormat_Depth24Plus,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                          });

  state.depth_texture.handle = tex;
  state.depth_texture.view   = wgpuTextureCreateView(tex, NULL);
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Geometry pass – group 0 (shared):
   * binding 0: uniform buffer (projection + view)
   * binding 1: storage buffer (GeometrySBO – atomic counter)
   * binding 2: storage buffer (head indices – atomic)
   * binding 3: storage buffer (linked list nodes)
   * binding 4: uniform buffer (screen dimensions) */
  {
    WGPUBindGroupLayoutEntry entries[5] = {
      /* binding 0: render pass UBO (vertex + fragment) */
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 2 * sizeof(mat4),
        },
      },
      /* binding 1: GeometrySBO (fragment) */
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Storage,
          .minBindingSize = 8,
        },
      },
      /* binding 2: head-index buffer (fragment) */
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Storage,
          .minBindingSize = 4,
        },
      },
      /* binding 3: linked-list buffer (fragment) */
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Storage,
          .minBindingSize = NODE_STRIDE,
        },
      },
      /* binding 4: screen dimensions (fragment) */
      {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 8,
        },
      },
    };

    state.bg_layouts.geometry_shared = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Geometry Shared BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Geometry pass – group 1 (per-object):
   * binding 0: uniform buffer (dynamic offset – model + color) */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = sizeof(mat4) + sizeof(vec4), /* 80 bytes */
        },
      },
    };

    state.bg_layouts.geometry_object = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Geometry Object BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Color pass – group 0:
   * binding 0: storage buffer (head indices – read only)
   * binding 1: storage buffer (linked list – read only)
   * binding 2: uniform buffer (screen dimensions) */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_ReadOnlyStorage,
          .minBindingSize = 4,
        },
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_ReadOnlyStorage,
          .minBindingSize = NODE_STRIDE,
        },
      },
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 8,
        },
      },
    };

    state.bg_layouts.color = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Color BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void destroy_bind_groups(void)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.geometry_shared)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.geometry_object)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.color)
}

static void init_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  uint32_t w        = (uint32_t)wgpu_context->width;
  uint32_t h        = (uint32_t)wgpu_context->height;

  /* Geometry pass – group 0 (shared) */
  {
    WGPUBindGroupEntry entries[5] = {
      {
        .binding = 0,
        .buffer  = state.render_pass_ubo,
        .offset  = 0,
        .size    = 2 * sizeof(mat4),
      },
      {
        .binding = 1,
        .buffer  = state.geometry_pass.geometry_sbo,
        .offset  = 0,
        .size    = 8,
      },
      {
        .binding = 2,
        .buffer  = state.geometry_pass.head_index_buf,
        .offset  = 0,
        .size    = (uint64_t)w * h * sizeof(uint32_t),
      },
      {
        .binding = 3,
        .buffer  = state.geometry_pass.linked_list,
        .offset  = 0,
        .size    = (uint64_t)state.geometry_pass.max_node_count * NODE_STRIDE,
      },
      {
        .binding = 4,
        .buffer  = state.geometry_pass.screen_dim_buf,
        .offset  = 0,
        .size    = 8,
      },
    };

    state.bind_groups.geometry_shared = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Geometry Shared BG"),
                .layout     = state.bg_layouts.geometry_shared,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Geometry pass – group 1 (per-object dynamic uniform) */
  {
    WGPUBindGroupEntry entries[1] = {
      {
        .binding = 0,
        .buffer  = state.object_ubo.buffer,
        .offset  = 0,
        .size    = sizeof(mat4) + sizeof(vec4), /* 80 bytes */
      },
    };

    state.bind_groups.geometry_object = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Geometry Object BG"),
                .layout     = state.bg_layouts.geometry_object,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Color pass – group 0 */
  {
    WGPUBindGroupEntry entries[3] = {
      {
        .binding = 0,
        .buffer  = state.geometry_pass.head_index_buf,
        .offset  = 0,
        .size    = (uint64_t)w * h * sizeof(uint32_t),
      },
      {
        .binding = 1,
        .buffer  = state.geometry_pass.linked_list,
        .offset  = 0,
        .size    = (uint64_t)state.geometry_pass.max_node_count * NODE_STRIDE,
      },
      {
        .binding = 2,
        .buffer  = state.geometry_pass.screen_dim_buf,
        .offset  = 0,
        .size    = 8,
      },
    };

    state.bind_groups.color
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label  = STRVIEW("Color BG"),
                                            .layout = state.bg_layouts.color,
                                            .entryCount = ARRAY_SIZE(entries),
                                            .entries    = entries,
                                          });
  }
}

/* -------------------------------------------------------------------------- *
 * Pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* --- Geometry pipeline --- */
  {
    WGPUBindGroupLayout layouts[2] = {
      state.bg_layouts.geometry_shared,
      state.bg_layouts.geometry_object,
    };

    state.pipeline_layouts.geometry = wgpuDeviceCreatePipelineLayout(
      device, &(WGPUPipelineLayoutDescriptor){
                .label                = STRVIEW("Geometry Pipeline Layout"),
                .bindGroupLayoutCount = ARRAY_SIZE(layouts),
                .bindGroupLayouts     = layouts,
              });

    WGPUShaderModule vs_module
      = wgpu_create_shader_module(device, geometry_vertex_shader_wgsl);
    WGPUShaderModule fs_module
      = wgpu_create_shader_module(device, geometry_fragment_shader_wgsl);

    /* Vertex layout: only position (vec3f) from gltf_vertex_t */
    WGPUVertexAttribute position_attr = {
      .shaderLocation = 0,
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, position),
    };

    WGPUVertexBufferLayout vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &position_attr,
    };

    /* No color targets for the geometry pipeline (fragment-only storage
     * writes). Depth test is disabled (Always) — OIT needs ALL fragments.
     * We still need the depth attachment because WebGPU requires at least
     * one attachment on a render pipeline / render pass. */
    state.pipelines.geometry = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Geometry Pipeline"),
                .layout = state.pipeline_layouts.geometry,
                .vertex = {
                  .module      = vs_module,
                  .entryPoint  = STRVIEW("main"),
                  .bufferCount = 1,
                  .buffers     = &vb_layout,
                },
                .primitive = {
                  .topology = WGPUPrimitiveTopology_TriangleList,
                  .cullMode = WGPUCullMode_None,
                },
                .depthStencil = &(WGPUDepthStencilState){
                  .format            = WGPUTextureFormat_Depth24Plus,
                  .depthWriteEnabled = WGPUOptionalBool_False,
                  .depthCompare      = WGPUCompareFunction_Always,
                },
                .fragment = &(WGPUFragmentState){
                  .module      = fs_module,
                  .entryPoint  = STRVIEW("main"),
                  .targetCount = 0,
                  .targets     = NULL,
                },
                .multisample = {
                  .count = 1,
                  .mask  = 0xFFFFFFFF,
                },
              });

    wgpuShaderModuleRelease(vs_module);
    wgpuShaderModuleRelease(fs_module);
  }

  /* --- Color pipeline --- */
  {
    state.pipeline_layouts.color = wgpuDeviceCreatePipelineLayout(
      device, &(WGPUPipelineLayoutDescriptor){
                .label                = STRVIEW("Color Pipeline Layout"),
                .bindGroupLayoutCount = 1,
                .bindGroupLayouts     = &state.bg_layouts.color,
              });

    WGPUShaderModule vs_module
      = wgpu_create_shader_module(device, color_vertex_shader_wgsl);
    WGPUShaderModule fs_module
      = wgpu_create_shader_module(device, color_fragment_shader_wgsl);

    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* No depth test — compositing is done in the shader */
    state.pipelines.color = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Color Pipeline"),
                .layout = state.pipeline_layouts.color,
                .vertex = {
                  .module      = vs_module,
                  .entryPoint  = STRVIEW("main"),
                  .bufferCount = 0,
                },
                .primitive = {
                  .topology = WGPUPrimitiveTopology_TriangleList,
                  .cullMode = WGPUCullMode_None,
                },
                .depthStencil = NULL,
                .fragment = &(WGPUFragmentState){
                  .module      = fs_module,
                  .entryPoint  = STRVIEW("main"),
                  .targetCount = 1,
                  .targets     = &color_target,
                },
                .multisample = {
                  .count = 1,
                  .mask  = 0xFFFFFFFF,
                },
              });

    wgpuShaderModuleRelease(vs_module);
    wgpuShaderModuleRelease(fs_module);
  }
}

/* -------------------------------------------------------------------------- *
 * Compute pipeline to clear head-index buffer to 0xFFFFFFFF
 * -------------------------------------------------------------------------- */

static void destroy_clear_pass(void)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.clear_pass.bind_group)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.clear_pass.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.clear_pass.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.clear_pass.bg_layout)
}

static void init_clear_pass(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Bind group layout: one storage buffer (read_write) */
  WGPUBindGroupLayoutEntry entries[1] = {
    {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = 4,
      },
    },
  };

  state.clear_pass.bg_layout
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .label = STRVIEW("Clear BGL"),
                                                .entryCount = 1,
                                                .entries    = entries,
                                              });

  state.clear_pass.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Clear Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.clear_pass.bg_layout,
            });

  WGPUShaderModule cs_module
    = wgpu_create_shader_module(device, clear_head_index_shader_wgsl);

  state.clear_pass.pipeline = wgpuDeviceCreateComputePipeline(
    device, &(WGPUComputePipelineDescriptor){
              .label   = STRVIEW("Clear Compute Pipeline"),
              .layout  = state.clear_pass.pipeline_layout,
              .compute = {
                .module     = cs_module,
                .entryPoint = STRVIEW("main"),
              },
            });

  wgpuShaderModuleRelease(cs_module);
}

static void update_clear_pass_bind_group(struct wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.clear_pass.bind_group)

  uint32_t w = (uint32_t)wgpu_context->width;
  uint32_t h = (uint32_t)wgpu_context->height;

  WGPUBindGroupEntry entries[1] = {
    {
      .binding = 0,
      .buffer  = state.geometry_pass.head_index_buf,
      .offset  = 0,
      .size    = (uint64_t)w * h * sizeof(uint32_t),
    },
  };

  state.clear_pass.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Clear BG"),
                            .layout     = state.clear_pass.bg_layout,
                            .entryCount = 1,
                            .entries    = entries,
                          });
}

/* -------------------------------------------------------------------------- *
 * Draw model helper (inline in render loop – used via macros below)
 * -------------------------------------------------------------------------- */

#define DRAW_MODEL_PRIMITIVES(pass, model)                                     \
  do {                                                                         \
    for (uint32_t _n = 0; _n < (model)->linear_node_count; _n++) {             \
      gltf_node_t* _node = (model)->linear_nodes[_n];                          \
      if (!_node->mesh)                                                        \
        continue;                                                              \
      for (uint32_t _p = 0; _p < _node->mesh->primitive_count; _p++) {         \
        gltf_primitive_t* _prim = &_node->mesh->primitives[_p];                \
        if (_prim->has_indices && _prim->index_count > 0) {                    \
          wgpuRenderPassEncoderDrawIndexed(pass, _prim->index_count, 1,        \
                                           _prim->first_index, 0, 0);          \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){0.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Order Independent Transparency", NULL,
          ImGuiWindowFlags_AlwaysAutoResize);

  igText("Objects: 125 spheres + 2 cubes");
  igText("Nodes per pixel: %u", NODE_COUNT);
  igText("Resolution: %u x %u", (uint32_t)wgpu_context->width,
         (uint32_t)wgpu_context->height);
  igText("Max linked-list nodes: %u", state.geometry_pass.max_node_count);

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Resize handling
 * -------------------------------------------------------------------------- */

static void on_resize(struct wgpu_context_t* wgpu_context)
{
  /* Recreate size-dependent resources */
  destroy_geometry_pass_resources();
  init_geometry_pass_resources(wgpu_context);

  destroy_bind_groups();
  init_bind_groups(wgpu_context);

  update_clear_pass_bind_group(wgpu_context);

  init_depth_texture(wgpu_context);

  /* Update camera aspect */
  camera_update_aspect_ratio(&state.camera, (float)wgpu_context->width
                                              / (float)wgpu_context->height);
}

/* -------------------------------------------------------------------------- *
 * Callbacks
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();

  init_camera(wgpu_context);
  load_models(wgpu_context);

  if (!state.models_loaded) {
    return EXIT_FAILURE;
  }

  init_uniform_buffers(wgpu_context);
  init_geometry_pass_resources(wgpu_context);
  init_depth_texture(wgpu_context);
  init_bind_group_layouts(wgpu_context);
  init_bind_groups(wgpu_context);
  init_pipelines(wgpu_context);
  init_clear_pass(wgpu_context);
  update_clear_pass_bind_group(wgpu_context);
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update camera */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  camera_update(&state.camera, delta_time);
  update_uniform_buffers(wgpu_context);

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;
  uint32_t w        = (uint32_t)wgpu_context->width;
  uint32_t h        = (uint32_t)wgpu_context->height;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* --- Step 1: Clear head-index buffer to 0xFFFFFFFF via compute --- */
  {
    uint32_t pixel_count     = w * h;
    uint32_t workgroup_count = (pixel_count + 255) / 256;

    WGPUComputePassEncoder cpass
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass, state.clear_pass.pipeline);
    wgpuComputePassEncoderSetBindGroup(cpass, 0, state.clear_pass.bind_group, 0,
                                       NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cpass, workgroup_count, 1, 1);
    wgpuComputePassEncoderEnd(cpass);
    wgpuComputePassEncoderRelease(cpass);
  }

  /* --- Step 2: Reset geometry SBO counter to 0 --- */
  wgpuCommandEncoderClearBuffer(cmd_enc, state.geometry_pass.geometry_sbo, 0,
                                sizeof(uint32_t));

  /* --- Step 3: Geometry render pass (no color attachments) --- */
  {
    WGPURenderPassDescriptor geom_rp_desc = {
      .colorAttachmentCount   = 0,
      .colorAttachments       = NULL,
      .depthStencilAttachment = &(WGPURenderPassDepthStencilAttachment){
        .view            = state.depth_texture.view,
        .depthLoadOp     = WGPULoadOp_Clear,
        .depthStoreOp    = WGPUStoreOp_Discard,
        .depthClearValue = 1.0f,
        .stencilLoadOp   = WGPULoadOp_Undefined,
        .stencilStoreOp  = WGPUStoreOp_Undefined,
      },
    };

    WGPURenderPassEncoder geom_pass
      = wgpuCommandEncoderBeginRenderPass(cmd_enc, &geom_rp_desc);

    wgpuRenderPassEncoderSetViewport(geom_pass, 0.0f, 0.0f, (float)w, (float)h,
                                     0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(geom_pass, 0, 0, w, h);
    wgpuRenderPassEncoderSetPipeline(geom_pass, state.pipelines.geometry);
    wgpuRenderPassEncoderSetBindGroup(
      geom_pass, 0, state.bind_groups.geometry_shared, 0, NULL);

    uint32_t obj_idx = 0;

    /* Draw 125 spheres */
    wgpuRenderPassEncoderSetVertexBuffer(
      geom_pass, 0, state.sphere_buffers.vertex_buffer, 0, WGPU_WHOLE_SIZE);
    if (state.sphere_buffers.index_buffer) {
      wgpuRenderPassEncoderSetIndexBuffer(
        geom_pass, state.sphere_buffers.index_buffer, WGPUIndexFormat_Uint32, 0,
        WGPU_WHOLE_SIZE);
    }

    for (int32_t x = 0; x < 5; x++) {
      for (int32_t y = 0; y < 5; y++) {
        for (int32_t z = 0; z < 5; z++) {
          uint32_t dyn_offset = obj_idx * state.object_ubo.stride;
          wgpuRenderPassEncoderSetBindGroup(
            geom_pass, 1, state.bind_groups.geometry_object, 1, &dyn_offset);

          DRAW_MODEL_PRIMITIVES(geom_pass, &state.models.sphere);

          obj_idx++;
        }
      }
    }

    /* Draw 2 cubes */
    wgpuRenderPassEncoderSetVertexBuffer(
      geom_pass, 0, state.cube_buffers.vertex_buffer, 0, WGPU_WHOLE_SIZE);
    if (state.cube_buffers.index_buffer) {
      wgpuRenderPassEncoderSetIndexBuffer(
        geom_pass, state.cube_buffers.index_buffer, WGPUIndexFormat_Uint32, 0,
        WGPU_WHOLE_SIZE);
    }

    for (uint32_t x = 0; x < 2; x++) {
      uint32_t dyn_offset = obj_idx * state.object_ubo.stride;
      wgpuRenderPassEncoderSetBindGroup(
        geom_pass, 1, state.bind_groups.geometry_object, 1, &dyn_offset);

      DRAW_MODEL_PRIMITIVES(geom_pass, &state.models.cube);

      obj_idx++;
    }

    wgpuRenderPassEncoderEnd(geom_pass);
    wgpuRenderPassEncoderRelease(geom_pass);
  }

  /* --- Step 4: Color render pass (fullscreen compositing) --- */
  {
    state.color_attachment.view = wgpu_context->swapchain_view;

    WGPURenderPassEncoder color_pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.color_render_pass_desc);

    wgpuRenderPassEncoderSetViewport(color_pass, 0.0f, 0.0f, (float)w, (float)h,
                                     0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(color_pass, 0, 0, w, h);
    wgpuRenderPassEncoderSetPipeline(color_pass, state.pipelines.color);
    wgpuRenderPassEncoderSetBindGroup(color_pass, 0, state.bind_groups.color, 0,
                                      NULL);
    wgpuRenderPassEncoderDraw(color_pass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(color_pass);
    wgpuRenderPassEncoderRelease(color_pass);
  }

  /* --- Finalize & submit --- */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    on_resize(wgpu_context);
  }

  camera_on_input_event(&state.camera, input_event);
}

static void shutdown_func(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Destroy pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.geometry)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.color)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.geometry)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.color)

  /* Destroy clear compute pass */
  destroy_clear_pass();

  /* Destroy bind groups */
  destroy_bind_groups();

  /* Destroy bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layouts.geometry_shared)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layouts.geometry_object)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layouts.color)

  /* Destroy uniform buffers */
  if (state.render_pass_ubo) {
    wgpuBufferRelease(state.render_pass_ubo);
  }
  if (state.object_ubo.buffer) {
    wgpuBufferRelease(state.object_ubo.buffer);
  }

  /* Destroy geometry pass resources */
  destroy_geometry_pass_resources();

  /* Destroy depth texture */
  wgpu_destroy_texture(&state.depth_texture);

  /* Destroy model buffers */
  if (state.sphere_buffers.vertex_buffer) {
    wgpuBufferRelease(state.sphere_buffers.vertex_buffer);
  }
  if (state.sphere_buffers.index_buffer) {
    wgpuBufferRelease(state.sphere_buffers.index_buffer);
  }
  if (state.cube_buffers.vertex_buffer) {
    wgpuBufferRelease(state.cube_buffers.vertex_buffer);
  }
  if (state.cube_buffers.index_buffer) {
    wgpuBufferRelease(state.cube_buffers.index_buffer);
  }

  /* Destroy models */
  if (state.models_loaded) {
    gltf_model_destroy(&state.models.sphere);
    gltf_model_destroy(&state.models.cube);
  }
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Order Independent Transparency",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown_func,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}
