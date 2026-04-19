/* -------------------------------------------------------------------------- *
 * WebGPU Example - Terrain Tessellation
 *
 * Demonstrates heightmap-based terrain rendering with multi-layer texture
 * blending, distance fog, and a skysphere background. Since WebGPU does not
 * support hardware tessellation shaders, the terrain is tessellated using a
 * GPU compute shader that samples the heightmap and computes Sobel-filter
 * normals per vertex, mimicking the Vulkan TCS/TES pipeline on the GPU.
 *
 * A coarse 64×64 grid (matching the Vulkan patch size) and a fine 256×256
 * grid share the same displaced vertex buffer.  The tessellation toggle
 * switches between the two index buffers, reproducing the Vulkan behaviour
 * of reducing geometric detail while keeping displacement intact.
 *
 * Features:
 * - GPU compute shader for heightmap displacement and normal generation
 * - 6-layer terrain texture blending based on height
 * - Distance-based exponential fog
 * - Skysphere background (glTF sphere model)
 * - Wireframe overlay (line-list topology)
 * - GUI: tessellation toggle, displacement factor, wireframe toggle
 * - Pipeline statistics display (vertex / triangle counts)
 * - First-person camera with mouse + keyboard control
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/terraintessellation
 * -------------------------------------------------------------------------- */

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

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

#include "core/camera.h"
#include "core/gltf_model.h"
#include "core/image_loader.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations - defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* terrain_shader_wgsl;
static const char* skysphere_shader_wgsl;
static const char* compute_tess_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define PATCH_SIZE 64
#define TERRAIN_GRID_SIZE 256
#define TERRAIN_LAYER_COUNT 6
#define TERRAIN_ARRAY_TEX_SIZE 512
#define HEIGHTMAP_TEX_SIZE 1024
#define FETCH_BUFFER_SIZE (5 * 1024 * 1024)

/* -------------------------------------------------------------------------- *
 * Types
 * -------------------------------------------------------------------------- */

typedef struct {
  float pos[3];
  float normal[3];
  float uv[2];
} terrain_vertex_t; /* 32 bytes */

/* Terrain uniform buffer (std140-compatible) */
typedef struct {
  float projection[16]; /* mat4x4f  offset 0   */
  float modelview[16];  /* mat4x4f  offset 64  */
  float light_pos[4];   /* vec4f    offset 128 */
  float disp_factor;    /* f32      offset 144 */
  float pad[3];         /* padding  offset 148 */
} terrain_ubo_t;        /* 160 bytes */

/* Skysphere uniform buffer */
typedef struct {
  float mvp[16]; /* mat4x4f  offset 0 */
} sky_ubo_t;     /* 64 bytes */

/* Compute shader uniform buffer */
typedef struct {
  float disp_factor; /* f32    offset 0  */
  uint32_t pad[3];   /* padding to 16 bytes */
} compute_ubo_t;     /* 16 bytes */

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Terrain geometry */
  struct {
    WGPUBuffer input_vertex_buffer; /* flat mesh (Storage, read by compute)  */
    WGPUBuffer vertex_buffer; /* displaced mesh (Vertex|Storage, output) */
    WGPUBuffer index_buffer;
    WGPUBuffer wire_index_buffer;
    WGPUBuffer coarse_index_buffer;
    WGPUBuffer coarse_wire_index_buffer;
    uint32_t index_count;
    uint32_t wire_index_count;
    uint32_t coarse_index_count;
    uint32_t coarse_wire_index_count;
    uint32_t vertex_count;
  } terrain;

  /* Skysphere model */
  struct {
    gltf_model_t model;
    WGPUBuffer vertex_buffer;
    WGPUBuffer index_buffer;
    bool loaded;
  } sky;

  /* Textures and samplers */
  struct {
    WGPUTexture heightmap;
    WGPUTextureView heightmap_view;
    WGPUSampler heightmap_sampler;
    WGPUTexture terrain_array;
    WGPUTextureView terrain_array_view;
    WGPUSampler terrain_array_sampler;
    WGPUTexture skysphere;
    WGPUTextureView skysphere_view;
    WGPUSampler skysphere_sampler;
    bool terrain_array_ready;
    bool skysphere_ready;
  } tex;

  /* Fetch buffers for async texture loading */
  uint8_t fetch_buffer[FETCH_BUFFER_SIZE];

  /* Uniform data (CPU side) */
  terrain_ubo_t terrain_ubo;
  sky_ubo_t sky_ubo;
  compute_ubo_t compute_ubo;

  /* Uniform GPU buffers */
  struct {
    wgpu_buffer_t terrain;
    wgpu_buffer_t sky;
    wgpu_buffer_t compute;
  } uniform_bufs;

  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout terrain;
    WGPUBindGroupLayout skysphere;
    WGPUBindGroupLayout compute;
  } bg_layouts;

  /* Pipeline layouts */
  struct {
    WGPUPipelineLayout terrain;
    WGPUPipelineLayout skysphere;
    WGPUPipelineLayout compute;
  } pipe_layouts;

  /* Bind groups */
  struct {
    WGPUBindGroup terrain;
    WGPUBindGroup skysphere;
    WGPUBindGroup compute;
  } bind_groups;

  /* Render pipelines */
  struct {
    WGPURenderPipeline terrain;
    WGPURenderPipeline wireframe;
    WGPURenderPipeline skysphere;
  } pipelines;

  /* Compute pipeline */
  WGPUComputePipeline compute_pipeline;

  /* Render pass */
  WGPURenderPassColorAttachment color_att;
  WGPURenderPassDepthStencilAttachment depth_att;
  WGPURenderPassDescriptor render_pass_desc;

  /* GUI settings */
  struct {
    bool tessellation;
    bool wireframe;
    float displacement_factor;
  } settings;

  /* Timing */
  uint64_t last_frame_time;

  /* Readiness */
  bool initialized;
} state = {
  .settings =
    {
      .tessellation        = true,
      .wireframe           = false,
      .displacement_factor = 32.0f,
    },
  /* Render pass descriptors */
  .color_att =
    {
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.47f, 0.5f, 0.67f, 1.0f},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    },
  .depth_att =
    {
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    },
  .render_pass_desc =
    {
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.color_att,
      .depthStencilAttachment = &state.depth_att,
    },
};

/* -------------------------------------------------------------------------- *
 * Heightmap loading (synchronous)
 * -------------------------------------------------------------------------- */

static uint8_t* load_heightmap(int* out_w, int* out_h)
{
  int channels    = 0;
  uint8_t* pixels = image_pixels_from_file(
    "assets/textures/terrain_heightmap_r16.png", out_w, out_h, &channels, 1);
  if (!pixels) {
    printf("ERROR: Failed to load terrain heightmap!\n");
  }
  return pixels;
}

/* -------------------------------------------------------------------------- *
 * Terrain mesh generation
 * -------------------------------------------------------------------------- */

static void generate_terrain_mesh(wgpu_context_t* wgpu_context,
                                  const uint8_t* heightdata, int hm_dim)
{
  UNUSED_VAR(heightdata);
  UNUSED_VAR(hm_dim);

  const uint32_t grid  = TERRAIN_GRID_SIZE;
  const float wx       = 2.0f;
  const float wz       = 2.0f;
  const float uv_scale = 1.0f;

  /* --- Flat vertices (Y=0, placeholder normals) --- */
  /* Displacement + normals will be computed by the GPU compute shader. */
  const uint32_t vert_count = grid * grid;
  terrain_vertex_t* verts
    = (terrain_vertex_t*)malloc(vert_count * sizeof(terrain_vertex_t));
  if (!verts) {
    return;
  }

  for (uint32_t x = 0; x < grid; x++) {
    for (uint32_t y = 0; y < grid; y++) {
      uint32_t idx      = x + y * grid;
      verts[idx].pos[0] = (float)x * wx + wx / 2.0f - (float)grid * wx / 2.0f;
      verts[idx].pos[1] = 0.0f; /* flat — displaced by compute */
      verts[idx].pos[2] = (float)y * wz + wz / 2.0f - (float)grid * wz / 2.0f;
      verts[idx].normal[0] = 0.0f;
      verts[idx].normal[1] = 1.0f; /* +Y up placeholder */
      verts[idx].normal[2] = 0.0f;
      verts[idx].uv[0]     = ((float)x / (float)(grid - 1)) * uv_scale;
      verts[idx].uv[1]     = ((float)y / (float)(grid - 1)) * uv_scale;
    }
  }

  /* --- Triangle indices ------------------------------------------------- */
  const uint32_t w           = grid - 1;
  const uint32_t tri_count   = w * w * 2;
  const uint32_t index_count = tri_count * 3;
  uint32_t* indices = (uint32_t*)malloc(index_count * sizeof(uint32_t));
  if (!indices) {
    free(verts);
    return;
  }
  uint32_t idx = 0;
  for (uint32_t x = 0; x < w; x++) {
    for (uint32_t y = 0; y < w; y++) {
      uint32_t v0 = x + y * grid;
      uint32_t v1 = x + (y + 1) * grid;
      uint32_t v2 = x + 1 + (y + 1) * grid;
      uint32_t v3 = x + 1 + y * grid;
      /* Triangle 1 (CCW from above) */
      indices[idx++] = v0;
      indices[idx++] = v1;
      indices[idx++] = v2;
      /* Triangle 2 (CCW from above) */
      indices[idx++] = v0;
      indices[idx++] = v2;
      indices[idx++] = v3;
    }
  }

  /* --- Wireframe indices (line-list) ------------------------------------ */
  /* Horizontal + vertical + diagonal lines */
  const uint32_t wire_count
    = grid * w + w * grid + w * w; /* h + v + diag lines */
  const uint32_t wire_idx_count = wire_count * 2;
  uint32_t* wire_indices = (uint32_t*)malloc(wire_idx_count * sizeof(uint32_t));
  if (!wire_indices) {
    free(verts);
    free(indices);
    return;
  }
  uint32_t wi = 0;
  /* Horizontal lines */
  for (uint32_t y = 0; y < grid; y++) {
    for (uint32_t x = 0; x < w; x++) {
      wire_indices[wi++] = x + y * grid;
      wire_indices[wi++] = x + 1 + y * grid;
    }
  }
  /* Vertical lines */
  for (uint32_t x = 0; x < grid; x++) {
    for (uint32_t y = 0; y < w; y++) {
      wire_indices[wi++] = x + y * grid;
      wire_indices[wi++] = x + (y + 1) * grid;
    }
  }
  /* Diagonal lines */
  for (uint32_t x = 0; x < w; x++) {
    for (uint32_t y = 0; y < w; y++) {
      wire_indices[wi++] = x + y * grid;
      wire_indices[wi++] = x + 1 + (y + 1) * grid;
    }
  }

  /* --- Upload to GPU ---------------------------------------------------- */
  WGPUDevice device = wgpu_context->device;

  size_t vb_size = vert_count * sizeof(terrain_vertex_t);

  /* Input vertex buffer: flat mesh, read by compute shader */
  state.terrain.input_vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Terrain Input VB"),
              .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
              .size  = vb_size,
              .mappedAtCreation = true,
            });
  void* vdata
    = wgpuBufferGetMappedRange(state.terrain.input_vertex_buffer, 0, vb_size);
  memcpy(vdata, verts, vb_size);
  wgpuBufferUnmap(state.terrain.input_vertex_buffer);
  free(verts);

  /* Output vertex buffer: displaced mesh, written by compute, read by render */
  state.terrain.vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Terrain Output VB"),
              .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage,
              .size  = vb_size,
            });

  size_t ib_size             = index_count * sizeof(uint32_t);
  state.terrain.index_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Terrain IB"),
              .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
              .size  = ib_size,
              .mappedAtCreation = true,
            });
  void* idata
    = wgpuBufferGetMappedRange(state.terrain.index_buffer, 0, ib_size);
  memcpy(idata, indices, ib_size);
  wgpuBufferUnmap(state.terrain.index_buffer);
  free(indices);

  size_t wib_size                 = wire_idx_count * sizeof(uint32_t);
  state.terrain.wire_index_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Terrain Wire IB"),
              .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
              .size  = wib_size,
              .mappedAtCreation = true,
            });
  void* widata
    = wgpuBufferGetMappedRange(state.terrain.wire_index_buffer, 0, wib_size);
  memcpy(widata, wire_indices, wib_size);
  wgpuBufferUnmap(state.terrain.wire_index_buffer);
  free(wire_indices);

  /* --- Coarse index buffers (tessellation OFF = Vulkan tess level 1) ---- */
  const uint32_t coarse = PATCH_SIZE; /* 64 */
  const uint32_t cstep  = grid / coarse;
  const uint32_t cw     = coarse - 1;

  /* Coarse triangle indices */
  const uint32_t ci_count = cw * cw * 2 * 3;
  uint32_t* ci            = (uint32_t*)malloc(ci_count * sizeof(uint32_t));
  uint32_t ci_idx         = 0;
  for (uint32_t cx = 0; cx < cw; cx++) {
    for (uint32_t cy = 0; cy < cw; cy++) {
      uint32_t v0  = (cx * cstep) + (cy * cstep) * grid;
      uint32_t v1  = (cx * cstep) + ((cy + 1) * cstep) * grid;
      uint32_t v2  = ((cx + 1) * cstep) + ((cy + 1) * cstep) * grid;
      uint32_t v3  = ((cx + 1) * cstep) + (cy * cstep) * grid;
      ci[ci_idx++] = v0;
      ci[ci_idx++] = v1;
      ci[ci_idx++] = v2;
      ci[ci_idx++] = v0;
      ci[ci_idx++] = v2;
      ci[ci_idx++] = v3;
    }
  }
  size_t ci_size                    = ci_count * sizeof(uint32_t);
  state.terrain.coarse_index_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Terrain Coarse IB"),
              .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
              .size  = ci_size,
              .mappedAtCreation = true,
            });
  void* cidata
    = wgpuBufferGetMappedRange(state.terrain.coarse_index_buffer, 0, ci_size);
  memcpy(cidata, ci, ci_size);
  wgpuBufferUnmap(state.terrain.coarse_index_buffer);
  free(ci);

  /* Coarse wireframe indices (line-list) */
  const uint32_t cwl_h     = coarse * cw; /* horizontal */
  const uint32_t cwl_v     = cw * coarse; /* vertical */
  const uint32_t cwl_d     = cw * cw;     /* diagonal */
  const uint32_t cwi_count = (cwl_h + cwl_v + cwl_d) * 2;
  uint32_t* cwi            = (uint32_t*)malloc(cwi_count * sizeof(uint32_t));
  uint32_t cwi_idx         = 0;
  for (uint32_t cy = 0; cy < coarse; cy++) {
    for (uint32_t cx = 0; cx < cw; cx++) {
      cwi[cwi_idx++] = (cx * cstep) + (cy * cstep) * grid;
      cwi[cwi_idx++] = ((cx + 1) * cstep) + (cy * cstep) * grid;
    }
  }
  for (uint32_t cx = 0; cx < coarse; cx++) {
    for (uint32_t cy = 0; cy < cw; cy++) {
      cwi[cwi_idx++] = (cx * cstep) + (cy * cstep) * grid;
      cwi[cwi_idx++] = (cx * cstep) + ((cy + 1) * cstep) * grid;
    }
  }
  for (uint32_t cx = 0; cx < cw; cx++) {
    for (uint32_t cy = 0; cy < cw; cy++) {
      cwi[cwi_idx++] = (cx * cstep) + (cy * cstep) * grid;
      cwi[cwi_idx++] = ((cx + 1) * cstep) + ((cy + 1) * cstep) * grid;
    }
  }
  size_t cwi_size                        = cwi_count * sizeof(uint32_t);
  state.terrain.coarse_wire_index_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Terrain Coarse Wire IB"),
              .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
              .size  = cwi_size,
              .mappedAtCreation = true,
            });
  void* cwidata = wgpuBufferGetMappedRange(
    state.terrain.coarse_wire_index_buffer, 0, cwi_size);
  memcpy(cwidata, cwi, cwi_size);
  wgpuBufferUnmap(state.terrain.coarse_wire_index_buffer);
  free(cwi);

  state.terrain.vertex_count            = vert_count;
  state.terrain.index_count             = index_count;
  state.terrain.wire_index_count        = wire_idx_count;
  state.terrain.coarse_index_count      = ci_count;
  state.terrain.coarse_wire_index_count = cwi_count;
}

/* -------------------------------------------------------------------------- *
 * Heightmap GPU texture
 * -------------------------------------------------------------------------- */

static void create_heightmap_texture(wgpu_context_t* wgpu_context,
                                     const uint8_t* pixels, int w, int h)
{
  WGPUDevice device = wgpu_context->device;

  state.tex.heightmap
    = wgpuDeviceCreateTexture(device, &(WGPUTextureDescriptor){
                                        .label = STRVIEW("Heightmap"),
                                        .usage = WGPUTextureUsage_TextureBinding
                                                 | WGPUTextureUsage_CopyDst,
                                        .dimension = WGPUTextureDimension_2D,
                                        .size   = {(uint32_t)w, (uint32_t)h, 1},
                                        .format = WGPUTextureFormat_R8Unorm,
                                        .mipLevelCount = 1,
                                        .sampleCount   = 1,
                                      });

  wgpuQueueWriteTexture(
    wgpu_context->queue,
    &(WGPUTexelCopyTextureInfo){.texture = state.tex.heightmap}, pixels,
    (size_t)(w * h),
    &(WGPUTexelCopyBufferLayout){.bytesPerRow  = (uint32_t)w,
                                 .rowsPerImage = (uint32_t)h},
    &(WGPUExtent3D){(uint32_t)w, (uint32_t)h, 1});

  state.tex.heightmap_view = wgpuTextureCreateView(
    state.tex.heightmap, &(WGPUTextureViewDescriptor){
                           .format          = WGPUTextureFormat_R8Unorm,
                           .dimension       = WGPUTextureViewDimension_2D,
                           .mipLevelCount   = 1,
                           .arrayLayerCount = 1,
                         });

  state.tex.heightmap_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Heightmap Sampler"),
              .addressModeU  = WGPUAddressMode_MirrorRepeat,
              .addressModeV  = WGPUAddressMode_MirrorRepeat,
              .addressModeW  = WGPUAddressMode_MirrorRepeat,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .maxAnisotropy = 1,
            });
}

/* -------------------------------------------------------------------------- *
 * Terrain texture array (2D array with 6 layers)
 * -------------------------------------------------------------------------- */

static void create_terrain_array_texture(wgpu_context_t* wgpu_context,
                                         const uint8_t* pixels, int w,
                                         int total_h, int layer_count)
{
  WGPUDevice device = wgpu_context->device;
  int layer_h       = total_h / layer_count;

  state.tex.terrain_array = wgpuDeviceCreateTexture(
    device,
    &(WGPUTextureDescriptor){
      .label     = STRVIEW("Terrain Layers"),
      .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size      = {(uint32_t)w, (uint32_t)layer_h, (uint32_t)layer_count},
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });

  size_t layer_bytes = (size_t)(w * layer_h * 4);
  for (int i = 0; i < layer_count; i++) {
    wgpuQueueWriteTexture(wgpu_context->queue,
                          &(WGPUTexelCopyTextureInfo){
                            .texture = state.tex.terrain_array,
                            .origin  = {0, 0, (uint32_t)i},
                          },
                          pixels + (size_t)i * layer_bytes, layer_bytes,
                          &(WGPUTexelCopyBufferLayout){
                            .bytesPerRow  = (uint32_t)(w * 4),
                            .rowsPerImage = (uint32_t)layer_h,
                          },
                          &(WGPUExtent3D){(uint32_t)w, (uint32_t)layer_h, 1});
  }

  state.tex.terrain_array_view = wgpuTextureCreateView(
    state.tex.terrain_array, &(WGPUTextureViewDescriptor){
                               .format    = WGPUTextureFormat_RGBA8Unorm,
                               .dimension = WGPUTextureViewDimension_2DArray,
                               .mipLevelCount   = 1,
                               .baseArrayLayer  = 0,
                               .arrayLayerCount = (uint32_t)layer_count,
                             });

  state.tex.terrain_array_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Terrain Layer Sampler"),
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_Repeat,
              .addressModeW  = WGPUAddressMode_Repeat,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .maxAnisotropy = 4,
            });

  state.tex.terrain_array_ready = true;
}

/* -------------------------------------------------------------------------- *
 * Skysphere texture
 * -------------------------------------------------------------------------- */

static void create_skysphere_texture(wgpu_context_t* wgpu_context,
                                     const uint8_t* pixels, int w, int h)
{
  WGPUDevice device = wgpu_context->device;

  state.tex.skysphere
    = wgpuDeviceCreateTexture(device, &(WGPUTextureDescriptor){
                                        .label = STRVIEW("Skysphere"),
                                        .usage = WGPUTextureUsage_TextureBinding
                                                 | WGPUTextureUsage_CopyDst,
                                        .dimension = WGPUTextureDimension_2D,
                                        .size   = {(uint32_t)w, (uint32_t)h, 1},
                                        .format = WGPUTextureFormat_RGBA8Unorm,
                                        .mipLevelCount = 1,
                                        .sampleCount   = 1,
                                      });

  wgpuQueueWriteTexture(
    wgpu_context->queue,
    &(WGPUTexelCopyTextureInfo){.texture = state.tex.skysphere}, pixels,
    (size_t)(w * h * 4),
    &(WGPUTexelCopyBufferLayout){
      .bytesPerRow  = (uint32_t)(w * 4),
      .rowsPerImage = (uint32_t)h,
    },
    &(WGPUExtent3D){(uint32_t)w, (uint32_t)h, 1});

  state.tex.skysphere_view = wgpuTextureCreateView(
    state.tex.skysphere, &(WGPUTextureViewDescriptor){
                           .format          = WGPUTextureFormat_RGBA8Unorm,
                           .dimension       = WGPUTextureViewDimension_2D,
                           .mipLevelCount   = 1,
                           .arrayLayerCount = 1,
                         });

  state.tex.skysphere_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Skysphere Sampler"),
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .maxAnisotropy = 1,
            });

  state.tex.skysphere_ready = true;
}

/* -------------------------------------------------------------------------- *
 * Placeholder textures (shown while async loading is in progress)
 * -------------------------------------------------------------------------- */

static void create_placeholder_textures(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* 1x1 gray terrain array */
  uint8_t gray[4] = {128, 128, 128, 255};

  state.tex.terrain_array = wgpuDeviceCreateTexture(
    device,
    &(WGPUTextureDescriptor){
      .label     = STRVIEW("Terrain Layers Placeholder"),
      .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size      = {1, 1, TERRAIN_LAYER_COUNT},
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });
  for (int i = 0; i < TERRAIN_LAYER_COUNT; i++) {
    wgpuQueueWriteTexture(
      wgpu_context->queue,
      &(WGPUTexelCopyTextureInfo){
        .texture = state.tex.terrain_array,
        .origin  = {0, 0, (uint32_t)i},
      },
      gray, 4,
      &(WGPUTexelCopyBufferLayout){.bytesPerRow = 4, .rowsPerImage = 1},
      &(WGPUExtent3D){1, 1, 1});
  }
  state.tex.terrain_array_view = wgpuTextureCreateView(
    state.tex.terrain_array, &(WGPUTextureViewDescriptor){
                               .format    = WGPUTextureFormat_RGBA8Unorm,
                               .dimension = WGPUTextureViewDimension_2DArray,
                               .mipLevelCount   = 1,
                               .arrayLayerCount = TERRAIN_LAYER_COUNT,
                             });
  state.tex.terrain_array_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Terrain Layer Sampler Placeholder"),
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_Repeat,
              .addressModeW  = WGPUAddressMode_Repeat,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .maxAnisotropy = 1,
            });

  /* 1x1 blue skysphere */
  uint8_t blue[4] = {120, 128, 171, 255};

  state.tex.skysphere = wgpuDeviceCreateTexture(
    device,
    &(WGPUTextureDescriptor){
      .label     = STRVIEW("Skysphere Placeholder"),
      .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size      = {1, 1, 1},
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });
  wgpuQueueWriteTexture(
    wgpu_context->queue,
    &(WGPUTexelCopyTextureInfo){.texture = state.tex.skysphere}, blue, 4,
    &(WGPUTexelCopyBufferLayout){.bytesPerRow = 4, .rowsPerImage = 1},
    &(WGPUExtent3D){1, 1, 1});
  state.tex.skysphere_view = wgpuTextureCreateView(
    state.tex.skysphere, &(WGPUTextureViewDescriptor){
                           .format          = WGPUTextureFormat_RGBA8Unorm,
                           .dimension       = WGPUTextureViewDimension_2D,
                           .mipLevelCount   = 1,
                           .arrayLayerCount = 1,
                         });
  state.tex.skysphere_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Skysphere Sampler Placeholder"),
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .maxAnisotropy = 1,
            });
}

/* -------------------------------------------------------------------------- *
 * Async texture fetch callbacks
 * -------------------------------------------------------------------------- */

/* Forward declaration for bind group recreation */
static void create_bind_groups(wgpu_context_t* wgpu_context);

static wgpu_context_t* s_wgpu_ctx = NULL; /* set in init() */

static void terrain_array_fetch_cb(const sfetch_response_t* response)
{
  if (response->failed) {
    printf("ERROR: terrain texture array fetch failed (%d)\n",
           response->error_code);
    return;
  }
  if (!response->fetched) {
    return;
  }

  int w, h, channels;
  uint8_t* pixels
    = image_pixels_from_memory((const uint8_t*)response->data.ptr,
                               (int)response->data.size, &w, &h, &channels, 4);
  if (!pixels) {
    printf("ERROR: failed to decode terrain texture array\n");
    return;
  }

  /* Destroy placeholder */
  WGPU_RELEASE_RESOURCE(TextureView, state.tex.terrain_array_view)
  WGPU_RELEASE_RESOURCE(Texture, state.tex.terrain_array)
  WGPU_RELEASE_RESOURCE(Sampler, state.tex.terrain_array_sampler)

  create_terrain_array_texture(s_wgpu_ctx, pixels, w, h, TERRAIN_LAYER_COUNT);
  image_free(pixels);

  /* Rebuild bind groups with the real texture */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.terrain)
  create_bind_groups(s_wgpu_ctx);
}

static void skysphere_fetch_cb(const sfetch_response_t* response)
{
  if (response->failed) {
    printf("ERROR: skysphere texture fetch failed (%d)\n",
           response->error_code);
    return;
  }
  if (!response->fetched) {
    return;
  }

  int w, h, channels;
  uint8_t* pixels
    = image_pixels_from_memory((const uint8_t*)response->data.ptr,
                               (int)response->data.size, &w, &h, &channels, 4);
  if (!pixels) {
    printf("ERROR: failed to decode skysphere texture\n");
    return;
  }

  /* Destroy placeholder */
  WGPU_RELEASE_RESOURCE(TextureView, state.tex.skysphere_view)
  WGPU_RELEASE_RESOURCE(Texture, state.tex.skysphere)
  WGPU_RELEASE_RESOURCE(Sampler, state.tex.skysphere_sampler)

  create_skysphere_texture(s_wgpu_ctx, pixels, w, h);
  image_free(pixels);

  /* Rebuild bind groups with the real texture */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.skysphere)
  create_bind_groups(s_wgpu_ctx);
}

/* -------------------------------------------------------------------------- *
 * Skysphere model loading
 * -------------------------------------------------------------------------- */

static void load_skysphere_model(wgpu_context_t* wgpu_context)
{
  bool ok = gltf_model_load_from_file(&state.sky.model,
                                      "assets/models/sphere.gltf", 1.0f);
  if (!ok) {
    printf("ERROR: Failed to load sphere.gltf\n");
    return;
  }

  WGPUDevice device = wgpu_context->device;
  gltf_model_t* m   = &state.sky.model;

  size_t vb_sz            = m->vertex_count * sizeof(gltf_vertex_t);
  state.sky.vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Sky VB"),
              .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
              .size  = vb_sz,
              .mappedAtCreation = true,
            });
  void* vd = wgpuBufferGetMappedRange(state.sky.vertex_buffer, 0, vb_sz);
  memcpy(vd, m->vertices, vb_sz);
  wgpuBufferUnmap(state.sky.vertex_buffer);

  if (m->index_count > 0) {
    size_t ib_sz           = m->index_count * sizeof(uint32_t);
    state.sky.index_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Sky IB"),
                .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                .size  = ib_sz,
                .mappedAtCreation = true,
              });
    void* id = wgpuBufferGetMappedRange(state.sky.index_buffer, 0, ib_sz);
    memcpy(id, m->indices, ib_sz);
    wgpuBufferUnmap(state.sky.index_buffer);
  }

  state.sky.loaded = true;
}

/* -------------------------------------------------------------------------- *
 * Camera setup
 * -------------------------------------------------------------------------- */

static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type           = CameraType_FirstPerson;
  state.camera.movement_speed = 10.0f;
  state.camera.rotation_speed = 0.25f;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;

  /* Vulkan original: position(18, 22.5, 57.5), rotation(-12, 159, 0)
   * Note: camera_set_position already negates Y internally, so pass Vulkan
   * values directly (do NOT use VKY_TO_WGPU_VEC3 — that creates a double
   * negation). VKY_TO_WGPU_CAM_ROT is still needed for rotation since
   * camera_set_rotation does not negate pitch. */
  camera_set_position(&state.camera, (vec3){18.0f, 22.5f, 57.5f});
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-12.0f, 159.0f, 0.0f));
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 512.0f);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void create_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.uniform_bufs.terrain = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Terrain UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(terrain_ubo_t),
                  });

  state.uniform_bufs.sky = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sky UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(sky_ubo_t),
                  });

  state.uniform_bufs.compute = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(compute_ubo_t),
                  });
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  camera_t* cam = &state.camera;

  /* Terrain UBO */
  glm_mat4_copy(cam->matrices.perspective, (vec4*)state.terrain_ubo.projection);
  glm_mat4_copy(cam->matrices.view, (vec4*)state.terrain_ubo.modelview);

  /* Light position: Vulkan computes lightPos.y = -0.5 - displacementFactor
   * (placing light just below terrain peaks). For WebGPU Y-up with positive
   * displacement, the equivalent is +0.5 + displacementFactor. */
  state.terrain_ubo.light_pos[0] = -48.0f;
  state.terrain_ubo.light_pos[1] = 0.5f + state.settings.displacement_factor;
  state.terrain_ubo.light_pos[2] = 46.0f;
  state.terrain_ubo.light_pos[3] = 0.0f;

  /* Displacement factor — always applied; tessellation toggle selects mesh
     resolution (coarse vs. fine) but displacement remains constant. */
  state.terrain_ubo.disp_factor = state.settings.displacement_factor;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_bufs.terrain.buffer,
                       0, &state.terrain_ubo, sizeof(terrain_ubo_t));

  /* Compute UBO */
  state.compute_ubo.disp_factor = state.settings.displacement_factor;
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_bufs.compute.buffer,
                       0, &state.compute_ubo, sizeof(compute_ubo_t));

  /* Skysphere UBO: projection * rotation-only view */
  mat4 sky_view;
  glm_mat4_identity(sky_view);
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      sky_view[r][c] = cam->matrices.view[r][c];
    }
  }
  mat4 mvp;
  glm_mat4_mul(cam->matrices.perspective, sky_view, mvp);
  glm_mat4_copy(mvp, (vec4*)state.sky_ubo.mvp);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_bufs.sky.buffer, 0,
                       &state.sky_ubo, sizeof(sky_ubo_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void create_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Terrain: UBO + heightmap (sampler+tex) + layer array (sampler+tex) */
  WGPUBindGroupLayoutEntry terrain_entries[5] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer     = {.type = WGPUBufferBindingType_Uniform,
                     .minBindingSize = sizeof(terrain_ubo_t)},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    },
    [3] = {
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
    [4] = {
      .binding    = 4,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2DArray},
    },
  };
  state.bg_layouts.terrain
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .label = STRVIEW("Terrain BGL"),
                                                .entryCount = 5,
                                                .entries    = terrain_entries,
                                              });

  /* Skysphere: UBO + sampler + tex */
  WGPUBindGroupLayoutEntry sky_entries[3] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer     = {.type = WGPUBufferBindingType_Uniform,
                     .minBindingSize = sizeof(sky_ubo_t)},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    },
  };
  state.bg_layouts.skysphere
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .label = STRVIEW("Sky BGL"),
                                                .entryCount = 3,
                                                .entries    = sky_entries,
                                              });

  /* Compute: input verts (storage r), output verts (storage rw), UBO,
     heightmap tex + sampler */
  size_t vb_size
    = (size_t)TERRAIN_GRID_SIZE * TERRAIN_GRID_SIZE * sizeof(terrain_vertex_t);
  WGPUBindGroupLayoutEntry compute_entries[5] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {.type = WGPUBufferBindingType_ReadOnlyStorage,
                     .minBindingSize = vb_size},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {.type = WGPUBufferBindingType_Storage,
                     .minBindingSize = vb_size},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {.type = WGPUBufferBindingType_Uniform,
                     .minBindingSize = sizeof(compute_ubo_t)},
    },
    [3] = {
      .binding    = 3,
      .visibility = WGPUShaderStage_Compute,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    },
    [4] = {
      .binding    = 4,
      .visibility = WGPUShaderStage_Compute,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
  };
  state.bg_layouts.compute
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .label = STRVIEW("Compute BGL"),
                                                .entryCount = 5,
                                                .entries    = compute_entries,
                                              });
}

/* -------------------------------------------------------------------------- *
 * Pipeline layouts
 * -------------------------------------------------------------------------- */

static void create_pipeline_layouts(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.pipe_layouts.terrain = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Terrain PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.bg_layouts.terrain,
            });

  state.pipe_layouts.skysphere = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Sky PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.bg_layouts.skysphere,
            });

  state.pipe_layouts.compute = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Compute PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.bg_layouts.compute,
            });
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void create_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Terrain */
  if (!state.bind_groups.terrain) {
    WGPUBindGroupEntry terrain_bg_entries[5] = {
      [0] = {.binding = 0,
             .buffer  = state.uniform_bufs.terrain.buffer,
             .size    = sizeof(terrain_ubo_t)},
      [1] = {.binding = 1, .sampler = state.tex.heightmap_sampler},
      [2] = {.binding = 2, .textureView = state.tex.heightmap_view},
      [3] = {.binding = 3, .sampler = state.tex.terrain_array_sampler},
      [4] = {.binding = 4, .textureView = state.tex.terrain_array_view},
    };
    state.bind_groups.terrain
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label  = STRVIEW("Terrain BG"),
                                            .layout = state.bg_layouts.terrain,
                                            .entryCount = 5,
                                            .entries    = terrain_bg_entries,
                                          });
  }

  /* Skysphere */
  if (!state.bind_groups.skysphere) {
    WGPUBindGroupEntry sky_bg_entries[3] = {
      [0] = {.binding = 0,
             .buffer  = state.uniform_bufs.sky.buffer,
             .size    = sizeof(sky_ubo_t)},
      [1] = {.binding = 1, .sampler = state.tex.skysphere_sampler},
      [2] = {.binding = 2, .textureView = state.tex.skysphere_view},
    };
    state.bind_groups.skysphere = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Sky BG"),
                .layout     = state.bg_layouts.skysphere,
                .entryCount = 3,
                .entries    = sky_bg_entries,
              });
  }

  /* Compute */
  if (!state.bind_groups.compute) {
    size_t vb_size = (size_t)TERRAIN_GRID_SIZE * TERRAIN_GRID_SIZE
                     * sizeof(terrain_vertex_t);
    WGPUBindGroupEntry compute_bg_entries[5] = {
      [0] = {.binding = 0,
             .buffer  = state.terrain.input_vertex_buffer,
             .size    = vb_size},
      [1]
      = {.binding = 1, .buffer = state.terrain.vertex_buffer, .size = vb_size},
      [2] = {.binding = 2,
             .buffer  = state.uniform_bufs.compute.buffer,
             .size    = sizeof(compute_ubo_t)},
      [3] = {.binding = 3, .textureView = state.tex.heightmap_view},
      [4] = {.binding = 4, .sampler = state.tex.heightmap_sampler},
    };
    state.bind_groups.compute
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label  = STRVIEW("Compute BG"),
                                            .layout = state.bg_layouts.compute,
                                            .entryCount = 5,
                                            .entries    = compute_bg_entries,
                                          });
  }
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void create_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* --- Shared vertex layout for terrain --------------------------------- */
  WGPUVertexAttribute terrain_attrs[3] = {
    {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
    {.format = WGPUVertexFormat_Float32x3, .offset = 12, .shaderLocation = 1},
    {.format = WGPUVertexFormat_Float32x2, .offset = 24, .shaderLocation = 2},
  };
  WGPUVertexBufferLayout terrain_vbl = {
    .arrayStride    = sizeof(terrain_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 3,
    .attributes     = terrain_attrs,
  };

  /* --- Terrain shader module -------------------------------------------- */
  WGPUShaderModule terrain_sm
    = wgpu_create_shader_module(device, terrain_shader_wgsl);

  /* --- Terrain fill pipeline -------------------------------------------- */
  WGPUColorTargetState terrain_ct = {
    .format    = wgpu_context->render_format,
    .writeMask = WGPUColorWriteMask_All,
  };
  state.pipelines.terrain = wgpuDeviceCreateRenderPipeline(
    device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Terrain Pipeline"),
      .layout = state.pipe_layouts.terrain,
      .vertex =
        {
          .module      = terrain_sm,
          .entryPoint  = STRVIEW("vs_terrain"),
          .bufferCount = 1,
          .buffers     = &terrain_vbl,
        },
      .primitive =
        {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Back,
        },
      .depthStencil =
        &(WGPUDepthStencilState){
          .format            = wgpu_context->depth_stencil_format,
          .depthWriteEnabled = true,
          .depthCompare      = WGPUCompareFunction_LessEqual,
        },
      .multisample = {.count = 1, .mask = ~0u},
      .fragment =
        &(WGPUFragmentState){
          .module      = terrain_sm,
          .entryPoint  = STRVIEW("fs_terrain"),
          .targetCount = 1,
          .targets     = &terrain_ct,
        },
    });

  /* --- Wireframe pipeline (same VS, solid-color FS, line-list) ---------- */
  WGPUColorTargetState wire_ct = {
    .format    = wgpu_context->render_format,
    .writeMask = WGPUColorWriteMask_All,
  };
  state.pipelines.wireframe = wgpuDeviceCreateRenderPipeline(
    device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Wireframe Pipeline"),
      .layout = state.pipe_layouts.terrain,
      .vertex =
        {
          .module      = terrain_sm,
          .entryPoint  = STRVIEW("vs_terrain"),
          .bufferCount = 1,
          .buffers     = &terrain_vbl,
        },
      .primitive =
        {
          .topology  = WGPUPrimitiveTopology_LineList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_None,
        },
      .depthStencil =
        &(WGPUDepthStencilState){
          .format            = wgpu_context->depth_stencil_format,
          .depthWriteEnabled = true,
          .depthCompare      = WGPUCompareFunction_LessEqual,
        },
      .multisample = {.count = 1, .mask = ~0u},
      .fragment =
        &(WGPUFragmentState){
          .module      = terrain_sm,
          .entryPoint  = STRVIEW("fs_wireframe"),
          .targetCount = 1,
          .targets     = &wire_ct,
        },
    });

  wgpuShaderModuleRelease(terrain_sm);

  /* --- Skysphere shader module ------------------------------------------ */
  WGPUShaderModule sky_sm
    = wgpu_create_shader_module(device, skysphere_shader_wgsl);

  /* --- Skysphere vertex layout (gltf_vertex_t stride) ------------------- */
  WGPUVertexAttribute sky_attrs[3] = {
    {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
    {.format = WGPUVertexFormat_Float32x3, .offset = 12, .shaderLocation = 1},
    {.format = WGPUVertexFormat_Float32x2, .offset = 24, .shaderLocation = 2},
  };
  WGPUVertexBufferLayout sky_vbl = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 3,
    .attributes     = sky_attrs,
  };

  WGPUColorTargetState sky_ct = {
    .format    = wgpu_context->render_format,
    .writeMask = WGPUColorWriteMask_All,
  };
  state.pipelines.skysphere = wgpuDeviceCreateRenderPipeline(
    device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Sky Pipeline"),
      .layout = state.pipe_layouts.skysphere,
      .vertex =
        {
          .module      = sky_sm,
          .entryPoint  = STRVIEW("vs_sky"),
          .bufferCount = 1,
          .buffers     = &sky_vbl,
        },
      .primitive =
        {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Front, /* render inside of sphere */
        },
      .depthStencil =
        &(WGPUDepthStencilState){
          .format            = wgpu_context->depth_stencil_format,
          .depthWriteEnabled = false, /* sky behind everything */
          .depthCompare      = WGPUCompareFunction_LessEqual,
        },
      .multisample = {.count = 1, .mask = ~0u},
      .fragment =
        &(WGPUFragmentState){
          .module      = sky_sm,
          .entryPoint  = STRVIEW("fs_sky"),
          .targetCount = 1,
          .targets     = &sky_ct,
        },
    });

  wgpuShaderModuleRelease(sky_sm);

  /* --- Compute tessellation pipeline ------------------------------------ */
  WGPUShaderModule compute_sm
    = wgpu_create_shader_module(device, compute_tess_shader_wgsl);

  state.compute_pipeline = wgpuDeviceCreateComputePipeline(
    device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Terrain Compute Pipeline"),
      .layout  = state.pipe_layouts.compute,
      .compute = {
        .module     = compute_sm,
        .entryPoint = STRVIEW("cs_tessellate"),
      },
    });

  wgpuShaderModuleRelease(compute_sm);
}

/* -------------------------------------------------------------------------- *
 * Draw helpers
 * -------------------------------------------------------------------------- */

static void draw_sky(WGPURenderPassEncoder pass)
{
  if (!state.sky.loaded) {
    return;
  }
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.sky.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  if (state.sky.index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.sky.index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  }
  gltf_model_t* m = &state.sky.model;
  for (uint32_t n = 0; n < m->linear_node_count; n++) {
    gltf_node_t* node = m->linear_nodes[n];
    if (!node->mesh) {
      continue;
    }
    for (uint32_t p = 0; p < node->mesh->primitive_count; p++) {
      gltf_primitive_t* prim = &node->mesh->primitives[p];
      if (prim->has_indices && prim->index_count > 0) {
        wgpuRenderPassEncoderDrawIndexed(pass, prim->index_count, 1,
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
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Terrain Tessellation", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    igCheckbox("Tessellation", &state.settings.tessellation);

    imgui_overlay_input_float(
      "Displacement", &state.settings.displacement_factor, 0.5f, "%.2f");
    if (state.settings.displacement_factor < 0.0f) {
      state.settings.displacement_factor = 0.0f;
    }

    igCheckbox("Wireframe", &state.settings.wireframe);
  }

  if (igCollapsingHeaderBoolPtr("Pipeline statistics", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    uint32_t tri_ib = state.settings.tessellation ?
                        state.terrain.index_count :
                        state.terrain.coarse_index_count;
    imgui_overlay_text("VS invocations: %u", state.terrain.vertex_count);
    imgui_overlay_text("Triangles: %u", tri_ib / 3);
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
    camera_update_aspect_ratio(&state.camera,
                               (float)input_event->window_width
                                 / (float)input_event->window_height);
    return;
  }

  if (!imgui_overlay_want_capture_mouse()) {
    camera_on_input_event(&state.camera, input_event);
  }

  /* WASD / arrow key movement (flight-simulator style, like Vulkan version) */
  state.camera.keys.up
    = input_event->keys_down[KEY_W] || input_event->keys_down[KEY_UP];
  state.camera.keys.down
    = input_event->keys_down[KEY_S] || input_event->keys_down[KEY_DOWN];
  state.camera.keys.left
    = input_event->keys_down[KEY_A] || input_event->keys_down[KEY_LEFT];
  state.camera.keys.right
    = input_event->keys_down[KEY_D] || input_event->keys_down[KEY_RIGHT];
}

/* -------------------------------------------------------------------------- *
 * Lifecycle
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  s_wgpu_ctx = wgpu_context;

  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 4,
    .num_channels = 1,
    .num_lanes    = 1,
    .logger.func  = slog_func,
  });

  /* Camera */
  init_camera(wgpu_context);

  /* Load heightmap synchronously (needed for mesh generation) */
  int hm_w = 0, hm_h = 0;
  uint8_t* hm_pixels = load_heightmap(&hm_w, &hm_h);
  if (!hm_pixels) {
    return EXIT_FAILURE;
  }

  /* Generate terrain mesh from heightmap */
  generate_terrain_mesh(wgpu_context, hm_pixels, hm_w);

  /* Create heightmap GPU texture */
  create_heightmap_texture(wgpu_context, hm_pixels, hm_w, hm_h);
  image_free(hm_pixels);

  /* Load skysphere model */
  load_skysphere_model(wgpu_context);

  /* Create placeholder textures for async-loaded assets */
  create_placeholder_textures(wgpu_context);

  /* Start async texture fetching */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/terrain_texturearray_rgba.png",
    .callback = terrain_array_fetch_cb,
    .buffer   = {.ptr = state.fetch_buffer, .size = FETCH_BUFFER_SIZE},
  });
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/skysphere_rgba.png",
    .callback = skysphere_fetch_cb,
    .buffer   = {.ptr = state.fetch_buffer, .size = FETCH_BUFFER_SIZE},
  });

  /* Uniform buffers */
  create_uniform_buffers(wgpu_context);

  /* Bind group layouts / pipeline layouts */
  create_bind_group_layouts(wgpu_context);
  create_pipeline_layouts(wgpu_context);

  /* Bind groups (with placeholder textures initially) */
  create_bind_groups(wgpu_context);

  /* Render pipelines */
  create_pipelines(wgpu_context);

  /* ImGui */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async file loads */
  sfetch_dowork();

  /* Frame timing */
  uint64_t now = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = now;
  }
  float dt              = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Camera */
  camera_update(&state.camera, dt);

  /* Uniforms */
  update_uniform_buffers(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, dt);
  render_gui(wgpu_context);

  /* --- Compute pass: displace terrain vertices on GPU ------------------- */
  {
    WGPUCommandEncoder comp_enc
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPUComputePassEncoder comp_pass
      = wgpuCommandEncoderBeginComputePass(comp_enc, NULL);
    wgpuComputePassEncoderSetPipeline(comp_pass, state.compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(comp_pass, 0, state.bind_groups.compute,
                                       0, NULL);
    /* 256*256 = 65536 vertices, workgroup_size(64) → 1024 workgroups */
    const uint32_t workgroups = (state.terrain.vertex_count + 63) / 64;
    wgpuComputePassEncoderDispatchWorkgroups(comp_pass, workgroups, 1, 1);
    wgpuComputePassEncoderEnd(comp_pass);
    WGPUCommandBuffer comp_cmd = wgpuCommandEncoderFinish(comp_enc, NULL);
    wgpuQueueSubmit(wgpu_context->queue, 1, &comp_cmd);
    wgpuComputePassEncoderRelease(comp_pass);
    wgpuCommandBufferRelease(comp_cmd);
    wgpuCommandEncoderRelease(comp_enc);
  }

  /* --- Render ----------------------------------------------------------- */
  state.color_att.view = wgpu_context->swapchain_view;
  state.depth_att.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(enc, &state.render_pass_desc);

  uint32_t w = (uint32_t)wgpu_context->width;
  uint32_t h = (uint32_t)wgpu_context->height;
  wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);

  /* (1) Skysphere */
  wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.skysphere);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_groups.skysphere, 0,
                                    NULL);
  draw_sky(pass);

  /* (2) Terrain */
  if (state.settings.wireframe) {
    wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.wireframe);
  }
  else {
    wgpuRenderPassEncoderSetPipeline(pass, state.pipelines.terrain);
  }
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_groups.terrain, 0,
                                    NULL);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.terrain.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);

  /* Select index buffer: tessellation ON = fine (256×256), OFF = coarse (64×64)
   */
  if (state.settings.wireframe) {
    WGPUBuffer wib = state.settings.tessellation ?
                       state.terrain.wire_index_buffer :
                       state.terrain.coarse_wire_index_buffer;
    uint32_t wic   = state.settings.tessellation ?
                       state.terrain.wire_index_count :
                       state.terrain.coarse_wire_index_count;
    wgpuRenderPassEncoderSetIndexBuffer(pass, wib, WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(pass, wic, 1, 0, 0, 0);
  }
  else {
    WGPUBuffer ib = state.settings.tessellation ?
                      state.terrain.index_buffer :
                      state.terrain.coarse_index_buffer;
    uint32_t ic   = state.settings.tessellation ?
                      state.terrain.index_count :
                      state.terrain.coarse_index_count;
    wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(pass, ic, 1, 0, 0, 0);
  }

  wgpuRenderPassEncoderEnd(pass);

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd);

  wgpuRenderPassEncoderRelease(pass);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  /* ImGui render */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Terrain buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain.input_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain.wire_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain.coarse_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain.coarse_wire_index_buffer)

  /* Sky model buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.sky.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.sky.index_buffer)
  if (state.sky.loaded) {
    gltf_model_destroy(&state.sky.model);
  }

  /* Textures */
  WGPU_RELEASE_RESOURCE(TextureView, state.tex.heightmap_view)
  WGPU_RELEASE_RESOURCE(Texture, state.tex.heightmap)
  WGPU_RELEASE_RESOURCE(Sampler, state.tex.heightmap_sampler)
  WGPU_RELEASE_RESOURCE(TextureView, state.tex.terrain_array_view)
  WGPU_RELEASE_RESOURCE(Texture, state.tex.terrain_array)
  WGPU_RELEASE_RESOURCE(Sampler, state.tex.terrain_array_sampler)
  WGPU_RELEASE_RESOURCE(TextureView, state.tex.skysphere_view)
  WGPU_RELEASE_RESOURCE(Texture, state.tex.skysphere)
  WGPU_RELEASE_RESOURCE(Sampler, state.tex.skysphere_sampler)

  /* Uniform buffers */
  wgpu_destroy_buffer(&state.uniform_bufs.terrain);
  wgpu_destroy_buffer(&state.uniform_bufs.sky);
  wgpu_destroy_buffer(&state.uniform_bufs.compute);

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.terrain)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.skysphere)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.compute)

  /* Layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layouts.terrain)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layouts.skysphere)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bg_layouts.compute)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipe_layouts.terrain)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipe_layouts.skysphere)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipe_layouts.compute)

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.terrain)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.wireframe)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.skysphere)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute_pipeline)
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Terrain Tessellation",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader Code
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* terrain_shader_wgsl = CODE(
  /* ---- Uniforms ---- */
  struct TerrainUniforms {
    projection : mat4x4f,
    modelview  : mat4x4f,
    light_pos  : vec4f,
    disp_factor: f32,
  }

  @group(0) @binding(0) var<uniform> ubo : TerrainUniforms;

  /* ---- Textures (fragment only) ---- */
  @group(0) @binding(1) var heightmap_sampler : sampler;
  @group(0) @binding(2) var heightmap_texture : texture_2d<f32>;
  @group(0) @binding(3) var layer_sampler     : sampler;
  @group(0) @binding(4) var layer_texture     : texture_2d_array<f32>;

  /* ---- Vertex I/O ---- */
  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
  }

  struct VertexOutput {
    @builtin(position) clip_pos  : vec4f,
    @location(0)       normal    : vec3f,
    @location(1)       uv        : vec2f,
    @location(2)       view_vec  : vec3f,
    @location(3)       light_vec : vec3f,
    @location(4)       eye_pos   : vec3f,
    @location(5)       world_pos : vec3f,
  }

  /* ---- Terrain vertex shader ---- */
  @vertex
  fn vs_terrain(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;

    /* Vertices are already displaced by the compute shader. */
    var pos = vec4f(input.position, 1.0);

    output.clip_pos  = ubo.projection * ubo.modelview * pos;
    output.normal    = input.normal;
    output.uv        = input.uv;
    output.view_vec  = -pos.xyz;
    output.light_vec = normalize(ubo.light_pos.xyz + output.view_vec);
    output.world_pos = pos.xyz;
    output.eye_pos   = (ubo.modelview * pos).xyz;

    return output;
  }

  /* ---- Height-based multi-layer sampling ---- */
  fn sample_terrain_layer(uv : vec2f) -> vec3f {
    let layers = array<vec2f, 6>(
      vec2f(-10.0, 10.0),
      vec2f(  5.0, 45.0),
      vec2f( 45.0, 80.0),
      vec2f( 75.0, 100.0),
      vec2f( 95.0, 140.0),
      vec2f(140.0, 190.0)
    );

    var color = vec3f(0.0);
    let height = textureSample(heightmap_texture, heightmap_sampler, uv).r * 255.0;

    for (var i = 0u; i < 6u; i = i + 1u) {
      let range_val = layers[i].y - layers[i].x;
      let weight    = max(0.0, (range_val - abs(height - layers[i].y)) / range_val);
      color += weight * textureSample(layer_texture, layer_sampler, uv * 16.0, i).rgb;
    }

    return color;
  }

  /* ---- Exponential fog ---- */
  fn fog(density : f32, frag_coord : vec4f) -> f32 {
    let LOG2 : f32 = -1.442695;
    let dist = frag_coord.z / frag_coord.w * 0.1;
    let d    = density * dist;
    return 1.0 - clamp(exp2(d * d * LOG2), 0.0, 1.0);
  }

  /* ---- Terrain fragment shader ---- */
  @fragment
  fn fs_terrain(input : VertexOutput) -> @location(0) vec4f {
    let N = normalize(input.normal);
    let L = normalize(input.light_vec);

    let ambient = vec3f(0.5);
    let diffuse = max(dot(N, L), 0.0) * vec3f(1.0);

    var color = vec4f((ambient + diffuse) * sample_terrain_layer(input.uv), 1.0);

    let fog_color = vec4f(0.47, 0.5, 0.67, 0.0);
    return mix(color, fog_color, fog(0.25, input.clip_pos));
  }

  /* ---- Wireframe fragment shader (solid white) ---- */
  @fragment
  fn fs_wireframe(input : VertexOutput) -> @location(0) vec4f {
    return vec4f(1.0, 1.0, 1.0, 1.0);
  }
);

static const char* skysphere_shader_wgsl = CODE(
  struct SkyUniforms {
    mvp : mat4x4f,
  }

  @group(0) @binding(0) var<uniform> ubo         : SkyUniforms;
  @group(0) @binding(1) var sky_sampler           : sampler;
  @group(0) @binding(2) var sky_texture           : texture_2d<f32>;

  struct SkyVSOut {
    @builtin(position) clip_pos : vec4f,
    @location(0)       uv       : vec2f,
  }

  @vertex
  fn vs_sky(
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
  ) -> SkyVSOut {
    var out : SkyVSOut;
    out.clip_pos = ubo.mvp * vec4f(position, 1.0);
    out.uv       = uv;
    return out;
  }

  @fragment
  fn fs_sky(input : SkyVSOut) -> @location(0) vec4f {
    let color = textureSample(sky_texture, sky_sampler, input.uv);
    return vec4f(color.rgb, 1.0);
  }
);

/* --- Compute tessellation shader --------------------------------------- */
static const char* compute_tess_shader_wgsl = CODE(
  /* Vertex layout matches terrain_vertex_t: 8 × f32 = 32 bytes */
  struct TerrainVertex {
    px: f32, py: f32, pz: f32,
    nx: f32, ny: f32, nz: f32,
    u: f32,  v: f32,
  }

  struct ComputeParams {
    disp_factor: f32,
  }

  @group(0) @binding(0) var<storage, read>       input_verts  : array<TerrainVertex>;
  @group(0) @binding(1) var<storage, read_write>  output_verts : array<TerrainVertex>;
  @group(0) @binding(2) var<uniform>              params       : ComputeParams;
  @group(0) @binding(3) var heightmap_tex  : texture_2d<f32>;
  @group(0) @binding(4) var heightmap_samp : sampler;

  @compute @workgroup_size(64)
  fn cs_tessellate(@builtin(global_invocation_id) gid : vec3u) {
    let idx = gid.x;
    let total = arrayLength(&input_verts);
    if (idx >= total) { return; }

    var vert = input_verts[idx];

    /* Sample heightmap at vertex UV */
    let uv = vec2f(vert.u, vert.v);
    let height = textureSampleLevel(heightmap_tex, heightmap_samp, uv, 0.0).r;

    /* Displace Y upward (WebGPU Y-up convention) */
    vert.py = height * params.disp_factor;

    /* Compute normal via Sobel filter on heightmap */
    let dims = vec2f(textureDimensions(heightmap_tex, 0));
    let texel = 1.0 / dims;

    let h00 = textureSampleLevel(heightmap_tex, heightmap_samp,
                                 uv + vec2f(-texel.x, -texel.y), 0.0).r;
    let h10 = textureSampleLevel(heightmap_tex, heightmap_samp,
                                 uv + vec2f(0.0, -texel.y), 0.0).r;
    let h20 = textureSampleLevel(heightmap_tex, heightmap_samp,
                                 uv + vec2f(texel.x, -texel.y), 0.0).r;
    let h01 = textureSampleLevel(heightmap_tex, heightmap_samp,
                                 uv + vec2f(-texel.x, 0.0), 0.0).r;
    let h21 = textureSampleLevel(heightmap_tex, heightmap_samp,
                                 uv + vec2f(texel.x, 0.0), 0.0).r;
    let h02 = textureSampleLevel(heightmap_tex, heightmap_samp,
                                 uv + vec2f(-texel.x, texel.y), 0.0).r;
    let h12 = textureSampleLevel(heightmap_tex, heightmap_samp,
                                 uv + vec2f(0.0, texel.y), 0.0).r;
    let h22 = textureSampleLevel(heightmap_tex, heightmap_samp,
                                 uv + vec2f(texel.x, texel.y), 0.0).r;

    /* Sobel gradients */
    let gx = (h00 - h20) + 2.0 * (h01 - h21) + (h02 - h22);
    let gz = (h00 + 2.0 * h10 + h20) - (h02 + 2.0 * h12 + h22);
    let d  = max(0.0, 1.0 - gx * gx - gz * gz);
    let n_y = 0.25 * sqrt(d);

    /* For positive Y displacement, normal = (-dh/dx, up, -dh/dz) */
    let n = normalize(vec3f(-gx * 2.0, n_y, -gz * 2.0));
    vert.nx = n.x;
    vert.ny = n.y;
    vert.nz = n.z;

    output_verts[idx] = vert;
  }
);
// clang-format on
