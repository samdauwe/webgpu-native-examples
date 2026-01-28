#include "common_shaders.h"
#include "meshes.h"
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

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
 * WebGPU Example - Timestamp Query
 *
 * This example demonstrates using Timestamp Queries to measure the duration of
 * a render pass.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/timestampQuery
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* black_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Performance Counter
 * A minimalistic perf timer class that computes mean + stddev online
 * -------------------------------------------------------------------------- */

typedef struct perf_counter_t {
  uint32_t sample_count;
  double accumulated;
  double accumulated_sq;
} perf_counter_t;

static void perf_counter_init(perf_counter_t* counter)
{
  counter->sample_count   = 0;
  counter->accumulated    = 0.0;
  counter->accumulated_sq = 0.0;
}

static void perf_counter_add_sample(perf_counter_t* counter, double value)
{
  counter->sample_count += 1;
  counter->accumulated += value;
  counter->accumulated_sq += value * value;
}

static double perf_counter_get_average(const perf_counter_t* counter)
{
  if (counter->sample_count == 0) {
    return 0.0;
  }
  return counter->accumulated / (double)counter->sample_count;
}

static double perf_counter_get_stddev(const perf_counter_t* counter)
{
  if (counter->sample_count == 0) {
    return 0.0;
  }
  const double avg = perf_counter_get_average(counter);
  const double variance
    = (counter->accumulated_sq / (double)counter->sample_count) - (avg * avg);
  return sqrt(fmax(0.0, variance));
}

/* -------------------------------------------------------------------------- *
 * Timestamp Query example
 * -------------------------------------------------------------------------- */

/* State struct */
static struct {
  /* Mesh data */
  cube_mesh_t cube_mesh;
  wgpu_buffer_t vertices;
  /* Uniform buffer */
  WGPUBuffer uniform_buffer;
  WGPUBindGroup uniform_bind_group;
  /* View matrices */
  struct {
    mat4 projection;
    mat4 view;
    mat4 model_view_projection;
  } view_matrices;
  /* Pipeline */
  WGPURenderPipeline pipeline;
  /* Depth texture */
  wgpu_texture_t depth_texture;
  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* Timestamp query support */
  bool has_timestamp_query;
  WGPUQuerySet query_set;
  WGPUBuffer timestamp_buffer;
  WGPUBuffer timestamp_map_buffer;
  /* Performance statistics */
  perf_counter_t render_pass_duration_counter;
  double last_average_ms;
  double last_stddev_ms;
  /* Timing for ImGui */
  uint64_t last_frame_time;
  /* Initialization flag */
  bool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.5f, 0.5f, 0.5f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Undefined,
    .stencilStoreOp    = WGPUStoreOp_Undefined,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .view_matrices = {
    .projection            = GLM_MAT4_IDENTITY_INIT,
    .view                  = GLM_MAT4_IDENTITY_INIT,
    .model_view_projection = GLM_MAT4_IDENTITY_INIT,
  },
};

/* Prepare the cube geometry */
static void init_cube_mesh(void)
{
  cube_mesh_init(&state.cube_mesh);
}

/* Create a vertex buffer from the cube data */
static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Vertices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(state.cube_mesh.vertex_array),
                    .initial.data = state.cube_mesh.vertex_array,
                  });
}

/* Initialize depth texture */
static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  /* Release previous depth texture if exists */
  wgpu_destroy_texture(&state.depth_texture);

  /* Create the depth texture */
  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->width,
    .height             = wgpu_context->height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Depth - Texture"),
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth24Plus,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.depth_texture.handle
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(state.depth_texture.handle != NULL);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_desc = {
    .label           = STRVIEW("Depth - Texture view"),
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  state.depth_texture.view
    = wgpuTextureCreateView(state.depth_texture.handle, &texture_view_desc);
  ASSERT(state.depth_texture.view != NULL);
}

/* Initialize view matrices */
static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  state.view_matrices.projection);
}

/* Update transformation matrix */
static void update_transformation_matrix(void)
{
  const float now     = stm_sec(stm_now());
  const float sin_now = sin(now);
  const float cos_now = cos(now);

  /* View matrix */
  glm_mat4_identity(state.view_matrices.view);
  glm_translate(state.view_matrices.view, (vec3){0.0f, 0.0f, -4.0f});
  glm_rotate(state.view_matrices.view, 1.0f, (vec3){sin_now, cos_now, 0.0f});

  /* Model view projection matrix */
  glm_mat4_mul(state.view_matrices.projection, state.view_matrices.view,
               state.view_matrices.model_view_projection);
}

/* Initialize uniform buffer */
static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  init_view_matrices(wgpu_context);

  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Uniform buffer"),
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(mat4), /* 4x4 matrix */
    });
  ASSERT(state.uniform_buffer != NULL);
}

/* Update uniform buffers */
static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  update_transformation_matrix();

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       &state.view_matrices.model_view_projection,
                       sizeof(mat4));
}

/* Initialize pipeline */
static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, basic_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, black_fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(cube, state.cube_mesh.vertex_size,
                            /* Attribute location 0: Position */
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                                               state.cube_mesh.position_offset),
                            /* Attribute location 1: UV */
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2,
                                               state.cube_mesh.uv_offset))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Timestamp query - Render pipeline"),
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &cube_vertex_buffer_layout,
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
      .frontFace = WGPUFrontFace_CCW,
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
      .count = 1,
      .mask  = 0xffffffff,
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

/* Initialize bind group */
static void init_bind_group(wgpu_context_t* wgpu_context)
{
  state.uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Uniform - Bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline, 0),
      .entryCount = 1,
      .entries    = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniform_buffer,
        .offset  = 0,
        .size    = sizeof(mat4),
      },
    });
  ASSERT(state.uniform_bind_group != NULL);
}

/* Initialize timestamp query resources */
static void init_timestamp_query(wgpu_context_t* wgpu_context)
{
  /* Check for timestamp query support */
  state.has_timestamp_query
    = wgpuAdapterHasFeature(wgpu_context->adapter,
                            WGPUFeatureName_TimestampQuery)
      && wgpuDeviceHasFeature(wgpu_context->device,
                              WGPUFeatureName_TimestampQuery);

  if (!state.has_timestamp_query) {
    return;
  }

  /* Create query set for timestamps (2 queries: begin and end) */
  state.query_set = wgpuDeviceCreateQuerySet(
    wgpu_context->device, &(WGPUQuerySetDescriptor){
                            .label = STRVIEW("Timestamp query set"),
                            .type  = WGPUQueryType_Timestamp,
                            .count = 2,
                          });
  ASSERT(state.query_set != NULL);

  /* Create a buffer where to store the result of GPU queries */
  const uint32_t timestamp_byte_size = 8; /* timestamps are uint64 */
  state.timestamp_buffer             = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
                  .label = STRVIEW("Timestamp resolve buffer"),
                  .size  = 2 * timestamp_byte_size,
                  .usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_QueryResolve,
    });
  ASSERT(state.timestamp_buffer != NULL);

  /* Create a buffer to map the result back to the CPU */
  state.timestamp_map_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Timestamp map buffer"),
      .size  = 2 * timestamp_byte_size,
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
    });
  ASSERT(state.timestamp_map_buffer != NULL);

  /* Initialize perf counter */
  perf_counter_init(&state.render_pass_duration_counter);
}

/* Callback for processing timestamp results */
static void on_timestamp_buffer_mapped(WGPUMapAsyncStatus status,
                                       WGPUStringView message, void* userdata1,
                                       void* userdata2)
{
  UNUSED_VAR(message);
  UNUSED_VAR(userdata1);
  UNUSED_VAR(userdata2);

  if (status != WGPUMapAsyncStatus_Success) {
    return;
  }

  const int64_t* timestamps = (const int64_t*)wgpuBufferGetConstMappedRange(
    state.timestamp_map_buffer, 0, 2 * sizeof(int64_t));

  if (timestamps) {
    /* Subtract the begin time from the end time */
    int64_t elapsed_ns = timestamps[1] - timestamps[0];

    /* It's possible elapsed_ns is negative which means it's invalid
     * (see spec https://gpuweb.github.io/gpuweb/#timestamp) */
    if (elapsed_ns >= 0) {
      /* Convert from nanoseconds to milliseconds */
      double elapsed_ms = (double)elapsed_ns * 1e-6;
      perf_counter_add_sample(&state.render_pass_duration_counter, elapsed_ms);

      /* Update cached values for display */
      state.last_average_ms
        = perf_counter_get_average(&state.render_pass_duration_counter);
      state.last_stddev_ms
        = perf_counter_get_stddev(&state.render_pass_duration_counter);
    }
  }

  wgpuBufferUnmap(state.timestamp_map_buffer);
}

/* Try to initiate timestamp download */
static void try_initiate_timestamp_download(void)
{
  if (!state.has_timestamp_query) {
    return;
  }

  WGPUBufferMapState map_state
    = wgpuBufferGetMapState(state.timestamp_map_buffer);
  if (map_state != WGPUBufferMapState_Unmapped) {
    return;
  }

  wgpuBufferMapAsync(state.timestamp_map_buffer, WGPUMapMode_Read, 0,
                     2 * sizeof(int64_t),
                     (WGPUBufferMapCallbackInfo){
                       .mode     = WGPUCallbackMode_AllowProcessEvents,
                       .callback = on_timestamp_buffer_mapped,
                     });
}

/* Input event callback */
static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
    init_view_matrices(wgpu_context);
  }
}

/* Render GUI */
static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Set window position closer to upper left corner */
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});

  /* Set initial window size */
  igSetNextWindowSize((ImVec2){340.0f, 0.0f}, ImGuiCond_FirstUseEver);

  /* Build GUI */
  igBegin("Timestamp Query", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (state.has_timestamp_query) {
    igText("Render Pass duration: %.3f ms +/- %.3f ms", state.last_average_ms,
           state.last_stddev_ms);
  }
  else {
    igTextColored((ImVec4){1.0f, 0.5f, 0.0f, 1.0f},
                  "Timestamp queries are not supported");
  }

  igEnd();
}

/* Initialize */
static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  /* Initialize sokol_time */
  stm_setup();

  /* Initialize resources */
  init_cube_mesh();
  init_vertex_buffer(wgpu_context);
  init_depth_texture(wgpu_context);
  init_uniform_buffer(wgpu_context);
  init_pipeline(wgpu_context);
  init_bind_group(wgpu_context);
  init_timestamp_query(wgpu_context);

  /* Initialize ImGui overlay */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;

  return EXIT_SUCCESS;
}

/* Render frame */
static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Calculate delta time for ImGui */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);

  /* Render GUI */
  render_gui(wgpu_context);

  /* Update matrix data */
  update_uniform_buffers(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* Update render pass descriptor */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth_texture.view;

  /* Add timestamp writes if supported */
  WGPUPassTimestampWrites timestamp_writes = {0};
  if (state.has_timestamp_query) {
    timestamp_writes.querySet                    = state.query_set;
    timestamp_writes.beginningOfPassWriteIndex   = 0;
    timestamp_writes.endOfPassWriteIndex         = 1;
    state.render_pass_descriptor.timestampWrites = &timestamp_writes;
  }

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Begin render pass */
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.uniform_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(rpass_enc, state.cube_mesh.vertex_count, 1, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);

  /* Resolve timestamp queries and copy to mappable buffer */
  if (state.has_timestamp_query) {
    /* Resolve query results into resolve buffer */
    wgpuCommandEncoderResolveQuerySet(cmd_enc, state.query_set, 0, 2,
                                      state.timestamp_buffer, 0);

    /* Copy values to the mappable buffer if it's unmapped */
    WGPUBufferMapState map_state
      = wgpuBufferGetMapState(state.timestamp_map_buffer);
    if (map_state == WGPUBufferMapState_Unmapped) {
      wgpuCommandEncoderCopyBufferToBuffer(cmd_enc, state.timestamp_buffer, 0,
                                           state.timestamp_map_buffer, 0,
                                           2 * sizeof(int64_t));
    }
  }

  /* Get command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Try to download the timestamp */
  try_initiate_timestamp_download();

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* Shutdown */
static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Shutdown ImGui overlay */
  imgui_overlay_shutdown();

  /* Release timestamp query resources */
  if (state.has_timestamp_query) {
    WGPU_RELEASE_RESOURCE(QuerySet, state.query_set)
    WGPU_RELEASE_RESOURCE(Buffer, state.timestamp_buffer)
    WGPU_RELEASE_RESOURCE(Buffer, state.timestamp_map_buffer)
  }

  /* Release other resources */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.uniform_bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  wgpu_destroy_texture(&state.depth_texture);
}

int main(void)
{
  /* Request timestamp-query feature */
  WGPUFeatureName required_features[] = {WGPUFeatureName_TimestampQuery};

  wgpu_start(&(wgpu_desc_t){
    .title                  = "Timestamp Query",
    .init_cb                = init,
    .frame_cb               = frame,
    .shutdown_cb            = shutdown,
    .input_event_cb         = input_event_cb,
    .required_features      = required_features,
    .required_feature_count = 1,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* black_fragment_shader_wgsl = CODE(
  @fragment
  fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
);
// clang-format on
