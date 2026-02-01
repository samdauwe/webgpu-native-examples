#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <stdbool.h>

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
 * WebGPU Example - Volume Rendering - Texture 3D
 *
 * This example shows how to render volumes with WebGPU using a 3D texture. It
 * demonstrates simple direct volume rendering for photometric content through
 * ray marching in a fragment shader, where a full-screen triangle determines
 * the color from ray start and step size values as set in the vertex shader.
 * This implementation employs data from the BrainWeb Simulated Brain Database,
 * with decompression streams, to save disk space and network traffic.
 *
 * The original raw data is generated using the BrainWeb Simulated Brain
 * Database:
 * https://brainweb.bic.mni.mcgill.ca/brainweb/
 * before processingin a custom Python script:
 * https://github.com/webgpu/webgpu-samples/tree/main/public/assets/img/volume/t1_icbm_normal_1mm_pn0_rf0.py).
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/volumeRenderingTexture3D
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* volume_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Volume Rendering - Texture 3D example
 * -------------------------------------------------------------------------- */

#define VOLUME_TEXTURE_WIDTH (180u)
#define VOLUME_TEXTURE_HEIGHT (216u)
#define VOLUME_TEXTURE_DEPTH (180u)
#define VOLUME_TEXTURE_SIZE                                                    \
  (VOLUME_TEXTURE_WIDTH * VOLUME_TEXTURE_HEIGHT * VOLUME_TEXTURE_DEPTH)

/* State struct */
static struct {
  wgpu_buffer_t uniform_buffer_vs;
  struct {
    mat4 inverse_model_view_projection_matrix;
  } ubo_vs;
  struct {
    mat4 projection;
    mat4 view;
  } view_matrices;
  struct {
    struct {
      WGPUTexture texture;
      WGPUTextureView view;
      WGPUSampler sampler;
      uint8_t data[VOLUME_TEXTURE_SIZE];
      bool is_dirty;
    } volume;
    struct {
      WGPUTexture texture;
      WGPUTextureView framebuffer;
    } multisampled;
  } textures;
  WGPURenderPipeline render_pipeline;
  WGPUBindGroup uniform_bind_group;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  struct {
    bool rotate_camera;
    float near;
    float far;
    WGPUTextureFormat texture_format;
  } params;
  float last_frame_ms;
  float rotation;
  const uint32_t sample_count;
  uint64_t last_imgui_frame_time;
  bool initialized;
} state = {
  .ubo_vs = {
    .inverse_model_view_projection_matrix = GLM_MAT4_ZERO_INIT,
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Discard,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
  },
  .params = {
    .rotate_camera  = true,
    .near           = 4.3f,
    .far            = 4.4f,
    .texture_format = WGPUTextureFormat_R8Unorm,
  },
  .sample_count = 4,
};

static void
get_inverse_model_view_projection_matrix(wgpu_context_t* wgpu_context,
                                         float delta_time, mat4* dest)
{
  /* View matrix */
  glm_mat4_identity(state.view_matrices.view);
  glm_translate(state.view_matrices.view, (vec3){0.0f, 0.0f, -4.0f});
  if (state.params.rotate_camera) {
    state.rotation += delta_time;
  }
  glm_rotate(state.view_matrices.view, 1.0f,
             (vec3){sin(state.rotation), cos(state.rotation), 0.0f});

  /* Projection matrix */
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;
  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, state.params.near, state.params.far,
                  state.view_matrices.projection);
  glm_mat4_mul(state.view_matrices.projection, state.view_matrices.view, *dest);
  glm_mat4_inv(*dest, *dest);
}

static void
update_inverse_model_view_projection_matrix(wgpu_context_t* wgpu_context)
{
  const float now        = stm_ms(stm_now());
  const float delta_time = (now - state.last_frame_ms) / 1000.0f;
  state.last_frame_ms    = now;

  get_inverse_model_view_projection_matrix(
    wgpu_context, delta_time,
    &state.ubo_vs.inverse_model_view_projection_matrix);
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update the inverse model-view-projection matrix */
  update_inverse_model_view_projection_matrix(wgpu_context);

  /* Map uniform buffer and update it */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_vs.buffer, 0,
                       &state.ubo_vs.inverse_model_view_projection_matrix,
                       state.uniform_buffer_vs.size);
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Create vertex shader uniform buffer block */
  state.uniform_buffer_vs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex shader - Uniform buffer block",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(state.ubo_vs), /* 4x4 matrix */
                  });

  /* Set uniform buffer block data */
  state.last_frame_ms = stm_ms(stm_now());
  update_uniform_buffers(wgpu_context);
}

static void init_multisampled_framebuffer(wgpu_context_t* wgpu_context)
{
  /* Create the multi-sampled texture */
  WGPUTextureDescriptor multisampled_frame_desc = {
    .label         = STRVIEW("Multi-sampled - Texture"),
    .size          = (WGPUExtent3D){
      .width               = wgpu_context->width,
      .height              = wgpu_context->height,
      .depthOrArrayLayers  = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = state.sample_count,
    .dimension     = WGPUTextureDimension_2D,
    .format        = wgpu_context->render_format,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.textures.multisampled.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &multisampled_frame_desc);
  ASSERT(state.textures.multisampled.texture != NULL);

  /* Create the multi-sampled texture view */
  state.textures.multisampled.framebuffer
    = wgpuTextureCreateView(state.textures.multisampled.texture,
                            &(WGPUTextureViewDescriptor){
                              .label  = STRVIEW("Multi-sampled - Texture view"),
                              .format = multisampled_frame_desc.format,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                            });
  ASSERT(state.textures.multisampled.framebuffer != NULL);
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
  ASSERT(response->data.ptr != NULL)
  ASSERT(response->data.size == VOLUME_TEXTURE_SIZE)
  state.textures.volume.is_dirty = true;
}

static void update_volume_texture_date(wgpu_context_t* wgpu_context)
{
  const uint32_t width           = VOLUME_TEXTURE_WIDTH;
  const uint32_t height          = VOLUME_TEXTURE_HEIGHT;
  const uint32_t depth           = VOLUME_TEXTURE_DEPTH;
  const uint32_t block_length    = 1;
  const uint32_t bytes_per_block = 1;
  const uint32_t blocks_wide     = ceil(width / (float)block_length);
  const uint32_t blocks_high     = ceil(height / (float)block_length);
  const uint32_t bytes_per_row   = blocks_wide * bytes_per_block;

  /* Copy volume data to texture */
  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo) {
                          .texture = state.textures.volume.texture,
                          .mipLevel = 0,
                          .origin = (WGPUOrigin3D) {
                            .x = 0,
                            .y = 0,
                            .z = 0,
                          },
                          .aspect = WGPUTextureAspect_All,
                        },
                        state.textures.volume.data, ARRAY_SIZE(state.textures.volume.data),
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = bytes_per_row,
                          .rowsPerImage = blocks_high,
                        },
                        &(WGPUExtent3D){
                          .width              = width,
                          .height             = height,
                          .depthOrArrayLayers = depth,
                        });
}

static void init_volume_texture(wgpu_context_t* wgpu_context)
{
  const uint32_t width      = VOLUME_TEXTURE_WIDTH;
  const uint32_t height     = VOLUME_TEXTURE_HEIGHT;
  const uint32_t depth      = VOLUME_TEXTURE_DEPTH;
  const uint32_t mip_levels = 1;

  /* Read volume data from file */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/volume/"
                "t1_icbm_normal_1mm_pn0_rf0_180x216x180_uint8_1x1.bin",
    .callback = fetch_callback,
    .buffer   = SFETCH_RANGE(state.textures.volume.data),
  });

  /* Create the volume texture */
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Volume - Texture"),
    .size          =   (WGPUExtent3D) {
      .width              = width,
      .height             = height,
      .depthOrArrayLayers = depth,
    },
    .mipLevelCount = mip_levels,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_3D,
    .format        = state.params.texture_format,
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
  };
  state.textures.volume.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(state.textures.volume.texture != NULL);

  /* Create a sampler with linear filtering for smooth interpolation. */
  state.textures.volume.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label        = STRVIEW("Volume - Texture sampler"),
                            .addressModeU = WGPUAddressMode_ClampToEdge,
                            .addressModeV = WGPUAddressMode_ClampToEdge,
                            .addressModeW = WGPUAddressMode_ClampToEdge,
                            .magFilter    = WGPUFilterMode_Linear,
                            .minFilter    = WGPUFilterMode_Linear,
                            .mipmapFilter = WGPUMipmapFilterMode_Linear,
                            .maxAnisotropy = 16,
                          });
  ASSERT(state.textures.volume.sampler != NULL);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = STRVIEW("Volume - Texture view"),
    .dimension       = WGPUTextureViewDimension_3D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  state.textures.volume.view
    = wgpuTextureCreateView(state.textures.volume.texture, &texture_view_dec);
  ASSERT(state.textures.volume.view != NULL);
}

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  /* Bind Group */
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Vertex shader uniform buffer */
      .binding = 0,
      .buffer  = state.uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1: Fragment shader image sampler */
      .binding = 1,
      .sampler = state.textures.volume.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      /* Binding 2 : Fragment shader texture view */
      .binding     = 2,
      .textureView = state.textures.volume.view,
    },
  };

  state.uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Uniform - Bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(state.render_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(state.uniform_bind_group != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule volume_shader_module
    = wgpu_create_shader_module(wgpu_context->device, volume_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Volume texture 3d - Render pipeline"),
    .vertex = {
      .module      = volume_shader_module,
      .entryPoint  = STRVIEW("vertex_main"),
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("fragment_main"),
      .module      = volume_shader_module,
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
    .multisample = {
       .count = state.sample_count,
       .mask  = 0xffffffff
    },
  };

  state.render_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.render_pipeline != NULL);

  wgpuShaderModuleRelease(volume_shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
    });
    init_uniform_buffer(wgpu_context);
    init_multisampled_framebuffer(wgpu_context);
    init_volume_texture(wgpu_context);
    init_pipeline(wgpu_context);
    init_bind_group(wgpu_context);
    imgui_overlay_init(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

/* Render GUI */
static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Set window position closer to upper left corner */
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});

  igBegin("Volume Rendering Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Rotate camera checkbox */
  igCheckbox("rotateCamera", &state.params.rotate_camera);

  /* Near/far sliders */
  imgui_overlay_slider_float("near", &state.params.near, 2.0f, 7.0f, "%.1f");
  imgui_overlay_slider_float("far", &state.params.far, 2.0f, 7.0f, "%.1f");

  igEnd();
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    /* Recreate multisampled framebuffer on resize */
    WGPU_RELEASE_RESOURCE(Texture, state.textures.multisampled.texture)
    WGPU_RELEASE_RESOURCE(TextureView, state.textures.multisampled.framebuffer)
    init_multisampled_framebuffer(wgpu_context);
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Update texture when pixel data loaded */
  if (state.textures.volume.is_dirty) {
    update_volume_texture_date(wgpu_context);
    state.textures.volume.is_dirty = false;
  }

  /* Calculate GUI frame time */
  const uint64_t now = stm_now();
  const float dt_sec
    = (float)stm_sec(stm_diff(now, state.last_imgui_frame_time));
  state.last_imgui_frame_time = now;

  /* Prepare imgui frame */
  imgui_overlay_new_frame(wgpu_context, dt_sec);

  /* Render GUI */
  render_gui(wgpu_context);

  /* Update uniform data */
  update_uniform_buffers(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = state.textures.multisampled.framebuffer;
  state.color_attachment.resolveTarget = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.uniform_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderDraw(rpass_enc, 3, 1, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Render imgui overlay */
  imgui_overlay_render(wgpu_context);

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
  imgui_overlay_shutdown();

  WGPU_RELEASE_RESOURCE(Texture, state.textures.volume.texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.volume.view)
  WGPU_RELEASE_RESOURCE(Sampler, state.textures.volume.sampler)
  WGPU_RELEASE_RESOURCE(Texture, state.textures.multisampled.texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.multisampled.framebuffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.uniform_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Volume Rendering - Texture 3D",
    .init_cb        = init,
    .frame_cb       = frame,
    .input_event_cb = input_event_cb,
    .shutdown_cb    = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* volume_shader_wgsl = CODE(
  struct Uniforms {
    inverseModelViewProjectionMatrix : mat4x4f,
  }

  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  @group(0) @binding(1) var mySampler: sampler;
  @group(0) @binding(2) var myTexture: texture_3d<f32>;

  struct VertexOutput {
    @builtin(position) Position : vec4f,
    @location(0) near : vec3f,
    @location(1) step : vec3f,
  }

  const NumSteps = 64u;

  @vertex
  fn vertex_main(
    @builtin(vertex_index) VertexIndex : u32
  ) -> VertexOutput {
    var pos = array<vec2f, 3>(
      vec2(-1.0, 3.0),
      vec2(-1.0, -1.0),
      vec2(3.0, -1.0)
    );
    var xy = pos[VertexIndex];
    var near = uniforms.inverseModelViewProjectionMatrix * vec4f(xy, 0.0, 1);
    var far = uniforms.inverseModelViewProjectionMatrix * vec4f(xy, 1, 1);
    near /= near.w;
    far /= far.w;
    return VertexOutput(
      vec4f(xy, 0.0, 1.0),
      near.xyz,
      (far.xyz - near.xyz) / f32(NumSteps)
    );
  }

  @fragment
  fn fragment_main(
    @location(0) near: vec3f,
    @location(1) step: vec3f
  ) -> @location(0) vec4f {
    var rayPos = near;
    var result = 0.0;
    for (var i = 0u; i < NumSteps; i++) {
      let texCoord = (rayPos.xyz + 1.0) * 0.5;
      let sample =
        textureSample(myTexture, mySampler, texCoord).r * 4.0 / f32(NumSteps);
      let intersects =
        all(rayPos.xyz < vec3f(1.0)) && all(rayPos.xyz > vec3f(-1.0));
      result += select(0.0, (1.0 - result) * sample, intersects && result < 1.0);
      rayPos += step;
    }
    return vec4f(vec3f(result), 1.0);
  }
);
// clang-format on
