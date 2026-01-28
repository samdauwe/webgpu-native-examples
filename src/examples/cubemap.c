#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

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
 * WebGPU Example - Cubemap
 *
 * This example shows how to render and sample from a cubemap texture.
 *
 * The example uses a simplified approach: instead of rendering a cube geometry,
 * it draws a fullscreen triangle and uses the inverse view-projection matrix
 * to compute the cubemap sampling direction in the fragment shader. This means:
 *   - No vertex buffers needed
 *   - No depth texture needed
 *   - Only 3 vertices drawn instead of 36
 *
 * Ref: https://github.com/webgpu/webgpu-samples/tree/main/sample/cubemap
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

static const char* sample_cubemap_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Cubemap example
 * -------------------------------------------------------------------------- */

/* Room for loading all cubemap faces in parallel */
#define NUM_FACES (6)
#define FACE_WIDTH (1024)
#define FACE_HEIGHT (1024)
#define FACE_NUM_BYTES (FACE_WIDTH * FACE_HEIGHT * 4)

/* State struct */
static struct {
  struct {
    WGPUBindGroup bind_group;
    WGPUBindGroupLayout bind_group_layout;
  } cubemap;
  wgpu_buffer_t uniform_buffer;
  struct {
    mat4 projection;
    mat4 model;
    mat4 view;
    mat4 tmp;
    mat4 view_direction_projection_inverse;
  } view_matrices;
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
    WGPUSampler sampler;
    WGPUBool is_dirty;
  } cubemap_texture;
  uint8_t cubemap_pixels[NUM_FACES][FACE_NUM_BYTES];
  int load_count;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1, 0.2, 0.3, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  }
};

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : Transform - visible to both vertex and fragment */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(mat4), /* 4x4 matrix */
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1 : Sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2 : Texture view */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_Cube,
        .multisampled  = false,
      },
      .storageTexture = {0},
    }
  };
  state.cubemap.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Cubemap - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.cubemap.bind_group_layout != NULL);

  /* Create the pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Cubemap - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.cubemap.bind_group_layout,
    });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_cubemap_texture(wgpu_context_t* wgpu_context)
{
  /* Texture */
  {
    WGPUTextureDescriptor tdesc = {
      .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
               | WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {FACE_WIDTH, FACE_HEIGHT, NUM_FACES},
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.cubemap_texture.handle
      = wgpuDeviceCreateTexture(wgpu_context->device, &tdesc);
  }

  /* Texture view */
  {
    WGPUTextureViewDescriptor view_desc = {
      .format          = WGPUTextureFormat_RGBA8Unorm,
      .dimension       = WGPUTextureViewDimension_Cube,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = NUM_FACES,
      .aspect          = WGPUTextureAspect_All,
      .usage           = WGPUTextureUsage_TextureBinding,
    };
    state.cubemap_texture.view
      = wgpuTextureCreateView(state.cubemap_texture.handle, &view_desc);
  }

  /* Texture sampler */
  {
    WGPUSamplerDescriptor sampler_desc = {
      .addressModeU  = WGPUAddressMode_ClampToEdge,
      .addressModeV  = WGPUAddressMode_ClampToEdge,
      .addressModeW  = WGPUAddressMode_ClampToEdge,
      .magFilter     = WGPUFilterMode_Linear,
      .minFilter     = WGPUFilterMode_Linear,
      .mipmapFilter  = WGPUMipmapFilterMode_Linear,
      .lodMinClamp   = 0,
      .lodMaxClamp   = 1,
      .compare       = WGPUCompareFunction_Undefined,
      .maxAnisotropy = 1,
    };
    state.cubemap_texture.sampler
      = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  }
}

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
  stbi_uc* decoded_pixels    = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (decoded_pixels) {
    assert(img_width == FACE_WIDTH);
    assert(img_height == FACE_HEIGHT);
    memcpy((void*)response->buffer.ptr, decoded_pixels, FACE_NUM_BYTES);
    stbi_image_free(decoded_pixels);
    ++state.load_count;
  }
}

/* Fetch the 6 separate images for negative/positive x, y, z axis of a cubemap
 * and upload it into a GPUTexture.
 * The order of the array layers is [+X, -X, +Y, -Y, +Z, -Z] */
static void fetch_cubemap_texture(void)
{
  static const char* cubemap_paths[NUM_FACES] = {
    "assets/textures/cubemaps/bridge2_px.jpg", /* +X Right  */
    "assets/textures/cubemaps/bridge2_nx.jpg", /* -X Left   */
    "assets/textures/cubemaps/bridge2_py.jpg", /* +Y Top    */
    "assets/textures/cubemaps/bridge2_ny.jpg", /* -Y Bottom */
    "assets/textures/cubemaps/bridge2_pz.jpg", /* +Z Back   */
    "assets/textures/cubemaps/bridge2_nz.jpg", /* -Z Front  */
  };
  for (int i = 0; i < NUM_FACES; i++) {
    sfetch_send(&(sfetch_request_t){
      .path     = cubemap_paths[i],
      .callback = fetch_callback,
      .buffer   = SFETCH_RANGE(state.cubemap_pixels[i]),
    });
  }
  state.cubemap_texture.is_dirty = 1;
}

static void update_texture_pixels(wgpu_context_t* wgpu_context)
{
  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create a host-visible staging buffers that contains the raw image data for
   * each face of the cubemap */
  WGPUBuffer staging_buffers[6] = {0};
  for (uint32_t face = 0; face < NUM_FACES; ++face) {
    WGPUBufferDescriptor staging_buffer_desc = {
      .usage            = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
      .size             = FACE_NUM_BYTES,
      .mappedAtCreation = true,
    };
    staging_buffers[face]
      = wgpuDeviceCreateBuffer(wgpu_context->device, &staging_buffer_desc);
    ASSERT(staging_buffers[face])
  }

  for (uint32_t face = 0; face < NUM_FACES; ++face) {
    /* Copy texture data into staging buffer */
    void* mapping
      = wgpuBufferGetMappedRange(staging_buffers[face], 0, FACE_NUM_BYTES);
    ASSERT(mapping)
    memcpy(mapping, state.cubemap_pixels[face], FACE_NUM_BYTES);
    wgpuBufferUnmap(staging_buffers[face]);

    /* Upload staging buffer to texture */
    wgpuCommandEncoderCopyBufferToTexture(cmd_encoder,
        /* Source */
        &(WGPUTexelCopyBufferInfo) {
          .buffer = staging_buffers[face],
          .layout = (WGPUTexelCopyBufferLayout) {
            .offset       = 0,
            .bytesPerRow  = FACE_WIDTH * 4,
            .rowsPerImage = FACE_HEIGHT,
          },
        },
        /* Destination */
        &(WGPUTexelCopyTextureInfo){
          .texture  = state.cubemap_texture.handle,
          .mipLevel = 0,
          .origin = (WGPUOrigin3D) {
              .x = 0,
              .y = 0,
              .z = face,
          },
          .aspect = WGPUTextureAspect_All,
        },
        /* Copy size */
        &(WGPUExtent3D){
          .width              = FACE_WIDTH,
          .height             = FACE_HEIGHT,
          .depthOrArrayLayers = 1,
        });
  }

  WGPUCommandBuffer command_buffer
    = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  /* Submit command buffer */
  ASSERT(command_buffer != NULL)
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  /* Release command buffer */
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

  /* Clean up staging resources */
  for (uint32_t face = 0; face < NUM_FACES; ++face) {
    WGPU_RELEASE_RESOURCE(Buffer, staging_buffers[face]);
  }

  state.cubemap_texture.is_dirty = 0;
}

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 3000.0f,
                  state.view_matrices.projection);

  /* Model matrix - identity */
  glm_mat4_identity(state.view_matrices.model);

  /* View matrix - identity */
  glm_mat4_identity(state.view_matrices.view);

  /* Other matrices */
  glm_mat4_identity(state.view_matrices.tmp);
  glm_mat4_identity(state.view_matrices.view_direction_projection_inverse);
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Setup the view matrices for the camera */
  init_view_matrices(wgpu_context);

  /* Uniform buffer - 4x4 matrix */
  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cubemap - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(mat4),
                  });
  ASSERT(state.uniform_buffer.buffer != NULL);
}

/* Compute camera movement:
 * It rotates around Y axis with a slight pitch movement. */
static void update_transformation_matrix(void)
{
  const float now = stm_ms(stm_now()) / 800.0f;

  /* Apply rotation to view matrix */
  glm_mat4_copy(state.view_matrices.view, state.view_matrices.tmp);
  glm_rotate(state.view_matrices.tmp, (GLM_PI / 10.f) * sin(now),
             (vec3){1.0f, 0.0f, 0.0f});
  glm_rotate(state.view_matrices.tmp, now * 0.2f, (vec3){0.f, 1.f, 0.f});

  /* Compute model-view-projection matrix */
  mat4 mvp;
  glm_mat4_mul(state.view_matrices.tmp, state.view_matrices.model, mvp);
  glm_mat4_mul(state.view_matrices.projection, mvp, mvp);

  /* Compute inverse for cubemap sampling direction */
  glm_mat4_inv(mvp, state.view_matrices.view_direction_projection_inverse);
}

static void update_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Update the transformation matrix */
  update_transformation_matrix();

  /* Upload to GPU */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer, 0,
                       &state.view_matrices.view_direction_projection_inverse,
                       sizeof(mat4));
}

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Transform */
      .binding = 0,
      .buffer  = state.uniform_buffer.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1 : Sampler */
      .binding = 1,
      .sampler = state.cubemap_texture.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
       /* Binding 2 : Texture view */
      .binding     = 2,
      .textureView = state.cubemap_texture.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Cubemap - Bind group"),
    .layout     = state.cubemap.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.cubemap.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.cubemap.bind_group != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule shader_module = wgpu_create_shader_module(
    wgpu_context->device, sample_cubemap_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* No vertex buffers needed - positions are generated in the vertex shader */
  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Cubemap - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = shader_module,
      .entryPoint  = STRVIEW("mainVS"),
      .bufferCount = 0,
      .buffers     = NULL,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("mainFS"),
      .module      = shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_None,
      .frontFace = WGPUFrontFace_CCW
    },
    /* No depth stencil - not needed for fullscreen cubemap sampling */
    .depthStencil = NULL,
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = NUM_FACES,
      .num_channels = 1,
      .num_lanes    = NUM_FACES,
      .logger.func  = slog_func,
    });
    init_pipeline_layout(wgpu_context);
    init_cubemap_texture(wgpu_context);
    fetch_cubemap_texture();
    init_uniform_buffer(wgpu_context);
    init_bind_group(wgpu_context);
    init_pipeline(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Update texture when pixel data loaded */
  if (state.cubemap_texture.is_dirty && state.load_count == NUM_FACES) {
    update_texture_pixels(wgpu_context);
  }

  /* Update matrix data */
  update_uniform_buffer(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Record render commands - draw 3 vertices for fullscreen triangle */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.cubemap.bind_group, 0,
                                    0);
  wgpuRenderPassEncoderDraw(rpass_enc, 3, 1, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present */
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

  WGPU_RELEASE_RESOURCE(Texture, state.cubemap_texture.handle);
  WGPU_RELEASE_RESOURCE(TextureView, state.cubemap_texture.view);
  WGPU_RELEASE_RESOURCE(Sampler, state.cubemap_texture.sampler);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.cubemap.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.cubemap.bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Cubemap",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* sample_cubemap_shader_wgsl = CODE(
  @group(0) @binding(0) var<uniform> viewDirectionProjectionInverse: mat4x4f;
  @group(0) @binding(1) var mySampler: sampler;
  @group(0) @binding(2) var myTexture: texture_cube<f32>;

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(1) direction: vec4f,
  };

  @vertex
  fn mainVS(
    @builtin(vertex_index) vertexIndex: u32
  ) -> VertexOutput {
    // A triangle large enough to cover all of clip space.
    let pos = array(
      vec2f(-1, -1),
      vec2f(-1,  3),
      vec2f( 3, -1),
    );
    let p = pos[vertexIndex];
    // We return the position twice. Once for @builtin(position)
    // Once for the fragment shader. The values in the fragment shader
    // will go from -1,-1 to 1,1 across the entire texture.
    return VertexOutput(
      vec4f(p, 0, 1),
      vec4f(p, -1, 1),
    );
  }

  @fragment
  fn mainFS(
    in: VertexOutput,
  ) -> @location(0) vec4f {
    // orient the direction to the view
    let t = viewDirectionProjectionInverse * in.direction;
    // remove the perspective.
    let uvw = normalize(t.xyz / t.w);
    return textureSample(myTexture, mySampler, uvw);
  }
);
// clang-format on
