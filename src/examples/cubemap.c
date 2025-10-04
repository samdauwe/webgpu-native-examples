#include "common_shaders.h"
#include "meshes.h"
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
 * Ref: https://github.com/webgpu/webgpu-samples/tree/main/sample/cubemap
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

static const char* sample_cubemap_fragment_shader_wgsl;

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
  cube_mesh_t cube_mesh;
  struct {
    WGPUBindGroup uniform_buffer_bind_group;
    WGPUBindGroupLayout bind_group_layout;
    struct {
      mat4 model_view_projection;
    } view_mtx;
  } cube;
  wgpu_buffer_t vertices;
  wgpu_buffer_t uniform_buffer_vs;
  struct {
    mat4 projection;
    mat4 model;
    mat4 view;
    mat4 tmp;
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
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1, 0.2, 0.3, 1.0},
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
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  }
};

static void init_cube_mesh(void)
{
  cube_mesh_init(&state.cube_mesh);
}

static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  /* Create a vertex buffer from the cube data. */
  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Vertex data",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(state.cube_mesh.vertex_array),
                    .initial.data = state.cube_mesh.vertex_array,
                  });
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : Transform */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(mat4), // 4x4 matrix
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
  state.cube.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Cube - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.cube.bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Pipeline - Bind group layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts = &state.cube.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_cubemap_texture(wgpu_context_t* wgpu_context)
{
  /* Texture */
  {
    WGPUTextureDescriptor tdesc = {
      .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size      = {FACE_WIDTH, FACE_HEIGHT, NUM_FACES},
      .format    = WGPUTextureFormat_RGBA8Unorm,
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
      .addressModeU  = WGPUAddressMode_Repeat,
      .addressModeV  = WGPUAddressMode_Repeat,
      .addressModeW  = WGPUAddressMode_Repeat,
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
    assert(img_width == FACE_HEIGHT);
    memcpy((void*)response->buffer.ptr, decoded_pixels, FACE_NUM_BYTES);
    stbi_image_free(decoded_pixels);
    ++state.load_count;
  }
}

// Fetch the 6 separate images for negative/positive x, y, z axis of a cubemap
// and upload it into a GPUTexture.
static void fetch_cubemap_texture(void)
{
  // The order of the array layers is [+X, -X, +Y, -Y, +Z, -Z]
  static const char* cubemap_paths[NUM_FACES] = {
    "assets/textures/cubemaps/bridge2_px.jpg", /* Right  */
    "assets/textures/cubemaps/bridge2_nx.jpg", /* Left   */
    "assets/textures/cubemaps/bridge2_py.jpg", /* Top    */
    "assets/textures/cubemaps/bridge2_ny.jpg", /* Bottom */
    "assets/textures/cubemaps/bridge2_pz.jpg", /* Back   */
    "assets/textures/cubemaps/bridge2_nz.jpg", /* Front  */
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

  // Create a host-visible staging buffers that contains the raw image data for
  // each face of the cubemap
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

    // Upload staging buffer to texture
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

  /* Sumbit commmand buffer and cleanup */
  ASSERT(command_buffer != NULL)

  /* Submit to the queue */
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  /* Release command buffer */
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

  /* Clean up staging resources and pixel data */
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

  /* Model matrix */
  glm_mat4_identity(state.view_matrices.model);
  glm_scale(state.view_matrices.model, (vec3){1000.0f, 1000.0f, 1000.0f});

  /* Model view projection matrix */
  glm_mat4_identity(state.cube.view_mtx.model_view_projection);

  /* Other matrices */
  glm_mat4_identity(state.view_matrices.view);
  glm_mat4_identity(state.view_matrices.tmp);
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Setup the view matrices for the camera */
  init_view_matrices(wgpu_context);

  /* Uniform buffer */
  state.uniform_buffer_vs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(mat4), // 4x4 matrix
                  });
  ASSERT(state.uniform_buffer_vs.buffer != NULL);
}

/* Compute camera movement:                               */
/* It rotates around Y axis with a slight pitch movement. */
static void update_transformation_matrix(void)
{
  const float now = stm_ms(stm_now()) / 800.0f;

  /* View matrix */
  glm_mat4_copy(state.view_matrices.view, state.view_matrices.tmp);
  glm_rotate(state.view_matrices.tmp, (PI / 10.f) * sin(now),
             (vec3){1.0f, 0.0f, 0.0f});
  glm_rotate(state.view_matrices.tmp, now * 0.2f, (vec3){0.f, 1.f, 0.f});

  glm_mat4_mul(state.view_matrices.tmp, state.view_matrices.model,
               state.cube.view_mtx.model_view_projection);
  glm_mat4_mul(state.view_matrices.projection,
               state.cube.view_mtx.model_view_projection,
               state.cube.view_mtx.model_view_projection);
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update the model-view-projection matrix */
  update_transformation_matrix();

  /* Map uniform buffer and update it */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_vs.buffer, 0,
                       &state.cube.view_mtx.model_view_projection,
                       state.uniform_buffer_vs.size);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Transform */
      .binding = 0,
      .buffer  = state.uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer_vs.size,
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
    .label      = STRVIEW("Cube uniform buffer - Bind group"),
    .layout     = state.cube.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.cube.uniform_buffer_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.cube.uniform_buffer_bind_group != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, basic_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, sample_cubemap_fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(textured_cube, state.cube_mesh.vertex_size,
                            /* Attribute location 0: Position */
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                                               state.cube_mesh.position_offset),
                            /* Attribute location 1: UV */
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2,
                                               state.cube_mesh.uv_offset))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Cubemap - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &textured_cube_vertex_buffer_layout,
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
      .cullMode  = WGPUCullMode_None,
      .frontFace = WGPUFrontFace_CCW
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
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
    init_cube_mesh();
    init_vertex_buffer(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_cubemap_texture(wgpu_context);
    fetch_cubemap_texture();
    init_uniform_buffers(wgpu_context);
    init_bind_groups(wgpu_context);
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
  update_uniform_buffers(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                    state.cube.uniform_buffer_bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(rpass_enc, state.cube_mesh.vertex_count, 1, 0, 0);
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

  WGPU_RELEASE_RESOURCE(Texture, state.cubemap_texture.handle);
  WGPU_RELEASE_RESOURCE(TextureView, state.cubemap_texture.view);
  WGPU_RELEASE_RESOURCE(Sampler, state.cubemap_texture.sampler);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.cube.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.cube.uniform_buffer_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
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
static const char* sample_cubemap_fragment_shader_wgsl = CODE(
  @group(0) @binding(1) var mySampler: sampler;
  @group(0) @binding(2) var myTexture: texture_cube<f32>;

  @fragment
  fn main(
    @location(0) fragUV: vec2<f32>,
    @location(1) fragPosition: vec4<f32>
  ) -> @location(0) vec4<f32> {
    // Our camera and the skybox cube are both centered at (0, 0, 0) so we can
    // use the cube geomtry position to get viewing vector to sample the cube
    // texture. The magnitude of the vector doesn't matter.
    var cubemapVec = fragPosition.xyz - vec3(0.5);
    return textureSample(myTexture, mySampler, cubemapVec);
  }
);
// clang-format on
