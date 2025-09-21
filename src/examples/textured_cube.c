#include "common_shaders.h"
#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include "sokol_fetch.h"

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
 * WebGPU Example - Textured Cube
 *
 * This example shows how to bind and sample textures.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/texturedCube
 * https://github.com/gfx-rs/wgpu-rs/tree/master/examples/cube
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* sampled_texture_mix_color_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Textured Cube example
 * -------------------------------------------------------------------------- */

/* State struct */
static struct {
  cube_mesh_t cube_mesh;
  wgpu_buffer_t vertices;
  struct {
    struct {
      WGPUBindGroup handle;
      bool is_dirty;
    } bind_group;
    WGPUBindGroupLayout bind_group_layout;
    struct {
      mat4 model_view_projection;
    } view_mtx;
  } cube;
  wgpu_buffer_t uniform_buffer_vs;
  struct {
    mat4 projection;
    mat4 view;
  } view_matrices;
  wgpu_texture_t texture;
  uint8_t file_buffer[512 * 512 * 4];
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
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
  .render_pass_dscriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  }
};

/* Prepare the cube geometry */
static void init_cube_mesh(void)
{
  cube_mesh_init(&state.cube_mesh);
}

/* Create a vertex buffer from the cube data. */
static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Textured Cube - Vertices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(state.cube_mesh.vertex_array),
                    .initial.data = state.cube_mesh.vertex_array,
                  });
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Transform */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(mat4), /* 4x4 matrix */
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Texture view */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    }
  };
  state.cube.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Textured Cube - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    });
  ASSERT(state.cube.bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Textured Cube - Render pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.cube.bind_group_layout,
    });
  ASSERT(state.pipeline_layout != NULL);
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
    state.texture.desc = (wgpu_texture_desc_t){
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
    state.texture.desc.is_dirty = true;
  }
}

static void init_texture(wgpu_context_t* wgpu_context)
{
  state.texture = wgpu_create_color_bars_texture(wgpu_context, 16, 16);
}

static void fetch_texture(void)
{
  /* Start loading the image file */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/Di-3d.png",
    .callback = fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
  });
}

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  // Projection matrix
  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  state.view_matrices.projection);
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Setup the view matrices for the camera */
  init_view_matrices(wgpu_context);

  /* Uniform buffer */
  state.uniform_buffer_vs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Camera view matrices - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(mat4), // 4x4 matrix
                  });
  ASSERT(state.uniform_buffer_vs.buffer != NULL);
}

static void update_transformation_matrix(void)
{
  const float now = stm_sec(stm_now());

  /* View matrix */
  glm_mat4_identity(state.view_matrices.view);
  glm_translate(state.view_matrices.view, (vec3){0.0f, 0.0f, -4.0f});
  glm_rotate(state.view_matrices.view, 1.0f, (vec3){sin(now), cos(now), 0.0f});

  /* Model view projection matrix */
  glm_mat4_identity(state.cube.view_mtx.model_view_projection);
  glm_mat4_mul(state.view_matrices.projection, state.view_matrices.view,
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

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.cube.bind_group.handle)

  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = state.uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .sampler = state.texture.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = state.texture.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Cube uniform - Buffer bind group"),
    .layout     = state.cube.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.cube.bind_group.handle
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.cube.bind_group.handle != NULL);
  state.cube.bind_group.is_dirty = false;
}

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, basic_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, sampled_texture_mix_color_fragment_shader_wgsl);

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
    .label  = STRVIEW("Textured cubes - Render pipeline"),
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
      .cullMode  = WGPUCullMode_Back,
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
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
    });
    init_cube_mesh();
    init_vertex_buffer(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_texture(wgpu_context);
    fetch_texture();
    init_bind_group(wgpu_context);
    init_pipelines(wgpu_context);
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

  /* Recreate texture when pixel data loaded */
  if (state.texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.texture);
    FREE_TEXTURE_PIXELS(state.texture);
    /* Upddate the bindgroup */
    init_bind_group(wgpu_context);
  }

  /* Update matrix data */
  update_uniform_buffers(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_dscriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.cube.bind_group.handle,
                                    0, 0);
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

  wgpu_destroy_texture(&state.texture);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.cube.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.cube.bind_group.handle)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Textured Cube",
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
static const char* sampled_texture_mix_color_fragment_shader_wgsl = CODE(
  @group(0) @binding(1) var mySampler: sampler;
  @group(0) @binding(2) var myTexture: texture_2d<f32>;

  @fragment
  fn main(
    @location(0) fragUV: vec2f,
    @location(1) fragPosition: vec4f
  ) -> @location(0) vec4f {
    return textureSample(myTexture, mySampler, fragUV) * fragPosition;
  }
);
// clang-format on
