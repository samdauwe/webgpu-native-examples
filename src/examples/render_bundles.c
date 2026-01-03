#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

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
 * WebGPU Example - Render Bundles
 *
 * This example shows how to use render bundles. It renders a large number of
 * meshes individually as a proxy for a more complex scene in order to
 * demonstrate the reduction in time spent to issue render commands. (Typically
 * a scene like this would make use of instancing to reduce draw overhead.)
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/renderBundles
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* mesh_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Render Bundles example
 * -------------------------------------------------------------------------- */

#define MAX_ASTEROID_COUNT 10000u

/* Renderable */
typedef struct renderable_t {
  wgpu_buffer_t vertices;
  wgpu_buffer_t indices;
} renderable_t;

/* State struct */
static struct {
  /* Scene objects */
  struct {
    renderable_t planet;
    renderable_t asteroids[5];
  } scene;
  /* Renderables with uniforms and bind groups */
  struct {
    renderable_t* renderable;
    wgpu_buffer_t uniforms;
    WGPUBindGroup bind_group;
  } renderables[1 + MAX_ASTEROID_COUNT];
  uint32_t renderables_length;
  /* Textures */
  struct {
    wgpu_texture_t planet;
    wgpu_texture_t moon;
  } textures;
  /* File buffers for async loading */
  struct {
    uint8_t planet[1024 * 1024 * 4];
    uint8_t moon[1024 * 1024 * 4];
  } file_buffers;
  uint32_t textures_loaded;
  /* View matrices */
  struct {
    mat4 transform;
    mat4 view;
    mat4 projection;
    mat4 model_view_projection;
  } view_matrices;
  /* Uniform buffer for frame */
  wgpu_buffer_t uniform_buffer;
  /* Frame bind group */
  WGPUBindGroup frame_bind_group;
  /* Settings */
  struct {
    bool use_render_bundles;
    int32_t asteroid_count;
  } settings;
  /* Pipeline and render bundle */
  WGPURenderPipeline pipeline;
  WGPURenderBundle render_bundle;
  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* Initialization flag */
  WGPUBool initialized;
} state = {
  .renderables_length = 0,
  .textures_loaded = 0,
  .settings = {
    .use_render_bundles = true,
    .asteroid_count     = 5000,
  },
  .view_matrices = {
    .transform             = GLM_MAT4_IDENTITY_INIT,
    .view                  = GLM_MAT4_IDENTITY_INIT,
    .projection            = GLM_MAT4_IDENTITY_INIT,
    .model_view_projection = GLM_MAT4_IDENTITY_INIT,
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
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
  .initialized = false,
};

/* Forward declarations */
static void update_render_bundle(wgpu_context_t* wgpu_context);
static void init_bind_groups(wgpu_context_t* wgpu_context);

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  const uint32_t uniform_buffer_size = 4 * 16; /* 4x4 matrix */

  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = uniform_buffer_size,
                  });
  ASSERT(state.uniform_buffer.buffer != NULL);
}

/* Fetch callback for texture loading */
static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Texture fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* Load image from memory */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);

  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = img_width,
        .height             = img_height,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels = {
        .ptr  = pixels,
        .size = img_width * img_height * 4,
      },
    };
    texture->desc.is_dirty = true;
    state.textures_loaded++;
    /* Note: Don't free pixels here - they will be freed after texture upload */
  }
}

static void init_textures(wgpu_context_t* wgpu_context)
{
  /* Create placeholder textures */
  state.textures.planet = wgpu_create_color_bars_texture(wgpu_context, NULL);
  state.textures.moon   = wgpu_create_color_bars_texture(wgpu_context, NULL);

  /* Start async loading */
  wgpu_texture_t* planet_tex = &state.textures.planet;
  wgpu_texture_t* moon_tex   = &state.textures.moon;

  sfetch_send(&(sfetch_request_t){
    .path      = "assets/textures/saturn.jpg",
    .callback  = fetch_callback,
    .buffer    = SFETCH_RANGE(state.file_buffers.planet),
    .user_data = {
      .ptr  = &planet_tex,
      .size = sizeof(wgpu_texture_t*),
    },
  });

  sfetch_send(&(sfetch_request_t){
    .path      = "assets/textures/moon.jpg",
    .callback  = fetch_callback,
    .buffer    = SFETCH_RANGE(state.file_buffers.moon),
    .user_data = {
      .ptr  = &moon_tex,
      .size = sizeof(wgpu_texture_t*),
    },
  });
}

/**
 * Helper functions to create the required meshes and bind groups for each
 * sphere.
 */
static void create_sphere_renderable(wgpu_context_t* wgpu_context,
                                     renderable_t* renderable, float radius,
                                     uint32_t width_segments,
                                     uint32_t height_segments, float randomness)
{
  /* Create sphere mesh */
  sphere_mesh_t sphere_mesh = {0};
  sphere_mesh_init(&sphere_mesh, radius, width_segments, height_segments,
                   randomness);

  /* Create a vertex buffer from the sphere data. */
  renderable->vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sphere - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sphere_mesh.vertices.length * sizeof(float),
                    .initial.data = sphere_mesh.vertices.data,
                    .count        = sphere_mesh.vertices.length,
                  });
  ASSERT(renderable->vertices.buffer != NULL);

  /* Create an index buffer from the sphere data. */
  renderable->indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sphere - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sphere_mesh.indices.length * sizeof(uint16_t),
                    .initial.data = sphere_mesh.indices.data,
                    .count        = sphere_mesh.indices.length,
                  });
  ASSERT(renderable->indices.buffer != NULL);

  /* Cleanup */
  sphere_mesh_destroy(&sphere_mesh);
}

static void create_frame_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entry = {
    .binding = 0,
    .buffer  = state.uniform_buffer.buffer,
    .offset  = 0,
    .size    = state.uniform_buffer.size,
  };
  state.frame_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Frame - Bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline, 0),
      .entryCount = 1,
      .entries    = &bg_entry,
    });
  ASSERT(state.frame_bind_group != NULL);
}

static void create_sphere_bind_group(wgpu_context_t* wgpu_context,
                                     uint64_t renderable_id,
                                     wgpu_texture_t* texture, mat4 transform)
{
  /* Uniform buffer */
  const uint32_t uniform_buffer_size        = 4 * 16; /* 4x4 matrix */
  state.renderables[renderable_id].uniforms = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Renderable - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = uniform_buffer_size,
                    .initial.data = transform,
                  });

  /* Bind group */
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = state.renderables[renderable_id].uniforms.buffer,
      .offset  = 0,
      .size    = state.renderables[renderable_id].uniforms.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .sampler = texture->sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = texture->view,
    },
  };
  state.renderables[renderable_id].bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Renderables - Bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline, 1),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(state.renderables[renderable_id].bind_group != NULL);
}

/* Initialize all bind groups (called when textures are updated) */
static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Only update if we have valid bind groups already */
  for (uint32_t i = 0; i < state.renderables_length; ++i) {
    if (state.renderables[i].bind_group) {
      WGPU_RELEASE_RESOURCE(BindGroup, state.renderables[i].bind_group)

      wgpu_texture_t* texture
        = (i == 0) ? &state.textures.planet : &state.textures.moon;

      WGPUBindGroupEntry bg_entries[3] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .buffer  = state.renderables[i].uniforms.buffer,
          .offset  = 0,
          .size    = state.renderables[i].uniforms.size,
        },
        [1] = (WGPUBindGroupEntry) {
          .binding = 1,
          .sampler = texture->sampler,
        },
        [2] = (WGPUBindGroupEntry) {
          .binding     = 2,
          .textureView = texture->view,
        },
      };

      state.renderables[i].bind_group = wgpuDeviceCreateBindGroup(
        wgpu_context->device,
        &(WGPUBindGroupDescriptor){
          .label      = STRVIEW("Renderables - Bind group"),
          .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline, 1),
          .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
          .entries    = bg_entries,
        });
      ASSERT(state.renderables[i].bind_group != NULL);
    }
  }

  /* Rebuild render bundle with new bind groups */
  update_render_bundle(wgpu_context);
}

/* Create one large central planet surrounded by a large ring of asteroids */
static void init_scene(wgpu_context_t* wgpu_context)
{
  /* Planet */
  create_sphere_renderable(wgpu_context, &state.scene.planet, 1.0f, 32, 16,
                           0.0f);
  create_sphere_bind_group(wgpu_context, 0, &state.textures.planet,
                           state.view_matrices.transform);
  state.renderables[0].renderable = &state.scene.planet;
  state.renderables_length++;

  /* Asteroids */
  create_sphere_renderable(wgpu_context, &state.scene.asteroids[0], 0.01f, 8, 6,
                           0.15f);
  create_sphere_renderable(wgpu_context, &state.scene.asteroids[1], 0.013f, 8,
                           6, 0.15f);
  create_sphere_renderable(wgpu_context, &state.scene.asteroids[2], 0.017f, 8,
                           6, 0.15f);
  create_sphere_renderable(wgpu_context, &state.scene.asteroids[3], 0.02f, 8, 6,
                           0.15f);
  create_sphere_renderable(wgpu_context, &state.scene.asteroids[4], 0.03f, 16,
                           8, 0.15f);
}

static void ensure_enough_asteroids(wgpu_context_t* wgpu_context)
{
  mat4 tmp_mat              = GLM_MAT4_IDENTITY_INIT;
  uint32_t asteroids_length = (uint32_t)ARRAY_SIZE(state.scene.asteroids);
  float radius = 0.0f, angle = 0.0f, x = 0.0f, y = 0.0f, z = 0.0f;
  for (uint32_t i = state.renderables_length;
       i <= (uint32_t)state.settings.asteroid_count; ++i) {
    /* Place copies of the asteroid in a ring. */
    radius = random_float() * 1.7f + 1.25f;
    angle  = random_float() * PI * 2.0f;
    x      = sin(angle) * radius;
    y      = (random_float() - 0.5f) * 0.015f;
    z      = cos(angle) * radius;

    glm_mat4_identity(tmp_mat);
    glm_translate_to(tmp_mat, (vec3){x, y, z}, state.view_matrices.transform);
    glm_rotate_x(state.view_matrices.transform, random_float() * PI,
                 state.view_matrices.transform);
    glm_rotate_y(state.view_matrices.transform, random_float() * PI,
                 state.view_matrices.transform);
    state.renderables[i].renderable
      = &state.scene.asteroids[i % asteroids_length];
    create_sphere_bind_group(wgpu_context, i, &state.textures.moon,
                             state.view_matrices.transform);
    state.renderables_length++;
  }
}

static void init_depth_stencil(wgpu_context_t* wgpu_context)
{
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Depth stencil texture"),
    .size          = (WGPUExtent3D){
       .width              = wgpu_context->width,
       .height             = wgpu_context->height,
       .depthOrArrayLayers = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth24Plus,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  wgpu_context->depth_stencil_tex
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(wgpu_context->depth_stencil_tex != NULL);

  WGPUTextureViewDescriptor view_desc = {
    .label           = STRVIEW("Depth stencil texture view"),
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_DepthOnly,
  };
  wgpu_context->depth_stencil_view
    = wgpuTextureCreateView(wgpu_context->depth_stencil_tex, &view_desc);
  ASSERT(wgpu_context->depth_stencil_view != NULL);

  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  /* Vertex buffer layout */
  sphere_mesh_layout_t sphere_mesh_layout = {0};
  sphere_mesh_layout_init(&sphere_mesh_layout);
  WGPU_VERTEX_BUFFER_LAYOUT(
    sphere, sphere_mesh_layout.vertex_stride,
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                       sphere_mesh_layout.positions_offset),
    /* Attribute location 1: normal */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       sphere_mesh_layout.normal_offset),
    /* Attribute location 2: uv */
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x2,
                       sphere_mesh_layout.uv_offset))

  /* Create shader module */
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, mesh_shader_wgsl);
  ASSERT(shader_module != NULL);

  /* Create render pipeline */
  state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Sphere mesh - Render pipeline"),
      .vertex = (WGPUVertexState){
        .module     = shader_module,
        .entryPoint = STRVIEW("vertexMain"),
        .bufferCount = 1,
        .buffers     = &sphere_vertex_buffer_layout,
      },
      .primitive = (WGPUPrimitiveState){
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        /* Backface culling since the sphere is solid piece of geometry */
        .cullMode  = WGPUCullMode_Back,
      },
      .fragment = &(WGPUFragmentState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format            = WGPUTextureFormat_Depth24Plus,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });
  ASSERT(state.pipeline != NULL);

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, shader_module);
}

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 0.1f, 100.0f,
                  state.view_matrices.projection);

  /* Model-view projection matrix */
  glm_mat4_identity(state.view_matrices.model_view_projection);
}

static void update_transformation_matrix(float time)
{
  mat4 tmp_mat = GLM_MAT4_IDENTITY_INIT;
  glm_translate_to(tmp_mat, (vec3){0.0f, 0.0f, -4.0f},
                   state.view_matrices.view);
  /* Tilt the view matrix so the planet looks like it's off-axis. */
  glm_rotate_z(state.view_matrices.view, PI * 0.1f, state.view_matrices.view);
  glm_rotate_x(state.view_matrices.view, PI * 0.1f, state.view_matrices.view);
  /* Rotate the view matrix slowly so the planet appears to spin. */
  glm_rotate_y(state.view_matrices.view, time * 0.05f,
               state.view_matrices.view);

  glm_mat4_mul(state.view_matrices.projection, state.view_matrices.view,
               state.view_matrices.model_view_projection);
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  const float now = stm_sec(stm_now());
  update_transformation_matrix(now);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer, 0,
                       &state.view_matrices.model_view_projection,
                       sizeof(mat4));
}

static void update_render_bundle(wgpu_context_t* wgpu_context);

/**
 * Render bundles function as partial, limited render passes, so we can use the
 * same code both to render the scene normally and to build the render bundle.
 */
#define RENDER_SCENE(Type, rpass_enc)                                          \
  if (rpass_enc) {                                                             \
    wgpu##Type##SetPipeline(rpass_enc, state.pipeline);                        \
    wgpu##Type##SetBindGroup(rpass_enc, 0, state.frame_bind_group, 0, 0);      \
    /* Loop through every renderable object and draw them individually.        \
     * (Because many of these meshes are repeated, with only the transforms    \
     * differing, instancing would be highly effective here. This sample       \
     * intentionally avoids using instancing in order to emulate a more        \
     * complex scene, which helps demonstrate the potential time savings a     \
     * render bundle can provide.) */                                          \
    int32_t count = 0;                                                         \
    for (uint32_t ri = 0; ri < state.renderables_length; ++ri) {               \
      wgpu##Type##SetBindGroup(rpass_enc, 1, state.renderables[ri].bind_group, \
                               0, 0);                                          \
      wgpu##Type##SetVertexBuffer(                                             \
        rpass_enc, 0, state.renderables[ri].renderable->vertices.buffer, 0,    \
        WGPU_WHOLE_SIZE);                                                      \
      wgpu##Type##SetIndexBuffer(                                              \
        rpass_enc, state.renderables[ri].renderable->indices.buffer,           \
        WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);                           \
      wgpu##Type##DrawIndexed(rpass_enc,                                       \
                              state.renderables[ri].renderable->indices.count, \
                              1, 0, 0, 0);                                     \
      if (++count > state.settings.asteroid_count) {                           \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
  }

/*
 * The render bundle can be encoded once and re-used as many times as needed.
 * Because it encodes all of the commands needed to render at the GPU level,
 * those commands will not need to execute the associated C code upon execution
 * or be re-validated, which can represent a significant time savings.
 *
 * However, because render bundles are immutable once created, they are only
 * appropriate for rendering content where the same commands will be executed
 * every time, with the only changes being the contents of the buffers and
 * textures used. Cases where the executed commands differ from frame-to-frame,
 * such as when using frustum or occlusion culling, will not benefit from using
 * render bundles as much.
 */
static void update_render_bundle(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(RenderBundle, state.render_bundle)

  WGPURenderBundleEncoder render_bundle_encoder
    = wgpuDeviceCreateRenderBundleEncoder(
      wgpu_context->device,
      &(WGPURenderBundleEncoderDescriptor){
        .label              = STRVIEW("Scene - Bundle encoder"),
        .colorFormatCount   = 1,
        .colorFormats       = &wgpu_context->render_format,
        .depthStencilFormat = WGPUTextureFormat_Depth24Plus,
        .sampleCount        = 1,
      });
  RENDER_SCENE(RenderBundleEncoder, render_bundle_encoder)
  state.render_bundle
    = wgpuRenderBundleEncoderFinish(render_bundle_encoder, NULL);
  ASSERT(state.render_bundle != NULL);

  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, render_bundle_encoder)
}

/* Input event callback */
static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    /* Recreate depth stencil texture on resize */
    init_depth_stencil(wgpu_context);
    /* Update view matrices with new aspect ratio */
    init_view_matrices(wgpu_context);
  }
}

/* Render frame */
static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async file requests */
  sfetch_dowork();

  /* Update textures if they were loaded */
  bool textures_updated = false;
  if (state.textures.planet.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.textures.planet);
    FREE_TEXTURE_PIXELS(state.textures.planet);
    textures_updated = true;
  }
  if (state.textures.moon.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.textures.moon);
    FREE_TEXTURE_PIXELS(state.textures.moon);
    textures_updated = true;
  }
  if (textures_updated) {
    /* Rebuild bind groups with new textures */
    init_bind_groups(wgpu_context);
  }

  /* Update uniform buffers */
  update_uniform_buffers(wgpu_context);

  /* Set target frame buffer */
  state.color_attachment.view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  ASSERT(cmd_enc != NULL);

  /* Begin render pass */
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);
  ASSERT(rpass_enc != NULL);

  if (state.settings.use_render_bundles) {
    /* Executing a bundle is equivalent to calling all of the commands encoded
     * in the render bundle as part of the current render pass. */
    wgpuRenderPassEncoderExecuteBundles(rpass_enc, 1, &state.render_bundle);
  }
  else {
    /* Alternatively, the same render commands can be encoded manually, which
     * can take longer since each command needs to be interpreted and
     * re-validated each time. */
    RENDER_SCENE(RenderPassEncoder, rpass_enc)
  }

  /* End render pass */
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)

  /* Get command buffer */
  WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(cmd_enc, NULL);
  ASSERT(cmd_buf != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc)

  /* Submit command buffer to queue */
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buf);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buf)

  return EXIT_SUCCESS;
}

/* Initialize function */
static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  /* Initialize sokol time and fetch */
  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 2,
    .num_channels = 1,
    .num_lanes    = 1,
  });

  /* Initialize pipeline and resources */
  init_pipeline(wgpu_context);
  init_uniform_buffer(wgpu_context);
  init_textures(wgpu_context);
  init_scene(wgpu_context);
  ensure_enough_asteroids(wgpu_context);
  init_depth_stencil(wgpu_context);
  init_view_matrices(wgpu_context);
  create_frame_bind_group(wgpu_context);
  update_render_bundle(wgpu_context);

  state.initialized = true;

  return EXIT_SUCCESS;
}

/* Shutdown/cleanup function */
static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Release render bundle and bind groups */
  WGPU_RELEASE_RESOURCE(RenderBundle, state.render_bundle)
  WGPU_RELEASE_RESOURCE(BindGroup, state.frame_bind_group)

  /* Release scene resources */
  wgpu_destroy_buffer(&state.scene.planet.vertices);
  wgpu_destroy_buffer(&state.scene.planet.indices);
  for (uint32_t i = 0u; i < (uint32_t)ARRAY_SIZE(state.scene.asteroids); ++i) {
    renderable_t* renderable = &state.scene.asteroids[i];
    wgpu_destroy_buffer(&renderable->vertices);
    wgpu_destroy_buffer(&renderable->indices);
  }

  /* Release renderables */
  for (uint32_t ri = 0u; ri < state.renderables_length; ++ri) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.renderables[ri].bind_group)
    wgpu_destroy_buffer(&state.renderables[ri].uniforms);
  }

  /* Release textures */
  wgpu_destroy_texture(&state.textures.moon);
  wgpu_destroy_texture(&state.textures.planet);

  /* Release uniform buffer and pipeline */
  wgpu_destroy_buffer(&state.uniform_buffer);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)

  /* Shutdown sokol fetch */
  sfetch_shutdown();
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Render Bundles",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* mesh_shader_wgsl = CODE(
  struct Uniforms {
    viewProjectionMatrix : mat4x4f
  }
  @group(0) @binding(0) var<uniform> uniforms : Uniforms;

  @group(1) @binding(0) var<uniform> modelMatrix : mat4x4f;

  struct VertexInput {
    @location(0) position : vec4f,
    @location(1) normal : vec3f,
    @location(2) uv : vec2f
  }

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) normal: vec3f,
    @location(1) uv : vec2f,
  }

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.position = uniforms.viewProjectionMatrix * modelMatrix * input.position;
    output.normal = normalize((modelMatrix * vec4(input.normal, 0)).xyz);
    output.uv = input.uv;
    return output;
  }

  @group(1) @binding(1) var meshSampler: sampler;
  @group(1) @binding(2) var meshTexture: texture_2d<f32>;

  // Static directional lighting
  const lightDir = vec3f(1, 1, 1);
  const dirColor = vec3(1);
  const ambientColor = vec3f(0.05);

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let textureColor = textureSample(meshTexture, meshSampler, input.uv);

    // Very simplified lighting algorithm.
    let lightColor = saturate(ambientColor + max(dot(input.normal, lightDir), 0.0) * dirColor);

    return vec4f(textureColor.rgb * lightColor, textureColor.a);
  }
);
// clang-format on
