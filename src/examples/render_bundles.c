#include "example_base.h"
#include "meshes.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

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

#define MAX_ASTEROID_COUNT 10000u

// Renderables
typedef struct renderable_t {
  wgpu_buffer_t vertices;
  wgpu_buffer_t indices;
} renderable_t;

static struct {
  renderable_t planet;
  renderable_t asteroids[5];
} scene = {0};

static struct {
  renderable_t* renderable;
  wgpu_buffer_t uniforms;
  WGPUBindGroup bind_group;
} renderables[1 + MAX_ASTEROID_COUNT] = {0};
static uint32_t renderables_length    = 0;

// Texture
static struct {
  texture_t planet;
  texture_t moon;
  WGPUSampler sampler;
} textures = {0};

// View matrices
static struct {
  mat4 transform;
  mat4 view;
  mat4 projection;
  mat4 model_view_projection;
} view_matrices = {
  .transform             = GLM_MAT4_IDENTITY_INIT,
  .view                  = GLM_MAT4_IDENTITY_INIT,
  .projection            = GLM_MAT4_IDENTITY_INIT,
  .model_view_projection = GLM_MAT4_IDENTITY_INIT,
};

// Uniform buffer
static wgpu_buffer_t uniform_buffer = {0};

// Frame bind group
static WGPUBindGroup frame_bind_group = NULL;

// Settings
static struct {
  bool use_render_bundles;
  int32_t asteroid_count;
} settings = {
  .use_render_bundles = true,
  .asteroid_count     = 500,
};

// Mesh render pipeline
static WGPURenderPipeline mesh_render_pipeline = NULL;

// Render bundle
static WGPURenderBundle render_bundle = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Other variables
static const char* example_title = "Render Bundles";
static bool prepared             = false;

static void prepare_uniform_buffer(wgpu_context_t* wgpu_context)
{
  const uint32_t uniform_buffer_size = 4 * 16; /* 4x4 matrix */

  uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniform bufer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = uniform_buffer_size,
                  });
  ASSERT(uniform_buffer.buffer != NULL);
}

static void prepare_planet_texture(wgpu_context_t* wgpu_context)
{
  const char* file                                    = "textures/saturn.jpg";
  wgpu_texture_load_options wgpu_texture_load_options = {
    .label        = "Saturn texture",
    .format       = WGPUTextureFormat_RGBA8Unorm,
    .address_mode = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                    | WGPUTextureUsage_RenderAttachment,
  };
  textures.planet = wgpu_create_texture_from_file(wgpu_context, file,
                                                  &wgpu_texture_load_options);
  ASSERT(textures.planet.texture != NULL);
}

static void prepare_moon_texture(wgpu_context_t* wgpu_context)
{
  const char* file                                    = "textures/moon.jpg";
  wgpu_texture_load_options wgpu_texture_load_options = {
    .label        = "Moon texture",
    .format       = WGPUTextureFormat_RGBA8Unorm,
    .address_mode = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                    | WGPUTextureUsage_RenderAttachment,
  };
  textures.moon = wgpu_create_texture_from_file(wgpu_context, file,
                                                &wgpu_texture_load_options);
  ASSERT(textures.moon.texture != NULL);
}

static void prepare_texture_sampler(wgpu_context_t* wgpu_context)
{
  // Create linear sampler
  WGPUSamplerDescriptor sampler_desc = {
    .label         = "Linear texture sampler",
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .magFilter     = WGPUFilterMode_Linear,
    .minFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  textures.sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(textures.sampler != NULL);
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
                    .label = "Sphere vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sphere_mesh.vertices.length * sizeof(float),
                    .initial.data = sphere_mesh.vertices.data,
                    .count        = sphere_mesh.vertices.length,
                  });
  ASSERT(renderable->vertices.buffer != NULL);

  /* Create an index buffer from the sphere data. */
  renderable->indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sphere index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sphere_mesh.indices.length * sizeof(uint16_t),
                    .initial.data = sphere_mesh.indices.data,
                    .count        = sphere_mesh.indices.length,
                  });
  ASSERT(renderable->indices.buffer != NULL);

  /* Cleanup */
  sphere_mesh_destroy(&sphere_mesh);
}

static void create_create_frame_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entry = (WGPUBindGroupEntry){
    .binding = 0,
    .buffer  = uniform_buffer.buffer,
    .offset  = 0,
    .size    = uniform_buffer.size,
  };
  frame_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label  = "Frame bind group",
      .layout = wgpuRenderPipelineGetBindGroupLayout(mesh_render_pipeline, 0),
      .entryCount = 1,
      .entries    = &bg_entry,
    });
}

static void create_sphere_bind_group(wgpu_context_t* wgpu_context,
                                     uint64_t renderable_id, texture_t* texture,
                                     mat4 transform)
{
  /* Uniform buffer */
  const uint32_t uniform_buffer_size  = 4 * 16; /* 4x4 matrix */
  renderables[renderable_id].uniforms = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Renderable uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = uniform_buffer_size,
                    .initial.data = transform,
                  });

  /* Bind group */
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = renderables[renderable_id].uniforms.buffer,
      .offset  = 0,
      .size    = renderables[renderable_id].uniforms.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .sampler = textures.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = texture->view,
    },
  };
  renderables[renderable_id].bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label  = "Bind group",
      .layout = wgpuRenderPipelineGetBindGroupLayout(mesh_render_pipeline, 1),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(renderables[renderable_id].bind_group != NULL);
}

/* Create one large central planet surrounded by a large ring of asteroids */
static void prepare_scene(wgpu_context_t* wgpu_context)
{
  /* Planet */
  create_sphere_renderable(wgpu_context, &scene.planet, 1.0f, 32, 16, 0.0f);
  create_sphere_bind_group(wgpu_context, 0, &textures.planet,
                           view_matrices.transform);
  renderables[0].renderable = &scene.planet;
  renderables_length++;

  /* Asteroids */
  create_sphere_renderable(wgpu_context, &scene.asteroids[0], 0.01f, 8, 6,
                           0.15f);
  create_sphere_renderable(wgpu_context, &scene.asteroids[1], 0.013f, 8, 6,
                           0.15f);
  create_sphere_renderable(wgpu_context, &scene.asteroids[2], 0.017f, 8, 6,
                           0.15f);
  create_sphere_renderable(wgpu_context, &scene.asteroids[3], 0.02f, 8, 6,
                           0.15f);
  create_sphere_renderable(wgpu_context, &scene.asteroids[4], 0.03f, 16, 8,
                           0.15f);
}

static void ensure_enough_asteroids(wgpu_context_t* wgpu_context)
{
  mat4 tmp_mat              = GLM_MAT4_IDENTITY_INIT;
  uint32_t asteroids_length = (uint32_t)ARRAY_SIZE(scene.asteroids);
  float radius = 0.0f, angle = 0.0f, x = 0.0f, y = 0.0f, z = 0.0f;
  for (uint32_t i = renderables_length; i <= (uint32_t)settings.asteroid_count;
       ++i) {
    /* Place copies of the asteroid in a ring. */
    radius = random_float() * 1.7f + 1.25f;
    angle  = random_float() * PI * 2.0f;
    x      = sin(angle) * radius;
    y      = (random_float() - 0.5f) * 0.015f;
    z      = cos(angle) * radius;

    glm_mat4_identity(tmp_mat);
    glm_translate_to(tmp_mat, (vec3){x, y, z}, view_matrices.transform);
    glm_rotate_x(view_matrices.transform, random_float() * PI,
                 view_matrices.transform);
    glm_rotate_y(view_matrices.transform, random_float() * PI,
                 view_matrices.transform);
    renderables[i].renderable = &scene.asteroids[i % asteroids_length];
    create_sphere_bind_group(wgpu_context, i, &textures.moon,
                             view_matrices.transform);
    renderables_length++;
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
    .view       = NULL, /* Assigned later */
    .depthSlice = ~0,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor) {
      .r = 0.0f,
      .g = 0.0f,
      .b = 0.0f,
      .a = 1.0f,
    },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context,
                          &(deph_stencil_texture_creation_options){
                            .format = WGPUTextureFormat_Depth24Plus,
                          });

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    // Backface culling since the sphere is solid piece of geometry.
    // Faces pointing away from the camera will be occluded by faces pointing
    // toward the camera.
    .cullMode = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Enable depth testing so that the fragment closest to the camera is rendered
  // in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout */
  sphere_mesh_layout_t sphere_mesh_layout = {0};
  sphere_mesh_layout_init(&sphere_mesh_layout);
  WGPU_VERTEX_BUFFER_LAYOUT(
    sphere, sphere_mesh_layout.vertex_stride,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                       sphere_mesh_layout.positions_offset),
    // Attribute location 1: normal
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       sphere_mesh_layout.normal_offset),
    // Attribute location 2: uv
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x2,
                       sphere_mesh_layout.uv_offset))

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      /* Vertex shader WGSL */
      .label = "Vertex shader WGSL",
      .file  = "shaders/render_bundles/mesh.wgsl",
      .entry = "vertexMain"
    },
    .buffer_count = 1,
    .buffers      = &sphere_vertex_buffer_layout,
  });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
    .shader_desc = (wgpu_shader_desc_t){
        /* Fragment shader WGSL */
        .label = "Fragment shader WGSL",
        .file  = "shaders/render_bundles/mesh.wgsl",
        .entry = "fragmentMain"
       },
      .target_count = 1,
      .targets = &color_target_state,
  });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Create rendering pipeline using the specified states */
  mesh_render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Sphere mesh render pipeline",
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(mesh_render_pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  /* Projection matrix */
  glm_mat4_identity(view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 0.1f, 100.0f,
                  view_matrices.projection);

  /* Model-view projection matrix */
  glm_mat4_identity(view_matrices.model_view_projection);
}

static void update_transformation_matrix(wgpu_example_context_t* context)
{
  mat4 tmp_mat = GLM_MAT4_IDENTITY_INIT;
  glm_translate_to(tmp_mat, (vec3){0.0f, 0.0f, -4.0f}, view_matrices.view);
  const float now = context->frame.timestamp_millis / 1000.0f;
  /* Tilt the view matrix so the planet looks like it's off-axis. */
  glm_rotate_z(view_matrices.view, PI * 0.1f, view_matrices.view);
  glm_rotate_x(view_matrices.view, PI * 0.1f, view_matrices.view);
  /* Rotate the view matrix slowly so the planet appears to spin. */
  glm_rotate_y(view_matrices.view, now * 0.05f, view_matrices.view);

  glm_mat4_mul(view_matrices.projection, view_matrices.view,
               view_matrices.model_view_projection);
}

static void update_uniform_buffer(wgpu_example_context_t* context)
{
  update_transformation_matrix(context);

  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer.buffer, 0,
                          &view_matrices.model_view_projection, sizeof(mat4));
}

static void update_render_bundle(wgpu_context_t* wgpu_context);

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    imgui_overlay_checkBox(context->imgui_overlay, "Use Render Bundles",
                           &settings.use_render_bundles);
    if (imgui_overlay_slider_int(context->imgui_overlay, "Asteroid Count",
                                 &settings.asteroid_count, 500, 10000)) {
      /**
       * If the content of the scene changes the render bundle must be
       * recreated.
       */
      ensure_enough_asteroids(context->wgpu_context);
      update_render_bundle(context->wgpu_context);
    }
  }
}

/**
 * Render bundles function as partial, limited render passes, so we can use the
 * same code both to render the scene normally and to build the render bundle.
 */
#define RENDER_SCENE(Type, rpass_enc)                                          \
  if (rpass_enc) {                                                             \
    wgpu##Type##SetPipeline(rpass_enc, mesh_render_pipeline);                  \
    wgpu##Type##SetBindGroup(rpass_enc, 0, frame_bind_group, 0, 0);            \
    /**                                                                        \
     *  Loop through every renderable object and draw them individually.       \
     * (Because many of these meshes are repeated, with only the transforms    \
     * differing, instancing would be highly effective here. This sample       \
     * intentionally avoids using instancing in order to emulate a more        \
     * complex scene, which helps demonstrate the potential time savings a     \
     * render bundle can provide.)                                             \
     */                                                                        \
    int32_t count = 0;                                                         \
    for (uint32_t ri = 0; ri < renderables_length; ++ri) {                     \
      wgpu##Type##SetBindGroup(rpass_enc, 1, renderables[ri].bind_group, 0,    \
                               0);                                             \
      wgpu##Type##SetVertexBuffer(rpass_enc, 0,                                \
                                  renderables[ri].renderable->vertices.buffer, \
                                  0, WGPU_WHOLE_SIZE);                         \
      wgpu##Type##SetIndexBuffer(rpass_enc,                                    \
                                 renderables[ri].renderable->indices.buffer,   \
                                 WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);  \
      wgpu##Type##DrawIndexed(                                                 \
        rpass_enc, renderables[ri].renderable->indices.count, 1, 0, 0, 0);     \
      if (++count > settings.asteroid_count) {                                 \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
  }

/*
 * The render bundle can be encoded once and re-used as many times as needed.
 * Because it encodes all of the commands needed to render at the GPU level,
 * those commands will not need to execute the associated JavaScript code upon
 * execution or be re-validated, which can represent a significant time savings.
 *
 * However, because render bundles are immutable once created, they are only
 * appropriate for rendering content where the same commands will be executed
 * every time, with the only changes being the contents of the buffers and
 * textures used. Cases where the executed commands differ from frame-to-frame,
 * such as when using frustrum or occlusion culling, will not benefit from
 * using render bundles as much.
 */
static void update_render_bundle(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(RenderBundle, render_bundle)

  WGPUTextureFormat color_formats[1] = {wgpu_context->swap_chain.format};
  WGPURenderBundleEncoder render_bundle_encoder
    = wgpuDeviceCreateRenderBundleEncoder(
      wgpu_context->device,
      &(WGPURenderBundleEncoderDescriptor){
        .label              = "Scene bundle encoder",
        .colorFormatCount   = (uint32_t)ARRAY_SIZE(color_formats),
        .colorFormats       = color_formats,
        .depthStencilFormat = WGPUTextureFormat_Depth24Plus,
        .sampleCount        = 1,
      });
  RENDER_SCENE(RenderBundleEncoder, render_bundle_encoder)
  render_bundle = wgpuRenderBundleEncoderFinish(render_bundle_encoder, NULL);
  ASSERT(render_bundle != NULL);

  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, render_bundle_encoder)
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  if (settings.use_render_bundles) {
    /**
     * Executing a bundle is equivalent to calling all of the commands encoded
     * in the render bundle as part of the current render pass.
     */
    wgpuRenderPassEncoderExecuteBundles(wgpu_context->rpass_enc, 1,
                                        &render_bundle);
  }
  else {
    /**
     * Alternatively, the same render commands can be encoded manually, which
     * can take longer since each command needs to be interpreted and
     * re-validated each time.
     */
    RENDER_SCENE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_pipeline(context->wgpu_context);
    prepare_uniform_buffer(context->wgpu_context);
    prepare_planet_texture(context->wgpu_context);
    prepare_moon_texture(context->wgpu_context);
    prepare_texture_sampler(context->wgpu_context);
    prepare_scene(context->wgpu_context);
    ensure_enough_asteroids(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepare_view_matrices(context->wgpu_context);
    create_create_frame_bind_group(context->wgpu_context);
    update_render_bundle(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int example_draw(wgpu_example_context_t* context)
{
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit command buffers to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  if (!context->paused) {
    update_uniform_buffer(context);
  }
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(RenderBundle, render_bundle)
  WGPU_RELEASE_RESOURCE(BindGroup, frame_bind_group)
  wgpu_destroy_buffer(&scene.planet.vertices);
  wgpu_destroy_buffer(&scene.planet.indices);
  for (uint32_t i = 0u; i < (uint32_t)ARRAY_SIZE(scene.asteroids); ++i) {
    renderable_t* renderable = &scene.asteroids[i];
    wgpu_destroy_buffer(&renderable->vertices);
    wgpu_destroy_buffer(&renderable->indices);
  }
  for (uint32_t ri = 0u; ri < renderables_length; ++ri) {
    WGPU_RELEASE_RESOURCE(BindGroup, renderables[ri].bind_group)
    wgpu_destroy_buffer(&renderables[ri].uniforms);
  }
  WGPU_RELEASE_RESOURCE(Sampler, textures.sampler)
  wgpu_destroy_texture(&textures.moon);
  wgpu_destroy_texture(&textures.planet);
  wgpu_destroy_buffer(&uniform_buffer);
  WGPU_RELEASE_RESOURCE(RenderPipeline, mesh_render_pipeline)
}

void example_render_bundles(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title                       = example_title,
      .overlay                     = true,
      .overlay_deph_stencil_format = WGPUTextureFormat_Depth24Plus,
  },
  .example_initialize_func = &example_initialize,
  .example_render_func     = &example_render,
  .example_destroy_func    = &example_destroy
  });
  // clang-format on
}
