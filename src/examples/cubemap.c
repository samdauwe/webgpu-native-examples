#include "common_shaders.h"
#include "example_base.h"
#include "meshes.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Cubemap
 *
 * This example shows how to render and sample from a cubemap texture.
 *
 * Ref: https://github.com/austinEng/webgpu-samples/tree/main/src/sample/cubemap
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

static const char* sample_cubemap_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Cubemap example
 * -------------------------------------------------------------------------- */

// Cube mesh
static cube_mesh_t cube_mesh = {0};

// Cube struct
static struct {
  WGPUBindGroup uniform_buffer_bind_group;
  WGPUBindGroupLayout bind_group_layout;
  struct {
    mat4 model_view_projection;
  } view_mtx;
} cube = {0};

// Vertex buffer
static wgpu_buffer_t vertices = {0};

// Uniform buffer block object
static wgpu_buffer_t uniform_buffer_vs = {0};

static struct {
  mat4 projection;
  mat4 model;
  mat4 view;
  mat4 tmp;
} view_matrices = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Texture and sampler
static texture_t cubemap_texture = {0};

// Other variables
static const char* example_title = "Cubemap";
static bool prepared             = false;

// Prepare the cube geometry
static void prepare_cube_mesh(void)
{
  cube_mesh_init(&cube_mesh);
}

/* Prepare vertex buffer */
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  /* Create a vertex buffer from the cube data. */
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Vertex data",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(cube_mesh.vertex_array),
                    .initial.data = cube_mesh.vertex_array,
                  });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
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
  cube.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Cube - Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(cube.bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = "Pipeline - Bind group layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &cube.bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

// Fetch the 6 separate images for negative/positive x, y, z axis of a cubemap
// and upload it into a GPUTexture.
static void prepare_cubemap_texture(wgpu_context_t* wgpu_context)
{
  // The order of the array layers is [+X, -X, +Y, -Y, +Z, -Z]
  static const char* cubemap[6] = {
    "textures/cubemaps/bridge2_px.jpg", /* Right  */
    "textures/cubemaps/bridge2_nx.jpg", /* Left   */
    "textures/cubemaps/bridge2_py.jpg", /* Top    */
    "textures/cubemaps/bridge2_ny.jpg", /* Bottom */
    "textures/cubemaps/bridge2_pz.jpg", /* Back   */
    "textures/cubemaps/bridge2_nz.jpg", /* Front  */
  };
  cubemap_texture = wgpu_create_texture_cubemap_from_files(
    wgpu_context, cubemap,
    &(struct wgpu_texture_load_options_t){
      .flip_y = false,
    });
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.1f,
        .g = 0.2f,
        .b = 0.3f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context,
                          &(struct deph_stencil_texture_creation_options_t){
                            .format = WGPUTextureFormat_Depth24PlusStencil8,
                          });

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  /* Projection matrix */
  glm_mat4_identity(view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 3000.0f,
                  view_matrices.projection);

  /* Model matrix */
  glm_mat4_identity(view_matrices.model);
  glm_scale(view_matrices.model, (vec3){1000.0f, 1000.0f, 1000.0f});

  /* Model view projection matrix */
  glm_mat4_identity(cube.view_mtx.model_view_projection);

  /* Other matrices */
  glm_mat4_identity(view_matrices.view);
  glm_mat4_identity(view_matrices.tmp);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Setup the view matrices for the camera
  prepare_view_matrices(context->wgpu_context);

  /* Uniform buffer */
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(mat4), // 4x4 matrix
    });
  ASSERT(uniform_buffer_vs.buffer != NULL);
}

/* Compute camera movement:                               */
/* It rotates around Y axis with a slight pitch movement. */
static void update_transformation_matrix(wgpu_example_context_t* context)
{
  const float now = context->frame.timestamp_millis / 800.0f;

  /* View matrix */
  glm_mat4_copy(view_matrices.view, view_matrices.tmp);
  glm_rotate(view_matrices.tmp, (PI / 10.f) * sin(now),
             (vec3){1.0f, 0.0f, 0.0f});
  glm_rotate(view_matrices.tmp, now * 0.2f, (vec3){0.f, 1.f, 0.f});

  glm_mat4_mul(view_matrices.tmp, view_matrices.model,
               cube.view_mtx.model_view_projection);
  glm_mat4_mul(view_matrices.projection, cube.view_mtx.model_view_projection,
               cube.view_mtx.model_view_projection);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  /* Update the model-view-projection matrix */
  update_transformation_matrix(context);

  /* Map uniform buffer and update it */
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &cube.view_mtx.model_view_projection,
                          uniform_buffer_vs.size);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Transform */
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1 : Sampler */
      .binding = 1,
      .sampler = cubemap_texture.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
       /* Binding 2 : Texture view */
      .binding     = 2,
      .textureView = cubemap_texture.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Cube uniform buffer - Bind group",
    .layout     = cube.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  cube.uniform_buffer_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(cube.uniform_buffer_bind_group != NULL);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    // Since we are seeing from inside of the cube and we are using the regular
    // cube geomtry data with outward-facing normals, the cullMode should be
    // 'front' or 'none'.
    .cullMode = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  // Enable depth testing so that the fragment closest to the camera
  // is rendered in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    textured_cube, cube_mesh.vertex_size,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                       cube_mesh.position_offset),
    // Attribute location 1: UV
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, cube_mesh.uv_offset))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
         wgpu_context, &(wgpu_vertex_state_t){
         .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label            = "Cubemap - Vertex shader WGSL",
            .wgsl_code.source = basic_vertex_shader_wgsl,
            .entry            = "main",
         },
         .buffer_count = 1,
         .buffers = &textured_cube_vertex_buffer_layout,
       });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
         wgpu_context, &(wgpu_fragment_state_t){
         .shader_desc = (wgpu_shader_desc_t){
            // Fragment shader WGSL
            .label            = "Cubemap - Fragment shader WGSL",
            .wgsl_code.source = sample_cubemap_fragment_shader_wgsl,
            .entry            = "main",
         },
         .target_count = 1,
         .targets = &color_target_state,
       });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Cubemap - Render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_cube_mesh();
    prepare_vertex_buffer(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_cubemap_texture(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_bind_groups(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  /* Bind cube vertex buffer (contains position and colors) */
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    cube.uniform_buffer_bind_group, 0, 0);

  /* Draw textured cube */
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, cube_mesh.vertex_count, 1,
                            0, 0);

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  /* Submit command buffer to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }

  if (!context->paused) {
    update_uniform_buffers(context);
  }

  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  wgpu_destroy_texture(&cubemap_texture);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, cube.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, cube.uniform_buffer_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_cubemap(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title   = example_title,
     .overlay = true,
     .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
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
