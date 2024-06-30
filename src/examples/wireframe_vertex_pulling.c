#include "example_base.h"
#include "meshes.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Wireframe and Thick-Line Rendering in WebGPU
 *
 * This example shows how to render a single indexed triangle model as mesh,
 * wireframe, or wireframe with thick lines, without the need to generate
 * additional buffers for line rendering.
 *
 * Uses vertex pulling to let the vertex shader decide which vertices to load,
 * which allows us to render indexed triangle meshes as wireframes or even
 * thick-wireframes.
 *  ** A normal wireframe is obtained by drawing 3 lines (6 vertices) per
 *     triangle. The vertex shader then uses the index buffer to load the
 *     triangle vertices in the order in which we need them to draw lines.
 *  ** A thick wireframe is obtained by rendering each of the 3 lines of a
 *     triangle as a quad (comprising 2 triangles). For each triangle of the
 *     indexed model, we are drawing a total of 3 lines/quads = 6 triangles = 18
 *     vertices. Each of these 18 vertices belongs to one of three lines, and
 *     each vertex shader invocation loads the start and end of the
 *     corresponding line. The line is then projected to screen space, and the
 *     orthoginal of the screen-space line direction is used to shift the
 *     vertices of each quad into the appropriate directions to obtain a thick
 *     line.
 *
 * Ref:
 * https://github.com/m-schuetz/webgpu_wireframe_thicklines
 * https://potree.org/permanent/wireframe_rendering/ (requires Chrome 96)
 * https://xeolabs.com/pdfs/OpenGLInsights.pdf
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* render_points_wgsl;
static const char* render_solid_mesh_wgsl;
static const char* render_wireframe_thick_wgsl;
static const char* render_wireframe_wgsl;

/* -------------------------------------------------------------------------- *
 * Wireframe and Thick-Line Rendering example
 * -------------------------------------------------------------------------- */

// Cube mesh
static indexed_cube_mesh_t indexed_cube_mesh = {0};

// Cube struct
typedef struct cube_t {
  WGPUBindGroup uniform_buffer_bind_group;
  WGPUBindGroupLayout bind_group_layout;
  // Vertex buffer
  wgpu_buffer_t positions;
  // Colors
  wgpu_buffer_t colors;
  // Index buffer
  wgpu_buffer_t indices;
  // Uniform buffer block object
  wgpu_buffer_t uniform_buffer_vs;
  // View matrices
  struct view_matrices_t {
    mat4 world;
    mat4 view;
    mat4 proj;
    float screen_width;
    float screen_height;
    float padding[12];
  } view_matrices;
} cube_t;
static cube_t cube = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Render modes
typedef enum render_mode_enum {
  RenderMode_Solid_Mesh      = 0,
  RenderMode_Points          = 1,
  RenderMode_Wireframe       = 2,
  RenderMode_Wireframe_Thick = 3,
} render_mode_enum;

static render_mode_enum current_render_mode = RenderMode_Solid_Mesh;

// Render pipeline
static WGPURenderPipeline render_pipelines[4] = {0};

// Other variables
static const char* example_title
  = "Wireframe and Thick-Line Rendering in WebGPU";
static bool prepared = false;

static void prepare_cube_mesh(void)
{
  indexed_cube_mesh_init(&indexed_cube_mesh);
}

static void prepare_storage_buffers(wgpu_context_t* wgpu_context)
{
  /* Create position buffer */
  cube.positions = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube position - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage,
                    .size  = sizeof(indexed_cube_mesh.vertex_array),
                    .initial.data = indexed_cube_mesh.vertex_array,
                  });

  /* Create color buffer */
  cube.colors = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube color - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage,
                    .size  = sizeof(indexed_cube_mesh.color_array),
                    .initial.data = indexed_cube_mesh.color_array,
                  });
}

static void prepare_index_buffer(wgpu_context_t* wgpu_context)
{
  /* Create index buffer */
  cube.indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Index buffer",
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size         = sizeof(indexed_cube_mesh.index_array),
                    .count        = indexed_cube_mesh.index_count,
                    .initial.data = indexed_cube_mesh.index_array,
                  });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /*  Binding 0: uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = cube.uniform_buffer_vs.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: positions */
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .hasDynamicOffset = false,
        .minBindingSize   = cube.positions.size,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2: colors */
      .binding    = 2,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .hasDynamicOffset = false,
        .minBindingSize   = cube.colors.size,
      },
      .sampler = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      /* Binding 3: indices */
      .binding    = 3,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_ReadOnlyStorage,
        .hasDynamicOffset = false,
        .minBindingSize   = cube.indices.size,
      },
      .sampler = {0},
    }
  };
  cube.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Cube - Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(cube.bind_group_layout != NULL)

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Cube - Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &cube.bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL)
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
        .r = 0.1f,
        .g = 0.2f,
        .b = 0.3f,
        .a = 1.0f,
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void update_view_matrices(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  glm_mat4_identity(cube.view_matrices.view);
  glm_translate(cube.view_matrices.view, (vec3){0.0f, 0.0f, -4.0f});
  const float now = context->frame.timestamp_millis / 1000.0f;
  glm_rotate(cube.view_matrices.view, 1.0f, (vec3){sin(now), cos(now), 0.0f});

  glm_mat4_identity(cube.view_matrices.world);
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;
  glm_mat4_identity(cube.view_matrices.proj);
  glm_perspective((2 * PI) / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  cube.view_matrices.proj);

  cube.view_matrices.screen_width  = (float)wgpu_context->surface.width;
  cube.view_matrices.screen_height = (float)wgpu_context->surface.height;
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  /* Update the view matrices */
  update_view_matrices(context);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, cube.uniform_buffer_vs.buffer,
                          0, &cube.view_matrices, cube.uniform_buffer_vs.size);
}

static void prepare_uniform_buffer(wgpu_example_context_t* context)
{
  /* Create uniform buffer */
  cube.uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Cube - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(cube.view_matrices),
    });
  ASSERT(cube.uniform_buffer_vs.buffer != NULL)

  /* Update uniform buffer */
  update_uniform_buffers(context);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0: uniform buffer */
      .binding = 0,
      .buffer  = cube.uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = cube.uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
       /* Binding 1: positions */
      .binding = 1,
      .buffer  = cube.positions.buffer,
      .offset  = 0,
      .size    = cube.positions.size,
    },
    [2] = (WGPUBindGroupEntry) {
      /* Binding 2: colors */
      .binding = 2,
      .buffer  = cube.colors.buffer,
      .offset  = 0,
      .size    = cube.colors.size,
    },
    [3] = (WGPUBindGroupEntry) {
      /* Binding 3: indices */
      .binding = 3,
      .buffer  = cube.indices.buffer,
      .offset  = 0,
      .size    = cube.indices.size,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Cube - Bind group layout",
    .layout     = cube.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  cube.uniform_buffer_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(cube.uniform_buffer_bind_group != NULL)
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Render pipeline: Solid mesh */
  {
    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
       wgpu_context, &(wgpu_vertex_state_t){
       .shader_desc = (wgpu_shader_desc_t){
         // Vertex shader WGSL
         .label             = "Render solid mesh - Vertex shader WGSL",
         .wgsl_code.source  = render_solid_mesh_wgsl,
         .entry             = "main_vertex",
       },
       .buffer_count = 0,
       .buffers      = NULL,
     });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
       wgpu_context, &(wgpu_fragment_state_t){
       .shader_desc = (wgpu_shader_desc_t){
         // Fragment shader WGSL
         .label             = "Render solid mesh - Fragment shader WGSL",
         .wgsl_code.source  = render_solid_mesh_wgsl,
         .entry             = "main_fragment",
       },
       .target_count = 1,
       .targets = &color_target_state,
     });

    // Create rendering pipeline using the specified states
    render_pipelines[(uint32_t)RenderMode_Solid_Mesh]
      = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                .label        = "Solid mesh - Render pipeline",
                                .layout       = pipeline_layout,
                                .primitive    = primitive_state,
                                .vertex       = vertex_state,
                                .fragment     = &fragment_state,
                                .depthStencil = &depth_stencil_state,
                                .multisample  = multisample_state,
                              });
    ASSERT(render_pipelines[(uint32_t)RenderMode_Solid_Mesh] != NULL);

    // Shader modules are no longer needed once the graphics pipeline has been
    // created
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Render pipeline: Points */
  {
    primitive_state.topology = WGPUPrimitiveTopology_PointList;

    /* Vertex state */
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
       wgpu_context, &(wgpu_vertex_state_t){
       .shader_desc = (wgpu_shader_desc_t){
         /* Vertex shader WGSL */
         .label            = "Render points - Vertex shader WGSL",
         .wgsl_code.source = render_points_wgsl,
         .entry            = "main_vertex",
       },
       .buffer_count = 0,
       .buffers      = NULL,
     });

    /* Fragment state */
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
       wgpu_context, &(wgpu_fragment_state_t){
       .shader_desc = (wgpu_shader_desc_t){
         /* Vertex shader WGSL */
         .label            = "Render points - Fragment shader WGSL",
         .wgsl_code.source = render_points_wgsl,
         .entry            = "main_fragment",
       },
       .target_count = 1,
       .targets      = &color_target_state,
     });

    /* Create rendering pipeline using the specified states */
    render_pipelines[(uint32_t)RenderMode_Points]
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device,
                                       &(WGPURenderPipelineDescriptor){
                                         .label  = "Points - Render pipeline",
                                         .layout = pipeline_layout,
                                         .primitive    = primitive_state,
                                         .vertex       = vertex_state,
                                         .fragment     = &fragment_state,
                                         .depthStencil = &depth_stencil_state,
                                         .multisample  = multisample_state,
                                       });
    ASSERT(render_pipelines[(uint32_t)RenderMode_Points] != NULL);

    // Shader modules are no longer needed once the graphics pipeline has been
    // created
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Render pipeline: Wireframe */
  {
    primitive_state.topology = WGPUPrimitiveTopology_LineList;

    /* Vertex state */
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
       wgpu_context, &(wgpu_vertex_state_t){
       .shader_desc = (wgpu_shader_desc_t){
         /* Vertex shader WGSL */
         .label            = "Render wireframe - Vertex shader WGSL",
         .wgsl_code.source = render_wireframe_wgsl,
         .entry            = "main_vertex",
       },
       .buffer_count = 0,
       .buffers      = NULL,
     });

    /* Fragment state */
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
       wgpu_context, &(wgpu_fragment_state_t){
       .shader_desc = (wgpu_shader_desc_t){
         /* Vertex shader WGSL */
         .label            = "Render wireframe - Fragment shader WGSL",
         .wgsl_code.source = render_wireframe_wgsl,
         .entry            = "main_fragment",
       },
       .target_count = 1,
       .targets      = &color_target_state,
     });

    /* Create rendering pipeline using the specified states */
    render_pipelines[(uint32_t)RenderMode_Wireframe]
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device,
                                       &(WGPURenderPipelineDescriptor){
                                         .label = "Wireframe - Render pipeline",
                                         .layout       = pipeline_layout,
                                         .primitive    = primitive_state,
                                         .vertex       = vertex_state,
                                         .fragment     = &fragment_state,
                                         .depthStencil = &depth_stencil_state,
                                         .multisample  = multisample_state,
                                       });
    ASSERT(render_pipelines[(uint32_t)RenderMode_Wireframe] != NULL);

    // Shader modules are no longer needed once the graphics pipeline has been
    // created
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Render pipeline: Wireframe Thick */
  {
    primitive_state.topology = WGPUPrimitiveTopology_TriangleList;

    /* Vertex state */
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
       wgpu_context, &(wgpu_vertex_state_t){
       .shader_desc = (wgpu_shader_desc_t){
         // Vertex shader WGSL
         .label            = "Render wireframe thick - Vertex shader WGSL",
         .wgsl_code.source = render_wireframe_thick_wgsl,
         .entry            = "main_vertex",
       },
       .buffer_count = 0,
       .buffers      = NULL,
     });

    /* Fragment state */
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
       wgpu_context, &(wgpu_fragment_state_t){
       .shader_desc = (wgpu_shader_desc_t){
         /* Vertex shader WGSL */
         .label            = "Render wireframe thick - Fragment shader WGSL",
         .wgsl_code.source = render_wireframe_thick_wgsl,
         .entry            = "main_fragment",
       },
       .target_count = 1,
       .targets      = &color_target_state,
     });

    /* Create rendering pipeline using the specified states */
    render_pipelines[(uint32_t)RenderMode_Wireframe_Thick]
      = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                .label  = "Wireframe thick - Render pipeline",
                                .layout = pipeline_layout,
                                .primitive    = primitive_state,
                                .vertex       = vertex_state,
                                .fragment     = &fragment_state,
                                .depthStencil = &depth_stencil_state,
                                .multisample  = multisample_state,
                              });
    ASSERT(render_pipelines[(uint32_t)RenderMode_Wireframe_Thick] != NULL);

    // Shader modules are no longer needed once the graphics pipeline has been
    // created
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_cube_mesh();
    prepare_storage_buffers(context->wgpu_context);
    prepare_index_buffer(context->wgpu_context);
    prepare_uniform_buffer(context);
    setup_pipeline_layout(context->wgpu_context);
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
    static const char* mode[4]
      = {"Solid", "Points", "Wireframe", "Wireframe Thick"};
    int32_t item_index = (int32_t)current_render_mode;
    if (imgui_overlay_combo_box(context->imgui_overlay, "Mode", &item_index,
                                mode, 4)) {
      current_render_mode = (render_mode_enum)item_index;
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(
    wgpu_context->rpass_enc, render_pipelines[(uint32_t)current_render_mode]);

  // Set the bind group
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    cube.uniform_buffer_bind_group, 0, 0);

  // Bind vertex buffers
  wgpuRenderPassEncoderSetVertexBuffer(
    wgpu_context->rpass_enc, 1, cube.positions.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                       cube.colors.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 3,
                                       cube.indices.buffer, 0, WGPU_WHOLE_SIZE);

  if (current_render_mode == RenderMode_Solid_Mesh) {
    // Bind index buffer
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, cube.indices.buffer, WGPUIndexFormat_Uint32, 0,
      cube.indices.size);
    // Draw indexed cube
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                     cube.indices.count, 1, 0, 0, 0);
  }
  else if (current_render_mode == RenderMode_Points) {
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc,
                              indexed_cube_mesh.vertex_count, 1, 0, 0);
  }
  else if (current_render_mode == RenderMode_Wireframe) {
    const uint32_t num_triangles = indexed_cube_mesh.index_count / 3;
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6 * num_triangles, 1, 0,
                              0);
  }
  else if (current_render_mode == RenderMode_Wireframe_Thick) {
    const uint32_t num_triangles = indexed_cube_mesh.index_count / 3;
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3 * 6 * num_triangles, 1,
                              0, 0);
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

static int example_draw(wgpu_example_context_t* context)
{
  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  /* Submit to queue */
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
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, cube.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, cube.uniform_buffer_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, cube.positions.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, cube.colors.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, cube.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, cube.uniform_buffer_vs.buffer)
  for (uint32_t i = 0; i < 4; ++i) {
    WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines[i])
  }
}

void example_wireframe_vertex_pulling(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title   = example_title,
     .overlay = true,
     .vsync   = true
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* render_points_wgsl = CODE(
  struct Uniforms {
    world           : mat4x4<f32>,
    view            : mat4x4<f32>,
    proj            : mat4x4<f32>,
    screen_width    : f32,
    screen_height   : f32
  }

  struct U32s {
      values : array<u32>
  }

  struct F32s {
    values : array<f32>
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<storage, read> positions : F32s;
  @binding(2) @group(0) var<storage, read> colors : U32s;
  @binding(3) @group(0) var<storage, read> indices : U32s;

  struct VertexInput {
    @builtin(instance_index) instanceID : u32,
    @builtin(vertex_index) vertexID : u32
  }

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>
  }

  @vertex
  fn main_vertex(vertex : VertexInput) -> VertexOutput {
    var position = vec4<f32>(
        positions.values[3u * vertex.vertexID + 0u],
        positions.values[3u * vertex.vertexID + 1u],
        positions.values[3u * vertex.vertexID + 2u],
        1.0
    );

    position = uniforms.proj * uniforms.view * uniforms.world * position;

    var color_u32 = colors.values[vertex.vertexID];
    var color = vec4<f32>(
        f32((color_u32 >>  0u) & 0xFFu) / 255.0,
        f32((color_u32 >>  8u) & 0xFFu) / 255.0,
        f32((color_u32 >> 16u) & 0xFFu) / 255.0,
        f32((color_u32 >> 24u) & 0xFFu) / 255.0,
    );

    var output : VertexOutput;
    output.position = position;
    output.color = color;

    return output;
  }

  struct FragmentInput {
    @location(0) color : vec4<f32>
  }

  struct FragmentOutput {
    @location(0) color : vec4<f32>
  }

  @fragment
  fn main_fragment(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = fragment.color;

    return output;
  }
);

static const char* render_solid_mesh_wgsl = CODE(
  struct Uniforms {
    world           : mat4x4<f32>,
    view            : mat4x4<f32>,
    proj            : mat4x4<f32>,
    screen_width    : f32,
    screen_height   : f32
  }

  struct U32s {
    values : array<u32>
  }

  struct F32s {
    values : array<f32>
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<storage, read> positions : F32s;
  @binding(2) @group(0) var<storage, read> colors : U32s;
  @binding(3) @group(0) var<storage, read> indices : U32s;

  struct VertexInput {
    @builtin(instance_index) instanceID : u32,
    @builtin(vertex_index) vertexID : u32
  }

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>
  }

  @vertex
  fn main_vertex(vertex : VertexInput) -> VertexOutput {
    var position = vec4<f32>(
        positions.values[3u * vertex.vertexID + 0u],
        positions.values[3u * vertex.vertexID + 1u],
        positions.values[3u * vertex.vertexID + 2u],
        1.0
    );

    position = uniforms.proj * uniforms.view * uniforms.world * position;

    var color_u32 = colors.values[vertex.vertexID];
    var color = vec4<f32>(
        f32((color_u32 >>  0u) & 0xFFu) / 255.0,
        f32((color_u32 >>  8u) & 0xFFu) / 255.0,
        f32((color_u32 >> 16u) & 0xFFu) / 255.0,
        f32((color_u32 >> 24u) & 0xFFu) / 255.0,
    );

    var output : VertexOutput;
    output.position = position;
    output.color = color;

    return output;
  }

  struct FragmentInput {
    @location(0) color : vec4<f32>
  }

  struct FragmentOutput {
    @location(0) color : vec4<f32>
  }

  @fragment
  fn main_fragment(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = fragment.color;

    return output;
  }
);

static const char* render_wireframe_thick_wgsl = CODE(
  struct Uniforms {
    world           : mat4x4<f32>,
    view            : mat4x4<f32>,
    proj            : mat4x4<f32>,
    screen_width    : f32,
    screen_height   : f32
  }

  struct U32s {
    values : array<u32>
  }

  struct F32s {
    values : array<f32>
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<storage, read> positions : F32s;
  @binding(2) @group(0) var<storage, read> colors : U32s;
  @binding(3) @group(0) var<storage, read> indices : U32s;

  struct VertexInput {
    @builtin(instance_index) instanceID : u32,
    @builtin(vertex_index) vertexID : u32
  }

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>
  }

  @vertex
  fn main_vertex(vertex : VertexInput) -> VertexOutput {
    var lineWidth = 5.0;

    var localToElement = array<u32, 6>(0u, 1u, 1u, 2u, 2u, 0u);

    var triangleIndex = vertex.vertexID / 18u;        // 18 vertices per triangle
    var localVertexIndex = vertex.vertexID % 18u;     // 18 vertices
    var localLineIndex = localVertexIndex / 6u;       // 3 lines, 6 vertices per line, 2 triangles per line

    var startElementIndex = indices.values[3u * triangleIndex + localLineIndex + 0u];
    var endElementIndex = indices.values[3u * triangleIndex + (localLineIndex + 1u) % 3u];

    var start = vec4<f32>(
      positions.values[3u * startElementIndex + 0u],
      positions.values[3u * startElementIndex + 1u],
      positions.values[3u * startElementIndex + 2u],
      1.0
    );

    var end = vec4<f32>(
      positions.values[3u * endElementIndex + 0u],
      positions.values[3u * endElementIndex + 1u],
      positions.values[3u * endElementIndex + 2u],
      1.0
    );

    var localIndex = vertex.vertexID % 6u;

    var position = start;
    var currElementIndex = startElementIndex;
    if (localIndex == 0u || localIndex == 3u|| localIndex == 5u){
      position = start;
      currElementIndex = startElementIndex;
    } else{
      position = end;
      currElementIndex = endElementIndex;
    }

    var worldPos = uniforms.world * position;
    var viewPos = uniforms.view * worldPos;
    var projPos = uniforms.proj * viewPos;

    var dirScreen : vec2<f32>;
    {
      var projStart = uniforms.proj * uniforms.view * uniforms.world * start;
      var projEnd = uniforms.proj * uniforms.view * uniforms.world * end;

      var screenStart = projStart.xy / projStart.w;
      var screenEnd = projEnd.xy / projEnd.w;

      dirScreen = normalize(screenEnd - screenStart);
    }

    { // apply pixel offsets to the 6 vertices of the quad
      var pxOffset = vec2<f32>(1.0, 0.0);

      // move vertices of quad sidewards
      if (localIndex == 0u || localIndex == 1u || localIndex == 3u){
        pxOffset = vec2<f32>(dirScreen.y, -dirScreen.x);
      } else{
        pxOffset = vec2<f32>(-dirScreen.y, dirScreen.x);
      }

      // move vertices of quad outwards
      if (localIndex == 0u || localIndex == 3u || localIndex == 5u){
         pxOffset = pxOffset - dirScreen;
      } else{
        pxOffset = pxOffset + dirScreen;
      }

      var screenDimensions = vec2<f32>(uniforms.screen_width, uniforms.screen_height);
      var adjusted = projPos.xy / projPos.w + lineWidth * pxOffset / screenDimensions;
      projPos = vec4<f32>(adjusted * projPos.w, projPos.zw);
    }

    var color_u32 = colors.values[currElementIndex];
    var color = vec4<f32>(
      f32((color_u32 >>  0u) & 0xFFu) / 255.0,
      f32((color_u32 >>  8u) & 0xFFu) / 255.0,
      f32((color_u32 >> 16u) & 0xFFu) / 255.0,
      f32((color_u32 >> 24u) & 0xFFu) / 255.0,
    );
    // var color = vec4<f32>(0.0, 1.0, 0.0, 1.0);

    var output : VertexOutput;
    output.position = projPos;
    output.color = color;

    return output;
  }

  struct FragmentInput {
    @location(0) color : vec4<f32>
  }

  struct FragmentOutput {
    @location(0) color : vec4<f32>
  }

  @fragment
  fn main_fragment(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = fragment.color;

    return output;
  }
);

static const char* render_wireframe_wgsl = CODE(
  struct Uniforms {
    world           : mat4x4<f32>,
    view            : mat4x4<f32>,
    proj            : mat4x4<f32>,
    screen_width    : f32,
    screen_height   : f32
  }

  struct U32s {
    values : array<u32>
  }

  struct F32s {
    values : array<f32>
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;
  @binding(1) @group(0) var<storage, read> positions : F32s;
  @binding(2) @group(0) var<storage, read> colors : U32s;
  @binding(3) @group(0) var<storage, read> indices : U32s;

  struct VertexInput {
    @builtin(instance_index) instanceID : u32,
    @builtin(vertex_index) vertexID : u32
  }

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>
  }

  @vertex
  fn main_vertex(vertex : VertexInput) -> VertexOutput {
    var localToElement = array<u32, 6>(0u, 1u, 1u, 2u, 2u, 0u);

    var triangleIndex = vertex.vertexID / 6u;
    var localVertexIndex = vertex.vertexID % 6u;

    var elementIndexIndex = 3u * triangleIndex + localToElement[localVertexIndex];
    var elementIndex = indices.values[elementIndexIndex];

    var position = vec4<f32>(
        positions.values[3u * elementIndex + 0u],
        positions.values[3u * elementIndex + 1u],
        positions.values[3u * elementIndex + 2u],
        1.0
    );

    position = uniforms.proj * uniforms.view * uniforms.world * position;

    var color_u32 = colors.values[elementIndex];
    var color = vec4<f32>(
        f32((color_u32 >>  0u) & 0xFFu) / 255.0,
        f32((color_u32 >>  8u) & 0xFFu) / 255.0,
        f32((color_u32 >> 16u) & 0xFFu) / 255.0,
        f32((color_u32 >> 24u) & 0xFFu) / 255.0,
    );

    var output : VertexOutput;
    output.position = position;
    output.color = color;

    return output;
  }

  struct FragmentInput {
    @location(0) color : vec4<f32>
  }

  struct FragmentOutput {
    @location(0) color : vec4<f32>
  }

  @fragment
  fn main_fragment(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = fragment.color;

    return output;
  }
);
// clang-format on
