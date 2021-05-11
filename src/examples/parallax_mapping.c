#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Parallax Mapping
 *
 * Implements multiple texture mapping methods to simulate depth based on
 * texture information: Normal mapping, parallax mapping, steep parallax mapping
 * and parallax occlusion mapping (best quality, worst performance).
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/parallaxmapping/parallaxmapping.cpp
 * -------------------------------------------------------------------------- */

static struct {
  texture_t color_map;
  // Normals and height are combined into one texture (height = alpha channel)
  texture_t normal_height_map;
} textures;

// Vertex layout for this example
typedef struct vertex_t {
  vec3 pos;
  vec2 uv;
  vec3 normal;
  vec4 tangent;
} vertex_t;

// Vertex buffer
static struct vertices_t {
  WGPUBuffer buffer;
  uint32_t count;
} vertices = {0};

// Index buffer
static struct indices_t {
  WGPUBuffer buffer;
  uint32_t count;
} indices = {0};

static struct {
  WGPUBuffer vertex_shader;
  WGPUBuffer fragment_shader;
} uniform_buffers;

static struct {
  struct {
    mat4 projection;
    mat4 view;
    mat4 model;
    vec4 light_pos;
    vec4 camera_pos;
  } vertex_shader;
  struct {
    float height_scale;
    // Basic parallax mapping needs a bias to look any good (and is hard to
    // tweak)
    float parallax_bias;
    // Number of layers for steep parallax and parallax occlusion (more layer =
    // better result for less performance)
    float num_layers;
    // (Parallax) mapping mode to use
    int32_t mapping_mode;
  } fragment_shader;
} ubos = {
  .vertex_shader = {
    .light_pos = {0.0f, -2.0f, 0.0f, 1.0f},
  },
  .fragment_shader = {
    .height_scale = 0.1f,
    .parallax_bias = -0.02f,
    .num_layers = 48.0f,
    .mapping_mode = 4,
  },
};

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachmentDescriptor rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

WGPUPipelineLayout pipeline_layout;
WGPURenderPipeline pipeline;
WGPUBindGroupLayout bind_group_layout;
WGPUBindGroup bind_group;

static const char* mapping_modes[5] = {
  "Color only",                 //
  "Normal mapping",             //
  "Parallax mapping",           //
  "Steep parallax mapping",     //
  "Parallax occlusion mapping", //
};

// Other variables
static const char* example_title = "Parallax Mapping";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  const float aspect_ratio = (float)context->wgpu_context->surface.width
                             / (float)context->wgpu_context->surface.height;

  context->timer_speed *= 0.5f;
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -2.5f});
  camera_set_rotation(context->camera, (vec3){0.0f, 15.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f, aspect_ratio, 0.0f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  textures.normal_height_map = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/rocks_normal_height_rgba.ktx");
  textures.color_map = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/rocks_color_rgba.ktx");
}

static void generate_quad(wgpu_context_t* wgpu_context)
{
  // Setup vertices for a single uv-mapped quad made from two triangles
  static const vertex_t vertices_data[4] = {
    [0] = {
      .pos     = {1.0f, 1.0f, 0.0f},
      .uv      = {1.0f, 1.0f},
      .normal  = {0.0f, 0.0f, 1.0f},
      .tangent = {1.0f, 0, 0, 0},
    },
    [1] = {
      .pos     = {-1.0f, 1.0f, 0.0f},
      .uv      = {0.0f, 1.0f},
      .normal  = {0.0f, 0.0f, 1.0f},
      .tangent = {1.0f, 0, 0, 0},
    },
    [2] = {
      .pos     = {-1.0f, -1.0f, 0.0f},
      .uv      = {0.0f, 0.0f},
      .normal  = {0.0f, 0.0f, 1.0f},
      .tangent = {1.0f, 0, 0, 0},
    },
    [3] = {
      .pos     = {1.0f, -1.0f, 0.0f},
      .uv      = {1.0f, 0.0f},
      .normal  = {0.0f, 0.0f, 1.0f},
      .tangent = {1.0f, 0, 0, 0},
    },
  };
  vertices.count              = (uint32_t)ARRAY_SIZE(vertices_data);
  uint32_t vertex_buffer_size = vertices.count * sizeof(vertex_t);

  // Setup indices
  static const uint16_t index_buffer[6] = {0, 1, 2, 2, 3, 0};
  indices.count                         = (uint32_t)ARRAY_SIZE(index_buffer);
  uint32_t index_buffer_size            = indices.count * sizeof(uint32_t);

  // Create vertex buffer
  vertices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, vertices_data, vertex_buffer_size, WGPUBufferUsage_Vertex);

  // Create index buffer
  indices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, index_buffer, index_buffer_size, WGPUBufferUsage_Index);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[6] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (Vertex shader)
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize = sizeof(ubos.vertex_shader),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Fragment shader color map image view
      .binding = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled = false,
      },
      .storageTexture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      // Binding 2: Fragment shader color map image sampler
      .binding = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type=WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      // Binding 3: Fragment combined normal and heightmap view
      .binding = 3,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled = false,
      },
      .storageTexture = {0},
    },
    [4] = (WGPUBindGroupLayoutEntry) {
      // Binding 4: Fragment combined normal and heightmap sampler
      .binding = 4,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type=WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [5] = (WGPUBindGroupLayoutEntry) {
      // Binding 5: Fragment shader uniform buffer
      .binding = 5,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize = sizeof(ubos.fragment_shader),
      },
      .sampler = {0},
    },
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL)

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this descriptor set layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL)
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  // Bind Group
  WGPUBindGroupEntry bg_entries[6] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0: Uniform buffer (Vertex shader)
      .binding = 0,
      .buffer = uniform_buffers.vertex_shader,
      .offset = 0,
      .size = sizeof(ubos.vertex_shader),
    },
    [1] = (WGPUBindGroupEntry) {
      // Binding 1: Fragment shader color map image view
      .binding = 1,
      .textureView = textures.color_map.view,
    },
    [2] = (WGPUBindGroupEntry) {
      // Binding 2: Fragment shader color map image sampler
      .binding = 2,
      .sampler = textures.color_map.sampler,
    },
    [3] = (WGPUBindGroupEntry) {
      // Binding 3: Fragment combined normal and heightmap view
      .binding = 3,
      .textureView = textures.normal_height_map.view,
    },
    [4] = (WGPUBindGroupEntry) {
      // Binding 4: Fragment combined normal and heightmap sampler
      .binding = 4,
      .sampler = textures.normal_height_map.sampler,
    },
    [5] = (WGPUBindGroupEntry) {
      // Binding 5: Fragment shader uniform buffer
      .binding = 5,
      .buffer = uniform_buffers.fragment_shader,
      .offset = 0,
      .size = sizeof(ubos.fragment_shader),
    },
  };

  bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .layout     = bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(bind_group != NULL)
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachmentDescriptor) {
      .view       = NULL,
      .attachment = NULL,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context);

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Construct the different states making up the pipeline

  // Rasterization state
  WGPURasterizationStateDescriptor rasterization_state
    = wgpu_create_rasterization_state_descriptor(
      &(create_rasterization_state_desc_t){
        .front_face = WGPUFrontFace_CW,
        .cull_mode  = WGPUCullMode_None,
      });

  // Color blend state
  WGPUColorStateDescriptor color_state_desc
    = wgpu_create_color_state_descriptor(&(create_color_state_desc_t){
      .format       = wgpu_context->swap_chain.format,
      .enable_blend = true,
    });

  // Depth and stencil state containing depth and stencil compare and test
  // operations
  WGPUDepthStencilStateDescriptor depth_stencil_state_desc
    = wgpu_create_depth_stencil_state_descriptor(
      &(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24PlusStencil8,
        .depth_write_enabled = true,
      });

  // Vertex input binding (=> Input assembly) description
  WGPU_VERTSTATE(
    quad, sizeof(vertex_t),
    /* Attribute descriptions */
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    // Attribute location 1: Texture coordinates
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, offsetof(vertex_t, uv)),
    // Attribute location 2: Vertex normal
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, normal)),
    // Attribute location 3: Vertex tangent
    WGPU_VERTATTR_DESC(3, WGPUVertexFormat_Float32x4,
                       offsetof(vertex_t, tangent)))

  // Shaders
  // Vertex shader
  wgpu_shader_t vert_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Vertex shader SPIR-V
                    .file = "shaders/parallax_mapping/parallax.vert.spv",
                  });
  // Fragment shader
  wgpu_shader_t frag_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Fragment shader SPIR-V
                    .file = "shaders/parallax_mapping/parallax.frag.spv",
                  });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .layout = pipeline_layout,
      // Vertex shader
      .vertexStage = vert_shader.programmable_stage_descriptor,
      // Fragment shader
      .fragmentStage = &frag_shader.programmable_stage_descriptor,
      // Rasterization state
      .rasterizationState     = &rasterization_state,
      .primitiveTopology      = WGPUPrimitiveTopology_TriangleList,
      .colorStateCount        = 1,
      .colorStates            = &color_state_desc,
      .depthStencilState      = &depth_stencil_state_desc,
      .vertexState            = &vert_state_quad,
      .sampleCount            = 1,
      .sampleMask             = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    });

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  wgpu_shader_release(&frag_shader);
  wgpu_shader_release(&vert_shader);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Vertex shader
  glm_mat4_copy(context->camera->matrices.perspective,
                ubos.vertex_shader.projection);
  glm_mat4_copy(context->camera->matrices.view, ubos.vertex_shader.view);
  glm_mat4_identity(ubos.vertex_shader.model);
  glm_scale(ubos.vertex_shader.model, (vec3){0.2f, 0.2f, 0.2f});

  if (!context->paused) {
    ubos.vertex_shader.light_pos[0]
      = sin(glm_rad(context->timer * 360.0f)) * 1.5f;
    ubos.vertex_shader.light_pos[2]
      = cos(glm_rad(context->timer * 360.0f)) * 1.5f;
  }

  glm_vec4(context->camera->position, -1.0f, ubos.vertex_shader.camera_pos);
  glm_vec4_scale(ubos.vertex_shader.camera_pos, -1.0f,
                 ubos.vertex_shader.camera_pos);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.vertex_shader,
                          0, &ubos.vertex_shader, sizeof(ubos.vertex_shader));

  // Fragment shader
  wgpu_queue_write_buffer(context->wgpu_context,
                          uniform_buffers.fragment_shader, 0,
                          &ubos.fragment_shader, sizeof(ubos.fragment_shader));
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Vertex shader uniform buffer
  {
    WGPUBufferDescriptor ubo_desc = {
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = sizeof(ubos.vertex_shader),
      .mappedAtCreation = false,
    };
    uniform_buffers.vertex_shader
      = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_desc);
  }

  // Fragment shader uniform buffer
  {
    WGPUBufferDescriptor ubo_desc = {
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = sizeof(ubos.fragment_shader),
      .mappedAtCreation = false,
    };
    uniform_buffers.fragment_shader
      = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_desc);
  }

  update_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    generate_quad(context->wgpu_context);
    load_assets(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    if (imgui_overlay_combo_box(context->imgui_overlay, "Mode",
                                &ubos.fragment_shader.mapping_mode,
                                mapping_modes, 5)) {
      update_uniform_buffers(context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  // Set the bind group
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  // Set viewport
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Bind triangle vertex buffer (contains position and colors)
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, 0);

  // Bind triangle index buffer
  wgpuRenderPassEncoderSetIndexBuffer(wgpu_context->rpass_enc, indices.buffer,
                                      WGPUIndexFormat_Uint16, 0, 0);

  // Draw indexed triangle
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, indices.count, 1, 0,
                                   0, 0);

  // End render pass
  wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
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
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  wgpu_destroy_texture(&textures.color_map);
  wgpu_destroy_texture(&textures.normal_height_map);
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.vertex_shader)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.fragment_shader)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
}

void example_parallax_mapping(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
