#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - glTF Scene Rendering
 *
 * Renders a complete scene loaded from an glTF 2.0 file. The sample uses the
 * glTF model loading functions, and adds data structures, functions and shaders
 * required to render a more complex scene using Crytek's Sponza model with
 * per-material pipelines and normal mapping.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/gltfscenerendering/gltfscenerendering.cpp
 * -------------------------------------------------------------------------- */

static struct gltf_model_t* gltf_model = NULL;

static struct {
  mat4 projection;
  mat4 view;
  vec4 light_pos;
  vec4 view_pos;
} ubo_scene = {
  .light_pos = {0.0f, 5.0f, 0.0f, 1.0f},
};

typedef struct {
  bool alpha_mask;
  float alpha_mask_cutoff;
} ubo_material_consts_t;

static struct {
  wgpu_buffer_t ubo_scene;
  struct {
    WGPUBuffer* buffers;
    uint32_t buffer_count;
  } ubo_material_consts;
} ubo_buffers = {0};

static struct {
  WGPUBindGroupLayout ubo_scene;
  WGPUBindGroupLayout ubo_primitive;
  WGPUBindGroupLayout textures;
} bind_group_layouts = {0};

static struct {
  WGPUBindGroup ubo_scene;
} bind_groups = {0};

static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};
static WGPUPipelineLayout pipeline_layout                        = NULL;

// Other variables
static const char* example_title = "glTF Scene Rendering";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera         = camera_create();
  context->camera->type   = CameraType_FirstPerson;
  context->camera->flip_y = false;
  camera_set_position(context->camera, (vec3){0.0f, 1.0f, 0.0f});
  camera_set_rotation(context->camera, (vec3){0.0f, -90.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags = WGPU_GLTF_FileLoadingFlags_None;
  gltf_model = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/Sponza/glTF/Sponza.gltf",
    .file_loading_flags = gltf_loading_flags,
  });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /*
   * This sample uses separate descriptor sets (and layouts) for the matrices
   * and materials (textures)
   */

  /* Bind group layout to pass scene data to the shader */
  {
    WGPUBindGroupLayoutEntry bgl_entry= {
        /* Binding 0: Uniform buffer (Vertex shader) => UBOScene */
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(ubo_scene),
        },
        .sampler = {0},
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Scene data - Bind group layout",
      .entryCount = 1,
      .entries    = &bgl_entry,
    };
    bind_group_layouts.ubo_scene
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.ubo_scene != NULL);
  }

  /* Bind group layout to pass mesh matrix to the shader */
  {
    WGPUBindGroupLayoutEntry bgl_entry = {
        /* Binding 0: Uniform buffer (Vertex shader) => Primitive */
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(mat4),
        },
        .texture = {0},
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Mesh matrix - Bind group layout",
      .entryCount = 1,
      .entries    = &bgl_entry,
    };
    bind_group_layouts.ubo_primitive
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.ubo_primitive != NULL);
  }

  /* Bind group layout for passing material textures and material constants */
  {
    WGPUBindGroupLayoutEntry bgl_entries[5] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: texture2D (Fragment shader) => Color map */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: sampler (Fragment shader) => Color map */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2: texture2D (Fragment shader) => Normal map */
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        /* Binding 3: sampler (Fragment shader) => Normal map */
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        /* Binding 4: Uniform buffer (Fragment shader) => MaterialConsts */
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(ubo_material_consts_t),
        },
        .texture = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Material - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    bind_group_layouts.textures
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.textures != NULL);
  }

  /*/ Pipeline layout using the bind group layouts */
  {
    WGPUBindGroupLayout bind_group_layout_sets[3] = {
      bind_group_layouts.ubo_scene,     /* set 0 */
      bind_group_layouts.textures,      /* set 1 */
      bind_group_layouts.ubo_primitive, /* set 2 */
    };
    /* Pipeline layout */
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "Pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layout_sets),
      .bindGroupLayouts     = bind_group_layout_sets,
    };
    pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                     &pipeline_layout_desc);
    ASSERT(pipeline_layout != NULL)
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Bind group for scene matrices */
  {
    WGPUBindGroupEntry bg_entry = {
      /* Binding 0: Uniform buffer (Vertex shader) => UBOScene */
      .binding = 0,
      .buffer  = ubo_buffers.ubo_scene.buffer,
      .offset  = 0,
      .size    = ubo_buffers.ubo_scene.size,
    };
    bind_groups.ubo_scene = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Scene matrices - Bind group",
                              .layout     = bind_group_layouts.ubo_scene,
                              .entryCount = 1,
                              .entries    = &bg_entry,
                            });
    ASSERT(bind_groups.ubo_scene != NULL)
  }

  /* Bind group for glTF model meshes */
  {
    wgpu_gltf_model_prepare_nodes_bind_group(gltf_model,
                                             bind_group_layouts.ubo_primitive);
  }

  /* Bind group for materials */
  {
    wgpu_gltf_materials_t materials = wgpu_gltf_model_get_materials(gltf_model);
    for (uint32_t i = 0; i < materials.material_count; ++i) {
      wgpu_gltf_material_t* material = &materials.materials[i];
      if (material->base_color_texture && material->normal_texture) {
        WGPUBindGroupEntry bg_entries[5] = {
            [0] = (WGPUBindGroupEntry) {
              /* Binding 0: texture2D (Fragment shader) => Color map */
              .binding     = 0,
              .textureView = material->base_color_texture->wgpu_texture.view,
            },
            [1] = (WGPUBindGroupEntry) {
              /* Binding 1: sampler (Fragment shader) => Color map */
              .binding = 1,
              .sampler = material->base_color_texture->wgpu_texture.sampler,
            },
            [2] = (WGPUBindGroupEntry) {
              /* Binding 2: texture2D (Fragment shader) => Normal map */
              .binding     = 2,
              .textureView = material->normal_texture->wgpu_texture.view,
            },
            [3] = (WGPUBindGroupEntry) {
              /* Binding 3: sampler (Fragment shader) => Normal map */
              .binding = 3,
              .sampler =  material->normal_texture->wgpu_texture.sampler,
            },
            [4] = (WGPUBindGroupEntry) {
              /* Binding 4: Uniform buffer (Fragment shader) => MaterialConsts */
              .binding = 4,
              .buffer  = ubo_buffers.ubo_material_consts.buffers[i],
              .offset  = 0,
              .size    = sizeof(ubo_material_consts_t),
            },
          };
        material->bind_group = wgpuDeviceCreateBindGroup(
          wgpu_context->device,
          &(WGPUBindGroupDescriptor){
            .label      = "Materials - Bind group",
            .layout     = bind_group_layouts.textures,
            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
            .entries    = bg_entries,
          });
        ASSERT(material->bind_group != NULL)
      }
    }
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.25f,
        .g = 0.25f,
        .b = 0.25f,
        .a = 1.0f,
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
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

  /* Vertex buffer layout */
  WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
    gltf_scene,
    /* Location 0: Position */
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    /* Location 1: Vertex normal */
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal),
    /* Location 2: Texture coordinates */
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_UV),
    /* Location 3: Vertex color */
    WGPU_GLTF_VERTATTR_DESC(3, WGPU_GLTF_VertexComponent_Color),
    /* Location 4: Vertex tangent */
    WGPU_GLTF_VERTATTR_DESC(4, WGPU_GLTF_VertexComponent_Tangent));

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader SPIR-V */
              .label = "glTF scene rendering - Vertex shader SPIR-V",
              .file  = "shaders/gltf_scene_rendering/scene.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &gltf_scene_vertex_buffer_layout,
          });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader SPIR-V */
              .label = "glTF scene rendering - Fragment shader SPIR-V",
              .file  = "shaders/gltf_scene_rendering/scene.frag.spv",
            },
            .target_count = 1,
            .targets      = &color_target_state,
          });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Render pipeline descriptor */
  WGPURenderPipelineDescriptor render_pipeline_descriptor = {
    .label        = "glTF scene - Rendering pipeline",
    .layout       = pipeline_layout,
    .primitive    = primitive_state,
    .vertex       = vertex_state,
    .fragment     = &fragment_state,
    .depthStencil = &depth_stencil_state,
    .multisample  = multisample_state,
  };

  // Instead of using a few fixed pipelines, we create one pipeline for each
  // material using the properties of that material
  wgpu_gltf_materials_t materials = wgpu_gltf_model_get_materials(gltf_model);
  for (uint32_t i = 0; i < materials.material_count; ++i) {
    wgpu_gltf_material_t* material = &materials.materials[i];
    /* For double sided materials, culling will be disabled */
    WGPUPrimitiveState* primitive_desc = &render_pipeline_descriptor.primitive;
    primitive_desc->cullMode
      = material->double_sided ? WGPUCullMode_None : WGPUCullMode_Back;
    material->pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &render_pipeline_descriptor);
    ASSERT(material->pipeline != NULL)
  }

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  camera_t* camera = context->camera;
  glm_mat4_copy(camera->matrices.perspective, ubo_scene.projection);
  glm_mat4_copy(camera->matrices.view, ubo_scene.view);
  glm_vec4_copy(camera->view_pos, ubo_scene.view_pos);
  wgpu_queue_write_buffer(context->wgpu_context, ubo_buffers.ubo_scene.buffer,
                          0, &ubo_scene, ubo_buffers.ubo_scene.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Scene uniform buffer */
  {
    ubo_buffers.ubo_scene = wgpu_create_buffer(
      context->wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Scene - Uniform buffer",
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
        .size  = sizeof(ubo_scene),
      });
    update_uniform_buffers(context);
  }

  /* Material constants uniform buffer */
  {
    wgpu_gltf_materials_t materials = wgpu_gltf_model_get_materials(gltf_model);
    ubo_buffers.ubo_material_consts.buffer_count = materials.material_count;
    ubo_buffers.ubo_material_consts.buffers
      = (WGPUBuffer*)calloc(materials.material_count, sizeof(WGPUBuffer));
    for (uint32_t i = 0; i < materials.material_count; ++i) {
      wgpu_gltf_material_t* material            = &materials.materials[i];
      ubo_material_consts_t ubo_material_consts = {
        .alpha_mask        = material->alpha_mode == AlphaMode_MASK,
        .alpha_mask_cutoff = material->alpha_cutoff,
      };
      ubo_buffers.ubo_material_consts.buffers[i] = wgpu_create_buffer_from_data(
        context->wgpu_context, &ubo_material_consts,
        sizeof(ubo_material_consts_t), WGPUBufferUsage_Uniform);
    }
  }
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.ubo_scene, 0, 0);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Draw plane */
  static wgpu_gltf_render_flags_enum_t render_flags
    = WGPU_GLTF_RenderFlags_BindImages;
  wgpu_gltf_model_draw(gltf_model, (wgpu_gltf_model_render_options_t){
                                     .render_flags        = render_flags,
                                     .bind_image_set      = 1,
                                     .bind_mesh_model_set = 2,
                                   });

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

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
  return example_draw(context);
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  wgpu_gltf_model_destroy(gltf_model);

  WGPU_RELEASE_RESOURCE(Buffer, ubo_buffers.ubo_scene.buffer)
  for (uint32_t i = 0; i < ubo_buffers.ubo_material_consts.buffer_count; ++i) {
    WGPU_RELEASE_RESOURCE(Buffer, ubo_buffers.ubo_material_consts.buffers[i])
  }
  free(ubo_buffers.ubo_material_consts.buffers);

  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.ubo_scene)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.ubo_primitive)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.textures)

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.ubo_scene)

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
}

void example_gltf_scene_rendering(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
