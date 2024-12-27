#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - glTF Vertex Skinning
 *
 * Shows how to load and display an animated scene from a glTF file using vertex
 * skinning.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/gltfskinning/gltfskinning.cpp
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* skinned_model_vertex_shader_wgsl;
static const char* skinned_model_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * glTF Vertex Skinning example
 * -------------------------------------------------------------------------- */

static struct gltf_model_t* gltf_model;

static struct {
  wgpu_buffer_t ubo_scene;
  struct {
    mat4 projection;
    mat4 view;
    vec4 light_pos;
  } ubo_scene_values;
} shader_data = {
  .ubo_scene_values.projection = GLM_MAT4_IDENTITY_INIT,
  .ubo_scene_values.view       = GLM_MAT4_IDENTITY_INIT,
  .ubo_scene_values.light_pos  = {5.0f, 5.0f, 5.0f, 1.0f},
};

static struct {
  WGPUBindGroupLayout ubo_scene;
  WGPUBindGroupLayout ubo_primitive;
  WGPUBindGroupLayout textures;
} bind_group_layouts;

static struct bind_group_t {
  WGPUBindGroup ubo_scene;
} bind_groups;

static WGPUPipelineLayout pipeline_layout;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass;

// Other variables
static const char* example_title = "glTF Vertex Skinning";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera         = camera_create();
  context->camera->type   = CameraType_LookAt;
  context->camera->flip_y = true;
  camera_set_position(context->camera, (vec3){0.0f, 0.75f, -2.0f});
  camera_set_rotation(context->camera, (vec3){90.0f, 180.0f, 90.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  const uint32_t gltf_loading_flags = WGPU_GLTF_FileLoadingFlags_None;
  gltf_model = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
    .wgpu_context       = wgpu_context,
    .filename           = "models/CesiumMan/glTF/CesiumMan.gltf",
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
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Vertex shader) => UBOScene
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(shader_data.ubo_scene_values),
        },
        .sampler = {0},
        },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "UBOScene - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    bind_group_layouts.ubo_scene
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.ubo_scene != NULL);
  }

  /* Bind group layout to pass the local matrices of a primitive to the shader
   */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Vertex shader) => UBOPrimitive
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout){
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(mat4),
        },
        .texture = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "UBOPrimitive - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    bind_group_layouts.ubo_primitive
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.ubo_primitive != NULL);
  }

  /* Bind group layout for passing material textures */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: texture2D (Fragment shader) => Color map
        .binding = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: sampler (Fragment shader) => Color map
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "Texture - Bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    bind_group_layouts.textures
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(bind_group_layouts.textures != NULL);
  }

  /* Pipeline layout using the bind group layouts */
  {
    // The pipeline layout uses three sets:
    // Set 0 = Scene matrices (VS)
    // Set 1 = Primitive matrices (VS)
    // Set 2 = Material texture (FS)
    WGPUBindGroupLayout bind_group_layout_sets[3] = {
      bind_group_layouts.ubo_scene,     /* set 0 */
      bind_group_layouts.ubo_primitive, /* set 1 */
      bind_group_layouts.textures,      /* set 2 */
    };
    // Pipeline layout
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

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Pass matrices to the shaders
  camera_t* camera = context->camera;
  glm_mat4_copy(camera->matrices.perspective,
                shader_data.ubo_scene_values.projection);
  glm_mat4_copy(camera->matrices.view, shader_data.ubo_scene_values.view);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, shader_data.ubo_scene.buffer,
                          0, &shader_data.ubo_scene_values,
                          shader_data.ubo_scene.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Scene uniform buffer
  shader_data.ubo_scene = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Scene vertex shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(shader_data.ubo_scene_values),
    });

  // Initialize uniform buffers
  update_uniform_buffers(context);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Bind group for scene matrices
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Uniform buffer (Vertex shader) => UBOScene
        .binding = 0,
        .buffer  = shader_data.ubo_scene.buffer,
        .offset  = 0,
        .size    = shader_data.ubo_scene.size,
      },
    };
    bind_groups.ubo_scene = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Scene matrices - Bind group",
                              .layout     = bind_group_layouts.ubo_scene,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.ubo_scene != NULL)
  }

  // Bind group for glTF model meshes
  {
    wgpu_gltf_model_prepare_nodes_bind_group(gltf_model,
                                             bind_group_layouts.ubo_primitive);
  }

  // Bind group for materials
  {
    wgpu_gltf_materials_t materials = wgpu_gltf_model_get_materials(gltf_model);
    for (uint32_t i = 0; i < materials.material_count; ++i) {
      wgpu_gltf_material_t* material = &materials.materials[i];
      if (material->base_color_texture) {
        WGPUBindGroupEntry bg_entries[2] = {
            [0] = (WGPUBindGroupEntry) {
              // Binding 0: texture2D (Fragment shader) => Color map
              .binding = 0,
              .textureView = material->base_color_texture->wgpu_texture.view,
            },
            [1] = (WGPUBindGroupEntry) {
              // Binding 1: sampler (Fragment shader) => Color map
              .binding = 1,
              .sampler =  material->base_color_texture->wgpu_texture.sampler,
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
  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL,
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.025f,
        .g = 0.025f,
        .b = 0.025f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Set clear sample for this example
  wgpu_context->depth_stencil.att_desc.depthClearValue = 1;

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  // Vertex buffer layout
  WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
    gltf_scene,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal),
    // Location 2: Texture coordinates
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_UV),
    // Location 3: Vertex color
    WGPU_GLTF_VERTATTR_DESC(3, WGPU_GLTF_VertexComponent_Color),
    // Location 4: Per-Vertex Joint indices
    WGPU_GLTF_VERTATTR_DESC(4, WGPU_GLTF_VertexComponent_Joint0),
    // Location 5: Per-Vertex Joint weights
    WGPU_GLTF_VERTATTR_DESC(5, WGPU_GLTF_VertexComponent_Weight0));

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader WGSL
              .label            = "GLTF skinned model - Vertex shader WGSL",
              .wgsl_code.source = skinned_model_vertex_shader_wgsl,
              .entry            = "main",
            },
            .buffer_count = 1,
            .buffers = &gltf_scene_vertex_buffer_layout,
          });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader WGSL
              .label            = "GLTF skinned model - Fragment shader WGSL",
              .wgsl_code.source = skinned_model_fragment_shader_wgsl,
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

  // Render pipeline descriptor
  WGPURenderPipelineDescriptor render_pipeline_descriptor = {
    .label        = "GLTF skinning - Render pipeline",
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
    // For double sided materials, culling will be disabled
    WGPUPrimitiveState* primitive_desc = &render_pipeline_descriptor.primitive;
    primitive_desc->cullMode
      = material->double_sided ? WGPUCullMode_None : WGPUCullMode_Back;
    material->pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &render_pipeline_descriptor);
    ASSERT(material->pipeline != NULL)
  }

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
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

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Bind scene matrices descriptor to set 0 */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.ubo_scene, 0, 0);

  /* Render GLTF model */
  static wgpu_gltf_render_flags_enum_t render_flags
    = WGPU_GLTF_RenderFlags_BindImages;
  wgpu_gltf_model_draw(gltf_model, (wgpu_gltf_model_render_options_t){
                                     .render_flags        = render_flags,
                                     .bind_mesh_model_set = 1,
                                     .bind_image_set      = 2,
                                   });

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

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  int draw_result = example_draw(context);
  if (!context->paused) {
    // gltf_model_update_animation(gltf_model, 0, context->frame_timer);
  }
  return draw_result;
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_uniform_buffers(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  wgpu_gltf_model_destroy(gltf_model);

  WGPU_RELEASE_RESOURCE(Buffer, shader_data.ubo_scene.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.ubo_scene)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.ubo_primitive)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.textures)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.ubo_scene)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
}

void example_gltf_skinning(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* skinned_model_vertex_shader_wgsl = CODE(
  struct UBOScene {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
    lightPos : vec4<f32>,
  };

  struct UBOPrimitive {
    model : mat4x4<f32>,
  };

  @group(0) @binding(0) var<uniform> uboScene : UBOScene;
  @group(1) @binding(0) var<uniform> primitive : UBOPrimitive;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) outNormal : vec3<f32>,
    @location(1) outColor : vec3<f32>,
    @location(2) outUV : vec2<f32>,
    @location(3) outViewVec : vec3<f32>,
    @location(4) outLightVec : vec3<f32>,
  };

  @vertex
  fn main(
    @location(0) inPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) inUV: vec2<f32>,
    @location(3) inColor: vec3<f32>,
    @location(4) inJointIndices: vec4<f32>,
    @location(5) inJointWeights: vec4<f32>
  ) -> Output {
    var output: Output;
    output.position = uboScene.projection * uboScene.view * primitive.model * vec4<f32>(inPos, 1.0);
    output.outNormal = mat3x3(
        primitive.model[0].xyz,
        primitive.model[1].xyz,
        primitive.model[2].xyz,
      ) * inNormal;
    output.outColor = inColor;
    output.outUV = inUV;
    let pos : vec4<f32> = primitive.model * vec4<f32>(inPos, 1.0);
    output.outLightVec = uboScene.lightPos.xyz - pos.xyz;
    output.outViewVec = -pos.xyz;
    return output;
  }
);

static const char* skinned_model_fragment_shader_wgsl = CODE(
  @group(2) @binding(0) var textureColorMap : texture_2d<f32>;
  @group(2) @binding(1) var samplerColorMap : sampler;

  @fragment
  fn main(
    @location(0) inNormal : vec3<f32>,
    @location(1) inColor : vec3<f32>,
    @location(2) inUV : vec2<f32>,
    @location(3) inViewVec : vec3<f32>,
    @location(4) inLightVec : vec3<f32>,
  ) -> @location(0) vec4<f32> {
    let textureColor : vec3<f32> = (textureSample(textureColorMap, samplerColorMap, inUV)).rgb;
    let color : vec4<f32> = vec4(textureColor, 1.0) * vec4(inColor, 1.0);
    let N : vec3<f32> = normalize(inNormal);
    let L : vec3<f32> = normalize(inLightVec);
    let V : vec3<f32> = normalize(inViewVec);
    let R : vec3<f32> = reflect(-L, N);
    let diffuse : vec3<f32> = max(dot(N, L), 0.5) * inColor;
    let specular : vec3<f32> = pow(max(dot(R, V), 0.0), 16.0) * vec3<f32>(0.75);
    return vec4<f32>(diffuse * color.rgb + specular, 1.0);
  }
);
// clang-format on
