#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"
#include "../webgpu/pbr.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Physical Based Rendering With Image Based Lighting
 *
 * Adds image based lighting from an hdr environment cubemap to the PBR
 * equation, using the surrounding environment as the light source. This adds an
 * even more realistic look the scene as the light contribution used by the
 * materials is now controlled by the environment. Also shows how to generate
 * the BRDF 2D-LUT and irradiance and filtered cube maps from the environment
 * map.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/pbribl
 * http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* pbr_ibl_vertex_shader_wgsl;
static const char* pbr_ibl_functions_fragment_shader_wgsl;
static const char* pbr_ibl_main_fragment_shader_wgsl;
static const char* skybox_vertex_shader_wgsl;
static const char* skybox_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Physical Based Rendering With Image Based Lighting example
 * -------------------------------------------------------------------------- */

#define SINGLE_ROW_OBJECT_COUNT 10u
#define ALIGNMENT 256u // 256-byte alignment

static bool display_skybox = true;

static struct {
  texture_t environment_cube;
  // Generated at runtime
  texture_t lut_brdf;
  texture_t irradiance_cube;
  texture_t prefiltered_cube;
} textures = {0};

static struct {
  struct gltf_model_t* skybox;
  struct {
    const char* name;
    const char* filelocation;
    struct gltf_model_t* object;
  } objects[4];
  int32_t object_index;
} models = {
  // clang-format off
  .objects = {
    { .name = "Sphere",    .filelocation = "models/sphere.gltf" },
    { .name = "Teapot",    .filelocation = "models/teapot.gltf" },
    { .name = "Torusknot", .filelocation = "models/torusknot.gltf" },
    { .name = "Venus",     .filelocation = "models/venus.gltf" },
  },
  // clang-format on
  .object_index = 0,
};

static struct {
  // Object vertex shader uniform buffer
  wgpu_buffer_t object;
  // Skybox vertex shader uniform buffer
  wgpu_buffer_t skybox;
  // Shared parameter uniform buffer
  wgpu_buffer_t ubo_params;
  // Material parameter uniform buffer
  struct {
    WGPUBuffer buffer;
    uint64_t buffer_size;
    uint64_t model_size;
  } material_params;
  // Object parameter uniform buffer
  struct {
    WGPUBuffer buffer;
    uint64_t buffer_size;
    uint64_t model_size;
  } object_params;
} uniform_buffers = {0};

static struct {
  mat4 projection;
  mat4 model;
  mat4 view;
  vec3 cam_pos;
} ubo_matrices = {0};

static struct {
  vec4 lights[4];
  float exposure;
  float gamma;
} ubo_params = {
  .exposure = 4.5f,
  .gamma    = 2.2f,
};

static struct matrial_params_dynamic_t {
  float roughness;
  float metallic;
  float specular;
  vec3 color;
  uint8_t padding[232];
} material_params_dynamic[SINGLE_ROW_OBJECT_COUNT] = {0};

static struct object_params_dynamic_t {
  vec3 position;
  uint8_t padding[244];
} object_params_dynamic[SINGLE_ROW_OBJECT_COUNT] = {0};

static struct {
  WGPURenderPipeline pbr;
  WGPURenderPipeline skybox;
} pipelines = {0};

static struct {
  WGPUBindGroup objects;
  WGPUBindGroup skybox;
} bind_groups = {0};

static struct {
  WGPUBindGroupLayout objects;
  WGPUBindGroupLayout skybox;
} bind_group_layouts = {0};

static struct {
  WGPUPipelineLayout pbr;
  WGPUPipelineLayout skybox;
} pipeline_layouts = {0};

static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Default materials to select from
static struct {
  const char* name;
  // Parameter block used as uniforms block
  struct {
    float roughness;
    float metallic;
    float specular;
    vec3 color;
  } params;
} materials[12] = {
  // clang-format off
  // Setup some default materials (source:
  // https://seblagarde.wordpress.com/2011/08/17/feeding-a-physical-based-lighting-mode/)
  { .name = "Gold",     .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 1.000000f, 0.765557f, 0.336057f } } },
  { .name = "Copper",   .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 0.955008f, 0.637427f, 0.538163f } } },
  { .name = "Chromium", .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 0.549585f, 0.556114f, 0.554256f } } },
  { .name = "Nickel",   .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 0.659777f, 0.608679f, 0.525649f } } },
  { .name = "Titanium", .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 0.541931f, 0.496791f, 0.449419f } } },
  { .name = "Cobalt",   .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 0.662124f, 0.654864f, 0.633732f } } },
  { .name = "Platinum", .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 0.672411f, 0.637331f, 0.585456f } } },
  // Testing materials
  { .name = "White",    .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 1.000000f, 1.000000f, 1.000000f } } },
  { .name = "Dark",     .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 0.100000f, 0.100000f, 0.100000f } } },
  { .name = "Black",    .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 0.000000f, 0.000000f, 0.000000f } } },
  { .name = "Red",      .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 1.000000f, 0.000000f, 0.000000f } } },
  { .name = "Blue",     .params = { .roughness = 0.0f, .metallic = 0.0f, .specular = 0.0f, .color = { 0.000000f, 0.000000f, 1.000000f } } },
  // clang-format on
};
static int32_t current_material_index = 4;

// Arrays used for GUI
static const char* material_names[12] = {
  // Default materials
  "Gold", "Copper", "Chromium", "Nickel", "Titanium", "Cobalt", "Platinum", //
  // Testing materials
  "White", "Dark", "Black", "Red", "Blue", //
};
static const char* object_names[4] = {"Sphere", "Teapot", "Torusknot", "Venus"};

static const char* example_title = "PBR With Image Based Lighting";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_FirstPerson;
  camera_set_movement_speed(context->camera, 4.0f);
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
  camera_set_rotation_speed(context->camera, 0.25f);

  camera_set_rotation(context->camera, (vec3){-3.75f, 180.0f, 0.0f});
  camera_set_position(context->camera, (vec3){0.55f, 0.85f, 12.0f});
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  // Load glTF models
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_DontLoadImages;
  // Skybox
  models.skybox
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/cube.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
  // Objects
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(models.objects); ++i) {
    models.objects[i].object
      = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
        .wgpu_context       = wgpu_context,
        .filename           = models.objects[i].filelocation,
        .file_loading_flags = gltf_loading_flags,
      });
  }
  // Cube map
  static const char* cubemap[6] = {
    "textures/cubemaps/pisa_cube_px.png", /* Right  */
    "textures/cubemaps/pisa_cube_nx.png", /* Left   */
    "textures/cubemaps/pisa_cube_py.png", /* Top    */
    "textures/cubemaps/pisa_cube_ny.png", /* Bottom */
    "textures/cubemaps/pisa_cube_pz.png", /* Back   */
    "textures/cubemaps/pisa_cube_nz.png", /* Front  */
  };
  textures.environment_cube = wgpu_create_texture_cubemap_from_files(
    wgpu_context, cubemap,
    &(struct wgpu_texture_load_options_t){
      .flip_y = true, /* Flip y to match pisa_cube.ktx hdr cubemap */
    });
  ASSERT(textures.environment_cube.texture)
}

static void setup_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Bind group layout for objects */
  {
    WGPUBindGroupLayoutEntry bgl_entries[10] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Vertex shader & Fragment shader)
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = uniform_buffers.object.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Uniform buffer (Fragment shader)
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = uniform_buffers.ubo_params.size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Dynamic uniform buffer (Fragment shader)
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = uniform_buffers.material_params.model_size,
        },
        .sampler = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        // Binding 3: Dynamic uniform buffer (Vertex shader)
        .binding    = 3,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = uniform_buffers.object_params.model_size,
        },
        .sampler = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        // Binding 4: Fragment shader image view
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [5] = (WGPUBindGroupLayoutEntry) {
        // Binding 5: Fragment shader image sampler
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [6] = (WGPUBindGroupLayoutEntry) {
        // Binding 6: Fragment shader image view
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [7] = (WGPUBindGroupLayoutEntry) {
        // Binding 7: Fragment shader image sampler
        .binding    = 7,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [8] = (WGPUBindGroupLayoutEntry) {
        // Binding 8: Fragment shader image view
        .binding    = 8,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [9] = (WGPUBindGroupLayoutEntry) {
        // Binding 9: Fragment shader image sampler
        .binding    = 9,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };
    bind_group_layouts.objects = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Objects - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.objects != NULL);
  }

  /* Bind group layout for skybox */
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Vertex shader uniform UBO
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = uniform_buffers.skybox.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Fragment uniform UBOParams
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = uniform_buffers.ubo_params.size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Fragment shader image view
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        // Binding 3: Fragment shader image sampler
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };

    // Create the bind group layout
    bind_group_layouts.skybox = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Skybox - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.skybox != NULL);
  }
}

static void setup_pipeline_layouts(wgpu_context_t* wgpu_context)
{
  /* Create the pipeline layout for objects */
  {
    pipeline_layouts.pbr = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label                = "PBR - Pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &bind_group_layouts.objects,
                            });
    ASSERT(pipeline_layouts.pbr != NULL);
  }

  /* Create the pipeline layout for skybox */
  {
    pipeline_layouts.skybox = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = "Skybox - Pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &bind_group_layouts.skybox,
                            });
    ASSERT(pipeline_layouts.skybox != NULL);
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Bind group for objects */
  {
    WGPUBindGroupEntry bg_entries[10] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Uniform buffer (Vertex shader & Fragment shader)
        .binding = 0,
        .buffer  = uniform_buffers.object.buffer,
        .offset  = 0,
        .size    = uniform_buffers.object.size,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Uniform buffer (Fragment shader)
        .binding = 1,
        .buffer  = uniform_buffers.ubo_params.buffer,
        .offset  = 0,
        .size    = uniform_buffers.ubo_params.size,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Dynamic uniform buffer (Fragment shader)
        .binding = 2,
        .buffer  = uniform_buffers.material_params.buffer,
        .offset  = 0,
        .size    = uniform_buffers.material_params.model_size,
      },
      [3] = (WGPUBindGroupEntry) {
        // Binding 3: Dynamic uniform buffer (Vertex shader)
        .binding = 3,
        .buffer  = uniform_buffers.object_params.buffer,
        .offset  = 0,
        .size    = uniform_buffers.object_params.model_size,
      },
      [4] = (WGPUBindGroupEntry) {
        // Binding 4: Fragment shader image view
        .binding     = 4,
        .textureView = textures.irradiance_cube.view
      },
      [5] = (WGPUBindGroupEntry) {
        // Binding 5: Fragment shader image sampler
        .binding = 5,
        .sampler = textures.irradiance_cube.sampler,
      },
      [6] = (WGPUBindGroupEntry) {
        // Binding 6: Fragment shader image view
        .binding     = 6,
        .textureView = textures.lut_brdf.view
      },
      [7] = (WGPUBindGroupEntry) {
        // Binding 7: Fragment shader image sampler
        .binding = 7,
        .sampler = textures.lut_brdf.sampler,
      },
      [8] = (WGPUBindGroupEntry) {
        // Binding 8: Fragment shader image view
        .binding     = 8,
        .textureView = textures.prefiltered_cube.view
      },
      [9] = (WGPUBindGroupEntry) {
        // Binding 9: Fragment shader image sampler
        .binding = 9,
        .sampler = textures.prefiltered_cube.sampler,
      },
    };

    bind_groups.objects = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Objects - Bind group",
                              .layout     = bind_group_layouts.objects,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.objects != NULL);
  }

  /* Bind group for skybox */
  {
    WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform UBO
        .binding = 0,
        .buffer  = uniform_buffers.skybox.buffer,
        .offset  = 0,
        .size    = uniform_buffers.skybox.size,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment uniform UBOParams
        .binding = 1,
        .buffer  = uniform_buffers.ubo_params.buffer,
        .offset  = 0,
        .size    = uniform_buffers.ubo_params.size,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader image view
        .binding     = 2,
        .textureView = textures.environment_cube.view
      },
      [3] = (WGPUBindGroupEntry) {
        // Binding 3: Fragment shader image sampler
        .binding = 3,
        .sampler = textures.environment_cube.sampler,
      },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Skybox - Bind group",
      .layout     = bind_group_layouts.skybox,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.skybox
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.skybox != NULL);
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
        .r = 0.1f,
        .g = 0.1f,
        .b = 0.1f,
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

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Construct the different states making up the pipeline */

  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
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
    skybox,
    /* Location 0: Position */
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    /* Location 1: Vertex normal */
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal),
    /* Location 2: UV */
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_UV));

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Skybox pipeline (background cube) */
  {
    primitive_state.cullMode              = WGPUCullMode_Front;
    depth_stencil_state.depthWriteEnabled = false;

    /* Vertex state */
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader WGSL */
              .label            = "Skybox - Vertex shader WGSL",
              .wgsl_code.source = skybox_vertex_shader_wgsl,
              .entry            = "main",
            },
            .buffer_count = 1,
            .buffers      = &skybox_vertex_buffer_layout,
          });

    /* Fragment state */
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader WGSL */
              .label            = "Skybox - Fragment shader WGSL",
              .wgsl_code.source = skybox_fragment_shader_wgsl,
              .entry            = "main",
            },
            .target_count = 1,
            .targets      = &color_target_state,
          });

    /* Create rendering pipeline using the specified states */
    pipelines.skybox = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Skybox - Render pipeline",
                              .layout       = pipeline_layouts.skybox,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.skybox != NULL);

    /* Partial cleanup */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* PBR pipeline */
  {
    primitive_state.cullMode = WGPUCullMode_None;

    /* Enable depth write */
    depth_stencil_state.depthWriteEnabled = true;

    /* Vertex state */
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader WGSL */
              .label            = "PBR IBL - Vertex shader WGSL",
              .wgsl_code.source = pbr_ibl_vertex_shader_wgsl,
              .entry            = "main",
            },
            .buffer_count = 1,
            .buffers      = &skybox_vertex_buffer_layout,
          });

    /* Fragment state */
    char* fragment_shader_wgsl
      = concat_strings(pbr_ibl_functions_fragment_shader_wgsl,
                       pbr_ibl_main_fragment_shader_wgsl, "\n");
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader WGSL */
              .label            = "PBR IBL - Fragment shader WGSL",
              .wgsl_code.source = fragment_shader_wgsl,
              .entry            = "main",
            },
            .target_count = 1,
            .targets      = &color_target_state,
          });
    free(fragment_shader_wgsl);

    /* Create rendering pipeline using the specified states */
    pipelines.pbr = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "PBR IBL - Render pipeline",
                              .layout       = pipeline_layouts.pbr,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.pbr != NULL);

    /* Partial cleanup */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void generate_brdf_lut(wgpu_context_t* wgpu_context)
{
  textures.lut_brdf = pbr_generate_brdf_lut(wgpu_context);
}

static void generate_irradiance_cube(wgpu_context_t* wgpu_context)
{
  textures.irradiance_cube = pbr_generate_irradiance_cube(
    wgpu_context, models.skybox, &textures.environment_cube);
}

static void generate_prefiltered_env_cube(wgpu_context_t* wgpu_context)
{
  textures.prefiltered_cube = pbr_generate_prefiltered_env_cube(
    wgpu_context, models.skybox, &textures.environment_cube);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  /* 3D object */
  camera_t* camera = context->camera;
  glm_mat4_copy(camera->matrices.perspective, ubo_matrices.projection);
  glm_mat4_copy(camera->matrices.view, ubo_matrices.view);
  glm_mat4_identity(ubo_matrices.model);
  glm_rotate(ubo_matrices.model,
             glm_rad(-90.0f + (models.object_index == 1 ? 45.0f : 0.0f)),
             (vec3){0.0f, 1.0f, 0.0f});
  glm_vec3_scale(camera->position, -1.0f, ubo_matrices.cam_pos);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.object.buffer,
                          0, &ubo_matrices, uniform_buffers.object.size);

  /* Skybox */
  mat3 mat3_tmp = GLM_MAT3_ZERO_INIT;
  glm_mat4_pick3(camera->matrices.view, mat3_tmp);
  glm_mat4_ins3(mat3_tmp, ubo_matrices.model);
  ubo_matrices.model[3][3] = 1.0f;
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.skybox.buffer,
                          0, &ubo_matrices, uniform_buffers.skybox.size);
}

static void update_dynamic_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Set objects positions and material properties */
  uint32_t obj_count = (uint32_t)SINGLE_ROW_OBJECT_COUNT;
  for (uint32_t x = 0; x < obj_count; ++x) {
    /* Set object position */
    vec3* pos = &object_params_dynamic[x].position;
    glm_vec3_copy((vec3){((float)(x - (obj_count / 2.0f))) * 2.15f, 0.0f, 0.0f},
                  *pos);
    /* Set material metallic and roughness properties */
    struct matrial_params_dynamic_t* mat_params = &material_params_dynamic[x];
    mat_params->roughness
      = 1.0f - glm_clamp((float)x / (float)obj_count, 0.005f, 1.0f);
    mat_params->metallic = glm_clamp((float)x / (float)obj_count, 0.005f, 1.0f);
    glm_vec3_copy(materials[current_material_index].params.color,
                  (*mat_params).color);
  }

  /* Update buffers */
  wgpu_queue_write_buffer(wgpu_context, uniform_buffers.object_params.buffer, 0,
                          &object_params_dynamic,
                          uniform_buffers.object_params.buffer_size);
  wgpu_queue_write_buffer(wgpu_context, uniform_buffers.material_params.buffer,
                          0, &material_params_dynamic,
                          uniform_buffers.material_params.buffer_size);
}

static void update_params(wgpu_context_t* wgpu_context)
{
  const float p = 15.0f;
  glm_vec4_copy((vec4){-p, -p * 0.5f, -p, 1.0f}, ubo_params.lights[0]);
  glm_vec4_copy((vec4){-p, -p * 0.5f, p, 1.0f}, ubo_params.lights[1]);
  glm_vec4_copy((vec4){p, -p * 0.5f, p, 1.0f}, ubo_params.lights[2]);
  glm_vec4_copy((vec4){p, -p * 0.5f, -p, 1.0f}, ubo_params.lights[3]);

  wgpu_queue_write_buffer(wgpu_context, uniform_buffers.ubo_params.buffer, 0,
                          &ubo_params, uniform_buffers.ubo_params.size);
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Object vertex shader uniform buffer */
  uniform_buffers.object = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Object vertex shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_matrices),
    });

  /* Skybox vertex shader uniform buffer */
  uniform_buffers.skybox = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Skybox vertex shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_matrices),
    });

  /* Shared parameter uniform buffer */
  uniform_buffers.ubo_params = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Shared parameter - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_params),
    });

  /* Material parameter uniform buffer */
  {
    uniform_buffers.material_params.model_size = sizeof(vec3) + sizeof(vec3);
    uniform_buffers.material_params.buffer_size
      = calc_constant_buffer_byte_size(sizeof(material_params_dynamic));
    WGPUBufferDescriptor ubo_desc = {
      .label            = "Material parameter - Uniform buffer",
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = uniform_buffers.material_params.buffer_size,
      .mappedAtCreation = false,
    };
    uniform_buffers.material_params.buffer
      = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_desc);
  }

  /* Object parameter uniform buffer */
  {
    uniform_buffers.object_params.model_size = sizeof(vec4);
    uniform_buffers.object_params.buffer_size
      = calc_constant_buffer_byte_size(sizeof(object_params_dynamic));
    WGPUBufferDescriptor ubo_desc = {
      .label            = "Object parameter - Uniform buffer",
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = uniform_buffers.object_params.buffer_size,
      .mappedAtCreation = false,
    };
    uniform_buffers.object_params.buffer
      = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_desc);
  }

  update_uniform_buffers(context);
  update_dynamic_uniform_buffer(context->wgpu_context);
  update_params(context->wgpu_context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    generate_brdf_lut(context->wgpu_context);
    generate_irradiance_cube(context->wgpu_context);
    generate_prefiltered_env_cube(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_bind_group_layouts(context->wgpu_context);
    setup_pipeline_layouts(context->wgpu_context);
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
    if (imgui_overlay_combo_box(context->imgui_overlay, "Material",
                                &current_material_index, material_names, 11)) {
      update_dynamic_uniform_buffer(context->wgpu_context);
    }
    if (imgui_overlay_combo_box(context->imgui_overlay, "Object type",
                                &models.object_index, object_names, 4)) {
      update_dynamic_uniform_buffer(context->wgpu_context);
    }
    if (imgui_overlay_input_float(context->imgui_overlay, "Exposure",
                                  &ubo_params.exposure, 0.1f, "%.2f")) {
      update_params(context->wgpu_context);
    }
    if (imgui_overlay_input_float(context->imgui_overlay, "Gamma",
                                  &ubo_params.gamma, 0.1f, "%.2f")) {
      update_params(context->wgpu_context);
    }
    imgui_overlay_checkBox(context->imgui_overlay, "Skybox", &display_skybox);
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

  /* Skybox */
  if (display_skybox) {
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.skybox);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.skybox, 0, 0);
    wgpu_gltf_model_draw(models.skybox, (wgpu_gltf_model_render_options_t){0});
  }

  /* Objects */
  {
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.pbr);

    for (uint32_t i = 0; i < (uint32_t)SINGLE_ROW_OBJECT_COUNT; ++i) {
      uint32_t dynamic_offset     = i * (uint32_t)ALIGNMENT;
      uint32_t dynamic_offsets[2] = {dynamic_offset, dynamic_offset};
      // Bind the bind group for rendering a mesh using the dynamic offset
      wgpuRenderPassEncoderSetBindGroup(
        wgpu_context->rpass_enc, 0, bind_groups.objects, 2, dynamic_offsets);
      // Draw object
      wgpu_gltf_model_draw(models.objects[models.object_index].object,
                           (wgpu_gltf_model_render_options_t){0});
    }
  }

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
  wgpu_destroy_texture(&textures.environment_cube);
  wgpu_destroy_texture(&textures.lut_brdf);
  wgpu_destroy_texture(&textures.irradiance_cube);
  wgpu_destroy_texture(&textures.prefiltered_cube);
  wgpu_gltf_model_destroy(models.skybox);
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(models.objects); ++i) {
    wgpu_gltf_model_destroy(models.objects[i].object);
  }
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.object.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.skybox.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.ubo_params.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.material_params.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.object_params.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.pbr)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.skybox)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.objects)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.skybox)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.objects)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.skybox)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.pbr)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.skybox)
}

void example_pbr_ibl(int argc, char* argv[])
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
static const char* pbr_ibl_vertex_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4<f32>,
    model : mat4x4<f32>,
    view : mat4x4<f32>,
    camPos : vec3<f32>,
  };

  struct ObjectParams {
    objPos : vec3<f32>,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(3) var<uniform> object : ObjectParams;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) worldPos : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
  };

  @vertex
  fn main(
    @location(0) inPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) inUV: vec2<f32>
  ) -> Output {
    var output: Output;
    let locPos : vec3<f32> = (ubo.model * vec4<f32>(inPos, 1.0)).xyz;
    output.worldPos = locPos + object.objPos;
    output.normal = mat3x3(
        ubo.model[0].xyz,
        ubo.model[1].xyz,
        ubo.model[2].xyz,
      ) * inNormal;
    output.uv = inUV;
    output.uv.y = 1.0 - inUV.y;
    output.position = ubo.projection * ubo.view * vec4<f32>(output.worldPos, 1.0);
    return output;
  }
);

static const char* pbr_ibl_functions_fragment_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4<f32>,
    model : mat4x4<f32>,
    view : mat4x4<f32>,
    camPos : vec3<f32>,
  };

  const LIGHTS_ARRAY_LENGTH = 4;

  struct UBOParams {
    lights : array<vec4<f32>, LIGHTS_ARRAY_LENGTH>,
    exposure : f32,
    gamma : f32,
  };

  struct MaterialParams {
    roughness : f32,
    metallic : f32,
    specular : f32,
    r : f32,
    g : f32,
    b : f32,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var<uniform> uboParams : UBOParams;
  @group(0) @binding(2) var<uniform> material : MaterialParams;
  @group(0) @binding(4) var textureIrradiance: texture_cube<f32>;
  @group(0) @binding(5) var samplerIrradiance: sampler;
  @group(0) @binding(6) var textureBRDFLUT: texture_2d<f32>;
  @group(0) @binding(7) var samplerBRDFLUT: sampler;
  @group(0) @binding(8) var texturePrefilteredMap: texture_cube<f32>;
  @group(0) @binding(9) var samplerPrefilteredMap: sampler;

  const PI = 3.1415926535897932384626433832795;

  fn ALBEDO() -> vec3f {
    return vec3<f32>(material.r, material.g, material.b);
  }

  // From http://filmicgames.com/archives/75
  fn Uncharted2Tonemap(x : vec3<f32>) -> vec3f {
    let A : f32 = 0.15;
    let B : f32 = 0.50;
    let C : f32 = 0.10;
    let D : f32 = 0.20;
    let E : f32 = 0.02;
    let F : f32 = 0.30;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
  }

  // Normal Distribution function ----------------------------------------------
  fn D_GGX(dotNH : f32, roughness : f32) -> f32 {
    let alpha : f32 = roughness * roughness;
    let alpha2 : f32 = alpha * alpha;
    let denom : f32 = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
    return (alpha2)/(PI * denom*denom);
  }

  // Geometric Shadowing function ----------------------------------------------
  fn G_SchlicksmithGGX(dotNL : f32, dotNV : f32, roughness : f32) -> f32 {
    let r : f32 = (roughness + 1.0);
    let k : f32 = (r*r) / 8.0;
    let GL : f32 = dotNL / (dotNL * (1.0 - k) + k);
    let GV : f32 = dotNV / (dotNV * (1.0 - k) + k);
    return GL * GV;
  }

  // Fresnel function ----------------------------------------------------------
  fn F_Schlick(cosTheta : f32, F0 : vec3<f32>) -> vec3f {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
  }

  fn F_SchlickR(cosTheta : f32, F0 : vec3<f32>, roughness : f32) -> vec3f {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
  }

  fn prefilteredReflection(R : vec3<f32>, roughness : f32) -> vec3f {
    let MAX_REFLECTION_LOD : f32 = 9.0; // todo: param/const
    let lod : f32 = roughness * MAX_REFLECTION_LOD;
    let lodf : f32 = floor(lod);
    let lodc : f32 = ceil(lod);
    let a : vec3<f32> = textureSampleLevel(texturePrefilteredMap, samplerPrefilteredMap, R, lodf).rgb;
    let b : vec3<f32> = textureSampleLevel(texturePrefilteredMap, samplerPrefilteredMap, R, lodc).rgb;
    return mix(a, b, lod - lodf);
  }

  fn specularContribution(L : vec3<f32>, V : vec3<f32>, N : vec3<f32>, F0 : vec3<f32>,
                          metallic : f32, roughness : f32) -> vec3f {
    // Precalculate vectors and dot products
    let H : vec3<f32> = normalize(V + L);
    let dotNH : f32 = clamp(dot(N, H), 0.0, 1.0);
    let dotNV : f32 = clamp(dot(N, V), 0.0, 1.0);
    let dotNL : f32 = clamp(dot(N, L), 0.0, 1.0);

    // Light color fixed
    let lightColor : vec3<f32> = vec3(1.0);

    var color : vec3<f32> = vec3(0.0);

    if (dotNL > 0.0) {
      // D = Normal distribution (Distribution of the microfacets)
      let D : f32 = D_GGX(dotNH, roughness);
      // G = Geometric shadowing term (Microfacets shadowing)
      let G : f32 = G_SchlicksmithGGX(dotNL, dotNV, roughness);
      // F = Fresnel factor (Reflectance depending on angle of incidence)
      let F : vec3<f32> = F_Schlick(dotNV, F0);
      let spec : vec3<f32> = D * F * G / (4.0 * dotNL * dotNV + 0.001);
      let kD : vec3<f32> = (vec3<f32>(1.0) - F) * (1.0 - metallic);
      color += (kD * ALBEDO() / PI + spec) * dotNL;
    }

    return color;
  }
);

static const char* pbr_ibl_main_fragment_shader_wgsl = CODE(
  @fragment
  fn main(
    @location(0) inWorldPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) inUV: vec2<f32>
  ) -> @location(0) vec4<f32> {
    let N : vec3<f32> = normalize(inNormal);
    let V : vec3<f32> = normalize(ubo.camPos - inWorldPos);
    let R : vec3<f32> = reflect(-V, N);

    let metallic : f32 = material.metallic;
    let roughness : f32 = material.roughness;

    var F0 : vec3<f32> = vec3(0.04);
    F0 = mix(F0, ALBEDO(), metallic);

    var Lo : vec3<f32> = vec3(0.0);
    for (var i : u32 = 0; i < LIGHTS_ARRAY_LENGTH; i++) {
      let L : vec3<f32> = normalize(uboParams.lights[i].xyz - inWorldPos);
      Lo += specularContribution(L, V, N, F0, metallic, roughness);
    }

    let brdf : vec2<f32> = textureSample(textureBRDFLUT, samplerBRDFLUT, vec2<f32>(max(dot(N, V), 0.0), roughness)).rg;
    let reflection : vec3<f32> = prefilteredReflection(R, roughness).rgb;
    let irradiance : vec3<f32> = textureSample(textureIrradiance, samplerIrradiance, N).rgb;

    // Diffuse based on irradiance
    let diffuse : vec3<f32> = irradiance * ALBEDO();

    let F : vec3<f32> = F_SchlickR(max(dot(N, V), 0.0), F0, roughness);

    // Specular reflectance
    let specular : vec3<f32> = reflection * (F * brdf.x + brdf.y);

    // Ambient part
    var kD : vec3<f32> = 1.0 - F;
    kD *= 1.0 - metallic;
    let ambient : vec3<f32> = (kD * diffuse + specular);

    var color : vec3<f32> = ambient + Lo;

    // Tone mapping
    color = Uncharted2Tonemap(color * uboParams.exposure);
    color = color * (1.0f / Uncharted2Tonemap(vec3<f32>(11.2f)));
    // Gamma correction
    color = pow(color, vec3<f32>(1.0f / uboParams.gamma));

    return vec4<f32>(color, 1.0);
  }
);

static const char* skybox_vertex_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4<f32>,
    model : mat4x4<f32>,
    view : mat4x4<f32>,
    camPos : vec3<f32>,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) outUVW : vec3<f32>
  };

  @vertex
  fn main(
    @location(0) inPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) inUV: vec2<f32>
  ) -> Output {
    var output: Output;
    output.outUVW = inPos;
    output.position = ubo.projection * ubo.model * vec4<f32>(inPos.xyz, 1.0);
    return output;
  }
);

static const char* skybox_fragment_shader_wgsl = CODE(
  struct UBOParams {
    lights : array<vec4<f32>, 4>,
    exposure : f32,
    gamma : f32,
  };

  @group(0) @binding(1) var<uniform> uboParams : UBOParams;
  @group(0) @binding(2) var textureEnv: texture_cube<f32>;
  @group(0) @binding(3) var samplerEnv: sampler;

  // From http://filmicworlds.com/blog/filmic-tonemapping-operators/
  fn Uncharted2Tonemap(color : vec3<f32>) -> vec3f {
    let A : f32 = 0.15;
    let B : f32 = 0.50;
    let C : f32 = 0.10;
    let D : f32 = 0.20;
    let E : f32 = 0.02;
    let F : f32 = 0.30;
    let W : f32 = 11.2;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
  }

  @fragment
  fn main(
    @location(0) inUVW: vec3<f32>
  ) -> @location(0) vec4<f32> {
    var color : vec3<f32> = textureSample(textureEnv, samplerEnv, inUVW * vec3<f32>(1.0, -1.0, 1.0)).rgb;
    // Tone mapping
    color = Uncharted2Tonemap(color * uboParams.exposure);
    color = color * (1.0f / Uncharted2Tonemap(vec3<f32>(11.2f)));
    // Gamma correction
    color = pow(color, vec3<f32>(1.0f / uboParams.gamma));
    return vec4<f32>(color, 1.0);
  }
);
// clang-format on
