#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"

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

#define SINGLE_ROW_OBJECT_COUNT 10u
#define ALIGNMENT 256u // 256-byte alignment

#define BRDF_LUT_DIM 512
#define IRRADIANCE_CUBE_DIM 64
#define IRRADIANCE_CUBE_NUM_MIPS 7 // ((uint32_t)(floor(log2(dim)))) + 1;
#define PREFILTERED_CUBE_DIM 512
#define PREFILTERED_CUBE_NUM_MIPS 10 // ((uint32_t)(floor(log2(dim)))) + 1;

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
static int32_t current_material_index = 9;

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
      .flip_y = true, // Flip y to match pisa_cube.ktx hdr cubemap
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
                              .label      = "Objects bind group layout",
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
                              .label      = "Skybox bind group layout",
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
                              .label                = "PBR pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &bind_group_layouts.objects,
                            });
    ASSERT(pipeline_layouts.pbr != NULL);
  }

  /* Create the pipeline layout for skybox */
  {
    pipeline_layouts.skybox = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label                = "Skybox pipeline layout",
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
                              .label      = "Objects bind group",
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
      .label      = "Skybox bind group",
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
  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.1f,
        .g = 0.1f,
        .b = 0.1f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Construct the different states making up the pipeline

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
    skybox,
    // Location 0: Position
    WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
    // Location 1: Vertex normal
    WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal),
    // Location 2: UV
    WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_UV));

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Skybox pipeline (background cube)
  {
    primitive_state.cullMode              = WGPUCullMode_Front;
    depth_stencil_state.depthWriteEnabled = false;

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader SPIR-V
              .label = "Skybox vertex shader",
              .file  = "shaders/pbr_ibl/skybox.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &skybox_vertex_buffer_layout,
          });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader SPIR-V
              .label = "Skybox fragment shader",
              .file  = "shaders/pbr_ibl/skybox.frag.spv",
            },
            .target_count = 1,
            .targets      = &color_target_state,
          });

    // Create rendering pipeline using the specified states
    pipelines.skybox = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Skybox render pipeline",
                              .layout       = pipeline_layouts.skybox,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.skybox != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  // PBR pipeline
  {
    primitive_state.cullMode = WGPUCullMode_None;

    // Enable depth write
    depth_stencil_state.depthWriteEnabled = true;

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader SPIR-V
              .label = "PBR IBL vertex shader",
              .file  = "shaders/pbr_ibl/pbribl.vert.spv",
            },
            .buffer_count = 1,
            .buffers      = &skybox_vertex_buffer_layout,
          });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader SPIR-V
              .label = "PBR IBL fragment shader",
              .file  = "shaders/pbr_ibl/pbribl.frag.spv",
            },
            .target_count = 1,
            .targets      = &color_target_state,
          });

    // Create rendering pipeline using the specified states
    pipelines.pbr = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "PBR render pipeline",
                              .layout       = pipeline_layouts.pbr,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipelines.pbr != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static uint64_t calc_constant_buffer_byte_size(uint64_t byte_size)
{
  return (byte_size + 255) & ~255;
}

// Generate a BRDF integration map used as a look-up-table (stores roughness /
// NdotV)
static void generate_brdf_lut(wgpu_context_t* wgpu_context)
{
  const WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
  const int32_t dim              = (int32_t)BRDF_LUT_DIM;

  /* Texture dimensions */
  WGPUExtent3D texture_extent = {
    .width              = dim,
    .height             = dim,
    .depthOrArrayLayers = 1,
  };

  /* Create the texture */
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "LUT BRDF texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = format,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    textures.lut_brdf.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(textures.lut_brdf.texture != NULL);
  }

  /* Create the texture view */
  {
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "LUT BRDF texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    textures.lut_brdf.view
      = wgpuTextureCreateView(textures.lut_brdf.texture, &texture_view_dec);
    ASSERT(textures.lut_brdf.view != NULL);
  }

  /* Create the sampler */
  {
    textures.lut_brdf.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = "LUT BRDF texture sampler",
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUFilterMode_Linear,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(textures.lut_brdf.sampler != NULL);
  }

  /* Look-up-table (from BRDF) pipeline */
  WGPURenderPipeline pipeline = NULL;
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
      .format    = format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .label = "Gen BRDF LUT vertex shader",
                .file  = "shaders/pbr_ibl/genbrdflut.vert.spv",
              },
              .buffer_count = 0,
              .buffers      = NULL,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Gen BRDF LUT fragment shader",
                .file  = "shaders/pbr_ibl/genbrdflut.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    // Create rendering pipeline using the specified states
    pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Gen BRDF LUT render pipeline",
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = NULL,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Create the actual renderpass */
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass = {
    .color_attachment[0]= (WGPURenderPassColorAttachment) {
        .view       = textures.lut_brdf.view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
     },
  };
  render_pass.render_pass_descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachment,
    .depthStencilAttachment = NULL,
  };

  /* Render */
  {
    wgpu_context->cmd_enc
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.render_pass_descriptor);
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                     (float)dim, (float)dim, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u, dim,
                                        dim);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);

    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

    WGPUCommandBuffer command_buffer
      = wgpuCommandEncoderFinish(wgpu_context->cmd_enc, NULL);
    ASSERT(command_buffer != NULL);
    WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

    // Sumbit commmand buffer and cleanup
    wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)
  }

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline);
}

// Generate an irradiance cube map from the environment cube map
static void generate_irradiance_cube(wgpu_context_t* wgpu_context)
{
  const WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
  const int32_t dim              = (int32_t)IRRADIANCE_CUBE_DIM;
  const uint32_t num_mips        = (uint32_t)IRRADIANCE_CUBE_NUM_MIPS;
  ASSERT(num_mips == ((uint32_t)(floor(log2(dim)))) + 1);
  const uint32_t array_layer_count = 6u; // Cube map

  /** Pre-filtered cube map **/
  // Texture dimensions
  WGPUExtent3D texture_extent = {
    .width              = dim,
    .height             = dim,
    .depthOrArrayLayers = array_layer_count,
  };

  // Create the texture
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Irradiance cube texture",
      .size          = texture_extent,
      .mipLevelCount = num_mips,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = format,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst
               | WGPUTextureUsage_TextureBinding,
    };
    textures.irradiance_cube.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(textures.irradiance_cube.texture != NULL);
  }

  // Create the texture view
  {
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Irradiance cube texture view",
      .dimension       = WGPUTextureViewDimension_Cube,
      .format          = format,
      .baseMipLevel    = 0,
      .mipLevelCount   = num_mips,
      .baseArrayLayer  = 0,
      .arrayLayerCount = array_layer_count,
    };
    textures.irradiance_cube.view = wgpuTextureCreateView(
      textures.irradiance_cube.texture, &texture_view_dec);
    ASSERT(textures.irradiance_cube.view != NULL);
  }

  // Create the sampler
  {
    textures.irradiance_cube.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label        = "Irradiance cube texture sampler",
                              .addressModeU = WGPUAddressMode_ClampToEdge,
                              .addressModeV = WGPUAddressMode_ClampToEdge,
                              .addressModeW = WGPUAddressMode_ClampToEdge,
                              .minFilter    = WGPUFilterMode_Linear,
                              .magFilter    = WGPUFilterMode_Linear,
                              .mipmapFilter = WGPUFilterMode_Linear,
                              .lodMinClamp  = 0.0f,
                              .lodMaxClamp  = (float)num_mips,
                              .maxAnisotropy = 1,
                            });
    ASSERT(textures.irradiance_cube.sampler != NULL);
  }

  // Framebuffer for offscreen rendering
  struct {
    WGPUTexture texture;
    WGPUTextureView texture_views[6 * (uint32_t)IRRADIANCE_CUBE_NUM_MIPS];
  } offscreen;

  // Offscreen framebuffer
  {
    // Color attachment
    {
      // Create the texture
      WGPUTextureDescriptor texture_desc = {
        .label         = "irradiance_cube_offscreen_texture",
        .size          = (WGPUExtent3D) {
          .width              = dim,
          .height             = dim,
          .depthOrArrayLayers = array_layer_count,
        },
        .mipLevelCount = num_mips,
        .sampleCount   = 1,
        .dimension     = WGPUTextureDimension_2D,
        .format        = format,
        .usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding
                 | WGPUTextureUsage_RenderAttachment,
      };
      offscreen.texture
        = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
      ASSERT(offscreen.texture != NULL);

      // Create the texture views
      uint32_t idx = 0;
      for (uint32_t array_layer = 0; array_layer < array_layer_count;
           ++array_layer) {
        for (uint32_t i = 0; i < num_mips; ++i) {
          idx = (array_layer * num_mips) + i;
          WGPUTextureViewDescriptor texture_view_dec = {
            .label           = "irradiance_cube_offscreen_texture_view",
            .aspect          = WGPUTextureAspect_All,
            .dimension       = WGPUTextureViewDimension_2D,
            .format          = texture_desc.format,
            .baseMipLevel    = i,
            .mipLevelCount   = 1,
            .baseArrayLayer  = array_layer,
            .arrayLayerCount = 1,
          };
          offscreen.texture_views[idx]
            = wgpuTextureCreateView(offscreen.texture, &texture_view_dec);
          ASSERT(offscreen.texture_views[idx] != NULL);
        }
      }
    }
  }

  struct push_block_vs_t {
    mat4 mvp;
    uint8_t padding[192];
  } push_block_vs[(uint32_t)IRRADIANCE_CUBE_NUM_MIPS * 6];

  struct push_block_fs_t {
    float delta_phi;
    float delta_theta;
    uint8_t padding[248];
  } push_block_fs[(uint32_t)IRRADIANCE_CUBE_NUM_MIPS * 6];

  // Update shader push constant block data
  {
    mat4 matrices[6] = {
      GLM_MAT4_IDENTITY_INIT, /* POSITIVE_X */
      GLM_MAT4_IDENTITY_INIT, /* NEGATIVE_X */
      GLM_MAT4_IDENTITY_INIT, /* POSITIVE_Y */
      GLM_MAT4_IDENTITY_INIT, /* NEGATIVE_Y */
      GLM_MAT4_IDENTITY_INIT, /* POSITIVE_Z */
      GLM_MAT4_IDENTITY_INIT, /* NEGATIVE_Z */
    };
    // NEGATIVE_X
    glm_rotate(matrices[0], glm_rad(90.0f), (vec3){0.0f, 1.0f, 0.0f});
    glm_rotate(matrices[0], glm_rad(180.0f), (vec3){1.0f, 0.0f, 0.0f});
    // NEGATIVE_X
    glm_rotate(matrices[1], glm_rad(-90.0f), (vec3){0.0f, 1.0f, 0.0f});
    glm_rotate(matrices[1], glm_rad(180.0f), (vec3){1.0f, 0.0f, 0.0f});
    // POSITIVE_Y
    glm_rotate(matrices[2], glm_rad(90.0f), (vec3){1.0f, 0.0f, 0.0f});
    // NEGATIVE_Y
    glm_rotate(matrices[3], glm_rad(-90.0f), (vec3){1.0f, 0.0f, 0.0f});
    // POSITIVE_Z
    glm_rotate(matrices[4], glm_rad(180.0f), (vec3){1.0f, 0.0f, 0.0f});
    // NEGATIVE_Z
    glm_rotate(matrices[5], glm_rad(180.0f), (vec3){0.0f, 0.0f, 1.0f});

    mat4 projection = GLM_MAT4_IDENTITY_INIT;
    glm_perspective(PI / 2.0f, 1.0f, 0.1f, 512.0f, projection);
    // Sampling deltas
    const float delta_phi   = (2.0f * PI) / 180.0f;
    const float delta_theta = (0.5f * PI) / 64.0f;
    uint32_t idx            = 0;
    for (uint32_t m = 0; m < num_mips; ++m) {
      for (uint32_t f = 0; f < 6; ++f) {
        idx = (m * 6) + f;
        // Set vertex shader push constant block
        glm_mat4_mul(projection, matrices[f], push_block_vs[idx].mvp);
        // Set fragment shader push constant block
        push_block_fs[idx].delta_phi   = delta_phi;
        push_block_fs[idx].delta_theta = delta_theta;
      }
    }
  }

  static struct {
    // Vertex shader parameter uniform buffer
    struct {
      WGPUBuffer buffer;
      uint64_t buffer_size;
      uint64_t model_size;
    } vs;
    // Fragment parameter uniform buffer
    struct {
      WGPUBuffer buffer;
      uint64_t buffer_size;
      uint64_t model_size;
    } fs;
  } irradiance_cube_ubos = {0};

  // Vertex shader parameter uniform buffer
  {
    irradiance_cube_ubos.vs.model_size = sizeof(mat4);
    irradiance_cube_ubos.vs.buffer_size
      = calc_constant_buffer_byte_size(sizeof(push_block_vs));
    irradiance_cube_ubos.vs.buffer = wgpu_create_buffer_from_data(
      wgpu_context, push_block_vs, irradiance_cube_ubos.vs.buffer_size,
      WGPUBufferUsage_Uniform);
  }

  // Fragment shader parameter uniform buffer
  {
    irradiance_cube_ubos.fs.model_size = sizeof(float) * 2;
    irradiance_cube_ubos.fs.buffer_size
      = calc_constant_buffer_byte_size(sizeof(push_block_fs));
    irradiance_cube_ubos.fs.buffer = wgpu_create_buffer_from_data(
      wgpu_context, push_block_fs, irradiance_cube_ubos.fs.buffer_size,
      WGPUBufferUsage_Uniform);
  }

  // Bind group layout
  WGPUBindGroupLayout bind_group_layout = NULL;
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Vertex shader uniform UBO
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = irradiance_cube_ubos.vs.model_size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Fragment shader uniform UBO
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = irradiance_cube_ubos.fs.model_size,
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
    bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layout != NULL);
  }

  // Bind group
  WGPUBindGroup bind_group = NULL;
  {
    WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform UBO
        .binding = 0,
        .buffer  = irradiance_cube_ubos.vs.buffer,
        .offset  = 0,
        .size    = irradiance_cube_ubos.vs.model_size,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment shader uniform UBO
        .binding = 1,
        .buffer  = irradiance_cube_ubos.fs.buffer,
        .offset  = 0,
        .size    = irradiance_cube_ubos.fs.model_size,
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
    bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_group != NULL);
  }

  // Pipeline layout
  WGPUPipelineLayout pipeline_layout = NULL;
  {
    pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label                = "Pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &bind_group_layout,
                            });
    ASSERT(pipeline_layout != NULL);
  }

  // Irradiance cube map pipeline
  WGPURenderPipeline pipeline = NULL;
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
      .format    = format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex buffer layout
    WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
      skybox,
      // Location 0: Position
      WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position));

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-
                .label = "Filtercube vertex shader",
                .file  = "shaders/pbr_ibl/filtercube.vert.spv",
              },
             .buffer_count = 1,
             .buffers      = &skybox_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "Irradiancecube fragment shader",
                .file  = "shaders/pbr_ibl/irradiancecube.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
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
                              .label  = "irradiance_cube_map_render_pipeline",
                              .layout = pipeline_layout,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = NULL,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  // Create the actual renderpass
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass = {
    .color_attachment[0]= (WGPURenderPassColorAttachment) {
        .view       = NULL, /* Assigned later */
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.2f,
          .a = 0.0f,
        },
     },
  };
  render_pass.render_pass_descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachment,
    .depthStencilAttachment = NULL,
  };

  // Render
  {
    wgpu_context->cmd_enc
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

    uint32_t idx         = 0;
    float viewport_width = 0.0f, viewport_height = 0.0f;
    for (uint32_t m = 0; m < num_mips; ++m) {
      viewport_width  = (float)(dim * pow(0.5f, m));
      viewport_height = (float)(dim * pow(0.5f, m));
      for (uint32_t f = 0; f < 6; ++f) {
        render_pass.color_attachment[0].view
          = offscreen.texture_views[(f * num_mips) + m];
        idx = (m * 6) + f;
        // Render scene from cube face's point of view
        wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
          wgpu_context->cmd_enc, &render_pass.render_pass_descriptor);
        wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                         viewport_width, viewport_height, 0.0f,
                                         1.0f);
        wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                            (uint32_t)viewport_width,
                                            (uint32_t)viewport_height);
        wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
        // Calculate the dynamic offsets
        uint32_t dynamic_offset     = idx * (uint32_t)ALIGNMENT;
        uint32_t dynamic_offsets[2] = {dynamic_offset, dynamic_offset};
        // Bind the bind group for rendering a mesh using the dynamic offset
        wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                          bind_group, 2, dynamic_offsets);
        // Draw object
        wgpu_gltf_model_draw(models.skybox,
                             (wgpu_gltf_model_render_options_t){0});
        // End render pass
        wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
        WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
      }
    }

    // Copy region for transfer from framebuffer to cube face
    for (uint32_t m = 0; m < num_mips; ++m) {
      WGPUExtent3D copy_size = (WGPUExtent3D){
        .width              = (float)(dim * pow(0.5f, m)),
        .height             = (float)(dim * pow(0.5f, m)),
        .depthOrArrayLayers = array_layer_count,
      };
      wgpuCommandEncoderCopyTextureToTexture(
        wgpu_context->cmd_enc,
        // source
        &(WGPUImageCopyTexture){
          .texture  = offscreen.texture,
          .mipLevel = m,
        },
        // destination
        &(WGPUImageCopyTexture){
          .texture  = textures.irradiance_cube.texture,
          .mipLevel = m,
        },
        // copySize
        &copy_size);
    }

    WGPUCommandBuffer command_buffer
      = wgpuCommandEncoderFinish(wgpu_context->cmd_enc, NULL);
    ASSERT(command_buffer != NULL);
    WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

    // Sumbit commmand buffer and cleanup
    wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)
  }

  // Cleanup
  WGPU_RELEASE_RESOURCE(Texture, offscreen.texture)
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(offscreen.texture_views); ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, offscreen.texture_views[i])
  }
  WGPU_RELEASE_RESOURCE(Buffer, irradiance_cube_ubos.vs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, irradiance_cube_ubos.fs.buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
}

// Prefilter environment cubemap
// See
// https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
static void generate_prefiltered_cube(wgpu_context_t* wgpu_context)
{
  const WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
  const int32_t dim              = (int32_t)PREFILTERED_CUBE_DIM;
  const uint32_t num_mips        = (uint32_t)PREFILTERED_CUBE_NUM_MIPS;
  ASSERT(num_mips == ((uint32_t)(floor(log2(dim)))) + 1)
  const uint32_t array_layer_count = 6; // Cube map

  /** Pre-filtered cube map **/
  // Texture dimensions
  WGPUExtent3D texture_extent = {
    .width              = dim,
    .height             = dim,
    .depthOrArrayLayers = array_layer_count,
  };

  // Create the texture
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Prefiltered cube texture",
      .size          = texture_extent,
      .mipLevelCount = num_mips,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = format,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst
               | WGPUTextureUsage_TextureBinding,
    };
    textures.prefiltered_cube.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(textures.prefiltered_cube.texture != NULL);
  }

  // Create the texture view
  {
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Prefiltered cube texture view",
      .dimension       = WGPUTextureViewDimension_Cube,
      .format          = format,
      .baseMipLevel    = 0,
      .mipLevelCount   = num_mips,
      .baseArrayLayer  = 0,
      .arrayLayerCount = array_layer_count,
    };
    textures.prefiltered_cube.view = wgpuTextureCreateView(
      textures.prefiltered_cube.texture, &texture_view_dec);
    ASSERT(textures.prefiltered_cube.view != NULL);
  }

  // Create the sampler
  {
    textures.prefiltered_cube.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label = "prefiltered_cube_texture_sampler",
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUFilterMode_Linear,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = (float)num_mips,
                              .maxAnisotropy = 1,
                            });
    ASSERT(textures.prefiltered_cube.sampler != NULL);
  }

  // Framebuffer for offscreen rendering
  struct {
    WGPUTexture texture;
    WGPUTextureView texture_views[6 * (uint32_t)PREFILTERED_CUBE_NUM_MIPS];
  } offscreen;

  // Offscreen framebuffer
  {
    // Color attachment
    {
      // Create the texture
      WGPUTextureDescriptor texture_desc = {
        .label         = "prefiltered_cube_offscreen_texture",
        .size          = (WGPUExtent3D) {
          .width              = dim,
          .height             = dim,
          .depthOrArrayLayers = array_layer_count,
        },
        .mipLevelCount = num_mips,
        .sampleCount   = 1,
        .dimension     = WGPUTextureDimension_2D,
        .format        = format,
        .usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding
                 | WGPUTextureUsage_RenderAttachment,
      };
      offscreen.texture
        = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
      ASSERT(offscreen.texture != NULL)

      // Create the texture views
      uint32_t idx = 0;
      for (uint32_t array_layer = 0; array_layer < array_layer_count;
           ++array_layer) {
        for (uint32_t i = 0; i < num_mips; ++i) {
          idx = (array_layer * num_mips) + i;
          WGPUTextureViewDescriptor texture_view_dec = {
            .label           = "prefiltered_cube_offscreen_texture_view",
            .aspect          = WGPUTextureAspect_All,
            .dimension       = WGPUTextureViewDimension_2D,
            .format          = texture_desc.format,
            .baseMipLevel    = i,
            .mipLevelCount   = 1,
            .baseArrayLayer  = array_layer,
            .arrayLayerCount = 1,
          };
          offscreen.texture_views[idx]
            = wgpuTextureCreateView(offscreen.texture, &texture_view_dec);
          ASSERT(offscreen.texture_views[idx] != NULL)
        }
      }
    }
  }

  struct push_block_vs_t {
    mat4 mvp;
    uint8_t padding[192];
  } push_block_vs[(uint32_t)PREFILTERED_CUBE_NUM_MIPS * 6];

  struct push_block_fs_t {
    float roughness;
    uint32_t num_samples;
    uint8_t padding[248];
  } push_block_fs[(uint32_t)PREFILTERED_CUBE_NUM_MIPS * 6];

  // Update shader push constant block data
  {
    mat4 matrices[6] = {
      GLM_MAT4_IDENTITY_INIT, // POSITIVE_X
      GLM_MAT4_IDENTITY_INIT, // NEGATIVE_X
      GLM_MAT4_IDENTITY_INIT, // POSITIVE_Y
      GLM_MAT4_IDENTITY_INIT, // NEGATIVE_Y
      GLM_MAT4_IDENTITY_INIT, // POSITIVE_Z
      GLM_MAT4_IDENTITY_INIT, // NEGATIVE_Z
    };
    // NEGATIVE_X
    glm_rotate(matrices[0], glm_rad(90.0f), (vec3){0.0f, 1.0f, 0.0f});
    glm_rotate(matrices[0], glm_rad(180.0f), (vec3){1.0f, 0.0f, 0.0f});
    // NEGATIVE_X
    glm_rotate(matrices[1], glm_rad(-90.0f), (vec3){0.0f, 1.0f, 0.0f});
    glm_rotate(matrices[1], glm_rad(180.0f), (vec3){1.0f, 0.0f, 0.0f});
    // POSITIVE_Y
    glm_rotate(matrices[2], glm_rad(90.0f), (vec3){1.0f, 0.0f, 0.0f});
    // NEGATIVE_Y
    glm_rotate(matrices[3], glm_rad(-90.0f), (vec3){1.0f, 0.0f, 0.0f});
    // POSITIVE_Z
    glm_rotate(matrices[4], glm_rad(180.0f), (vec3){1.0f, 0.0f, 0.0f});
    // NEGATIVE_Z
    glm_rotate(matrices[5], glm_rad(180.0f), (vec3){0.0f, 0.0f, 1.0f});

    mat4 projection = GLM_MAT4_IDENTITY_INIT;
    glm_perspective(PI / 2.0f, 1.0f, 0.1f, 512.0f, projection);
    // Sampling deltas
    uint32_t idx = 0;
    for (uint32_t m = 0; m < num_mips; ++m) {
      for (uint32_t f = 0; f < 6; ++f) {
        idx = (m * 6) + f;
        // Set vertex shader push constant block
        glm_mat4_mul(projection, matrices[f], push_block_vs[idx].mvp);
        // Set fragment shader push constant block
        push_block_fs[idx].roughness   = (float)m / (float)(num_mips - 1);
        push_block_fs[idx].num_samples = 32u;
      }
    }
  }

  static struct {
    // Vertex shader parameter uniform buffer
    struct {
      WGPUBuffer buffer;
      uint64_t buffer_size;
      uint64_t model_size;
    } vs;
    // Fragment parameter uniform buffer
    struct {
      WGPUBuffer buffer;
      uint64_t buffer_size;
      uint64_t model_size;
    } fs;
  } prefiltered_cube_ubos;

  // Vertex shader parameter uniform buffer
  {
    prefiltered_cube_ubos.vs.model_size = sizeof(mat4);
    prefiltered_cube_ubos.vs.buffer_size
      = calc_constant_buffer_byte_size(sizeof(push_block_vs));
    prefiltered_cube_ubos.vs.buffer = wgpu_create_buffer_from_data(
      wgpu_context, push_block_vs, prefiltered_cube_ubos.vs.buffer_size,
      WGPUBufferUsage_Uniform);
  }

  // Fragment shader parameter uniform buffer
  {
    prefiltered_cube_ubos.fs.model_size = sizeof(float) + sizeof(uint32_t);
    prefiltered_cube_ubos.fs.buffer_size
      = calc_constant_buffer_byte_size(sizeof(push_block_fs));
    prefiltered_cube_ubos.fs.buffer = wgpu_create_buffer_from_data(
      wgpu_context, push_block_fs, prefiltered_cube_ubos.fs.buffer_size,
      WGPUBufferUsage_Uniform);
  }

  // Bind group layout
  WGPUBindGroupLayout bind_group_layout = NULL;
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Vertex shader uniform UBO
        .binding   = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = prefiltered_cube_ubos.vs.model_size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Fragment shader uniform UBO
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = prefiltered_cube_ubos.fs.model_size,
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
    bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layout != NULL);
  }

  // Bind group
  WGPUBindGroup bind_group = NULL;
  {
    WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform UBO
        .binding = 0,
        .buffer  = prefiltered_cube_ubos.vs.buffer,
        .offset  = 0,
        .size    = prefiltered_cube_ubos.vs.model_size,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment shader uniform UBO
        .binding = 1,
        .buffer  = prefiltered_cube_ubos.fs.buffer,
        .offset  = 0,
        .size    = prefiltered_cube_ubos.fs.model_size,
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
    bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_group != NULL);
  }

  // Pipeline layout
  WGPUPipelineLayout pipeline_layout = NULL;
  {
    pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &bind_group_layout,
                            });
    ASSERT(pipeline_layout != NULL)
  }

  // Prefiltered cube map pipeline
  WGPURenderPipeline pipeline = NULL;
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
      .format    = format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex buffer layout
    WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
      skybox,
      // Location 0: Position
      WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position));

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .label = "filtercube_vertex_shaders",
                .file  = "shaders/pbr_ibl/filtercube.vert.spv",
              },
             .buffer_count = 1,
             .buffers = &skybox_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .label = "prefilterenvmap_fragment_shaders",
                .file  = "shaders/pbr_ibl/prefilterenvmap.frag.spv",
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
                              .label  = "prefiltered_cube_map_render_pipeline",
                              .layout = pipeline_layout,
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = NULL,
                              .multisample  = multisample_state,
                            });
    ASSERT(pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  // Create the actual renderpass
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass = {
    .color_attachment[0]= (WGPURenderPassColorAttachment) {
        .view       = NULL, /* Assigned later */
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.2f,
          .a = 0.0f,
        },
     },
  };
  render_pass.render_pass_descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachment,
    .depthStencilAttachment = NULL,
  };

  // Render
  {
    wgpu_context->cmd_enc
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

    uint32_t idx         = 0;
    float viewport_width = 0.0f, viewport_height = 0.0f;
    for (uint32_t m = 0; m < num_mips; ++m) {
      viewport_width  = (float)(dim * pow(0.5f, m));
      viewport_height = (float)(dim * pow(0.5f, m));
      for (uint32_t f = 0; f < 6; ++f) {
        render_pass.color_attachment[0].view
          = offscreen.texture_views[(f * num_mips) + m];
        idx = (m * 6) + f;
        // Render scene from cube face's point of view
        wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
          wgpu_context->cmd_enc, &render_pass.render_pass_descriptor);
        wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                         viewport_width, viewport_height, 0.0f,
                                         1.0f);
        wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                            (uint32_t)viewport_width,
                                            (uint32_t)viewport_height);
        wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
        // Calculate the dynamic offsets
        uint32_t dynamic_offset     = idx * (uint32_t)ALIGNMENT;
        uint32_t dynamic_offsets[2] = {dynamic_offset, dynamic_offset};
        // Bind the bind group for rendering a mesh using the dynamic offset
        wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                          bind_group, 2, dynamic_offsets);
        // Draw object
        wgpu_gltf_model_draw(models.skybox,
                             (wgpu_gltf_model_render_options_t){0});
        // End render pass
        wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
        WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
      }
    }

    // Copy region for transfer from framebuffer to cube face
    for (uint32_t m = 0; m < num_mips; ++m) {
      WGPUExtent3D copy_size = (WGPUExtent3D){
        .width              = (float)(dim * pow(0.5f, m)),
        .height             = (float)(dim * pow(0.5f, m)),
        .depthOrArrayLayers = array_layer_count,
      };
      wgpuCommandEncoderCopyTextureToTexture(
        wgpu_context->cmd_enc,
        // source
        &(WGPUImageCopyTexture){
          .texture  = offscreen.texture,
          .mipLevel = m,
        },
        // destination
        &(WGPUImageCopyTexture){
          .texture  = textures.prefiltered_cube.texture,
          .mipLevel = m,
        },
        // copySize
        &copy_size);
    }

    WGPUCommandBuffer command_buffer
      = wgpuCommandEncoderFinish(wgpu_context->cmd_enc, NULL);
    ASSERT(command_buffer != NULL);
    WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

    // Sumbit commmand buffer and cleanup
    wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)
  }

  // Cleanup
  WGPU_RELEASE_RESOURCE(Texture, offscreen.texture)
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(offscreen.texture_views); ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, offscreen.texture_views[i])
  }
  WGPU_RELEASE_RESOURCE(Buffer, prefiltered_cube_ubos.vs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, prefiltered_cube_ubos.fs.buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // 3D object
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

  // Skybox
  mat3 mat3_tmp = GLM_MAT3_ZERO_INIT;
  glm_mat4_pick3(camera->matrices.view, mat3_tmp);
  glm_mat4_ins3(mat3_tmp, ubo_matrices.model);
  ubo_matrices.model[3][3] = 1.0f;
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.skybox.buffer,
                          0, &ubo_matrices, uniform_buffers.skybox.size);
}

static void update_dynamic_uniform_buffer(wgpu_context_t* wgpu_context)
{
  // Set objects positions and material properties
  uint32_t obj_count = (uint32_t)SINGLE_ROW_OBJECT_COUNT;
  for (uint32_t x = 0; x < obj_count; ++x) {
    // Set object position
    vec3* pos = &object_params_dynamic[x].position;
    glm_vec3_copy((vec3){((float)(x - (obj_count / 2.0f))) * 2.15f, 0.0f, 0.0f},
                  *pos);
    // Set material metallic and roughness properties
    struct matrial_params_dynamic_t* mat_params = &material_params_dynamic[x];
    mat_params->roughness
      = 1.0f - glm_clamp((float)x / (float)obj_count, 0.005f, 1.0f);
    mat_params->metallic = glm_clamp((float)x / (float)obj_count, 0.005f, 1.0f);
    glm_vec3_copy(materials[current_material_index].params.color,
                  (*mat_params).color);
  }

  // Update buffers
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

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Object vertex shader uniform buffer
  uniform_buffers.object = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_matrices),
    });

  // Skybox vertex shader uniform buffer
  uniform_buffers.skybox = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_matrices),
    });

  // Shared parameter uniform buffer
  uniform_buffers.ubo_params = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_params),
    });

  // Material parameter uniform buffer
  {
    uniform_buffers.material_params.model_size = sizeof(vec3) + sizeof(vec3);
    uniform_buffers.material_params.buffer_size
      = calc_constant_buffer_byte_size(sizeof(material_params_dynamic));
    WGPUBufferDescriptor ubo_desc = {
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = uniform_buffers.material_params.buffer_size,
      .mappedAtCreation = false,
    };
    uniform_buffers.material_params.buffer
      = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_desc);
  }

  // Object parameter uniform buffer
  {
    uniform_buffers.object_params.model_size = sizeof(vec4);
    uniform_buffers.object_params.buffer_size
      = calc_constant_buffer_byte_size(sizeof(object_params_dynamic));
    WGPUBufferDescriptor ubo_desc = {
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
    generate_prefiltered_cube(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_bind_group_layouts(context->wgpu_context);
    setup_pipeline_layouts(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
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
  // Set target frame buffer
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  // Set viewport
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Skybox
  if (display_skybox) {
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.skybox);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.skybox, 0, 0);
    wgpu_gltf_model_draw(models.skybox, (wgpu_gltf_model_render_options_t){0});
  }

  // Objects
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
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
