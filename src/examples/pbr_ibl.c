#include "example_base.h"
#include "examples.h"

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

#define GRID_DIM 7

static bool display_skybox = true;

static struct {
  texture_t environment_cube;
} textures;

static struct {
  struct gltf_model_t* skybox;
} models;

static struct {
  WGPUBuffer object;
  WGPUBuffer skybox;
  WGPUBuffer params;
} uniform_buffers;

static struct {
  mat4 projection;
  mat4 model;
  mat4 view;
  vec3 cam_pos;
} ubo_matrices;

static struct {
  vec4 lights[4];
  float exposure;
  float gamma;
} ubo_params = {
  .exposure = 4.5f,
  .gamma    = 2.2f,
};

static struct {
  WGPURenderPipeline skybox;
} pipelines;

static struct {
  WGPUBindGroup skybox;
} bind_groups;

static struct {
  WGPUBindGroupLayout skybox;
} bind_group_layouts;

WGPUPipelineLayout pipeline_layout;

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
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices;
  // Skybox
  models.skybox
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/cube.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
  // Load cube map
  static const char* cubemap[6] = {
    "textures/cubemaps/pisa_cube_px.jpg", // Right
    "textures/cubemaps/pisa_cube_nx.jpg", // Left
    "textures/cubemaps/pisa_cube_py.jpg", // Top
    "textures/cubemaps/pisa_cube_ny.jpg", // Bottom
    "textures/cubemaps/pisa_cube_pz.jpg", // Back
    "textures/cubemaps/pisa_cube_nz.jpg", // Front
  };
  textures.environment_cube = wgpu_create_texture_cubemap_from_files(
    wgpu_context, cubemap,
    &(struct wgpu_texture_load_options_t){
      .flip_y = false,
    });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout for skybox
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Vertex shader uniform UBO
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = sizeof(ubo_matrices),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Fragment uniform UBOParams
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize = sizeof(ubo_params),
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Fragment shader image view
        .binding = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        // Binding 3: Fragment shader image sampler
        .binding = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };

    // Create the bind group layout
    bind_group_layouts.skybox = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.skybox != NULL)
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Bind group for skybox
  {
    WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: Vertex shader uniform UBO
        .binding = 0,
        .buffer = uniform_buffers.skybox,
        .offset = 0,
        .size = sizeof(ubo_matrices),
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment uniform UBOParams
        .binding = 1,
        .buffer = uniform_buffers.params,
        .offset = 0,
        .size = sizeof(ubo_params),
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader image view
        .binding = 2,
        .textureView = textures.environment_cube.view
      },
      [3] = (WGPUBindGroupEntry) {
        // Binding 3: Fragment shader image sampler
        .binding = 3,
        .sampler = textures.environment_cube.sampler,
      },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .layout     = bind_group_layouts.skybox,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.skybox
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.skybox != NULL)
  }
}
