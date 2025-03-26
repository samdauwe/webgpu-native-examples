#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader Ray Tracing
 *
 * Simple GPU ray tracer with shadows and reflections using a compute shader. No
 * scene geometry is rendered in the graphics pass.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/computeraytracing
 * -------------------------------------------------------------------------- */

#define HIGH_QUALITY 0

#if !HIGH_QUALITY
#define TEX_DIM 1024u
#else
#define TEX_DIM 2048u
#endif

static texture_t texture_compute_target = {0};
static uint32_t current_id
  = 0; // Id used to identify objects by the ray tracing shader

// Resources for the graphics part of the example
static struct {
  WGPUBindGroupLayout
    bind_group_layout; // Raytraced image display shader binding layout
  WGPUBindGroup
    bind_group_pre_compute;    // Raytraced image display shader bindings before
                               // compute shader image manipulation
  WGPUBindGroup bind_group;    // Raytraced image display shader bindings after
                               // compute shader image manipulation
  WGPURenderPipeline pipeline; // Raytraced image display pipeline
  WGPUPipelineLayout pipeline_layout; // Layout of the graphics pipeline
} graphics = {0};

// Resources for the compute part of the example
static struct {
  struct {
    struct wgpu_buffer_t
      spheres; // (Shader) storage buffer object with scene spheres
    struct wgpu_buffer_t
      planes; // (Shader) storage buffer object with scene planes
  } storage_buffers;
  struct wgpu_buffer_t
    uniform_buffer; // Uniform buffer object containing scene data
  WGPUBindGroupLayout bind_group_layout; // Compute shader binding layout
  WGPUBindGroup bind_group;              // Compute shader bindings
  WGPUPipelineLayout pipeline_layout;    // Layout of the compute pipeline
  WGPUComputePipeline pipeline;          // Compute raytracing pipeline
  struct compute_ubo_t {                 // Compute shader uniform block object
    vec3 lightPos;
    float aspectRatio; // Aspect ratio of the viewport
    vec4 fogColor;
    struct {
      vec3 pos;
      vec3 lookat;
      float fov;
    } camera;
  } ubo;
} compute = {0};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// SSBO sphere declaration
typedef struct sphere_t {
  vec3 pos;
  float radius;
  vec3 diffuse;
  float specular;
  uint32_t id; // Id used to identify sphere for raytracing
  int32_t _pad[3];
} sphere_t;

// SSBO plane declaration
typedef struct plane_t {
  vec3 normal;
  float distance;
  vec3 diffuse;
  float specular;
  uint32_t id;
  int32_t _pad[3];
} plane_t;

// Other variables
static const char* example_title = "Compute Shader Ray Tracing";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 512.0f);
  camera_set_rotation(context->camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_translation(context->camera, (vec3){0.0f, 0.0f, -4.0f});
  camera_set_rotation_speed(context->camera, 0.0f);
  camera_set_movement_speed(context->camera, 2.5f);
}

// Prepare a texture target that is used to store compute shader calculations
static void prepare_texture_target(wgpu_context_t* wgpu_context, texture_t* tex,
                                   uint32_t width, uint32_t height,
                                   WGPUTextureFormat format)
{
  // Prepare blit target texture
  tex->size.width              = width;
  tex->size.height             = height;
  tex->size.depthOrArrayLayers = 1;
  tex->mip_level_count         = 1;
  tex->format                  = format;

  tex->texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label         = "Blit target - Texture",
      .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = (WGPUExtent3D){
        .width               = tex->size.width,
        .height              = tex->size.height,
        .depthOrArrayLayers  = tex->size.depthOrArrayLayers,
      },
      .format        = tex->format,
      .mipLevelCount = tex->mip_level_count,
      .sampleCount   = 1,
    });

  // Create the texture view
  tex->view = wgpuTextureCreateView(tex->texture,
                                    &(WGPUTextureViewDescriptor){
                                      .label     = "Blit target - Texture view",
                                      .format    = tex->format,
                                      .dimension = WGPUTextureViewDimension_2D,
                                      .baseMipLevel    = 0,
                                      .mipLevelCount   = tex->mip_level_count,
                                      .baseArrayLayer  = 0,
                                      .arrayLayerCount = 1,
                                    });

  // Create sampler
  tex->sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Blit target - Texture sampler",
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
}

void init_sphere(sphere_t* sphere, vec3 pos, float radius, vec3 diffuse,
                 float specular)
{
  pos[1] *= -1.0f; /* flip y */

  sphere->id = current_id++;
  glm_vec3_copy(pos, sphere->pos);
  sphere->radius = radius;
  glm_vec3_copy(diffuse, sphere->diffuse);
  sphere->specular = specular;
}

void init_plane(plane_t* plane, vec3 normal, float distance, vec3 diffuse,
                float specular)
{
  plane->id = current_id++;
  glm_vec3_copy(normal, plane->normal);
  plane->distance = distance;
  glm_vec3_copy(diffuse, plane->diffuse);
  plane->specular = specular;
}

// Setup and fill the compute shader storage buffers containing primitives for
// the raytraced scene
static void prepare_storage_buffers(wgpu_context_t* wgpu_context)
{
  // Spheres
  static sphere_t spheres[3] = {0};
  init_sphere(&spheres[0], (vec3){1.75f, -0.5f, 0.0f}, 1.0f,
              (vec3){0.0f, 1.0f, 0.0f}, 32.0f);
  init_sphere(&spheres[1], (vec3){0.0f, 1.0f, -0.5f}, 1.0f,
              (vec3){0.65f, 0.77f, 0.97f}, 32.0f);
  init_sphere(&spheres[2], (vec3){-1.75f, -0.75f, -0.5f}, 1.25f,
              (vec3){0.9f, 0.76f, 0.46f}, 32.0f);
  uint64_t storage_buffer_size = ARRAY_SIZE(spheres) * sizeof(sphere_t);

  // Stage
  // The SSBO will be used as a storage buffer for the compute pipeline and as a
  // vertex buffer in the graphics pipeline
  compute.storage_buffers.spheres = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Spheres compute - Storage buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size         = storage_buffer_size,
                    .initial.data = &spheres,
                  });

  // Planes
  static plane_t planes[6] = {0};
  const float room_dim     = 4.0f;
  init_plane(&planes[0], (vec3){0.0f, 1.0f, 0.0f}, room_dim,
             (vec3){1.0f, 1.0f, 1.0f}, 32.0f);
  init_plane(&planes[1], (vec3){0.0f, -1.0f, 0.0f}, room_dim,
             (vec3){1.0f, 1.0f, 1.0f}, 32.0f);
  init_plane(&planes[2], (vec3){0.0f, 0.0f, 1.0f}, room_dim,
             (vec3){1.0f, 1.0f, 1.0f}, 32.0f);
  init_plane(&planes[3], (vec3){0.0f, 0.0f, -1.0f}, room_dim,
             (vec3){0.0f, 0.0f, 0.0f}, 32.0f);
  init_plane(&planes[4], (vec3){-1.0f, 0.0f, 0.0f}, room_dim,
             (vec3){1.0f, 0.0f, 0.0f}, 32.0f);
  init_plane(&planes[5], (vec3){1.0f, 0.0f, 0.0f}, room_dim,
             (vec3){0.0f, 1.0f, 0.0f}, 32.0f);
  storage_buffer_size = ARRAY_SIZE(planes) * sizeof(plane_t);

  // Stage
  // The SSBO will be used as a storage buffer for the compute pipeline and as a
  // vertex buffer in the graphics pipeline
  compute.storage_buffers.planes = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Plane compute - Storage buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size         = storage_buffer_size,
                    .initial.data = &planes,
                  });
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
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

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0 : Fragment shader image
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
      // Binding 1 : Fragment shader sampler
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = "Graphics - Bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  graphics.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(graphics.bind_group_layout != NULL);

  // Create the pipeline layout
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .label                = "Graphics - Pipeline layout",
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &graphics.bind_group_layout,
  };
  graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &pipeline_layout_desc);
  ASSERT(graphics.pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0 : Fragment shader image
      .binding     = 0,
      .textureView = texture_compute_target.view,
    },
    [1] = (WGPUBindGroupEntry) {
       // Binding 1 : Fragment shader sampler
      .binding = 1,
      .sampler = texture_compute_target.sampler,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Graphics pipeline - Bind group",
    .layout     = graphics.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };

  graphics.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(graphics.bind_group != NULL);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
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
      .depth_write_enabled = false,
    });

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader SPIR-V
                      .label = "Texture - Vertex shader SPIR-V",
                      .file = "shaders/compute_ray_tracing/texture.vert.spv",
                    },
                    .buffer_count = 0,
                    .buffers      = NULL,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader SPIR-V
                      .label = "Fragment - Vertex shader SPIR-V",
                      .file = "shaders/compute_ray_tracing/texture.frag.spv",
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
  graphics.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Graphics - Render pipeline",
                            .layout       = graphics.pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

/* Prepare the compute pipeline that generates the ray traced image */
static void prepare_compute(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : Storage image (raytraced output) */
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .storageTexture = (WGPUStorageTextureBindingLayout) {
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA8Unorm,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1 : Uniform buffer block */
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = compute.uniform_buffer.size,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2: Shader storage buffer for the spheres */
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = compute.storage_buffers.spheres.size,
      },
      .sampler = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      /* Binding 3: Shader storage buffer for the planes */
      .binding    = 3,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = compute.storage_buffers.planes.size,
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = "Compute - Bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  compute.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(compute.bind_group_layout != NULL);

  WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &compute.bind_group_layout,
  };
  compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &compute_pipeline_layout_desc);
  ASSERT(compute.pipeline_layout != NULL);

  /* Compute pipeline bind group */
  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0: Output storage image */
      .binding     = 0,
      .textureView = texture_compute_target.view,
    },
    [1] = (WGPUBindGroupEntry) {
     /* Binding 1 : Uniform buffer */
      .binding = 1,
      .buffer  = compute.uniform_buffer.buffer,
      .offset  = 0,
      .size    = compute.uniform_buffer.size,
    },
    [2] = (WGPUBindGroupEntry) {
     /* Binding 2: Shader storage buffer for the spheres */
      .binding = 2,
      .buffer  = compute.storage_buffers.spheres.buffer,
      .offset  = 0,
      .size    = compute.storage_buffers.spheres.size,
    },
    [3] = (WGPUBindGroupEntry) {
     /* Binding 3: Shader storage buffer for the planes */
      .binding = 3,
      .buffer  = compute.storage_buffers.planes.buffer,
      .offset  = 0,
      .size    = compute.storage_buffers.planes.size,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Compute - Bind group",
    .layout     = compute.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  compute.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);

  /* Compute shader */
  wgpu_shader_t particle_comp_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    /* Compute shader SPIR-V */
                    .label = "Ray tracing - Compute shader",
                    .file  = "shaders/compute_ray_tracing/raytracing.comp.spv",
                  });

  /* Create pipeline */
  compute.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = "Compute pipeline",
      .layout  = compute.pipeline_layout,
      .compute = particle_comp_shader.programmable_stage_descriptor,
    });

  /* Partial clean-up */
  wgpu_shader_release(&particle_comp_shader);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  const float timer = context->timer;

  compute.ubo.lightPos[0]
    = 0.0f + sin(glm_rad(timer * 360.0f)) * cos(glm_rad(timer * 360.0f)) * 2.0f;
  compute.ubo.lightPos[1] = 0.0f + sin(glm_rad(timer * 360.0f)) * 2.0f;
  compute.ubo.lightPos[2] = 0.0f + cos(glm_rad(timer * 360.0f)) * 2.0f;
  glm_vec3_scale(context->camera->position, -1.0f, compute.ubo.camera.pos);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, compute.uniform_buffer.buffer,
                          0, &compute.ubo, compute.uniform_buffer.size);
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Default values */
  glm_vec4_zero(compute.ubo.fogColor);
  glm_vec3_copy((vec3){0.0f, 0.0f, 4.0f}, compute.ubo.camera.pos);
  glm_vec3_copy((vec3){0.0f, 0.5f, 0.0f}, compute.ubo.camera.lookat);
  compute.ubo.camera.fov  = 10.0f;
  compute.ubo.aspectRatio = context->window_size.aspect_ratio;
  context->timer_speed *= 0.25f;

  /* Compute shader parameter uniform buffer block */
  compute.uniform_buffer = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Compute shader parameter - Uniform buffer block",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(compute.ubo),
    });

  /* Update uniform buffer */
  update_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    prepare_storage_buffers(context->wgpu_context);
    prepare_uniform_buffers(context);
    prepare_texture_target(context->wgpu_context, &texture_compute_target,
                           TEX_DIM, TEX_DIM, WGPUTextureFormat_RGBA8Unorm);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepare_compute(context->wgpu_context);
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

  /* Compute pass: generated ray traced image */
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    /* Dispatch the compute job */
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                      compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                       compute.bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      wgpu_context->cpass_enc, texture_compute_target.size.width / 16,
      texture_compute_target.size.height / 16, 1);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  // Display ray traced image generated by compute shader as a full screen quad
  // Quad vertices are generated in the vertex shader
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      graphics.bind_group, 0, 0);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL);
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
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  // Textures
  wgpu_destroy_texture(&texture_compute_target);

  // Graphics
  WGPU_RELEASE_RESOURCE(BindGroupLayout, graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, graphics.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, graphics.pipeline)

  // Compute
  WGPU_RELEASE_RESOURCE(Buffer, compute.storage_buffers.spheres.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, compute.storage_buffers.planes.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, compute.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, compute.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, compute.pipeline)
}

void example_compute_ray_tracing(int argc, char* argv[])
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
