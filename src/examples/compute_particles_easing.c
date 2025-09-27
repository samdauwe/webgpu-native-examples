#include "webgpu/wgpu_common.h"

#define SOKOL_FETCH_IMPL
#include "sokol_fetch.h"

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader Particle Easing
 *
 * Particle system using compute shaders. Particle data is stored in a shader
 * storage buffer, particle movement is implemented using easing functions.
 *
 * Ref:
 * https://redcamel.github.io/webgpu/14_compute
 * https://github.com/redcamel/webgpu/tree/master/14_compute
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* particle_compute_shader_wgsl_p1;
static const char* particle_compute_shader_wgsl_p2;
static const char* particle_vertex_shader_wgsl;
static const char* particle_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Compute Shader Particle Easing example
 * -------------------------------------------------------------------------- */

#define PARTICLE_NUM (60000u)
#define PROPERTY_NUM (40u)
#define WORKGROUP_SIZE (256)

/* State struct */
static struct {
  wgpu_buffer_t vertices;
  float initial_particle_data[PARTICLE_NUM * PROPERTY_NUM];
  struct {
    WGPUBindGroupLayout uniforms_bind_group_layout;
    WGPUBindGroup uniforms_bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
  } graphics;
  struct {
    wgpu_buffer_t sim_param_buffer;
    wgpu_buffer_t particle_buffer;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup particle_bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
  } compute;
  wgpu_texture_t particle_texture;
  uint8_t file_buffer[64 * 64 * 4];
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  struct {
    float time;
    float min_life;
    float max_life;
  } sim_param_data;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_dscriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .sim_param_data = {
    .time     = 0.0f,     /* startTime time */
    .min_life = 2000.0f,  /* Min lifeRange  */
    .max_life = 10000.0f, /* Max lifeRange  */
  },
};

static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  const float t_scale                     = 0.005f;
  static const float vertex_buffer[6 * 6] = {
    // clang-format off
    -t_scale, -t_scale, 0.0f, 1.0f, 0.0f, 0.0f, //
     t_scale, -t_scale, 0.0f, 1.0f, 0.0f, 1.0f, //
    -t_scale,  t_scale, 0.0f, 1.0f, 1.0f, 0.0f, //
    //
    -t_scale,  t_scale, 0.0f, 1.0f, 1.0f, 0.0f, //
     t_scale, -t_scale, 0.0f, 1.0f, 0.0f, 1.0f, //
     t_scale,  t_scale, 0.0f, 1.0f, 1.0f, 1.0f, //
    // clang-format on
  };

  /* Create vertex buffer */
  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Particle - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_buffer),
                    .count = (uint32_t)ARRAY_SIZE(vertex_buffer),
                    .initial.data = vertex_buffer,
                  });
}

static void init_particle_buffer(wgpu_context_t* wgpu_context)
{
  /* Particle data */
  const float current_time = stm_sec(stm_now());
  for (uint32_t i = 0; i < (uint32_t)PARTICLE_NUM; ++i) {
    const float life = random_float() * 8000.0f + 2000.0f;
    const float age  = random_float() * life;
    state.initial_particle_data[PROPERTY_NUM * i + 0]
      = current_time - age;                                   // start time
    state.initial_particle_data[PROPERTY_NUM * i + 1] = life; // life
    // position
    state.initial_particle_data[PROPERTY_NUM * i + 4]
      = random_float() * 2 - 1; // x
    state.initial_particle_data[PROPERTY_NUM * i + 5]
      = random_float() * 2 - 1; // y
    state.initial_particle_data[PROPERTY_NUM * i + 6]
      = random_float() * 2 - 1; // z
    state.initial_particle_data[PROPERTY_NUM * i + 7] = 0.0f;
    // scale
    state.initial_particle_data[PROPERTY_NUM * i + 8]  = 0.0f; // scaleX
    state.initial_particle_data[PROPERTY_NUM * i + 9]  = 0.0f; // scaleY
    state.initial_particle_data[PROPERTY_NUM * i + 10] = 0.0f; // scaleZ
    state.initial_particle_data[PROPERTY_NUM * i + 11] = 0.0f;
    // x
    state.initial_particle_data[PROPERTY_NUM * i + 12] = 0.0f; // startValue
    state.initial_particle_data[PROPERTY_NUM * i + 13]
      = random_float() * 2.0f - 1.0f; // endValue
    state.initial_particle_data[PROPERTY_NUM * i + 14]
      = (int)(random_float() * 27.0f); // ease
    // y
    state.initial_particle_data[PROPERTY_NUM * i + 16] = 0.0f; // startValue
    state.initial_particle_data[PROPERTY_NUM * i + 17]
      = random_float() * 2.0f - 1.0f; // endValue
    state.initial_particle_data[PROPERTY_NUM * i + 18]
      = (int)(random_float() * 27.0f); // ease
    // z
    state.initial_particle_data[PROPERTY_NUM * i + 20] = 0.0f; // startValue
    state.initial_particle_data[PROPERTY_NUM * i + 21]
      = random_float() * 2.0f - 1.0f; // endValue
    state.initial_particle_data[PROPERTY_NUM * i + 22]
      = (int)(random_float() * 27.0f); // ease
    // scaleX
    const float t_scale                                = random_float() * 12.0f;
    state.initial_particle_data[PROPERTY_NUM * i + 24] = 0.0f;    // startValue
    state.initial_particle_data[PROPERTY_NUM * i + 25] = t_scale; // endValue
    state.initial_particle_data[PROPERTY_NUM * i + 26] = 0.0f;    // ease
    // scaleY
    state.initial_particle_data[PROPERTY_NUM * i + 28] = 0.0f;    // startValue
    state.initial_particle_data[PROPERTY_NUM * i + 29] = t_scale; // endValue
    state.initial_particle_data[PROPERTY_NUM * i + 30] = 0.0f;    // ease
    // scaleZ
    state.initial_particle_data[PROPERTY_NUM * i + 32] = 0.0f;    // startValue
    state.initial_particle_data[PROPERTY_NUM * i + 33] = t_scale; // endValue
    state.initial_particle_data[PROPERTY_NUM * i + 34] = 0.0f;    // ease
    // alpha
    state.initial_particle_data[PROPERTY_NUM * i + 36]
      = random_float();                                        // startValue
    state.initial_particle_data[PROPERTY_NUM * i + 37] = 0.0f; // endValue
    state.initial_particle_data[PROPERTY_NUM * i + 38]
      = (int)(random_float() * 27.0f);                         // ease
    state.initial_particle_data[PROPERTY_NUM * i + 39] = 0.0f; // value
  }

  /* Create vertex buffer */
  state.compute.particle_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute particle - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size         = sizeof(state.initial_particle_data),
                    .initial.data = state.initial_particle_data,
                  });
}

static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("File fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* The file data has been fetched, since we provided a big-enough buffer we
   * can be sure that all data has been loaded here */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc = (wgpu_texture_desc_t){
                .extent = (WGPUExtent3D) {
                  .width              = img_width,
                  .height             = img_height,
                  .depthOrArrayLayers = 4,
      },
                .format = WGPUTextureFormat_RGBA8Unorm,
                .pixels = {
                  .ptr  = pixels,
                  .size = img_width * img_height * 4,
      },
    };
    texture->desc.is_dirty = true;
  }
}

static void init_particle_texture(wgpu_context_t* wgpu_context)
{
  /* Dummy particle texture */
  state.particle_texture = wgpu_create_color_bars_texture(wgpu_context, 16, 16);

  /* Start loading the image file */
  const char* particle_texture_path = "assets/textures/particle.png";
  wgpu_texture_t* texture           = &state.particle_texture;
  sfetch_send(&(sfetch_request_t){
    .path      = particle_texture_path,
    .callback  = fetch_callback,
    .buffer    = SFETCH_RANGE(state.file_buffer),
    .user_data = {
      .ptr  = &texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });
}

static void init_compute_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : SimParams */
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.sim_param_data),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1 : ParticlesA */
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = sizeof(state.initial_particle_data),
      },
      .sampler = {0},
    }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = STRVIEW("Compute - Bind group layout"),
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  state.compute.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.compute.bind_group_layout != NULL)

  WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
    .label                = STRVIEW("Compute - Pipeline layout"),
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &state.compute.bind_group_layout,
  };
  state.compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &compute_pipeline_layout_desc);
  ASSERT(state.compute.pipeline_layout != NULL)
}

static void init_compute_bind_group(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : SimParams */
      .binding = 0,
      .buffer  = state.compute.sim_param_buffer.buffer,
      .size    =  state.compute.sim_param_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
     /* Binding 1 : Particles A */
      .binding = 1,
      .buffer  = state.compute.particle_buffer.buffer,
      .offset  = 0,
      .size    = state.compute.particle_buffer.size,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Compute pipeline - Bind group"),
    .layout     = state.compute.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.compute.particle_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
}

static char* get_full_compute_shader(void)
{
  size_t len1      = strlen(particle_compute_shader_wgsl_p1);
  size_t len2      = strlen(particle_compute_shader_wgsl_p2);
  size_t total_len = len1 + len2 + 1; /* +1 for null terminator */

  char* full_shader = malloc(total_len);
  if (full_shader == NULL) {
    return NULL; /* Handle allocation failure */
  }

  snprintf(full_shader, total_len, "%s%s", particle_compute_shader_wgsl_p1,
           particle_compute_shader_wgsl_p2);

  return full_shader;
}

static void init_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Compute shader */
  char* particle_compute_shader_wgsl = get_full_compute_shader();
  if (particle_compute_shader_wgsl == NULL) {
    printf("Could not create full compute shader");
    return;
  }
  WGPUShaderModule comp_shader_module = wgpu_create_shader_module(
    wgpu_context->device, particle_compute_shader_wgsl);
  free(particle_compute_shader_wgsl);

  /* Create pipeline */
  state.compute.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Particle - Compute pipeline"),
      .layout  = state.compute.pipeline_layout,
      .compute = {
        .module     = comp_shader_module,
        .entryPoint = STRVIEW("main"),
      },
    });
  ASSERT(state.compute.pipeline != NULL);

  /* Partial cleanup */
  wgpuShaderModuleRelease(comp_shader_module);
}

static void init_graphics_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Uniforms bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : sampler */
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1 : texture */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = STRVIEW("Uniforms - Bind group layout"),
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  state.graphics.uniforms_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.graphics.uniforms_bind_group_layout != NULL)

  /* Create the pipeline layout */
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .label                = STRVIEW("Render - Pipeline layout"),
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &state.graphics.uniforms_bind_group_layout,
  };
  state.graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &pipeline_layout_desc);
  ASSERT(state.graphics.pipeline_layout != NULL);
}

static void init_graphics_bind_group(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.uniforms_bind_group)

  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
       /* Binding 0 : sampler */
      .binding = 0,
      .sampler = state.particle_texture.sampler,
    },
    [1] = (WGPUBindGroupEntry) {
       /* Binding 1 : texture view */
      .binding     = 1,
      .textureView = state.particle_texture.view,
    }
  };

  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Graphics uniforms - Bind group"),
    .layout     = state.graphics.uniforms_bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };

  state.graphics.uniforms_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.graphics.uniforms_bind_group != NULL)
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.sim_param_data.time += (1.0 / 60.f) * 1000.0f;

  /* Map uniform buffer and update it */
  wgpuQueueWriteBuffer(
    wgpu_context->queue, state.compute.sim_param_buffer.buffer, 0,
    &state.sim_param_data, state.compute.sim_param_buffer.size);
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Compute shader uniform buffer block */
  state.compute.sim_param_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute shader - Uniform buffer block",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.sim_param_data),
                  });

  /* Update uniform buffer */
  update_uniform_buffers(wgpu_context);
}

static void init_graphics_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, particle_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, particle_fragment_shader_wgsl);

  /* Color target state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);
  {
    blend_state.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend_state.color.dstFactor = WGPUBlendFactor_One;
    blend_state.color.operation = WGPUBlendOperation_Add;
  }
  {
    blend_state.alpha.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend_state.alpha.dstFactor = WGPUBlendFactor_One;
    blend_state.alpha.operation = WGPUBlendOperation_Add;
  }

  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = false,
    });

  /* Vertex buffer layout */
  WGPUVertexBufferLayout buffers[2] = {0};

  /* Instanced particles buffer */
  buffers[0].arrayStride              = PROPERTY_NUM * 4;
  buffers[0].stepMode                 = WGPUVertexStepMode_Instance;
  buffers[0].attributeCount           = 3;
  WGPUVertexAttribute attributes_0[3] = {0};
  {
    /* position */
    attributes_0[0] = (WGPUVertexAttribute){
      .shaderLocation = 0,
      .offset         = 4 * 4,
      .format         = WGPUVertexFormat_Float32x3,
    };
    /* scale */
    attributes_0[1] = (WGPUVertexAttribute){
      .shaderLocation = 1,
      .offset         = 8 * 4,
      .format         = WGPUVertexFormat_Float32x3,
    };
    /* alpha */
    attributes_0[2] = (WGPUVertexAttribute){
      .shaderLocation = 2,
      .offset         = 39 * 4,
      .format         = WGPUVertexFormat_Float32,
    };
  }
  buffers[0].attributes = attributes_0;

  /* vertex buffer */
  buffers[1].arrayStride              = 6 * 4;
  buffers[1].stepMode                 = WGPUVertexStepMode_Vertex;
  buffers[1].attributeCount           = 2;
  WGPUVertexAttribute attributes_1[2] = {0};
  {
    /* position*/
    attributes_1[0] = (WGPUVertexAttribute){
      .shaderLocation = 3,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x4,
    };
    /* scale*/
    attributes_1[1] = (WGPUVertexAttribute){
      .shaderLocation = 4,
      .offset         = 4 * 4,
      .format         = WGPUVertexFormat_Float32x2,
    };
  }
  buffers[1].attributes = attributes_1;

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Particle - Render pipeline"),
    .layout = state.graphics.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = (uint32_t) ARRAY_SIZE(buffers),
      .buffers     = buffers,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets     = &color_target_state,
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.graphics.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.graphics.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
    });
    init_vertex_buffer(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_particle_buffer(wgpu_context);
    init_particle_texture(wgpu_context);
    init_compute_pipeline_layout(wgpu_context);
    init_compute_bind_group(wgpu_context);
    init_compute_pipeline(wgpu_context);
    init_graphics_pipeline_layout(wgpu_context);
    init_graphics_bind_group(wgpu_context);
    init_graphics_pipeline(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Recreate texture when pixel data loaded */
  if (state.particle_texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.particle_texture);
    FREE_TEXTURE_PIXELS(state.particle_texture);
    /* Upddate the bindgroup */
    init_graphics_bind_group(wgpu_context);
  }

  /* Update matrix data */
  update_uniform_buffers(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass: Compute particle movement */
  {
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    /* Dispatch the compute job */
    wgpuComputePassEncoderSetPipeline(cpass_enc, state.compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(
      cpass_enc, 0, state.compute.particle_bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      cpass_enc, (uint32_t)ceilf(PARTICLE_NUM / WORKGROUP_SIZE), 1, 1);
    wgpuComputePassEncoderEnd(cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, cpass_enc)
  }

  /* Render pass: Draw the particle system using the update vertex buffer */
  {
    state.color_attachment.view         = wgpu_context->swapchain_view;
    state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_dscriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                      state.graphics.uniforms_bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass_enc, 0, state.compute.particle_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 1, state.vertices.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(rpass_enc, 6, PARTICLE_NUM, 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Textures */
  wgpu_destroy_texture(&state.particle_texture);

  /* Vertices */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)

  /* Graphics pipeline */
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.graphics.uniforms_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.uniforms_bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)

  /* Compute pipeline */
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.sim_param_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.particle_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute.particle_bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Compute Shader Particle Easing",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* particle_compute_shader_wgsl_p1 = CODE(
  const PARTICLE_NUM: u32 = 60000u;

  struct Info {
    startValue: f32,
    endValue: f32,
    easeType: f32,
    value: f32,
  };

  struct InfoGroup {
    infoX: Info,
    infoY: Info,
    infoZ: Info,
  };

  struct Particle {
    startTime: f32,
    life: f32,
    valuePosition: vec4<f32>,
    valueScale: vec4<f32>,
    infoPosition: InfoGroup,
    infoScale: InfoGroup,
    infoAlpha: Info,
  };

  struct SimParams {
    time: f32,
    minLife: f32,
    maxLife: f32,
  };

  @group(0) @binding(0) var<uniform> params: SimParams;

  @group(0) @binding(1) var<storage, read_write> particlesA: array<Particle, PARTICLE_NUM>;

  fn rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
  }

  const PI: f32 = 3.141592653589793;
  const HPI: f32 = PI * 0.5;
  const PI2: f32 = PI * 2.0;

  fn calEasing(n_in: f32, type_in: f32) -> f32 {
    var n = n_in;
    switch (i32(type_in)) {
      // linear
      case 0: { }
      // QuintIn
      case 1: { n = n * n * n * n * n; }
      // QuintOut
      case 2: { n = ((n - 1.0) * n * n * n * n) + 1.0; }
      // QuintInOut
      case 3: {
        n = n * 2.0;
        if (n < 1.0) {
          n = n * n * n * n * n * 0.5;
        } else {
          n = n - 2.0;
          n = 0.5 * (n * n * n * n * n + 2.0);
        }
      }
      // BackIn
      case 4: { n = n * n * (n * 1.70158 + n - 1.70158); }
      // BackOut
      case 5: { n = (n - 1.0) * n * (n * 1.70158 + n + 1.70158) + 1.0; }
      // BackInOut
      case 6: {
        n = n * 2.0;
        if (n < 1.0) {
          n = 0.5 * n * n * (n * 1.70158 + n - 1.70158);
        } else {
          n = n - 2.0;
          n = 0.5 * (n * n * (n * 1.70158 + n + 1.70158) + 2.0);  // Note: Added +2 for consistency with other InOut patterns
        }
      }
      // CircIn
      case 7: { n = -1.0 * (sqrt(1.0 - n * n) - 1.0); }
      // CircOut
      case 8: { n = sqrt(1.0 - (n - 1.0) * n); }
      // CircInOut
      case 9: {
        n = n * 2.0;
        if (n < 1.0) {
          n = -0.5 * (sqrt(1.0 - n * n) - 1.0);
        } else {
          n = n - 2.0;
          n = 0.5 * (sqrt(1.0 - n * n) + 1.0);
        }
      }
      // CubicIn
      case 10: { n = n * n * n; }
      // CubicOut
      case 11: { n = ((n - 1.0) * n * n) + 1.0; }
      // CubicInOut
      case 12: {
        n = n * 2.0;
        if (n < 1.0) {
          n = n * n * n * 0.5;
        } else {
          n = n - 2.0;
          n = 0.5 * (n * n * n + 2.0);
        }
      }
      // ExpoIn
      case 13: {
        if (n == 0.0) {
          n = 0.0;
        } else {
          n = pow(2.0, 10.0 * (n - 1.0));
        }
      }
      // ExpoOut
      case 14: {
        if (n == 1.0) {
          n = 1.0;
        } else {
          n = -pow(2.0, -10.0 * n) + 1.0;
        }
      }
      // ExpoInOut
      case 15: {
        n = n * 2.0;
        if (n < 1.0) {
          if (n == 0.0) {
            n = 0.0;
          } else {
            n = 0.5 * pow(2.0, 10.0 * (n - 1.0));
          }
        } else {
          if (n == 2.0) {
            n = 1.0;
          } else {
            n = -0.5 * pow(2.0, -10.0 * (n - 1.0)) + 1.0;
          }
        }
      }
      // QuadIn
      case 16: { n = n * n; }
      // QuadOut
      case 17: { n = (2.0 - n) * n; }
      // QuadInOut
      case 18: {
        n = n * 2.0;
        if (n < 1.0) {
          n = n * n * 0.5;
        } else {
          n = n - 1.0;
          n = 0.5 * ((2.0 - n) * n + 1.0);
        }
      }
      // QuartIn
      case 19: { n = n * n * n * n; }
      // QuartOut
      case 20: { n = 1.0 - ((n - 1.0) * n * n * n); }
      // QuartInOut
      case 21: {
        n = n * 2.0;
        if (n < 1.0) {
          n = n * n * n * n * 0.5;
        } else {
          n = n - 2.0;
          n = 1.0 - (n * n * n * n * 0.5);
        }
      }
      // SineIn
      case 22: { n = -cos(n * HPI) + 1.0; }
      // SineOut
      case 23: { n = sin(n * HPI); }
      // SineInOut
      case 24: { n = (-cos(n * PI) + 1.0) * 0.5; }
      // ElasticIn
      case 25: {
        if (n == 0.0) {
          n = 0.0;
        } else if (n == 1.0) {
          n = 1.0;
        } else {
          n = n - 1.0;
          n = -1.0 * pow(2.0, 10.0 * n) * sin((n - 0.075) * PI2 / 0.3);
        }
      }
      // ElasticOut
      case 26: {
        if (n == 0.0) {
          n = 0.0;
        } else if (n == 1.0) {
          n = 1.0;
        } else {
          n = pow(2.0, -10.0 * n) * sin((n - 0.075) * PI2 / 0.3) + 1.0;
        }
      }
      // ElasticInOut
      case 27: {
        if (n == 0.0) {
          return 0.0;
        }
        if (n == 1.0) {
          return 1.0;
        }
        n = n * 2.0;
        if (n < 1.0) {
          n = n - 1.0;
          n = -0.5 * pow(2.0, 10.0 * n) * sin((n - 0.075) * PI2 / 0.3);
        } else {
          n = n - 1.0;
          n = 0.5 * pow(2.0, -10.0 * n) * sin((n - 0.075) * PI2 / 0.3) + 1.0;
        }
      }
      default: { }
    }
    return n;
  }
);

static const char* particle_compute_shader_wgsl_p2 = CODE(
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= PARTICLE_NUM) {
      return;
    }

    var targetParticle = particlesA[index];

    var n: f32;
    let age = params.time - targetParticle.startTime;
    var lifeRatio = age / targetParticle.life;

    if (lifeRatio >= 1.0) {
      particlesA[index].startTime = params.time;
      var t0 = rand(vec2<f32>(params.minLife, params.maxLife) + params.time) * params.maxLife;
      t0 = max(params.minLife, t0);
      particlesA[index].life = t0;
      lifeRatio = 0.0;
    }

    // position
    n = lifeRatio;
    n = calEasing(n, targetParticle.infoPosition.infoX.easeType);
    particlesA[index].valuePosition.x = targetParticle.infoPosition.infoX.startValue + (targetParticle.infoPosition.infoX.endValue - targetParticle.infoPosition.infoX.startValue) * n;
    n = lifeRatio;
    n = calEasing(n, targetParticle.infoPosition.infoY.easeType);
    particlesA[index].valuePosition.y = targetParticle.infoPosition.infoY.startValue + (targetParticle.infoPosition.infoY.endValue - targetParticle.infoPosition.infoY.startValue) * n;
    n = lifeRatio;
    n = calEasing(n, targetParticle.infoPosition.infoZ.easeType);
    particlesA[index].valuePosition.z = targetParticle.infoPosition.infoZ.startValue + (targetParticle.infoPosition.infoZ.endValue - targetParticle.infoPosition.infoZ.startValue) * n;

    // scale
    n = lifeRatio;
    n = calEasing(n, targetParticle.infoScale.infoX.easeType);
    particlesA[index].valueScale.x = targetParticle.infoScale.infoX.startValue + (targetParticle.infoScale.infoX.endValue - targetParticle.infoScale.infoX.startValue) * n;
    n = lifeRatio;
    n = calEasing(n, targetParticle.infoScale.infoY.easeType);
    particlesA[index].valueScale.y = targetParticle.infoScale.infoY.startValue + (targetParticle.infoScale.infoY.endValue - targetParticle.infoScale.infoY.startValue) * n;
    n = lifeRatio;
    n = calEasing(n, targetParticle.infoScale.infoZ.easeType);
    particlesA[index].valueScale.z = targetParticle.infoScale.infoZ.startValue + (targetParticle.infoScale.infoZ.endValue - targetParticle.infoScale.infoZ.startValue) * n;

    // alpha
    n = lifeRatio;
    n = calEasing(n, targetParticle.infoAlpha.easeType);
    particlesA[index].infoAlpha.value = targetParticle.infoAlpha.startValue + (targetParticle.infoAlpha.endValue - targetParticle.infoAlpha.startValue) * n;
  }
);

static const char* particle_vertex_shader_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) tUV: vec2<f32>,
    @location(1) vAlpha: f32,
  };

  @vertex
  fn main(
    @location(0) position: vec3<f32>,
    @location(1) scale: vec3<f32>,
    @location(2) alpha: f32,
    @location(3) a_pos: vec4<f32>,
    @location(4) a_uv: vec2<f32>,
  ) -> VertexOutput {
    var output: VertexOutput;

    let ratio: f32 = 976.0 / 1920.0;
    let scaleMTX: mat4x4<f32> = mat4x4<f32>(
      vec4<f32>(scale.x, 0.0, 0.0, 0.0),
      vec4<f32>(0.0, scale.y, 0.0, 0.0),
      vec4<f32>(0.0, 0.0, scale.z, 0.0),
      vec4<f32>(position, 1.0)
    );
    output.pos = scaleMTX * vec4<f32>(a_pos.x, a_pos.y / ratio, a_pos.z, 1.0);
    output.tUV = a_uv;
    output.vAlpha = alpha;

    return output;
  }
);

static const char* particle_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var uSampler: sampler;
  @group(0) @binding(1) var uTexture: texture_2d<f32>;

  struct FragmentOutput {
    @location(0) outColor: vec4<f32>,
  };

  @fragment
  fn main(
    @location(0) tUV: vec2<f32>,
    @location(1) vAlpha: f32,
  ) -> FragmentOutput {
    var output: FragmentOutput;

    output.outColor = textureSample(uTexture, uSampler, tUV);
    output.outColor = vec4<f32>(mix(output.outColor.rgb, vec3<f32>(vAlpha, 0.0, 0.0), 1.0 - vAlpha), output.outColor.a);
    output.outColor.a *= vAlpha;

    return output;
  }
);
// clang-format on
