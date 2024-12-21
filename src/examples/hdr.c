#include "example_base.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - High Dynamic Range Rendering
 *
 * Implements a high dynamic range rendering pipeline using 16/32 bit floating
 * point precision for all internal formats, textures and calculations,
 * including a bloom pass, manual exposure and tone mapping.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/hdr
 * -------------------------------------------------------------------------- */

#define NUMBER_OF_CONSTANTS 3u
#define ALIGNMENT 256u /* 256-byte alignment */

static bool bloom          = false;
static bool display_skybox = true;

static struct {
  texture_t envmap;
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
  .objects = {
    // clang-format off
    { .name = "Sphere",    .filelocation = "models/sphere.gltf"    },
    { .name = "Teapot",    .filelocation = "models/teapot.gltf"    },
    { .name = "Torusknot", .filelocation = "models/torusknot.gltf" },
    { .name = "Venus",     .filelocation = "models/venus.gltf"     },
    // clang-format on
  },
  .object_index = 1,
};
static const char* object_names[4] = {"Sphere", "Teapot", "Torusknot", "Venus"};

static struct {
  wgpu_buffer_t matrices;
  wgpu_buffer_t params;
  struct {
    WGPUBuffer buffer;
    uint64_t buffer_size;
    uint64_t model_size;
  } dynamic;
} uniform_buffers = {0};

static struct {
  mat4 projection;
  mat4 model_view;
  mat4 inverse_modelview;
} ubo_matrices = {0};

static struct {
  float exposure;
} ubo_params = {
  .exposure = 1.0f,
};

static struct {
  int value;
  uint8_t padding[252];
} ubo_constants[NUMBER_OF_CONSTANTS] = {0};

static struct {
  WGPURenderPipeline skybox;
  WGPURenderPipeline reflect;
  WGPURenderPipeline composition;
  WGPURenderPipeline bloom[2];
} pipelines = {0};

static struct {
  WGPUPipelineLayout models;
  WGPUPipelineLayout composition;
  WGPUPipelineLayout bloom_filter;
} pipeline_layouts = {0};

static struct {
  WGPUBindGroup object;
  WGPUBindGroup skybox;
  WGPUBindGroup composition;
  WGPUBindGroup bloom_filter;
} bind_groups = {0};

static struct {
  WGPUBindGroupLayout models;
  WGPUBindGroupLayout composition;
  WGPUBindGroupLayout bloom_filter;
} bind_group_layouts = {0};

typedef enum wgpu_render_pass_attachment_type_t {
  WGPU_RENDER_PASS_COLOR_ATTACHMENT_TYPE         = 0x00000001,
  WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_TYPE = 0x00000002,
} wgpu_render_pass_attachment_type_t;

/* Framebuffer for offscreen rendering */
typedef struct {
  WGPUTexture texture;
  WGPUTextureView texture_view;
  WGPUTextureFormat format;
} frame_buffer_attachment_t;

static struct {
  uint32_t width, height;
  frame_buffer_attachment_t color[2];
  frame_buffer_attachment_t depth;
  struct {
    WGPURenderPassColorAttachment color_attachment[2];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass_desc;
  WGPUSampler sampler;
} offscreen_pass = {0};

static struct {
  uint32_t width, height;
  frame_buffer_attachment_t color[1];
  struct {
    WGPURenderPassColorAttachment color_attachment[1];
    WGPURenderPassDescriptor render_pass_descriptor;
  } render_pass_desc;
  WGPUSampler sampler;
} filter_pass = {0};

static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

static WGPUTextureFormat depth_format = WGPUTextureFormat_Depth24PlusStencil8;

static const char* example_title = "High Dynamic Range Rendering";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 0.0f, -6.0f});
  camera_set_rotation(context->camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 256.0f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  /* Load glTF models */
  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_FlipY;
  models.skybox
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/cube.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(models.objects); ++i) {
    models.objects[i].object
      = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
        .wgpu_context       = wgpu_context,
        .filename           = models.objects[i].filelocation,
        .file_loading_flags = gltf_loading_flags,
      });
  }
  /* Load cube map */
  static const char* cubemap[6] = {
    "textures/cubemaps/uffizi_cube_px.png", /* Right  */
    "textures/cubemaps/uffizi_cube_nx.png", /* Left   */
    "textures/cubemaps/uffizi_cube_py.png", /* Top    */
    "textures/cubemaps/uffizi_cube_ny.png", /* Bottom */
    "textures/cubemaps/uffizi_cube_pz.png", /* Back   */
    "textures/cubemaps/uffizi_cube_nz.png", /* Front  */
  };
  textures.envmap = wgpu_create_texture_cubemap_from_files(
    wgpu_context, cubemap,
    &(struct wgpu_texture_load_options_t){
      .flip_y = true, /* Flip y to match uffizi_cube_nz.ktx hdr cubemap */
    });
  ASSERT(textures.envmap.texture != NULL);
}

void create_attachment(wgpu_context_t* wgpu_context, const char* texture_label,
                       WGPUTextureFormat format,
                       wgpu_render_pass_attachment_type_t attachment_type,
                       frame_buffer_attachment_t* attachment)
{
  /* Create the texture extent */
  WGPUExtent3D texture_extent = {
    .width              = offscreen_pass.width,
    .height             = offscreen_pass.height,
    .depthOrArrayLayers = 1,
  };

  /* Texture usage flags */
  WGPUTextureUsageFlags usage_flags = WGPUTextureUsage_RenderAttachment;
  if (attachment_type == WGPU_RENDER_PASS_COLOR_ATTACHMENT_TYPE) {
    usage_flags = usage_flags | WGPUTextureUsage_TextureBinding;
  }
  else if (attachment_type == WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_TYPE) {
    usage_flags = usage_flags | WGPUTextureUsage_CopySrc;
  }

  /* Texture format */
  attachment->format = format;

  /* Create the texture */
  WGPUTextureDescriptor texture_desc = {
    .label         = texture_label,
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = attachment->format,
    .usage         = usage_flags,
  };
  attachment->texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(attachment->texture);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "Texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  attachment->texture_view
    = wgpuTextureCreateView(attachment->texture, &texture_view_dec);
  ASSERT(attachment->texture_view);
}

/**
 * Prepare a new framebuffer and attachments for offscreen rendering (G-Buffer)
 */
static void prepare_offscreen(wgpu_context_t* wgpu_context)
{
  /* Offscreen render pass */
  {
    offscreen_pass.width  = wgpu_context->surface.width;
    offscreen_pass.height = wgpu_context->surface.height;

    /* Color attachments */

    /* Two floating point color buffers */
    create_attachment(
      wgpu_context, "Offscreen - Color texure 1", WGPUTextureFormat_RGBA8Unorm,
      WGPU_RENDER_PASS_COLOR_ATTACHMENT_TYPE, &offscreen_pass.color[0]);
    create_attachment(
      wgpu_context, "Offscreen - Color texure 2", WGPUTextureFormat_RGBA8Unorm,
      WGPU_RENDER_PASS_COLOR_ATTACHMENT_TYPE, &offscreen_pass.color[1]);
    /* Depth attachment */
    create_attachment(wgpu_context, "Offscreen - Depth texture", depth_format,
                      WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_TYPE,
                      &offscreen_pass.depth);

    /* Init attachment properties */

    /* Color attachment */
    for (uint32_t i = 0; i < 2; ++i) {
      offscreen_pass.render_pass_desc.color_attachment[i]
          = (WGPURenderPassColorAttachment) {
            .view       = offscreen_pass.color[i].texture_view,
            .depthSlice = ~0,
            .loadOp     = WGPULoadOp_Clear,
            .storeOp    = WGPUStoreOp_Store,
            .clearValue = (WGPUColor) {
              .r = 0.0f,
              .g = 0.0f,
              .b = 0.0f,
              .a = 0.0f,
            },
        };
    }

    /* Depth stencil attachment */
    offscreen_pass.render_pass_desc.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view              = offscreen_pass.depth.texture_view,
        .depthLoadOp       = WGPULoadOp_Clear,
        .depthStoreOp      = WGPUStoreOp_Store,
        .depthClearValue   = 1.0f,
        .stencilLoadOp     = WGPULoadOp_Clear,
        .stencilStoreOp    = WGPUStoreOp_Store,
        .stencilClearValue = 0,
      };

    /* Render pass descriptor */
    offscreen_pass.render_pass_desc.render_pass_descriptor
      = (WGPURenderPassDescriptor){
        .label                = "Offscreen - Render pass descriptor",
        .colorAttachmentCount = 2,
        .colorAttachments = offscreen_pass.render_pass_desc.color_attachment,
        .depthStencilAttachment
        = &offscreen_pass.render_pass_desc.depth_stencil_attachment,
      };

    /* Create sampler to sample from the color attachments */
    offscreen_pass.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = "Texture - Sampler",
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Nearest,
                              .magFilter     = WGPUFilterMode_Nearest,
                              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(offscreen_pass.sampler != NULL);
  }

  /* Bloom separable filter pass */
  {
    filter_pass.width  = wgpu_context->surface.width;
    filter_pass.height = wgpu_context->surface.height;

    /* Color attachments */

    /* Floating point color buffer */
    create_attachment(
      wgpu_context, "Bloom color - Texture", WGPUTextureFormat_RGBA8Unorm,
      WGPU_RENDER_PASS_COLOR_ATTACHMENT_TYPE, &filter_pass.color[0]);

    /* Init attachment properties */

    /* Color attachment */
    filter_pass.render_pass_desc.color_attachment[0]
          = (WGPURenderPassColorAttachment) {
            .view       = filter_pass.color[0].texture_view,
            .depthSlice = ~0,
            .loadOp     = WGPULoadOp_Clear,
            .storeOp    = WGPUStoreOp_Store,
            .clearValue = (WGPUColor) {
              .r = 0.0f,
              .g = 0.0f,
              .b = 0.0f,
              .a = 0.0f,
            },
        };

    /* Render pass descriptor */
    filter_pass.render_pass_desc.render_pass_descriptor
      = (WGPURenderPassDescriptor){
        .label                  = "Filter - Render pass descriptor",
        .colorAttachmentCount   = 1,
        .colorAttachments       = filter_pass.render_pass_desc.color_attachment,
        .depthStencilAttachment = NULL,
      };

    /* Create sampler to sample from the color attachment */
    filter_pass.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = "Filter pass - Texture sampler",
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Nearest,
                              .magFilter     = WGPUFilterMode_Nearest,
                              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
  }
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout for models */
  {
    WGPUBindGroupLayoutEntry bgl_entries[5] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Vertex / fragment shader uniform buffer */
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(ubo_matrices),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: Fragment shader image view */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2: Fragment shader image sampler */
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        /* Binding 3: Fragment shader uniform buffer */
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(ubo_params),
        },
        .sampler = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        /* Binding 4:  Vertex / fragment shader dynamic uniform buffer */
        .binding    = 4,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = sizeof(ubo_constants[0].value),
        },
        .sampler = {0},
      },
    };

    /* Create the bind group layout */
    bind_group_layouts.models = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Models - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.models != NULL);

    /* Create the pipeline layout */
    pipeline_layouts.models = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = "Models - Pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &bind_group_layouts.models,
                            });
    ASSERT(pipeline_layouts.models != NULL);
  }

  /* Bind group layout for bloom filter & G-Buffer composition */
  {
    WGPUBindGroupLayoutEntry bgl_entries[5] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Fragment shader image view */
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
        /* Binding 1: Fragment shader image sampler */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2: Fragment shader image view */
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
        /* Binding 3: Fragment shader image sampler */
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        /* Binding 4: fragment shader dynamic uniform buffer */
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = sizeof(ubo_constants[0].value),
        },
        .sampler = {0},
      },
    };

    /* Bloom filter pipeline layout */
    {
      /* Create the bind group layout */
      bind_group_layouts.bloom_filter = wgpuDeviceCreateBindGroupLayout(
        wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                                .label = "Bloom filter - Bind group layout",
                                .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                                .entries    = bgl_entries,
                              });
      ASSERT(bind_group_layouts.bloom_filter != NULL);

      /* Create the pipeline layout */
      pipeline_layouts.bloom_filter = wgpuDeviceCreatePipelineLayout(
        wgpu_context->device,
        &(WGPUPipelineLayoutDescriptor){
          .label                = "Bloom filter - Pipeline layout",
          .bindGroupLayoutCount = 1,
          .bindGroupLayouts     = &bind_group_layouts.bloom_filter,
        });
      ASSERT(pipeline_layouts.bloom_filter != NULL);
    }

    /* G-Buffer composition */
    {
      /* Create the bind group layout */
      bind_group_layouts.composition = wgpuDeviceCreateBindGroupLayout(
        wgpu_context->device,
        &(WGPUBindGroupLayoutDescriptor){
          .label      = "G-Buffer composition - Bind group layout",
          .entryCount = 4u,
          .entries    = bgl_entries,
        });
      ASSERT(bind_group_layouts.composition != NULL);

      /* Create the pipeline layout */
      pipeline_layouts.composition = wgpuDeviceCreatePipelineLayout(
        wgpu_context->device,
        &(WGPUPipelineLayoutDescriptor){
          .label                = "G-Buffer composition - Pipeline layout",
          .bindGroupLayoutCount = 1,
          .bindGroupLayouts     = &bind_group_layouts.composition,
        });
      ASSERT(pipeline_layouts.composition != NULL);
    }
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Model bind groups */
  {
    WGPUBindGroupEntry bg_entries[5] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0: Vertex / fragment shader uniform buffer */
        .binding = 0,
        .buffer  = uniform_buffers.matrices.buffer,
        .offset  = 0,
        .size    = uniform_buffers.matrices.size,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: Fragment shader image view */
        .binding     = 1,
        .textureView = textures.envmap.view,
      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 2: Fragment shader image sampler */
        .binding = 2,
        .sampler = textures.envmap.sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        /* Binding 3: Fragment shader uniform buffer */
        .binding = 3,
        .buffer  = uniform_buffers.params.buffer,
        .offset  = 0,
        .size    = uniform_buffers.params.size,
      },
      [4] = (WGPUBindGroupEntry) {
        /* Binding 4: Vertex / fragment shader dynamic uniform buffer */
        .binding = 4,
        .buffer  = uniform_buffers.dynamic.buffer,
        .offset  = 0,
        .size    = sizeof(ubo_constants[0].value),
      },
    };

    /* 3D object bind group */
    {
      WGPUBindGroupDescriptor bg_desc = {
        .label      = "3D object - Bind group",
        .layout     = bind_group_layouts.models,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      };
      bind_groups.object
        = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
      ASSERT(bind_groups.object != NULL);
    }

    /* Skybox bind group */
    {
      WGPUBindGroupDescriptor bg_desc = {
        .label      = "Skybox - Bind group",
        .layout     = bind_group_layouts.models,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      };
      bind_groups.skybox
        = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
      ASSERT(bind_groups.skybox != NULL);
    }
  }

  /* Bloom filter bind group */
  {
    WGPUBindGroupEntry bg_entries[5] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0: Fragment shader image view */
        .binding     = 0,
        .textureView = offscreen_pass.color[0].texture_view
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: Fragment shader image sampler */
        .binding = 1,
        .sampler = offscreen_pass.sampler,
      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 2: Fragment shader image view */
        .binding     = 2,
        .textureView = offscreen_pass.color[1].texture_view
      },
      [3] = (WGPUBindGroupEntry) {
        /* Binding 3: Fragment shader image sampler */
        .binding = 3,
        .sampler = offscreen_pass.sampler,
      },
      [4] = (WGPUBindGroupEntry) {
        /* Binding 4: fragment shader dynamic uniform buffer */
        .binding = 4,
        .buffer  = uniform_buffers.dynamic.buffer,
        .offset  = 0,
        .size    = sizeof(ubo_constants[0].value),
      },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Bloom filter - Bind group",
      .layout     = bind_group_layouts.bloom_filter,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.bloom_filter
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.bloom_filter != NULL);
  }

  /* Composition bind group */
  {
    WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0: Fragment shader image view */
        .binding    = 0,
        .textureView = offscreen_pass.color[0].texture_view
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: Fragment shader image sampler */
        .binding = 1,
        .sampler = offscreen_pass.sampler,
      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 2: Fragment shader image view */
        .binding     = 2,
        .textureView = filter_pass.color[0].texture_view
      },
      [3] = (WGPUBindGroupEntry) {
        /*Binding 3: Fragment shader image sampler */
        .binding = 3,
        .sampler = filter_pass.sampler,
      },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Composition - Bind group",
      .layout     = bind_group_layouts.composition,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    bind_groups.composition
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(bind_groups.composition != NULL);
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
        .r = 0.025f,
        .g = 0.025f,
        .b = 0.025f,
        .a = 1.000f,
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = (uint32_t)ARRAY_SIZE(rp_color_att_descriptors),
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  /* Multisample state */
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Full screen pipelines */

  /* Final fullscreen composition pass pipeline */
  {
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* Vertex state */
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
          wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader SPIR-V */
              .label = "Composition - Vertex shader SPIR-V",
              .file  = "shaders/hdr/composition.vert.spv",
            },
            /* Empty vertex input state, full screen triangles are generated by the vertex shader */
            .buffer_count = 0,
            .buffers      = NULL,
          });

    /* Fragment state */
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
          wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader SPIR-V */
              .label = "Composition - Fragment shader SPIR-V",
              .file  = "shaders/hdr/composition.frag.spv",
            },
            .target_count = 1,
            .targets      = &color_target_state_desc,
          });

    /* Create rendering pipeline using the specified states */
    pipelines.composition = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Composition - Render pipeline",
                              .layout       = pipeline_layouts.composition,
                              .primitive    = primitive_state_desc,
                              .vertex       = vertex_state_desc,
                              .fragment     = &fragment_state_desc,
                              .depthStencil = &depth_stencil_state_desc,
                              .multisample  = multisample_state_desc,
                            });
    ASSERT(pipelines.composition != NULL);

    /* Partial cleanup */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }

  /* Bloom pass */
  {
    /* Additive blending */
    WGPUBlendState blend_state_radial_blur = {
      .color.operation = WGPUBlendOperation_Add,
      .color.srcFactor = WGPUBlendFactor_One,
      .color.dstFactor = WGPUBlendFactor_One,
      .alpha.operation = WGPUBlendOperation_Add,
      .alpha.srcFactor = WGPUBlendFactor_SrcAlpha,
      .alpha.dstFactor = WGPUBlendFactor_DstAlpha,
    };
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .blend     = &blend_state_radial_blur,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* Vertex state */
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
          wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader SPIR-V */
              .label = "Bloom - Vertex shader SPIR-V",
              .file  = "shaders/hdr/bloom.vert.spv",
            },
            /* Empty vertex input state, full screen triangles are generated by the vertex shader */
            .buffer_count = 0,
            .buffers = NULL,
          });

    /* Fragment state */
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
          wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader SPIR-V */
              .label = "Bloom - Fragment shader SPIR-V",
              .file  = "shaders/hdr/bloom.frag.spv",
            },
            .target_count = 1,
            .targets = &color_target_state_desc,
          });

    /* First bloom filter pass (into separate framebuffer) */
    color_target_state_desc.format = wgpu_context->swap_chain.format;
    pipelines.bloom[0]             = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                          .label        = "Bloom 1 - Render pipeline",
                                          .layout       = pipeline_layouts.bloom_filter,
                                          .primitive    = primitive_state_desc,
                                          .vertex       = vertex_state_desc,
                                          .fragment     = &fragment_state_desc,
                                          .depthStencil = &depth_stencil_state_desc,
                                          .multisample  = multisample_state_desc,
                            });
    ASSERT(pipelines.bloom[0] != NULL);

    /* Second bloom filter pass (into separate framebuffer) */
    color_target_state_desc.format = filter_pass.color[0].format;
    pipelines.bloom[1]             = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                          .label        = "Bloom 2 - render pipeline",
                                          .layout       = pipeline_layouts.bloom_filter,
                                          .primitive    = primitive_state_desc,
                                          .vertex       = vertex_state_desc,
                                          .fragment     = &fragment_state_desc,
                                          .depthStencil = NULL,
                                          .multisample  = multisample_state_desc,
                            });
    ASSERT(pipelines.bloom[1] != NULL);
  }

  /* Object rendering pipelines */
  {
    /* Use vertex input state from glTF model setup */
    WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
      gltf_model,
      /* Location 0: Position */
      WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
      /* Location 1: Vertex normal */
      WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_Normal));

    /* Color target state */
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state_desc[2] = {
      [0] = (WGPUColorTargetState){
        .format    = offscreen_pass.color[0].format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
      [1] = (WGPUColorTargetState){
        .format    = offscreen_pass.color[1].format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    };

    /* Vertex state */
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
          wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader SPIR-V */
              .label = "G-Buffer - Vertex shader SPIR-V",
              .file  = "shaders/hdr/gbuffer.vert.spv",
            },
            .buffer_count = 1,
            .buffers = &gltf_model_vertex_buffer_layout,
          });

    /* Fragment state */
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
          wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader SPIR-V */
              .label = "G-Buffer - Fragment shader SPIR-V",
              .file = "shaders/hdr/gbuffer.frag.spv",
            },
            .target_count = 2,
            .targets = color_target_state_desc,
          });

    /* Skybox pipeline (background cube) */
    {
      primitive_state_desc.cullMode              = WGPUCullMode_Back;
      depth_stencil_state_desc.depthWriteEnabled = false;

      /* Create rendering pipeline using the specified  */
      pipelines.skybox = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                .label        = "Skybox - Render pipeline",
                                .layout       = pipeline_layouts.models,
                                .primitive    = primitive_state_desc,
                                .vertex       = vertex_state_desc,
                                .fragment     = &fragment_state_desc,
                                .depthStencil = &depth_stencil_state_desc,
                                .multisample  = multisample_state_desc,
                              });
      ASSERT(pipelines.skybox);
    }

    /* Object rendering pipeline */
    {
      /* Enable depth write */
      depth_stencil_state_desc.depthWriteEnabled = true;
      /* Flip cull mode */
      primitive_state_desc.cullMode = WGPUCullMode_Front;

      /* Create rendering pipeline using the specified states */
      pipelines.reflect = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                .label        = "Reflect - Render pipeline",
                                .layout       = pipeline_layouts.models,
                                .primitive    = primitive_state_desc,
                                .vertex       = vertex_state_desc,
                                .fragment     = &fragment_state_desc,
                                .depthStencil = &depth_stencil_state_desc,
                                .multisample  = multisample_state_desc,
                              });
      ASSERT(pipelines.reflect);
    }

    /* Partial cleanup */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  camera_t* camera = context->camera;

  glm_mat4_copy(camera->matrices.perspective, ubo_matrices.projection);
  glm_mat4_copy(camera->matrices.view, ubo_matrices.model_view);
  glm_mat4_inv(camera->matrices.view, ubo_matrices.inverse_modelview);

  wgpu_queue_write_buffer(context->wgpu_context,
                          uniform_buffers.matrices.buffer, 0, &ubo_matrices,
                          uniform_buffers.matrices.size);
}

static void update_params(wgpu_example_context_t* context)
{
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.params.buffer,
                          0, &ubo_params, uniform_buffers.params.size);
}

static void update_dynamic_uniform_buffers(wgpu_example_context_t* context)
{
  /* Set constant values */
  for (uint32_t i = 0u; i < NUMBER_OF_CONSTANTS; ++i) {
    ubo_constants[i].value = (int32_t)i;
  }

  /* Update buffer */
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.dynamic.buffer,
                          0, &ubo_constants,
                          uniform_buffers.dynamic.buffer_size);
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Matrices vertex shader uniform buffer */
  uniform_buffers.matrices = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Matrices vertex shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_matrices),
    });

  /* Params */
  uniform_buffers.params = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Parameters - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_matrices),
    });

  /* Uniform buffer object with constants */
  uniform_buffers.dynamic.model_size  = sizeof(int);
  uniform_buffers.dynamic.buffer_size = sizeof(ubo_constants);
  uniform_buffers.dynamic.buffer      = wgpuDeviceCreateBuffer(
    context->wgpu_context->device,
    &(WGPUBufferDescriptor){
           .label            = "Object with constants - Uniform buffer",
           .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
           .size             = uniform_buffers.dynamic.buffer_size,
           .mappedAtCreation = false,
    });

  /* Initialize uniform buffers */
  update_uniform_buffers(context);
  update_params(context);
  update_dynamic_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_uniform_buffers(context);
    prepare_offscreen(context->wgpu_context);
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
    if (imgui_overlay_combo_box(context->imgui_overlay, "Object Type",
                                &models.object_index, object_names, 4)) {
      update_uniform_buffers(context);
    }
    if (imgui_overlay_input_float(context->imgui_overlay, "Exposure",
                                  &ubo_params.exposure, 0.025f, "%.3f")) {
      update_params(context);
    }
    imgui_overlay_checkBox(context->imgui_overlay, "Bloom", &bloom);
    imgui_overlay_checkBox(context->imgui_overlay, "Skybox", &display_skybox);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /*
   * First pass: Render scene to offscreen framebuffer
   */
  {
    /* Create render pass encoder for encoding drawing commands */
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc,
      &offscreen_pass.render_pass_desc.render_pass_descriptor);

    /* Set viewport */
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                     (float)offscreen_pass.width,
                                     (float)offscreen_pass.height, 0.0f, 1.0f);

    /* Set scissor rectangle */
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        offscreen_pass.width,
                                        offscreen_pass.height);

    /* Skybox */
    if (display_skybox) {
      wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                       pipelines.skybox);
      uint32_t dynamic_offset = 0 * (uint32_t)ALIGNMENT;
      wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                        bind_groups.skybox, 1, &dynamic_offset);
      wgpu_gltf_model_draw(models.skybox,
                           (wgpu_gltf_model_render_options_t){0});
    }

    /* 3D oject */
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.reflect);
    uint32_t dynamic_offset = 1 * (uint32_t)ALIGNMENT;
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.object, 1, &dynamic_offset);
    wgpu_gltf_model_draw(models.objects[models.object_index].object,
                         (wgpu_gltf_model_render_options_t){0});

    /* End render pass */
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /*
   * Second render pass: First bloom pass
   */
  if (bloom) {
    /* Bloom filter */

    /* Create render pass encoder for encoding drawing commands */
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc,
      &filter_pass.render_pass_desc.render_pass_descriptor);

    /* Set viewport */
    wgpuRenderPassEncoderSetViewport(wgpu_context->rpass_enc, 0.0f, 0.0f,
                                     (float)filter_pass.width,
                                     (float)filter_pass.height, 0.0f, 1.0f);

    /* Set scissor rectangle */
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        filter_pass.width, filter_pass.height);

    /* Render */
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.bloom[1]);
    uint32_t dynamic_offset = 0 * (uint32_t)ALIGNMENT;
    wgpuRenderPassEncoderSetBindGroup(
      wgpu_context->rpass_enc, 0, bind_groups.bloom_filter, 1, &dynamic_offset);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);

    /* End render pass */
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /*
   * Third render pass: Scene rendering with applied second bloom pass (when
   * enabled)
   */
  {
    /* Final composition */

    /* Set target frame buffer */
    rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

    /* Create render pass encoder for encoding drawing commands */
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);

    /* Set viewport */
    wgpuRenderPassEncoderSetViewport(
      wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
      (float)wgpu_context->surface.height, 0.0f, 1.0f);

    /* Set scissor rectangle */
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        wgpu_context->surface.width,
                                        wgpu_context->surface.height);

    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.composition, 0, 0);

    /* Scene */
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.composition);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);

    /* Bloom */
    if (bloom) {
      wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                       pipelines.bloom[0]);
      uint32_t dynamic_offset = 1 * (uint32_t)ALIGNMENT;
      wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                        bind_groups.bloom_filter, 1,
                                        &dynamic_offset);
      wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);
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
  int result = example_draw(context);
  if (context->camera->updated) {
    update_uniform_buffers(context);
  }
  return result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);

  wgpu_destroy_texture(&textures.envmap);

  wgpu_gltf_model_destroy(models.skybox);
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(models.objects); ++i) {
    wgpu_gltf_model_destroy(models.objects[i].object);
  }

  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.matrices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.params.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.dynamic.buffer)

  WGPU_RELEASE_RESOURCE(Texture, offscreen_pass.color[0].texture)
  WGPU_RELEASE_RESOURCE(Texture, offscreen_pass.color[1].texture)
  WGPU_RELEASE_RESOURCE(Texture, offscreen_pass.depth.texture)
  WGPU_RELEASE_RESOURCE(TextureView, offscreen_pass.color[0].texture_view)
  WGPU_RELEASE_RESOURCE(TextureView, offscreen_pass.color[1].texture_view)
  WGPU_RELEASE_RESOURCE(TextureView, offscreen_pass.depth.texture_view)
  WGPU_RELEASE_RESOURCE(Sampler, offscreen_pass.sampler)

  WGPU_RELEASE_RESOURCE(Texture, filter_pass.color[0].texture)
  WGPU_RELEASE_RESOURCE(TextureView, filter_pass.color[0].texture_view)
  WGPU_RELEASE_RESOURCE(Sampler, filter_pass.sampler)

  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.skybox)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.reflect)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.composition)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.bloom[0])
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.bloom[1])

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.models)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.composition)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.bloom_filter)

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.object)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.skybox)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.composition)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.bloom_filter)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.models)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.composition)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.bloom_filter)
}

void example_hdr(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func= &example_initialize,
    .example_render_func    = &example_render,
    .example_destroy_func   = &example_destroy,
  });
  // clang-format on
}
