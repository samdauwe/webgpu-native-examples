#include "webgpu/wgpu_common.h"

#define SOKOL_FETCH_IMPL
#include "sokol_fetch.h"

#define SOKOL_LOG_IMPL
#include "sokol_log.h"

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

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Tile Map
 *
 * This example shows how to render tile maps using WebGPU. The map is rendered
 * using two textures. One is the tileset, the other is a texture representing
 * the map itself. Each pixel encodes the x/y coords of the tile from the
 * tileset to draw.
 *
 * Ref:
 * https://github.com/toji/webgpu-test/tree/main/webgpu-tilemap
 * https://blog.tojicode.com/2012/07/sprite-tile-maps-on-gpu.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* tile_map_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Forward declaration
 * -------------------------------------------------------------------------- */

static void fetch_callback(const sfetch_response_t* response);

/* -------------------------------------------------------------------------- *
 * Tile set
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_texture_t texture;
  float tile_size;
  uint8_t file_buffer[128 * 128 * 4];
} tile_set_t;

static void tile_set_create(tile_set_t* this, wgpu_context_t* wgpu_context,
                            const char* file_path, float tile_size)
{
  this->texture = wgpu_create_color_bars_texture(wgpu_context, 16, 16);
  /*sfetch_send(&(sfetch_request_t){
    .path      = file_path,
    .callback  = fetch_callback,
    .buffer    = SFETCH_RANGE(this->file_buffer),
    .user_data = {
      .ptr = &this->texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });*/
  this->tile_size = tile_size;
}

static void tile_set_destroy(tile_set_t* this)
{
  wgpu_destroy_texture(&this->texture);
  this->tile_size = 0.0f;
}

/* -------------------------------------------------------------------------- *
 * Tile map layer
 * -------------------------------------------------------------------------- */

typedef struct {
  float x;
  float y;
  float tile_size;
  float scale;
  float padding[4];
} tile_data_t;

typedef struct {
  wgpu_context_t* wgpu_context;
  WGPUBindGroupLayout bind_group_layout;
  tile_set_t* tile_set;
  WGPUSampler tile_map_sampler;
  wgpu_texture_t texture;
  uint8_t file_buffer[64 * 64 * 4];
  struct {
    wgpu_buffer_t buffer;
    tile_data_t data;
    WGPUBindGroup bind_group;
  } uniform;
} tile_map_layer_t;

static void tile_map_layer_create_init_bind_group(tile_map_layer_t* this);

static void tile_map_layer_create(tile_map_layer_t* this,
                                  wgpu_context_t* wgpu_context,
                                  const char* texture_path,
                                  WGPUBindGroupLayout bind_group_layout,
                                  tile_set_t* tile_set,
                                  WGPUSampler tile_map_sampler)
{
  this->wgpu_context      = wgpu_context;
  this->bind_group_layout = bind_group_layout;
  this->tile_set          = tile_set;
  this->tile_map_sampler  = tile_map_sampler;

  /* Init uniform data */
  {
    this->uniform.data.x         = 0.0f;
    this->uniform.data.y         = 0.0f;
    this->uniform.data.tile_size = tile_set->tile_size;
    this->uniform.data.scale     = 4.0f;
  }

  /* Tile map layer texture */
  this->texture = wgpu_create_color_bars_texture(wgpu_context, 16, 16);

  /* Start loading the image file */
  /*sfetch_send(&(sfetch_request_t){
    .path      = texture_path,
    .callback  = fetch_callback,
    .buffer    = SFETCH_RANGE(this->file_buffer),
    .user_data = {
      .ptr = &this->texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });*/

  /* Create uniform buffer */
  this->uniform.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Tile map - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(tile_data_t),
                  });

  /* Create bind group */
  tile_map_layer_create_init_bind_group(this);
}

static void tile_map_layer_create_init_bind_group(tile_map_layer_t* this)
{
  WGPU_RELEASE_RESOURCE(BindGroup, this->uniform.bind_group)

  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Tile map uniforms */
      .binding = 0,
      .buffer  = this->uniform.buffer.buffer,
      .offset  = 0,
      .size    = this->uniform.buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1 : Map texture */
      .binding     = 1,
      .textureView = this->texture.view,
    },
    [2] = (WGPUBindGroupEntry) {
      /* Binding 2 : Sprite texture */
      .binding     = 2,
      .textureView = this->tile_set->texture.view,
    },
    [3] = (WGPUBindGroupEntry) {
      /* Binding 3 : Sprite sampler */
      .binding = 3,
      .sampler = this->tile_map_sampler,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Tile map - Layer bind group"),
    .layout     = this->bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  this->uniform.bind_group
    = wgpuDeviceCreateBindGroup(this->wgpu_context->device, &bg_desc);
  ASSERT(this->uniform.bind_group != NULL);
}

static void tile_map_layer_destroy(tile_map_layer_t* this)
{
  wgpu_destroy_texture(&this->texture);
  wgpu_destroy_buffer(&this->uniform.buffer);
  WGPU_RELEASE_RESOURCE(BindGroup, this->uniform.bind_group)
}

static void tile_map_layer_write_uniform(tile_map_layer_t* this)
{
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->uniform.buffer.buffer,
                       0, &this->uniform.data, this->uniform.buffer.size);
}

/* -------------------------------------------------------------------------- *
 * Tile map renderer
 * -------------------------------------------------------------------------- */

#define MAX_TILE_MAP_LAYER_COUNT (2u)

typedef struct {
  wgpu_context_t* wgpu_context;
  WGPUTextureFormat color_format;
  WGPURenderPipeline pipeline;
  WGPUBindGroupLayout bind_group_layout;
  WGPUPipelineLayout pipeline_layout;
  WGPUSampler sampler;
  tile_set_t tile_set;
  tile_map_layer_t tile_map_layers[MAX_TILE_MAP_LAYER_COUNT];
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
} tile_map_renderer_t;

static void tile_map_renderer_init_defaults(tile_map_renderer_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void tile_map_renderer_create(tile_map_renderer_t* this,
                                     wgpu_context_t* wgpu_context,
                                     WGPUTextureFormat color_format)
{
  tile_map_renderer_init_defaults(this);
  this->wgpu_context = wgpu_context;
  this->color_format = color_format;

  /* Tile map texture sampler */
  {
    this->sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label = STRVIEW("Tile map - Texture sampler"),
                              .addressModeU  = WGPUAddressMode_Repeat,
                              .addressModeV  = WGPUAddressMode_Repeat,
                              .addressModeW  = WGPUAddressMode_Repeat,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Nearest,
                              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(this->sampler != NULL);
  }

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0 : Tile map uniforms */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(tile_data_t),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1 : Map texture */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2 : Sprite texture */
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
        /* Binding 3 : Sprite sampler */
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("Tile map - Bind group layout"),
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Pipeline layout */
  {
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Tile map - Render pipeline layout"),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &this->bind_group_layout,
      });
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Render pipeline */
  {
    /* Color target state */
    WGPUBlendState blend_state = (WGPUBlendState){
      .color.operation = WGPUBlendOperation_Add,
      .color.srcFactor = WGPUBlendFactor_SrcAlpha,
      .color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
      .alpha.operation = WGPUBlendOperation_Add,
      .alpha.srcFactor = WGPUBlendFactor_One,
      .alpha.dstFactor = WGPUBlendFactor_One,
    };

    WGPUShaderModule tile_map_shader_module
      = wgpu_create_shader_module(wgpu_context->device, tile_map_shader_wgsl);

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Tile map - Render pipeline"),
      .layout = this->pipeline_layout,
      .vertex = {
        .module      = tile_map_shader_module,
        .entryPoint  = STRVIEW("vertexMain"),
      },
      .fragment = &(WGPUFragmentState) {
        .entryPoint  = STRVIEW("fragmentMain"),
        .module      = tile_map_shader_module,
        .targetCount = 1,
        .targets = &(WGPUColorTargetState) {
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_Back,
        .frontFace = WGPUFrontFace_CCW
      },
      .multisample = {
         .count = 1,
         .mask  = 0xffffffff
      },
    };

    this->pipeline
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(this->pipeline != NULL);

    /* Cleanup shaders */
    WGPU_RELEASE_RESOURCE(ShaderModule, tile_map_shader_module);
  }

  /* Render pass */
  {
    /* Color attachment */
    this->color_attachment = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
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
    this->render_pass_dscriptor = (WGPURenderPassDescriptor){
      .label                = STRVIEW("Render pass descriptor"),
      .colorAttachmentCount = 1,
      .colorAttachments     = &this->color_attachment,
    };
  }
}

static void tile_map_renderer_destroy(tile_map_renderer_t* this)
{
  for (uint32_t i = 0; i < MAX_TILE_MAP_LAYER_COUNT; ++i) {
    tile_map_layer_destroy(&this->tile_map_layers[i]);
  }
  tile_set_destroy(&this->tile_set);
  WGPU_RELEASE_RESOURCE(Sampler, this->sampler)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
}

static void tile_map_renderer_create_tileset(tile_map_renderer_t* this,
                                             const char* file_path,
                                             float tile_size)
{
  tile_set_destroy(&this->tile_set);
  tile_set_create(&this->tile_set, this->wgpu_context, file_path, tile_size);
}

static void tile_map_renderer_create_tile_map_layer(tile_map_renderer_t* this,
                                                    uint32_t layer_index,
                                                    const char* texture_path,
                                                    tile_set_t* tile_set)
{
  if (layer_index >= MAX_TILE_MAP_LAYER_COUNT) {
    return;
  }

  tile_map_layer_t* tile_map_layer = &this->tile_map_layers[layer_index];
  tile_map_layer_create(tile_map_layer, this->wgpu_context, texture_path,
                        this->bind_group_layout, tile_set, this->sampler);
}

static int tile_map_renderer_draw(tile_map_renderer_t* this)
{
  WGPUDevice device = this->wgpu_context->device;
  WGPUQueue queue   = this->wgpu_context->queue;

  this->color_attachment.view = this->wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &this->render_pass_dscriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, this->pipeline);

  // Draw tile map layers: Layer rendering is back-to-front to ensure proper
  // transparency, which means there can be quite a bit of unnecessary overdraw.
  for (int32_t i = MAX_TILE_MAP_LAYER_COUNT - 1; i >= 0; --i) {
    wgpuRenderPassEncoderSetBindGroup(
      rpass_enc, 0, this->tile_map_layers[i].uniform.bind_group, 0, 0);
    wgpuRenderPassEncoderDraw(rpass_enc, 3, 1, 0, 0);
  }

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void tile_map_renderer_update_textures(tile_map_renderer_t* this)
{
  int8_t is_dirty = this->tile_set.texture.desc.is_dirty;
  for (uint32_t i = 0; i < MAX_TILE_MAP_LAYER_COUNT; ++i) {
    is_dirty = is_dirty && this->tile_map_layers[i].texture.desc.is_dirty;
  }

  if (is_dirty) {
    /* Recreate tile set texture */
    wgpu_recreate_texture(this->wgpu_context, &this->tile_set.texture);
    FREE_TEXTURE_PIXELS(this->tile_set.texture);

    /* Recreate tile map layers texture */
    for (uint32_t i = 0; i < MAX_TILE_MAP_LAYER_COUNT; ++i) {
      wgpu_recreate_texture(this->wgpu_context,
                            &this->tile_map_layers[i].texture);
      FREE_TEXTURE_PIXELS(this->tile_map_layers[i].texture);
      /* Upddate the bind group */
      tile_map_layer_create_init_bind_group(&this->tile_map_layers[i]);
    }
  }
}

static void tile_map_renderer_update_tile_map_layers(tile_map_renderer_t* this,
                                                     float t)
{
  tile_data_t *layer0 = &this->tile_map_layers[0].uniform.data,
              *layer1 = &this->tile_map_layers[1].uniform.data;

  layer0->x = floor((sin(t / 1000.0f) + 1) * 256);
  layer0->y = floor((cos(t / 500.0f) + 1) * 256);
  layer1->x = floor(layer0->x / 2.0f);
  layer1->y = floor(layer0->y / 2.0f);

  layer1->scale = layer0->scale = (sin(t / 3000.0f) + 2) * 2;

  for (uint32_t i = 0; i < MAX_TILE_MAP_LAYER_COUNT; ++i) {
    tile_map_layer_write_uniform(&this->tile_map_layers[i]);
  }
}

/* -------------------------------------------------------------------------- *
 * Fetch callback
 * -------------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------------- *
 * Tile map example
 * -------------------------------------------------------------------------- */

static struct {
  const char* tile_set;
  const char* tile_map_layers[MAX_TILE_MAP_LAYER_COUNT];
} texture_paths = {
  .tile_set = "assets/textures/spelunky-tiles.png",
  .tile_map_layers = {
    "assets/textures/spelunky0.png", /* Tile map layer 0 */
    "assets/textures/spelunky1.png", /* Tile map layer 1 */
  },
};

/* Tile map renderer */
static tile_map_renderer_t tile_map_renderer = {0};

/* Other variables */
static int8_t initialized = false;

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 3,
      .num_channels = 1,
      .num_lanes    = 1,
      .logger.func  = slog_func,
    });
    tile_map_renderer_create(&tile_map_renderer, wgpu_context,
                             wgpu_context->render_format);
    tile_map_renderer_create_tileset(&tile_map_renderer, texture_paths.tile_set,
                                     16.0f);
    for (uint32_t i = 0; i < MAX_TILE_MAP_LAYER_COUNT; ++i) {
      tile_map_renderer_create_tile_map_layer(&tile_map_renderer, i,
                                              texture_paths.tile_map_layers[i],
                                              &tile_map_renderer.tile_set);
    }
    initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  if (!initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Update texture when pixel data loaded */
  tile_map_renderer_update_textures(&tile_map_renderer);

  /* Update the uniform data for every layer */
  tile_map_renderer_update_tile_map_layers(&tile_map_renderer,
                                           stm_ms(stm_now()));

  return tile_map_renderer_draw(&tile_map_renderer);
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  sfetch_shutdown();
  tile_map_renderer_destroy(&tile_map_renderer);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Tile Map",
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
static const char* tile_map_shader_wgsl = CODE(
  const pos = array<vec2f, 3>(
    vec2f(-1, -1), vec2f(-1, 3), vec2f(3, -1));

  @vertex
  fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = pos[i];
    return vec4f(p, 0, 1);
  }

  struct TilemapUniforms {
    viewOffset: vec2f,
    tileSize: f32,
    tileScale: f32,
  }
  @group(0) @binding(0) var<uniform> tileUniforms: TilemapUniforms;

  @group(0) @binding(1) var tileTexture: texture_2d<f32>;
  @group(0) @binding(2) var spriteTexture: texture_2d<f32>;
  @group(0) @binding(3) var spriteSampler: sampler;

  fn getTile(pixelCoord: vec2f) -> vec2u {
    let scaledTileSize = tileUniforms.tileSize * tileUniforms.tileScale;
    let texCoord = vec2u(pixelCoord / scaledTileSize) % textureDimensions(tileTexture);
    return vec2u(textureLoad(tileTexture, texCoord, 0).xy * 256);
  }

  fn getSpriteCoord(tile: vec2u, pixelCoord: vec2f) -> vec2f {
    let scaledTileSize = tileUniforms.tileSize * tileUniforms.tileScale;
    let texelSize = vec2f(1) / vec2f(textureDimensions(spriteTexture));
    let halfTexel = texelSize * 0.5;
    let tileRange = tileUniforms.tileSize * texelSize;

    // Get the UV within the tile
    let spriteCoord = ((pixelCoord % scaledTileSize) / scaledTileSize) * tileRange;

    // Clamp the coords to within half a texel of the edge of the sprite so that we
    // never accidentally sample from a neighboring sprite.
    let clampedSpriteCoord = clamp(spriteCoord, halfTexel, tileRange - halfTexel);

    // Get the UV of the upper left corner of the sprite
    let spriteOffset = vec2f(tile) * tileRange;

    // Return the real UV of the sprite to sample.
    return clampedSpriteCoord + spriteOffset;
  }

  @fragment
  fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let pixelCoord = pos.xy + tileUniforms.viewOffset;

    // Get the sprite index from the tilemap
    let tile = getTile(pixelCoord);

    // Get the UV within the sprite
    let spriteCoord = getSpriteCoord(tile, pixelCoord);

    // Sample the sprite and return the color
    let spriteColor = textureSample(spriteTexture, spriteSampler, spriteCoord);
    if ((tile.x == 256 && tile.y == 256) || spriteColor.a < 0.01) {
      discard;
    }
    return spriteColor;
  }
);
// clang-format on
