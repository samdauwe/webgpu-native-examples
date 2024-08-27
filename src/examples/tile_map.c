#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

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
 * Tile set
 * -------------------------------------------------------------------------- */

typedef struct {
  texture_t texture;
  float tile_size;
} tile_set_t;

static void tile_set_create(tile_set_t* this, wgpu_context_t* wgpu_context,
                            const char* file_path, float tile_size)
{
  this->texture = wgpu_create_texture_from_file(wgpu_context, file_path, NULL);
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
  texture_t texture;
  struct {
    wgpu_buffer_t buffer;
    tile_data_t data;
    WGPUBindGroup bind_group;
  } uniform;
} tile_map_layer_t;

static void tile_map_layer_create(tile_map_layer_t* this,
                                  wgpu_context_t* wgpu_context,
                                  const char* texture_path,
                                  WGPUBindGroupLayout bind_group_layout,
                                  tile_set_t* tile_set,
                                  WGPUSampler tile_map_sampler)
{
  this->wgpu_context = wgpu_context;

  /* Init uniform data */
  {
    this->uniform.data.x         = 0.0f;
    this->uniform.data.y         = 0.0f;
    this->uniform.data.tile_size = tile_set->tile_size;
    this->uniform.data.scale     = 4.0f;
  }

  /* Tile map layer texture */
  this->texture
    = wgpu_create_texture_from_file(wgpu_context, texture_path, NULL);

  /* Create uniform buffer */
  this->uniform.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Tile map - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(tile_data_t),
                  });

  /* Create bind group */
  {
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
        .textureView = tile_set->texture.view,
      },
      [3] = (WGPUBindGroupEntry) {
        /* Binding 3 : Sprite sampler */
        .binding = 3,
        .sampler = tile_map_sampler,
      }
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Tile map - Layer bind group",
      .layout     = bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->uniform.bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->uniform.bind_group != NULL);
  }
}

static void tile_map_layer_destroy(tile_map_layer_t* this)
{
  wgpu_destroy_texture(&this->texture);
  wgpu_destroy_buffer(&this->uniform.buffer);
  WGPU_RELEASE_RESOURCE(BindGroup, this->uniform.bind_group)
}

static void tile_map_layer_write_uniform(tile_map_layer_t* this)
{
  wgpu_queue_write_buffer(this->wgpu_context, this->uniform.buffer.buffer, 0,
                          &this->uniform.data, this->uniform.buffer.size);
}

/* -------------------------------------------------------------------------- *
 * Tile map renderer
 * -------------------------------------------------------------------------- */

#define MAX_TILE_MAP_LAYER_COUNT 2u

typedef struct {
  wgpu_context_t* wgpu_context;
  WGPUTextureFormat color_format;
  WGPURenderPipeline pipeline;
  WGPUBindGroupLayout bind_group_layout;
  WGPUPipelineLayout pipeline_layout;
  WGPUSampler sampler;
  tile_set_t tile_set;
  tile_map_layer_t tile_map_layers[MAX_TILE_MAP_LAYER_COUNT];
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } render_pass;
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
                              .label         = "Tile map - Texture sampler",
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
                              .label      = "Tile map - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Pipeline layout */
  {
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = "Tile map - Render pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &this->bind_group_layout,
                            });
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Render pipeline */
  {
    /* Primitive state */
    WGPUPrimitiveState primitive_state_desc = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    /* Color target state */
    WGPUBlendState blend_state = (WGPUBlendState){
      .color.operation = WGPUBlendOperation_Add,
      .color.srcFactor = WGPUBlendFactor_SrcAlpha,
      .color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
      .alpha.operation = WGPUBlendOperation_Add,
      .alpha.srcFactor = WGPUBlendFactor_One,
      .alpha.dstFactor = WGPUBlendFactor_One,
    };
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .format    = this->color_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* Vertex state */
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
                      .shader_desc = (wgpu_shader_desc_t){
                        /* Vertex shader WGSL */
                        .label            = "Tile map - Vertex shader",
                        .wgsl_code.source = tile_map_shader_wgsl,
                        .entry            = "vertexMain",
                      },
                      .buffer_count = 0,
                      .buffers = NULL,
                    });

    /* Fragment state */
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
                      .shader_desc = (wgpu_shader_desc_t){
                        /* Fragment shader WGSL */
                        .label            = "Tile map - Fragment shader",
                        .wgsl_code.source = tile_map_shader_wgsl,
                        .entry            = "fragmentMain",
                      },
                      .target_count = 1,
                      .targets = &color_target_state_desc,
                    });

    /* Multisample state */
    WGPUMultisampleState multisample_state_desc
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    /* Create rendering pipeline using the specified states */
    this->pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label       = "Tile map - Render pipeline",
                              .layout      = this->pipeline_layout,
                              .primitive   = primitive_state_desc,
                              .vertex      = vertex_state_desc,
                              .fragment    = &fragment_state_desc,
                              .multisample = multisample_state_desc,
                            });

    /* Cleanup shaders */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }

  /* Render pass */
  {
    /* Color attachment */
    this->render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
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
    this->render_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = "Render pass descriptor",
      .colorAttachmentCount   = 1,
      .colorAttachments       = this->render_pass.color_attachments,
      .depthStencilAttachment = NULL,
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

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer
tile_map_renderer_build_command_buffer(tile_map_renderer_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;
  this->render_pass.color_attachments[0].view
    = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &this->render_pass.descriptor);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, this->pipeline);

  // Draw tile map layers: Layer rendering is back-to-front to ensure proper
  // transparency, which means there can be quite a bit of unnecessary overdraw.
  for (int32_t i = MAX_TILE_MAP_LAYER_COUNT - 1; i >= 0; --i) {
    wgpuRenderPassEncoderSetBindGroup(
      wgpu_context->rpass_enc, 0, this->tile_map_layers[i].uniform.bind_group,
      0, 0);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);
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

static int tile_map_renderer_draw(tile_map_renderer_t* this)
{
  wgpu_example_context_t* context = this->wgpu_context->context;

  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = tile_map_renderer_build_command_buffer(this);

  /* Submit to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
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
 * Tile map example
 * -------------------------------------------------------------------------- */

static struct {
  const char* tile_set;
  const char* tile_map_layers[MAX_TILE_MAP_LAYER_COUNT];
} texture_paths = {
  .tile_set = "textures/spelunky-tiles.png",
  .tile_map_layers = {
    "textures/spelunky0.png", /* Tile map layer 0 */
    "textures/spelunky1.png", /* Tile map layer 1 */
  },
};

/* Tile map renderer */
static tile_map_renderer_t tile_map_renderer = {0};

/* Other variables */
static const char* example_title = "Tile Map";
static bool prepared             = false;

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    wgpu_context_t* wgpu_context = context->wgpu_context;
    tile_map_renderer_create(&tile_map_renderer, wgpu_context,
                             wgpu_context->swap_chain.format);
    tile_map_renderer_create_tileset(&tile_map_renderer, texture_paths.tile_set,
                                     16.0f);
    for (uint32_t i = 0; i < MAX_TILE_MAP_LAYER_COUNT; ++i) {
      tile_map_renderer_create_tile_map_layer(&tile_map_renderer, i,
                                              texture_paths.tile_map_layers[i],
                                              &tile_map_renderer.tile_set);
    }
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  if (!context->paused) {
    /* Update the uniform data for every layer */
    tile_map_renderer_update_tile_map_layers(&tile_map_renderer,
                                             context->frame.timestamp_millis);
  }
  return tile_map_renderer_draw(&tile_map_renderer);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  tile_map_renderer_destroy(&tile_map_renderer);
}

void example_tile_map(int argc, char* argv[])
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
