#include "context.h"

#include <stdlib.h>
#include <string.h>

#include "../core/log.h"
#include "../core/macro.h"
#include "../core/window.h"

#include "../webgpu/texture.h"

#include "../../lib/wgpu_native/wgpu_native.h"

/* WebGPU context creating/releasing */
wgpu_context_t* wgpu_context_create(wgpu_context_create_options_t* options)
{
  wgpu_context_t* context = (wgpu_context_t*)malloc(sizeof(wgpu_context_t));
  memset(context, 0, sizeof(wgpu_context_t));

  context->swap_chain.present_mode
    = options ?
        (options->vsync ? WGPUPresentMode_Fifo : WGPUPresentMode_Mailbox) :
        WGPUPresentMode_Mailbox;

  return context;
}

void wgpu_context_release(wgpu_context_t* wgpu_context)
{
  if (wgpu_context->texture_client != NULL) {
    wgpu_texture_client_destroy(wgpu_context->texture_client);
    wgpu_context->texture_client = NULL;
  }

  WGPU_RELEASE_RESOURCE(TextureView, wgpu_context->depth_stencil.texture_view);
  WGPU_RELEASE_RESOURCE(Texture, wgpu_context->depth_stencil.texture);
  WGPU_RELEASE_RESOURCE(SwapChain, wgpu_context->swap_chain.instance);
  WGPU_RELEASE_RESOURCE(Queue, wgpu_context->queue);
  WGPU_RELEASE_RESOURCE(Device, wgpu_context->device);

  free(wgpu_context);
}

/* WebGPU info functions */
void wgpu_get_context_info(char (*adapter_info)[256])
{
  wgpu_get_adapter_info(adapter_info);
}

/* WebGPU context helper functions */
WGPUBuffer wgpu_create_buffer_from_data(wgpu_context_t* wgpu_context,
                                        const void* data, size_t size,
                                        WGPUBufferUsage usage)
{
  WGPUBufferDescriptor buffer_desc = {
    .usage = WGPUBufferUsage_CopyDst | usage,
    .size  = size,
  };
  WGPUBuffer buffer
    = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
  wgpu_queue_write_buffer(wgpu_context, buffer, 0, data, size);
  return buffer;
}

void wgpu_create_device_and_queue(wgpu_context_t* wgpu_context)
{
  wgpu_log_available_adapters();

  /* WebGPU adapter creation */
  wgpu_context->adapter = wgpu_request_adapter(&(WGPURequestAdapterOptions){
    .powerPreference = WGPUPowerPreference_HighPerformance,
  });

  /* WebGPU device creation */
  WGPUFeatureName required_features[2] = {
    WGPUFeatureName_TextureCompressionBC,
    WGPUFeatureName_BGRA8UnormStorage,
  };
  WGPUDeviceDescriptor deviceDescriptor = {
    .requiredFeatureCount = (uint32_t)ARRAY_SIZE(required_features),
    .requiredFeatures     = required_features,
  };
  wgpu_context->device
    = wgpuAdapterCreateDevice(wgpu_context->adapter, &deviceDescriptor);
  wgpuDeviceSetUncapturedErrorCallback(
    wgpu_context->device, &wgpu_error_callback, (void*)wgpu_context);

  /* Query device features */
  static const WGPUFeatureName feature_names[WGPU_FEATURE_COUNT] = {
    WGPUFeatureName_Depth32FloatStencil8,
    WGPUFeatureName_TimestampQuery,
    WGPUFeatureName_TextureCompressionBC,
    WGPUFeatureName_TextureCompressionETC2,
    WGPUFeatureName_TextureCompressionASTC,
    WGPUFeatureName_IndirectFirstInstance,
    WGPUFeatureName_BGRA8UnormStorage,
    WGPUFeatureName_DepthClipControl,
    WGPUFeatureName_ShaderF16,
    WGPUFeatureName_DawnInternalUsages,
    WGPUFeatureName_DawnMultiPlanarFormats,
  };
  for (uint32_t i = 0; i < WGPU_FEATURE_COUNT; ++i) {
    wgpu_context->features[i].feature_name = feature_names[i];
    wgpu_context->features[i].is_supported
      = wgpuDeviceHasFeature(wgpu_context->device, feature_names[i]);
  }

  /* Get the default queue from the device */
  wgpu_context->queue = wgpuDeviceGetQueue(wgpu_context->device);
}

bool wgpu_has_feature(wgpu_context_t* wgpu_context,
                      WGPUFeatureName feature_name)
{
  bool has_feature = false;
  for (uint32_t i = 0; i < WGPU_FEATURE_COUNT; ++i) {
    if (wgpu_context->features[i].feature_name == feature_name) {
      has_feature = wgpu_context->features[i].is_supported;
      break;
    }
  }
  return has_feature;
}

void wgpu_setup_window_surface(wgpu_context_t* wgpu_context, void* window)
{
  wgpu_context->surface.instance = window_get_surface((window_t*)window);
  window_get_size((window_t*)window, &wgpu_context->surface.width,
                  &wgpu_context->surface.height);
}

void wgpu_setup_deph_stencil(
  wgpu_context_t* wgpu_context,
  struct deph_stencil_texture_creation_options_t* options)
{
  if ((wgpu_context->depth_stencil.texture != NULL)
      && (wgpu_context->depth_stencil.texture_view != NULL)) {
    return;
  }

  WGPUTextureFormat format = options != NULL ?
                               (options->format != WGPUTextureFormat_Undefined ?
                                  options->format :
                                  WGPUTextureFormat_Depth24PlusStencil8) :
                               WGPUTextureFormat_Depth24PlusStencil8;
  uint32_t sample_count = options != NULL ? MAX(1, options->sample_count) : 1;

  WGPUTextureDescriptor depth_texture_desc = {
    .usage         = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
    .format        = format,
    .dimension     = WGPUTextureDimension_2D,
    .mipLevelCount = 1,
    .sampleCount   = sample_count,
    .size          = (WGPUExtent3D) {
      .width               = wgpu_context->surface.width,
      .height              = wgpu_context->surface.height,
      .depthOrArrayLayers  = 1,
     },
  };
  wgpu_context->depth_stencil.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_texture_desc);

  WGPUTextureViewDescriptor depth_texture_view_dec = {
    .format          = depth_texture_desc.format,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  wgpu_context->depth_stencil.texture_view = wgpuTextureCreateView(
    wgpu_context->depth_stencil.texture, &depth_texture_view_dec);

  wgpu_context->depth_stencil.att_desc = (WGPURenderPassDepthStencilAttachment){
    .view              = wgpu_context->depth_stencil.texture_view,
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilClearValue = 0,
  };

  // stencilLoadOp & stencilStoreOp must be set if the attachment has stencil
  // aspect or stencilReadOnly is false
  if (format == WGPUTextureFormat_Depth24PlusStencil8) {
    wgpu_context->depth_stencil.att_desc.stencilLoadOp  = WGPULoadOp_Clear;
    wgpu_context->depth_stencil.att_desc.stencilStoreOp = WGPUStoreOp_Store;
  }
}

void wgpu_setup_swap_chain(wgpu_context_t* wgpu_context)
{
  /* Create the swap chain */
  WGPUSwapChainDescriptor swap_chain_descriptor = {
    .usage       = WGPUTextureUsage_RenderAttachment,
    .format      = WGPUTextureFormat_BGRA8Unorm,
    .width       = wgpu_context->surface.width,
    .height      = wgpu_context->surface.height,
    .presentMode = wgpu_context->swap_chain.present_mode,
  };
  if (wgpu_context->swap_chain.instance) {
    wgpuSwapChainRelease(wgpu_context->swap_chain.instance);
  }
  wgpu_context->swap_chain.instance = wgpuDeviceCreateSwapChain(
    wgpu_context->device, wgpu_context->surface.instance,
    &swap_chain_descriptor);
  ASSERT(wgpu_context->swap_chain.instance);

  /* Find a suitable depth format */
  wgpu_context->swap_chain.format = swap_chain_descriptor.format;
}

void wgpu_error_callback(WGPUErrorType error_type, char const* message,
                         void* userdata)
{
  UNUSED_VAR(userdata);

  const char* error_type_name = "";
  switch (error_type) {
    case WGPUErrorType_Validation:
      error_type_name = "Validation";
      break;
    case WGPUErrorType_OutOfMemory:
      error_type_name = "Out of memory";
      break;
    case WGPUErrorType_Unknown:
      error_type_name = "Unknown";
      break;
    case WGPUErrorType_DeviceLost:
      error_type_name = "Device lost";
      break;
    default:
      return;
  }

  log_error("Error(%d) %s: %s", (int)error_type, error_type_name, message);
}

/* Methods of Queue */
void wgpu_queue_write_buffer(wgpu_context_t* wgpu_context, WGPUBuffer buffer,
                             uint64_t buffer_offset, void const* data,
                             size_t size)
{
  wgpuQueueWriteBuffer(wgpu_context->queue, buffer, buffer_offset, data, size);
}

/* Render helper functions */

/* Get a new command buffer */
WGPUCommandBuffer wgpu_get_command_buffer(WGPUCommandEncoder cmd_encoder)
{
  return wgpuCommandEncoderFinish(cmd_encoder, NULL);
}

WGPUTextureView wgpu_swap_chain_get_current_image(wgpu_context_t* wgpu_context)
{
  wgpu_context->swap_chain.frame_buffer
    = wgpuSwapChainGetCurrentTextureView(wgpu_context->swap_chain.instance);
  return wgpu_context->swap_chain.frame_buffer;
}

/* End the command buffers and submit it to the queue */
void wgpu_flush_command_buffers(wgpu_context_t* wgpu_context,
                                WGPUCommandBuffer* command_buffers,
                                uint32_t command_buffer_count)
{
  ASSERT(command_buffers != NULL)

  /* Submit to the queue */
  wgpuQueueSubmit(wgpu_context->queue, command_buffer_count, command_buffers);

  /* Release command buffer */
  for (uint32_t i = 0; i < command_buffer_count; ++i) {
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffers[i])
  }
}

void wgpu_swap_chain_present(wgpu_context_t* wgpu_context)
{
  wgpuSwapChainPresent(wgpu_context->swap_chain.instance);

  WGPU_RELEASE_RESOURCE(TextureView, wgpu_context->swap_chain.frame_buffer)
}

/* Texture client creation */
void wgpu_create_texture_client(wgpu_context_t* wgpu_context)
{
  if (wgpu_context->texture_client == NULL) {
    wgpu_context->texture_client = wgpu_texture_client_create(wgpu_context);
  }
}

/* Pipeline state factories */
WGPUBlendState wgpu_create_blend_state(bool enable_blend)
{
  WGPUBlendComponent blend_component_descriptor = {
    .operation = WGPUBlendOperation_Add,
  };

  if (enable_blend) {
    blend_component_descriptor.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend_component_descriptor.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
  }
  else {
    blend_component_descriptor.srcFactor = WGPUBlendFactor_One;
    blend_component_descriptor.dstFactor = WGPUBlendFactor_Zero;
  }

  return (WGPUBlendState){
    .color = blend_component_descriptor,
    .alpha = blend_component_descriptor,
  };
}

WGPUDepthStencilState
wgpu_create_depth_stencil_state(create_depth_stencil_state_desc_t* desc)
{
  WGPUStencilFaceState stencil_state_face_descriptor = {
    .compare     = WGPUCompareFunction_Always,
    .failOp      = WGPUStencilOperation_Keep,
    .depthFailOp = WGPUStencilOperation_Keep,
    .passOp      = WGPUStencilOperation_Keep,
  };

  return (WGPUDepthStencilState){
    .depthWriteEnabled   = desc->depth_write_enabled,
    .format              = desc->format,
    .depthCompare        = WGPUCompareFunction_LessEqual,
    .stencilFront        = stencil_state_face_descriptor,
    .stencilBack         = stencil_state_face_descriptor,
    .stencilReadMask     = 0xFFFFFFFF,
    .stencilWriteMask    = 0xFFFFFFFF,
    .depthBias           = 0,
    .depthBiasSlopeScale = 0.0f,
    .depthBiasClamp      = 0.0f,
  };
}

WGPUMultisampleState
wgpu_create_multisample_state_descriptor(create_multisample_state_desc_t* desc)
{
  return (WGPUMultisampleState){
    .count                  = desc ? desc->sample_count : 1,
    .mask                   = 0xFFFFFFFF,
    .alphaToCoverageEnabled = false,
  };
}
