#include "context.h"

#include <stdlib.h>
#include <string.h>

#include "../core/log.h"
#include "../core/macro.h"
#include "../core/platform.h"

#include "../webgpu/buffer.h"

#include "../../lib/wgpu_native/wgpu_native.h"

/* WebGPU context creating/releasing */
wgpu_context_t* wgpu_context_create()
{
  wgpu_context_t* context = (wgpu_context_t*)malloc(sizeof(wgpu_context_t));
  memset(context, 0, sizeof(wgpu_context_t));
  return context;
}

void wgpu_context_release(wgpu_context_t* wgpu_context)
{
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
  // WebGPU device creation
  wgpu_context->device = wgpu_create_device(WGPUBackendType_Vulkan);
  wgpuDeviceSetUncapturedErrorCallback(
    wgpu_context->device, &wgpu_error_callback, (void*)wgpu_context);

  // Get the default queue from the device
  wgpu_context->queue = wgpuDeviceGetQueue(wgpu_context->device);
}

void wgpu_create_surface(wgpu_context_t* wgpu_context, void* window)
{
  wgpu_context->surface.instance
    = window_get_surface(wgpu_context->device, (window_t*)window);
  window_get_size((window_t*)window, &wgpu_context->surface.width,
                  &wgpu_context->surface.height);
}

void wgpu_setup_deph_stencil(wgpu_context_t* wgpu_context)
{
  WGPUTextureDescriptor depth_texture_desc = {
    .usage         = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
    .format        = WGPUTextureFormat_Depth24PlusStencil8,
    .dimension     = WGPUTextureDimension_2D,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .size          = (WGPUExtent3D) {
      .width               = wgpu_context->surface.width,
      .height              = wgpu_context->surface.height,
      .depthOrArrayLayers  = 1,
     },
  };
  wgpu_context->depth_stencil.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_texture_desc);

  WGPUTextureViewDescriptor depth_texture_view_dec = {
    .format          = WGPUTextureFormat_Depth24PlusStencil8,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  wgpu_context->depth_stencil.texture_view = wgpuTextureCreateView(
    wgpu_context->depth_stencil.texture, &depth_texture_view_dec);

  wgpu_context->depth_stencil.att_desc
    = (WGPURenderPassDepthStencilAttachmentDescriptor){
      .view           = wgpu_context->depth_stencil.texture_view,
      .depthLoadOp    = WGPULoadOp_Clear,
      .depthStoreOp   = WGPUStoreOp_Store,
      .clearDepth     = 1.0f,
      .stencilLoadOp  = WGPULoadOp_Clear,
      .stencilStoreOp = WGPUStoreOp_Store,
      .clearStencil   = 0,
    };
}

void wgpu_setup_swap_chain(wgpu_context_t* wgpu_context)
{
  wgpu_context->swap_chain.instance = wgpu_create_swap_chain(
    wgpu_context->device, wgpu_context->surface.instance,
    wgpu_context->surface.width, wgpu_context->surface.height);

  // Find a suitable depth format
  wgpu_context->swap_chain.format
    = wgpu_get_swap_chain_preferred_format(wgpu_context->device);
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

// Get a new command buffer
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

// End the command buffers and submit it to the queue
void wgpu_flush_command_buffers(wgpu_context_t* wgpu_context,
                                WGPUCommandBuffer* command_buffers,
                                uint32_t command_buffer_count)
{
  ASSERT(command_buffers != NULL)

  // Submit to the queue
  wgpuQueueSubmit(wgpu_context->queue, command_buffer_count, command_buffers);

  // Release command buffer
  for (uint32_t i = 0; i < command_buffer_count; ++i) {
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffers[i])
  }
}

void wgpu_swap_chain_present(wgpu_context_t* wgpu_context)
{
  wgpuSwapChainPresent(wgpu_context->swap_chain.instance);

  WGPU_RELEASE_RESOURCE(TextureView, wgpu_context->swap_chain.frame_buffer)
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
