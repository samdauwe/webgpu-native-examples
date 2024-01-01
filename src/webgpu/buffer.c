#include "buffer.h"

#include <string.h>

#include "../core/macro.h"

#include "context.h"

wgpu_buffer_t wgpu_create_buffer(struct wgpu_context_t* wgpu_context,
                                 const wgpu_buffer_desc_t* desc)
{
  /* Ensure that buffer size is a multiple of 4 */
  const uint32_t size = (desc->size + 3) & ~3;

  wgpu_buffer_t wgpu_buffer = {
    .usage = desc->usage,
    .size  = size,
    .count = desc->count,
  };

  WGPUBufferDescriptor buffer_desc = {
    .label            = desc->label,
    .usage            = desc->usage,
    .size             = size,
    .mappedAtCreation = false,
  };

  const uint32_t initial_size
    = (desc->initial.size == 0) ? desc->size : desc->initial.size;

  if (desc->initial.data && initial_size > 0 && initial_size <= desc->size) {
    buffer_desc.mappedAtCreation = true;
    WGPUBuffer buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(buffer != NULL);
    void* mapping = wgpuBufferGetMappedRange(buffer, 0, size);
    ASSERT(mapping != NULL);
    memcpy(mapping, desc->initial.data, initial_size);
    wgpuBufferUnmap(buffer);
    wgpu_buffer.buffer = buffer;
  }
  else {
    wgpu_buffer.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(wgpu_buffer.buffer != NULL);
  }
  return wgpu_buffer;
}

void wgpu_destroy_buffer(wgpu_buffer_t* buffer)
{
  WGPU_RELEASE_RESOURCE(Buffer, buffer->buffer)
}

void wgpu_record_copy_data_to_buffer(struct wgpu_context_t* wgpu_context,
                                     wgpu_buffer_t* buff, uint32_t buff_offset,
                                     uint32_t buff_size, const void* data,
                                     uint32_t data_size)
{
  ASSERT(wgpu_context->cmd_enc != NULL);
  ASSERT(data && data_size > 0);
  /*ASSERT(
    buff->buffer && buff_size >= buff_offset + data_size
    && (buff->usage & (WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc)));*/
  ASSERT(buff_size % 4 == 0);

  WGPUBufferDescriptor staging_buffer_desc = {
    .usage            = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc,
    .size             = buff_size,
    .mappedAtCreation = true,
  };

  WGPUBuffer staging
    = wgpuDeviceCreateBuffer(wgpu_context->device, &staging_buffer_desc);
  ASSERT(staging != NULL);
  void* mapping = wgpuBufferGetMappedRange(staging, 0, buff_size);
  ASSERT(mapping != NULL);
  memcpy(mapping, data, data_size);
  wgpuBufferUnmap(staging);

  wgpuCommandEncoderCopyBufferToBuffer(wgpu_context->cmd_enc, staging, 0,
                                       buff->buffer, buff_offset, buff_size);
  WGPU_RELEASE_RESOURCE(Buffer, staging);
}

WGPUCommandBuffer wgpu_copy_buffer_to_texture(
  struct wgpu_context_t* wgpu_context, WGPUImageCopyBuffer* buffer_copy_view,
  WGPUImageCopyTexture* texture_copy_view, WGPUExtent3D* texture_size)
{
  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  wgpuCommandEncoderCopyBufferToTexture(cmd_encoder, buffer_copy_view,
                                        texture_copy_view, texture_size);
  WGPUCommandBuffer command_buffer
    = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  ASSERT(command_buffer != NULL);

  return command_buffer;
}

void copy_padding_buffer(unsigned char* dst, unsigned char* src, int32_t width,
                         int32_t height, int32_t kPadding)
{
  unsigned char* s = src;
  unsigned char* d = dst;
  for (int32_t i = 0; i < height; ++i) {
    memcpy(d, s, width * 4);
    s += width * 4;
    d += kPadding * 4;
  }
}

uint64_t calc_constant_buffer_byte_size(uint64_t byte_size)
{
  return (byte_size + 255) & ~255;
}
