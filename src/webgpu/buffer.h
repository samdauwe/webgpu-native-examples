#ifndef BUFFER_H_
#define BUFFER_H_

#include <stdint.h>

#include <dawn/webgpu.h>

/* Forward declarations */
struct wgpu_context_t;

/* WebGPU buffer */
typedef struct wgpu_buffer_desc_t {
  const char* label;
  WGPUBufferUsage usage;
  uint32_t size;
  uint32_t count; /* numer of elements in the buffer (optional) */
  struct {
    const void* data;
    uint32_t size;
  } initial;
} wgpu_buffer_desc_t;

typedef struct wgpu_buffer_t {
  WGPUBuffer buffer;
  WGPUBufferUsage usage;
  uint32_t size;
  uint32_t count; /* numer of elements in the buffer (optional) */
} wgpu_buffer_t;

/* WebGPU buffer creating  / destroy */
wgpu_buffer_t wgpu_create_buffer(struct wgpu_context_t* wgpu_context,
                                 const wgpu_buffer_desc_t* desc);
void wgpu_destroy_buffer(wgpu_buffer_t* buffer);

/*
 * Copies data into buff.buffer via a temporary staging buffer, doesn't submit
 * the resulting command
 */
void wgpu_record_copy_data_to_buffer(struct wgpu_context_t* wgpu_context,
                                     wgpu_buffer_t* buff, uint32_t buff_offset,
                                     uint32_t buff_size, const void* data,
                                     uint32_t data_size);

WGPUCommandBuffer wgpu_copy_buffer_to_texture(
  struct wgpu_context_t* wgpu_context, WGPUImageCopyBuffer* buffer_copy_view,
  WGPUImageCopyTexture* texture_copy_view, WGPUExtent3D* texture_size);

void copy_padding_buffer(unsigned char* dst, unsigned char* src, int32_t width,
                         int32_t height, int32_t kPadding);
uint64_t calc_constant_buffer_byte_size(uint64_t byte_size);

#endif // BUFFER_H_
