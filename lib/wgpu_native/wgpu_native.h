#ifndef WGPU_NATIVE_H
#define WGPU_NATIVE_H

#include <dawn/webgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

void wgpu_log_available_adapters();
void wgpu_get_adapter_info(char (*adapter_info)[256]);
WGPUAdapter wgpu_request_adapter(WGPURequestAdapterOptions* options);
WGPUSurface wgpu_create_surface(void* display, void* window_handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // WGPU_NATIVE_H
