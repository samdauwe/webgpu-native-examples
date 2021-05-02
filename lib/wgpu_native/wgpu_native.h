#ifndef WGPU_NATIVE_H
#define WGPU_NATIVE_H

#include <dawn/webgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

void wgpu_log_available_adapters();
void wgpu_get_adapter_info(char (*adapter_info)[256]);
void* wgpu_get_backend_instance(WGPUDevice device);
WGPUDevice wgpu_create_device(WGPUBackendType type);
WGPUSwapChain wgpu_create_swap_chain(WGPUDevice device, void* surface, int width, int height);
WGPUTextureFormat wgpu_get_swap_chain_preferred_format(WGPUDevice device);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // WGPU_NATIVE_H
