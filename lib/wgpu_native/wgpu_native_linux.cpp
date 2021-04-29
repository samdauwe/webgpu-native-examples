#include "wgpu_native.h"

#include <string.h>

#if __has_include("vulkan/vulkan.h")
#define DAWN_ENABLE_BACKEND_VULKAN
#endif

#ifdef DAWN_ENABLE_BACKEND_VULKAN
#include <vulkan/vulkan.h>
#endif

//****************************************************************************/

#include <dawn/dawn_proc.h>
#include <dawn/webgpu_cpp.h>
#ifdef DAWN_ENABLE_BACKEND_VULKAN
#include <dawn_native/VulkanBackend.h>
#endif
#include <dawn_native/NullBackend.h>

namespace impl {

/*
 * Chosen backend type for \c #device.
 */
WGPUBackendType backend;

/*
 * WebGPU graphics API-specific device, created from a \c dawn_native::Adapter
 * and optional feature requests. This should wrap the same underlying device
 * for the same configuration.
 */
WGPUDevice device;

/*
 * Something needs to hold onto this since the address is passed to the WebGPU
 * native API, exposing the type-specific swap chain implementation. The struct
 * gets filled out on calling the respective XXX::CreateNativeSwapChainImpl(),
 * binding the WebGPU device and native window, then its raw pointer is passed
 * into WebGPU as a 64-bit int. The browser API doesn't have an equivalent
 * (since the swap chain is created from the canvas directly).
 *
 * Is the struct copied or does it need holding for the lifecycle of the swap
 * chain, i.e. can it just be a temporary?
 *
 * After calling wgpuSwapChainRelease() does it also call swapImpl::Destroy()
 * to delete the underlying NativeSwapChainImpl(), invalidating this struct?
 */
static DawnSwapChainImplementation swapImpl;

/*
 * Preferred swap chain format, obtained in the browser via a promise to
 * GPUCanvasContext::getSwapChainPreferredFormat(). In Dawn we can call this
 * directly in NativeSwapChainImpl::GetPreferredFormat() (which is hard-coded
 * with D3D, for example, to RGBA8Unorm, but queried for others). For the D3D
 * back-end calling wgpuSwapChainConfigure ignores the passed preference and
 * asserts if it's not the preferred choice.
 */
static WGPUTextureFormat swapPref;

//********************************** Helpers *********************************/

/**
 * Analogous to the browser's \c GPU.requestAdapter().
 * \n
 * The returned \c Adapter is a wrapper around the underlying Dawn adapter (and
 * owned by the single Dawn instance).
 *
 * \todo we might be interested in whether the \c AdapterType is discrete or integrated for
 * power-management reasons
 *
 * \param[in] type1st first choice of \e backend type (e.g. \c WGPUBackendType_D3D12)
 * \param[in] type2nd optional fallback \e backend type (or \c WGPUBackendType_Null to pick the
 * first choice or nothing) \return the best choice adapter or an empty adapter wrapper
 */
static dawn_native::Adapter requestAdapter(WGPUBackendType type1st,
                                           WGPUBackendType type2nd = WGPUBackendType_Null)
{
  static dawn_native::Instance instance;
  instance.DiscoverDefaultAdapters();
  wgpu::AdapterProperties properties;
  std::vector<dawn_native::Adapter> adapters = instance.GetAdapters();
  for (auto it = adapters.begin(); it != adapters.end(); ++it) {
    it->GetProperties(&properties);
    if (static_cast<WGPUBackendType>(properties.backendType) == type1st) {
      return *it;
    }
  }
  if (type2nd) {
    for (auto it = adapters.begin(); it != adapters.end(); ++it) {
      it->GetProperties(&properties);
      if (static_cast<WGPUBackendType>(properties.backendType) == type2nd) {
        return *it;
      }
    }
  }
  return dawn_native::Adapter();
}

static const char* backendTypeName(wgpu::BackendType t) {
  switch (t) {
    case wgpu::BackendType::Null:     return "Null";
    case wgpu::BackendType::D3D11:    return "D3D11";
    case wgpu::BackendType::D3D12:    return "D3D12";
    case wgpu::BackendType::Metal:    return "Metal";
    case wgpu::BackendType::Vulkan:   return "Vulkan";
    case wgpu::BackendType::OpenGL:   return "OpenGL";
    case wgpu::BackendType::OpenGLES: return "OpenGLES";
  }
  return "?";
}

static const char* adapterTypeName(wgpu::AdapterType t) {
  switch (t) {
    case wgpu::AdapterType::DiscreteGPU:   return "DiscreteGPU";
    case wgpu::AdapterType::IntegratedGPU: return "IntegratedGPU";
    case wgpu::AdapterType::CPU:           return "CPU";
    case wgpu::AdapterType::Unknown:       return "Unknown";
  }
  return "?";
}

static void logAvailableAdapters() {
  static dawn_native::Instance instance;
  instance.DiscoverDefaultAdapters();
  fprintf(stderr, "Available adapters:\n");
  for (auto&& a : instance.GetAdapters()) {
    wgpu::AdapterProperties p;
    a.GetProperties(&p);
    fprintf(stderr, "  %s (%s)\n"
      "    deviceID=%u, vendorID=0x%x, BackendType::%s, AdapterType::%s\n",
      p.name, p.driverDescription,
      p.deviceID, p.vendorID, backendTypeName(p.backendType), adapterTypeName(p.adapterType));
  }
}

/**
 * Creates an API-specific swap chain implementation in \c #swapImpl and stores
 * the \c #swapPref.
 */
static void initSwapChain(WGPUBackendType backend, VkSurfaceKHR surface)
{
  switch (backend) {
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    case WGPUBackendType_Vulkan:
      if (impl::swapImpl.userData == nullptr) {
        impl::swapImpl = dawn_native::vulkan::CreateNativeSwapChainImpl(impl::device, surface);
        impl::swapPref = dawn_native::vulkan::GetNativeSwapChainPreferredFormat(&impl::swapImpl);
      }
      break;
#endif
    default:
      if (impl::swapImpl.userData == nullptr) {
        impl::swapImpl = dawn_native::null::CreateNativeSwapChainImpl();
        impl::swapPref = WGPUTextureFormat_Undefined;
      }
      break;
  }
}

/**
 * Dawn error handling callback (adheres to \c WGPUErrorCallback).
 *
 * \param[in] message error string
 */
static void printError(WGPUErrorType /*type*/, const char* message, void*)
{
  puts(message);
}

} // namespace impl

//******************************** Public API ********************************/

void wgpu_log_available_adapters()
{
  impl::logAvailableAdapters();
}

void* wgpu_get_backend_instance(WGPUDevice device)
{
  return dawn_native::vulkan::GetInstance(device);
}

WGPUDevice wgpu_create_device(WGPUBackendType type)
{
  if (type > WGPUBackendType_OpenGLES) {
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    type = WGPUBackendType_Vulkan;
#endif
  }
  if (dawn_native::Adapter adapter = impl::requestAdapter(type)) {
    wgpu::AdapterProperties properties;
    adapter.GetProperties(&properties);
    dawn_native::DeviceDescriptor devDesc;
    devDesc.requiredExtensions.push_back("texture_compression_bc");
    impl::backend = static_cast<WGPUBackendType>(properties.backendType);
    impl::device  = adapter.CreateDevice(&devDesc);
    if (!impl::device) {
      impl::device = adapter.CreateDevice();
    }
    DawnProcTable procs(dawn_native::GetProcs());
    procs.deviceSetUncapturedErrorCallback(impl::device, impl::printError, nullptr);
    dawnProcSetProcs(&procs);
  }
  return impl::device;
}

WGPUSwapChain wgpu_create_swap_chain(WGPUDevice device, void* surface, int width, int height)
{
  impl::initSwapChain(impl::backend, reinterpret_cast<VkSurfaceKHR>(surface));

  WGPUSwapChainDescriptor swapDesc = {};
  swapDesc.implementation          = reinterpret_cast<uintptr_t>(&impl::swapImpl);
  WGPUSwapChain swapchain          = wgpuDeviceCreateSwapChain(device, nullptr, &swapDesc);

  wgpuSwapChainConfigure(swapchain, impl::swapPref, WGPUTextureUsage_RenderAttachment, width,
                         height);
  return swapchain;
}

WGPUTextureFormat wgpu_get_swap_chain_preferred_format(WGPUDevice /*device*/)
{
  return impl::swapPref;
}
