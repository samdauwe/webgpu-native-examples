#include "wgpu_native.h"

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp.h>

#include <string.h>

#include <cstdio>
#include <memory>

//****************************** Implementation *******************************/

namespace WGPUImpl {

#ifdef DEBUG
#define dlog(format, ...)                                                      \
  ({                                                                           \
    fprintf(stderr, format " \e[2m(%s %d)\e[0m\n", ##__VA_ARGS__,              \
            __FUNCTION__, __LINE__);                                           \
    fflush(stderr);                                                            \
  })
#else
#define dlog(...)                                                              \
  do {                                                                         \
  } while (0)
#endif

static const char* BackendTypeName(wgpu::BackendType);
static const char* AdapterTypeName(wgpu::AdapterType);

static struct {
  struct {
    DawnProcTable procTable;
    std::unique_ptr<dawn::native::Instance> instance = nullptr;
  } dawn_native;
  struct {
    dawn::native::Adapter handle;
    wgpu::BackendType backendType;
    struct {
      const char* name;
      const char* typeName;
      const char* backendName;
    } info;
  } adapter;
  bool initialized = false;
} gpuContext = {};

static void Initialize()
{
  if (gpuContext.initialized) {
    return;
  }

  // Set up the native procs for the global proctable
  gpuContext.dawn_native.procTable = dawn::native::GetProcs();
  dawnProcSetProcs(&gpuContext.dawn_native.procTable);
  gpuContext.dawn_native.instance = std::make_unique<dawn::native::Instance>();
  // Discovers adapters
  (void)gpuContext.dawn_native.instance->EnumerateAdapters();
  gpuContext.dawn_native.instance->EnableBackendValidation(true);
  gpuContext.dawn_native.instance->SetBackendValidationLevel(
    dawn::native::BackendValidationLevel::Full);

  // Dawn backend type.
  // Default to D3D12, Metal, Vulkan, OpenGL in that order as D3D12 and Metal
  // are the preferred on their respective platforms, and Vulkan is preferred to
  // OpenGL
  gpuContext.adapter.backendType =
#if defined(DAWN_ENABLE_BACKEND_D3D12)
    wgpu::BackendType::D3D12;
#elif defined(DAWN_ENABLE_BACKEND_METAL)
    wgpu::BackendType::Metal;
#elif defined(DAWN_ENABLE_BACKEND_VULKAN)
    wgpu::BackendType::Vulkan;
#elif defined(DAWN_ENABLE_BACKEND_OPENGL)
    wgpu::BackendType::OpenGL;
#else
#error
#endif
  gpuContext.adapter.handle = nullptr;
  gpuContext.initialized    = true;
}

static void SetAdapterInfo(const wgpu::AdapterProperties& ap)
{
  gpuContext.adapter.info.name        = ap.name;
  gpuContext.adapter.info.typeName    = AdapterTypeName(ap.adapterType);
  gpuContext.adapter.info.backendName = BackendTypeName(ap.backendType);
}

static WGPUAdapter RequestAdapter(WGPURequestAdapterOptions* options)
{
  Initialize();

  WGPUPowerPreference powerPreference
    = options ?
        (options->powerPreference == WGPUPowerPreference_HighPerformance ?
           WGPUPowerPreference_HighPerformance :
           WGPUPowerPreference_LowPower) :
        WGPUPowerPreference_LowPower;

  // Search available adapters for a good match, in the following priority
  // order
  std::vector<wgpu::AdapterType> typePriority;
  if (powerPreference == WGPUPowerPreference_LowPower) {
    // low power
    typePriority = std::vector<wgpu::AdapterType>{
      wgpu::AdapterType::IntegratedGPU,
      wgpu::AdapterType::DiscreteGPU,
      wgpu::AdapterType::CPU,
    };
  }
  else if (powerPreference == WGPUPowerPreference_HighPerformance) {
    // high performance
    typePriority = std::vector<wgpu::AdapterType>{
      wgpu::AdapterType::DiscreteGPU,
      wgpu::AdapterType::IntegratedGPU,
      wgpu::AdapterType::CPU,
    };
  }

  std::vector<dawn::native::Adapter> adapters
    = gpuContext.dawn_native.instance->EnumerateAdapters();
  for (auto reqType : typePriority) {
    for (const dawn::native::Adapter& adapter : adapters) {
      wgpu::AdapterProperties ap;
      adapter.GetProperties(&ap);
      if (ap.adapterType == reqType
          && (reqType == wgpu::AdapterType::CPU
              || ap.backendType == gpuContext.adapter.backendType)) {
        gpuContext.adapter.handle = adapter;
        SetAdapterInfo(ap);
        dlog("Selected adapter %s (device=0x%x vendor=0x%x type=%s/%s)",
             ap.name, ap.deviceID, ap.vendorID,
             gpuContext.adapter.info.typeName,
             gpuContext.adapter.info.backendName);
        return gpuContext.adapter.handle.Get();
      }
    }
  }

  return nullptr;
}

static void LogAvailableAdapters()
{
  Initialize();

  fprintf(stderr, "Available adapters:\n");
  for (auto&& a : gpuContext.dawn_native.instance->EnumerateAdapters()) {
    wgpu::AdapterProperties p;
    a.GetProperties(&p);
    fprintf(
      stderr,
      "  %s (%s)\n"
      "    deviceID=%u, vendorID=0x%x, BackendType::%s, AdapterType::%s\n",
      p.name, p.driverDescription, p.deviceID, p.vendorID,
      BackendTypeName(p.backendType), AdapterTypeName(p.adapterType));
  }
}

static void GetAdapterInfo(char (*adapter_info)[256])
{
  strncpy(adapter_info[0], gpuContext.adapter.info.name, 256);
  strncpy(adapter_info[1], gpuContext.adapter.info.typeName, 256);
  strncpy(adapter_info[2], gpuContext.adapter.info.backendName, 256);
}

static std::unique_ptr<wgpu::ChainedStruct> SurfaceDescriptor(void* display,
                                                              void* window)
{
#if defined(WIN32)
  std::unique_ptr<wgpu::SurfaceDescriptorFromWindowsHWND> desc
    = std::make_unique<wgpu::SurfaceDescriptorFromWindowsHWND>();
  desc->hwnd      = window;
  desc->hinstance = GetModuleHandle(nullptr);
  return std::move(desc);
#elif defined(__linux__) // X11
  std::unique_ptr<wgpu::SurfaceDescriptorFromXlibWindow> desc
    = std::make_unique<wgpu::SurfaceDescriptorFromXlibWindow>();
  desc->display = display;
  desc->window  = *((uint32_t*)window);
  return std::move(desc);
#endif

  return nullptr;
}

static WGPUSurface CreateSurface(void* display, void* window)
{
  std::unique_ptr<wgpu::ChainedStruct> sd = SurfaceDescriptor(display, window);
  wgpu::SurfaceDescriptor descriptor;
  descriptor.nextInChain = sd.get();
  wgpu::Surface surface = wgpu::Instance(gpuContext.dawn_native.instance->Get())
                            .CreateSurface(&descriptor);
  if (!surface) {
    return nullptr;
  }
  WGPUSurface surf = surface.Get();
  wgpuSurfaceReference(surf);
  return surf;
}

static const char* BackendTypeName(wgpu::BackendType t)
{
  switch (t) {
    case wgpu::BackendType::Null:
      return "Null";
    case wgpu::BackendType::WebGPU:
      return "WebGPU";
    case wgpu::BackendType::D3D11:
      return "D3D11";
    case wgpu::BackendType::D3D12:
      return "D3D12";
    case wgpu::BackendType::Metal:
      return "Metal";
    case wgpu::BackendType::Vulkan:
      return "Vulkan";
    case wgpu::BackendType::OpenGL:
      return "OpenGL";
    case wgpu::BackendType::OpenGLES:
      return "OpenGL ES";
    default:
      break;
  }
  return "?";
}
static const char* AdapterTypeName(wgpu::AdapterType t)
{
  switch (t) {
    case wgpu::AdapterType::DiscreteGPU:
      return "Discrete GPU";
    case wgpu::AdapterType::IntegratedGPU:
      return "Integrated GPU";
    case wgpu::AdapterType::CPU:
      return "CPU";
    case wgpu::AdapterType::Unknown:
      return "Unknown";
  }
  return "?";
}

} // namespace WGPUImpl

//******************************** Public API *********************************/

void wgpu_log_available_adapters()
{
  WGPUImpl::LogAvailableAdapters();
}

void wgpu_get_adapter_info(char (*adapter_info)[256])
{
  WGPUImpl::GetAdapterInfo(adapter_info);
}

WGPUAdapter wgpu_request_adapter(WGPURequestAdapterOptions* options)
{
  return WGPUImpl::RequestAdapter(options);
}

WGPUSurface wgpu_create_surface(void* display, void* window_handle)
{
  return WGPUImpl::CreateSurface(display, window_handle);
}
