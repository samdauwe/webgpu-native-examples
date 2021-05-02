#ifndef CONTEXT_H
#define CONTEXT_H

#include <dawn/webgpu.h>

#define WGPU_RELEASE_RESOURCE(Type, Name)                                      \
  if (Name) {                                                                  \
    wgpu##Type##Release(Name);                                                 \
    Name = NULL;                                                               \
  }

#define WGPU_VERTATTR_DESC(l, f, o)                                            \
  (WGPUVertexAttribute)                                                        \
  {                                                                            \
    .shaderLocation = l, .format = f, .offset = o,                             \
  }

#define WGPU_VERTBUFFERLAYOUT_DESC(s, a)                                       \
  {                                                                            \
    .arrayStride    = s,                                                       \
    .stepMode       = WGPUInputStepMode_Vertex,                                \
    .attributeCount = sizeof(a) / sizeof(a[0]),                                \
    .attributes     = a,                                                       \
  };

#define WPU_VERTEXSTATE_DESC(b)                                                \
  {                                                                            \
    .vertexBufferCount = 1, .vertexBuffers = &b,                               \
  }

#define WGPU_VERTSTATE(name, bindSize, ...)                                    \
  WGPUVertexAttribute vertAttrDesc##name[] = {__VA_ARGS__};                    \
  WGPUVertexBufferLayout name##VertBuffLayoutDesc                              \
    = WGPU_VERTBUFFERLAYOUT_DESC(bindSize, vertAttrDesc##name);                \
  WGPUVertexStateDescriptor vert_state_##name                                  \
    = WPU_VERTEXSTATE_DESC(name##VertBuffLayoutDesc);

#define MAX_COMMAND_BUFFER_COUNT 256

/* Initializers */

/* Forward declarations */
struct wgpu_buffer_t;

/* WebGPU context */
typedef struct wgpu_context_t {
  void* context;
  WGPUDevice device;
  WGPUQueue queue;
  WGPUInstance instance;
  struct {
    void* instance;
    uint32_t width;
    uint32_t height;
  } surface;
  struct {
    WGPUSwapChain instance;
    WGPUTextureFormat format;
    WGPUTextureView frame_buffer;
  } swap_chain;
  WGPUCommandEncoder cmd_enc;       // Command encoder
  WGPURenderPassEncoder rpass_enc;  // Render pass encoder
  WGPUComputePassEncoder cpass_enc; // Compute pass encoder
  struct {
    WGPUTexture texture;
    WGPUTextureView texture_view;
    WGPURenderPassDepthStencilAttachmentDescriptor att_desc;
  } depth_stencil;
  struct {
    uint32_t command_buffer_count;
    WGPUCommandBuffer command_buffers[MAX_COMMAND_BUFFER_COUNT];
  } submit_info;
} wgpu_context_t;

/* WebGPU context creating/releasing */
wgpu_context_t* wgpu_context_create();
void wgpu_context_release(wgpu_context_t* wgpu_context);

/* WebGPU info functions */
void wgpu_get_context_info(char (*adapter_info)[256]);

/* WebGPU context helper functions */
WGPUBuffer wgpu_create_buffer_from_data(wgpu_context_t* wgpu_context,
                                        const void* data, size_t size,
                                        WGPUBufferUsage usage);
void wgpu_create_device_and_queue(wgpu_context_t* wgpu_context);
void wgpu_create_surface(wgpu_context_t* wgpu_context, void* window);
void wgpu_setup_deph_stencil(wgpu_context_t* wgpu_context);
void wgpu_setup_swap_chain(wgpu_context_t* wgpu_context);
void wgpu_error_callback(WGPUErrorType type, char const* message,
                         void* userdata);

/* Methods of Queue */
void wgpu_queue_write_buffer(wgpu_context_t* wgpu_context, WGPUBuffer buffer,
                             uint64_t buffer_offset, void const* data,
                             size_t size);

/* Render helper functions */
WGPUCommandBuffer wgpu_get_command_buffer(WGPUCommandEncoder cmd_encoder);
WGPUTextureView wgpu_swap_chain_get_current_image(wgpu_context_t* wgpu_context);
void wgpu_flush_command_buffers(wgpu_context_t* wgpu_context,
                                WGPUCommandBuffer* command_buffers,
                                uint32_t command_buffer_count);
void wgpu_swap_chain_present(wgpu_context_t* wgpu_context);

/* Pipeline state factories */
typedef struct create_color_state_desc_t {
  WGPUTextureFormat format;
  bool enable_blend;
} create_color_state_desc_t;
WGPUColorStateDescriptor
wgpu_create_color_state_descriptor(create_color_state_desc_t* desc);

typedef struct create_depth_stencil_state_desc_t {
  WGPUTextureFormat format;
  bool depth_write_enabled;
} create_depth_stencil_state_desc_t;
WGPUDepthStencilStateDescriptor wgpu_create_depth_stencil_state_descriptor(
  create_depth_stencil_state_desc_t* desc);

typedef struct create_rasterization_state_desc_t {
  WGPUFrontFace front_face;
  WGPUCullMode cull_mode;
} create_rasterization_state_desc_t;
WGPURasterizationStateDescriptor wgpu_create_rasterization_state_descriptor(
  create_rasterization_state_desc_t* desc);

#endif
