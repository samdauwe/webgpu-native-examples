#ifndef CONTEXT_H
#define CONTEXT_H

#include <stdbool.h>
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
    .stepMode       = WGPUVertexStepMode_Vertex,                               \
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

#define WGPU_VERTEX_BUFFER_LAYOUT(name, bind_size, ...)                        \
  WGPUVertexAttribute vert_attr_desc_##name[] = {__VA_ARGS__};                 \
  WGPUVertexBufferLayout name##_vertex_buffer_layout                           \
    = WGPU_VERTBUFFERLAYOUT_DESC(bind_size, vert_attr_desc_##name);

#define MAX_COMMAND_BUFFER_COUNT 256
#define WGPU_FEATURE_COUNT 12u

/* Initializers */

/* Forward declarations */
struct wgpu_buffer_t;
struct wgpu_texture_client_t;

/* WebGPU context create options */
typedef struct wgpu_context_create_options_t {
  bool vsync;
} wgpu_context_create_options_t;

/* WebGPU context */
typedef struct wgpu_context_t {
  void* context;
  WGPUAdapter adapter;
  WGPUDevice device;
  WGPUQueue queue;
  WGPUInstance instance;
  struct {
    WGPUFeatureName feature_name;
    bool is_supported;
  } features[WGPU_FEATURE_COUNT];
  struct {
    void* instance;
    uint32_t width;
    uint32_t height;
  } surface;
  struct {
    WGPUSwapChain instance;
    WGPUTextureFormat format;
    WGPUTextureView frame_buffer;
    WGPUPresentMode present_mode;
  } swap_chain;
  WGPUCommandEncoder cmd_enc;       /* Command encoder */
  WGPURenderPassEncoder rpass_enc;  /* Render pass encoder */
  WGPUComputePassEncoder cpass_enc; /* Compute pass encoder */
  struct {
    WGPUTexture texture;
    WGPUTextureView texture_view;
    WGPURenderPassDepthStencilAttachment att_desc;
  } depth_stencil;
  struct {
    uint32_t command_buffer_count;
    WGPUCommandBuffer command_buffers[MAX_COMMAND_BUFFER_COUNT];
  } submit_info;
  struct wgpu_texture_client_t* texture_client;
} wgpu_context_t;

/* WebGPU context creating/releasing */
wgpu_context_t* wgpu_context_create(wgpu_context_create_options_t* options);
void wgpu_context_release(wgpu_context_t* wgpu_context);

/* WebGPU info functions */
void wgpu_get_context_info(char (*adapter_info)[256]);
bool wgpu_has_feature(wgpu_context_t* wgpu_context,
                      WGPUFeatureName feature_name);

/* WebGPU context helper functions */
typedef struct deph_stencil_texture_creation_options_t {
  WGPUTextureFormat format;
  uint32_t sample_count;
} deph_stencil_texture_creation_options;

WGPUBuffer wgpu_create_buffer_from_data(wgpu_context_t* wgpu_context,
                                        const void* data, size_t size,
                                        WGPUBufferUsage usage);
void wgpu_create_device_and_queue(wgpu_context_t* wgpu_context);
void wgpu_setup_window_surface(wgpu_context_t* wgpu_context, void* window);
void wgpu_setup_deph_stencil(
  wgpu_context_t* wgpu_context,
  struct deph_stencil_texture_creation_options_t* options);
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

/* Texture client creation */
void wgpu_create_texture_client(wgpu_context_t* wgpu_context);

/* Pipeline state factories */
WGPUBlendState wgpu_create_blend_state(bool enable_blend);

typedef struct create_depth_stencil_state_desc_t {
  WGPUTextureFormat format;
  bool depth_write_enabled;
} create_depth_stencil_state_desc_t;
WGPUDepthStencilState
wgpu_create_depth_stencil_state(create_depth_stencil_state_desc_t* desc);

typedef struct create_multisample_state_desc_t {
  uint32_t sample_count;
} create_multisample_state_desc_t;
WGPUMultisampleState
wgpu_create_multisample_state_descriptor(create_multisample_state_desc_t* desc);

#endif
