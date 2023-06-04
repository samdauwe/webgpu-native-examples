#ifndef SHADER_H
#define SHADER_H

#include "context.h"

/* WebGPU shader */
typedef struct wgpu_shader_desc_t {
  const char* label;
  const char* file; /* file has priority over byte code & WGSL code */
  struct {
    const uint8_t* data;
    const uint32_t size;
  } byte_code; /* SPIR-V bytecode */
  struct {
    const char* source;
  } wgsl_code; /* WGSL source code ( ref: https://www.w3.org/TR/WGSL ) */
  const char* entry;
  struct {
    uint32_t count;
    WGPUConstantEntry const* entries;
  } constants; /* pipeline shader constants*/
} wgpu_shader_desc_t;

typedef struct wgpu_shader_t {
  WGPUProgrammableStageDescriptor programmable_stage_descriptor;
  WGPUShaderModule module;
} wgpu_shader_t;

/* Helper functions */
WGPUShaderModule
wgpu_create_shader_module_from_spirv_file(WGPUDevice device,
                                          const char* filename);
WGPUShaderModule wgpu_create_shader_module_from_wgsl_file(WGPUDevice device,
                                                          const char* filename);
WGPUShaderModule wgpu_create_shader_module_from_spirv_bytecode(
  WGPUDevice device, const uint8_t* data, const uint32_t size);
WGPUShaderModule wgpu_create_shader_module_from_wgsl(WGPUDevice device,
                                                     const char* source);
WGPUShaderModule
wgpu_create_shader_module(wgpu_context_t* wgpu_context,
                          const wgpu_shader_desc_t* shader_desc);

/* Shader creating/releasing */
wgpu_shader_t wgpu_shader_create(wgpu_context_t* wgpu_context,
                                 const wgpu_shader_desc_t* desc);
void wgpu_shader_release(wgpu_shader_t* shader);

typedef struct wgpu_vertex_state_t {
  wgpu_shader_desc_t shader_desc;
  uint32_t constant_count;
  WGPUConstantEntry const* constants;
  uint32_t buffer_count;
  WGPUVertexBufferLayout const* buffers;
} wgpu_vertex_state_t;
WGPUVertexState wgpu_create_vertex_state(wgpu_context_t* wgpu_context,
                                         const wgpu_vertex_state_t* desc);

typedef struct wgpu_fragment_state_t {
  wgpu_shader_desc_t shader_desc;
  uint32_t constant_count;
  WGPUConstantEntry const* constants;
  uint32_t target_count;
  WGPUColorTargetState const* targets;
} wgpu_fragment_state_t;
WGPUFragmentState wgpu_create_fragment_state(wgpu_context_t* wgpu_context,
                                             const wgpu_fragment_state_t* desc);

#endif
