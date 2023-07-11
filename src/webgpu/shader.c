#include "shader.h"

#include <stdlib.h>
#include <string.h>

#include "../core/file.h"
#include "../core/log.h"
#include "../core/macro.h"

static void
wgpu_compilation_info_callback(WGPUCompilationInfoRequestStatus status,
                               WGPUCompilationInfo const* compilationInfo,
                               void* userdata)
{
  UNUSED_VAR(userdata);
  if (status == WGPUCompilationInfoRequestStatus_Error) {
    for (uint32_t m = 0; m < compilationInfo->messageCount; ++m) {
      WGPUCompilationMessage message = compilationInfo->messages[m];
      log_error("lineNum: %u, linePos: %u, Error: %s", message.lineNum,
                message.linePos, message.message);
    }
  }
}

WGPUShaderModule wgpu_create_shader_module_from_spirv_file(WGPUDevice device,
                                                           const char* filename)
{
  file_read_result_t result;
  read_file(filename, &result, 0);
  log_debug("Read file: %s, size: %d bytes\n", filename, result.size);
  WGPUShaderModule shader_module
    = wgpu_create_shader_module_from_spirv_bytecode(device, result.data,
                                                    result.size);
  free(result.data);
  return shader_module;
}

WGPUShaderModule wgpu_create_shader_module_from_wgsl_file(WGPUDevice device,
                                                          const char* filename)
{
  file_read_result_t result;
  read_file(filename, &result, 1);
  log_debug("Read file: %s, size: %d bytes\n", filename, result.size);
  WGPUShaderModule shader_module
    = wgpu_create_shader_module_from_wgsl(device, (char*)result.data);
  free(result.data);
  return shader_module;
}

WGPUShaderModule
wgpu_create_shader_module(wgpu_context_t* wgpu_context,
                          const wgpu_shader_desc_t* shader_desc)
{
  WGPUShaderModule shader_module = NULL;

  if (shader_desc->file != NULL) {
    /* WebGPU Shader from file */
    if (filename_has_extension(shader_desc->file, "spv")) {
      shader_module = wgpu_create_shader_module_from_spirv_file(
        wgpu_context->device, shader_desc->file);
    }
    else if (filename_has_extension(shader_desc->file, "wgsl")) {
      shader_module = wgpu_create_shader_module_from_wgsl_file(
        wgpu_context->device, shader_desc->file);
    }
  }
  else if ((shader_desc->byte_code.data != NULL)
           && (shader_desc->byte_code.size != 0)) {
    /* WebGPU Shader from SPIR-V bytecode */
    shader_module = wgpu_create_shader_module_from_spirv_bytecode(
      wgpu_context->device, shader_desc->byte_code.data,
      shader_desc->byte_code.size);
  }
  else if (shader_desc->wgsl_code.source != NULL) {
    /* WebGPU Shader from WGSL code */
    shader_module = wgpu_create_shader_module_from_wgsl(
      wgpu_context->device, shader_desc->wgsl_code.source);
  }

  return shader_module;
}

WGPUShaderModule wgpu_create_shader_module_from_spirv_bytecode(
  WGPUDevice device, const uint8_t* data, const uint32_t size)
{
  ASSERT(data && size > 0 && size % 4 == 0);

  WGPUShaderModuleSPIRVDescriptor shader_module_spirv_desc = {
    .chain   = {
      .sType = WGPUSType_ShaderModuleSPIRVDescriptor,
    },
    .codeSize = size / sizeof(uint32_t),
    .code     = (const uint32_t*)data,
  };

  WGPUShaderModuleDescriptor shader_module_desc = {
    .nextInChain = (WGPUChainedStruct const*)&shader_module_spirv_desc,
  };

  WGPUShaderModule shader_module
    = wgpuDeviceCreateShaderModule(device, &shader_module_desc);
  wgpuShaderModuleGetCompilationInfo(shader_module,
                                     wgpu_compilation_info_callback, NULL);

  return shader_module;
}

WGPUShaderModule wgpu_create_shader_module_from_wgsl(WGPUDevice device,
                                                     const char* source)
{
  WGPUShaderModuleWGSLDescriptor shader_module_wgsl_desc = {
    .chain  = {
      .sType = WGPUSType_ShaderModuleWGSLDescriptor,
    },
    .code = source,
  };

  WGPUShaderModuleDescriptor shader_module_desc = {
    .nextInChain = &shader_module_wgsl_desc.chain,
  };

  WGPUShaderModule shader_module
    = wgpuDeviceCreateShaderModule(device, &shader_module_desc);
  wgpuShaderModuleGetCompilationInfo(shader_module,
                                     wgpu_compilation_info_callback, NULL);

  return shader_module;
}

wgpu_shader_t wgpu_shader_create(wgpu_context_t* wgpu_context,
                                 const wgpu_shader_desc_t* desc)
{
  ASSERT(desc->file || (desc->byte_code.data && desc->byte_code.size > 0)
         || desc->wgsl_code.source);
  ASSERT(wgpu_context && wgpu_context->device);

  wgpu_shader_t shader = {0};
  shader.module        = wgpu_create_shader_module(wgpu_context, desc);
  ASSERT(shader.module);

  shader.programmable_stage_descriptor = (WGPUProgrammableStageDescriptor){
    .module        = shader.module,
    .entryPoint    = desc->entry ? desc->entry : "main",
    .constantCount = desc->constants.count,
    .constants     = desc->constants.entries,
  };

  return shader;
}

void wgpu_shader_release(wgpu_shader_t* shader)
{
  ASSERT(shader->module);
  WGPU_RELEASE_RESOURCE(ShaderModule, shader->module);
}

WGPUVertexState wgpu_create_vertex_state(wgpu_context_t* wgpu_context,
                                         const wgpu_vertex_state_t* desc)
{
  ASSERT(desc);
  wgpu_shader_desc_t const* shader_desc = &desc->shader_desc;

  ASSERT(shader_desc
         && (shader_desc->file
             || (shader_desc->byte_code.data && shader_desc->byte_code.size > 0)
             || shader_desc->wgsl_code.source));
  ASSERT(wgpu_context && wgpu_context->device);

  WGPUVertexState vertex_state = {0};
  vertex_state.module = wgpu_create_shader_module(wgpu_context, shader_desc);
  ASSERT(vertex_state.module);

  vertex_state.entryPoint    = shader_desc->entry ? shader_desc->entry : "main",
  vertex_state.constantCount = desc->constant_count,
  vertex_state.constants     = desc->constants;
  vertex_state.bufferCount   = desc->buffer_count,
  vertex_state.buffers       = desc->buffers;

  return vertex_state;
}

WGPUFragmentState wgpu_create_fragment_state(wgpu_context_t* wgpu_context,
                                             const wgpu_fragment_state_t* desc)
{
  ASSERT(desc);
  wgpu_shader_desc_t const* shader_desc = &desc->shader_desc;

  ASSERT(shader_desc
         && (shader_desc->file
             || (shader_desc->byte_code.data && shader_desc->byte_code.size > 0)
             || shader_desc->wgsl_code.source));
  ASSERT(wgpu_context && wgpu_context->device);

  WGPUFragmentState fragment_state = {0};
  fragment_state.module = wgpu_create_shader_module(wgpu_context, shader_desc);
  ASSERT(fragment_state.module);

  fragment_state.entryPoint = shader_desc->entry ? shader_desc->entry : "main",
  fragment_state.constantCount = desc->constant_count,
  fragment_state.constants     = desc->constants;
  fragment_state.targetCount   = desc->target_count,
  fragment_state.targets       = desc->targets;

  return fragment_state;
}
