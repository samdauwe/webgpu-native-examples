#include "shader.h"

#include <stdlib.h>

#include "../core/file.h"
#include "../core/log.h"
#include "../core/macro.h"

WGPUShaderModule wgpu_create_shader_module_from_spirv_file(WGPUDevice device,
                                                           const char* filename)
{
  file_read_result_t result;
  read_file(filename, &result);
  log_debug("Read file: %s, size: %d bytes\n", filename, result.size);
  WGPUShaderModule shader_module
    = wgpu_create_shader_module_from_spirv_bytecode(device, result.data,
                                                    result.size);
  free(result.data);
  return shader_module;
}

WGPUShaderModule wgpu_create_shader_module_from_spirv_bytecode(
  WGPUDevice device, const uint8_t* data, const uint32_t size)
{
  ASSERT(data && size > 0 && size % 4 == 0);

  WGPUShaderModuleSPIRVDescriptor shader_module_spirv_desc = {
    .chain.sType = WGPUSType_ShaderModuleSPIRVDescriptor,
    .codeSize    = size / sizeof(uint32_t),
    .code        = (const uint32_t*)data,
  };

  WGPUShaderModuleDescriptor shader_module_desc = {
    .nextInChain = (WGPUChainedStruct const*)&shader_module_spirv_desc,
  };

  return wgpuDeviceCreateShaderModule(device, &shader_module_desc);
}

wgpu_shader_t wgpu_shader_create(wgpu_context_t* wgpu_context,
                                 const wgpu_shader_desc_t* desc)
{
  ASSERT(desc->file || (desc->byte_code.data && desc->byte_code.size > 0));
  ASSERT(wgpu_context && wgpu_context->device);

  wgpu_shader_t shader;
  if (desc->file) {
    shader.module = wgpu_create_shader_module_from_spirv_file(
      wgpu_context->device, desc->file);
  }
  else {
    shader.module = wgpu_create_shader_module_from_spirv_bytecode(
      wgpu_context->device, desc->byte_code.data, desc->byte_code.size);
  }
  ASSERT(shader.module);

  shader.programmable_stage_descriptor = (WGPUProgrammableStageDescriptor){
    .module     = shader.module,
    .entryPoint = desc->entry ? desc->entry : "main",
  };

  return shader;
}

void wgpu_shader_release(wgpu_shader_t* shader)
{
  ASSERT(shader->module);
  WGPU_RELEASE_RESOURCE(ShaderModule, shader->module);
}
