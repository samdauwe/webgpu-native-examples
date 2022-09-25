#include "example_base.h"
#include "examples.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Fluid Simulation
 *
 * WebGPU demo featuring an implementation of Jos Stam's "Real-Time Fluid
 * Dynamics for Games" paper.
 *
 * Ref:
 * JavaScript version: https://github.com/indiana-dev/WebGPU-Fluid-Simulation
 * Jos Stam Paper :
 * https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
 * Nvidia GPUGem's Chapter 38 :
 * https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu
 * Jamie Wong's Fluid simulation :
 * https://jamie-wong.com/2016/08/05/webgl-fluid-simulation/ PavelDoGreat's
 * Fluid simulation : https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
 * -------------------------------------------------------------------------- */

#define MAX_DIMENSIONS 3

static struct {
  uint32_t grid_size;
  uint32_t dye_size;
  uint32_t dye_w;
  uint32_t dye_h;
  uint32_t sim_speed;
  bool contain_fluid;
  float velocity_add_intensity;
  float velocity_add_radius;
  float velocity_diffusion;
  float dye_add_intensity;
  float dye_add_radius;
  float dye_diffusion;
  float viscosity;
  uint32_t vorticity;
  uint32_t pressure_iterations;
} settings_t = {
  .grid_size              = 512,
  .dye_size               = 2048,
  .sim_speed              = 5,
  .contain_fluid          = true,
  .velocity_add_intensity = 0.1f,
  .velocity_add_radius    = 0.0001f,
  .velocity_diffusion     = 0.9999f,
  .dye_add_intensity      = 4.0f,
  .dye_add_radius         = 0.001f,
  .dye_diffusion          = 0.994f,
  .viscosity              = 0.8f,
  .vorticity              = 2,
  .pressure_iterations    = 100,
};

/* Creates and manage multi-dimensional buffers by creating a buffer for each
 * dimension */
typedef struct {
  wgpu_context_t* wgpu_context;          /* The WebGPU context*/
  uint32_t dims;                         /* Number of dimensions */
  uint32_t buffer_size;                  /* Size of the buffer in bytes */
  uint32_t w;                            /* Buffer width */
  uint32_t h;                            /* Buffer height */
  wgpu_buffer_t buffers[MAX_DIMENSIONS]; /* Multi-dimensional buffers */
} dynamic_buffer_t;

static void dynamic_buffer_init_defaults(dynamic_buffer_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void dynamic_buffer_init(dynamic_buffer_t* this,
                                wgpu_context_t* wgpu_context, uint32_t dims,
                                uint32_t w, uint32_t h)
{
  dynamic_buffer_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->dims         = dims;
  this->buffer_size  = w * h * 4;
  this->w            = w;
  this->h            = h;

  assert(dims <= MAX_DIMENSIONS);
  for (uint32_t dim = 0; dim < dims; ++dim) {
    WGPUBufferDescriptor buffer_desc = {
      .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc
               | WGPUBufferUsage_CopyDst,
      .size             = this->buffer_size,
      .mappedAtCreation = false,
    };
    this->buffers[dim] = (wgpu_buffer_t){
      .buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc),
      .usage  = buffer_desc.usage,
      .size   = buffer_desc.size,
    };
  }
}
