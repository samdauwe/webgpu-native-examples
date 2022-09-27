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
} settings = {
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

static struct {
  vec2 current;
  vec2 last;
  vec2 velocity;
} mouse_infos = {
  .current  = GLM_VEC2_ZERO_INIT,
  .last     = GLM_VEC2_ZERO_INIT,
  .velocity = GLM_VEC2_ZERO_INIT,
};

/**
 * Creates and manage multi-dimensional buffers by creating a buffer for each
 * dimension
 */
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

static void dynamic_buffer_destroy(dynamic_buffer_t* this)
{
  for (uint32_t i = 0; i < this->dims; ++i) {
    wgpu_destroy_buffer(&this->buffers[i]);
  }
}

/**
 * Copy each buffer to another DynamicBuffer's buffers.
 * If the dimensions don't match, the last non-empty dimension will be copied
 * instead
 */
static void dynamic_buffer_copy_to(dynamic_buffer_t* this,
                                   dynamic_buffer_t* buffer,
                                   WGPUCommandEncoder command_encoder)
{
  for (uint32_t i = 0; i < MAX(this->dims, buffer->dims); ++i) {
    wgpuCommandEncoderCopyBufferToBuffer(
      command_encoder, this->buffers[MIN(i, this->dims - 1)].buffer, 0,
      buffer->buffers[MIN(i, buffer->dims - 1)].buffer, 0, this->buffer_size);
  }
}

/* Reset all the buffers */
static void dynamic_buffer_clear(dynamic_buffer_t* this)
{
  float* empty_buffer = (float*)malloc(this->buffer_size);
  memset(empty_buffer, 0, this->buffer_size);

  for (uint32_t i = 0; i < this->dims; ++i) {
    wgpu_queue_write_buffer(this->wgpu_context, this->buffers[i].buffer, 0,
                            empty_buffer, this->buffer_size);
  }

  free(empty_buffer);
}

static struct {
  dynamic_buffer_t velocity;
  dynamic_buffer_t velocity0;

  dynamic_buffer_t dye;
  dynamic_buffer_t dye0;

  dynamic_buffer_t divergence;
  dynamic_buffer_t divergence0;

  dynamic_buffer_t pressure;
  dynamic_buffer_t pressure0;

  dynamic_buffer_t vorticity;
} dynamic_buffers;

typedef enum {
  UNIFORM_TIME,
  UNIFORM_DT,
  UNIFORM_MOUSE_INFOS,
  UNIFORM_GRID_SIZE,
  UNIFORM_SIM_SPEED,
  UNIFORM_VELOCITY_ADD_INTENSITY,
  UNIFORM_VELOCITY_ADD_RADIUS,
  UNIFORM_VELOCITY_DIFFUSION,
  UNIFORM_DYE_ADD_INTENSITY,
  UNIFORM_DYE_ADD_RADIUS,
  UNIFORM_DYE_ADD_DIFFUSION,
  UNIFORM_VISCOSITY,
  UNIFORM_VORTICITY,
  UNIFORM_CONTAIN_FLUID,
  UNIFORM_MOUSE_TYPE,
  UNIFORM_COUNT,
} uniform_type_t;

/* Manage uniform buffers relative to the compute shaders & the gui */
typedef struct {
  size_t size;
  bool needs_update;
} uniform_t;

static struct {
  uniform_t time;
  uniform_t dt;
  uniform_t mouse;
  uniform_t grid;
  uniform_t simSpeed;
  uniform_t vel_force;
  uniform_t vel_radius;
  uniform_t vel_diff;
  uniform_t dye_force;
  uniform_t dye_radius;
  uniform_t dye_diff;
  uniform_t viscosity;
  uniform_t u_vorticity;
  uniform_t contain_fluid;
  uniform_t u_symmetry;
} uniforms;
