#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

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

typedef enum {
  DYNAMIC_BUFFER_VELOCITY,
  DYNAMIC_BUFFER_DYE,
  DYNAMIC_BUFFER_DIVERGENCE,
  DYNAMIC_BUFFER_PRESSURE,
  DYNAMIC_BUFFER_VORTICITY,
  DYNAMIC_BUFFER_RGB,
} dynamic_buffer_type_t;

static struct {
  uint32_t grid_size;
  uint32_t grid_w;
  uint32_t grid_h;
  uint32_t dye_size;
  uint32_t dye_w;
  uint32_t dye_h;
  uint32_t rdx;
  uint32_t dye_rdx;
  float dx;
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
  dynamic_buffer_type_t buffer_view;
  float dt;
  float time;
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
  .buffer_view            = DYNAMIC_BUFFER_DYE,
  .dt                     = 0.0f,
  .time                   = 0.0f,
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

/* -------------------------------------------------------------------------- *
 * Dynamic buffer
 * -------------------------------------------------------------------------- */

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

  /* The r,g,b buffer containing the data to render */
  dynamic_buffer_t rgb_buffer;
} dynamic_buffers;

/* Initialize dynamic buffers */
static void dynamic_buffers_init(wgpu_context_t* wgpu_context)
{
  dynamic_buffer_init(&dynamic_buffers.velocity, wgpu_context, 2,
                      settings.grid_w, settings.grid_h);
  dynamic_buffer_init(&dynamic_buffers.velocity0, wgpu_context, 2,
                      settings.grid_w, settings.grid_h);

  dynamic_buffer_init(&dynamic_buffers.dye, wgpu_context, 3, settings.dye_w,
                      settings.dye_h);
  dynamic_buffer_init(&dynamic_buffers.dye0, wgpu_context, 3, settings.dye_w,
                      settings.dye_h);

  dynamic_buffer_init(&dynamic_buffers.divergence, wgpu_context, 1,
                      settings.grid_w, settings.grid_h);
  dynamic_buffer_init(&dynamic_buffers.divergence0, wgpu_context, 1,
                      settings.grid_w, settings.grid_h);

  dynamic_buffer_init(&dynamic_buffers.pressure, wgpu_context, 1,
                      settings.grid_w, settings.grid_h);
  dynamic_buffer_init(&dynamic_buffers.pressure0, wgpu_context, 1,
                      settings.grid_w, settings.grid_h);

  dynamic_buffer_init(&dynamic_buffers.vorticity, wgpu_context, 1,
                      settings.grid_w, settings.grid_h);
}

static void dynamic_buffers_destroy()
{
  dynamic_buffer_destroy(&dynamic_buffers.velocity);
  dynamic_buffer_destroy(&dynamic_buffers.velocity0);

  dynamic_buffer_destroy(&dynamic_buffers.dye);
  dynamic_buffer_destroy(&dynamic_buffers.dye0);

  dynamic_buffer_destroy(&dynamic_buffers.divergence);
  dynamic_buffer_destroy(&dynamic_buffers.divergence0);

  dynamic_buffer_destroy(&dynamic_buffers.pressure);
  dynamic_buffer_destroy(&dynamic_buffers.pressure0);

  dynamic_buffer_destroy(&dynamic_buffers.vorticity);

  dynamic_buffer_destroy(&dynamic_buffers.rgb_buffer);
}

/* -------------------------------------------------------------------------- *
 * Uniforms
 * -------------------------------------------------------------------------- */

typedef enum {
  UNIFORM_TIME,                   /* time */
  UNIFORM_DT,                     /* dt */
  UNIFORM_MOUSE_INFOS,            /* mouseInfos */
  UNIFORM_GRID_SIZE,              /* gridSize */
  UNIFORM_SIM_SPEED,              /* sim_speed */
  UNIFORM_VELOCITY_ADD_INTENSITY, /* velocity_add_intensity */
  UNIFORM_VELOCITY_ADD_RADIUS,    /* velocity_add_radius */
  UNIFORM_VELOCITY_DIFFUSION,     /* velocity_diffusion */
  UNIFORM_DYE_ADD_INTENSITY,      /* dye_add_intensity */
  UNIFORM_DYE_ADD_RADIUS,         /* dye_add_radius */
  UNIFORM_DYE_ADD_DIFFUSION,      /* dye_diffusion */
  UNIFORM_VISCOSITY,              /* viscosity */
  UNIFORM_VORTICITY,              /* vorticity */
  UNIFORM_CONTAIN_FLUID,          /* contain_fluid */
  UNIFORM_MOUSE_TYPE,             /* mouse_type */
  UNIFORM_RENDER_INTENSITY,       /* render_intensity_multiplier */
  UNIFORM_RENDER_DYE,             /* render_dye_buffer */
  UNIFORM_COUNT,
} uniform_type_t;

#define MAX_UNIFORM_VALUE_COUNT 7

/* Manage uniform buffers relative to the compute shaders & the gui */
typedef struct {
  uniform_type_t type;
  float values[MAX_UNIFORM_VALUE_COUNT];
  size_t size;
  bool always_update;
  bool needs_update;
  wgpu_buffer_t buffer;
} uniform_t;

static struct {
  uniform_t time;
  uniform_t dt;
  uniform_t mouse;
  uniform_t grid;
  uniform_t sim_speed;
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
  uniform_t u_render_intensity;
  uniform_t u_render_dye;
} uniforms;

static uniform_t* global_uniforms[UNIFORM_COUNT] = {0};

static void uniform_init_defaults(uniform_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void uniform_init(uniform_t* this, wgpu_context_t* wgpu_context,
                         uniform_type_t type, uint32_t size, float const* value)
{
  uniform_init_defaults(this);

  this->type         = type;
  this->size         = size;
  this->needs_update = false;

  this->always_update = (size == 1);

  if (this->size == 1 || value != NULL) {
    this->buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .usage = WGPUBufferUsage_Uniform
                                                    | WGPUBufferUsage_CopyDst,
                                           .size = this->size * sizeof(float),
                                           .initial.data = value ? value : 0,
                                         });
    if (value) {
      memcpy(this->values, value, size);
    }
  }
  else {
    this->buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .usage = WGPUBufferUsage_Uniform
                                                    | WGPUBufferUsage_CopyDst,
                                           .size = this->size * sizeof(float),
                                         });
  }

  global_uniforms[type] = this;
}

static void uniform_destroy(uniform_t* this)
{
  wgpu_destroy_buffer(&this->buffer);
}

/* Update the GPU buffer if the value has changed */
static void uniform_update(uniform_t* this, wgpu_context_t* wgpu_context,
                           float* value, uint32_t value_count)
{
  if (this->needs_update || this->always_update || value != NULL) {
    wgpu_queue_write_buffer(
      wgpu_context, this->buffer.buffer, 0, value ? value : this->values,
      (value_count ? MIN(this->size, value_count) : this->size)
        * sizeof(float));
    this->needs_update = false;
  }
}

/* Initialize uniforms */
static void uniforms_buffers_init(wgpu_context_t* wgpu_context)
{
  uniform_init(&uniforms.time, wgpu_context, UNIFORM_TIME, 1, NULL);
  uniform_init(&uniforms.dt, wgpu_context, UNIFORM_DT, 1, NULL);
  uniform_init(&uniforms.mouse, wgpu_context, UNIFORM_MOUSE_INFOS, 4, NULL);
  const float values[7] = {
    settings.grid_w, settings.grid_h, settings.dye_w,   settings.dye_h,
    settings.dx,     settings.rdx,    settings.dye_rdx,
  };
  uniform_init(&uniforms.grid, wgpu_context, UNIFORM_GRID_SIZE, 7, values);
  uniform_init(&uniforms.sim_speed, wgpu_context, UNIFORM_SIM_SPEED, 1, NULL);
  uniform_init(&uniforms.vel_force, wgpu_context,
               UNIFORM_VELOCITY_ADD_INTENSITY, 1, NULL);
  uniform_init(&uniforms.vel_radius, wgpu_context, UNIFORM_VELOCITY_ADD_RADIUS,
               1, NULL);
  uniform_init(&uniforms.vel_diff, wgpu_context, UNIFORM_VELOCITY_DIFFUSION, 1,
               NULL);
  uniform_init(&uniforms.dye_force, wgpu_context, UNIFORM_DYE_ADD_INTENSITY, 1,
               NULL);
  uniform_init(&uniforms.dye_radius, wgpu_context, UNIFORM_DYE_ADD_RADIUS, 1,
               NULL);
  uniform_init(&uniforms.dye_diff, wgpu_context, UNIFORM_DYE_ADD_DIFFUSION, 1,
               NULL);
  uniform_init(&uniforms.viscosity, wgpu_context, UNIFORM_VISCOSITY, 1, NULL);
  uniform_init(&uniforms.u_vorticity, wgpu_context, UNIFORM_VORTICITY, 1, NULL);
  uniform_init(&uniforms.contain_fluid, wgpu_context, UNIFORM_CONTAIN_FLUID, 1,
               NULL);
  uniform_init(&uniforms.u_symmetry, wgpu_context, UNIFORM_MOUSE_TYPE, 1, NULL);
}

/* Destruct uniforms */
static void uniforms_buffers_destroy()
{
  uniform_destroy(&uniforms.time);
  uniform_destroy(&uniforms.dt);
  uniform_destroy(&uniforms.mouse);
  uniform_destroy(&uniforms.grid);
  uniform_destroy(&uniforms.sim_speed);
  uniform_destroy(&uniforms.vel_force);
  uniform_destroy(&uniforms.vel_radius);
  uniform_destroy(&uniforms.vel_diff);
  uniform_destroy(&uniforms.dye_force);
  uniform_destroy(&uniforms.dye_radius);
  uniform_destroy(&uniforms.dye_diff);
  uniform_destroy(&uniforms.viscosity);
  uniform_destroy(&uniforms.u_vorticity);
  uniform_destroy(&uniforms.contain_fluid);
  uniform_destroy(&uniforms.u_symmetry);
}

/* -------------------------------------------------------------------------- *
 * Program
 * -------------------------------------------------------------------------- */

#define PROGRAM_MAX_BUFFER_COUNT 3u
#define PROGRAM_MAX_UNIFORM_COUNT 8u

/* Creates a shader module, compute pipeline & bind group to use with the GPU */
typedef struct {
  uint32_t dispatch_x; /* Dispatch workers width */
  uint32_t dispatch_y; /* Dispatch workers height */
  WGPUComputePipeline compute_pipeline;
  WGPUBindGroup bind_group;
} program_t;

static struct {
  program_t advect_dye_program;
  program_t advect_program;
  program_t boundary_div_program;
  program_t boundary_pressure_program;
  program_t boundary_program;
  program_t checker_program;
  program_t clear_pressure_program;
  program_t divergence_program;
  program_t gradient_subtract_program;
  program_t pressure_program;
  program_t update_dye_program;
  program_t update_program;
  program_t vorticity_confinment_program;
  program_t vorticity_program;
} programs;

static void program_init_defaults(program_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void program_init(program_t* this, wgpu_context_t* wgpu_context,
                         dynamic_buffer_t** buffers, uint32_t buffer_count,
                         uniform_t** uniforms, uint32_t uniform_count,
                         const char* shader_wgsl_path, uint32_t dispatch_x,
                         uint32_t dispatch_y)
{
  program_init_defaults(this);

  /* Create the shader module using the WGSL string and use it to create a
   * compute pipeline with 'auto' binding layout */
  {
    wgpu_shader_t comp_shader
      = wgpu_shader_create(wgpu_context, &(wgpu_shader_desc_t){
                                           /* Compute shader WGSL */
                                           .file  = shader_wgsl_path,
                                           .entry = "main",
                                         });
    this->compute_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .compute = comp_shader.programmable_stage_descriptor,
      });
    wgpu_shader_release(&comp_shader);
  }

  /* Concat the buffer & uniforms and format the entries to the right WebGPU
   * format */
  WGPUBindGroupEntry
    bg_entries[PROGRAM_MAX_BUFFER_COUNT + PROGRAM_MAX_UNIFORM_COUNT];
  uint32_t bge_i = 0;
  {
    for (uint32_t i = 0; i < buffer_count; ++i, ++bge_i) {
      bg_entries[bge_i] = (WGPUBindGroupEntry){
        .binding = bge_i,
        .buffer  = buffers[i]->buffers[0].buffer,
        .offset  = 0,
        .size    = buffers[i]->buffers[0].size,
      };
    }
    for (uint32_t i = 0; i < uniform_count; ++i, ++bge_i) {
      bg_entries[bge_i] = (WGPUBindGroupEntry){
        .binding = bge_i,
        .buffer  = uniforms[i]->buffer.buffer,
        .offset  = 0,
        .size    = uniforms[i]->size,
      };
    }
  }

  /* Create the bind group using these entries & auto-layout detection */
  {
    WGPUBindGroupDescriptor bg_desc = {
      .layout = wgpuComputePipelineGetBindGroupLayout(this->compute_pipeline,
                                                      0 /* index */),
      .entryCount = bge_i,
      .entries    = bg_entries,
    };
    this->bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  }

  this->dispatch_x = dispatch_x;
  this->dispatch_y = dispatch_y;
}

static void program_destroy(program_t* this)
{
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->compute_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

/* Dispatch the compute pipeline to the GPU */
static void program_dispatch(program_t* this,
                             WGPUComputePassEncoder pass_encoder)
{
  wgpuComputePassEncoderSetPipeline(pass_encoder, this->compute_pipeline);
  wgpuComputePassEncoderSetBindGroup(pass_encoder, 0, this->bind_group, 0,
                                     NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    pass_encoder, (uint32_t)ceil((float)this->dispatch_x / 8.0f),
    (uint32_t)ceil((float)this->dispatch_y / 8.0f), 1);
}

static void init_advect_dye_program(program_t* this,
                                    wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.dye0,     /* in_quantity */
    &dynamic_buffers.velocity, /* in_velocity */
    &dynamic_buffers.dye,      /* out_quantity */
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid, /* */
    &uniforms.dt,   /* */
  };
  const char* shader_wgsl_path = "advect_dye_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.dye_w, settings.dye_h);
}

static void init_advect_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.velocity0, /* in_quantity */
    &dynamic_buffers.velocity0, /* in_velocity */
    &dynamic_buffers.velocity,  /* out_quantity */
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid, /* */
    &uniforms.dt,   /* */
  };
  const char* shader_wgsl_path = "advect_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_boundary_div_program(program_t* this,
                                      wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.divergence0, /* in_quantity */
    &dynamic_buffers.divergence,  /* out_quantity */
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid, /* */
  };
  const char* shader_wgsl_path = "boundary_pressure_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_boundary_pressure_program(program_t* this,
                                           wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.pressure0, /* in_quantity */
    &dynamic_buffers.pressure,  /* out_quantity */
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid, /* */
  };
  const char* shader_wgsl_path = "boundary_pressure_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_boundary_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.velocity,  /* in_quantity */
    &dynamic_buffers.velocity0, /* out_quantity */
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid,          /* */
    &uniforms.contain_fluid, /* */
  };
  const char* shader_wgsl_path = "boundary_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_clear_pressure_program(program_t* this,
                                        wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.pressure,  /* in_quantity */
    &dynamic_buffers.pressure0, /* out_quantity */
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid,      /* */
    &uniforms.viscosity, /* */
  };
  const char* shader_wgsl_path = "clear_pressure_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_checker_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[1] = {
    &dynamic_buffers.dye, /* */
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid, /* */
  };
  const char* shader_wgsl_path = "checkerboard_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.dye_w, settings.dye_h);
}

static void init_divergence_program(program_t* this,
                                    wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.velocity0,   /* in_velocity */
    &dynamic_buffers.divergence0, /* out_divergence */
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid, /* */
  };
  const char* shader_wgsl_path = "divergence_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_gradient_subtract_program(program_t* this,
                                           wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.pressure,  /* in_pressure */
    &dynamic_buffers.velocity0, /* in_velocity */
    &dynamic_buffers.velocity,  /* out_velocity */
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid, /* */
  };
  const char* shader_wgsl_path = "gradient_subtract_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_pressure_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.pressure,   /* in_pressure */
    &dynamic_buffers.divergence, /* in_divergence */
    &dynamic_buffers.pressure0,  /* out_pressure */
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid, /* */
  };
  const char* shader_wgsl_path = "pressure_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_update_dye_program(program_t* this,
                                    wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.dye,  /* in_quantity */
    &dynamic_buffers.dye0, /* out_quantity */
  };
  uniform_t* program_uniforms[8] = {
    &uniforms.grid,       /* */
    &uniforms.mouse,      /* */
    &uniforms.dye_force,  /* */
    &uniforms.dye_radius, /* */
    &uniforms.dye_diff,   /* */
    &uniforms.time,       /* */
    &uniforms.dt,         /* */
    &uniforms.u_symmetry, /* */
  };
  const char* shader_wgsl_path = "update_dye_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.dye_w, settings.dye_h);
}

static void init_update_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.velocity,  /* in_quantity */
    &dynamic_buffers.velocity0, /* out_quantity */
  };
  uniform_t* program_uniforms[8] = {
    &uniforms.grid,       /* */
    &uniforms.mouse,      /* */
    &uniforms.vel_force,  /* */
    &uniforms.vel_radius, /* */
    &uniforms.vel_diff,   /* */
    &uniforms.dt,         /* */
    &uniforms.time,       /* */
    &uniforms.u_symmetry, /* */
  };
  const char* shader_wgsl_path = "update_velocity_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_vorticity_confinment_program(program_t* this,
                                              wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.velocity,  /* in_velocity */
    &dynamic_buffers.vorticity, /* out_vorticity */
    &dynamic_buffers.velocity0, /* out_vorticity */
  };
  uniform_t* program_uniforms[3] = {
    &uniforms.grid,        /* */
    &uniforms.dt,          /* */
    &uniforms.u_vorticity, /* */
  };
  const char* shader_wgsl_path = "vorticity_confinment_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

static void init_vorticity_program(program_t* this,
                                   wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.velocity,  /* in_velocity */
    &dynamic_buffers.vorticity, /* out_vorticity */
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid, /* */
  };
  const char* shader_wgsl_path = "vorticity_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_path,
               settings.grid_w, settings.grid_h);
}

/* Init programs */
static void programs_init(wgpu_context_t* wgpu_context)
{
  init_advect_dye_program(&programs.advect_dye_program, wgpu_context);
  init_advect_program(&programs.advect_program, wgpu_context);
  init_boundary_div_program(&programs.boundary_div_program, wgpu_context);
  init_boundary_pressure_program(&programs.boundary_pressure_program,
                                 wgpu_context);
  init_boundary_program(&programs.boundary_program, wgpu_context);
  init_clear_pressure_program(&programs.clear_pressure_program, wgpu_context);
  init_checker_program(&programs.checker_program, wgpu_context);
  init_divergence_program(&programs.divergence_program, wgpu_context);
  init_gradient_subtract_program(&programs.gradient_subtract_program,
                                 wgpu_context);
  init_pressure_program(&programs.pressure_program, wgpu_context);
  init_update_dye_program(&programs.update_dye_program, wgpu_context);
  init_update_program(&programs.update_program, wgpu_context);
  init_vorticity_confinment_program(&programs.vorticity_confinment_program,
                                    wgpu_context);
  init_vorticity_program(&programs.vorticity_program, wgpu_context);
}

static void programs_destroy()
{
  program_destroy(&programs.advect_dye_program);
  program_destroy(&programs.advect_program);
  program_destroy(&programs.boundary_div_program);
  program_destroy(&programs.boundary_pressure_program);
  program_destroy(&programs.boundary_program);
  program_destroy(&programs.clear_pressure_program);
  program_destroy(&programs.checker_program);
  program_destroy(&programs.divergence_program);
  program_destroy(&programs.gradient_subtract_program);
  program_destroy(&programs.pressure_program);
  program_destroy(&programs.update_dye_program);
  program_destroy(&programs.update_program);
  program_destroy(&programs.vorticity_confinment_program);
  program_destroy(&programs.vorticity_program);
}

/* -------------------------------------------------------------------------- *
 * Initialization
 * -------------------------------------------------------------------------- */

/* Downscale if necessary to prevent crashes */
static WGPUExtent3D get_valid_dimensions(uint32_t w, uint32_t h,
                                         uint64_t max_buffer_size,
                                         uint64_t max_canvas_size)
{
  float down_ratio = 1.0f;

  /* Prevent buffer size overflow */
  if (w * h * 4 >= max_buffer_size) {
    down_ratio = sqrt(max_buffer_size / (float)(w * h * 4));
  }

  /* Prevent canvas size overflow */
  if (w > max_canvas_size) {
    down_ratio = max_canvas_size / (float)w;
  }
  else if (h > max_canvas_size) {
    down_ratio = max_canvas_size / (float)h;
  }

  return (WGPUExtent3D){
    .width  = floor(w * down_ratio),
    .height = floor(h * down_ratio),
  };
}

/* Fit to screen while keeping the aspect ratio */
static WGPUExtent3D get_preferred_dimensions(uint32_t size,
                                             wgpu_context_t* wgpu_context,
                                             uint64_t max_buffer_size,
                                             uint64_t max_canvas_size)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  uint32_t w = 0, h = 0;

  if (wgpu_context->surface.height < wgpu_context->surface.width) {
    w = floor(size * aspect_ratio);
    h = size;
  }
  else {
    w = size;
    h = floor(size / aspect_ratio);
  }

  return get_valid_dimensions(w, h, max_buffer_size, max_canvas_size);
}

/* Init buffer & canvas dimensions to fit the screen while keeping the aspect
 * ratio and downscaling the dimensions if they exceed the device capabilities
 */
static void init_sizes(wgpu_context_t* wgpu_context)
{
  uint64_t max_buffer_size          = 0;
  uint64_t max_canvas_size          = 0;
  WGPUSupportedLimits device_limits = {0};
  if (wgpuAdapterGetLimits(wgpu_context->adapter, &device_limits)) {
    max_buffer_size = device_limits.limits.maxStorageBufferBindingSize;
    max_canvas_size = device_limits.limits.maxTextureDimension2D;
  }

  /* Calculate simulation buffer dimensions */
  WGPUExtent3D grid_size = get_preferred_dimensions(
    settings.grid_size, wgpu_context, max_buffer_size, max_canvas_size);
  settings.grid_w = grid_size.width;
  settings.grid_h = grid_size.height;

  /* Calculate dye & canvas buffer dimensions */
  WGPUExtent3D dye_size = get_preferred_dimensions(
    settings.dye_size, wgpu_context, max_buffer_size, max_canvas_size);
  settings.dye_w = dye_size.width;
  settings.dye_h = dye_size.height;

  /* Useful values for the simulation */
  settings.rdx     = settings.grid_size * 4;
  settings.dye_rdx = settings.dye_size * 4;
  settings.dx      = 1.0f / settings.rdx;
}

/* -------------------------------------------------------------------------- *
 * Render
 * -------------------------------------------------------------------------- */

/* Renders 3 (r, g, b) storage buffers to the canvas */
static struct {
  /* Vertex buffer */
  wgpu_buffer_t vertex_buffer;

  /* Render pipeline */
  WGPURenderPipeline render_pipeline;

  /* Bind groups stores the resources bound to the binding points in a shader */
  WGPUBindGroup render_bind_group;

  /* Render pass descriptor for frame buffer writes */
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } render_pass;
} render_program = {0};

// Shaders
// clang-format off
static const char* shader_wgsl = CODE(
  struct GridSize {
    w : f32,
    h : f32,
    dyeW: f32,
    dyeH: f32,
    dx : f32,
    rdx : f32,
    dyeRdx : f32
  }

  struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(1) uv : vec2<f32>,
  };

  @group(0) @binding(0) var<storage, read_write> fieldX : array<f32>;
  @group(0) @binding(1) var<storage, read_write> fieldY : array<f32>;
  @group(0) @binding(2) var<storage, read_write> fieldZ : array<f32>;
  @group(0) @binding(3) var<uniform> uGrid : GridSize;
  @group(0) @binding(4) var<uniform> multiplier : f32;
  @group(0) @binding(5) var<uniform> isRenderingDye : f32;

  @vertex
  fn vertex_main(@location(0) position: vec4<f32>) -> VertexOut
  {
    var output : VertexOut;
    output.position = position;
    output.uv = position.xy*.5+.5;
    return output;
  }

  @fragment
  fn fragment_main(fragData : VertexOut) -> @location(0) vec4<f32>
  {
    var w = uGrid.dyeW;
    var h = uGrid.dyeH;

    if (isRenderingDye != 1.) {
      w = uGrid.w;
      h = uGrid.h;
    }

    let fuv = vec2<f32>((floor(fragData.uv*vec2(w, h))));
    let id = u32(fuv.x + fuv.y * w);

    let r = fieldX[id];
    let g = fieldY[id];
    let b = fieldZ[id];
    var col = vec3(r, g, b);

    if (r == g && r == b) {
      if (r < 0.) {col = mix(vec3(0.), vec3(0., 0., 1.), abs(r));}
      else {col = mix(vec3(0.), vec3(1., 0., 0.), r);}
    }
    return vec4(col, 1) * multiplier;
  }
);
// clang-format on

static void render_program_prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  const float vertices[24] = {
    -1, -1, 0, 1, -1, 1, 0, 1, 1, -1, 0, 1,
    1,  -1, 0, 1, -1, 1, 0, 1, 1, 1,  0, 1,
  };

  render_program.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertices),
                    .initial.data = vertices,
                  });
}

static void render_program_destroy()
{
  wgpu_destroy_buffer(&render_program.vertex_buffer);
  uniform_destroy(&uniforms.u_render_intensity);
  uniform_destroy(&uniforms.u_render_dye);
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_program.render_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, render_program.render_bind_group)
}

static void render_program_prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    fluid_simulation, 16, WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, 0))

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label            = "vertex_shader_wgsl",
                  .wgsl_code.source = shader_wgsl,
                  .entry            = "vertex_main",
                },
                .buffer_count = 1,
                .buffers      = &fluid_simulation_vertex_buffer_layout,
              });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .label            = "fragment_shader_wgsl",
                  .wgsl_code.source = shader_wgsl,
                  .entry            = "fragment_main",
                },
                .target_count = 1,
                .targets      = &color_target_state,
              });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  render_program.render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "fluid_simulation_render_pipeline",
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(render_program.render_pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void render_program_setup_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[6] = {
    /* Binding 0 : fieldX */
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = dynamic_buffers.rgb_buffer.buffers[0].buffer,
      .size    = dynamic_buffers.rgb_buffer.buffer_size,
    },
    /* Binding 1 : fieldY */
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .buffer  = dynamic_buffers.rgb_buffer.buffers[1].buffer,
      .size    = dynamic_buffers.rgb_buffer.buffer_size,
    },
    /* Binding 2 : fieldZ */
    [2] = (WGPUBindGroupEntry) {
      .binding = 2,
      .buffer  = dynamic_buffers.rgb_buffer.buffers[2].buffer,
      .size    = dynamic_buffers.rgb_buffer.buffer_size,
    },
    /* Binding 3 : uGrid */
    [3] = (WGPUBindGroupEntry) {
      .binding = 3,
      .buffer  = uniforms.grid.buffer.buffer,
      .size    = uniforms.grid.buffer.size,
    },
    /* Binding 4 : multiplier */
    [4] = (WGPUBindGroupEntry) {
      .binding = 4,
      .buffer  = uniforms.u_render_intensity.buffer.buffer,
      .size    = uniforms.u_render_intensity.buffer.size,
    },
    /* Binding 4 : isRenderingDye */
    [5] = (WGPUBindGroupEntry) {
      .binding = 5,
      .buffer  = uniforms.u_render_dye.buffer.buffer,
      .size    = uniforms.u_render_dye.buffer.size,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label = "render bind group",
    .layout
    = wgpuRenderPipelineGetBindGroupLayout(render_program.render_pipeline, 0),
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  render_program.render_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(render_program.render_bind_group != NULL);
}

/* The r,g,b buffer containing the data to render */
static void render_program_setup_rgb_buffer(wgpu_context_t* wgpu_context)
{
  dynamic_buffer_init(&dynamic_buffers.rgb_buffer, wgpu_context, /* dims: */ 3,
                      /* w: */ settings.dye_w, /* h: */ settings.dye_h);
}

/* Uniforms */
static void render_program_setup_render_uniforms(wgpu_context_t* wgpu_context)
{
  const float value = 1;

  uniform_init(&uniforms.u_render_intensity, wgpu_context,
               UNIFORM_RENDER_INTENSITY, 1, &value);
  uniform_init(&uniforms.u_render_dye, wgpu_context, UNIFORM_RENDER_DYE, 1,
               &value);
}

static void render_program_setup_render_pass()
{
  /* Color attachment */
  render_program.render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  /* Render pass descriptor */
  render_program.render_pass.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_program.render_pass.color_attachments,
    .depthStencilAttachment = NULL,
  };
}

static void render_program_initialize(wgpu_context_t* wgpu_context)
{
  render_program_prepare_vertex_buffer(wgpu_context);
  render_program_prepare_pipelines(wgpu_context);
  render_program_setup_rgb_buffer(wgpu_context);
  render_program_setup_render_uniforms(wgpu_context);
  render_program_setup_bind_group(wgpu_context);
  render_program_setup_render_pass();
}

/* Dispatch a draw command to render on the canvas */
static void render_program_dispatch(wgpu_context_t* wgpu_context,
                                    WGPUCommandEncoder command_encoder)
{
  render_program.render_pass.color_attachments[0].view
    = wgpu_context->swap_chain.frame_buffer;

  WGPURenderPassEncoder render_pass_encoder = wgpuCommandEncoderBeginRenderPass(
    command_encoder, &render_program.render_pass.descriptor);

  wgpuRenderPassEncoderSetPipeline(render_pass_encoder,
                                   render_program.render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass_encoder, 0,
                                    render_program.render_bind_group, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(render_pass_encoder, 0,
                                       render_program.vertex_buffer.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(render_pass_encoder, 6, 1, 0, 0);
  wgpuRenderPassEncoderEnd(render_pass_encoder);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass_encoder)
}

/* -------------------------------------------------------------------------- *
 * Fluid simulation
 * -------------------------------------------------------------------------- */

static const char* example_title = "Fluid Simulation";
static bool prepared             = false;

static struct {
  uint64_t loop;
  float last_frame;
} simulation = {
  .loop       = 0,
  .last_frame = 0,
};

/* Simulation reset */
static void simulation_reset()
{
  dynamic_buffer_clear(&dynamic_buffers.velocity);
  dynamic_buffer_clear(&dynamic_buffers.dye);
  dynamic_buffer_clear(&dynamic_buffers.pressure);

  settings.time   = 0.0f;
  simulation.loop = 0;
}

/* Fluid simulation step */
static void
simulation_dispatch_compute_pipeline(WGPUComputePassEncoder pass_encoder)
{
  /* Add velocity and dye at the mouse position */
  program_dispatch(&programs.update_dye_program, pass_encoder);
  program_dispatch(&programs.update_program, pass_encoder);

  /* Advect the velocity field through itself */
  program_dispatch(&programs.advect_program, pass_encoder);
  program_dispatch(&programs.boundary_program, pass_encoder);

  /* Compute the divergence */
  program_dispatch(&programs.divergence_program, pass_encoder);
  program_dispatch(&programs.boundary_div_program, pass_encoder);

  /* Solve the jacobi-pressure equation */
  for (uint32_t i = 0; i < settings.pressure_iterations; ++i) {
    program_dispatch(&programs.pressure_program, pass_encoder);
    /* boundary conditions */
    program_dispatch(&programs.boundary_pressure_program, pass_encoder);
  }

  /* Subtract the pressure from the velocity field */
  program_dispatch(&programs.gradient_subtract_program, pass_encoder);
  program_dispatch(&programs.clear_pressure_program, pass_encoder);

  /* Compute & apply vorticity confinment */
  program_dispatch(&programs.vorticity_program, pass_encoder);
  program_dispatch(&programs.vorticity_confinment_program, pass_encoder);

  /* Advect the dye through the velocity field */
  program_dispatch(&programs.advect_dye_program, pass_encoder);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    init_sizes(context->wgpu_context);
    dynamic_buffers_init(context->wgpu_context);
    uniforms_buffers_init(context->wgpu_context);
    programs_init(context->wgpu_context);
    render_program_initialize(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    if (imgui_overlay_button(context->imgui_overlay, "Reset")) {
      simulation_reset();
    }
  }
}

/* Render loop */
static WGPUCommandBuffer
build_simulation_step_command_buffer(wgpu_example_context_t* context)
{
  /* WebGPU context */
  wgpu_context_t* wgpu_context = context->wgpu_context;

  /* Update time */
  const float now = context->frame.timestamp_millis;
  settings.dt     = MIN(1.0f / 60.0f, (now - simulation.last_frame) / 1000.0f)
                * settings.sim_speed;
  settings.time += settings.dt;
  simulation.last_frame = now;

  /* Update uniforms */
  for (uint32_t i = 0; i < (uint32_t)UNIFORM_COUNT; ++i) {
    uniform_update(global_uniforms[i], wgpu_context, NULL, 0);
  }

  /* Update mouse uniform */
  glm_vec2_sub(mouse_infos.current, mouse_infos.last, mouse_infos.velocity);
  float mouse_values[4] = {
    mouse_infos.current[0], mouse_infos.current[1],   /* current */
    mouse_infos.velocity[0], mouse_infos.velocity[1], /* velocity */
  };
  uniform_update(&uniforms.mouse, wgpu_context, mouse_values, 4);
  glm_vec2_copy(mouse_infos.current, mouse_infos.last);

  /* Compute fluid */
  render_program.render_pass.color_attachments[0].view
    = wgpu_context->swap_chain.frame_buffer;
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    simulation_dispatch_compute_pipeline(wgpu_context->cpass_enc);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  dynamic_buffer_copy_to(&dynamic_buffers.velocity0, &dynamic_buffers.velocity,
                         wgpu_context->cmd_enc);
  dynamic_buffer_copy_to(&dynamic_buffers.pressure0, &dynamic_buffers.pressure,
                         wgpu_context->cmd_enc);

  /* Copy the selected buffer to the render program */
  if (settings.buffer_view == DYNAMIC_BUFFER_DYE) {
    dynamic_buffer_copy_to(&dynamic_buffers.dye, &dynamic_buffers.rgb_buffer,
                           wgpu_context->cmd_enc);
  }
  else if (settings.buffer_view == DYNAMIC_BUFFER_VELOCITY) {
    dynamic_buffer_copy_to(&dynamic_buffers.velocity,
                           &dynamic_buffers.rgb_buffer, wgpu_context->cmd_enc);
  }
  else if (settings.buffer_view == DYNAMIC_BUFFER_DIVERGENCE) {
    dynamic_buffer_copy_to(&dynamic_buffers.divergence,
                           &dynamic_buffers.rgb_buffer, wgpu_context->cmd_enc);
  }
  else if (settings.buffer_view == DYNAMIC_BUFFER_PRESSURE) {
    dynamic_buffer_copy_to(&dynamic_buffers.pressure,
                           &dynamic_buffers.rgb_buffer, wgpu_context->cmd_enc);
  }
  else if (settings.buffer_view == DYNAMIC_BUFFER_VORTICITY) {
    dynamic_buffer_copy_to(&dynamic_buffers.vorticity,
                           &dynamic_buffers.rgb_buffer, wgpu_context->cmd_enc);
  }

  /* Draw fluid */
  render_program_dispatch(wgpu_context, wgpu_context->cmd_enc);

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL)
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_simulation_step_command_buffer(context);

  // Submit to queue
  submit_command_buffers(context);

  // Send commands to the GPU
  submit_frame(context);

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  dynamic_buffers_destroy();
  uniforms_buffers_destroy();
  programs_destroy();
  render_program_destroy();
}

void example_fluid_simulation(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
    .title  = example_title,
    .overlay = true,
    .vsync   = true,
  },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
