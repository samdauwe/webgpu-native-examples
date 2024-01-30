#include "example_base.h"

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

#define MAX_DIMENSIONS 3u

typedef enum {
  RENDER_MODE_CLASSIC,
  RENDER_MODE_SMOKE_2D,
  RENDER_MODE_SMOKE_3D_SHADOWS,
  RENDER_MODE_DEBUG_VELOCITY,
  RENDER_MODE_DEBUG_DIVERGENCE,
  RENDER_MODE_DEBUG_PRESSURE,
  RENDER_MODE_DEBUG_VORTICITY,
} render_modes_t;

typedef enum {
  DYNAMIC_BUFFER_VELOCITY,
  DYNAMIC_BUFFER_DYE,
  DYNAMIC_BUFFER_DIVERGENCE,
  DYNAMIC_BUFFER_PRESSURE,
  DYNAMIC_BUFFER_VORTICITY,
  DYNAMIC_BUFFER_RGB,
} dynamic_buffer_type_t;

typedef enum {
  INPUT_SYMMETRY_NONE,
  INPUT_SYMMETRY_HORIZONTAL,
  INPUT_SYMMETRY_VERTICAL,
  INPUT_SYMMETRY_BOTH,
  INPUT_SYMMETRY_CENTER,
} input_symmetry_t;

static struct {
  float render_mode;
  float grid_size;
  uint32_t grid_w;
  uint32_t grid_h;
  uint32_t dye_size;
  uint32_t dye_w;
  uint32_t dye_h;
  uint32_t rdx;
  uint32_t dye_rdx;
  float dx;
  float sim_speed;
  float contain_fluid;
  float velocity_add_intensity;
  float velocity_add_radius;
  float velocity_diffusion;
  float dye_add_intensity;
  float dye_add_radius;
  float dye_diffusion;
  float viscosity;
  float vorticity;
  float render_intensity_multiplier;
  float render_dye_buffer;
  int32_t pressure_iterations;
  dynamic_buffer_type_t buffer_view;
  float input_symmetry;
  int32_t raymarch_steps;
  float smoke_density;
  float enable_shadows;
  float shadow_intensity;
  float smoke_height;
  float light_height;
  float light_intensity;
  float light_falloff;
  float dt;
  float time;
  vec4 mouse;
} settings = {
  .render_mode                 = (float)RENDER_MODE_CLASSIC,
  .grid_size                   = 128.0f,
  .dye_size                    = 1024,
  .sim_speed                   = 5.0f,
  .contain_fluid               = 1.0f,
  .velocity_add_intensity      = 0.2f,
  .velocity_add_radius         = 0.0002f,
  .velocity_diffusion          = 0.9999f,
  .dye_add_intensity           = 1.0f,
  .dye_add_radius              = 0.001f,
  .dye_diffusion               = 0.98f,
  .viscosity                   = 0.8f,
  .vorticity                   = 2.0f,
  .render_intensity_multiplier = 1.0f,
  .render_dye_buffer           = 1.0f,
  .pressure_iterations         = 20,
  .buffer_view                 = DYNAMIC_BUFFER_DYE,
  .input_symmetry              = (float)INPUT_SYMMETRY_NONE,
  .raymarch_steps              = 12.0f,
  .smoke_density               = 40.0f,
  .enable_shadows              = 1.0f,
  .shadow_intensity            = 25.0f,
  .smoke_height                = 0.2f,
  .light_height                = 1.0f,
  .light_intensity             = 1.0f,
  .light_falloff               = 1.0f,
  .dt                          = 0.0f,
  .time                        = 0.0f,
  .mouse                       = GLM_VEC4_ZERO_INIT,
};

static struct {
  vec2 current;
  vec2 last;
  vec2 velocity;
} mouse_infos = {
  .current  = GLM_VEC2_ZERO_INIT,
  .last     = {0.0f, 1.0f}, /* y position needs to be inverted */
  .velocity = GLM_VEC2_ZERO_INIT,
};

/* -------------------------------------------------------------------------- *
 * Dynamic buffers
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
      .label = "Dynamic buffer",
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

/* Buffers */
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
} dynamic_buffers = {0};

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

static void dynamic_buffers_destroy(void)
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
}

/* -------------------------------------------------------------------------- *
 * Uniforms
 * -------------------------------------------------------------------------- */

typedef enum {
  UNIFORM_RENDER_MODE,                 /* render_mode */
  UNIFORM_TIME,                        /* time */
  UNIFORM_DT,                          /* dt */
  UNIFORM_MOUSE_INFOS,                 /* mouseInfos */
  UNIFORM_GRID_SIZE,                   /* gridSize */
  UNIFORM_SIM_SPEED,                   /* sim_speed */
  UNIFORM_VELOCITY_ADD_INTENSITY,      /* velocity_add_intensity */
  UNIFORM_VELOCITY_ADD_RADIUS,         /* velocity_add_radius */
  UNIFORM_VELOCITY_DIFFUSION,          /* velocity_diffusion */
  UNIFORM_DYE_ADD_INTENSITY,           /* dye_add_intensity */
  UNIFORM_DYE_ADD_RADIUS,              /* dye_add_radius */
  UNIFORM_DYE_ADD_DIFFUSION,           /* dye_diffusion */
  UNIFORM_VISCOSITY,                   /* viscosity */
  UNIFORM_VORTICITY,                   /* vorticity */
  UNIFORM_CONTAIN_FLUID,               /* contain_fluid */
  UNIFORM_MOUSE_TYPE,                  /* mouse_type */
  UNIFORM_SMOKE_PARAMETERS,            /* smoke_parameters */
  UNIFORM_RENDER_INTENSITY_MULTIPLIER, /* render_intensity_multiplier */
  UNIFORM_COUNT,
} uniform_type_t;

#define MAX_UNIFORM_VALUE_COUNT 8u

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
  uniform_t render_mode;
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
  uniform_t vorticity;
  uniform_t contain_fluid;
  uniform_t symmetry;
  uniform_t smoke_parameters;
  uniform_t render_intensity;
} uniforms = {0};

static uniform_t* global_uniforms[UNIFORM_COUNT] = {0};

static void uniform_init_defaults(uniform_t* this)
{
  memset(this, 0, sizeof(*this));
}

static float* uniform_get_setting_value(uniform_type_t type)
{
  switch (type) {
    case UNIFORM_RENDER_MODE: /* render_mode */
      return &settings.render_mode;
    case UNIFORM_TIME: /* time */
      return &settings.time;
    case UNIFORM_DT: /* dt */
      return &settings.dt;
    case UNIFORM_MOUSE_INFOS: /* mouseInfos */
      return settings.mouse;
    case UNIFORM_SIM_SPEED: /* sim_speed */
      return &settings.sim_speed;
    case UNIFORM_VELOCITY_ADD_INTENSITY: /* velocity_add_intensity */
      return &settings.velocity_add_intensity;
    case UNIFORM_VELOCITY_ADD_RADIUS: /* velocity_add_radius */
      return &settings.velocity_add_radius;
    case UNIFORM_VELOCITY_DIFFUSION: /* velocity_diffusion */
      return &settings.velocity_diffusion;
    case UNIFORM_DYE_ADD_INTENSITY: /* dye_add_intensity */
      return &settings.dye_add_intensity;
    case UNIFORM_DYE_ADD_RADIUS: /* dye_add_radius */
      return &settings.dye_add_radius;
    case UNIFORM_DYE_ADD_DIFFUSION: /* dye_diffusion */
      return &settings.dye_diffusion;
    case UNIFORM_VISCOSITY: /* viscosity */
      return &settings.viscosity;
    case UNIFORM_VORTICITY: /* vorticity */
      return &settings.vorticity;
    case UNIFORM_CONTAIN_FLUID: /* contain_fluid */
      return &settings.contain_fluid;
    case UNIFORM_MOUSE_TYPE: /* mouse_type */
      return &settings.input_symmetry;
    case UNIFORM_RENDER_INTENSITY_MULTIPLIER: /* render_intensity_multiplier */
      return &settings.render_intensity_multiplier;
    default:
      return NULL;
  }
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
    float const* buff_value = value ? value : uniform_get_setting_value(type);
    this->buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Uniform buffer",
                                           .usage = WGPUBufferUsage_Uniform
                                                    | WGPUBufferUsage_CopyDst,
                                           .size = this->size * sizeof(float),
                                           .initial.data = buff_value,
                                         });
    if (buff_value) {
      memcpy(this->values, buff_value, this->size * sizeof(float));
    }
  }
  else {
    this->buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Uniform buffer",
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
                           float const* value, uint32_t value_count)
{
  if (this->needs_update || this->always_update || value != NULL) {
    float const* buff_value
      = value ? value : uniform_get_setting_value(this->type);
    if (value_count == 1 && buff_value[0] == value[0]) {
      return;
    }
    if (buff_value) {
      memcpy(this->values, buff_value, this->size * sizeof(float));
    }
    uint32_t buff_size
      = (value_count ? MIN(this->size, value_count) : this->size)
        * sizeof(float);
    wgpu_queue_write_buffer(wgpu_context, this->buffer.buffer, 0, this->values,
                            buff_size);
    this->needs_update = false;
  }
}

/* Initialize uniforms */
static void uniforms_buffers_init(wgpu_context_t* wgpu_context)
{
  float uniform_value = settings.render_mode;
  uniform_init(&uniforms.render_mode, wgpu_context, UNIFORM_RENDER_MODE, 1,
               &uniform_value);
  uniform_init(&uniforms.time, wgpu_context, UNIFORM_TIME, 1, NULL);
  uniform_init(&uniforms.dt, wgpu_context, UNIFORM_DT, 1, NULL);
  uniform_init(&uniforms.mouse, wgpu_context, UNIFORM_MOUSE_INFOS, 4, NULL);
  const float grid_values[7] = {
    settings.grid_w, settings.grid_h, settings.dye_w,   settings.dye_h,
    settings.dx,     settings.rdx,    settings.dye_rdx,
  };
  uniform_init(&uniforms.grid, wgpu_context, UNIFORM_GRID_SIZE, 7, grid_values);
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
  uniform_init(&uniforms.vorticity, wgpu_context, UNIFORM_VORTICITY, 1, NULL);
  uniform_init(&uniforms.contain_fluid, wgpu_context, UNIFORM_CONTAIN_FLUID, 1,
               NULL);
  uniform_init(&uniforms.symmetry, wgpu_context, UNIFORM_MOUSE_TYPE, 1, NULL);
  const float smoke_parameter_values[8] = {
    settings.raymarch_steps,   settings.smoke_density, settings.enable_shadows,
    settings.shadow_intensity, settings.smoke_height,  settings.light_height,
    settings.light_intensity,  settings.light_falloff,
  };
  uniform_init(&uniforms.smoke_parameters, wgpu_context,
               UNIFORM_SMOKE_PARAMETERS, 8, smoke_parameter_values);
  uniform_value = 1;
  uniform_init(&uniforms.render_intensity, wgpu_context,
               UNIFORM_RENDER_INTENSITY_MULTIPLIER, 1, &uniform_value);
}

/* Destruct uniforms */
static void uniforms_buffers_destroy(void)
{
  uniform_destroy(&uniforms.render_mode);
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
  uniform_destroy(&uniforms.vorticity);
  uniform_destroy(&uniforms.contain_fluid);
  uniform_destroy(&uniforms.symmetry);
  uniform_destroy(&uniforms.smoke_parameters);
  uniform_destroy(&uniforms.render_intensity);
}

/* -------------------------------------------------------------------------- *
 * Programs
 * -------------------------------------------------------------------------- */

#define PROGRAM_MAX_BUFFER_COUNT (3u * 3u)
#define PROGRAM_MAX_UNIFORM_COUNT 8u

/* Creates a shader module, compute pipeline & bind group to use with the GPU */
typedef struct {
  uint32_t dispatch_x; /* Dispatch workers width */
  uint32_t dispatch_y; /* Dispatch workers height */
  WGPUComputePipeline compute_pipeline;
  WGPUBindGroup bind_group;
} program_t;

static struct {
  program_t checker_program;
  program_t update_dye_program;
  program_t update_program;
  program_t advect_program;
  program_t boundary_program;
  program_t divergence_program;
  program_t boundary_div_program;
  program_t pressure_program;
  program_t boundary_pressure_program;
  program_t gradient_subtract_program;
  program_t advect_dye_program;
  program_t clear_pressure_program;
  program_t vorticity_program;
  program_t vorticity_confinment_program;
  program_t render_program;
} programs = {0};

static void program_init_defaults(program_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void program_init(program_t* this, wgpu_context_t* wgpu_context,
                         dynamic_buffer_t** buffers, uint32_t buffer_count,
                         uniform_t** uniforms, uint32_t uniform_count,
                         const char* shader_wgsl_filename, uint32_t dispatch_x,
                         uint32_t dispatch_y)
{
  program_init_defaults(this);

  /*
   * Create the shader module using the WGSL string and use it to create a
   * compute pipeline with 'auto' binding layout
   */
  {
    char shader_wgsl_path[STRMAX] = {0};
    snprintf(shader_wgsl_path, strlen(shader_wgsl_filename) + 25 + 1,
             "shaders/fluid_simulation/%s", shader_wgsl_filename);
    wgpu_shader_t comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      /* Compute shader WGSL */
                      .label = "Fluid simulation compute shader WGSL",
                      .file  = shader_wgsl_path,
                      .entry = "main",
                    });
    this->compute_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .compute = comp_shader.programmable_stage_descriptor,
      });
    ASSERT(this->compute_pipeline != NULL);
    wgpu_shader_release(&comp_shader);
  }

  /*
   * Concat the buffer & uniforms and format the entries to the right WebGPU
   * format
   */
  WGPUBindGroupEntry
    bg_entries[PROGRAM_MAX_BUFFER_COUNT + PROGRAM_MAX_UNIFORM_COUNT];
  uint32_t bge_i = 0;
  {
    for (uint32_t i = 0; i < buffer_count; ++i) {
      for (uint32_t d = 0; d < buffers[i]->dims; ++d) {
        bg_entries[bge_i] = (WGPUBindGroupEntry){
          .binding = bge_i,
          .buffer  = buffers[i]->buffers[d].buffer,
          .offset  = 0,
          .size    = buffers[i]->buffers[d].size,
        };
        ++bge_i;
      }
    }
    for (uint32_t i = 0; i < uniform_count; ++i, ++bge_i) {
      bg_entries[bge_i] = (WGPUBindGroupEntry){
        .binding = bge_i,
        .buffer  = uniforms[i]->buffer.buffer,
        .offset  = 0,
        .size    = uniforms[i]->buffer.size,
      };
    }
  }

  /* Create the bind group using these entries & auto-layout detection */
  {
    WGPUBindGroupDescriptor bg_desc = {
      .label  = "Bind group",
      .layout = wgpuComputePipelineGetBindGroupLayout(this->compute_pipeline,
                                                      0 /* index */),
      .entryCount = bge_i,
      .entries    = bg_entries,
    };
    this->bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->bind_group != NULL);
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
  const char* shader_wgsl_filename = "advect_dye_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
  const char* shader_wgsl_filename = "advect_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
  const char* shader_wgsl_filename = "boundary_pressure_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
  const char* shader_wgsl_filename = "boundary_pressure_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
  const char* shader_wgsl_filename = "boundary_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
  const char* shader_wgsl_filename = "clear_pressure_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
               settings.grid_w, settings.grid_h);
}

static void init_checker_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[1] = {
    &dynamic_buffers.dye, /* */
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid, /* */
    &uniforms.time, /* */
  };
  const char* shader_wgsl_filename = "checkerboard_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
  const char* shader_wgsl_filename = "divergence_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
  const char* shader_wgsl_filename = "gradient_subtract_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
  const char* shader_wgsl_filename = "pressure_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
    &uniforms.symmetry,   /* */
  };
  const char* shader_wgsl_filename = "update_dye_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
    &uniforms.symmetry,   /* */
  };
  const char* shader_wgsl_filename = "update_velocity_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
    &uniforms.grid,      /* */
    &uniforms.dt,        /* */
    &uniforms.vorticity, /* */
  };
  const char* shader_wgsl_filename = "vorticity_confinment_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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
  const char* shader_wgsl_filename = "vorticity_shader.wgsl";
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), shader_wgsl_filename,
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

static void programs_destroy(void)
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
  const WGPUExtent3D grid_size = get_preferred_dimensions(
    settings.grid_size, wgpu_context, max_buffer_size, max_canvas_size);
  settings.grid_w = grid_size.width;
  settings.grid_h = grid_size.height;

  /* Calculate dye & canvas buffer dimensions */
  settings.dye_w = (float)wgpu_context->surface.width;
  settings.dye_h = (float)wgpu_context->surface.height;

  /* Useful values for the simulation */
  settings.rdx     = settings.grid_size * 4;
  settings.dye_rdx = settings.dye_size * 4;
  settings.dx      = 1.0f / settings.rdx;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* render_program_shader_wgsl_vertex_main;
static const char* render_program_shader_wgsl_fragment_main;

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

static void render_program_prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  // clang-format off
  const float vertices[24] = {
    -1, -1, 0, 1, -1, 1, 0, 1, 1, -1, 0, 1,
     1, -1, 0, 1, -1, 1, 0, 1, 1,  1, 0, 1,
  };
  // clang-format on

  render_program.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertices),
                    .initial.data = vertices,
                  });
}

static void render_program_destroy(void)
{
  wgpu_destroy_buffer(&render_program.vertex_buffer);
  dynamic_buffer_destroy(&dynamic_buffers.rgb_buffer);
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

  /* WGSL Shader*/
  char* program_shader_wgsl
    = concat_strings(render_program_shader_wgsl_vertex_main,
                     render_program_shader_wgsl_fragment_main, "\n");

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label            = "Vertex shader WGSL",
                  .wgsl_code.source = program_shader_wgsl,
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
                  .label            = "Fragment shader WGSL",
                  .wgsl_code.source = program_shader_wgsl,
                  .entry            = "fragment_main",
                },
                .target_count = 1,
                .targets      = &color_target_state,
              });

  free(program_shader_wgsl);

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
  WGPUBindGroupEntry bg_entries[9] = {
    /* Binding 0 : fieldX */
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = dynamic_buffers.rgb_buffer.buffers[0].buffer,
      .size    = dynamic_buffers.rgb_buffer.buffers[0].size,
    },
    /* Binding 1 : fieldY */
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .buffer  = dynamic_buffers.rgb_buffer.buffers[1].buffer,
      .size    = dynamic_buffers.rgb_buffer.buffers[1].size,
    },
    /* Binding 2 : fieldZ */
    [2] = (WGPUBindGroupEntry) {
      .binding = 2,
      .buffer  = dynamic_buffers.rgb_buffer.buffers[2].buffer,
      .size    = dynamic_buffers.rgb_buffer.buffers[2].size,
    },
    /* Binding 3 : uGrid */
    [3] = (WGPUBindGroupEntry) {
      .binding = 3,
      .buffer  = uniforms.grid.buffer.buffer,
      .size    = uniforms.grid.buffer.size,
    },
    /* Binding 4 : uTime */
    [4] = (WGPUBindGroupEntry) {
      .binding = 4,
      .buffer  = uniforms.time.buffer.buffer,
      .size    = uniforms.time.buffer.size,
    },
    /* Binding 5 : uMouse */
    [5] = (WGPUBindGroupEntry) {
      .binding = 5,
      .buffer  = uniforms.mouse.buffer.buffer,
      .size    = uniforms.mouse.buffer.size,
    },
    /* Binding 6 : uRenderMode */
    [6] = (WGPUBindGroupEntry) {
      .binding = 6,
      .buffer  = uniforms.render_mode.buffer.buffer,
      .size    = uniforms.render_mode.buffer.size,
    },
    /* Binding 7 : uRenderIntensity */
    [7] = (WGPUBindGroupEntry) {
      .binding = 7,
      .buffer  = uniforms.render_intensity.buffer.buffer,
      .size    = uniforms.render_intensity.buffer.size,
    },
    /* Binding 8 : uSmokeParameters */
    [8] = (WGPUBindGroupEntry) {
      .binding = 8,
      .buffer  = uniforms.smoke_parameters.buffer.buffer,
      .size    = uniforms.smoke_parameters.buffer.size,
    },
  };

  WGPUBindGroupDescriptor bg_desc = {
    .label = "Render bind group",
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

static void render_program_setup_render_pass(void)
{
  /* Color attachment */
  render_program.render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  /* Render pass descriptor */
  render_program.render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = &render_program.render_pass.color_attachments[0],
    .depthStencilAttachment = NULL,
  };
}

static void render_program_init(wgpu_context_t* wgpu_context)
{
  render_program_prepare_vertex_buffer(wgpu_context);
  render_program_prepare_pipelines(wgpu_context);
  render_program_setup_rgb_buffer(wgpu_context);
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
static void simulation_reset(void)
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
  for (int32_t i = 0; i < settings.pressure_iterations; ++i) {
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
    /* Init buffers, uniforms and programs */
    dynamic_buffers_init(context->wgpu_context);
    uniforms_buffers_init(context->wgpu_context);
    programs_init(context->wgpu_context);
    render_program_init(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
#if _DEBUG_RENDER_MODES_
    static const char* render_modes[7] = {
      "Classic",           "Smoke 2D",           "Smoke 3D + Shadows",
      "Debug - Velocity",  "Debug - Divergence", "Debug - Pressure",
      "Debug - Vorticity",
    };
    static const float render_intensity_multipliers[7]
      = {1, 1, 1, 100, 10, 1e6, 1};
    int32_t render_mode_int = (int32_t)settings.render_mode;
    if (imgui_overlay_combo_box(context->imgui_overlay, "Mouse Symmetry",
                                &render_mode_int, render_modes,
                                ARRAY_SIZE(render_modes))) {
      uniform_update(&uniforms.render_intensity, context->wgpu_context,
                     &render_intensity_multipliers[render_mode_int], 1);
      settings.render_mode = (float)render_mode_int;
    }
#endif
    imgui_overlay_slider_int(context->imgui_overlay, "Pressure Iterations",
                             &settings.pressure_iterations, 0, 50);
    static const char* symmetry_types[5]
      = {"None", "Horizontal", "Vertical", "Both", "Center"};
    int32_t symmetry_value_int = (int32_t)settings.input_symmetry;
    if (imgui_overlay_combo_box(context->imgui_overlay, "Mouse Symmetry",
                                &symmetry_value_int, symmetry_types,
                                ARRAY_SIZE(symmetry_types))) {
      settings.input_symmetry = (float)symmetry_value_int;
      uniform_update(&uniforms.symmetry, context->wgpu_context,
                     &settings.input_symmetry, 1);
    }
    if (imgui_overlay_button(context->imgui_overlay, "Reset")) {
      simulation_reset();
    }
    if (imgui_overlay_header("Smoke Parameters")) {
      imgui_overlay_slider_int(context->imgui_overlay, "3D resolution",
                               &settings.raymarch_steps, 5, 20);
      imgui_overlay_slider_float(context->imgui_overlay, "Light Elevation",
                                 &settings.light_height, 0.5f, 1.0f, "%.3f");
      imgui_overlay_slider_float(context->imgui_overlay, "Light Intensity",
                                 &settings.light_intensity, 0.0f, 1.0f, "%.3f");
      imgui_overlay_slider_float(context->imgui_overlay, "Light Falloff",
                                 &settings.light_falloff, 0.5f, 10.0f, "%.3f");
      bool enable_shadows = settings.enable_shadows != 0.0f;
      if (imgui_overlay_checkBox(context->imgui_overlay, "Enable Shadows",
                                 &enable_shadows)) {
        settings.enable_shadows = enable_shadows ? 1.0f : 0.0f;
      }
      imgui_overlay_slider_float(context->imgui_overlay, "Shadow Intensity",
                                 &settings.shadow_intensity, 0.0f, 50.0f,
                                 "%.3f");
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

  /* Updated  mouse state */
  mouse_infos.current[0]
    = context->mouse_position[0] / context->wgpu_context->surface.width;
  mouse_infos.current[1]
    = 1.0f - context->mouse_position[1] / context->wgpu_context->surface.height;

  /* Update custom uniform */
  glm_vec2_sub(mouse_infos.current, mouse_infos.last, mouse_infos.velocity);
  float mouse_values[4] = {
    mouse_infos.current[0], mouse_infos.current[1],   /* current */
    mouse_infos.velocity[0], mouse_infos.velocity[1], /* velocity */
  };
  uniform_update(&uniforms.mouse, wgpu_context, mouse_values,
                 (uint32_t)ARRAY_SIZE(mouse_values));
  glm_vec2_copy(mouse_infos.current, mouse_infos.last);
  float smoke_parameter_values[8] = {
    settings.raymarch_steps,   settings.smoke_density, settings.enable_shadows,
    settings.shadow_intensity, settings.smoke_height,  settings.light_height,
    settings.light_intensity,  settings.light_falloff,
  };
  uniform_update(&uniforms.smoke_parameters, wgpu_context,
                 smoke_parameter_values,
                 (uint32_t)ARRAY_SIZE(smoke_parameter_values));

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

  /* Configure render mode */
  if ((int)settings.render_mode == RENDER_MODE_DEBUG_VELOCITY) {
    dynamic_buffer_copy_to(&dynamic_buffers.velocity,
                           &dynamic_buffers.rgb_buffer, wgpu_context->cmd_enc);
  }
  else if ((int)settings.render_mode == RENDER_MODE_DEBUG_DIVERGENCE) {
    dynamic_buffer_copy_to(&dynamic_buffers.divergence,
                           &dynamic_buffers.rgb_buffer, wgpu_context->cmd_enc);
  }
  else if ((int)settings.render_mode == RENDER_MODE_DEBUG_PRESSURE) {
    dynamic_buffer_copy_to(&dynamic_buffers.pressure,
                           &dynamic_buffers.rgb_buffer, wgpu_context->cmd_enc);
  }
  else if ((int)settings.render_mode == RENDER_MODE_DEBUG_VORTICITY) {
    dynamic_buffer_copy_to(&dynamic_buffers.vorticity,
                           &dynamic_buffers.rgb_buffer, wgpu_context->cmd_enc);
  }
  else {
    dynamic_buffer_copy_to(&dynamic_buffers.dye, &dynamic_buffers.rgb_buffer,
                           wgpu_context->cmd_enc);
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
    .title   = example_title,
    .overlay = true,
    .vsync   = true,
  },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
/**
 * @brief 3D Smoke Rendering inspired from @xjorma's shader:
 * @ref https://www.shadertoy.com/view/WlVyRV
 */
static const char* render_program_shader_wgsl_vertex_main = CODE(
  // -- STRUCT_GRID_SIZE -- //
  struct GridSize {
    w : f32,
    h : f32,
    dyeW: f32,
    dyeH: f32,
    dx : f32,
    rdx : f32,
    dyeRdx : f32
  }
  // -- STRUCT_GRID_SIZE -- //

  // -- STRUCT_MOUSE -- //
  struct Mouse {
    pos: vec2<f32>,
    vel: vec2<f32>,
  }
  // -- STRUCT_MOUSE -- //

  struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(1) uv : vec2<f32>,
  };

  struct SmokeData {
    raymarchSteps: f32,
    smokeDensity: f32,
    enableShadows: f32,
    shadowIntensity: f32,
    smokeHeight: f32,
    lightHeight: f32,
    lightIntensity: f32,
    lightFalloff: f32,
  }

  @group(0) @binding(0) var<storage, read> fieldX : array<f32>;
  @group(0) @binding(1) var<storage, read> fieldY : array<f32>;
  @group(0) @binding(2) var<storage, read> fieldZ : array<f32>;
  @group(0) @binding(3) var<uniform> uGrid : GridSize;
  @group(0) @binding(4) var<uniform> uTime : f32;
  @group(0) @binding(5) var<uniform> uMouse : Mouse;
  @group(0) @binding(6) var<uniform> isRenderingDye : f32;
  @group(0) @binding(7) var<uniform> multiplier : f32;
  @group(0) @binding(8) var<uniform> smokeData : SmokeData;

  @vertex
  fn vertex_main(@location(0) position: vec4<f32>) -> VertexOut
  {
    var output : VertexOut;
    output.position = position;
    output.uv = position.xy*.5+.5;
    return output;
  }
);

static const char* render_program_shader_wgsl_fragment_main = CODE(
  fn hash12(p: vec2<f32>) -> f32
  {
    var p3: vec3<f32>  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }

  fn getDye(pos : vec3<f32>) -> vec3<f32>
  {
    var uv = pos.xy;
    uv.x *= uGrid.h / uGrid.w;
    uv = uv * 0.5 + 0.5;

    if(max(uv.x, uv.y) > 1. || min(uv.x, uv.y) < 0.) {
      return vec3(0);
    }

    uv = floor(uv*vec2(uGrid.dyeW, uGrid.dyeH));
    let id = u32(uv.x + uv.y * uGrid.dyeW);

    return vec3(fieldX[id], fieldY[id], fieldZ[id]);
  }

  fn getLevel(dye: vec3<f32>) -> f32
  {
    return max(dye.r, max(dye.g, dye.b));
  }

  fn getMousePos() -> vec2<f32> {
    var pos = uMouse.pos;
    pos = (pos - .5) * 2.;
    pos.x *= uGrid.w / uGrid.h;
    return pos;
  }

  fn getShadow(p: vec3<f32>, lightPos: vec3<f32>, fogSlice: f32) -> f32 {
    let lightDir: vec3<f32> = normalize(lightPos - p);
    let lightDist: f32 = pow(max(0., dot(lightPos - p, lightPos - p) - smokeData.lightIntensity + 1.), smokeData.lightFalloff);
    var shadowDist: f32 = 0.;

    for (var i: f32 = 1.; i <= smokeData.raymarchSteps; i += 1.) {
      let sp: vec3<f32> = p + mix(0., lightDist*smokeData.smokeHeight, i / smokeData.raymarchSteps) * lightDir;
      if (sp.z > smokeData.smokeHeight) {
        break;
      }

      let height: f32 = getLevel(getDye(sp)) * smokeData.smokeHeight;
      shadowDist += min(max(0., height - sp.z), fogSlice);
    }

    return exp(-shadowDist * smokeData.shadowIntensity) / lightDist;
  }

  @fragment
  fn fragment_main(fragData : VertexOut) -> @location(0) vec4<f32>
  {
    var w = uGrid.dyeW;
    var h = uGrid.dyeH;

    if (isRenderingDye != 2.) {
      if (isRenderingDye > 1.) {
        w = uGrid.w;
        h = uGrid.h;
      }

      let fuv = vec2<f32>((floor(fragData.uv*vec2(w, h))));
      let id = u32(fuv.x + fuv.y * w);

      let r = fieldX[id] + uTime * 0. + uMouse.pos.x * 0.;
      let g = fieldY[id];
      let b = fieldZ[id];
      var col = vec3(r, g, b);

      if (isRenderingDye > 1.) {
        if (r < 0.) {col = mix(vec3(0.), vec3(0., 0., 1.), abs(r));}
        else {col = mix(vec3(0.), vec3(1., 0., 0.), r);}
      }

      return vec4(col * multiplier, 1);
    }

    var uv: vec2<f32> = fragData.uv * 2. - 1.;
    uv.x *= uGrid.dyeW / uGrid.dyeH;
    // let rd: vec3<f32> = normalize(vec3(uv, -1));
    // let ro: vec3<f32> = vec3(0,0,1);

    let theta = -1.5708;
    let phi = 3.141592 + 0.0001;// - (uMouse.pos.y - .5);
    let parralax = 20.;
    var ro: vec3<f32> = parralax * vec3(sin(phi)*cos(theta),cos(phi),sin(phi)*sin(theta));
    let cw = normalize(-ro);
    let cu = normalize(cross(cw, vec3(0, 0, 1)));
    let cv = normalize(cross(cu, cw));
    let ca = mat3x3(cu, cv, cw);
    var rd =  ca*normalize(vec3(uv, parralax));
    ro = ro.xzy; rd = rd.xzy;

    let bgCol: vec3<f32> = vec3(0,0,0);
    let fogSlice = smokeData.smokeHeight / smokeData.raymarchSteps;

    let near: f32 = (smokeData.smokeHeight - ro.z) / rd.z;
    let far: f32  = -ro.z / rd.z;

    let m = getMousePos();
    let lightPos: vec3<f32> = vec3(m, smokeData.lightHeight);

    var transmittance: f32 = 1.;
    var col: vec3<f32> = vec3(0.35,0.35,0.35) * 0.;

    for (var i: f32 = 0.; i <= smokeData.raymarchSteps; i += 1.) {
      let p: vec3<f32> = ro + mix(near, far, i / smokeData.raymarchSteps) * rd;

      let dyeColor: vec3<f32> = getDye(p);
      let height: f32 = getLevel(dyeColor) * smokeData.smokeHeight;
      let smple: f32 = min(max(0., height - p.z), fogSlice);

      if (smple > .0001) {
        var shadow: f32 = 1.;

        if (smokeData.enableShadows > 0.) {
          shadow = getShadow(p, lightPos, fogSlice);
        }

        let dens: f32 = smple*smokeData.smokeDensity;

        col += shadow * dens * transmittance * dyeColor;
        transmittance *= 1. - dens;
      }
    }

    return vec4(mix(bgCol, col, 1. - transmittance), 1);
  }
);
// clang-format on
