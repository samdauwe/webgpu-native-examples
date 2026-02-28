#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <cglm/cglm.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* render_program_shader_wgsl_vertex_main;
static const char* render_program_shader_wgsl_fragment_main;

/* -------------------------------------------------------------------------- *
 * Utility: read a text file into a malloc'd string
 * -------------------------------------------------------------------------- */

static char* read_file_to_string(const char* path)
{
  FILE* file = fopen(path, "rb");
  if (!file) {
    fprintf(stderr, "Failed to open file: %s\n", path);
    return NULL;
  }
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  char* buffer = (char*)malloc((size_t)size + 1);
  if (buffer) {
    size_t read  = fread(buffer, 1, (size_t)size, file);
    buffer[read] = '\0';
  }
  fclose(file);
  return buffer;
}

static char* concat_strings(const char* a, const char* b, const char* sep)
{
  size_t la    = strlen(a);
  size_t lb    = strlen(b);
  size_t ls    = strlen(sep);
  char* result = (char*)malloc(la + lb + ls + 1);
  memcpy(result, a, la);
  memcpy(result + la, sep, ls);
  memcpy(result + la + ls, b, lb);
  result[la + ls + lb] = '\0';
  return result;
}

/* -------------------------------------------------------------------------- *
 * Enums & Settings
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
  .last     = {0.0f, 1.0f},
  .velocity = GLM_VEC2_ZERO_INIT,
};

/* Mouse position in pixels, tracked via input events */
static struct {
  float x;
  float y;
} mouse_pixel_pos = {0};

/* -------------------------------------------------------------------------- *
 * Dynamic buffers
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  uint32_t dims;
  uint32_t buffer_size;
  uint32_t w;
  uint32_t h;
  wgpu_buffer_t buffers[MAX_DIMENSIONS];
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
      .label = STRVIEW("Dynamic - Storage buffer"),
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

static void dynamic_buffer_clear(dynamic_buffer_t* this)
{
  float* empty_buffer = (float*)malloc(this->buffer_size);
  memset(empty_buffer, 0, this->buffer_size);

  for (uint32_t i = 0; i < this->dims; ++i) {
    wgpuQueueWriteBuffer(this->wgpu_context->queue, this->buffers[i].buffer, 0,
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

  dynamic_buffer_t rgb_buffer;
} dynamic_buffers = {0};

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
  UNIFORM_RENDER_MODE,
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
  UNIFORM_SMOKE_PARAMETERS,
  UNIFORM_RENDER_INTENSITY_MULTIPLIER,
  UNIFORM_COUNT,
} uniform_type_t;

#define MAX_UNIFORM_VALUE_COUNT 8u

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
    case UNIFORM_RENDER_MODE:
      return &settings.render_mode;
    case UNIFORM_TIME:
      return &settings.time;
    case UNIFORM_DT:
      return &settings.dt;
    case UNIFORM_MOUSE_INFOS:
      return settings.mouse;
    case UNIFORM_SIM_SPEED:
      return &settings.sim_speed;
    case UNIFORM_VELOCITY_ADD_INTENSITY:
      return &settings.velocity_add_intensity;
    case UNIFORM_VELOCITY_ADD_RADIUS:
      return &settings.velocity_add_radius;
    case UNIFORM_VELOCITY_DIFFUSION:
      return &settings.velocity_diffusion;
    case UNIFORM_DYE_ADD_INTENSITY:
      return &settings.dye_add_intensity;
    case UNIFORM_DYE_ADD_RADIUS:
      return &settings.dye_add_radius;
    case UNIFORM_DYE_ADD_DIFFUSION:
      return &settings.dye_diffusion;
    case UNIFORM_VISCOSITY:
      return &settings.viscosity;
    case UNIFORM_VORTICITY:
      return &settings.vorticity;
    case UNIFORM_CONTAIN_FLUID:
      return &settings.contain_fluid;
    case UNIFORM_MOUSE_TYPE:
      return &settings.input_symmetry;
    case UNIFORM_RENDER_INTENSITY_MULTIPLIER:
      return &settings.render_intensity_multiplier;
    default:
      return NULL;
  }
}

static void uniform_init(uniform_t* this, wgpu_context_t* wgpu_context,
                         uniform_type_t type, uint32_t size, float const* value)
{
  uniform_init_defaults(this);

  this->type          = type;
  this->size          = size;
  this->needs_update  = false;
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
    wgpuQueueWriteBuffer(wgpu_context->queue, this->buffer.buffer, 0,
                         this->values, buff_size);
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

typedef struct {
  uint32_t dispatch_x;
  uint32_t dispatch_y;
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

  /* Read the WGSL shader file and create a shader module */
  {
    char shader_wgsl_path[256] = {0};
    snprintf(shader_wgsl_path, sizeof(shader_wgsl_path),
             "assets/shaders/fluid_simulation/%s", shader_wgsl_filename);
    char* shader_source = read_file_to_string(shader_wgsl_path);
    ASSERT(shader_source != NULL);

    WGPUShaderModule shader_module
      = wgpu_create_shader_module(wgpu_context->device, shader_source);
    ASSERT(shader_module != NULL);

    this->compute_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Fluid simulation - Compute pipeline"),
        .compute = (WGPUComputeState){
          .module     = shader_module,
          .entryPoint = STRVIEW("main"),
        },
      });
    ASSERT(this->compute_pipeline != NULL);

    wgpuShaderModuleRelease(shader_module);
    free(shader_source);
  }

  /* Build bind group entries from buffers + uniforms */
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

  /* Create bind group using auto-layout detection */
  {
    WGPUBindGroupLayout layout
      = wgpuComputePipelineGetBindGroupLayout(this->compute_pipeline, 0);
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Fluid simulation - Compute bind group"),
        .layout     = layout,
        .entryCount = bge_i,
        .entries    = bg_entries,
      });
    ASSERT(this->bind_group != NULL);
    wgpuBindGroupLayoutRelease(layout);
  }

  this->dispatch_x = dispatch_x;
  this->dispatch_y = dispatch_y;
}

static void program_destroy(program_t* this)
{
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->compute_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

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

/* -- Program init functions ----------------------------------------------- */

static void init_advect_dye_program(program_t* this,
                                    wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.dye0,
    &dynamic_buffers.velocity,
    &dynamic_buffers.dye,
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid,
    &uniforms.dt,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), "advect_dye_shader.wgsl",
               settings.dye_w, settings.dye_h);
}

static void init_advect_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.velocity0,
    &dynamic_buffers.velocity0,
    &dynamic_buffers.velocity,
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid,
    &uniforms.dt,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), "advect_shader.wgsl",
               settings.grid_w, settings.grid_h);
}

static void init_boundary_div_program(program_t* this,
                                      wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.divergence0,
    &dynamic_buffers.divergence,
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid,
  };
  program_init(
    this, wgpu_context, program_buffers, (uint32_t)ARRAY_SIZE(program_buffers),
    program_uniforms, (uint32_t)ARRAY_SIZE(program_uniforms),
    "boundary_pressure_shader.wgsl", settings.grid_w, settings.grid_h);
}

static void init_boundary_pressure_program(program_t* this,
                                           wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.pressure0,
    &dynamic_buffers.pressure,
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid,
  };
  program_init(
    this, wgpu_context, program_buffers, (uint32_t)ARRAY_SIZE(program_buffers),
    program_uniforms, (uint32_t)ARRAY_SIZE(program_uniforms),
    "boundary_pressure_shader.wgsl", settings.grid_w, settings.grid_h);
}

static void init_boundary_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.velocity,
    &dynamic_buffers.velocity0,
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid,
    &uniforms.contain_fluid,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), "boundary_shader.wgsl",
               settings.grid_w, settings.grid_h);
}

static void init_clear_pressure_program(program_t* this,
                                        wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.pressure,
    &dynamic_buffers.pressure0,
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid,
    &uniforms.viscosity,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms),
               "clear_pressure_shader.wgsl", settings.grid_w, settings.grid_h);
}

static void init_checker_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[1] = {
    &dynamic_buffers.dye,
  };
  uniform_t* program_uniforms[2] = {
    &uniforms.grid,
    &uniforms.time,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms),
               "checkerboard_shader.wgsl", settings.dye_w, settings.dye_h);
}

static void init_divergence_program(program_t* this,
                                    wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.velocity0,
    &dynamic_buffers.divergence0,
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), "divergence_shader.wgsl",
               settings.grid_w, settings.grid_h);
}

static void init_gradient_subtract_program(program_t* this,
                                           wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.pressure,
    &dynamic_buffers.velocity0,
    &dynamic_buffers.velocity,
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid,
  };
  program_init(
    this, wgpu_context, program_buffers, (uint32_t)ARRAY_SIZE(program_buffers),
    program_uniforms, (uint32_t)ARRAY_SIZE(program_uniforms),
    "gradient_subtract_shader.wgsl", settings.grid_w, settings.grid_h);
}

static void init_pressure_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.pressure,
    &dynamic_buffers.divergence,
    &dynamic_buffers.pressure0,
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), "pressure_shader.wgsl",
               settings.grid_w, settings.grid_h);
}

static void init_update_dye_program(program_t* this,
                                    wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.dye,
    &dynamic_buffers.dye0,
  };
  uniform_t* program_uniforms[8] = {
    &uniforms.grid,       &uniforms.mouse,    &uniforms.dye_force,
    &uniforms.dye_radius, &uniforms.dye_diff, &uniforms.time,
    &uniforms.dt,         &uniforms.symmetry,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), "update_dye_shader.wgsl",
               settings.dye_w, settings.dye_h);
}

static void init_update_program(program_t* this, wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.velocity,
    &dynamic_buffers.velocity0,
  };
  uniform_t* program_uniforms[8] = {
    &uniforms.grid,       &uniforms.mouse,    &uniforms.vel_force,
    &uniforms.vel_radius, &uniforms.vel_diff, &uniforms.dt,
    &uniforms.time,       &uniforms.symmetry,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms),
               "update_velocity_shader.wgsl", settings.grid_w, settings.grid_h);
}

static void init_vorticity_confinment_program(program_t* this,
                                              wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[3] = {
    &dynamic_buffers.velocity,
    &dynamic_buffers.vorticity,
    &dynamic_buffers.velocity0,
  };
  uniform_t* program_uniforms[3] = {
    &uniforms.grid,
    &uniforms.dt,
    &uniforms.vorticity,
  };
  program_init(
    this, wgpu_context, program_buffers, (uint32_t)ARRAY_SIZE(program_buffers),
    program_uniforms, (uint32_t)ARRAY_SIZE(program_uniforms),
    "vorticity_confinment_shader.wgsl", settings.grid_w, settings.grid_h);
}

static void init_vorticity_program(program_t* this,
                                   wgpu_context_t* wgpu_context)
{
  dynamic_buffer_t* program_buffers[2] = {
    &dynamic_buffers.velocity,
    &dynamic_buffers.vorticity,
  };
  uniform_t* program_uniforms[1] = {
    &uniforms.grid,
  };
  program_init(this, wgpu_context, program_buffers,
               (uint32_t)ARRAY_SIZE(program_buffers), program_uniforms,
               (uint32_t)ARRAY_SIZE(program_uniforms), "vorticity_shader.wgsl",
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

static WGPUExtent3D get_valid_dimensions(uint32_t w, uint32_t h,
                                         uint64_t max_buffer_size,
                                         uint64_t max_canvas_size)
{
  float down_ratio = 1.0f;

  if (w * h * 4 >= max_buffer_size) {
    down_ratio = sqrt(max_buffer_size / (float)(w * h * 4));
  }

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

static WGPUExtent3D get_preferred_dimensions(uint32_t size,
                                             wgpu_context_t* wgpu_context,
                                             uint64_t max_buffer_size,
                                             uint64_t max_canvas_size)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  uint32_t w = 0, h = 0;

  if (wgpu_context->height < wgpu_context->width) {
    w = floor(size * aspect_ratio);
    h = size;
  }
  else {
    w = size;
    h = floor(size / aspect_ratio);
  }

  return get_valid_dimensions(w, h, max_buffer_size, max_canvas_size);
}

static void init_sizes(wgpu_context_t* wgpu_context)
{
  uint64_t max_buffer_size = 0;
  uint64_t max_canvas_size = 0;
  WGPULimits device_limits = {0};
  if (wgpuAdapterGetLimits(wgpu_context->adapter, &device_limits)
      == WGPUStatus_Success) {
    max_buffer_size = device_limits.maxStorageBufferBindingSize;
    max_canvas_size = device_limits.maxTextureDimension2D;
  }

  const WGPUExtent3D grid_size = get_preferred_dimensions(
    settings.grid_size, wgpu_context, max_buffer_size, max_canvas_size);
  settings.grid_w = grid_size.width;
  settings.grid_h = grid_size.height;

  settings.dye_w = (float)wgpu_context->width;
  settings.dye_h = (float)wgpu_context->height;

  settings.rdx     = settings.grid_size * 4;
  settings.dye_rdx = settings.dye_size * 4;
  settings.dx      = 1.0f / settings.rdx;
}

/* -------------------------------------------------------------------------- *
 * Render
 * -------------------------------------------------------------------------- */

static struct {
  wgpu_buffer_t vertex_buffer;
  WGPURenderPipeline render_pipeline;
  WGPUBindGroup render_bind_group;

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
                    .label = "Render program - Vertex buffer",
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
  /* Combine vertex and fragment WGSL into one shader module */
  char* combined_shader_wgsl
    = concat_strings(render_program_shader_wgsl_vertex_main,
                     render_program_shader_wgsl_fragment_main, "\n");

  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, combined_shader_wgsl);
  ASSERT(shader_module != NULL);
  free(combined_shader_wgsl);

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    fluid_simulation, 16, WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, 0))

  /* Blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  render_program.render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Fluid simulation - Render pipeline"),
      .vertex = (WGPUVertexState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("vertex_main"),
        .bufferCount = 1,
        .buffers     = &fluid_simulation_vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("fragment_main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });
  ASSERT(render_program.render_pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);
}

static void render_program_setup_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[9] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = dynamic_buffers.rgb_buffer.buffers[0].buffer,
      .size    = dynamic_buffers.rgb_buffer.buffers[0].size,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = dynamic_buffers.rgb_buffer.buffers[1].buffer,
      .size    = dynamic_buffers.rgb_buffer.buffers[1].size,
    },
    [2] = (WGPUBindGroupEntry){
      .binding = 2,
      .buffer  = dynamic_buffers.rgb_buffer.buffers[2].buffer,
      .size    = dynamic_buffers.rgb_buffer.buffers[2].size,
    },
    [3] = (WGPUBindGroupEntry){
      .binding = 3,
      .buffer  = uniforms.grid.buffer.buffer,
      .size    = uniforms.grid.buffer.size,
    },
    [4] = (WGPUBindGroupEntry){
      .binding = 4,
      .buffer  = uniforms.time.buffer.buffer,
      .size    = uniforms.time.buffer.size,
    },
    [5] = (WGPUBindGroupEntry){
      .binding = 5,
      .buffer  = uniforms.mouse.buffer.buffer,
      .size    = uniforms.mouse.buffer.size,
    },
    [6] = (WGPUBindGroupEntry){
      .binding = 6,
      .buffer  = uniforms.render_mode.buffer.buffer,
      .size    = uniforms.render_mode.buffer.size,
    },
    [7] = (WGPUBindGroupEntry){
      .binding = 7,
      .buffer  = uniforms.render_intensity.buffer.buffer,
      .size    = uniforms.render_intensity.buffer.size,
    },
    [8] = (WGPUBindGroupEntry){
      .binding = 8,
      .buffer  = uniforms.smoke_parameters.buffer.buffer,
      .size    = uniforms.smoke_parameters.buffer.size,
    },
  };

  WGPUBindGroupLayout layout
    = wgpuRenderPipelineGetBindGroupLayout(render_program.render_pipeline, 0);
  render_program.render_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Fluid simulation - Render bind group"),
      .layout     = layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(render_program.render_bind_group != NULL);
  wgpuBindGroupLayoutRelease(layout);
}

static void render_program_setup_rgb_buffer(wgpu_context_t* wgpu_context)
{
  dynamic_buffer_init(&dynamic_buffers.rgb_buffer, wgpu_context, 3,
                      settings.dye_w, settings.dye_h);
}

static void render_program_setup_render_pass(void)
{
  render_program.render_pass.color_attachments[0]
    = (WGPURenderPassColorAttachment){
      .view       = NULL, /* Assigned later */
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor){0.0f, 0.0f, 0.0f, 1.0f},
    };

  render_program.render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = STRVIEW("Fluid simulation - Render pass"),
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_program.render_pass.color_attachments,
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

static void render_program_dispatch(wgpu_context_t* wgpu_context,
                                    WGPUCommandEncoder command_encoder)
{
  render_program.render_pass.color_attachments[0].view
    = wgpu_context->swapchain_view;

  WGPURenderPassEncoder render_pass_encoder = wgpuCommandEncoderBeginRenderPass(
    command_encoder, &render_program.render_pass.descriptor);

  wgpuRenderPassEncoderSetPipeline(render_pass_encoder,
                                   render_program.render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass_encoder, 0,
                                    render_program.render_bind_group, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(render_pass_encoder, 0,
                                       render_program.vertex_buffer.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(render_pass_encoder, 6, 1, 0, 0);
  wgpuRenderPassEncoderEnd(render_pass_encoder);
  wgpuRenderPassEncoderRelease(render_pass_encoder);
}

/* -------------------------------------------------------------------------- *
 * Fluid simulation
 * -------------------------------------------------------------------------- */

static struct {
  uint64_t loop;
  uint64_t last_frame_ticks;
  float last_frame;
  WGPUBool initialized;
} simulation = {
  .loop             = 0,
  .last_frame_ticks = 0,
  .last_frame       = 0.0f,
  .initialized      = false,
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
  program_dispatch(&programs.update_dye_program, pass_encoder);
  program_dispatch(&programs.update_program, pass_encoder);

  program_dispatch(&programs.advect_program, pass_encoder);
  program_dispatch(&programs.boundary_program, pass_encoder);

  program_dispatch(&programs.divergence_program, pass_encoder);
  program_dispatch(&programs.boundary_div_program, pass_encoder);

  for (int32_t i = 0; i < settings.pressure_iterations; ++i) {
    program_dispatch(&programs.pressure_program, pass_encoder);
    program_dispatch(&programs.boundary_pressure_program, pass_encoder);
  }

  program_dispatch(&programs.gradient_subtract_program, pass_encoder);
  program_dispatch(&programs.clear_pressure_program, pass_encoder);

  program_dispatch(&programs.vorticity_program, pass_encoder);
  program_dispatch(&programs.vorticity_confinment_program, pass_encoder);

  program_dispatch(&programs.advect_dye_program, pass_encoder);
}

/* -- GUI ------------------------------------------------------------------ */

static void render_gui(wgpu_context_t* wgpu_context)
{
  const uint64_t now = stm_now();
  const float dt_sec
    = (float)stm_sec(stm_diff(now, simulation.last_frame_ticks));
  simulation.last_frame_ticks = now;

  imgui_overlay_new_frame(wgpu_context, dt_sec);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
    igSliderInt("Pressure Iterations", &settings.pressure_iterations, 0, 50,
                "%d");

    static const char* symmetry_types[5]
      = {"None", "Horizontal", "Vertical", "Both", "Center"};
    int32_t symmetry_value_int = (int32_t)settings.input_symmetry;
    if (igCombo("Mouse Symmetry", &symmetry_value_int, symmetry_types, 5, 5)) {
      settings.input_symmetry = (float)symmetry_value_int;
      uniform_update(&uniforms.symmetry, wgpu_context, &settings.input_symmetry,
                     1);
    }

    if (igButton("Reset", (ImVec2){0, 0})) {
      simulation_reset();
    }

    if (igCollapsingHeader("Smoke Parameters", 0)) {
      igSliderInt("3D resolution", &settings.raymarch_steps, 5, 20, "%d");
      igSliderFloat("Light Elevation", &settings.light_height, 0.5f, 1.0f,
                    "%.3f", 0);
      igSliderFloat("Light Intensity", &settings.light_intensity, 0.0f, 1.0f,
                    "%.3f", 0);
      igSliderFloat("Light Falloff", &settings.light_falloff, 0.5f, 10.0f,
                    "%.3f", 0);
      bool enable_shadows = settings.enable_shadows != 0.0f;
      if (igCheckbox("Enable Shadows", &enable_shadows)) {
        settings.enable_shadows = enable_shadows ? 1.0f : 0.0f;
      }
      igSliderFloat("Shadow Intensity", &settings.shadow_intensity, 0.0f, 50.0f,
                    "%.3f", 0);
    }
  }

  igEnd();
}

/* -- Init / Frame / Shutdown ---------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();

    init_sizes(wgpu_context);
    dynamic_buffers_init(wgpu_context);
    uniforms_buffers_init(wgpu_context);
    programs_init(wgpu_context);
    render_program_init(wgpu_context);
    imgui_overlay_init(wgpu_context);

    simulation.initialized = true;
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!simulation.initialized) {
    return EXIT_FAILURE;
  }

  /* Update time */
  const float now = (float)stm_ms(stm_now()) / 1000.0f;
  settings.dt
    = MIN(1.0f / 60.0f, (now - simulation.last_frame)) * settings.sim_speed;
  settings.time += settings.dt;
  simulation.last_frame = now;

  /* Update uniforms */
  for (uint32_t i = 0; i < (uint32_t)UNIFORM_COUNT; ++i) {
    uniform_update(global_uniforms[i], wgpu_context, NULL, 0);
  }

  /* Update mouse state */
  mouse_infos.current[0] = mouse_pixel_pos.x / (float)wgpu_context->width;
  mouse_infos.current[1]
    = 1.0f - mouse_pixel_pos.y / (float)wgpu_context->height;

  glm_vec2_sub(mouse_infos.current, mouse_infos.last, mouse_infos.velocity);
  float mouse_values[4] = {
    mouse_infos.current[0],
    mouse_infos.current[1],
    mouse_infos.velocity[0],
    mouse_infos.velocity[1],
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

  /* Render GUI */
  render_gui(wgpu_context);

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Compute pass */
  {
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    simulation_dispatch_compute_pipeline(cpass_enc);
    wgpuComputePassEncoderEnd(cpass_enc);
    wgpuComputePassEncoderRelease(cpass_enc);
  }

  /* Copy buffers */
  dynamic_buffer_copy_to(&dynamic_buffers.velocity0, &dynamic_buffers.velocity,
                         cmd_enc);
  dynamic_buffer_copy_to(&dynamic_buffers.pressure0, &dynamic_buffers.pressure,
                         cmd_enc);

  /* Configure render mode */
  if ((int)settings.render_mode == RENDER_MODE_DEBUG_VELOCITY) {
    dynamic_buffer_copy_to(&dynamic_buffers.velocity,
                           &dynamic_buffers.rgb_buffer, cmd_enc);
  }
  else if ((int)settings.render_mode == RENDER_MODE_DEBUG_DIVERGENCE) {
    dynamic_buffer_copy_to(&dynamic_buffers.divergence,
                           &dynamic_buffers.rgb_buffer, cmd_enc);
  }
  else if ((int)settings.render_mode == RENDER_MODE_DEBUG_PRESSURE) {
    dynamic_buffer_copy_to(&dynamic_buffers.pressure,
                           &dynamic_buffers.rgb_buffer, cmd_enc);
  }
  else if ((int)settings.render_mode == RENDER_MODE_DEBUG_VORTICITY) {
    dynamic_buffer_copy_to(&dynamic_buffers.vorticity,
                           &dynamic_buffers.rgb_buffer, cmd_enc);
  }
  else {
    dynamic_buffer_copy_to(&dynamic_buffers.dye, &dynamic_buffers.rgb_buffer,
                           cmd_enc);
  }

  /* Draw fluid */
  render_program_dispatch(wgpu_context, cmd_enc);

  /* Submit */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  /* Render imgui overlay */
  imgui_overlay_render(wgpu_context);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  dynamic_buffers_destroy();
  uniforms_buffers_destroy();
  programs_destroy();
  render_program_destroy();
  imgui_overlay_shutdown();
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  /* Track mouse position for simulation */
  if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    mouse_pixel_pos.x = input_event->mouse_x;
    mouse_pixel_pos.y = input_event->mouse_y;
  }
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title           = "Fluid Simulation",
    .no_depth_buffer = true,
    .init_cb         = init,
    .frame_cb        = frame,
    .input_event_cb  = input_event_cb,
    .shutdown_cb     = shutdown,
  });

  return EXIT_SUCCESS;
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
