#include "webgpu/wgpu_common.h"

#include "core/camera.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Gears
 *
 * WebGPU interpretation of glxgears. Procedurally generates and animates
 * multiple gears.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/gears
 *
 * Note:
 * WebGPU’s coordinate systems match DirectX and Metal’s coordinate systems in a
 * graphics pipeline. This means that WebGPU uses a left-handed coordinate
 * system (Ref: https://github.com/gpuweb/gpuweb/issues/416).
 * The coordinate system used by Vulkan is a right-handed system.
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* gears_vertex_shader_wgsl;
static const char* gears_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * WebGPU Gear class definition
 * -------------------------------------------------------------------------- */

typedef struct vertex_t {
  vec3 pos;
  vec3 normal;
  vec3 color;
} vertex_t;

static void initialize_vertex(vertex_t* v, vec3 p, vec3 n, vec3 c)
{
  v->pos[0]    = p[0];
  v->pos[1]    = p[1];
  v->pos[2]    = p[2];
  v->color[0]  = c[0];
  v->color[1]  = c[1];
  v->color[2]  = c[2];
  v->normal[0] = n[0];
  v->normal[1] = n[1];
  v->normal[2] = n[2];
}

typedef struct gear_info_t {
  float inner_radius;
  float outer_radius;
  float width;
  int num_teeth;
  float tooth_depth;
  vec3 color;
  vec3 pos;
  float rot_speed;
  float rot_offset;
} gear_info_t;

typedef struct gear_ubo_t {
  mat4 projection;
  mat4 model;
  mat4 normal;
  mat4 view;
  vec3 light_pos;
} gear_ubo_t;

typedef struct webgpu_gear_t {
  // Reference to the WebGPU context
  wgpu_context_t* wgpu_context;
  // WGPU bind group
  WGPUBindGroup bind_group;
  // Vertex buffer
  struct {
    wgpu_buffer_t buffer;
    vertex_t* data;
  } vbo;
  // Index buffer
  struct {
    wgpu_buffer_t buffer;
    uint32_t* data;
  } ibo;
  // Uniform buffer
  struct {
    wgpu_buffer_t buffer;
    gear_ubo_t data;
  } ubo;
  vec3 color;
  vec3 pos;
  float rot_speed;
  float rot_offset;
} webgpu_gear_t;

static void webgpu_gear_prepare_uniform_buffer(webgpu_gear_t* gear);

int32_t webgpu_gear_new_vertex(vertex_t* vertex_buffer, int32_t* vertex_counter,
                               float x, float y, float z, vec3 normal,
                               vec3 color)
{
  vertex_buffer[*vertex_counter] = (vertex_t){0};
  initialize_vertex(&vertex_buffer[*vertex_counter], (vec3){x, y, z}, normal,
                    color);
  ++(*vertex_counter);
  return (*vertex_counter) - 1;
}

void webgpu_gear_new_face(uint32_t* index_buffer, int32_t* index_counter, int a,
                          int b, int c)
{
  index_buffer[(*index_counter)++] = a;
  index_buffer[(*index_counter)++] = b;
  index_buffer[(*index_counter)++] = c;
}

static webgpu_gear_t* webgpu_gear_create(wgpu_context_t* wgpu_context)
{
  webgpu_gear_t* gear = (webgpu_gear_t*)malloc(sizeof(webgpu_gear_t));
  gear->wgpu_context  = wgpu_context;
  gear->vbo.data      = NULL;
  gear->ibo.data      = NULL;

  return gear;
}

static void webgpu_gear_destroy(webgpu_gear_t* gear)
{
  // Clean up WebGPU resources
  WGPU_RELEASE_RESOURCE(Buffer, gear->ubo.buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, gear->vbo.buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, gear->ibo.buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, gear->bind_group)
  if (gear->vbo.data) {
    free(gear->vbo.data);
  }
  if (gear->ibo.data) {
    free(gear->ibo.data);
  }
  free(gear);
}

static void webgpu_gear_generate(webgpu_gear_t* gear, gear_info_t* gearinfo)
{
  glm_vec3_copy(gearinfo->color, gear->color);
  glm_vec3_copy(gearinfo->pos, gear->pos);
  gear->rot_offset = gearinfo->rot_offset;
  gear->rot_speed  = gearinfo->rot_speed;

  /* Vertex buffer */
  gear->vbo.buffer.count = (6         /* front face */
                            + 4       /* front sides of teeth */
                            + 6       /* back face */
                            + 4       /* back sides of teeth */
                            + (4 * 5) /* draw outward faces of teeth */
                            )
                           * gearinfo->num_teeth;
  gear->vbo.buffer.size  = gear->vbo.buffer.count * sizeof(vertex_t);
  gear->vbo.data         = (vertex_t*)malloc(gear->vbo.buffer.size);
  vertex_t* vbd          = gear->vbo.data; // alias
  int32_t vertex_counter = 0;

  /* Index buffer */
  gear->ibo.buffer.count = (4         /* front face */
                            + 2       /* front sides of teeth */
                            + 4       /* back face */
                            + 2       /* back sides of teeth */
                            + (2 * 5) /* draw outward faces of teeth */
                            )
                           * 3 * gearinfo->num_teeth;
  gear->ibo.buffer.size = gear->ibo.buffer.count * sizeof(uint32_t);
  gear->ibo.data        = (uint32_t*)malloc(gear->ibo.buffer.size);
  uint32_t* ibd         = gear->ibo.data; // alias
  int32_t index_counter = 0;

  int i;
  float r0, r1, r2;
  float ta, da;
  float u1, v1, u2, v2, len;
  float cos_ta, cos_ta_1da, cos_ta_2da, cos_ta_3da, cos_ta_4da;
  float sin_ta, sin_ta_1da, sin_ta_2da, sin_ta_3da, sin_ta_4da;
  int32_t ix0, ix1, ix2, ix3, ix4, ix5;

  r0 = gearinfo->inner_radius;
  r1 = gearinfo->outer_radius - gearinfo->tooth_depth / 2.0f;
  r2 = gearinfo->outer_radius + gearinfo->tooth_depth / 2.0f;
  da = 2.0f * PI / gearinfo->num_teeth / 4.0f;

  vec3 normal = GLM_VEC3_ZERO_INIT;

  for (i = 0; i < gearinfo->num_teeth; ++i) {
    ta = i * 2.0f * PI / gearinfo->num_teeth;

    cos_ta     = cos(ta);
    cos_ta_1da = cos(ta + da);
    cos_ta_2da = cos(ta + 2.0f * da);
    cos_ta_3da = cos(ta + 3.0f * da);
    cos_ta_4da = cos(ta + 4.0f * da);
    sin_ta     = sin(ta);
    sin_ta_1da = sin(ta + da);
    sin_ta_2da = sin(ta + 2.0f * da);
    sin_ta_3da = sin(ta + 3.0f * da);
    sin_ta_4da = sin(ta + 4.0f * da);

    u1  = r2 * cos_ta_1da - r1 * cos_ta;
    v1  = r2 * sin_ta_1da - r1 * sin_ta;
    len = sqrt(u1 * u1 + v1 * v1);
    u1 /= len;
    v1 /= len;
    u2 = r1 * cos_ta_3da - r2 * cos_ta_2da;
    v2 = r1 * sin_ta_3da - r2 * sin_ta_2da;

    // front face
    glm_vec3_copy((vec3){0.0f, 0.0f, 1.0f}, normal);
    ix0 = webgpu_gear_new_vertex(vbd, &vertex_counter, r0 * cos_ta, r0 * sin_ta,
                                 gearinfo->width * 0.5f, normal, gear->color);
    ix1 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta, r1 * sin_ta,
                                 gearinfo->width * 0.5f, normal, gear->color);
    ix2 = webgpu_gear_new_vertex(vbd, &vertex_counter, r0 * cos_ta, r0 * sin_ta,
                                 gearinfo->width * 0.5f, normal, gear->color);
    ix3 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_3da,
                                 r1 * sin_ta_3da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix4 = webgpu_gear_new_vertex(vbd, &vertex_counter, r0 * cos_ta_4da,
                                 r0 * sin_ta_4da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix5 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_4da,
                                 r1 * sin_ta_4da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    webgpu_gear_new_face(ibd, &index_counter, ix0, ix1, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix1, ix3, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix2, ix3, ix4);
    webgpu_gear_new_face(ibd, &index_counter, ix3, ix5, ix4);

    // front sides of teeth
    glm_vec3_copy((vec3){0.0f, 0.0f, 1.0f}, normal);
    ix0 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta, r1 * sin_ta,
                                 gearinfo->width * 0.5f, normal, gear->color);
    ix1 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_1da,
                                 r2 * sin_ta_1da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix2 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_3da,
                                 r1 * sin_ta_3da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix3 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_2da,
                                 r2 * sin_ta_2da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    webgpu_gear_new_face(ibd, &index_counter, ix0, ix1, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix1, ix3, ix2);

    // back face
    glm_vec3_copy((vec3){0.0f, 0.0f, -1.0f}, normal);
    ix0 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta, r1 * sin_ta,
                                 -gearinfo->width * 0.5f, normal, gear->color);
    ix1 = webgpu_gear_new_vertex(vbd, &vertex_counter, r0 * cos_ta, r0 * sin_ta,
                                 -gearinfo->width * 0.5f, normal, gear->color);
    ix2 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_3da,
                                 r1 * sin_ta_3da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix3 = webgpu_gear_new_vertex(vbd, &vertex_counter, r0 * cos_ta, r0 * sin_ta,
                                 -gearinfo->width * 0.5f, normal, gear->color);
    ix4 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_4da,
                                 r1 * sin_ta_4da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix5 = webgpu_gear_new_vertex(vbd, &vertex_counter, r0 * cos_ta_4da,
                                 r0 * sin_ta_4da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    webgpu_gear_new_face(ibd, &index_counter, ix0, ix1, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix1, ix3, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix2, ix3, ix4);
    webgpu_gear_new_face(ibd, &index_counter, ix3, ix5, ix4);

    // back sides of teeth
    glm_vec3_copy((vec3){0.0f, 0.0f, -1.0f}, normal);
    ix0 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_3da,
                                 r1 * sin_ta_3da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix1 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_2da,
                                 r2 * sin_ta_2da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix2 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta, r1 * sin_ta,
                                 -gearinfo->width * 0.5f, normal, gear->color);
    ix3 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_1da,
                                 r2 * sin_ta_1da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    webgpu_gear_new_face(ibd, &index_counter, ix0, ix1, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix1, ix3, ix2);

    // draw outward faces of teeth
    glm_vec3_copy((vec3){v1, -u1, 0.0f}, normal);
    ix0 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta, r1 * sin_ta,
                                 gearinfo->width * 0.5f, normal, gear->color);
    ix1 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta, r1 * sin_ta,
                                 -gearinfo->width * 0.5f, normal, gear->color);
    ix2 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_1da,
                                 r2 * sin_ta_1da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix3 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_1da,
                                 r2 * sin_ta_1da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    webgpu_gear_new_face(ibd, &index_counter, ix0, ix1, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix1, ix3, ix2);

    glm_vec3_copy((vec3){cos_ta, sin_ta, 0.0f}, normal);
    ix0 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_1da,
                                 r2 * sin_ta_1da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix1 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_1da,
                                 r2 * sin_ta_1da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix2 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_2da,
                                 r2 * sin_ta_2da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix3 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_2da,
                                 r2 * sin_ta_2da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    webgpu_gear_new_face(ibd, &index_counter, ix0, ix1, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix1, ix3, ix2);

    glm_vec3_copy((vec3){v2, -u2, 0.0f}, normal);
    ix0 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_2da,
                                 r2 * sin_ta_2da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix1 = webgpu_gear_new_vertex(vbd, &vertex_counter, r2 * cos_ta_2da,
                                 r2 * sin_ta_2da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix2 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_3da,
                                 r1 * sin_ta_3da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix3 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_3da,
                                 r1 * sin_ta_3da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    webgpu_gear_new_face(ibd, &index_counter, ix0, ix1, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix1, ix3, ix2);

    glm_vec3_copy((vec3){cos_ta, sin_ta, 0.0f}, normal);
    ix0 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_3da,
                                 r1 * sin_ta_3da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix1 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_3da,
                                 r1 * sin_ta_3da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix2 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_4da,
                                 r1 * sin_ta_4da, gearinfo->width * 0.5f,
                                 normal, gear->color);
    ix3 = webgpu_gear_new_vertex(vbd, &vertex_counter, r1 * cos_ta_4da,
                                 r1 * sin_ta_4da, -gearinfo->width * 0.5f,
                                 normal, gear->color);
    webgpu_gear_new_face(ibd, &index_counter, ix0, ix1, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix1, ix3, ix2);

    // draw inside radius cylinder
    ix0 = webgpu_gear_new_vertex(gear->vbo.data, &vertex_counter, r0 * cos_ta,
                                 r0 * sin_ta, -gearinfo->width * 0.5f,
                                 (vec3){-cos_ta, -sin_ta, 0.0f}, gear->color);
    ix1 = webgpu_gear_new_vertex(gear->vbo.data, &vertex_counter, r0 * cos_ta,
                                 r0 * sin_ta, gearinfo->width * 0.5f,
                                 (vec3){-cos_ta, -sin_ta, 0.0f}, gear->color);
    ix2 = webgpu_gear_new_vertex(
      gear->vbo.data, &vertex_counter, r0 * cos_ta_4da, r0 * sin_ta_4da,
      -gearinfo->width * 0.5f, (vec3){-cos_ta_4da, -sin_ta_4da, 0.0f},
      gear->color);
    ix3 = webgpu_gear_new_vertex(
      gear->vbo.data, &vertex_counter, r0 * cos_ta_4da, r0 * sin_ta_4da,
      gearinfo->width * 0.5f, (vec3){-cos_ta_4da, -sin_ta_4da, 0.0f},
      gear->color);
    webgpu_gear_new_face(ibd, &index_counter, ix0, ix1, ix2);
    webgpu_gear_new_face(ibd, &index_counter, ix1, ix3, ix2);
  }

  // Vertex buffer
  gear->vbo.buffer = wgpu_create_buffer(
    gear->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Gear vertex buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
      .size         = gear->vbo.buffer.size,
      .count        = gear->vbo.buffer.count,
      .initial.data = gear->vbo.data,
    });

  // Index buffer
  gear->ibo.buffer
    = wgpu_create_buffer(gear->wgpu_context, &(wgpu_buffer_desc_t){
                                               .label = "Gear index buffer",
                                               .usage = WGPUBufferUsage_CopyDst
                                                        | WGPUBufferUsage_Index,
                                               .size  = gear->ibo.buffer.size,
                                               .count = gear->ibo.buffer.count,
                                               .initial.data = gear->ibo.data,
                                             });

  // Uniform buffer
  webgpu_gear_prepare_uniform_buffer(gear);
}

static void webgpu_gear_draw(webgpu_gear_t* gear, WGPURenderPassEncoder rpass)
{
  wgpuRenderPassEncoderSetBindGroup(rpass, 0, gear->bind_group, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(rpass, 0, gear->vbo.buffer.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rpass, gear->ibo.buffer.buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rpass, gear->ibo.buffer.count, 1, 0, 0, 1);
}

static void webgpu_gear_update_uniform_buffer(webgpu_gear_t* gear,
                                              mat4 perspective, mat4 view,
                                              float timer)
{
  gear_ubo_t* ubo = &gear->ubo.data;
  glm_mat4_copy(perspective, ubo->projection);
  glm_mat4_copy(view, ubo->view);
  glm_mat4_copy(GLM_MAT4_IDENTITY, ubo->model);
  glm_translate(ubo->model, gear->pos);
  glm_rotate(ubo->model, glm_rad((gear->rot_speed * timer) + gear->rot_offset),
             (vec3){0.0f, 0.0f, 1.0f});
  mat4 mul_mat = GLM_MAT4_ZERO_INIT;
  glm_mat4_mul(ubo->view, ubo->model, mul_mat);
  glm_mat4_inv(mul_mat, mul_mat);
  glm_mat4_transpose_to(mul_mat, ubo->normal);
  glm_vec3_copy((vec3){0.0f, 0.0f, 2.5f}, ubo->light_pos);
  ubo->light_pos[0] = sin(glm_rad(timer)) * 8.0f;
  ubo->light_pos[2] = cos(glm_rad(timer)) * 8.0f;

  wgpuQueueWriteBuffer(gear->wgpu_context->queue, gear->ubo.buffer.buffer, 0,
                       ubo, gear->ubo.buffer.size);
}

static void webgpu_gear_setup_bind_group(webgpu_gear_t* gear,
                                         WGPUBindGroupLayout bind_group_layout)
{
  WGPUBindGroupDescriptor bg_desc = {
    .layout     = bind_group_layout,
    .entryCount = 1,
    .entries    = &(WGPUBindGroupEntry) {
      // Binding 0 : Vertex shader uniform buffer
      .binding = 0,
      .buffer  = gear->ubo.buffer.buffer,
      .offset  = 0,
      .size    = gear->ubo.buffer.size,
    },
  };

  gear->bind_group
    = wgpuDeviceCreateBindGroup(gear->wgpu_context->device, &bg_desc);
  ASSERT(gear->bind_group != NULL);
}

static void webgpu_gear_prepare_uniform_buffer(webgpu_gear_t* gear)
{
  memset(&gear->ubo.data, 0, sizeof(gear_ubo_t));
  gear->ubo.buffer = wgpu_create_buffer(
    gear->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(gear_ubo_t),
    });
}

/* -------------------------------------------------------------------------- *
 * WebGPU Gears properties definition
 * -------------------------------------------------------------------------- */

typedef struct webgpu_gear_definition_t {
  float inner_radius;
  float outer_radius;
  float width;
  int32_t tooth_count;
  float tooth_depth;
  vec3 color;
  vec3 position;
  float rotation_speed;
  float rotation_offset;
} webgpu_gear_definition_t;

// Gear definitions
static webgpu_gear_definition_t gear_defs[3] = {
  [0] = { /* webgpu_gear_definition_t */
    .inner_radius    = 1.0f,
    .outer_radius    = 4.0f,
    .width           = 1.0f,
    .tooth_count     = 20,
    .tooth_depth     = 0.7f,
    .color           = {1.0f, 0.0f, 0.0f},
    .position        = {-3.0f, 0.0f, 0.0f},
    .rotation_speed  = 1.0f,
    .rotation_offset = 0.0f,
  },
  [1] = { /* webgpu_gear_definition_t */
    .inner_radius    = 0.5f,
    .outer_radius    = 2.0f,
    .width           = 2.0f,
    .tooth_count     = 10,
    .tooth_depth     = 0.7f,
    .color           = {0.0f, 1.0f, 0.2f},
    .position        = {3.1f, 0.0f, 0.0f},
    .rotation_speed  = -2.0f,
    .rotation_offset = -9.0f,
  },
  [2] = { /* webgpu_gear_definition_t */
    .inner_radius    = 1.3f,
    .outer_radius    = 2.0f,
    .width           = 0.5f,
    .tooth_count     = 10,
    .tooth_depth     = 0.7f,
    .color           = {0.0f, 0.0f, 1.0f},
    .position        = {-3.1f, -6.2f, 0.0f},
    .rotation_speed  = -2.0f,
    .rotation_offset = -30.0f,
  },
};

/* -------------------------------------------------------------------------- *
 * WebGPU Gears example
 * -------------------------------------------------------------------------- */

/* State struct */
static struct {
  webgpu_gear_t* gears[3];
  camera_t camera;
  WGPUBool view_updated;
  float timer;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
  WGPUBindGroupLayout bind_group_layout;
  /* Render pass descriptor for frame buffer writes */
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } render_pass;
  WGPUBool initialized;
} state = {
  .render_pass = {
    .color_attachment = {
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0, 0.0, 0.0, 1.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    },
    .depth_stencil_attachment = {
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    },
    .descriptor = {
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.render_pass.color_attachment,
      .depthStencilAttachment = &state.render_pass.depth_stencil_attachment,
    },
  }
};

static const uint32_t gears_count = 3;

static void init_vertices(wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < gears_count; ++i) {
    const float* gear_color    = gear_defs[i].color;
    const float* gear_position = gear_defs[i].position;
    gear_info_t gear_info      = {
           .inner_radius = gear_defs[i].inner_radius,
           .outer_radius = gear_defs[i].outer_radius,
           .width        = gear_defs[i].width,
           .num_teeth    = gear_defs[i].tooth_count,
           .tooth_depth  = gear_defs[i].tooth_depth,
           .color        = {gear_color[0], gear_color[1], gear_color[2]},
           .pos          = {gear_position[0], gear_position[1], gear_position[2]},
           .rot_speed    = gear_defs[i].rotation_speed,
           .rot_offset   = gear_defs[i].rotation_offset,
    };
    state.gears[i] = webgpu_gear_create(wgpu_context);
    webgpu_gear_generate(state.gears[i], &gear_info);
  }
}

/* Initialize camera */
static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type   = CameraType_LookAt;
  state.camera.flip_y = true;
  camera_set_position(&state.camera, (vec3){0.0f, 2.5f, -16.0f});
  camera_set_rotation(&state.camera, (vec3){23.75f, 41.25f, 21.0f});
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.001f, 256.0f);
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .entryCount = 1,
    .entries = &(WGPUBindGroupLayoutEntry) {
      /* Binding 0: Vertex shader uniform buffer */
      .binding   = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(gear_ubo_t),
      },
      .sampler = {0},
    },
  };
  state.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.bind_group_layout != NULL);

  /* Create the pipeline layout */
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &state.bind_group_layout,
  };
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                         &pipeline_layout_desc);
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_groups(void)
{
  for (uint32_t i = 0; i < gears_count; ++i) {
    webgpu_gear_setup_bind_group(state.gears[i], state.bind_group_layout);
  }
}

/* Create the graphics pipeline */
static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, gears_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, gears_fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    gear, sizeof(vertex_t),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    /* Attribute location 1: Normal */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, normal)),
    /* Attribute location 2: Color */
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, color)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Gears - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &gear_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_Back,
      .frontFace = WGPUFrontFace_CW
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static void init_depth_stencil(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  /* Depth stencil is created automatically by wgpu_context */
}

static void update_uniform_buffers(void)
{
  for (uint32_t i = 0; i < gears_count; ++i) {
    webgpu_gear_update_uniform_buffer(
      state.gears[i], state.camera.matrices.perspective,
      state.camera.matrices.view, state.timer * 360.0f);
  }
}

/* Initialize the gears example */
static int init(wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    init_camera(wgpu_context);
    init_vertices(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_pipelines(wgpu_context);
    init_bind_groups();
    init_depth_stencil(wgpu_context);
    update_uniform_buffers();
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

/* Update matrices and uniform buffers */
static void update_timer(void)
{
  static uint64_t start_time   = 0;
  static uint64_t current_time = 0;

  if (start_time == 0) {
    stm_setup();
    start_time = stm_now();
  }

  current_time = stm_now();
  state.timer  = (float)stm_ms(current_time - start_time) / 1000.0f * 0.25f;
}

/* Render frame */
static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update timer */
  update_timer();

  /* Update uniform buffers */
  update_uniform_buffers();

  /* Update render pass attachments */
  state.render_pass.color_attachment.view = wgpu_context->swapchain_view;
  state.render_pass.depth_stencil_attachment.view
    = wgpu_context->depth_stencil_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder */
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass.descriptor);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);

  /* Draw gears */
  for (uint32_t i = 0; i < gears_count; ++i) {
    webgpu_gear_draw(state.gears[i], rpass_enc);
  }

  /* End render pass */
  wgpuRenderPassEncoderEnd(rpass_enc);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit command buffer */
  ASSERT(command_buffer != NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(command_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

/* Handle input events */
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  UNUSED_VAR(wgpu_context);

  camera_on_input_event(&state.camera, input_event);
  state.view_updated = true;
}

/* Clean up resources */
static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout);

  for (uint32_t i = 0; i < gears_count; ++i) {
    webgpu_gear_destroy(state.gears[i]);
  }
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Gears",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* gears_vertex_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4<f32>,
    model : mat4x4<f32>,
    normal : mat4x4<f32>,
    view : mat4x4<f32>,
    lightpos : vec3<f32>,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) normal : vec3<f32>,
    @location(1) color : vec3<f32>,
    @location(2) eyePos : vec3<f32>,
    @location(3) lightVec : vec3<f32>,
  };

  @vertex
  fn main(
    @location(0) inPos: vec4<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) inColor: vec3<f32>
  ) -> Output {
    var output: Output;
    output.normal = normalize(mat3x3(
        ubo.normal[0].xyz,
        ubo.normal[1].xyz,
        ubo.normal[2].xyz
      ) * inNormal);
    output.color = inColor;
    let modelView : mat4x4<f32> = ubo.view * ubo.model;
    let pos : vec4<f32> = modelView * inPos;
    output.eyePos = (modelView * pos).xyz;
    let lightPos : vec4<f32> = vec4<f32>(ubo.lightpos, 1.0) * modelView;
    output.lightVec = normalize(lightPos.xyz - output.eyePos);
    output.position = ubo.projection * pos;
    return output;
  }
);

static const char* gears_fragment_shader_wgsl = CODE(
  @fragment
  fn main(
    @location(0) inNormal: vec3<f32>,
    @location(1) inColor: vec3<f32>,
    @location(2) inEyePos: vec3<f32>,
    @location(3) inLightVec: vec3<f32>
  ) -> @location(0) vec4<f32> {
    let Eye : vec3<f32> = normalize(-inEyePos);
    let Reflected : vec3<f32> = normalize(reflect(-inLightVec, inNormal));

    let IAmbient : vec4<f32> = vec4<f32>(0.2, 0.2, 0.2, 1.0);
    let IDiffuse : vec4<f32> = vec4<f32>(0.5, 0.5, 0.5, 0.5) * max(dot(inNormal, inLightVec), 0.0);
    let specular : f32 = 0.25;
    let ISpecular : vec4<f32> = vec4<f32>(0.5, 0.5, 0.5, 1.0) * pow(max(dot(Reflected, Eye), 0.0), 0.8) * specular;

    return vec4<f32>((IAmbient + IDiffuse) * vec4(inColor, 1.0) + ISpecular);
  }
);
// clang-format on
