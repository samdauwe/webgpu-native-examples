#include "example_base.h"

#include <math.h>
#include <string.h>

#include <cglm/cglm.h>

#include "../core/macro.h"
#include "../webgpu/imgui_overlay.h"

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

static void webgpu_gear_draw(webgpu_gear_t* gear)
{
  WGPURenderPassEncoder rpass = gear->wgpu_context->rpass_enc;

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

  wgpu_queue_write_buffer(gear->wgpu_context, gear->ubo.buffer.buffer, 0, ubo,
                          gear->ubo.buffer.size);
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
static webgpu_gear_t* wgpu_gears[3];
static const uint32_t wgpu_gears_count = (uint32_t)ARRAY_SIZE(wgpu_gears);

static void prepare_vertices(wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < wgpu_gears_count; ++i) {
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
    wgpu_gears[i] = webgpu_gear_create(wgpu_context);
    webgpu_gear_generate(wgpu_gears[i], &gear_info);
  }
}

/* -------------------------------------------------------------------------- *
 * WebGPU Gears example
 * -------------------------------------------------------------------------- */

// The pipeline layout
static WGPUPipelineLayout pipeline_layout;

// Pipeline
static WGPURenderPipeline pipeline;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

// The bind group layout
static WGPUBindGroupLayout bind_group_layout;

// Other variables
static const char* example_title = "Gears";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera         = camera_create();
  context->camera->type   = CameraType_LookAt;
  context->camera->flip_y = true;
  camera_set_position(context->camera, (vec3){0.0f, 2.5f, -16.0f});
  camera_set_rotation(context->camera, (vec3){23.75f, 41.25f, 21.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.001f, 256.0f);
  context->timer_speed *= 0.25f;
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .entryCount = 1,
    .entries = &(WGPUBindGroupLayoutEntry) {
      // Binding 0: Vertex shader uniform buffer
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
  bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this descriptor set layout
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &bind_group_layout,
  };
  pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                   &pipeline_layout_desc);
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_groups(void)
{
  for (uint32_t i = 0; i < wgpu_gears_count; ++i) {
    webgpu_gear_setup_bind_group(wgpu_gears[i], bind_group_layout);
  }
}

// Create the graphics pipeline
static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    gear, sizeof(vertex_t),
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    // Attribute location 1: Normal
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, normal)),
    // Attribute location 2: Color
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, color)))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label            = "Gears vertex shader",
                  .wgsl_code.source = gears_vertex_shader_wgsl,
                  .entry            = "main",
                },
                .buffer_count = 1,
                .buffers      = &gear_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .label            = "Gears fragment shader",
                  .wgsl_code.source = gears_fragment_shader_wgsl,
                  .entry            = "main",
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
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "solid_render_pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, // Assigned later
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

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  for (uint32_t i = 0; i < wgpu_gears_count; ++i) {
    webgpu_gear_update_uniform_buffer(
      wgpu_gears[i], context->camera->matrices.perspective,
      context->camera->matrices.view, context->timer * 360.0f);
  }
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    prepare_vertices(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_groups();
    update_uniform_buffers(context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  // Draw gears
  for (uint32_t i = 0; i < wgpu_gears_count; ++i) {
    webgpu_gear_draw(wgpu_gears[i]);
  }

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
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
    = build_command_buffer(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  // update the uniform buffer
  update_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout);

  for (uint32_t i = 0; i < wgpu_gears_count; ++i) {
    webgpu_gear_destroy(wgpu_gears[i]);
  }
}

void example_gears(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
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
