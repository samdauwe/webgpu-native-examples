#include "example_base.h"

#include <cJSON.h>
#include <string.h>

#include "../core/log.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Normal Mapping example
 *
 * This example demonstrates how to achieve normal mapping in WebGPU. A normal
 * map uses RGB information that corresponds directly with the X, Y and Z axis
 * in 3D space. This RGB information tells the 3D application the exact
 * direction of the surface normals are oriented in for each and every polygon.
 *
 * Ref:
 * https://github.com/Konstantin84UKR/webgpu_examples/tree/master/normalMap
 *
 * Note:
 * http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Vertex data - Torus Knot Mesh / Plane Mesh
 * -------------------------------------------------------------------------- */

typedef enum mesh_type_t {
  MESH_TYPE_TORUS_KNOT = 0,
  MESH_TYPE_PLANE      = 1,
  MESH_TYPE_COUNT      = 2,
} mesh_type_t;

#define TORUS_KNOT_VERTEX_COUNT 7893
#define TORUS_KNOT_FACES_COUNT 3000
#define TORUS_KNOT_INDEX_COUNT (TORUS_KNOT_FACES_COUNT * 3)
#define TORUS_KNOT_UV_COUNT 5262
#define TORUS_KNOT_NORMAL_COUNT 7893
#define TORUS_KNOT_TANGENTS_COUNT 7893
#define TORUS_KNOT_BITANGENTS_COUNT 7893

#define PLANE_VERTEX_COUNT 75
#define PLANE_FACES_COUNT 32
#define PLANE_INDEX_COUNT (TORUS_KNOT_FACES_COUNT * 3)
#define PLANE_UV_COUNT 50
#define PLANE_NORMAL_COUNT 75
#define PLANE_TANGENTS_COUNT 75
#define PLANE_BITANGENTS_COUNT 75

static struct torus_knot_mesh {
  float vertices[TORUS_KNOT_VERTEX_COUNT];
  uint32_t indices[TORUS_KNOT_INDEX_COUNT];
  float uvs[TORUS_KNOT_UV_COUNT];
  float normals[TORUS_KNOT_NORMAL_COUNT];
  float tangents[TORUS_KNOT_TANGENTS_COUNT];
  float bitangents[TORUS_KNOT_BITANGENTS_COUNT];
} torus_knot_mesh = {0};

static struct plane_mesh {
  float vertices[PLANE_VERTEX_COUNT];
  uint32_t indices[PLANE_INDEX_COUNT];
  float uvs[PLANE_UV_COUNT];
  float normals[PLANE_NORMAL_COUNT];
  float tangents[PLANE_TANGENTS_COUNT];
  float bitangents[PLANE_BITANGENTS_COUNT];
} plane_mesh = {0};

int32_t prepare_meshes(void)
{
  int32_t res = EXIT_FAILURE;

  file_read_result_t file_read_result = {0};
  read_file("meshes/model.json", &file_read_result, true);
  const char* const json_data = (const char* const)file_read_result.data;

  cJSON* model_json = cJSON_Parse(json_data);
  if (model_json == NULL) {
    const char* error_ptr = cJSON_GetErrorPtr();
    if (error_ptr != NULL) {
      log_error("Error before: %s", error_ptr);
    }
    goto load_json_end;
  }

  if (!cJSON_IsObject(model_json)
      || !cJSON_HasObjectItem(model_json, "meshes")) {
    log_error("Invalid mesh file, does not contain 'meshes' array");
    goto load_json_end;
  }

  /* Get meshes */
  for (int32_t mesh_id = 0; mesh_id < (int32_t)MESH_TYPE_COUNT; ++mesh_id) {
    const cJSON* meshes_array
      = cJSON_GetObjectItemCaseSensitive(model_json, "meshes");
    if (!cJSON_IsArray(meshes_array)) {
      log_error("'meshes' object item is not an array");
      goto load_json_end;
    }
    if (!(cJSON_GetArraySize(meshes_array) > mesh_id)) {
      log_error(
        "'meshes' array does not contain any mesh object for mesh type %d",
        mesh_id);
      goto load_json_end;
    }
    const cJSON* meshes_item = cJSON_GetArrayItem(meshes_array, mesh_id);

    if (!cJSON_IsObject(meshes_item)
        || !cJSON_HasObjectItem(meshes_item, "vertices")
        || !cJSON_HasObjectItem(meshes_item, "faces")
        || !cJSON_HasObjectItem(meshes_item, "texturecoords")
        || !cJSON_HasObjectItem(meshes_item, "normals")
        || !cJSON_HasObjectItem(meshes_item, "tangents")
        || !cJSON_HasObjectItem(meshes_item, "bitangents")) {
      log_error(
        "Invalid mesh object, does not contain 'vertices', 'faces', "
        "'texturecoords', 'normals', 'tangents', 'bitangents' array");
      goto load_json_end;
    }

    /* Parse vertices */
    {
      const cJSON* vertex_array = NULL;
      const cJSON* vertex_item  = NULL;

      vertex_array = cJSON_GetObjectItemCaseSensitive(meshes_item, "vertices");
      if (!cJSON_IsArray(vertex_array)) {
        log_error("vertices object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_VERTEX_COUNT :
                           PLANE_VERTEX_COUNT;
      ASSERT(cJSON_GetArraySize(vertex_array) == expectedSize);

      float* mesh_vertices = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                               torus_knot_mesh.vertices :
                               plane_mesh.vertices;
      uint32_t c           = 0;
      cJSON_ArrayForEach(vertex_item, vertex_array)
      {
        mesh_vertices[c++] = (float)vertex_item->valuedouble;
      }
    }

    /* Parse indices */
    {
      const cJSON* faces_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "faces");
      if (!cJSON_IsArray(faces_array)) {
        log_error("'faces' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_FACES_COUNT :
                           PLANE_FACES_COUNT;
      ASSERT(cJSON_GetArraySize(faces_array) == expectedSize);

      uint32_t* mesh_indices = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                                 torus_knot_mesh.indices :
                                 plane_mesh.indices;
      const cJSON* face_item = NULL;
      uint32_t c             = 0;
      cJSON_ArrayForEach(face_item, faces_array)
      {
        if (!(cJSON_GetArraySize(face_item) == 3)) {
          log_error("'face' item is not an array of size 3");
          goto load_json_end;
        }
        for (uint32_t i = 0; i < 3; ++i) {
          mesh_indices[c++]
            = (uint32_t)cJSON_GetArrayItem(face_item, i)->valueint;
        }
      }
    }

    /* Parse uvs */
    {
      const cJSON* texturecoord_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "texturecoords");
      if (!(cJSON_GetArraySize(texturecoord_array) > 0)) {
        log_error("'texturecoords' array does not contain any object");
        goto load_json_end;
      }
      const cJSON* texturecoords_item
        = cJSON_GetArrayItem(texturecoord_array, 0);
      if (!cJSON_IsArray(texturecoords_item)) {
        log_error("'texturecoords' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_UV_COUNT :
                           PLANE_UV_COUNT;
      ASSERT(cJSON_GetArraySize(texturecoords_item) == expectedSize);

      float* mesh_uvs                = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                                         torus_knot_mesh.uvs :
                                         plane_mesh.uvs;
      const cJSON* texturecoord_item = NULL;
      uint32_t c                     = 0;
      cJSON_ArrayForEach(texturecoord_item, texturecoords_item)
      {
        mesh_uvs[c++] = (float)texturecoord_item->valuedouble;
      }
    }

    /* Parse normals */
    {
      const cJSON* normal_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "normals");
      if (!cJSON_IsArray(normal_array)) {
        log_error("'normals' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_NORMAL_COUNT :
                           PLANE_NORMAL_COUNT;
      ASSERT(cJSON_GetArraySize(normal_array) == expectedSize);

      float* mesh_normals      = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                                   torus_knot_mesh.normals :
                                   plane_mesh.normals;
      const cJSON* normal_item = NULL;
      uint32_t c               = 0;
      cJSON_ArrayForEach(normal_item, normal_array)
      {
        mesh_normals[c++] = (float)normal_item->valuedouble;
      }
    }

    /* Parse tangents */
    {
      const cJSON* tangent_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "tangents");
      if (!cJSON_IsArray(tangent_array)) {
        log_error("'tangents' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_TANGENTS_COUNT :
                           PLANE_TANGENTS_COUNT;
      ASSERT(cJSON_GetArraySize(tangent_array) == expectedSize);

      float* mesh_tangents      = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                                    torus_knot_mesh.tangents :
                                    plane_mesh.tangents;
      const cJSON* tangent_item = NULL;
      uint32_t c                = 0;
      cJSON_ArrayForEach(tangent_item, tangent_array)
      {
        mesh_tangents[c++] = (float)tangent_item->valuedouble;
      }
    }

    /* Parse bitangents */
    {
      const cJSON* bitangent_array
        = cJSON_GetObjectItemCaseSensitive(meshes_item, "bitangents");
      if (!cJSON_IsArray(bitangent_array)) {
        log_error("'bitangents' object item is not an array");
        goto load_json_end;
      }

      int expectedSize = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                           TORUS_KNOT_BITANGENTS_COUNT :
                           PLANE_BITANGENTS_COUNT;
      ASSERT(cJSON_GetArraySize(bitangent_array) == expectedSize);

      float* mesh_bitangents      = (mesh_id == MESH_TYPE_TORUS_KNOT) ?
                                      torus_knot_mesh.bitangents :
                                      plane_mesh.bitangents;
      const cJSON* bitangent_item = NULL;
      uint32_t c                  = 0;
      cJSON_ArrayForEach(bitangent_item, bitangent_array)
      {
        mesh_bitangents[c++] = (float)bitangent_item->valuedouble;
      }
    }
  }

  res = EXIT_SUCCESS;

load_json_end:
  cJSON_Delete(model_json);
  free(file_read_result.data);

  return res;
}

/* -------------------------------------------------------------------------- *
 * Camera
 * -------------------------------------------------------------------------- */

typedef struct _camera_t {
  wgpu_context_t* wgpu_context;
  float speed_camera;
  float fovy;
  vec3 eye;
  vec3 front;
  vec3 up_world;
  vec3 right;
  vec3 up;
  vec3 look;
  mat4 projection_matrix;
  mat4 view_matrix;
  mat4 world_matrix;
  float yaw;
  float pitch;
  float delta_time;
} _camera_t;

void _camera_update(_camera_t* this);

static void _camera_init_defaults(_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  this->speed_camera = 0.01f;
  this->fovy         = 40.0f * PI / 180.0f;
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, this->up_world);
  glm_mat4_identity(this->world_matrix);

  this->yaw        = 90.0f * PI / 180.0f;
  this->pitch      = 0.0f;
  this->delta_time = 1.0f;
}

void _camera_init(_camera_t* this, wgpu_context_t* wgpu_context, vec3 eye,
                  vec3 front)
{
  _camera_init_defaults(this);

  this->wgpu_context = wgpu_context;

  glm_vec3_copy(eye, this->eye);
  glm_vec3_copy(front, this->front);
  glm_vec3_cross(this->front, this->up_world, this->right);
  glm_vec3_cross(this->right, this->front, this->up);
  glm_vec3_add(this->eye, this->front, this->look);

  _camera_update(this);
}

void _camera_update(_camera_t* this)
{
  /* View projection matrix */
  const float aspect_ratio = (float)this->wgpu_context->surface.width
                             / (float)this->wgpu_context->surface.height;
  glm_perspective(this->fovy, aspect_ratio, 0.1f, 500.0f,
                  this->projection_matrix);

  /* View matrix */
  glm_lookat(this->eye,        /* eye vector    */
             this->look,       /* center vector */
             this->up,         /* up vector     */
             this->view_matrix /* result matrix */
  );
}

void _camera_update_camera_vectors(_camera_t* this)
{
  glm_vec3_add(this->eye, this->front, this->look);
  glm_vec3_cross(this->front, this->up_world, this->right);
  glm_vec3_cross(this->right, this->front, this->up);

  vec3 fz = GLM_VEC3_ZERO_INIT, rx = GLM_VEC3_ZERO_INIT,
       uy = GLM_VEC3_ZERO_INIT;
  glm_vec3_normalize_to(this->front, fz);
  glm_vec3_normalize_to(this->right, rx);
  glm_vec3_normalize_to(this->up, uy);

  glm_mat4_copy(
    (mat4){
      {rx[0], rx[1], rx[2], this->eye[0]}, /* mat  */
      {uy[0], uy[1], uy[2], this->eye[1]}, /*      */
      {fz[0], fz[1], fz[2], this->eye[2]}, /*      */
      {0.0f, 0, 0.0f, 1.0f},               /* dest */
    },
    this->world_matrix);
}

void _camera_set_position(_camera_t* this, vec3 position)
{
  glm_vec3_copy(position, this->eye);
  _camera_update(this);
}

void _camera_set_look(_camera_t* this, vec3 position)
{
  glm_vec3_copy(position, this->front);
  _camera_update_camera_vectors(this);
  _camera_update(this);
}

void _camera_set_delta_time(_camera_t* this, float delta_time)
{
  this->delta_time = delta_time;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */
static const char* normal_map_vertex_shader_wgsl;
static const char* normal_map_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Normal Mapping example
 * -------------------------------------------------------------------------- */

/* Buffers */
static struct {
  struct {
    wgpu_buffer_t vertex;
    wgpu_buffer_t index;
    wgpu_buffer_t uv;
    wgpu_buffer_t normal;
    wgpu_buffer_t tangent;
    wgpu_buffer_t bitangent;
  } torus_knot;
  struct {
    wgpu_buffer_t vertex;
    wgpu_buffer_t index;
    wgpu_buffer_t uv;
    wgpu_buffer_t normal;
    wgpu_buffer_t tangent;
    wgpu_buffer_t bitangent;
  } plane;
  wgpu_buffer_t normal_map_vs_uniform_buffer;
  wgpu_buffer_t normal_map_fs_uniform_buffer;
  wgpu_buffer_t uniform_buffer_light;
} buffers = {0};

/* Textures and samplers */
static struct {
  texture_t diffuse;
  texture_t normal;
  texture_t specular;
  texture_t depth;
} textures = {0};

static struct {
  WGPUSampler normal_map;
} samplers = {0};

/* Uniform bind groups and render pipelines (and layout) */
static struct {
  WGPUBindGroupLayout normal_map;
} bind_group_layouts;

static struct {
  WGPUPipelineLayout normal_map;
} pipeline_layouts;

static struct {
  WGPUBindGroup normal_map;
} bind_groups = {0};

static struct {
  WGPURenderPipeline normal_map;
} pipelines = {0};

/* Render pass descriptor for frame buffer writes */
static struct {
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } normal_map;
} render_pass = {0};

/* Uniform data */
static struct {
  mat4 projection_matrix;
  mat4 view_matrix;
  mat4 model_matrix;
} view_matrices = {
  .projection_matrix = GLM_MAT4_IDENTITY_INIT,
  .view_matrix       = GLM_MAT4_IDENTITY_INIT,
  .model_matrix      = GLM_MAT4_IDENTITY_INIT,
};

static struct {
  vec4 eye_position;
  vec4 light_position;
} light_positions = {
  .eye_position   = {3.0f, 10.0f, 2.0f, 1.0f},
  .light_position = {5.0f, 5.0f, 5.0f, 1.0f},
};

static struct {
  mat4 projection_matrix;
  mat4 view_matrix;
  mat4 model_matrix;
} shadow_view_matrices = {
  .projection_matrix = GLM_MAT4_IDENTITY_INIT,
  .view_matrix       = GLM_MAT4_IDENTITY_INIT,
  .model_matrix      = GLM_MAT4_IDENTITY_INIT,
};

/* Camera animation */
static float time_old   = 0;
static _camera_t camera = {0};

/* Other variables */
static const char* example_title = "Normal Mapping example";
static bool prepared             = false;

static void initialize_camera(wgpu_context_t* wgpu_context)
{
  _camera_init(&camera, wgpu_context, (vec3){0.0, 0.0, 10.0} /* eye */,
               (vec3){0.0f, 0.0f, -1.0f} /* front */);
}

/**
 * @brief Generates a orthogonal projection matrix with the given bounds.
 * @ref https://glmatrix.net/docs/mat4.js.html
 *
 * @param {mat4} out mat4 frustum matrix will be written into
 * @param {number} left Left bound of the frustum
 * @param {number} right Right bound of the frustum
 * @param {number} bottom Bottom bound of the frustum
 * @param {number} top Top bound of the frustum
 * @param {number} near Near bound of the frustum
 * @param {number} far Far bound of the frustum
 * @returns {mat4} out
 */
static void glm_mat4_ortho(float left, float right, float bottom, float top,
                           float nearZ, float farZ, mat4 dest)
{
  float lr, bt, nf;

  lr = 1.0f / (left - right);
  bt = 1.0f / (bottom - top);
  nf = 1.0f / (nearZ - farZ);

  glm_mat4_zero(dest);

  dest[0][0] = -2.0f * lr;
  dest[1][1] = -2.0f * bt;
  dest[2][2] = 2.0f * nf;
  dest[3][0] = (left + right) * lr;
  dest[3][1] = (top + bottom) * bt;
  dest[3][2] = (farZ + nearZ) * nf;
  dest[3][3] = 1.0f;
}

static void prepare_uniform_data(wgpu_context_t* wgpu_context)
{
  /* View matrix */
  glm_lookat(light_positions.eye_position, /* eye vector    */
             (vec3){0.0f, 0.0f, 0.0f},     /* center vector */
             (vec3){0.0f, 1.0f, 0.0f},     /* up vector     */
             view_matrices.view_matrix     /* result matrix */
  );

  /* View projection matrix */
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;
  const float fovy = 40.0f * PI / 180.0f;
  glm_perspective(fovy, aspect_ratio, 1.f, 25.0f,
                  view_matrices.projection_matrix);

  /* Set camera position */
  _camera_set_position(&camera, (vec3){0.0f, 5.0f, 10.0f});
  _camera_set_look(&camera, (vec3){0.0, -0.5, -1.0});
  glm_vec3_copy(camera.eye, light_positions.eye_position);

  /* Shadow view matrix */
  glm_lookat(light_positions.light_position,  /* eye vector    */
             (vec3){0.0f, 0.0f, 0.0f},        /* center vector */
             (vec3){0.0f, 1.0f, 0.0f},        /* up vector     */
             shadow_view_matrices.view_matrix /* result matrix */
  );

  /* Shadow view projection matrix */
  glm_mat4_ortho(-6.0f, 6.0f, -6.0f, 6.0f, 1.0f, 35.0f,
                 shadow_view_matrices.projection_matrix);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  /* Time */
  const float now = context->frame.timestamp_millis;
  float dt        = now - time_old;
  time_old        = now;

  _camera_set_delta_time(&camera, dt);

  /* Rotate model matrix */
  glm_rotate_y(view_matrices.model_matrix, dt * 0.0002f,
               view_matrices.model_matrix);
  glm_mat4_copy(camera.projection_matrix, view_matrices.projection_matrix);
  glm_mat4_copy(camera.view_matrix, view_matrices.view_matrix);
  glm_mat4_copy(view_matrices.model_matrix, shadow_view_matrices.model_matrix);

  /* Map uniform buffers and update them */
  wgpu_queue_write_buffer(context->wgpu_context,
                          buffers.normal_map_vs_uniform_buffer.buffer, 0,
                          &view_matrices, sizeof(view_matrices));

  wgpu_queue_write_buffer(context->wgpu_context,
                          buffers.uniform_buffer_light.buffer, 64 + 64,
                          shadow_view_matrices.model_matrix, sizeof(mat4));

  wgpu_queue_write_buffer(context->wgpu_context,
                          buffers.normal_map_fs_uniform_buffer.buffer, 0,
                          camera.eye, sizeof(vec3));
}

static void prepare_buffers(wgpu_context_t* wgpu_context)
{
  //******************************* Torus Knot *******************************//

  /* Vertex buffer */
  buffers.torus_knot.vertex = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.vertices),
                    .initial.data = torus_knot_mesh.vertices,
                  });

  /* Index buffer */
  buffers.torus_knot.index = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(torus_knot_mesh.indices),
                    .initial.data = torus_knot_mesh.indices,
                    .count        = TORUS_KNOT_INDEX_COUNT,
                  });

  /* UV buffer */
  buffers.torus_knot.uv = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - UV buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.uvs),
                    .initial.data = torus_knot_mesh.uvs,
                  });

  /* Normal buffer */
  buffers.torus_knot.normal = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - Normal buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.normals),
                    .initial.data = torus_knot_mesh.normals,
                  });

  /* Tangent buffer */
  buffers.torus_knot.tangent = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - Tangents buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.tangents),
                    .initial.data = torus_knot_mesh.tangents,
                  });

  /* Bitangent buffer */
  buffers.torus_knot.bitangent = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot - Bitangents buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.bitangents),
                    .initial.data = torus_knot_mesh.bitangents,
                  });

  //********************************* Plane **********************************//

  /* Vertex buffer */
  buffers.plane.vertex = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Plane - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(plane_mesh.vertices),
                    .initial.data = plane_mesh.vertices,
                  });

  /* Index buffer */
  buffers.plane.index = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Plane - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(plane_mesh.indices),
                    .initial.data = plane_mesh.indices,
                    .count        = PLANE_INDEX_COUNT,
                  });

  /* UV buffer */
  buffers.plane.uv = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Plane - UV buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(plane_mesh.uvs),
                    .initial.data = plane_mesh.uvs,
                  });

  /* Normal buffer */
  buffers.plane.normal = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Plane - Normal buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(plane_mesh.normals),
                    .initial.data = plane_mesh.normals,
                  });

  /* Tangent buffer */
  buffers.plane.tangent = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Plane - Tangents buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(plane_mesh.tangents),
                    .initial.data = plane_mesh.tangents,
                  });

  /* Bitangent buffer */
  buffers.plane.bitangent = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Plane - Bitangents buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(plane_mesh.bitangents),
                    .initial.data = plane_mesh.bitangents,
                  });

  //***************************** Uniform Buffer *****************************//

  /* Normal map vertex shader uniform buffer */
  buffers.normal_map_vs_uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Normal map vertex shader - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(view_matrices),
                    .initial.data = &view_matrices,
                  });

  /* Normal map fragment shader uniform buffer */
  buffers.normal_map_fs_uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Normal map fragment shader - Uniform buffer 0",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(light_positions),
                    .initial.data = &light_positions,
                  });

  /* Light uniform buffer */
  buffers.uniform_buffer_light = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Light - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(shadow_view_matrices),
                    .initial.data = &shadow_view_matrices,
                  });
}

static void prepare_textures(wgpu_context_t* wgpu_context)
{
  /* Diffuse texture*/
  {
    const char* file = "textures/brics_diffuse.jpg";
    textures.diffuse = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  /* Normal texture*/
  {
    const char* file = "textures/brics_normal.jpg";
    textures.normal  = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  /* Specular texture*/
  {
    const char* file  = "textures/brics_specular.jpg";
    textures.specular = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  /* Samplers */
  {
    samplers.normal_map = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = "Normal map - Texture sampler",
                              .addressModeU  = WGPUAddressMode_Repeat,
                              .addressModeV  = WGPUAddressMode_Repeat,
                              .addressModeW  = WGPUAddressMode_Repeat,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(samplers.normal_map);
  }

  /* Depth texture */
  {
    textures.depth.texture =  wgpuDeviceCreateTexture(wgpu_context->device,
      &(WGPUTextureDescriptor) {
        .label         = "Depth - Texture",
        .usage         = WGPUTextureUsage_RenderAttachment,
        .dimension     = WGPUTextureDimension_2D,
        .format        = WGPUTextureFormat_Depth24Plus,
        .mipLevelCount = 1,
        .sampleCount   = 1,
        .size          = (WGPUExtent3D)  {
          .width               = wgpu_context->surface.width,
          .height              = wgpu_context->surface.height,
          .depthOrArrayLayers  = 1,
        },
      });

    textures.depth.view = wgpuTextureCreateView(
      textures.depth.texture, &(WGPUTextureViewDescriptor){
                                .label         = "Depth - Texture view",
                                .dimension     = WGPUTextureViewDimension_2D,
                                .format        = WGPUTextureFormat_Depth24Plus,
                                .mipLevelCount = 1,
                                .arrayLayerCount = 1,
                              });
  }
}

static void setup_render_passes(void)
{
  /* Color attachment */
  render_pass.normal_map.color_attachments[0] = (WGPURenderPassColorAttachment) {
    .view       = NULL, /* Assigned later */
    .depthSlice = ~0,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor) {
      .r = 0.3f,
      .g = 0.4f,
      .b = 0.5f,
      .a = 1.0f,
    },
  };

  /* Depth-stencil attachment */
  render_pass.normal_map.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view            = textures.depth.view,
      .depthClearValue = 1.0f,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
    };

  /* Render pass descriptor */
  render_pass.normal_map.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Normal map - Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.normal_map.color_attachments,
    .depthStencilAttachment = &render_pass.normal_map.depth_stencil_attachment,
  };
}

static void setup_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[7] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = buffers.normal_map_vs_uniform_buffer.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = buffers.normal_map_fs_uniform_buffer.size,
      },
      .sampler = {0},
    },
    [4] = (WGPUBindGroupLayoutEntry) {
      .binding    = 4,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = buffers.uniform_buffer_light.size,
      },
      .sampler = {0},
    },
    [5] = (WGPUBindGroupLayoutEntry) {
      .binding    = 5,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [6] = (WGPUBindGroupLayoutEntry) {
      .binding    = 6,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  bind_group_layouts.normal_map = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Normal map pass - Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layouts.normal_map != NULL);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayout bind_group_layout_list[1] = {
    bind_group_layouts.normal_map, /* group 0 */
  };
  pipeline_layouts.normal_map = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = "Normal map pass - Pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layout_list),
      .bindGroupLayouts     = bind_group_layout_list,
    });
  ASSERT(pipeline_layouts.normal_map != NULL);
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[7] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = buffers.normal_map_vs_uniform_buffer.buffer,
      .offset  = 0,
      .size    = buffers.normal_map_vs_uniform_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .sampler = samplers.normal_map,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = textures.diffuse.view,
    },
    [3] = (WGPUBindGroupEntry) {
      .binding = 3,
      .buffer  = buffers.normal_map_fs_uniform_buffer.buffer,
      .offset  = 0,
      .size    = buffers.normal_map_fs_uniform_buffer.size,
    },
    [4] = (WGPUBindGroupEntry) {
      .binding = 4,
      .buffer  = buffers.uniform_buffer_light.buffer,
      .offset  = 0,
      .size    = buffers.uniform_buffer_light.size,
    },
    [5] = (WGPUBindGroupEntry) {
      .binding     = 5,
      .textureView = textures.normal.view,
    },
    [6] = (WGPUBindGroupEntry) {
      .binding     = 6,
      .textureView = textures.specular.view,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Normal map pass - Bind group",
    .layout     = bind_group_layouts.normal_map,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  bind_groups.normal_map
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(bind_groups.normal_map != NULL);
}

static void prepare_normal_map_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  // Enable depth testing so that the fragment closest to the camera is rendered
  // in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPUVertexBufferLayout normal_map_vertex_buffer_layouts[5] = {0};
  {
    WGPUVertexAttribute attribute = {
      // Shader location 0 : position attribute
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    normal_map_vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 1 : uv attribute
      .shaderLocation = 1,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x2,
    };
    normal_map_vertex_buffer_layouts[1] = (WGPUVertexBufferLayout){
      .arrayStride    = 2 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 2 : normal attribute
      .shaderLocation = 2,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    normal_map_vertex_buffer_layouts[2] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 3 : tangent attribute
      .shaderLocation = 3,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    normal_map_vertex_buffer_layouts[3] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 4 : bitangent attribute
      .shaderLocation = 4,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    normal_map_vertex_buffer_layouts[4] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "Normal map - Vertex shader WGSL",
                      .wgsl_code.source = normal_map_vertex_shader_wgsl,
                      .entry            = "main",
                    },
                    .buffer_count = (uint32_t)ARRAY_SIZE(normal_map_vertex_buffer_layouts),
                    .buffers      = normal_map_vertex_buffer_layouts,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "Normal map - Fragment shader WGSL",
                      .wgsl_code.source = normal_map_fragment_shader_wgsl,
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
  pipelines.normal_map = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Normal map - Render pipeline",
                            .layout       = pipeline_layouts.normal_map,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipelines.normal_map != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_meshes();
    initialize_camera(context->wgpu_context);
    prepare_uniform_data(context->wgpu_context);
    prepare_buffers(context->wgpu_context);
    prepare_textures(context->wgpu_context);
    setup_bind_group_layout(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
    prepare_normal_map_pipeline(context->wgpu_context);
    setup_render_passes();
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Normap map render pass */
  {
    render_pass.normal_map.color_attachments[0].view
      = wgpu_context->swap_chain.frame_buffer;

    /* Begin render pass */
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.normal_map.descriptor);

    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.normal_map);

    /* Render torus knot mesh */
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         buffers.torus_knot.vertex.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                         buffers.torus_knot.uv.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                         buffers.torus_knot.normal.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 3,
                                         buffers.torus_knot.tangent.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 4,
                                         buffers.torus_knot.bitangent.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, buffers.torus_knot.index.buffer,
      WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.normal_map, 0, 0);
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                     TORUS_KNOT_INDEX_COUNT, 1, 0, 0, 0);

    /* Render plane mesh */
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         buffers.plane.vertex.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 1, buffers.plane.uv.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                         buffers.plane.normal.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 3,
                                         buffers.plane.tangent.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 4,
                                         buffers.plane.bitangent.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, buffers.plane.index.buffer,
      WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.normal_map, 0, 0);
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, PLANE_INDEX_COUNT,
                                     1, 0, 0, 0);

    /* End render pass */
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  /* Submit to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  wgpu_destroy_buffer(&buffers.torus_knot.vertex);
  wgpu_destroy_buffer(&buffers.torus_knot.index);
  wgpu_destroy_buffer(&buffers.torus_knot.uv);
  wgpu_destroy_buffer(&buffers.torus_knot.normal);
  wgpu_destroy_buffer(&buffers.torus_knot.tangent);
  wgpu_destroy_buffer(&buffers.torus_knot.bitangent);

  wgpu_destroy_buffer(&buffers.plane.vertex);
  wgpu_destroy_buffer(&buffers.plane.index);
  wgpu_destroy_buffer(&buffers.plane.uv);
  wgpu_destroy_buffer(&buffers.plane.normal);
  wgpu_destroy_buffer(&buffers.plane.tangent);
  wgpu_destroy_buffer(&buffers.plane.bitangent);

  wgpu_destroy_buffer(&buffers.normal_map_vs_uniform_buffer);
  wgpu_destroy_buffer(&buffers.normal_map_fs_uniform_buffer);
  wgpu_destroy_buffer(&buffers.uniform_buffer_light);

  wgpu_destroy_texture(&textures.diffuse);
  wgpu_destroy_texture(&textures.normal);
  wgpu_destroy_texture(&textures.specular);
  wgpu_destroy_texture(&textures.depth);

  WGPU_RELEASE_RESOURCE(Sampler, samplers.normal_map)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.normal_map)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.normal_map)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.normal_map)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.normal_map)
}

void example_normal_mapping(int argc, char* argv[])
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
static const char* normal_map_vertex_shader_wgsl = CODE(
  struct Uniform {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @group(0) @binding(0) var<uniform> uniforms : Uniform;

  struct UniformLight {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @group(0) @binding(4) var<uniform> uniformsLight : UniformLight;

  struct Output {
     @builtin(position) Position : vec4<f32>,
     @location(0) fragPosition : vec3<f32>,
     @location(1) fragUV : vec2<f32>,
     // @location(2) fragNormal : vec3<f32>,
     @location(3) shadowPos : vec3<f32>,
     @location(4) fragNor : vec3<f32>,
     @location(5) fragTangent : vec3<f32>,
     @location(6) fragBitangent : vec3<f32>
  };


  @vertex
  fn main(@location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>
  ) -> Output {
    var output: Output;
    output.Position = uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
    output.fragPosition = (uniforms.mMatrix * pos).xyz;
    output.fragUV = uv;
    //output.fragNormal  = (uniforms.mMatrix * vec4<f32>(normal,1.0)).xyz;

    // -----NORMAL --------------------------------

    var nMatrix : mat4x4<f32> = uniforms.mMatrix;
    nMatrix[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    let norm : vec3<f32>  = normalize((nMatrix * vec4<f32>(normal,1.0)).xyz);
    let tang : vec3<f32> = normalize((nMatrix * vec4<f32>(tangent,1.0)).xyz);
    let binormal : vec3<f32> = normalize((nMatrix * vec4<f32>(bitangent,1.0)).xyz);

    output.fragNor  = norm;
    output.fragTangent  = tang;
    output.fragBitangent  = binormal;


    let posFromLight: vec4<f32> = uniformsLight.pMatrix * uniformsLight.vMatrix * uniformsLight.mMatrix * pos;
    // Convert shadowPos XY to (0, 1) to fit texture UV
    output.shadowPos = vec3<f32>(posFromLight.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5), posFromLight.z);

    return output;
   }
);

static const char* normal_map_fragment_shader_wgsl = CODE(
  @binding(1) @group(0) var textureSampler : sampler;
  @binding(2) @group(0) var textureData : texture_2d<f32>;
  @binding(5) @group(0) var textureDataNormal : texture_2d<f32>;
  @binding(6) @group(0) var textureDataSpecular : texture_2d<f32>;

  struct Uniforms {
    eyePosition : vec4<f32>,
    lightPosition : vec4<f32>,
  };
  @binding(3) @group(0) var<uniform> uniforms : Uniforms;

  @fragment
  fn main(@location(0) fragPosition: vec3<f32>,
    @location(1) fragUV: vec2<f32>,
    //@location(2) fragNormal: vec3<f32>,
    @location(3) shadowPos: vec3<f32>,
    @location(4) fragNor: vec3<f32>,
    @location(5) fragTangent: vec3<f32>,
    @location(6) fragBitangent: vec3<f32>
  ) -> @location(0) vec4<f32> {
    let specularColor: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    let i = 4.0f;
    let textureColor: vec3<f32> = (textureSample(textureData, textureSampler, fragUV * i)).rgb;
    let texturSpecular: vec3<f32> = (textureSample(textureDataSpecular, textureSampler, fragUV * i)).rgb;

    var textureNormal: vec3<f32> = normalize(2.0 * (textureSample(textureDataNormal, textureSampler, fragUV * i)).rgb - 1.0);
    var colorNormal = normalize(vec3<f32>(textureNormal.x, textureNormal.y, textureNormal.z));
    colorNormal.y *= -1;

    var tbnMatrix : mat3x3<f32> = mat3x3<f32>(
      normalize(fragTangent),
      normalize(fragBitangent),
      normalize(fragNor)
    );

    colorNormal = normalize(tbnMatrix * colorNormal);

    let N: vec3<f32> = normalize(colorNormal.xyz);
    let L: vec3<f32> = normalize((uniforms.lightPosition).xyz - fragPosition.xyz);
    let V: vec3<f32> = normalize((uniforms.eyePosition).xyz - fragPosition.xyz);
    let H: vec3<f32> = normalize(L + V);

    let diffuse: f32 = 0.8 * max(dot(N, L), 0.0);
    let specular = pow(max(dot(N, H),0.0),100.0);

    let finalColor: vec3<f32> =  textureColor * diffuse + (texturSpecular * specular );
    // let finalColor:vec3<f32> =  colorNormal * 0.5 + 0.5;  //let color = N * 0.5 + 0.5;
    // let finalColor:vec3<f32> =  texturSpecular ;  //let color = N * 0.5 + 0.5;

    return vec4<f32>(finalColor, 1.0);
  }
);
// clang-format on
