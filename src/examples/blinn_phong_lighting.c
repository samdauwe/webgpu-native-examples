#include "example_base.h"

#include <cJSON.h>
#include <string.h>

#include "../core/log.h"
#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Blinn-Phong Lighting example
 *
 * This example demonstrates how to render a torus knot mesh with blinn-phong
 * lighting model.
 *
 * Ref:
 * https://github.com/Konstantin84UKR/webgpu_examples/tree/master/phong
 * https://github.com/jack1232/ebook-webgpu-lighting/tree/main/src/examples/ch04
 *
 * Note:
 * https://learnopengl.com/Advanced-Lighting/Advanced-Lighting
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Vertex data - Torus Knot
 * -------------------------------------------------------------------------- */

#define TORUS_KNOT_VERTEX_COUNT 7893
#define TORUS_KNOT_FACES_COUNT 3000
#define TORUS_KNOT_INDEX_COUNT (TORUS_KNOT_FACES_COUNT * 3)
#define TORUS_KNOT_UV_COUNT 5262
#define TORUS_KNOT_NORMAL_COUNT 7893

static struct torus_knot_mesh {
  float vertices[TORUS_KNOT_VERTEX_COUNT];
  uint32_t indices[TORUS_KNOT_INDEX_COUNT];
  float uvs[TORUS_KNOT_UV_COUNT];
  float normals[TORUS_KNOT_NORMAL_COUNT];
} torus_knot_mesh = {0};

int prepare_torus_knot_mesh(void)
{
  int res = EXIT_FAILURE;

  file_read_result_t file_read_result = {0};
  read_file("meshes/model.json", &file_read_result, true);
  const char* const json_data = (const char* const)file_read_result.data;

  const cJSON* meshes_array       = NULL;
  const cJSON* meshes_item        = NULL;
  const cJSON* vertex_array       = NULL;
  const cJSON* vertex_item        = NULL;
  const cJSON* faces_array        = NULL;
  const cJSON* face_item          = NULL;
  const cJSON* texturecoord_array = NULL;
  const cJSON* texturecoords_item = NULL;
  const cJSON* texturecoord_item  = NULL;
  const cJSON* normal_array       = NULL;
  const cJSON* normal_item        = NULL;
  cJSON* model_json               = cJSON_Parse(json_data);
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

  /* Get first mesh */
  meshes_array = cJSON_GetObjectItemCaseSensitive(model_json, "meshes");
  if (!cJSON_IsArray(meshes_array)) {
    log_error("'meshes' object item is not an array");
    goto load_json_end;
  }
  if (!(cJSON_GetArraySize(meshes_array) > 0)) {
    log_error("'meshes' array does not contain any mesh object");
    goto load_json_end;
  }
  meshes_item = cJSON_GetArrayItem(meshes_array, 0);

  if (!cJSON_IsObject(meshes_item)
      || !cJSON_HasObjectItem(meshes_item, "vertices")
      || !cJSON_HasObjectItem(meshes_item, "faces")
      || !cJSON_HasObjectItem(meshes_item, "texturecoords")
      || !cJSON_HasObjectItem(meshes_item, "normals")) {
    log_error(
      "Invalid mesh object, does not contain 'vertices', 'faces', "
      "'texturecoords', 'normals' array");
    goto load_json_end;
  }

  /* Parse vertices */
  {
    vertex_array = cJSON_GetObjectItemCaseSensitive(meshes_item, "vertices");
    if (!cJSON_IsArray(vertex_array)) {
      log_error("vertices object item is not an array");
      goto load_json_end;
    }

    ASSERT(cJSON_GetArraySize(vertex_array) == TORUS_KNOT_VERTEX_COUNT);

    uint32_t c = 0;
    cJSON_ArrayForEach(vertex_item, vertex_array)
    {
      torus_knot_mesh.vertices[c++] = (float)vertex_item->valuedouble;
    }
  }

  /* Parse indices */
  {
    faces_array = cJSON_GetObjectItemCaseSensitive(meshes_item, "faces");
    if (!cJSON_IsArray(faces_array)) {
      log_error("'faces' object item is not an array");
      goto load_json_end;
    }

    ASSERT(cJSON_GetArraySize(faces_array) == TORUS_KNOT_FACES_COUNT);

    uint32_t c = 0;
    cJSON_ArrayForEach(face_item, faces_array)
    {
      if (!(cJSON_GetArraySize(face_item) == 3)) {
        log_error("'face' item is not an array of size 3");
        goto load_json_end;
      }
      for (uint32_t i = 0; i < 3; ++i) {
        torus_knot_mesh.indices[c++]
          = (uint32_t)cJSON_GetArrayItem(face_item, i)->valueint;
      }
    }
  }

  /* Parse uvs */
  {
    texturecoord_array
      = cJSON_GetObjectItemCaseSensitive(meshes_item, "texturecoords");
    if (!(cJSON_GetArraySize(texturecoord_array) > 0)) {
      log_error("'texturecoords' array does not contain any object");
      goto load_json_end;
    }
    texturecoords_item = cJSON_GetArrayItem(texturecoord_array, 0);
    if (!cJSON_IsArray(texturecoords_item)) {
      log_error("'texturecoords' object item is not an array");
      goto load_json_end;
    }

    ASSERT(cJSON_GetArraySize(texturecoords_item) == TORUS_KNOT_UV_COUNT);

    uint32_t c = 0;
    cJSON_ArrayForEach(texturecoord_item, texturecoords_item)
    {
      torus_knot_mesh.uvs[c++] = (float)texturecoord_item->valuedouble;
    }
  }

  /* Parse normals */
  {
    normal_array = cJSON_GetObjectItemCaseSensitive(meshes_item, "normals");
    if (!cJSON_IsArray(normal_array)) {
      log_error("'normals' object item is not an array");
      goto load_json_end;
    }

    ASSERT(cJSON_GetArraySize(normal_array) == TORUS_KNOT_NORMAL_COUNT);

    uint32_t c = 0;
    cJSON_ArrayForEach(normal_item, normal_array)
    {
      torus_knot_mesh.normals[c++] = (float)normal_item->valuedouble;
    }
  }

  res = EXIT_SUCCESS;

load_json_end:
  cJSON_Delete(model_json);
  free(file_read_result.data);

  return res;
}

/* -------------------------------------------------------------------------- *
 * Vertex data - Sphere Geometry
 * -------------------------------------------------------------------------- */

typedef struct range_t {
  void* ptr;
  size_t byte_length;
  size_t length;
} range_t;

typedef struct sphere_geometry_t {
  float radius;
  uint32_t width_segments;
  uint32_t height_segments;
  float phi_start;
  float phi_length;
  float theta_start;
  float theta_length;
  // vertex positions, texture coordinates, normals, tangents, and vertex
  // indices
  range_t vertices;
  range_t uvs;
  range_t normals;
  range_t tangents;
  range_t indices;
} sphere_geometry_t;

static sphere_geometry_t sphere_geometry = {0};

static void sphere_geometry_init_defaults(sphere_geometry_t* this)
{
  memset(this, 0, sizeof(*this));
}

void sphere_geometry_init(sphere_geometry_t* this, float radius,
                          uint32_t width_segments, uint32_t height_segments,
                          float phi_start, float phi_length, float theta_start,
                          float theta_length)
{
  sphere_geometry_init_defaults(this);

  this->radius          = radius;
  this->width_segments  = width_segments;
  this->height_segments = height_segments;
  this->phi_start       = phi_start;
  this->phi_length      = phi_length;
  this->theta_start     = theta_start;
  this->theta_length    = theta_length;

  // Generate vertex positions, texture coordinates, normals, tangents, and
  // vertex indices
  const uint32_t vertex_count = (width_segments + 1) * (height_segments + 1);

  float* vertices   = (float*)malloc(vertex_count * 3 * sizeof(float));
  float* uvs        = (float*)malloc(vertex_count * 2 * sizeof(float));
  float* normals    = (float*)malloc(vertex_count * 3 * sizeof(float));
  float* tangents   = (float*)malloc(vertex_count * 3 * sizeof(float));
  uint32_t* indices = (uint32_t*)malloc(vertex_count * 6 * sizeof(uint32_t));

  size_t vertices_length = 0, uvs_length = 0, normals_length = 0,
         tangents_length = 0, indices_length = 0;

  for (uint32_t iy = 0; iy <= height_segments; iy++) {
    const float v     = iy / (float)height_segments;
    const float theta = theta_start + v * theta_length;

    for (uint32_t ix = 0; ix <= width_segments; ix++) {
      const float u   = ix / (float)width_segments;
      const float phi = phi_start + u * phi_length;

      // Calculate vertex position
      const float x = -radius * cos(phi) * sin(theta);
      const float y = radius * cos(theta);
      const float z = radius * sin(phi) * sin(theta);

      vertices[vertices_length++] = x;
      vertices[vertices_length++] = y;
      vertices[vertices_length++] = z;

      // Calculate texture coordinates
      uvs[uvs_length++] = u;
      uvs[uvs_length++] = v; // Invert v-axis to match the typical convention

      // Calculate normal vector
      normals[normals_length++] = x;
      normals[normals_length++] = y;
      normals[normals_length++] = z;

      // Calculate tangent vector (same for all vertices)
      // Assuming the tangent vector points along the positive X-axis
      tangents[tangents_length++] = radius * sin(phi);
      tangents[tangents_length++] = 0.0f;
      tangents[tangents_length++] = radius * cos(phi);

      if (iy < height_segments && ix < width_segments) {
        const uint32_t current_index = ix + iy * (width_segments + 1);
        const uint32_t next_index_x  = current_index + 1;
        const uint32_t next_index_y  = current_index + width_segments + 1;
        const uint32_t next_index_xy = next_index_y + 1;

        // Generate indices for two triangles of each face
        indices[indices_length++] = current_index;
        indices[indices_length++] = next_index_y;
        indices[indices_length++] = next_index_x;

        indices[indices_length++] = next_index_y;
        indices[indices_length++] = next_index_xy;
        indices[indices_length++] = next_index_x;
      }
    }
  }

  // Initialize sphere geometry
  this->vertices.ptr         = vertices;
  this->vertices.byte_length = vertices_length * sizeof(float);
  this->vertices.length      = vertices_length;

  this->uvs.ptr         = uvs;
  this->uvs.byte_length = uvs_length * sizeof(float);
  this->uvs.length      = uvs_length;

  this->normals.ptr         = normals;
  this->normals.byte_length = normals_length * sizeof(float);
  this->normals.length      = normals_length;

  this->tangents.ptr         = tangents;
  this->tangents.byte_length = tangents_length * sizeof(float);
  this->tangents.length      = tangents_length;

  this->indices.ptr         = indices;
  this->indices.byte_length = indices_length * sizeof(uint32_t);
  this->indices.length      = indices_length;
}

static void sphere_geometry_destroy(sphere_geometry_t* this)
{
  range_t* sphere_data[5] = {
    &this->vertices, &this->uvs,     &this->normals,
    &this->tangents, &this->indices,
  };

  for (uint32_t i = 0; i < 5; ++i) {
    range_t* r = sphere_data[i];
    if ((r->ptr != NULL) && (r->length > 0)) {
      free(r->ptr);
      r->ptr    = NULL;
      r->length = 0;
    }
    r->byte_length = 0;
  }
}

void prepare_sphere_geometry(void)
{
  sphere_geometry_init(&sphere_geometry, 0.1f, 16, 8, 0.0f, PI2, 0.0f, PI);
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */
static const char* blinn_phong_lighting_torus_knot_vertex_shader_wgsl;
static const char* blinn_phong_lighting_torus_knot_fragment_shader_wgsl;
static const char* blinn_phong_lighting_sphere_vertex_shader_wgsl;
static const char* blinn_phong_lighting_sphere_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Blinn-Phong Lighting example
 * -------------------------------------------------------------------------- */

/* Buffers */
static struct {
  struct {
    wgpu_buffer_t vertex;
    wgpu_buffer_t index;
    wgpu_buffer_t uv;
    wgpu_buffer_t normal;
    wgpu_buffer_t vs_uniform;
    wgpu_buffer_t fs_uniform;
  } torus_knot;
  struct {
    wgpu_buffer_t vertex;
    wgpu_buffer_t index;
    wgpu_buffer_t vs_uniform;
  } sphere;
} buffers = {0};

/* Texture and sampler */
static struct {
  texture_t torus_knot_face;
  texture_t depth;
} textures = {0};

/* Uniform bind group and render pipeline (and layout) */
static struct {
  WGPUBindGroup torus_knot;
  WGPUBindGroup sphere;
} bind_groups = {0};

static struct {
  WGPURenderPipeline torus_knot;
  WGPURenderPipeline sphere;
} pipelines = {0};

/* Render pass descriptor for frame buffer writes */
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

/* Uniform data */
static struct {
  mat4 projection_matrix;
  mat4 view_matrix;
  mat4 model_matrix;
} torus_knot_view_matrices = {
  .projection_matrix = GLM_MAT4_IDENTITY_INIT,
  .view_matrix       = GLM_MAT4_IDENTITY_INIT,
  .model_matrix      = GLM_MAT4_IDENTITY_INIT,
};
static struct {
  mat4 projection_matrix;
  mat4 view_matrix;
  mat4 model_matrix;
} sphere_view_matrices = {
  .projection_matrix = GLM_MAT4_IDENTITY_INIT,
  .view_matrix       = GLM_MAT4_IDENTITY_INIT,
  .model_matrix      = GLM_MAT4_IDENTITY_INIT,
};
static float time_old = 0.0f;

static struct {
  vec4 eye_position;
  vec4 light_position;
} light_positions = {
  .eye_position   = {0.0f, 1.0f, 8.0f, 1.0f},
  .light_position = {0.0f, 0.0f, 1.0f, 1.0f},
};

// Other variables
static const char* example_title = "Blinn-Phong Lighting";
static bool prepared             = false;

static void prepare_uniform_data(wgpu_context_t* wgpu_context)
{
  /* View matrix */
  glm_lookat(light_positions.eye_position,        // eye vector
             (vec3){0.0f, 0.0f, 0.0f},            // center vector
             (vec3){0.0f, 1.0f, 0.0f},            // up vector
             torus_knot_view_matrices.view_matrix // result matrix
  );
  glm_mat4_copy(torus_knot_view_matrices.view_matrix,
                sphere_view_matrices.view_matrix);

  /* View projection matrix */
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;
  const float fovy = 40.0f * PI / 180.0f;
  glm_perspective(fovy, aspect_ratio, 1.f, 25.0f,
                  torus_knot_view_matrices.projection_matrix);
  glm_mat4_copy(torus_knot_view_matrices.projection_matrix,
                sphere_view_matrices.projection_matrix);

  /* Translate model matrix for the sphere geometry */
  glm_translate(sphere_view_matrices.model_matrix,
                light_positions.light_position);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  /* Time */
  const float now = context->frame.timestamp_millis;
  const float dt  = now - time_old;
  time_old        = now;

  /* Update view matrix update */
  {
    glm_rotate_x(torus_knot_view_matrices.model_matrix, dt * 0.0002f,
                 torus_knot_view_matrices.model_matrix);
    glm_rotate_y(torus_knot_view_matrices.model_matrix, dt * 0.0002f,
                 torus_knot_view_matrices.model_matrix);
    glm_rotate_z(torus_knot_view_matrices.model_matrix, dt * 0.0002f,
                 torus_knot_view_matrices.model_matrix);

    // Map uniform buffer and update it
    wgpu_queue_write_buffer(
      context->wgpu_context, buffers.torus_knot.vs_uniform.buffer, 64 + 64,
      &torus_knot_view_matrices.model_matrix[0], sizeof(mat4));
  }

  /* Update light position */
  {
    light_positions.light_position[0] = sin(now * 0.001f) * 4.0f;

    /* Update the shere view matrix based on the light position */
    {
      glm_mat4_identity(sphere_view_matrices.model_matrix);
      glm_translate(sphere_view_matrices.model_matrix,
                    light_positions.light_position);

      /* Map uniform buffer and update it */
      wgpu_queue_write_buffer(context->wgpu_context,
                              buffers.sphere.vs_uniform.buffer, 64 + 64,
                              sphere_view_matrices.model_matrix, sizeof(mat4));
    }

    /* Map uniform buffer and update it */
    wgpu_queue_write_buffer(context->wgpu_context,
                            buffers.torus_knot.fs_uniform.buffer, 16,
                            &light_positions.light_position[0], sizeof(vec4));
  }
}

static void prepare_buffers(wgpu_context_t* wgpu_context)
{
  //******************************* Torus Knot *******************************//

  /* Vertex buffer */
  buffers.torus_knot.vertex = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.vertices),
                    .initial.data = torus_knot_mesh.vertices,
                  });

  /* Index buffer */
  buffers.torus_knot.index = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(torus_knot_mesh.indices),
                    .initial.data = torus_knot_mesh.indices,
                  });

  /* UV buffer */
  buffers.torus_knot.uv = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "UV buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.uvs),
                    .initial.data = torus_knot_mesh.uvs,
                  });

  /* Normal buffer */
  buffers.torus_knot.normal = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Normal buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(torus_knot_mesh.normals),
                    .initial.data = torus_knot_mesh.normals,
                  });

  /* Vertex shader uniform buffer */
  buffers.torus_knot.vs_uniform = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot vertex shader uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(torus_knot_view_matrices),
                    .initial.data = &torus_knot_view_matrices,
                  });

  /* Fragment shader uniform buffer */
  buffers.torus_knot.fs_uniform = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Torus knot fragment shader uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(light_positions),
                    .initial.data = &light_positions,
                  });

  //**************************** Sphere Geometry *****************************//

  /* Vertex buffer */
  buffers.sphere.vertex = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sphere vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sphere_geometry.vertices.byte_length,
                    .initial.data = sphere_geometry.vertices.ptr,
                  });

  /* Index buffer */
  buffers.sphere.index = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sphere index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sphere_geometry.indices.byte_length,
                    .initial.data = sphere_geometry.indices.ptr,
                    .count        = sphere_geometry.indices.length,
                  });

  /* Vertex shader uniform buffer */
  buffers.sphere.vs_uniform = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Sphere vertex shader uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(sphere_view_matrices),
                    .initial.data = &sphere_view_matrices,
                  });
}

static void prepare_textures(wgpu_context_t* wgpu_context)
{
  /* Torus knot face texture*/
  {
    const char* file = "textures/uv.jpg";
    textures.torus_knot_face
      = wgpu_create_texture_from_file(wgpu_context, file, NULL);
  }

  /* Depth texture */
  {
    textures.depth.texture =  wgpuDeviceCreateTexture(wgpu_context->device,
      &(WGPUTextureDescriptor) {
        .label         = "Depth texture",
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
                                .label         = "Depth texture view",
                                .dimension     = WGPUTextureViewDimension_2D,
                                .format        = WGPUTextureFormat_Depth24Plus,
                                .mipLevelCount = 1,
                                .arrayLayerCount = 1,
                              });
  }
}

static void setup_render_pass(void)
{
  /* Color attachment */
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
    .view       = NULL, /* Assigned later */
    .depthSlice = ~0,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor) {
      .r = 0.1f,
      .g = 0.2f,
      .b = 0.3f,
      .a = 1.0f,
    },
  };

  /* Depth-stecil attachment */
  render_pass.depth_stencil_attachment = (WGPURenderPassDepthStencilAttachment){
    .view            = textures.depth.view,
    .depthClearValue = 1.0f,
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
  };

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &render_pass.depth_stencil_attachment,
  };
}

static void setup_torus_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = buffers.torus_knot.vs_uniform.buffer,
      .offset  = 0,
      .size    = buffers.torus_knot.vs_uniform.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .sampler = textures.torus_knot_face.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = textures.torus_knot_face.view,
    },
    [3] = (WGPUBindGroupEntry) {
      .binding = 3,
      .buffer  = buffers.torus_knot.fs_uniform.buffer,
      .offset  = 0,
      .size    = buffers.torus_knot.fs_uniform.size,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Torus knot uniform buffer bind group",
    .layout     = wgpuRenderPipelineGetBindGroupLayout(pipelines.torus_knot, 0),
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  bind_groups.torus_knot
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(bind_groups.torus_knot != NULL);
}

static void setup_sphere_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = buffers.sphere.vs_uniform.buffer,
      .offset  = 0,
      .size    = buffers.sphere.vs_uniform.size,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Sphere uniform buffer bind group",
    .layout     = wgpuRenderPipelineGetBindGroupLayout(pipelines.sphere, 0),
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  bind_groups.sphere
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(bind_groups.sphere != NULL);
}

static void prepare_torus_knot_pipeline(wgpu_context_t* wgpu_context)
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
  WGPUVertexBufferLayout textured_torus_knot_vertex_buffer_layouts[3] = {0};
  {
    WGPUVertexAttribute attribute = {
      // Shader location 0 : position attribute
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    textured_torus_knot_vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
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
    textured_torus_knot_vertex_buffer_layouts[1] = (WGPUVertexBufferLayout){
      .arrayStride    = 2 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 2 : Normal attribute
      .shaderLocation = 2,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    textured_torus_knot_vertex_buffer_layouts[2] = (WGPUVertexBufferLayout){
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
                      .label            = "Blinn-Phong lighting torus knot vertex shader WGSL",
                      .wgsl_code.source = blinn_phong_lighting_torus_knot_vertex_shader_wgsl,
                      .entry            = "main",
                    },
                    .buffer_count = (uint32_t)ARRAY_SIZE(textured_torus_knot_vertex_buffer_layouts),
                    .buffers = textured_torus_knot_vertex_buffer_layouts,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "Blinn-Phong lighting torus knot fragment shader WGSL",
                      .wgsl_code.source = blinn_phong_lighting_torus_knot_fragment_shader_wgsl,
                      .entry = "main",
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
  pipelines.torus_knot = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label        = "Blinn phong lighting torus knot render pipeline",
      .primitive    = primitive_state,
      .vertex       = vertex_state,
      .fragment     = &fragment_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    });
  ASSERT(pipelines.torus_knot != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void prepare_sphere_pipeline(wgpu_context_t* wgpu_context)
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
  WGPUVertexBufferLayout shere_vertex_buffer_layouts[1] = {0};
  {
    WGPUVertexAttribute attribute = {
      // Shader location 0 : position attribute
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    shere_vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
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
                      .label            = "Blinn-Phong lighting sphere vertex shader WGSL",
                      .wgsl_code.source = blinn_phong_lighting_sphere_vertex_shader_wgsl,
                      .entry            = "main",
                    },
                    .buffer_count = (uint32_t)ARRAY_SIZE(shere_vertex_buffer_layouts),
                    .buffers      = shere_vertex_buffer_layouts,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "Blinn-Phong lighting sphere fragment shader WGSL",
                      .wgsl_code.source = blinn_phong_lighting_sphere_fragment_shader_wgsl,
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
  pipelines.sphere = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label        = "Blinn-Phong lighting sphere render pipeline",
      .primitive    = primitive_state,
      .vertex       = vertex_state,
      .fragment     = &fragment_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    });
  ASSERT(pipelines.sphere != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_torus_knot_mesh();
    prepare_sphere_geometry();
    prepare_uniform_data(context->wgpu_context);
    prepare_buffers(context->wgpu_context);
    prepare_textures(context->wgpu_context);
    prepare_torus_knot_pipeline(context->wgpu_context);
    prepare_sphere_pipeline(context->wgpu_context);
    setup_torus_bind_group(context->wgpu_context);
    setup_sphere_bind_group(context->wgpu_context);
    setup_render_pass();
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
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Begin render pass */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Record torus knot render pass */
  {
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.torus_knot);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         buffers.torus_knot.vertex.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                         buffers.torus_knot.uv.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                         buffers.torus_knot.normal.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, buffers.torus_knot.index.buffer,
      WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.torus_knot, 0, 0);
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                     TORUS_KNOT_INDEX_COUNT, 1, 0, 0, 0);
  }

  /* Record sphere render pass */
  {
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.sphere);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         buffers.sphere.vertex.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, buffers.sphere.index.buffer,
      WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.sphere, 0, 0);
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                     buffers.sphere.index.count, 1, 0, 0, 0);
  }

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

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

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_SUCCESS;
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
  sphere_geometry_destroy(&sphere_geometry);
  wgpu_destroy_buffer(&buffers.torus_knot.vertex);
  wgpu_destroy_buffer(&buffers.torus_knot.index);
  wgpu_destroy_buffer(&buffers.torus_knot.uv);
  wgpu_destroy_buffer(&buffers.torus_knot.normal);
  wgpu_destroy_buffer(&buffers.torus_knot.vs_uniform);
  wgpu_destroy_buffer(&buffers.torus_knot.fs_uniform);
  wgpu_destroy_buffer(&buffers.sphere.vertex);
  wgpu_destroy_buffer(&buffers.sphere.index);
  wgpu_destroy_texture(&textures.torus_knot_face);
  wgpu_destroy_texture(&textures.depth);
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.torus_knot)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.sphere)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.torus_knot)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.sphere)
}

void example_blinn_phong_lighting(int argc, char* argv[])
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
static const char* blinn_phong_lighting_torus_knot_vertex_shader_wgsl = CODE(
  struct Uniform {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @binding(0) @group(0) var<uniform> uniforms : Uniform;

  struct Output {
    @builtin(position) Position : vec4<f32>,
    @location(0) vPosition : vec4<f32>,
    @location(1) vUV : vec2<f32>,
    @location(2) vNormal : vec4<f32>,
  };

  @vertex
  fn main(
    @location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>
  ) -> Output {
    var output: Output;
    output.Position = uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
    output.vPosition = uniforms.mMatrix * pos;
    output.vUV = uv;
    output.vNormal   =  uniforms.mMatrix * vec4<f32>(normal,1.0);

    return output;
  }
);

static const char* blinn_phong_lighting_torus_knot_fragment_shader_wgsl = CODE(
  @binding(1) @group(0) var textureSampler : sampler;
  @binding(2) @group(0) var textureData : texture_2d<f32>;

  const PI : f32 = 3.1415926535897932384626433832795;

  struct Uniforms {
    eyePosition : vec4<f32>,
    lightPosition : vec4<f32>,
  };
  @binding(3) @group(0) var<uniform> uniforms : Uniforms;

  fn lin2rgb(lin: vec3<f32>) -> vec3<f32>{
    return pow(lin, vec3<f32>(1.0/2.2));
  }

  fn rgb2lin(rgb: vec3<f32>) -> vec3<f32>{
    return pow(rgb, vec3<f32>(2.2));
  }

  fn brdfPhong(lighDir: vec3<f32>,
    viewDir: vec3<f32>,
    halfDir: vec3<f32>,
    normal: vec3<f32>,
    phongDiffuseColor: vec3<f32>,
    phongSpecularColor: vec3<f32>,
    phongShiniess:f32) -> vec3<f32>{

    var color : vec3<f32> =  phongDiffuseColor;
    let specDot : f32 = max(dot(normal, halfDir),0.0);
    color +=  pow(specDot, phongShiniess) * phongSpecularColor;
    return color;
  }

  fn modifiedPhongBRDF(lighDir: vec3<f32>,
    viewDir: vec3<f32>,
    halfDir: vec3<f32>,
    normal: vec3<f32>,
    phongDiffuseColor: vec3<f32>,
    phongSpecularColor: vec3<f32>,
    phongShininess:f32) -> vec3<f32>{

    var color : vec3<f32> =  phongDiffuseColor / PI;
    let specDot : f32 = max(dot(normal, halfDir),0.0);
    let normalization = (phongShininess + 2.0) / (2.0 * PI);
    color +=  pow(specDot, phongShininess) * normalization * phongSpecularColor;
    return color;
  }

  @fragment
  fn main(
    @location(0) vPosition: vec4<f32>,
    @location(1) vUV: vec2<f32>,
    @location(2) vNormal:  vec4<f32>
  ) -> @location(0) vec4<f32> {
    let specularColor:vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    let diffuseColor:vec3<f32> = vec3<f32>(0.25, 0.5, 0.2);
    let lightColor:vec3<f32> = vec3<f32>(0.6, 0.7, 0.8);
    let ambientColor:vec3<f32> = vec3<f32>(0.1, 0.1, 0.15);
    let shiniess = 100.0;
    let flux = 10.0;

    let textureColor:vec3<f32> = (textureSample(textureData, textureSampler, vUV)).rgb;

    let N:vec3<f32> = normalize(vNormal.xyz);
    let L:vec3<f32> = normalize((uniforms.lightPosition).xyz - vPosition.xyz);
    let V:vec3<f32> = normalize((uniforms.eyePosition).xyz - vPosition.xyz);
    let H:vec3<f32> = normalize(L + V);


    let distlight = distance((uniforms.lightPosition).xyz, vPosition.xyz);
    let R = length((uniforms.lightPosition).xyz - vPosition.xyz);

    let ambient : vec3<f32> = rgb2lin(textureColor.rgb) * rgb2lin(ambientColor.rgb);

    let irradiance = flux / (4.0 * PI * distlight * distlight) * max(dot(N,L), 0.0); // pointLight
    //let irradiance : f32 = 1.0 * max(dot(N, L), 0.0); // sun

    let brdf = brdfPhong(L,V,H,N, rgb2lin(textureColor), rgb2lin(specularColor),shiniess);

    var radiance = brdf * irradiance * rgb2lin(lightColor.rgb) + ambient;

    let finalColor:vec3<f32> =  vec3<f32>(L.x,L.y,L.z);

    // return vec4<f32>(finalColor, 1.0);
    return vec4<f32>(lin2rgb(radiance), 1.0);
  }
);

static const char* blinn_phong_lighting_sphere_vertex_shader_wgsl = CODE(
  struct Uniform {
    pMatrix : mat4x4<f32>,
    vMatrix : mat4x4<f32>,
    mMatrix : mat4x4<f32>,
  };
  @binding(0) @group(0) var<uniform> uniforms : Uniform;

  struct Output {
    @builtin(position) Position : vec4<f32>
  };

  @vertex
  fn main(@location(0) pos: vec4<f32>) -> Output {
    var output: Output;
    output.Position = uniforms.pMatrix * uniforms.vMatrix * uniforms.mMatrix * pos;
    return output;
  }
);

static const char* blinn_phong_lighting_sphere_fragment_shader_wgsl = CODE(
  @fragment
  fn main() -> @location(0) vec4<f32> {
    let finalColor:vec3<f32> =  vec3<f32>(0.9,0.9,0.9);
    return vec4<f32>(finalColor, 1.0);
  }
);
// clang-format on
