#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

#define PAR_SHAPES_IMPLEMENTATION
#include "par_shapes.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Procedural Mesh
 *
 * This WebGPU sample shows how to efficiently draw several procedurally
 * generated meshes:
 *  - All vertices and indices are stored in one large vertex/index buffer
 *  - Two bind groups are used - frame bind group and draw bind group
 *  - Simple physically-based shading is used
 *  - Single-pass wireframe rendering
 *  - Main drawing loop is optimized and changes just one dynamic offset before
 *    each draw call
 *
 * Ref:
 * https://github.com/michal-z/zig-gamedev/tree/main/samples/procedural_mesh_wgpu
 * -------------------------------------------------------------------------- */

#define ALIGNMENT 256u // 256-byte alignment

/* -------------------------------------------------------------------------- *
 * WGSl Shaders
 * -------------------------------------------------------------------------- */

// Shaders
// clang-format off
static const char* vertex_shader_wgsl = CODE(
  struct DrawUniforms {
    object_to_world: mat4x4<f32>,
    basecolor_roughness: vec4<f32>,
  }
  @group(1) @binding(0) var<uniform> draw_uniforms: DrawUniforms;

  struct FrameUniforms {
    world_to_clip: mat4x4<f32>,
    camera_position: vec3<f32>,
  }
  @group(0) @binding(0) var<uniform> frame_uniforms: FrameUniforms;

  struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) barycentrics: vec3<f32>,
  }
  @stage(vertex) fn main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
  ) -> VertexOut {
    var output: VertexOut;
    output.position_clip = vec4<f32>(position, 1.0);
    // output.position_clip = vec4(position, 1.0) * draw_uniforms.object_to_world * frame_uniforms.world_to_clip;
    output.position = position;
    output.normal = normal * mat3x3(
      draw_uniforms.object_to_world[0].xyz,
      draw_uniforms.object_to_world[1].xyz,
      draw_uniforms.object_to_world[2].xyz,
    );
    let index = vertex_index % 3u;
    output.barycentrics = vec3(f32(index == 0u), f32(index == 1u), f32(index == 2u));
    return output;
  }
  );

static const char* fragment_shader_wgsl = CODE(
  struct DrawUniforms {
    object_to_world: mat4x4<f32>,
    basecolor_roughness: vec4<f32>,
  }
  @group(1) @binding(0) var<uniform> draw_uniforms: DrawUniforms;

  struct FrameUniforms {
    world_to_clip: mat4x4<f32>,
    camera_position: vec3<f32>,
  }
  @group(0) @binding(0) var<uniform> frame_uniforms: FrameUniforms;

  let pi = 3.1415926;

  fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

  // Trowbridge-Reitz GGX normal distribution function.
  fn distributionGgx(n: vec3<f32>, h: vec3<f32>, alpha: f32) -> f32 {
    let alpha_sq = alpha * alpha;
    let n_dot_h = saturate(dot(n, h));
    let k = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0;
    return alpha_sq / (pi * k * k);
  }

  fn geometrySchlickGgx(x: f32, k: f32) -> f32 {
    return x / (x * (1.0 - k) + k);
  }

  fn geometrySmith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, k: f32) -> f32 {
    let n_dot_v = saturate(dot(n, v));
    let n_dot_l = saturate(dot(n, l));
    return geometrySchlickGgx(n_dot_v, k) * geometrySchlickGgx(n_dot_l, k);
  }

  fn fresnelSchlick(h_dot_v: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3(1.0, 1.0, 1.0) - f0) * pow(1.0 - h_dot_v, 5.0);
  }

  @stage(fragment) fn main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) barycentrics: vec3<f32>,
  ) -> @location(0) vec4<f32> {
    let v = normalize(frame_uniforms.camera_position - position);
    let n = normalize(normal);

    let base_color = draw_uniforms.basecolor_roughness.xyz;
    let ao = 1.0;
    var roughness = draw_uniforms.basecolor_roughness.a;
    var metallic: f32;
    if (roughness < 0.0) { metallic = 1.0; } else { metallic = 0.0; }
    roughness = abs(roughness);

    let alpha = roughness * roughness;
    var k = alpha + 1.0;
    k = (k * k) / 8.0;
    var f0 = vec3(0.04);
    f0 = mix(f0, base_color, metallic);

    let light_positions = array<vec3<f32>, 4>(
        vec3(25.0, 15.0, 25.0),
        vec3(-25.0, 15.0, 25.0),
        vec3(25.0, 15.0, -25.0),
        vec3(-25.0, 15.0, -25.0),
    );
    let light_radiance = array<vec3<f32>, 4>(
        4.0 * vec3(0.0, 100.0, 250.0),
        8.0 * vec3(200.0, 150.0, 250.0),
        3.0 * vec3(200.0, 0.0, 0.0),
        9.0 * vec3(200.0, 150.0, 0.0),
    );

    var lo = vec3(0.0);
    for (var light_index: i32 = 0; light_index < 4; light_index = light_index + 1) {
        let lvec = light_positions[light_index] - position;

        let l = normalize(lvec);
        let h = normalize(l + v);

        let distance_sq = dot(lvec, lvec);
        let attenuation = 1.0 / distance_sq;
        let radiance = light_radiance[light_index] * attenuation;

        let f = fresnelSchlick(saturate(dot(h, v)), f0);

        let ndf = distributionGgx(n, h, alpha);
        let g = geometrySmith(n, v, l, k);

        let numerator = ndf * g * f;
        let denominator = 4.0 * saturate(dot(n, v)) * saturate(dot(n, l));
        let specular = numerator / max(denominator, 0.001);

        let ks = f;
        let kd = (vec3(1.0) - ks) * (1.0 - metallic);

        let n_dot_l = saturate(dot(n, l));
        lo = lo + (kd * base_color / pi + specular) * radiance * n_dot_l;
    }

    let ambient = vec3(0.03) * base_color * ao;
    var color = ambient + lo;
    color = color / (color + 1.0);
    color = pow(color, vec3(1.0 / 2.2));

    // wireframe
    var barys = barycentrics;
    barys.z = 1.0 - barys.x - barys.y;
    let deltas = fwidth(barys);
    let smoothing = deltas * 1.0;
    let thickness = deltas * 0.25;
    barys = smoothstep(thickness, thickness + smoothing, barys);
    let min_bary = min(barys.x, min(barys.y, barys.z));
    return vec4(min_bary * color, 1.0);
  }
);
// clang-format on

/* -------------------------------------------------------------------------- *
 * Math functions
 *
 * @ref
 * https://github.com/michal-z/zig-gamedev/blob/main/libs/zmath/src/zmath.zig
 * -------------------------------------------------------------------------- */

static void look_to_lh(vec3 eyepos, vec3 eyedir, vec3 updir, mat4* dest)
{
  vec3 ax = GLM_VEC3_ZERO_INIT, ay = GLM_VEC3_ZERO_INIT,
       az = GLM_VEC3_ZERO_INIT;
  glm_normalize_to(eyedir, az);
  glm_vec3_cross(updir, az, ax);
  glm_normalize(ax);
  glm_vec3_cross(az, ax, ay);
  glm_normalize(ay);

  mat4 mat = {
    {ax[0], ax[1], ax[2], -glm_vec3_dot(ax, eyepos)}, //
    {ay[0], ay[1], ay[2], -glm_vec3_dot(ay, eyepos)}, //
    {az[0], az[1], az[2], -glm_vec3_dot(az, eyepos)}, //
    {0.0f, 0.0f, 0.0f, 1.0f},                         //
  };
  glm_mat4_transpose_to(mat, *dest);
}

static void perspective_fov_lh(float fovy, float aspect, float near, float far,
                               mat4* dest)
{
  const vec2 scfov = {sin(0.5f * fovy), cos(0.5f * fovy)};

  ASSERT(near > 0.0f && far > 0.0f && far > near);
  ASSERT(!approx_eq_fabs_eps(scfov[0], 0.0, 0.001));
  ASSERT(!approx_eq_fabs_eps(far, near, 0.001));
  ASSERT(!approx_eq_fabs_eps(aspect, 0.0, 0.01));

  const float h = scfov[1] / scfov[0];
  const float w = h / aspect;
  const float r = far / (far - near);
  mat4 mat      = {
    {w, 0.0f, 0.0f, 0.0f},         //
    {0.0f, h, 0.0f, 0.0f},         //
    {0.0f, 0.0f, r, 1.0f},         //
    {0.0f, 0.0f, -r * near, 0.0f}, //
  };
  glm_mat4_copy(mat, *dest);
}

/* -------------------------------------------------------------------------- *
 * Triangle mesh generation
 *
 * @ref
 * https://github.com/michal-z/zig-gamedev/blob/main/libs/zmesh/src/Shape.zig
 * @see https://prideout.net/shapes
 * -------------------------------------------------------------------------- */

typedef struct {
  struct {
    uint16_t* data;
    size_t len;
  } indices;
  struct {
    vec3* data;
    size_t len;
  } positions;
  struct {
    vec3* data;
    size_t len;
  } normals;
  struct {
    vec2* data;
    size_t len;
  } texcoords;
  par_shapes_mesh* handle;
} shape_t;

static void shape_deinit(shape_t* mesh)
{
  if (mesh->indices.len > 0) {
    mesh->indices.data = NULL;
    mesh->indices.len  = 0;
  }
  if (mesh->positions.len > 0) {
    mesh->positions.data = NULL;
    mesh->positions.len  = 0;
  }
  if (mesh->normals.len > 0) {
    mesh->normals.data = NULL;
    mesh->normals.len  = 0;
  }
  if (mesh->texcoords.len > 0) {
    mesh->texcoords.data = NULL;
    mesh->texcoords.len  = 0;
  }
  if (mesh->handle != NULL) {
    par_shapes_free_mesh(mesh->handle);
    mesh->handle = NULL;
  }
}

static shape_t init_shape(par_shapes_mesh* handle)
{
  return (shape_t){
    .indices.data   = (uint16_t*)&(handle->triangles[0]),
    .indices.len    = handle->ntriangles * 3,
    .positions.data = (vec3*)&(handle->points[0]),
    .positions.len  = handle->npoints,
    .normals.data
    = (handle->normals != NULL) ? (vec3*)&(handle->normals[0]) : NULL,
    .normals.len = (handle->normals != NULL) ? handle->npoints : 0,
    .texcoords.data
    = (handle->tcoords != NULL) ? (vec2*)&(handle->tcoords[0]) : NULL,
    .texcoords.len = (handle->tcoords != NULL) ? handle->npoints : 0,
    .handle        = handle,
  };
}

static void shape_rotate(shape_t* mesh, float radians, float x, float y,
                         float z)
{
  par_shapes_rotate(mesh->handle, radians, (float[]){x, y, z});
  *mesh = init_shape(mesh->handle);
}

static void shape_unweld(shape_t* mesh)
{
  par_shapes_unweld(mesh->handle, true);
  *mesh = init_shape(mesh->handle);
}

static void shape_compute_normals(shape_t* mesh)
{
  par_shapes_compute_normals(mesh->handle);
  *mesh = init_shape(mesh->handle);
}

static shape_t init_trefoil_knot(int32_t slices, int32_t stacks, float radius)
{
  return init_shape(par_shapes_create_trefoil_knot(slices, stacks, radius));
}

/* -------------------------------------------------------------------------- *
 * Procedural Mesh Example
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 position;
  vec3 normal;
} vertex_t;

typedef struct {
  mat4 world_to_clip;
  vec3 camera_position;
} frame_uniforms_t;

typedef struct {
  mat4 object_to_world;
  vec4 basecolor_roughness;
} draw_uniforms_t;

typedef struct {
  uint32_t index_offset;
  int32_t vertex_offset;
  uint32_t num_indices;
  uint32_t num_vertices;
} mesh_t;

typedef struct {
  uint32_t mesh_index;
  vec3 position;
  vec4 basecolor_roughness;
} drawable_t;

static struct {
  WGPUBindGroupLayout draw_bind_group_layout;
  WGPUBindGroupLayout frame_bind_group_layout;
  WGPUPipelineLayout pipeline_layout;

  WGPUBindGroup draw_bind_group;
  WGPUBindGroup frame_bind_group;
  WGPURenderPipeline pipeline;

  uint32_t total_num_vertices;
  uint32_t total_num_indices;

  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  struct {
    wgpu_buffer_t frame;
    wgpu_buffer_t draw;
  } uniform_buffers;

  WGPUTexture depth_texture;
  WGPUTextureView depth_texture_view;

  // Render pass descriptor for frame buffer writes
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } render_pass;

  drawable_t drawables[1];
  mesh_t meshes[1];

  struct {
    vec3 position;
    vec3 forward;
    vec3 updir;
    float pitch;
    float yaw;
    mat4 cam_world_to_view;
    mat4 cam_view_to_clip;
    mat4 cam_world_to_clip;
  } camera;

  frame_uniforms_t frame_uniforms;
  draw_uniforms_t draw_uniforms;

  struct {
    float xpos;
    float ypos;
  } mouse;

  // Other variables
  const char* example_title;
  bool prepared;
} demo_state = {
  .camera.position          = {0.0f, 4.0f, -4.0f},
  .camera.forward           = {0.0f, 0.0f, 1.0f},
  .camera.updir             = {0.0f, 1.0f, 0.0f},
  .camera.pitch             = 0.15f * PI,
  .camera.yaw               = 0.0f,
  .camera.cam_world_to_view = GLM_MAT4_ZERO_INIT,
  .camera.cam_view_to_clip  = GLM_MAT4_ZERO_INIT,
  .camera.cam_world_to_clip = GLM_MAT4_ZERO_INIT,

  .mouse.xpos = 0.0f,
  .mouse.ypos = 0.0f,

  .example_title = "Procedural Mesh",
  .prepared      = false,
};

static void append_mesh(uint32_t mesh_index, shape_t* mesh, mesh_t* meshes,
                        uint16_t** meshes_indices, uint32_t* meshes_indices_len,
                        vec3** meshes_positions, uint32_t* meshes_positions_len,
                        vec3** meshes_normals, uint32_t* meshes_normals_len)
{
  meshes[mesh_index] = (mesh_t){
    .index_offset  = *meshes_indices_len,
    .vertex_offset = (int32_t)*meshes_positions_len,
    .num_indices   = mesh->indices.len,
    .num_vertices  = mesh->positions.len,
  };

  // Indices
  *meshes_indices_len += mesh->indices.len;
  uint32_t meshes_indices_size
    = (*meshes_indices_len) * sizeof(**meshes_indices);
  if (*meshes_indices == NULL) {
    *meshes_indices = (uint16_t*)malloc(meshes_indices_size);
  }
  else {
    *meshes_indices = (uint16_t*)realloc(*meshes_indices, meshes_indices_size);
  }
  memcpy(&(*meshes_indices)[*meshes_indices_len - mesh->indices.len],
         mesh->indices.data, mesh->indices.len * sizeof(*mesh->indices.data));

  // Positions
  *meshes_positions_len += mesh->positions.len;
  uint32_t meshes_positions_size
    = (*meshes_positions_len) * sizeof(**meshes_positions);
  if (*meshes_positions == NULL) {
    *meshes_positions = (vec3*)malloc(meshes_positions_size);
  }
  else {
    *meshes_positions
      = (vec3*)realloc(*meshes_positions, meshes_positions_size);
  }
  memcpy(&(*meshes_positions)[*meshes_positions_len - mesh->positions.len],
         mesh->positions.data,
         mesh->positions.len * sizeof(*mesh->positions.data));

  // Normals
  *meshes_normals_len += mesh->normals.len;
  uint32_t meshes_normals_size
    = (*meshes_normals_len) * sizeof(**meshes_normals);
  if (*meshes_normals == NULL) {
    *meshes_normals = (vec3*)malloc(meshes_normals_size);
  }
  else {
    *meshes_normals = (vec3*)realloc(*meshes_normals, meshes_normals_size);
  }
  memcpy(&(*meshes_normals)[*meshes_normals_len - mesh->normals.len],
         mesh->normals.data, mesh->normals.len * sizeof(*mesh->normals.data));
}

static void init_scene(drawable_t* drawables, mesh_t* meshes,
                       uint16_t** meshes_indices, uint32_t* meshes_indices_len,
                       vec3** meshes_positions, uint32_t* meshes_positions_len,
                       vec3** meshes_normals, uint32_t* meshes_normals_len)
{
  uint32_t mesh_index = 0;

  // Trefoil knot
  {
    shape_t mesh = init_trefoil_knot(10, 128, 0.8f);
    shape_rotate(&mesh, PI_2, 1.0, 0.0, 0.0);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {0.f, 1.f, 0.f},
      .basecolor_roughness = {0.0f, 0.7f, 0.0f, 0.6f},
    };

    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&mesh);
  }
}

static void prepare_vertex_and_index_buffer(wgpu_context_t* wgpu_context)
{
  uint16_t* meshes_indices      = NULL;
  uint32_t meshes_indices_len   = 0;
  vec3* meshes_positions        = NULL;
  uint32_t meshes_positions_len = 0;
  vec3* meshes_normals          = NULL;
  uint32_t meshes_normals_len   = 0;
  init_scene(demo_state.drawables, demo_state.meshes, &meshes_indices,
             &meshes_indices_len, &meshes_positions, &meshes_positions_len,
             &meshes_normals, &meshes_normals_len);

  demo_state.total_num_vertices = meshes_positions_len;
  demo_state.total_num_indices  = meshes_indices_len;

  // Create a vertex buffer
  {
    vertex_t* vertex_data
      = (vertex_t*)malloc(demo_state.total_num_vertices * sizeof(vertex_t));
    for (uint32_t i = 0; i < meshes_positions_len; ++i) {
      glm_vec3_copy(meshes_positions[i], vertex_data[i].position);
      glm_vec3_copy(meshes_normals[i], vertex_data[i].normal);
    }
    demo_state.vertex_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = demo_state.total_num_vertices * sizeof(vertex_t),
                      .initial.data = vertex_data,
                    });
    free(vertex_data);
  }

  // Create a index buffer
  demo_state.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = demo_state.total_num_indices * sizeof(uint16_t),
                    .initial.data = meshes_indices,
                  });

  // Cleanup allocated memory
  if (meshes_indices != NULL) {
    free(meshes_indices);
  }
  if (meshes_positions != NULL) {
    free(meshes_positions);
  }
  if (meshes_normals != NULL) {
    free(meshes_normals);
  }
}

static void setup_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Draw bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = sizeof(draw_uniforms_t),
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    demo_state.draw_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(demo_state.draw_bind_group_layout != NULL);
  }

  /* Frame bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize =  sizeof(frame_uniforms_t),
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    demo_state.frame_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(demo_state.frame_bind_group_layout != NULL);
  }
}

static void setup_render_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayout bind_group_layouts[2] = {
    demo_state.frame_bind_group_layout, // Group 1
    demo_state.draw_bind_group_layout,  // Group 0
  };
  demo_state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    });
  ASSERT(demo_state.pipeline_layout != NULL);
}

static void prepare_rendering_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  // Enable depth testing so that the fragment closest to the camera is rendered
  // in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth32Float,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(mesh, sizeof(vertex_t),
                            // Attribute location 0: Position
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                                               offsetof(vertex_t, position)),
                            // Attribute location 1: Normal
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                                               offsetof(vertex_t, normal)))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .wgsl_code.source = vertex_shader_wgsl,
                  .entry = "main",
                },
                .buffer_count = 1,
                .buffers = &mesh_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .wgsl_code.source = fragment_shader_wgsl,
                  .entry = "main",
                },
                .target_count = 1,
                .targets = &color_target_state_desc,
              });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  demo_state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "procedural_mesh_render_pipeline",
                            .layout       = demo_state.pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(demo_state.pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void update_camera(wgpu_context_t* wgpu_context)
{
  look_to_lh(demo_state.camera.position, demo_state.camera.forward,
             demo_state.camera.updir, &demo_state.camera.cam_world_to_view);
  perspective_fov_lh(0.25f * PI,
                     (float)wgpu_context->surface.width
                       / (float)wgpu_context->surface.height,
                     0.01f, 200.0f, &demo_state.camera.cam_view_to_clip);
  glm_mat4_mulN((mat4*[]){&demo_state.camera.cam_world_to_view,
                          &demo_state.camera.cam_view_to_clip},
                2, demo_state.camera.cam_world_to_clip);
}

static void update_frame_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Update camera matrices
  update_camera(wgpu_context);

  // Update "world to clip" (camera) xform
  glm_mat4_transpose_to(demo_state.camera.cam_world_to_clip,
                        demo_state.frame_uniforms.world_to_clip);
  glm_vec3_copy(demo_state.camera.position,
                demo_state.frame_uniforms.camera_position);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(wgpu_context, demo_state.uniform_buffers.frame.buffer,
                          0, &demo_state.frame_uniforms,
                          demo_state.uniform_buffers.frame.size);
}

static void update_draw_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Update "object to world" xform
  vec3* drawable_pos   = &demo_state.drawables[0].position;
  mat4 object_to_world = {
    {1.0f, 0.0f, 0.0f, 0.0f},                                           //
    {0.0f, 1.0f, 0.0f, 0.0f},                                           //
    {0.0f, 0.0f, 1.0f, 0.0f},                                           //
    {(*drawable_pos)[0], (*drawable_pos)[1], (*drawable_pos)[2], 1.0f}, //
  };
  glm_mat4_transpose(object_to_world);

  glm_mat4_copy(object_to_world, demo_state.draw_uniforms.object_to_world);
  glm_vec4_copy(demo_state.drawables[0].basecolor_roughness,
                demo_state.draw_uniforms.basecolor_roughness);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(wgpu_context, demo_state.uniform_buffers.draw.buffer,
                          0, &demo_state.draw_uniforms,
                          demo_state.uniform_buffers.draw.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Create a frame uniform buffer
  demo_state.uniform_buffers.frame = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(frame_uniforms_t),
    });

  // Create a draw uniform buffer
  demo_state.uniform_buffers.draw = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(draw_uniforms_t),
    });

  update_frame_uniform_buffers(context->wgpu_context);
  update_draw_uniform_buffers(context->wgpu_context);
}

static void prepare_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Frame bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = demo_state.uniform_buffers.frame.buffer,
        .offset  = 0,
        .size    = demo_state.uniform_buffers.frame.size,
      },
    };
    demo_state.frame_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = demo_state.frame_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(demo_state.frame_bind_group != NULL);
  }

  /* Draw bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = demo_state.uniform_buffers.draw.buffer,
        .offset  = 0,
        .size    = demo_state.uniform_buffers.draw.size,
      },
    };
    demo_state.draw_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = demo_state.draw_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(demo_state.draw_bind_group != NULL);
  }
}

static void prepare_depth_texture(wgpu_context_t* wgpu_context)
{
  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->surface.width,
    .height             = wgpu_context->surface.height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth32Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  demo_state.depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(demo_state.depth_texture != NULL);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  demo_state.depth_texture_view
    = wgpuTextureCreateView(demo_state.depth_texture, &texture_view_dec);
  ASSERT(demo_state.depth_texture_view != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  demo_state.render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, // Assigned later
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.1f,
        .g = 0.2f,
        .b = 0.3f,
        .a = 1.0f,
      },
  };

  // Depth stencil attachment
  demo_state.render_pass.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view            = demo_state.depth_texture_view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .clearDepth      = 1.0f,
      .depthClearValue = 1.0f,
      .clearStencil    = 0,
    };

  // Render pass descriptor
  demo_state.render_pass.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = demo_state.render_pass.color_attachments,
    .depthStencilAttachment = &demo_state.render_pass.depth_stencil_attachment,
  };
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertex_and_index_buffer(context->wgpu_context);
    setup_bind_group_layouts(context->wgpu_context);
    setup_render_pipeline_layout(context->wgpu_context);
    prepare_rendering_pipeline(context->wgpu_context);
    prepare_uniform_buffers(context);
    prepare_bind_groups(context->wgpu_context);
    prepare_depth_texture(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    demo_state.prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Info")) {
    UNUSED_VAR(context);
    // imgui_overlay_text(context->imgui_overlay,
    //                    "Left Mouse Button + drag  :  rotate camera");
    // imgui_overlay_text(context->imgui_overlay, "W, A, S, D  :  move camera");
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  demo_state.render_pass.color_attachments[0].view
    = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &demo_state.render_pass.descriptor);

  // Bind vertex buffer (contains positions and normals)
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       demo_state.vertex_buffer.buffer, 0,
                                       demo_state.vertex_buffer.size);

  // Bind index buffer
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, demo_state.index_buffer.buffer,
    WGPUIndexFormat_Uint16, 0, demo_state.index_buffer.size);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                   demo_state.pipeline);

  // Set the bind group
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 1,
                                    demo_state.frame_bind_group, 0, 0);

  // Draw indexed geometries
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(demo_state.drawables); ++i) {
    uint32_t dynamic_offset = i * ALIGNMENT;
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      demo_state.draw_bind_group, 0,
                                      &dynamic_offset);
    wgpuRenderPassEncoderDrawIndexed(
      wgpu_context->rpass_enc, demo_state.meshes[i].num_indices, 1,
      demo_state.meshes[i].index_offset, demo_state.meshes[i].vertex_offset, 0);
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
  if (!demo_state.prepared) {
    return 1;
  }
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  WGPU_RELEASE_RESOURCE(Buffer, demo_state.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, demo_state.index_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, demo_state.uniform_buffers.frame.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, demo_state.uniform_buffers.draw.buffer)

  WGPU_RELEASE_RESOURCE(Texture, demo_state.depth_texture)
  WGPU_RELEASE_RESOURCE(TextureView, demo_state.depth_texture_view)

  WGPU_RELEASE_RESOURCE(BindGroup, demo_state.draw_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, demo_state.frame_bind_group)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, demo_state.draw_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, demo_state.frame_bind_group_layout)

  WGPU_RELEASE_RESOURCE(PipelineLayout, demo_state.pipeline_layout)

  WGPU_RELEASE_RESOURCE(RenderPipeline, demo_state.pipeline)
}

void example_procedural_mesh(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title   = demo_state.example_title,
     .overlay = true,
     .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy
  });
  // clang-format on
}
