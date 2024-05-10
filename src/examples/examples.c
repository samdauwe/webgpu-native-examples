#include "examples.h"

#include <stdlib.h>
#include <string.h>

#include "../core/macro.h"

void example_a_buffer(int argc, char* argv[]);
void example_animometer(int argc, char* argv[]);
void example_aquarium(int argc, char* argv[]);
void example_basisu(int argc, char* argv[]);
void example_bind_groups(int argc, char* argv[]);
void example_blinn_phong_lighting(int argc, char* argv[]);
void example_bloom(int argc, char* argv[]);
void example_cameras(int argc, char* argv[]);
void example_clear_screen(int argc, char* argv[]);
void example_compute_boids(int argc, char* argv[]);
void example_compute_metaballs(int argc, char* argv[]);
void example_compute_n_body(int argc, char* argv[]);
void example_compute_particles(int argc, char* argv[]);
void example_compute_particles_easing(int argc, char* argv[]);
void example_compute_particles_webgpu_logo(int argc, char* argv[]);
void example_compute_ray_tracing(int argc, char* argv[]);
void example_compute_shader(int argc, char* argv[]);
void example_conservative_raster(int argc, char* argv[]);
void example_conway(int argc, char* argv[]);
void example_conway_paletted_blurring(int argc, char* argv[]);
void example_coordinate_system(int argc, char* argv[]);
void example_cornell_box(int argc, char* argv[]);
void example_cube_reflection(int argc, char* argv[]);
void example_cubemap(int argc, char* argv[]);
void example_deferred_rendering(int argc, char* argv[]);
void example_dynamic_uniform_buffer(int argc, char* argv[]);
void example_equirectangular_image(int argc, char* argv[]);
void example_fluid_simulation(int argc, char* argv[]);
void example_game_of_life(int argc, char* argv[]);
void example_gears(int argc, char* argv[]);
void example_gerstner_waves(int argc, char* argv[]);
void example_gltf_loading(int argc, char* argv[]);
void example_gltf_scene_rendering(int argc, char* argv[]);
void example_gltf_skinning(int argc, char* argv[]);
void example_hdr(int argc, char* argv[]);
void example_image_blur(int argc, char* argv[]);
void example_imgui_overlay(int argc, char* argv[]);
void example_immersive_video(int argc, char* argv[]);
void example_instanced_cube(int argc, char* argv[]);
void example_minimal(int argc, char* argv[]);
void example_msaa_line(int argc, char* argv[]);
void example_multi_sampling(int argc, char* argv[]);
void example_n_body_simulation(int argc, char* argv[]);
void example_normal_map(int argc, char* argv[]);
void example_normal_mapping(int argc, char* argv[]);
void example_occlusion_query(int argc, char* argv[]);
void example_offscreen_rendering(int argc, char* argv[]);
void example_out_of_bounds_viewport(int argc, char* argv[]);
void example_parallax_mapping(int argc, char* argv[]);
void example_pbr_basic(int argc, char* argv[]);
void example_pbr_ibl(int argc, char* argv[]);
void example_pbr_texture(int argc, char* argv[]);
void example_points(int argc, char* argv[]);
void example_post_processing(int argc, char* argv[]);
void example_pristine_grid(int argc, char* argv[]);
void example_prng(int argc, char* argv[]);
void example_procedural_mesh(int argc, char* argv[]);
void example_radial_blur(int argc, char* argv[]);
void example_render_bundles(int argc, char* argv[]);
void example_reversed_z(int argc, char* argv[]);
void example_sampler_parameters(int argc, char* argv[]);
void example_screenshot(int argc, char* argv[]);
void example_shadertoy(int argc, char* argv[]);
void example_shadow_mapping(int argc, char* argv[]);
void example_square(int argc, char* argv[]);
void example_stencil_buffer(int argc, char* argv[]);
void example_terrain_mesh(int argc, char* argv[]);
void example_text_overlay(int argc, char* argv[]);
void example_texture_3d(int argc, char* argv[]);
void example_texture_cubemap(int argc, char* argv[]);
void example_texture_mipmap_gen(int argc, char* argv[]);
void example_textured_cube(int argc, char* argv[]);
void example_textured_quad(int argc, char* argv[]);
void example_tile_map(int argc, char* argv[]);
void example_triangle(int argc, char* argv[]);
void example_two_cubes(int argc, char* argv[]);
void example_vertex_buffer(int argc, char* argv[]);
void example_video_uploading(int argc, char* argv[]);
void example_volume_rendering_texture_3d(int argc, char* argv[]);
void example_wireframe_vertex_pulling(int argc, char* argv[]);

static examplecase_t g_example_cases[] = {
  {"a_buffer", example_a_buffer},
  {"animometer", example_animometer},
  // {"aquarium", example_aquarium},
  {"basisu", example_basisu},
  {"bind_groups", example_bind_groups},
  {"blinn_phong_lighting", example_blinn_phong_lighting},
  {"bloom", example_bloom},
  {"cameras", example_cameras},
  {"clear_screen", example_clear_screen},
  {"compute_boids", example_compute_boids},
  {"compute_metaballs", example_compute_metaballs},
  {"compute_particles", example_compute_particles},
  {"compute_particles_easing", example_compute_particles_easing},
  {"compute_particles_webgpu_logo", example_compute_particles_webgpu_logo},
  {"compute_ray_tracing", example_compute_ray_tracing},
  {"compute_shader", example_compute_shader},
  {"conservative_raster", example_conservative_raster},
  {"conway", example_conway},
  {"conway_paletted_blurring", example_conway_paletted_blurring},
  {"coordinate_system", example_coordinate_system},
  {"cornell_box", example_cornell_box},
  {"cube_reflection", example_cube_reflection},
  {"cubemap", example_cubemap},
  {"deferred_rendering", example_deferred_rendering},
  {"dynamic_uniform_buffer", example_dynamic_uniform_buffer},
  {"equirectangular_image", example_equirectangular_image},
  {"fluid_simulation", example_fluid_simulation},
  {"game_of_life", example_game_of_life},
  {"gears", example_gears},
  {"gerstner_waves", example_gerstner_waves},
  {"gltf_loading", example_gltf_loading},
  {"gltf_scene_rendering", example_gltf_scene_rendering},
  {"gltf_skinning", example_gltf_skinning},
  {"hdr", example_hdr},
  {"image_blur", example_image_blur},
  {"imgui_overlay", example_imgui_overlay},
  {"immersive_video", example_immersive_video},
  {"instanced_cube", example_instanced_cube},
  {"minimal", example_minimal},
  {"msaa_line", example_msaa_line},
  {"multi_sampling", example_multi_sampling},
  {"n_body_simulation", example_n_body_simulation},
  {"normal_map", example_normal_map},
  {"normal_mapping", example_normal_mapping},
  {"occlusion_query", example_occlusion_query},
  {"offscreen_rendering", example_offscreen_rendering},
  {"out_of_bounds_viewport", example_out_of_bounds_viewport},
  {"parallax_mapping", example_parallax_mapping},
  {"pbr_basic", example_pbr_basic},
  {"pbr_ibl", example_pbr_ibl},
  {"pbr_texture", example_pbr_texture},
  {"points", example_points},
  {"post_processing", example_post_processing},
  {"pristine_grid", example_pristine_grid},
  {"prng", example_prng},
  {"procedural_mesh", example_procedural_mesh},
  {"radial_blur", example_radial_blur},
  {"render_bundles", example_render_bundles},
  {"reversed_z", example_reversed_z},
  {"sampler_parameters", example_sampler_parameters},
  {"screenshot", example_screenshot},
  {"shadertoy", example_shadertoy},
  {"shadow_mapping", example_shadow_mapping},
  {"square", example_square},
  {"stencil_buffer", example_stencil_buffer},
  {"terrain_mesh", example_terrain_mesh},
  {"text_overlay", example_text_overlay},
  {"texture_3d", example_texture_3d},
  {"texture_cubemap", example_texture_cubemap},
  {"texture_mipmap_gen", example_texture_mipmap_gen},
  {"textured_cube", example_textured_cube},
  {"textured_quad", example_textured_quad},
  {"tile_map", example_tile_map},
  {"triangle", example_triangle},
  {"two_cubes", example_two_cubes},
  {"vertex_buffer", example_vertex_buffer},
  {"video_uploading", example_video_uploading},
  {"volume_rendering_texture_3d", example_volume_rendering_texture_3d},
  {"wireframe_vertex_pulling", example_wireframe_vertex_pulling},
};

int get_number_of_examples(void)
{
  return ARRAY_SIZE(g_example_cases);
}

examplecase_t* get_examples(void)
{
  return g_example_cases;
}

examplecase_t* get_example_by_name(const char* example_name)
{
  examplecase_t* example     = NULL;
  const int num_examplecases = get_number_of_examples();
  for (int i = 0; i < num_examplecases; i++) {
    if (strcmp(g_example_cases[i].example_name, example_name) == 0) {
      example = &g_example_cases[i];
      break;
    }
  }
  return example;
}

examplecase_t* get_random_example(void)
{
  const int i = rand() % get_number_of_examples();
  return &g_example_cases[i];
}

void log_example_names(void)
{
  const int num_examplecases = get_number_of_examples();
  printf("Available examples (%d):\n", num_examplecases);
  for (int i = 0; i < num_examplecases; i++) {
    printf("  |- %s\n", g_example_cases[i].example_name);
  }
}
