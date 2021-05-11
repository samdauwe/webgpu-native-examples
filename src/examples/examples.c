#include "examples.h"

#include <stdlib.h>
#include <string.h>

#include "../core/log.h"
#include "../core/macro.h"

void example_animometer(int argc, char* argv[]);
void example_clear_screen(int argc, char* argv[]);
void example_compute_boids(int argc, char* argv[]);
void example_compute_n_body(int argc, char* argv[]);
void example_compute_particles(int argc, char* argv[]);
void example_compute_ray_tracing(int argc, char* argv[]);
void example_compute_shader(int argc, char* argv[]);
void example_coordinate_system(int argc, char* argv[]);
void example_cube_reflection(int argc, char* argv[]);
void example_deferred_rendering(int argc, char* argv[]);
void example_dynamic_uniform_buffer(int argc, char* argv[]);
void example_gears(int argc, char* argv[]);
void example_gltf_loading(int argc, char* argv[]);
void example_image_blur(int argc, char* argv[]);
void example_imgui_overlay(int argc, char* argv[]);
void example_instanced_cube(int argc, char* argv[]);
void example_msaa_line(int argc, char* argv[]);
void example_parallax_mapping(int argc, char* argv[]);
void example_reversed_z(int argc, char* argv[]);
void example_shadertoy(int argc, char* argv[]);
void example_shadow_mapping(int argc, char* argv[]);
void example_skybox(int argc, char* argv[]);
void example_textured_cube(int argc, char* argv[]);
void example_textured_quad(int argc, char* argv[]);
void example_triangle(int argc, char* argv[]);
void example_two_cubes(int argc, char* argv[]);
void example_video_uploading(int argc, char* argv[]);

static examplecase_t g_example_cases[] = {
  {"animometer", example_animometer},
  {"clear_screen", example_clear_screen},
  {"compute_boids", example_compute_boids},
  {"compute_n_body", example_compute_n_body},
  {"compute_particles", example_compute_particles},
  {"compute_ray_tracing", example_compute_ray_tracing},
  {"compute_shader", example_compute_shader},
  {"coordinate_system", example_coordinate_system},
  {"cube_reflection", example_cube_reflection},
  {"deferred_rendering", example_deferred_rendering},
  {"dynamic_uniform_buffer", example_dynamic_uniform_buffer},
  {"gears", example_gears},
  {"gltf_loading", example_gltf_loading},
  {"image_blur", example_image_blur},
  {"imgui_overlay", example_imgui_overlay},
  {"instanced_cube", example_instanced_cube},
  {"msaa_line", example_msaa_line},
  {"parallax_mapping", example_parallax_mapping},
  {"reversed_z", example_reversed_z},
  {"shadertoy", example_shadertoy},
  {"shadow_mapping", example_shadow_mapping},
  {"skybox", example_skybox},
  {"textured_cube", example_textured_cube},
  {"textured_quad", example_textured_quad},
  {"triangle", example_triangle},
  {"two_cubes", example_two_cubes},
  {"video_uploading", example_video_uploading},
};

static int get_number_of_examples()
{
  return ARRAY_SIZE(g_example_cases);
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

examplecase_t* get_random_example()
{
  const int i = rand() % get_number_of_examples();
  return &g_example_cases[i];
}

void log_example_names()
{
  const int num_examplecases = get_number_of_examples();
  log_info("available examples: ");
  for (int i = 0; i < num_examplecases; i++) {
    if (i != num_examplecases - 1) {
      log_info("%s, ", g_example_cases[i].example_name);
    }
    else {
      log_info("%s\n", g_example_cases[i].example_name);
    }
  }
}
