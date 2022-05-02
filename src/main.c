#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/api.h"
#include "core/argparse.h"
#include "examples/examples.h"

int main(int argc, char* argv[])
{
  srand((unsigned int)time(NULL));
  initialize_default_path();

  const char* example_name = NULL;
  int demo_mode = 0, window_width = 0, window_height = 0;
  struct argparse_option options[] = {
    OPT_BOOLEAN('?', "help", NULL, "show this help message and exit",
                argparse_help_cb, 0, OPT_NONEG),
    OPT_GROUP("Options"),
    OPT_STRING('s', "sample", &example_name, "sample to launch", NULL, 0, 0),
    OPT_INTEGER('w', "width", &window_width, "window width", NULL, 0, 0),
    OPT_INTEGER('h', "height", &window_height, "window height", NULL, 0, 0),
    OPT_BOOLEAN('d', "demo-mode", &demo_mode,
                "demo mode, this mode runs every example for 10 seconds", NULL,
                0, 0),
    OPT_END(),
  };

  struct argparse argparse;
  const char* const usages[] = {
    "wgpu_sample_launcher [options]",
    NULL,
  };
  argparse_init(&argparse, options, usages, 0);
  argparse_describe(
    &argparse, "\nWebGPU Native examples and demos launcher.",
    "\nThis command-line application launches WebGPU Native examples and "
    "provides several options to configure their run-time behavior.");
  char** argv_cpy   = argv_copy(argc, argv);
  int argparse_argc = argparse_parse(&argparse, argc, (const char**)argv_cpy);
  free(argv_cpy);

  if (argc == 0) {
    examplecase_t* example = get_random_example();
    printf("Randomly selected example: %s\n", example->example_name);
    example->example_func(argc, argv);
  }
  if (example_name != NULL) {
    examplecase_t* example = get_example_by_name(example_name);
    if (example == NULL) {
      fprintf(stderr, "Example not found: %s\n", example_name);
      log_example_names();
      return EXIT_FAILURE;
    }
    else {
      printf("Running example: %s\n", example->example_name);
      example->example_func(argc, argv);
    }
  }
  if (demo_mode != 0) {
    examplecase_t* examples = get_examples();
    uint32_t example_count  = get_number_of_examples();
    printf("Running demo mode, found %d examples\n", example_count);
    for (uint32_t i = 0; i < example_count; ++i) {
      printf("Running example: %s\n", examples[i].example_name);
    }
  }
  if (argparse_argc != 0) {
    printf("argc: %d\n", argparse_argc);
    for (int32_t i = 0; i < argparse_argc; ++i) {
      printf("argv[%d]: %s\n", i, *(argv + i));
    }
  }

  return EXIT_SUCCESS;
}
