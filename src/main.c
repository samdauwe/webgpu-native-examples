#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "core/api.h"
#include "examples/examples.h"

int main(int argc, char* argv[])
{
  examplecase_t* example = NULL;

  srand((unsigned int)time(NULL));
  initialize_default_path();

  const char* example_name = argv[1];
  example = argc > 1 ? get_example_by_name(example_name) : get_random_example();

  if (example) {
    log_info("Running example: %s\n", example->example_name);
    example->example_func(argc, argv);
  }
  else {
    log_error("Example not found: %s\n", example_name);
    log_example_names();
  }

  return EXIT_SUCCESS;
}
