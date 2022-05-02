#ifndef EXAMPLES_H
#define EXAMPLES_H

typedef void examplefunc_t(int argc, char* argv[]);
typedef struct {
  const char* example_name;
  examplefunc_t* example_func;
} examplecase_t;

int get_number_of_examples();
examplecase_t* get_examples();
examplecase_t* get_random_example();
examplecase_t* get_example_by_name(const char* example_name);
void log_example_names();

#endif /* EXAMPLES_H */
