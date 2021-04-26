#ifndef MACRO_H
#define MACRO_H

#include <assert.h>
#include <stdio.h>

#define EPSILON 1e-5f              /* epsilon */
#define PI 3.14159265358979323846f /* pi */

#define TO_RADIANS(degrees) ((PI / 180) * (degrees))
#define TO_DEGREES(radians) ((180 / PI) * (radians))

#define LINE_SIZE 256
#define PATH_SIZE 256

#define STRMAX 512

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define UNUSED_VAR(x) ((void)(x))
#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))

#define GET_DEFAULT_IF_ZERO(value, default_value) (value != NULL) ? value : default_value

#ifndef NDEBUG
#define ASSERT(expression)                                                     \
  {                                                                            \
    if (!(expression)) {                                                       \
      printf("Assertion(%s) failed: file \"%s\", line %d\n", #expression,      \
             __FILE__, __LINE__);                                              \
    }                                                                          \
  }
#else
#define ASSERT(expression) NULL;
#endif

#endif
