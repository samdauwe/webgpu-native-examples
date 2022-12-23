#ifndef MACRO_H
#define MACRO_H

#include <assert.h>
#include <stdio.h>

/* Define bool, false, true if not defined */
#ifndef __bool_true_false_are_defined
#define bool int
#define false 0
#define true 1
#define size_t uint64_t
#endif

/* Define NULL if not defined */
#ifndef NULL
#define NULL ((void*)0)
#endif

/* Constants */
#define E 2.71828182845904523536f        /* Euler's constant e */
#define LOG2E 1.44269504088896340736f    /* log2(e) */
#define LOG10E 0.434294481903251827651f  /* log10(e) */
#define LN2 0.693147180559945309417f     /* ln(2) */
#define LN10 2.30258509299404568402f     /* ln(10) */
#define PI 3.14159265358979323846f       /* pi */
#define PI2 6.28318530717958647692f      /* pi * 2 */
#define PI_2 1.57079632679489661923f     /* pi/2 */
#define PI_4 0.785398163397448309616f    /* pi/4 */
#define SQRT2 1.41421356237309504880f    /* sqrt(2) */
#define SQRT2_2 0.707106781186547524401f /* sqrt(2)/2 */

#define THOUSAND 1000L
#define MILLION 1000000L
#define BILLION 1000000000L

/* Constant used to define the minimal number value */
#define EPSILON 1e-5f /* epsilon */

#define TO_RADIANS(degrees) ((PI / 180) * (degrees))
#define TO_DEGREES(radians) ((180 / PI) * (radians))

#define LINE_SIZE 256
#define PATH_SIZE 256

#define STRMAX 512

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLAMP(x, lo, hi) (MIN(hi, MAX(lo, x)))

#define UNUSED_VAR(x) ((void)(x))
#define UNUSED_FUNCTION(x) ((void)(x))
#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))

#define CODE(...) #__VA_ARGS__

#define GET_DEFAULT_IF_ZERO(value, default_value)                              \
  (value != NULL) ? value : default_value

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
