#include "math.h"

#include <math.h>
#include <stdlib.h>

#include "macro.h"

float random_float_min_max(float min, float max)
{
  /* [min, max] */
  return ((max - min) * ((float)rand() / (float)RAND_MAX)) + min;
}

float random_float(void)
{
  return random_float_min_max(0.0f, 1.0f); /* [0, 1.0] */
}

int approx_eq_fabs_eps(float v0, float v1, float epsilon)
{
  return fabs(v1 - v0) < epsilon;
}

int approx_eq_fabs(float v0, float v1)
{
  return approx_eq_fabs_eps(v0, v1, EPSILON);
}

float clamp_float(float d, float min, float max)
{
  const float t = d < min ? min : d;
  return t > max ? max : t;
}
