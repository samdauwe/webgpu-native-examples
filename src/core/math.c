#include "math.h"

#include <stdlib.h>

float rand_float_min_max(float min, float max)
{
  /* [min, max] */
  return ((max - min) * ((float)rand() / (float)RAND_MAX)) + min;
}

float random_float()
{
  return rand_float_min_max(0.0f, 1.0f);
}
