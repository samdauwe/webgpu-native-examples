#include "frustum.h"

#include <string.h>

#include "macro.h"

/* frustum creating/releasing */

frustum_t* frustum_create(void)
{
  frustum_t* frustum = (frustum_t*)malloc(sizeof(frustum_t));
  memset(frustum, 0, sizeof(frustum_t));

  return frustum;
}

void frustum_release(frustum_t* frustum)
{
  free(frustum);
}

/* frustum updating */

void frustum_update(frustum_t* frustum, mat4 matrix)
{
  frustum->planes[Frustum_Side_Left][0] = matrix[0][3] + matrix[0][0];
  frustum->planes[Frustum_Side_Left][1] = matrix[1][3] + matrix[1][0];
  frustum->planes[Frustum_Side_Left][2] = matrix[2][3] + matrix[2][0];
  frustum->planes[Frustum_Side_Left][3] = matrix[3][3] + matrix[3][0];

  frustum->planes[Frustum_Side_Right][0] = matrix[0][3] - matrix[0][0];
  frustum->planes[Frustum_Side_Right][1] = matrix[1][3] - matrix[1][0];
  frustum->planes[Frustum_Side_Right][2] = matrix[2][3] - matrix[2][0];
  frustum->planes[Frustum_Side_Right][3] = matrix[3][3] - matrix[3][0];

  frustum->planes[Frustum_Side_Top][0] = matrix[0][3] - matrix[0][1];
  frustum->planes[Frustum_Side_Top][1] = matrix[1][3] - matrix[1][1];
  frustum->planes[Frustum_Side_Top][2] = matrix[2][3] - matrix[2][1];
  frustum->planes[Frustum_Side_Top][3] = matrix[3][3] - matrix[3][1];

  frustum->planes[Frustum_Side_Bottom][0] = matrix[0][3] + matrix[0][1];
  frustum->planes[Frustum_Side_Bottom][1] = matrix[1][3] + matrix[1][1];
  frustum->planes[Frustum_Side_Bottom][2] = matrix[2][3] + matrix[2][1];
  frustum->planes[Frustum_Side_Bottom][3] = matrix[3][3] + matrix[3][1];

  frustum->planes[Frustum_Side_Back][0] = matrix[0][3] + matrix[0][2];
  frustum->planes[Frustum_Side_Back][1] = matrix[1][3] + matrix[1][2];
  frustum->planes[Frustum_Side_Back][2] = matrix[2][3] + matrix[2][2];
  frustum->planes[Frustum_Side_Back][3] = matrix[3][3] + matrix[3][2];

  frustum->planes[Frustum_Side_Front][0] = matrix[0][3] - matrix[0][2];
  frustum->planes[Frustum_Side_Front][1] = matrix[1][3] - matrix[1][2];
  frustum->planes[Frustum_Side_Front][2] = matrix[2][3] - matrix[2][2];
  frustum->planes[Frustum_Side_Front][3] = matrix[3][3] - matrix[3][2];

  float length = 0.f;
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(frustum->planes); ++i) {
    length = sqrtf(frustum->planes[i][0] * frustum->planes[i][0]
                   + frustum->planes[i][1] * frustum->planes[i][1]
                   + frustum->planes[i][2] * frustum->planes[i][2]);
    glm_vec4_scale(frustum->planes[i], 1.0f / length, frustum->planes[i]);
  }
}

/* frustum checking */

bool frustum_check_sphere(frustum_t* frustum, vec3 pos, float radius)
{
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(frustum->planes); ++i) {
    if ((frustum->planes[i][0] * pos[0]) + (frustum->planes[i][1] * pos[1])
          + (frustum->planes[i][2] * pos[2]) + frustum->planes[i][3]
        <= -radius) {
      return false;
    }
  }
  return true;
}
