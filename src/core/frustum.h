#ifndef FRUSTUM_H
#define FRUSTUM_H

#include <cglm/cglm.h>

typedef enum {
  Frustum_Side_Left   = 0,
  Frustum_Side_Right  = 1,
  Frustum_Side_Top    = 2,
  Frustum_Side_Bottom = 3,
  Frustum_Side_Back   = 4,
  Frustum_Side_Front  = 5
} frustum_side_enum;

/**
 * @brief View frustum culling class
 */
typedef struct frustum_t {
  vec4 planes[6];
} frustum_t;

/* frustum creating/releasing */
frustum_t* frustum_create(void);
void frustum_release(frustum_t* frustum);

/* frustum updating */
void frustum_update(frustum_t* frustum, mat4 matrix);

/* frustum checking */
bool frustum_check_sphere(frustum_t* frustum, vec3 pos, float radius);

#endif
