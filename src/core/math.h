#ifndef MATH_H
#define MATH_H

/**
 * @brief Generates a random float number in range [min, max].
 * @param min minimum number
 * @param max maximum number
 * @return random float number in range [min, max]
 */
float random_float_min_max(float min, float max);

/**
 * @brief Generates a random float number in range [0.0f, 1.0f].
 * @return random float number in range [0.0f, 1.0f]
 */
float random_float(void);

/**
 * @brief Returns if value v0 and v1 are approximately equal using epsilon as
 * allowed error.
 * @return if value v0 and v1 are approximately equal
 */
int approx_eq_fabs_eps(float v0, float v1, float epsilon);
int approx_eq_fabs(float v0, float v1);

/**
 * @brief Clamps a float value between minimum and maximum.
 * @param d the number to clamp
 * @param min the minimum value
 * @param max the maximum value
 * @return the clamped value in range [min, max]
 */
float clamp_float(float d, float min, float max);

#endif /* MATH_H */
