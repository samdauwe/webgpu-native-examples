/*
 * WAjic-compatible sokol_time shim.
 *
 * Provides the subset of the sokol_time API used by the webgpu-native-examples:
 *   stm_setup()                — initialise the timer
 *   stm_now()                  — return current time in nanoseconds (uint64_t)
 *   stm_sec(uint64_t)          — convert nanoseconds to seconds (double)
 *   stm_ms(uint64_t)           — convert nanoseconds to milliseconds (double)
 *   stm_diff(new, old)         — difference in nanoseconds
 *   stm_laptime(uint64_t* last) — elapsed since last call, updates *last
 *
 * Timing source: performance.now() (sub-millisecond, monotonic).
 *
 * Usage — identical to sokol_time.h:
 *   #define WAJIC_TIME_IMPL   (in exactly one .c file)
 *   #include "wajic_time.h"
 */

#ifndef WAJIC_TIME_H
#define WAJIC_TIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void     stm_setup(void);
uint64_t stm_now(void);
uint64_t stm_diff(uint64_t new_ticks, uint64_t old_ticks);
double   stm_sec(uint64_t ticks);
double   stm_ms(uint64_t ticks);

/* stm_laptime: returns nanoseconds elapsed since *last_time and updates it.
 * Matches the sokol_time stm_laptime() signature exactly. */
static inline uint64_t stm_laptime(uint64_t* last_time)
{
  uint64_t dt  = 0;
  uint64_t now = stm_now();
  if (*last_time != 0) {
    dt = stm_diff(now, *last_time);
  }
  *last_time = now;
  return dt;
}

/* stm_since: nanoseconds elapsed since start_ticks (convenience wrapper). */
static inline uint64_t stm_since(uint64_t start_ticks)
{
  return stm_diff(stm_now(), start_ticks);
}

#ifdef __cplusplus
}
#endif

/* ========================================================================== */
#ifdef WAJIC_TIME_IMPL
/* ========================================================================== */

#include <wajic.h>

/* Returns performance.now() as a double (milliseconds since page load). */
WAJIC(double, _wajic_time_now_ms, (void),
{
    return performance.now();
})

static double _wajic_time_start_ms;

void stm_setup(void)
{
  _wajic_time_start_ms = _wajic_time_now_ms();
}

uint64_t stm_now(void)
{
  double elapsed_ms = _wajic_time_now_ms() - _wajic_time_start_ms;
  /* Convert milliseconds → nanoseconds (sokol_time's native unit). */
  return (uint64_t)(elapsed_ms * 1000000.0);
}

uint64_t stm_diff(uint64_t new_ticks, uint64_t old_ticks)
{
  return (new_ticks > old_ticks) ? (new_ticks - old_ticks) : 1;
}

double stm_sec(uint64_t ticks)
{
  return (double)ticks / 1000000000.0;
}

double stm_ms(uint64_t ticks)
{
  return (double)ticks / 1000000.0;
}

#endif /* WAJIC_TIME_IMPL */
#endif /* WAJIC_TIME_H */
