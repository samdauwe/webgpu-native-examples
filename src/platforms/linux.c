#include "../core/platform.h"

#include "../core/macro.h"

#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/time.h>

/* platform initialization */

void initialize_default_path(void)
{
  char path[PATH_SIZE];
  ssize_t bytes;
  int error;

  bytes = readlink("/proc/self/exe", path, PATH_SIZE - 1);
  assert(bytes != -1);
  path[bytes]         = '\0';
  *strrchr(path, '/') = '\0';

  error = chdir(path);
  assert(error == 0);
  error = chdir("assets");
  assert(error == 0);
}

/* misc platform functions */

static double get_native_time(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void get_local_time(date_t* current_date)
{
  struct timeval te;
  gettimeofday(&te, NULL);
  time_t T               = time(NULL);
  struct tm tm           = *localtime(&T);
  long long milliseconds = te.tv_sec * 1000LL + te.tv_usec / 1000;
  current_date->msec     = (int)(milliseconds % (1000));
  current_date->sec      = tm.tm_sec;
  current_date->min      = tm.tm_min;
  current_date->hour     = tm.tm_hour;
  current_date->day      = tm.tm_mday;
  current_date->month    = tm.tm_mon + 1;
  current_date->year     = tm.tm_year + 1900;
  current_date->day_sec  = ((float)current_date->msec) / 1000.0
                          + current_date->sec + current_date->min * 60
                          + current_date->hour * 3600;
  return;
}

float platform_get_time(void)
{
  static double initial = -1.0;
  if (initial < 0.0) {
    initial = get_native_time();
  }
  return (float)(get_native_time() - initial);
}
