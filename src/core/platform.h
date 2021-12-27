#ifndef PLATFORM_H
#define PLATFORM_H

/* date class */
typedef struct date_t {
  int msec;
  int sec;
  float day_sec;
  int min;
  int hour;
  int day;
  int month;
  int year;
} date_t;

/* platform initialization */
void initialize_default_path(void);

/* misc platform functions */
void get_local_time(date_t* current_date);
float platform_get_time(void);

#endif
