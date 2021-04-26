#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>

#include "input.h"

typedef struct window_config_t {
  uint32_t width;
  uint32_t height;
} window_config_t;

typedef struct window window_t;
typedef struct {
  void (*key_callback)(window_t* window, keycode_t key, int pressed);
  void (*button_callback)(window_t* window, button_t button, int pressed);
  void (*scroll_callback)(window_t* window, float offset);
} callbacks_t;

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
void platform_initialize(void);
void platform_terminate(void);

/* window related functions */
window_t* window_create(const char* title, uint32_t width, uint32_t height,
                        int resizable);
void window_destroy(window_t* window);
int window_should_close(window_t* window);
void window_set_title(window_t* window, const char* title);
void window_set_userdata(window_t* window, void* userdata);
void* window_get_userdata(window_t* window);
void* window_get_surface(void* device, window_t* window);
void window_get_size(window_t* window, uint32_t* width, uint32_t* height);
void window_get_aspect_ratio(window_t* window, float* aspect_ratio);

/* input related functions */
void input_poll_events(void);
int input_key_pressed(window_t* window, keycode_t key);
int input_button_pressed(window_t* window, button_t button);
void input_query_cursor(window_t* window, float* xpos, float* ypos);
void input_set_callbacks(window_t* window, callbacks_t callbacks);

/* misc platform functions */
void get_local_time(date_t* current_date);
float platform_get_time(void);

#endif
