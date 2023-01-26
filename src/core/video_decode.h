#ifndef VIDEO_DECODE_H
#define VIDEO_DECODE_H

#include <stdint.h>

int init_video_decode(void);
int open_video_file(const char* fname);
int get_video_dimension(int* width, int* height);
int get_video_pixformat(uint32_t* pixformat);
int get_video_buffer(void** buf);

int start_video_decode(void);

#endif
