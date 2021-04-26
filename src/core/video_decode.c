#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
#include <pthread.h>

/*
 * control play speed.
 *   0.1: 10 times slower
 *   1.0: play on normal speed.
 *   2.0: two times faster
 */
#define PLAY_SPEED (1.0);

static pthread_t s_decode_thread;
static AVFormatContext* s_fmt_ctx;
static AVCodecContext* s_dec_ctx;
static AVStream* s_video_st;
static int s_video_stream_index = -1;
static int s_video_w, s_video_h;
static int s_crop_w, s_crop_h;
static unsigned int s_video_fmt;
static int64_t s_duration_base;

static void* s_decode_buf = NULL;

int init_video_decode()
{
  return 0;
}

int open_video_file(const char* fname)
{
  AVFormatContext* fmt_ctx = NULL;
  AVCodec* dec;
  AVCodecContext* dec_ctx;
  int video_stream_index = -1;
  int ret;

  ret = avformat_open_input(&fmt_ctx, fname, NULL, NULL);
  if (ret < 0) {
    fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    return -1;
  }

  ret = avformat_find_stream_info(fmt_ctx, NULL);
  if (ret < 0) {
    fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    return ret;
  }

  av_dump_format(fmt_ctx, 0, fname, 0);

  /* select the video stream */
  ret = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &dec, 0);
  if (ret < 0) {
    fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    return ret;
  }
  video_stream_index = ret;

  /* create decoding context */
  dec_ctx = avcodec_alloc_context3(dec);
  if (dec_ctx == NULL) {
    fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    return AVERROR(ENOMEM);
  }
  avcodec_parameters_to_context(dec_ctx,
                                fmt_ctx->streams[video_stream_index]->codecpar);

  /* init the video decoder */
  ret = avcodec_open2(dec_ctx, dec, NULL);
  if (ret < 0) {
    fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    return ret;
  }

  s_fmt_ctx            = fmt_ctx;
  s_dec_ctx            = dec_ctx;
  s_video_st           = fmt_ctx->streams[video_stream_index];
  s_video_stream_index = video_stream_index;

  s_video_w   = dec_ctx->width;
  s_video_h   = dec_ctx->height;
  s_video_fmt = s_dec_ctx->pix_fmt;

#if 0
  if (s_video_w > s_video_h) {
    s_crop_w = s_crop_h = s_video_h;
  }
  else {
    s_crop_w = s_crop_h = s_video_w;
  }
#else
  s_crop_w = s_video_w;
  s_crop_h = s_video_h;
#endif

  fprintf(stdout, "-------------------------------------------\n");
  fprintf(stdout, " file  : %s\n", fname);
  fprintf(stdout, " format: %s\n", av_get_pix_fmt_name(s_video_fmt));
  fprintf(stdout, " size  : (%d, %d)\n", s_video_w, s_video_h);
  fprintf(stdout, " crop  : (%d, %d)\n", s_crop_w, s_crop_h);
  fprintf(stdout, "-------------------------------------------\n");

  return 0;
}

int get_video_dimension(int* width, int* height)
{
  *width  = s_crop_w;
  *height = s_crop_h;

  return 0;
}

int get_video_pixformat(int* pixformat)
{
  *pixformat = s_video_fmt;
  return 0;
}

int get_video_buffer(void** buf)
{
  *buf = s_decode_buf;
  return 0;
}

static int save_to_ppm(AVFrame* frame, int width, int height, int icnt)
{
  FILE* fp;
  char fname[64];

  sprintf(fname, "frame%08d.ppm", icnt);
  fp = fopen(fname, "wb");
  if (fp == NULL) {
    fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    return -1;
  }

  fprintf(fp, "P6\n%d %d\n255\n", width, height);
  unsigned char* color = (unsigned char*)malloc(width * 3);
  for (int y = 0; y < height; y++) {
    unsigned char* src8 = frame->data[0] + y * frame->linesize[0];
    for (int x = 0; x < width; x++) {
      color[x * 3 + 0] = *src8++;
      color[x * 3 + 1] = *src8++;
      color[x * 3 + 2] = *src8++;
      src8++;
    }
    fwrite(color, 1, width * 3, fp);
  }
  free(color);

  fclose(fp);

  return 0;
}

static int convert_to_rgba8888(AVFrame* frame, int ofstx, int ofsty, int width,
                               int height)
{
  if (s_decode_buf == NULL) {
    s_decode_buf = (unsigned char*)malloc(width * height * 4);
  }

  if (ofstx == 0 && ofsty == 0) {
    memcpy(s_decode_buf, frame->data[0], width * height * 4);
  }
  else {
    for (int y = 0; y < height; y++) {
      unsigned char* dst8 = s_decode_buf + y * width * 4;
      unsigned char* src8 = frame->data[0] + (y + ofsty) * frame->linesize[0];
      src8 += ofstx * 4;

      for (int x = 0; x < width; x++) {
        *dst8++ = *src8++;
        *dst8++ = *src8++;
        *dst8++ = *src8++;
        *dst8++ = *src8++;
      }
    }
  }
  return 0;
}

static int on_frame_decoded(AVFrame* frame)
{
  int dec_w = s_video_w;
  int dec_h = s_video_h;
  int ofstx = (s_video_w - s_crop_w) * 0.5f;
  int ofsty = (s_video_h - s_crop_h) * 0.5f;

  if (0) {
    static int i = 0;
    save_to_ppm(frame, dec_w, dec_h, i++);
  }

  convert_to_rgba8888(frame, ofstx, ofsty, s_crop_w, s_crop_h);

  return 0;
}

static void init_duration()
{
  s_duration_base = av_gettime();
}

static int64_t get_duration_us()
{
  int64_t duration = av_gettime() - s_duration_base;
  return duration;
}

static void sleep_to_pts(AVPacket* packet)
{
  int64_t pts_us = 0;
  if (packet->dts != AV_NOPTS_VALUE) {
    pts_us = packet->dts;
  }

  pts_us *= (av_q2d(s_video_st->time_base) * 1000 * 1000);
  pts_us /= PLAY_SPEED;

  int64_t delay_us = pts_us - get_duration_us();
  if (delay_us > 0) {
    av_usleep(delay_us);
  }
}
static void* decode_thread_main()
{
  AVFrame* frame    = av_frame_alloc();
  AVFrame* framergb = av_frame_alloc();

  if (frame == NULL || framergb == NULL) {
    fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    return 0;
  }

  int dec_w    = s_dec_ctx->width;
  int dec_h    = s_dec_ctx->height;
  int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGBA, dec_w, dec_h, 1);

  uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

  av_image_fill_arrays(framergb->data, framergb->linesize, buffer,
                       AV_PIX_FMT_RGBA, dec_w, dec_h, 1);

  struct SwsContext* sws_ctx
    = sws_getContext(dec_w, dec_h, s_dec_ctx->pix_fmt, dec_w, dec_h,
                     AV_PIX_FMT_RGBA, SWS_FAST_BILINEAR, NULL, NULL, NULL);
  if (sws_ctx == NULL) {
    fprintf(stderr, "Cannot initialize the sws context\n");
    return 0;
  }

  while (1) {
    AVPacket packet;
    int ret;

    init_duration();

    while (av_read_frame(s_fmt_ctx, &packet) >= 0) {
      if (packet.stream_index == s_video_stream_index) {
        ret = avcodec_send_packet(s_dec_ctx, &packet);
        if (ret < 0) {
          fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
          break;
        }

        while (ret >= 0) {
          ret = avcodec_receive_frame(s_dec_ctx, frame);
          if (ret == AVERROR(EAGAIN)) {
            // fprintf (stderr, "retry.\n");
            break;
          }
          else if (ret == AVERROR_EOF) {
            fprintf(stderr, "EOF.\n");
            break;
          }
          else if (ret < 0) {
            fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return 0;
          }

          sws_scale(sws_ctx, (const uint8_t* const*)frame->data,
                    frame->linesize, 0, dec_h, framergb->data,
                    framergb->linesize);

          sleep_to_pts(&packet);
          on_frame_decoded(framergb);
        }
      }

      av_packet_unref(&packet);
    }

    /* flush decoder */
    ret = avcodec_send_packet(s_dec_ctx, &packet);
    if (ret < 0) {
      fprintf(stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    }

    while (avcodec_receive_frame(s_dec_ctx, frame) == 0) {
      sws_scale(sws_ctx, (const uint8_t* const*)frame->data, frame->linesize, 0,
                dec_h, framergb->data, framergb->linesize);

      sleep_to_pts(&packet);
      on_frame_decoded(framergb);
    }

    /* rewind to restart */
    av_seek_frame(s_fmt_ctx, -1, 0, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(s_dec_ctx);
  }

  av_free(buffer);
  av_frame_free(&framergb);
  av_frame_free(&frame);

  return 0;
}

int start_video_decode()
{
  pthread_create(&s_decode_thread, NULL, decode_thread_main, NULL);
  return 0;
}
