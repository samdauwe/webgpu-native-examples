#ifndef TEXT_OVERLAY_H
#define TEXT_OVERLAY_H

#include "context.h"

typedef struct text_overlay text_overlay_t;

typedef enum text_overlay_text_align_enum {
  TextOverlay_Text_AlignLeft   = 0,
  TextOverlay_Text_AlignCenter = 1,
  TextOverlay_Text_AlignRight  = 2,
} text_overlay_text_align_enum;

/* text overlay creating/releasing */
text_overlay_t* text_overlay_create(wgpu_context_t* wgpu_context);
void text_overlay_release(text_overlay_t* text_overlay);

/* Prepare for text update */
void text_overlay_begin_text_update(text_overlay_t* text_overlay);
/* Add text to the current buffer */
void text_overlay_add_text(text_overlay_t* text_overlay, const char* text,
                           float x, float y,
                           text_overlay_text_align_enum align);
void text_overlay_add_formatted_text(text_overlay_t* text_overlay, float x,
                                     float y,
                                     text_overlay_text_align_enum align,
                                     const char* format_str, ...);
/* Finish text update */
void text_overlay_end_text_update(text_overlay_t* text_overlay);

/* Draw text data into a command buffer */
void text_overlay_draw_frame(text_overlay_t* text_overlay,
                             WGPUTextureView view);

#endif
