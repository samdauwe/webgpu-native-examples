#ifndef IMGUI_OVERLAY_H
#define IMGUI_OVERLAY_H

#include "context.h"

#include "../examples/example_base.h"

struct ImDrawData;
typedef struct imgui_overlay imgui_overlay_t;

/* imgui overlay creating/releasing */
imgui_overlay_t* imgui_overlay_create(wgpu_context_t* wgpu_context,
                                      WGPUTextureFormat format);
void imgui_overlay_release(imgui_overlay_t* imgui_overlay);

/* Property getters / setters */
float imgui_overlay_get_scale(imgui_overlay_t* imgui_overlay);

/* imgui overlay rendering */
void imgui_overlay_new_frame(imgui_overlay_t* imgui_overlay,
                             wgpu_example_context_t* context);
void imgui_overlay_render(imgui_overlay_t* imgui_overlay);
void imgui_overlay_draw_frame(imgui_overlay_t* imgui_overlay,
                              WGPUTextureView view);
bool imgui_overlay_want_capture_mouse(void);

bool imgui_overlay_header(const char* caption);
bool imgui_overlay_checkBox(imgui_overlay_t* imgui_overlay, const char* caption,
                            bool* value);
bool imgui_overlay_input_float(imgui_overlay_t* imgui_overlay,
                               const char* caption, float* value, float step,
                               const char* format);
bool imgui_overlay_slider_float(imgui_overlay_t* imgui_overlay,
                                const char* caption, float* value, float min,
                                float max, const char* format);
bool imgui_overlay_slider_int(imgui_overlay_t* imgui_overlay,
                              const char* caption, int32_t* value, int32_t min,
                              int32_t max);
bool imgui_overlay_combo_box(imgui_overlay_t* imgui_overlay,
                             const char* caption, int32_t* item_index,
                             const char** items, uint32_t item_count);
bool imgui_overlay_button(imgui_overlay_t* imgui_overlay, const char* caption);
void imgui_overlay_text(const char* format_str, ...);
bool imgui_overlay_color_edit4(imgui_overlay_t* imgui_overlay,
                               const char* caption, float color[4]);

#endif
