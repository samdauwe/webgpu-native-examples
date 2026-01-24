#ifndef IMGUI_OVERLAY_H
#define IMGUI_OVERLAY_H

/**
 * @file imgui_overlay.h
 * @brief Modular ImGui overlay implementation for WebGPU examples
 *
 * A clean, efficient, and modular ImGui overlay that can be easily integrated
 * into any WebGPU example.
 */

#include "wgpu_common.h"

#include <stdbool.h>

/* Forward declarations */
struct ImGuiContext;

/* -------------------------------------------------------------------------- *
 * ImGui Overlay API
 * -------------------------------------------------------------------------- */

/**
 * @brief Initialize ImGui overlay
 * @param wgpu_context WebGPU context
 * @return 0 on success, non-zero on failure
 */
int imgui_overlay_init(wgpu_context_t* wgpu_context);

/**
 * @brief Start a new ImGui frame
 * @param wgpu_context WebGPU context
 * @param delta_time Time elapsed since last frame (in seconds)
 */
void imgui_overlay_new_frame(wgpu_context_t* wgpu_context, float delta_time);

/**
 * @brief Render ImGui draw data
 *
 * This should be called after your main scene rendering to overlay the GUI.
 * It creates its own render pass that loads (preserves) the existing content.
 *
 * @param wgpu_context WebGPU context
 */
void imgui_overlay_render(wgpu_context_t* wgpu_context);

/**
 * @brief Handle input events for ImGui
 * @param wgpu_context WebGPU context
 * @param event Input event from the framework
 */
void imgui_overlay_handle_input(wgpu_context_t* wgpu_context,
                                const input_event_t* event);

/**
 * @brief Shutdown and cleanup ImGui overlay
 */
void imgui_overlay_shutdown(void);

/**
 * @brief Check if ImGui wants to capture mouse input
 * @return true if ImGui wants mouse input
 */
bool imgui_overlay_want_capture_mouse(void);

/**
 * @brief Check if ImGui wants to capture keyboard input
 * @return true if ImGui wants keyboard input
 */
bool imgui_overlay_want_capture_keyboard(void);

/* -------------------------------------------------------------------------- *
 * Convenience Widget Functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Create a collapsible header
 * @param caption Header text
 * @return true if header is open
 */
bool imgui_overlay_header(const char* caption);

/**
 * @brief Create a checkbox widget
 * @param caption Label text
 * @param value Pointer to boolean value
 * @return true if value changed
 */
bool imgui_overlay_checkbox(const char* caption, bool* value);

/**
 * @brief Create a slider for float values
 * @param caption Label text
 * @param value Pointer to float value
 * @param min Minimum value
 * @param max Maximum value
 * @param format Printf format string (e.g., "%.3f")
 * @return true if value changed
 */
bool imgui_overlay_slider_float(const char* caption, float* value, float min,
                                float max, const char* format);

/**
 * @brief Create a slider for int values
 * @param caption Label text
 * @param value Pointer to int value
 * @param min Minimum value
 * @param max Maximum value
 * @return true if value changed
 */
bool imgui_overlay_slider_int(const char* caption, int32_t* value, int32_t min,
                              int32_t max);

/**
 * @brief Create an input field for float values
 * @param caption Label text
 * @param value Pointer to float value
 * @param step Step increment
 * @param format Printf format string
 * @return true if value changed
 */
bool imgui_overlay_input_float(const char* caption, float* value, float step,
                               const char* format);

/**
 * @brief Create a combo box
 * @param caption Label text
 * @param item_index Pointer to selected index
 * @param items Array of item strings
 * @param item_count Number of items
 * @return true if selection changed
 */
bool imgui_overlay_combo_box(const char* caption, int32_t* item_index,
                             const char** items, uint32_t item_count);

/**
 * @brief Create a button
 * @param caption Button text
 * @return true if button was clicked
 */
bool imgui_overlay_button(const char* caption);

/**
 * @brief Display formatted text
 * @param format_str Printf format string
 */
void imgui_overlay_text(const char* format_str, ...);

/**
 * @brief Create a color edit widget
 * @param caption Label text
 * @param color Pointer to RGBA color array
 * @return true if color changed
 */
bool imgui_overlay_color_edit4(const char* caption, float color[4]);

#endif /* IMGUI_OVERLAY_H */
