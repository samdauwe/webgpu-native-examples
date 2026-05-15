/*
 * WAjic browser-native image loading.
 *
 * Decodes images using the browser's built-in decoder (createImageBitmap)
 * instead of stb_image in WASM.  This is more reliable and efficient:
 *   - No raw file data copied through WASM memory for decoding
 *   - Browser decoder is hardware-accelerated and battle-tested
 *   - Only decoded RGBA pixels are written to WASM memory
 *   - Avoids potential WASM memory view staleness issues with raw data
 *
 * Usage — define WAJIC_IMAGE_IMPL in exactly one .c file:
 *   #define WAJIC_IMAGE_IMPL
 *   #include "webgpu/wajic_image.h"
 *
 *   // In init:
 *   wajic_image_load("assets/textures/foo.png");
 *
 *   // Each frame:
 *   wajic_image_result_t img;
 *   if (wajic_image_poll(&img)) {
 *       if (img.pixels) {
 *           // Use img.pixels (width * height * 4 RGBA bytes)
 *           free(img.pixels);
 *       }
 *   }
 *
 * Limitations:
 *   - Only one image load at a time (sufficient for sequential loading).
 *   - Decoded format is always RGBA8 (4 channels).
 */

#ifndef WAJIC_IMAGE_H
#define WAJIC_IMAGE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wajic_image_result_t {
  uint8_t* pixels; /* RGBA pixel data — caller must free(), NULL on failure */
  int width;
  int height;
} wajic_image_result_t;

/* Start an async image load.  The browser fetches and decodes the image. */
void wajic_image_load(const char* url);

/* Poll for completion.  Returns true when the load has finished (success or
 * failure).  On success result->pixels is non-NULL. */
bool wajic_image_poll(wajic_image_result_t* result);

#ifdef __cplusplus
}
#endif

/* ========================================================================== */
#ifdef WAJIC_IMAGE_IMPL
/* ========================================================================== */

#include <string.h>
#include <wajic.h>

/* ── Internal state ─────────────────────────────────────────────────────── */

static struct {
  bool pending;
  bool done;
  uint8_t* pixels;
  int width;
  int height;
} _wajic_img;

/* ── JS interop ─────────────────────────────────────────────────────────── */

/*
 * Browser-side image loading pipeline:
 *   fetch(url) → response.blob() → createImageBitmap(blob)
 *   → canvas.drawImage() → getImageData() → RGBA pixels
 *   → ASM.malloc() for WASM buffer → MU8.set() → WAFNImageLoaded()
 *
 * The malloc() call happens AFTER decoding so we know the exact pixel size.
 * MU8 is re-validated against MEM.buffer after malloc (which may grow memory).
 */
WAJIC(void, _wajic_js_load_image, (const char* url),
{
    var urlStr = MStrGet(url);
    fetch(urlStr).then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + urlStr);
        return r.blob();
    })
    .then(function(blob) { return createImageBitmap(blob); })
    .then(function(bmp) {
        var w = bmp.width, h = bmp.height;
        var canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(bmp, 0, 0);
        bmp.close();
        var imgData = ctx.getImageData(0, 0, w, h);
        var pixelBytes = imgData.data.length;

        /* Allocate WASM buffer for decoded pixels */
        var ptr = ASM.malloc(pixelBytes);
        if (!ptr) {
            console.error('[wajic_image] malloc(' + pixelBytes + ') failed for ' + urlStr);
            ASM.WAFNImageLoaded(0, 0, 0);
            return;
        }

        /* Refresh MU8 — malloc may have grown WASM memory */
        if (MU8.buffer !== MEM.buffer) MU8 = new Uint8Array(MEM.buffer);
        MU8.set(imgData.data, ptr);
        ASM.WAFNImageLoaded(ptr, w, h);
    })
    ['catch'](function(err) {
        console.error('[wajic_image] Failed to load ' + urlStr + ': ' + err);
        ASM.WAFNImageLoaded(0, 0, 0);
    });
})

/* Exported callback — invoked by JS when image decoding completes. */
WA_EXPORT(WAFNImageLoaded)
void WAFNImageLoaded(void* pixels, int width, int height)
{
  _wajic_img.pixels = (uint8_t*)pixels;
  _wajic_img.width  = width;
  _wajic_img.height = height;
  _wajic_img.done   = true;
}

/* ── Public implementation ──────────────────────────────────────────────── */

void wajic_image_load(const char* url)
{
  memset(&_wajic_img, 0, sizeof(_wajic_img));
  _wajic_img.pending = true;
  _wajic_js_load_image(url);
}

bool wajic_image_poll(wajic_image_result_t* result)
{
  if (!_wajic_img.done)
    return false;

  if (result) {
    result->pixels = _wajic_img.pixels;
    result->width  = _wajic_img.width;
    result->height = _wajic_img.height;
  }

  _wajic_img.pending = false;
  _wajic_img.done    = false;
  return true;
}

#endif /* WAJIC_IMAGE_IMPL */
#endif /* WAJIC_IMAGE_H */
