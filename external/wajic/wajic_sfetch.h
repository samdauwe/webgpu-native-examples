/*
 * WAjic-compatible sokol_fetch shim.
 *
 * Provides the subset of the sokol_fetch (sfetch) API used by the
 * webgpu-native-examples, implemented on top of the browser Fetch API
 * via WAJIC JS interop.
 *
 * Supported operations:
 *   sfetch_setup()     — initialise (accepts sfetch_desc_t)
 *   sfetch_send()      — start an async fetch request
 *   sfetch_dowork()    — poll for completed fetches and fire callbacks
 *   sfetch_shutdown()  — tear down
 *
 * Limitations:
 *   - No streaming / chunk mode (chunk_size is ignored).
 *   - No channels / lanes scheduling (single FIFO queue).
 *   - Max concurrent requests is compile-time WAJIC_SFETCH_MAX_REQUESTS.
 *
 * Usage — similar to sokol_fetch.h:
 *   #define WAJIC_SFETCH_IMPL   (in exactly one .c file)
 *   #include "wajic_sfetch.h"
 */

#ifndef WAJIC_SFETCH_H
#define WAJIC_SFETCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Configuration ──────────────────────────────────────────────────────── */

#ifndef SFETCH_MAX_PATH
#define SFETCH_MAX_PATH (1024)
#endif
#ifndef SFETCH_MAX_USERDATA_UINT64
#define SFETCH_MAX_USERDATA_UINT64 (16)
#endif

#ifndef WAJIC_SFETCH_MAX_REQUESTS
#define WAJIC_SFETCH_MAX_REQUESTS (48)
#endif

/* ── Public types (API-compatible with sokol_fetch.h) ───────────────────── */

typedef struct sfetch_handle_t {
  uint32_t id;
} sfetch_handle_t;

typedef enum sfetch_error_t {
  SFETCH_ERROR_NO_ERROR = 0,
  SFETCH_ERROR_FILE_NOT_FOUND,
  SFETCH_ERROR_NO_BUFFER,
  SFETCH_ERROR_BUFFER_TOO_SMALL,
  SFETCH_ERROR_UNEXPECTED_EOF,
  SFETCH_ERROR_INVALID_HTTP_STATUS,
  SFETCH_ERROR_CANCELLED,
  SFETCH_ERROR_JS_OTHER,
} sfetch_error_t;

typedef struct sfetch_range_t {
  const void* ptr;
  size_t size;
} sfetch_range_t;

/* Convenience macro, mirrors sokol_fetch.h */
#define SFETCH_RANGE(x) (sfetch_range_t){ (const void*)&(x), sizeof(x) }

typedef struct sfetch_response_t {
  sfetch_handle_t handle;
  bool dispatched;
  bool fetched;
  bool paused;
  bool finished;
  bool failed;
  bool cancelled;
  sfetch_error_t error_code;
  uint32_t channel;
  uint32_t lane;
  const char* path;
  void* user_data;
  uint32_t data_offset;
  sfetch_range_t data;
  sfetch_range_t buffer;
  /* Inline storage so the response is self-contained and does not alias the
   * slot (which is freed before the callback fires in a naive implementation).
   * user_data and path are copied here; the public fields above point into
   * this storage. */
  char _path_storage[SFETCH_MAX_PATH];
  uint64_t _user_data_storage[SFETCH_MAX_USERDATA_UINT64];
} sfetch_response_t;

typedef struct sfetch_logger_t {
  void (*func)(const char*, uint32_t, uint32_t, const char*, uint32_t,
               const char*, void*);
  void* user_data;
} sfetch_logger_t;

typedef struct sfetch_allocator_t {
  void* (*alloc_fn)(size_t, void*);
  void (*free_fn)(void*, void*);
  void* user_data;
} sfetch_allocator_t;

typedef struct sfetch_desc_t {
  uint32_t max_requests;
  uint32_t num_channels;
  uint32_t num_lanes;
  sfetch_allocator_t allocator;
  sfetch_logger_t logger;
} sfetch_desc_t;

typedef struct sfetch_request_t {
  uint32_t channel;
  const char* path;
  void (*callback)(const sfetch_response_t*);
  uint32_t chunk_size;
  sfetch_range_t buffer;
  sfetch_range_t user_data;
} sfetch_request_t;

/* ── Public API ─────────────────────────────────────────────────────────── */

void sfetch_setup(const sfetch_desc_t* desc);
sfetch_handle_t sfetch_send(const sfetch_request_t* request);
void sfetch_dowork(void);
void sfetch_shutdown(void);
bool sfetch_valid(void);

#ifdef __cplusplus
}
#endif

/* ========================================================================== */
#ifdef WAJIC_SFETCH_IMPL
/* ========================================================================== */

#include <string.h>
#include <wajic.h>

/* ── Internal slot state ────────────────────────────────────────────────── */

typedef enum {
  _WSFETCH_SLOT_FREE = 0,
  _WSFETCH_SLOT_ACTIVE,  /* JS fetch in-flight */
  _WSFETCH_SLOT_DONE,    /* JS fetch completed — ready for callback */
} _wsfetch_slot_state_t;

typedef struct {
  _wsfetch_slot_state_t state;
  char path[SFETCH_MAX_PATH];
  void (*callback)(const sfetch_response_t*);
  void* buffer_ptr;
  uint32_t buffer_size;
  uint32_t fetched_size;
  sfetch_error_t error;
  uint64_t user_data[SFETCH_MAX_USERDATA_UINT64];
  uint32_t user_data_size;
  bool dynamic; /* true = buffer was allocated by JS via malloc */
} _wsfetch_slot_t;

static struct {
  bool valid;
  _wsfetch_slot_t slots[WAJIC_SFETCH_MAX_REQUESTS];
} _wsfetch;

/* ── JS interop ─────────────────────────────────────────────────────────── */

/*
 * JS-side fetch.  When the response arrives, copies data into the WASM
 * buffer and calls the exported WAFNSFetchDone(slot_index, fetched_size, error).
 *
 * Uses a retry loop with exponential backoff to handle transient network
 * failures that commonly occur during same-tab page navigations and reloads
 * (e.g. the browser reuses a TCP keep-alive connection that the server has
 * already closed).
 */
WAJIC(void, _wsfetch_js_send,
      (uint32_t slot_index, const char* url, void* buf_ptr, uint32_t buf_size),
{
    var urlStr = MStrGet(url);
    var maxRetries = 3;
    var attempt = 0;
    function doFetch() {
        fetch(urlStr, { cache: 'no-cache' }).then(function(resp) {
            if (!resp.ok) {
                console.error('[sfetch] HTTP ' + resp.status + ' for ' + urlStr);
                ASM.WAFNSFetchDone(slot_index, 0, 5 /*INVALID_HTTP_STATUS*/);
                return;
            }
            return resp.arrayBuffer();
        }).then(function(ab) {
            if (!ab) return;
            var u8 = new Uint8Array(ab);
            if (u8.length > buf_size) {
                console.error('[sfetch] Response too large (' + u8.length + ' > ' + buf_size + ') for ' + urlStr);
                ASM.WAFNSFetchDone(slot_index, 0, 3 /*BUFFER_TOO_SMALL*/);
                return;
            }
            /* Refresh MU8 — WASM memory may have grown since the fetch started */
            if (MU8.buffer !== MEM.buffer) MU8 = new Uint8Array(MEM.buffer);
            MU8.set(u8, buf_ptr);
            ASM.WAFNSFetchDone(slot_index, u8.length, 0 /*NO_ERROR*/);
        })['catch'](function(err) {
            attempt++;
            if (attempt < maxRetries) {
                console.warn('[sfetch] Fetch attempt ' + attempt + '/' + maxRetries + ' failed for ' + urlStr + ': ' + err + '. Retrying...');
                setTimeout(doFetch, attempt * 200);
            } else {
                console.error('[sfetch] All ' + maxRetries + ' fetch attempts failed for ' + urlStr + ': ' + err);
                ASM.WAFNSFetchDone(slot_index, 0, 7 /*JS_OTHER*/);
            }
        });
    }
    doFetch();
})

/* Exported callback invoked by JS when a fetch completes. */
WA_EXPORT(WAFNSFetchDone)
void WAFNSFetchDone(uint32_t slot_index, uint32_t fetched_size, uint32_t error)
{
  if (slot_index >= WAJIC_SFETCH_MAX_REQUESTS)
    return;
  _wsfetch_slot_t* s = &_wsfetch.slots[slot_index];
  if (s->state != _WSFETCH_SLOT_ACTIVE)
    return;
  s->fetched_size = fetched_size;
  s->error        = (sfetch_error_t)error;
  s->state        = _WSFETCH_SLOT_DONE;
}

/*
 * JS-side fetch with dynamic buffer allocation.
 *
 * The caller does not provide a pre-allocated buffer. Instead, JS fetches the
 * resource, determines its size, calls WASM malloc() to allocate exactly the
 * right amount of memory, and copies the data in. This eliminates the need
 * for stat() / pre-sizing.
 */
WAJIC(void, _wsfetch_js_send_dynamic,
      (uint32_t slot_index, const char* url),
{
    var urlStr = MStrGet(url);
    var maxRetries = 3;
    var attempt = 0;
    function doFetch() {
        fetch(urlStr, { cache: 'no-cache' }).then(function(resp) {
            if (!resp.ok) {
                console.error('[sfetch] HTTP ' + resp.status + ' for ' + urlStr);
                ASM.WAFNSFetchDoneDynamic(slot_index, 0, 0, 5);
                return;
            }
            return resp.arrayBuffer();
        }).then(function(ab) {
            if (!ab) return;
            var u8 = new Uint8Array(ab);
            var sz = u8.length;
            if (sz === 0) {
                console.error('[sfetch] Empty response for ' + urlStr);
                ASM.WAFNSFetchDoneDynamic(slot_index, 0, 0, 1);
                return;
            }
            /* Allocate WASM memory for the response data */
            var ptr = ASM.malloc(sz);
            if (!ptr) {
                console.error('[sfetch] malloc(' + sz + ') failed for ' + urlStr);
                ASM.WAFNSFetchDoneDynamic(slot_index, 0, 0, 7);
                return;
            }
            /* Refresh MU8 — malloc may have grown WASM memory */
            if (MU8.buffer !== MEM.buffer) MU8 = new Uint8Array(MEM.buffer);
            MU8.set(u8, ptr);
            ASM.WAFNSFetchDoneDynamic(slot_index, ptr, sz, 0);
        })['catch'](function(err) {
            attempt++;
            if (attempt < maxRetries) {
                console.warn('[sfetch] Fetch attempt ' + attempt + '/' + maxRetries + ' failed for ' + urlStr + ': ' + err + '. Retrying...');
                setTimeout(doFetch, attempt * 200);
            } else {
                console.error('[sfetch] All ' + maxRetries + ' fetch attempts failed for ' + urlStr + ': ' + err);
                ASM.WAFNSFetchDoneDynamic(slot_index, 0, 0, 7);
            }
        });
    }
    doFetch();
})

/* Exported callback invoked by JS when a dynamic fetch completes. */
WA_EXPORT(WAFNSFetchDoneDynamic)
void WAFNSFetchDoneDynamic(uint32_t slot_index, uint32_t buf_ptr,
                           uint32_t fetched_size, uint32_t error)
{
  if (slot_index >= WAJIC_SFETCH_MAX_REQUESTS)
    return;
  _wsfetch_slot_t* s = &_wsfetch.slots[slot_index];
  if (s->state != _WSFETCH_SLOT_ACTIVE)
    return;
  s->buffer_ptr   = (void*)(uintptr_t)buf_ptr;
  s->buffer_size  = fetched_size;
  s->fetched_size = fetched_size;
  s->error        = (sfetch_error_t)error;
  s->state        = _WSFETCH_SLOT_DONE;
}

/* ── Public implementation ──────────────────────────────────────────────── */

void sfetch_setup(const sfetch_desc_t* desc)
{
  (void)desc;
  memset(&_wsfetch, 0, sizeof(_wsfetch));
  _wsfetch.valid = true;
}

bool sfetch_valid(void)
{
  return _wsfetch.valid;
}

sfetch_handle_t sfetch_send(const sfetch_request_t* request)
{
  sfetch_handle_t invalid = {0};
  if (!request || !request->path || !request->callback)
    return invalid;

  /* Determine allocation mode:
   * - Pre-allocated: buffer.ptr != NULL, buffer.size > 0 (original behavior)
   * - Dynamic: buffer.ptr == NULL — JS allocates via malloc after response */
  bool dynamic = (!request->buffer.ptr || request->buffer.size == 0);

  /* Find a free slot. */
  for (uint32_t i = 0; i < WAJIC_SFETCH_MAX_REQUESTS; i++) {
    _wsfetch_slot_t* s = &_wsfetch.slots[i];
    if (s->state != _WSFETCH_SLOT_FREE)
      continue;

    s->state       = _WSFETCH_SLOT_ACTIVE;
    s->callback    = request->callback;
    s->dynamic     = dynamic;

    if (!dynamic) {
      s->buffer_ptr  = (void*)request->buffer.ptr;
      s->buffer_size = (uint32_t)request->buffer.size;
    } else {
      s->buffer_ptr  = NULL;
      s->buffer_size = 0;
    }

    size_t plen = strlen(request->path);
    if (plen >= SFETCH_MAX_PATH) plen = SFETCH_MAX_PATH - 1;
    memcpy(s->path, request->path, plen);
    s->path[plen] = '\0';

    /* Copy user data blob (up to SFETCH_MAX_USERDATA_UINT64 * 8 bytes). */
    s->user_data_size = 0;
    if (request->user_data.ptr && request->user_data.size > 0) {
      uint32_t max_ud = (uint32_t)(SFETCH_MAX_USERDATA_UINT64 * sizeof(uint64_t));
      uint32_t ud_sz  = (uint32_t)request->user_data.size;
      if (ud_sz > max_ud) ud_sz = max_ud;
      memcpy(s->user_data, request->user_data.ptr, ud_sz);
      s->user_data_size = ud_sz;
    }

    /* Fire the JS fetch. */
    if (dynamic)
      _wsfetch_js_send_dynamic(i, s->path);
    else
      _wsfetch_js_send(i, s->path, s->buffer_ptr, s->buffer_size);

    sfetch_handle_t h = {i + 1}; /* 1-based so 0 stays invalid */
    return h;
  }

  return invalid; /* no free slot */
}

void sfetch_dowork(void)
{
  for (uint32_t i = 0; i < WAJIC_SFETCH_MAX_REQUESTS; i++) {
    _wsfetch_slot_t* s = &_wsfetch.slots[i];
    if (s->state != _WSFETCH_SLOT_DONE)
      continue;

    /* Build a fully self-contained response by copying all slot data into
     * inline storage fields before the slot is freed.  This guarantees that
     * the callback receives a valid response regardless of slot lifecycle. */
    sfetch_response_t resp;
    memset(&resp, 0, sizeof(resp));
    resp.handle.id   = i + 1;
    resp.buffer.ptr  = s->buffer_ptr;
    resp.buffer.size = s->buffer_size;

    /* Copy path into inline storage and point the public field at it. */
    memcpy(resp._path_storage, s->path, sizeof(resp._path_storage));
    resp._path_storage[sizeof(resp._path_storage) - 1] = '\0';
    resp.path = resp._path_storage;

    /* Copy user_data blob into inline storage and point the public field. */
    if (s->user_data_size > 0) {
      memcpy(resp._user_data_storage, s->user_data, s->user_data_size);
      resp.user_data = resp._user_data_storage;
    }

    if (s->error == SFETCH_ERROR_NO_ERROR) {
      resp.fetched    = true;
      resp.finished   = true;
      resp.data.ptr   = s->buffer_ptr;
      resp.data.size  = s->fetched_size;
    }
    else {
      resp.failed     = true;
      resp.finished   = true;
      resp.error_code = s->error;
    }

    /* Save the callback, free the slot, then fire the callback.
     * The slot is freed first so it is available for re-use if the callback
     * calls sfetch_send() synchronously (a supported sokol_fetch pattern).
     * The response is safe because it no longer aliases the slot. */
    void (*cb)(const sfetch_response_t*) = s->callback;
    memset(s, 0, sizeof(*s)); /* slot → FREE */
    cb(&resp);
  }
}

void sfetch_shutdown(void)
{
  memset(&_wsfetch, 0, sizeof(_wsfetch));
}

#endif /* WAJIC_SFETCH_IMPL */
#endif /* WAJIC_SFETCH_H */
