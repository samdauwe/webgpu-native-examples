/*
  WAjic - WebAssembly JavaScript Interface Creator
  Copyright (C) 2020 Bernhard Schelling

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#pragma once

#include <wajic.h>

// Get file size of an embedded file
WAJIC_LIB(FILE, unsigned int, WaFileGetSize, (const char* name),
{
	var f = WebAssembly.Module.customSections(WM, '|'+MStrGet(name))[0];
	return f&&f.byteLength;
})

// Read from an embedded file into a prepared buffer of at least size (if size is 0 everything past start is read)
WAJIC_LIB(FILE, unsigned int, WaFileRead, (const char* name, void* ptr, unsigned int start WA_ARG(0), unsigned int size WA_ARG(0)),
{
	var a = new Uint8Array(WebAssembly.Module.customSections(WM, '|'+MStrGet(name))[0] || []), end = a.length;
	start = (start < end ? start : end);
	end = (!size || start + size > end ? end : start + size);
	MU8.set(a.subarray(start, end), ptr);
	return end - start;
})

// Read from an embedded file into a newly allocated buffer (if size is 0 everything past start is read)
WAJIC_LIB(FILE, unsigned char*, WaFileMallocRead, (const char* name, unsigned int* out_length, unsigned int start WA_ARG(0), unsigned int size WA_ARG(0)),
{
	var a = new Uint8Array(WebAssembly.Module.customSections(WM, '|'+MStrGet(name))[0] || []), end = a.length;
	start = (start < end ? start : end);
	end = (!size || start + size > end ? end : start + size);
	if (out_length) MU32[out_length>>2] = end - start;
	return MArrPut(a.subarray(start, end));
})

// Load data from a URL and pass the result (or error) back to wasm to a callback that has been marked with WA_EXPORT
WAJIC_LIB(FILE, void, WaFileLoadUrl, (const char* exported_callback, const char* url, void* userdata WA_ARG(0), const char* progress_callback WA_ARG(0), const void* postdata WA_ARG(0), unsigned int postlength WA_ARG(0), unsigned int timeout WA_ARG(0)),
{
	var xhr = new XMLHttpRequest(), cb = ASM[MStrGet(exported_callback)], prog = (progress_callback && ASM[MStrGet(progress_callback)]);
	if (!cb) throw 'bad callback';
	xhr.open((postlength ? 'POST' : 'GET'), MStrGet(url), true);
	xhr.responseType = 'arraybuffer';
	if (timeout) xhr.timeout = timeout;
	xhr.onload = function()
	{
		if (xhr.status == 200)
		{
			var ptr = MArrPut(new Uint8Array(xhr.response));
			cb(200, ptr, xhr.response.byteLength, userdata);
			ASM.free(ptr);
		}
		else cb(xhr.status, 0, 0, userdata);
	};
	if (prog) xhr.onprogress = function(e) { if (e.lengthComputable) prog(e.loaded, e.total, userdata); };
	xhr.ontimeout = xhr.onerror = function(event)
	{
		// This could be called synchronously by xhr.send() so force it to arrive async
		setTimeout(function() { cb(xhr.status||-1, 0, 0, userdata); });
	};
	if (postlength) { try { xhr.send(MU8.subarray(postdata, postdata+postlength)); } catch (e) { xhr.send(MU8.buffer.slice(postdata, postdata+postlength)); } }
	else xhr.send(null);
})
