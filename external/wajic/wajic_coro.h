/*
  WAjic - WebAssembly JavaScript Interface Creator
  Copyright (C) 2021 Bernhard Schelling

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

typedef void* WaCoro;
typedef int (*WaCoroEntry)(void* user_data);

WAJIC_LIB_WITH_INIT(CORO,
(
	var main_data, coro_current, coro_count = 0, coro_nums = {}, coro_asms = [0], org_started = WA.started, org_main;
	WA.started = ()=>{ (org_started && org_started()); WA.started = org_started; CoroHandler(); };
	function CoroHandler()
	{
		for (;;)
		{
			var nptr = (coro_current>>2)+4, n = MU32[nptr], fn;
			// States:
			// 0: Function ended without switching context
			// 1: New coro entering function for the first time
			// 2: Switching back into the coro after yielding to the browser
			// 3: Waiting with WaCoroWaitAnimFrame
			// 4: Waiting with WaCoroYield (wait as short as possible)
			// 5: Waiting with WaCoroSleep for a few milliseconds
			if (!n) return;
			if (n == 3) window.requestAnimationFrame(CoroHandler);
			if (n == 4) window.postMessage(9, "*");
			if (n > 4) setTimeout(CoroHandler, n - 5);
			if (n > 2) { MU32[nptr] = 2; return; }
			ASM.asyncify_stop_unwind();
			if (n == 2) ASM.asyncify_start_rewind(coro_current);

			if (fn = MU32[nptr-2])
				coro_asms[fn](MU32[nptr-1]);
			else
				org_main();
		}
	}
	function CoroCtxSwitch(n)
	{
		if (!main_data)
		{
			org_main = (ASM.main||ASM.__main_argc_argv||ASM.__original_main||ASM.__main_void||ASM.WajicMain);
			var ptr = (main_data = coro_current = ASM.malloc(20+WASM_STACK_SIZE))>>2;
			MU32[ptr+0] = main_data + 20;
			MU32[ptr+1] = main_data + 20 + WASM_STACK_SIZE;
			MU32[ptr+2] = 0;
			MU32[ptr+3] = 0;
			MU32[ptr+4] = 0;
		}
		if (MU32[(coro_current>>2)+4] == 2)
		{
			MU32[(coro_current>>2)+4] = 0;
			ASM.asyncify_stop_rewind();
			return false;
		}
		MU32[(coro_current>>2)+4] = n;
		ASM.asyncify_start_unwind(coro_current);
		return true;
	}
),
// Create a new coroutine by passing a c function and the name the function was exported as with WA_EXPORT
WaCoro, WaCoroInitNew, (WaCoroEntry fn, const char* fn_wa_export, void* user_data WA_ARG(0), int stack_size WA_ARG(0)),
{
	if (!stack_size) stack_size = WASM_STACK_SIZE;
	fn = coro_nums[fn_wa_export] || (coro_asms[++coro_count] = ASM[MStrGet(fn_wa_export)],coro_nums[fn_wa_export] = coro_count);
	var res = ASM.malloc(20+stack_size), ptr = res>>2;
	MU32[ptr+0] = res + 20;
	MU32[ptr+1] = res + 20 + stack_size;
	MU32[ptr+2] = fn;
	MU32[ptr+3] = user_data;
	MU32[ptr+4] = 1;
	return res;
})

// Free a coroutine
WAJIC_LIB(CORO, void, WaCoroFree, (WaCoro coro),
{
	ASM.free(coro);
})

// Switch context to a coroutine or back to main (by passing NULL)
WAJIC_LIB(CORO, void, WaCoroSwitch, (WaCoro to WA_ARG(0)),
{
	if (CoroCtxSwitch(2)) coro_current = (to || main_data);
})

// Yield to the browser and wait until the next canvas animation frame
WAJIC_LIB(CORO, void, WaCoroWaitAnimFrame, (),
{
	CoroCtxSwitch(3);
})

WAJIC_LIB_WITH_INIT(CORO,
(
	window.addEventListener("message", (evt) => { if (evt.data===9) CoroHandler(); });
),
// Yield to the browser for as short as possible
void, WaCoroYield, (),
{
	CoroCtxSwitch(4);
})

// Yield to the browser for a certain number of milliseconds (less than 4 will still wait 4 milliseconds in most browsers)
WAJIC_LIB(CORO, void, WaCoroSleep, (int ms),
{
	CoroCtxSwitch(5 + (ms < 0 ? 0 : ms));
})

// Define WA_CORO_IMPLEMENT_NANOSLEEP in one source file before including this header to get an implementation of the posix nanosleep function
#ifdef WA_CORO_IMPLEMENT_NANOSLEEP
#include <time.h>
#include <inttypes.h>
int nanosleep(const struct timespec *req, struct timespec *rem)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC_RAW, &t);
	uint64_t cur = ((uint64_t)t.tv_sec * 1000000000 + t.tv_nsec);
	uint64_t til = cur + ((uint64_t)req->tv_sec * 1000000000 + req->tv_nsec);
	while (cur < til)
	{
		uint64_t remain = til - cur;
		if (remain > 4500000) WaCoroSleep((remain - 500000) / 1000000);
		else WaCoroYield();
		clock_gettime(CLOCK_MONOTONIC_RAW, &t);
		cur = ((uint64_t)t.tv_sec * 1000000000 + t.tv_nsec);
	}
	if (rem) { rem->tv_sec = 0; rem->tv_nsec = 0; }
	return 0;
}
#endif
