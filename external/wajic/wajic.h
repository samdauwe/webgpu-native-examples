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

#ifdef __cplusplus
#define WA_EXTERN extern "C"
#define WA_ARG(v) = v
#else
#define WA_EXTERN extern
#define WA_ARG(v)
#endif

// Macro to generate a JavaScript function that can be called from C
#define WAJIC(ret, name, args, ...) WA_EXTERN __attribute__((import_module("J"), import_name(#name "\x11" #args "\x11" #__VA_ARGS__))) ret name args;

// Macro to generate a JavaScript function that can be called from C with additional shared init code
#define WAJIC_WITH_INIT(INIT, ret, name, args, ...) WA_EXTERN __attribute__((import_module("J"), import_name(#name "\x11" #args "\x11" #__VA_ARGS__ "\x11\x11" #INIT ))) ret name args;

// Macro to generate a JavaScript function sharing same init code that can be called from C
#define WAJIC_LIB(lib, ret, name, args, ...) WA_EXTERN __attribute__((import_module("J"), import_name(#name "\x11" #args "\x11" #__VA_ARGS__ "\x11" #lib))) ret name args;

// Macro to generate a JavaScript function that can be called from C also specifying shared init code
#define WAJIC_LIB_WITH_INIT(lib, INIT, ret, name, args, ...) WA_EXTERN __attribute__((import_module("J"), import_name(#name "\x11" #args "\x11" #__VA_ARGS__ "\x11" #lib "\x11" #INIT))) ret name args;

// Macro to make a C function available from JavaScript
#define WA_EXPORT(name) __attribute__((used, visibility("default"), export_name(#name)))

// Specially named types to support 64-bit passing (though it is recommended to pass by pointer)
typedef signed long long WAi64;
typedef unsigned long long WAu64;
