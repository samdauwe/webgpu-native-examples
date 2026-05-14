<h1><img src="https://wajic.github.io/wajic.png" alt="WAjic - WebAssembly JavaScript Interface Creator" width="600"></h1>

WAjic is a simple way to build C/C++ WebAssembly application with meaningful browser integrations like WebGL.

Inspired by [Emscripten's EM_JS macro](https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-call-javascript-from-native),
WAjic is intended to be a more direct approach to building that offers more control and customization while being straight forward from C/C++ code to the web browser.

It starts out with a single command to call the Clang compiler to produce a .wasm file.
The output can be loaded in the WAjic viewer (available [online](https://wajic.github.io/viewer/)) or further processed with
[WajicUp](#introducing-wajicup) (available as command line tool or [online](https://wajic.github.io/up/)) for customized deployment with minimal file size.

  * [Samples](#samples)
  * [Why](#why)
  * [Setup](#setup)
    * [Getting WAjic](#getting-wajic)
    * [Getting LLVM](#getting-llvm)
    * [Getting Node.js](#getting-nodejs)
  * [Building an Application](#building-an-application)
    * [Automatically Building](#automatically-building)
    * [Manually Building](#manually-building)
    * [Running wasm-opt](#running-wasm-opt)
  * [Introducing WAjicUp](#introducing-wajicup)
  * [Creating your own WAJIC functions](#creating-your-own-wajic-functions)
    * [Functions and objects available in WAJIC code](#functions-and-objects-available-in-wajic-code)
    * [Exporting functions](#exporting-functions)
    * [Shared Init Code Block](#shared-init-code-block)
    * [Libraries](#libraries)
  * [Advanced Features](#advanced-features)
    * [Embedding Files](#embedding-files)
    * [Loading URLs](#loading-urls)
    * [WebGL](#webgl)
    * [Coroutines](#coroutines)
  * [Notes](#notes)
    * [Files in this Repository](#files-in-this-repository)
    * [Clang Parameters](#clang-parameters)
    * [Debugging](#debugging)
    * [Compiling and Linking Separately](#compiling-and-linking-separately)
    * [Manually Building System Libraries](#manually-building-system-libraries)
    * [Directly Compiling with WAjicUp](#directly-compiling-with-wajicup)
    * [Using Symbolic Links](#using-symbolic-links)
  * [Missing Features](#missing-features)
  * [License](#license)

## Samples
Check out the [online sample gallery](https://wajic.github.io/samples/).

## Why
WebAssembly makes it possible to run C/C++ programs in the browser on a website. But WebAssembly alone has no system/user interface.
Code is only executed when asked to do so through JavaScript, and similarly it's output needs to be processed by JavaScript.

For example, to run a C program that renders 3D graphics with OpenGL on the web, every OpenGL system function needs to be implemented
in JavaScript to then call the appropriate WebGL API and pass data between WebAssembly and that API.

This is where WAjic comes in. WAjic provides a way to write these interface functions and libraries directly in C/C++ code.
Furthermore, compiled .wasm files run directly with a generic JavaScript loader on the web and in the command line.

## Setup

### Getting WAjic
To get started download this project from GitHub with the [`Download ZIP`](../../archive/master.zip) button or by cloning the repository.

For just compiling code you only need the header files, the full repository archive comes with some tools and samples as explained below.

### Getting LLVM
We need only Clang and wasm-ld from LLVM 8.0.0 or newer which is available on [the official LLVM releases page](https://releases.llvm.org/download.html).  
On Windows it's much simpler to use 7zip to extract just `clang.exe` and `wasm-ld.exe` instead of installing the whole suite. That's all we need.

### Getting Node.js
This is not required for building, but required for testing in the command-line and the tools explained below.

WebAssembly support and the tools run fine in Node.js version 8 and newer, and don't have any external package dependencies.
If you use Node only for this, just get the latest long-term-support executable.  
You can find the official Win64 EXE [here](https://nodejs.org/download/release/latest-dubnium/win-x64/node.exe).

## Building an Application
Building can be done automatically or manually by calling Clang in the command line or by using a build system like GNU Make.

### Automatically Building
First, make sure clang, wasm-ld and wasm-opt exist in the same directory as wajicup.js.  
See the section [Using Symbolic Links](#using-symbolic-links) for how to do this without fully copying these files.

Then just run the following command to get a fully packaged, optimized and independent html file of the first sample:

`node wajicup.js Samples/Basic.c Basic.html`

At first, this only builds raw C/C++ applications without the C/C++ standard libraries or dynamic memory allocations.  
To get support for these system libraries, just download the [pre-built system libraries and headers](https://github.com/schellingb/wajic/releases/tag/bin) and put it into the wajic directory.

Now we can build the rest of the samples, for example the WebGL sample, with:

`node wajicup.js Samples/WebGL.c WebGL.html`

See the sections [Introducing WAjicUp](#introducing-wajicup) and [Directly Compiling with WAjicUp](#directly-compiling-with-wajicup) for details and how to output separate html/js/wasm files.

### Manually Building
For example, to build the basic sample, run this command to create Basic.wasm:

`clang -I. -Os -target wasm32 -nostartfiles -nodefaultlibs -nostdinc -nostdinc++ -Wno-unused-command-line-argument -DNDEBUG -D__WAJIC__ -fvisibility=hidden -fno-rtti -fno-exceptions -fno-threadsafe-statics -Xlinker -strip-all -Xlinker -gc-sections -Xlinker -no-entry -Xlinker -allow-undefined -Xlinker -export=__wasm_call_ctors -Xlinker -export=main samples/Basic.c -o Basic.wasm`

The built .wasm file can be loaded in the [WAjic viewer](https://wajic.github.io/viewer/) or via Node.js CLI `node wajic.js Basic.wasm`.

This only builds raw C/C++ applications without the C/C++ standard libraries or dynamic memory allocations.  
To get support for these system libraries, just download the [pre-built system libraries and headers](https://github.com/schellingb/wajic/releases/tag/bin) and put it into the wajic directory.

Once you have the system files some more arguments need to be added to the build command.  
Now we can build the rest of the samples, for example the WebGL sample, with:

`clang -isystem./system/include/libcxx -isystem./system/lib/libcxx/include -isystem./system/include/compat -isystem./system/include -isystem./system/include/libc -isystem./system/lib/libc/musl/include -isystem./system/lib/libc/musl/arch/emscripten -isystem./system/lib/libc/musl/arch/generic -Xlinker ./system/system.bc -D__EMSCRIPTEN__ -D_LIBCPP_ABI_VERSION=2 -I. -Os -target wasm32 -nostartfiles -nodefaultlibs -nostdinc -nostdinc++ -Wno-unused-command-line-argument -DNDEBUG -D__WAJIC__ -fvisibility=hidden -fno-rtti -fno-exceptions -fno-threadsafe-statics -Xlinker -strip-all -Xlinker -gc-sections -Xlinker -no-entry -Xlinker -allow-undefined -Xlinker -export=__wasm_call_ctors -Xlinker -export=main -Xlinker -export=malloc -Xlinker -export=free samples/WebGL.c -o WebGL.wasm`

With the .wasm file ready and tested in the [WAjic viewer](https://wajic.github.io/viewer/), we can go on further optimizing the result.

If the program doesn't do WASM memory allocation from JavaScript code, `-Xlinker -export=malloc` can be removed.
Same with freeing memory from JavaScript code, `-Xlinker -export=free` can be removed.  
The [packing utility program](#introducing-wajicup) explained below will warn when there are unused or missing exports.

In the notes below you'll find [explanation for the various parameters](#clang-parameters) and also sample commands to [run the compiling and linking separately](#compiling-and-linking-separately).

If you want to build the system libraries yourself, check the [chapter about it below](#manually-building-system-libraries).

### Running wasm-opt
The tool wasm-opt from Binaryen provides a 10% to 15% size reduction of the generated .wasm files.
It also offers generation of shim functions to pass 64-bit numbers between C and JavaScript.  
Binary releases are available on the [Binaryen project page](https://github.com/WebAssembly/binaryen/releases).  
Feel free to extract only `wasm-opt` and ignore the rest.

Then run it over the wasm file with:

`wasm-opt --legalize-js-interface --low-memory-unused --converge -Os WebGL.wasm -o WebGL.wasm`

Functionality wise nothing changes (unless there are 64bit parameters passed to/from JavaScript) and it still runs in the browser just like before.

## Introducing WAjicUp
The WebAssembly JavaScript Interface Creator Utility Program is a tool bundled with WAjic that allows optimization and customized JavaScript/HTML generation.
It is available as command line tool (with Node.js) or [online](https://wajic.github.io/up/).

The most basic command:

`node wajicup.js WebGL.wasm WebGL.wasm`

This will minify the JavaScript code that has been embedded.

Next we can also generate multiple output files:

`node wajicup.js WebGL.wasm BuiltWebGL.wasm BuiltWebGL.js BuiltWebGL.html`

This will take out the JavaScript code embedded in the wasm file and put it into a standalone JavaScript loader file.
It even generates a simple HTML file that loads the loader (which then loads the wasm file).

Any combination of wasm/js/html file is supported. For example outputting just a single HTML file will make WAjicUp
embed both the WASM data and the JavaScript code inside it.

Some option switches are also available:

 Name           | Explanation
----------------|-----------------
 `-no_minify`   | Don't minify JavaScript code
 `-no_log`      | Remove all output logging
 `-streaming`   | Enable [WASM streaming](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly/instantiateStreaming) (needs web server support, new browser)
 `-rle`         | Use RLE compression when embedding the WASM file
 `-loadbar`     | Add a loading progress bar to the generated HTML
 `-node`        | Output JavaScript that runs in Node.js (CLI)
 `-embed N P`   | Embed data file with embed name N from path P (see [file embedding](#embedding-files))
 `-gzipreport`  | Report the potential output size with gzip compression
 `-args`        | Enable C program arguments (argc/argv) that are passed to main (can be customized in the WASM loading html)
 `-arg X`       | Passes X to the program arguments, can be specified multiple times (first arg must be the program name)
 `-template H`  | Uses HTML template file H instead of generating a default file.
 `-stacksize S` | Overrides the size of the stack from the WebAssembly default of just 64 kb
 `-cc "X"`      | When [compiling directly with WAjicUp](#directly-compiling-with-wajicup), this passes argument X to the compiler
 `-ld "Y"`      | When [compiling directly with WAjicUp](#directly-compiling-with-wajicup), this passes argument Y to the linker
 `-v`           | Be verbose about processed functions
 `-h`           | Show command line usage

## Creating your own WAJIC functions
With the WAJIC macro you can declare a function callable from C/C++ with a JavaScript code body that can then access all kinds of web APIs:

```C
WAJIC(void, ShowAlert, (const char* msg),
{
	Alert(MStrGet(msg));
})
```

The first line declares the function, it's return value and it's arguments as used by the C code.  
The function body after that is JavaScript code that can execute web APIs, manipulate the WASM memory and call other WASM functions, too.

```C
WAJIC(int, AddThenMultiply, (int factor),
{
	return ASM.Add(1, 2) * factor;
}

WA_EXPORT(Add) int Add(int a, int b)
{
	return a + b;
}
```

This creates a C function `Add` that adds two values and a JavaScript function (callable form C) `AddThenMultiply` that then invokes the Add function and returns its result multiplied by a factor.

```C
WAJIC(char*, GetDocumentTitle, (),
{
	return MStrPut(document.title)
})
```

This creates a function callable from C that returns the document.title string. Because this allocates memory in the JavaScript side with malloc, the C program needs to call free() on the returned pointer.

### Functions and objects available in WAJIC code
On the JavaScript side there are a handful of utility functions and variables available. Some of them were already shown off in the examples above.

 Name                          | Explanation
-------------------------------|-----------------
 `MStrPut(str)`                | Allocate memory and store a JavaScript string `str` encoded as UTF8 with a \0 null terminator on the WASM heap and return the new pointer.
 `MStrPut(str, ptr, buf_size)` | Store a JavaScript string `str` encoded as UTF8 with a \0 null terminator in a prepared WASM buffer at `ptr` of size `buf_size`. Returns the actual number of bytes written (excluding the null terminator). Returns 0 and does nothing if `buf_size` is 0.
 `MStrGet(ptr)`                | Read a \0 null terminated UTF8 string from the WASM memory at address `ptr`.
 `MStrGet(ptr, length)`        | Read a UTF8 string of size `length` bytes from the WASM memory at address `ptr`.
 `MArrPut(arr)`                | Allocate memory and store a JavaScript array/typed array/buffer on the WASM heap and return the new pointer.
 `ASM`                         | An object which contains all the exports from the WASM module. Its primary use is to call C/C++ functions/callbacks from WAJIC functions.
 `WM`                          | Gives access to the [WebAssembly module object](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly/Module), used for accessing [embedded files](#embedding-files).
 `MU8`                         | Access to the WASM memory as unsigned 8-bit integers
 `MU16`                        | Access to the WASM memory as unsigned 16-bit integers. Usually accessed by right shifting pointers by 1, like `MU16[ptr>>1] = 789`.
 `MU32`                        | Access to the WASM memory as unsigned 32-bit integers. Usually accessed by right shifting pointers by 2, like `MU32[ptr>>2] = 1`.
 `MI32`                        | Access to the WASM memory as signed 32-bit integers. Usually accessed by right shifting pointers by 2, like `MI32[ptr>>2] = -1`.
 `MF32`                        | Access to the WASM memory as 32-bit floats. Usually accessed by right shifting pointers by 2, like `MF32[ptr>>2] = 0.5`.
 `STOP`                        | A boolean variable that is set to true when the program aborts/crashes. If you use requestAnimationFrame/setInterval/setTimeout/event listeners you should check this first before continuing.
 `abort(code, msg)`            | This will abort a running program where `code` can be a predefined tag like 'BOOT', 'CRASH', 'MEM' or your own if you extend the WA.error function in the front-end. `msg` contains more error details.

### Exporting functions
To make a C/C++ function available to the JavaScript world (both for custom front-end scripts and WAJIC functions), you can tag them with WA_EXPORT(<name>).  
For an example, you can check the [code above](#creating-your-own-wajic-functions) and how the `Add` function is annotated with it.

### Shared Init Code Block
If you want to share functionality and variables between multiple WAJIC JavaScript functions, you can add a shared init code block.

```C
WAJIC_WITH_INIT(
(
	var myCounter = 1;
),
int, GetCounter, (),
{
	return myCounter;
})
WAJIC(void, AddCounter, (int num),
{
	myCounter += num;
})
```

Here the variable myCounter is defined outside of the two WAJIC functions and it is available to both.

Check the [SharedInit sample](https://wajic.github.io/samples/?SharedInit)

There is one caveat with init blocks, its availability is attached to the function it's defined together with.
So in the example above, if the C program never calls `GetCounter`, `AddCounter` will not work properly.
You have to make sure to attach the init block to a function that must be referenced by the main program.

### Libraries
There are two more macros specifically for encapsulating libraries into distinct function groups:  
`WAJIC_LIB` and `WAJIC_LIB_WITH_INIT`

These in itself are very similar to the base `WAJIC` and `WAJIC_WITH_INIT`, except that there is one more argument at the beginning.  
The main feature of library grouping is that the init code block is only shared between functions in the same library group.

```C
WAJIC_LIB_WITH_INIT(MYLIB, (...), int, InitMyLib, (...), {...})
WAJIC_LIB(MYLIB, void, DoSomethingInMyLib, (...), {...})
```

## Advanced Features

### Embedding Files
You can embed binary files with WAjicUp and read them in your program. To embed a file add the -embed parameter:

`node wajicup.js EmbedFile.wasm EmbedFile.wasm -embed MYFILE data.bin`

And then in your program you can use standard C functions to read embedded files:

```C
char buf[1024];
FILE* f = fopen("MYFILE", "rb");
int len = fread(buf, 1, 1024, f); // read the first 1024 bytes of the file
fclose(f);
```

Alternatively you can access the file contents with the wajic file API:

```C
#include <wajic_file.h>
unsigned int size = WaFileGetSize("MYFILE") // get the file size
unsigned int read = WaFileRead("MYFILE", data, start, sizeof(data)); // read sizeof(data) bytes from file at offset start
const char* file = (const char*)WaFileMallocRead("MYFILE", &size); // allocate memory and read the full file (optional start/offset)
free(file); // free the memory allocated by WaFileMallocRead
```

It is also possible to embed all files in a directory.

`node wajicup.js EmbedFile.wasm EmbedFile.wasm -embed somedir/ ../path/to/somedir/`

Check the [EmbedFile sample](https://wajic.github.io/samples/?EmbedFile) and the implementation in [wajic_file.h](wajic_file.h).

### Loading URLs
You can load data at URLs with optional progress updates during the download (for example to show a progress).
The URL can be relative to the HTML file that executes the WASM file.

To do so, you have to pass a string of the name of the callback function like this:

```C
#include <wajic_file.h>

// This function is called when the HTTP request finishes (or has an error)
WA_EXPORT(FinishCallback) void FinishCallback(int status, char* data, unsigned int length, void* userdata)
{ printf("Got response - status: %d - length: %u - data: '%.4s...' - userdata: %p\n", status, length, data, userdata); }

// This function is called periodically with download progress updates until download is complete
WA_EXPORT(ProgressCallback) void ProgressCallback(unsigned int loaded, unsigned int total, void* userdata)
{ printf("Progress - loaded: %u - total: %u - userdata: %p\n", loaded, total, userdata); }

WaFileLoadUrl("FinishCallback", url, (void*)0x1234, "ProgressCallback");
```

The third parameter is a user data pointer which can be anything that will be given back in the callbacks so if you
do multiple LoadUrl requests you know which one is which. POST requests and setting a custom timeout are also available.

Check the [LoadUrl sample](https://wajic.github.io/samples/?LoadUrl) and the implementation in [wajic_file.h](wajic_file.h).

### WebGL
Currently WAjic comes with a WebGL version 1 header that emulates OpenGL ES 2.0 API which in itself is a subset of desktop OpenGL 2.0/3.0.

OpenGL ES based means vertex/fragment shaders required, no fixed function pipeline and also no unbuffered vertex attribute arrays.

There is no shader code transformation, shaders need to be written with WebGL compatibility in mind (i.e. explicit float precision, no f suffix for floats).

Check the [WebGL sample](https://wajic.github.io/samples/?WebGL) for how to set up a canvas and render something.

### Coroutines

Coroutines allows execution to suspend (yield back to the browser) or switch between function contexts. It can be used to suspend a running program mid-function (wait for
time to pass or until the next animation frame) or to emulate threads.

Check the [Coroutine sample](https://wajic.github.io/samples/?Coroutine) and the implementation in [wajic_coro.h](wajic_coro.h).

If you don't use WAjicUp, you will need to run another step of wasm-opt with the following command:

`wasm-opt --asyncify Coroutine.wasm -o Coroutine.wasm`

## Notes

### Files in this Repository
 File                                  | Explanation
---------------------------------------|-----------------
[wajic.h](wajic.h)                     | The main header defining the WAJIC macros as well as WA_EXPORT
[wajic_gl.h](wajic_gl.h)               | Header defining the [WebGL functionality](#webgl)
[wajic_file.h](wajic_file.h)           | Header defining functions for dealing with [embedded files](#embedding-files) and [loading URLs](#loading-urls)
[wajic_coro.h](wajic_coro.h)           | Header defining functions for dealing with [Coroutine functionality](#coroutines)
[wajic.js](wajic.js)                   | The generic WASM loader that extracts WAJIC functions and instantiates them in JavaScript. Compatible with web and Node.js (commandline).
[wajic.minified.js](wajic.minified.js) | Minified version of wajic.js.
[wajic.mk](wajic.mk)                   | A GNU make makefile to build [the system libraries](#manually-building-system-libraries) as well as wasm files.
[wajicup.js](wajicup.js)               | WAjic [Utility Program](#introducing-wajicup) for optimizing of wasm files and generating front-ends/loaders.
[wajicup.html](wajicup.html)           | Web UI for WAjicUp to use it without Node.js (also available [online](https://wajic.github.io/up/)).
[viewer.html](viewer.html)             | Viewer tool to easily load and test built wasm files (also available [online](https://wajic.github.io/viewer/)).

### Clang Parameters
 Parameter                                  | Explanation
--------------------------------------------|-----------------
 `-I<wajic directory>`                      | Add the WAjic home directory to the include search path list (for wajic.h, etc)
 `-Os`                                      | Optimize for performance and output size, see [clang manual](https://clang.llvm.org/docs/CommandGuide/clang.html#code-generation-options)
 `-target wasm32`                           | Build for the WebAssembly target
 `-nostartfiles`                            | Avoid clang trying to build for a native console application (don't link with crt1.o)
 `-nodefaultlibs`                           | Don't link with the clang standard libraries (don't link with libc.a)
 `-nostdinc`                                | Remove the internal system directories from the include search path
 `-nostdinc++`                              | Remove the internal C++ directories from the include search path
 `-Wno-unused-command-line-argument`        | Don't complain if we have C++ exclusive arguments (like -fno-rtti) when building a C program
 `-DNDEBUG`                                 | Define the macro `NDEBUG` (removes debug overhead in some libraries)
 `-D__WAJIC__`                              | Define the macro `__WAJIC__` to allow checking if we're building with the WAjic headers available
 `-fvisibility=hidden`                      | Mark all functions and symbols as hidden so unused code can get removed
 `-fno-rtti`                                | Disable C++ run-time type information
 `-fno-exceptions`                          | Disable C++ exceptions
 `-fno-threadsafe-statics`                  | Do not emit code to make initialization of local statics thread safe
 `-Xlinker -strip-all`                      | Strip debug information from the output
 `-Xlinker -gc-sections`                    | Strip unused functions and symbols from the output
 `-Xlinker -no-entry`                       | Disable entry symbol of native application (the JavaScript loader does this for us)
 `-Xlinker -allow-undefined`                | Allow undefined symbols (the functions imported from JavaScript are undefined during compiling)
 `-Xlinker -export=__wasm_call_ctors`       | Export the special startup function which constructs global objects
 `-Xlinker -export=main`                    | Export the main function if available (will do nothing if it doesn't exist)
 `-Xlinker -export=malloc`                  | Export the malloc function to do memory allocation from JavaScript
 `-Xlinker -export=free`                    | Export the free function to free allocated memory from JavaScript
 `Samples/Basic.c`                          | Set the list of source files
 `-o Basic.wasm`                            | Defines the output file
 `-isystem<wajic dir>/system/<directories>` | When building with the C/C++ standard libraries, this sets the required include search paths
 `-Xlinker <wajic dir>/system/system.bc`    | When building with the C/C++ standard libraries, this links against the precompiled library
 `-D__EMSCRIPTEN__`                         | When building with the C/C++ standard libraries, this is required for musl-libc for WASM
 `-D_LIBCPP_ABI_VERSION=2`                  | When building with the C++ standard library, this is a required macro for WebAssembly builds

### Debugging
In Chrome it is easily possible to debug the JavaScript code that has been embedded in the wasm file.
Open the Developer tools in your browser and place a breakpoint on the line of code where `WebAssembly.instantiate` is called.
Then check the Scope view and find the imports object that is passed to the function call and look into it. There you'll find clickable
values tagged with `[[FunctionLocation]]`. After following the location, click the "Pretty print" button in the lower left of the source
view to make things readable. Breakpoints and everything is supported.

Sadly Firefox is not yet on the same level regarding debugging of functions generated at runtime, but hopefully in the future it will.

### Compiling and Linking Separately
To build one of the samples by calling the compiler separately from the linker, first call clang for each source file to create an object file with .o extension:

`clang -cc1 -triple wasm32 -emit-obj -fcolor-diagnostics -I. -isystem./system/include/libcxx -isystem./system/lib/libcxx/include -isystem./system/include/compat -isystem./system/include -isystem./system/include/libc -isystem./system/lib/libc/musl/include -isystem./system/lib/libc/musl/arch/emscripten -isystem./system/lib/libc/musl/arch/generic -fno-common -mconstructor-aliases -fvisibility hidden -fno-threadsafe-statics -fgnuc-version=4.2.1 -D__WAJIC__ -D__EMSCRIPTEN__ -D_LIBCPP_ABI_VERSION=2 -DNDEBUG -x c -std=c99 -Os samples/WebGL.c -o WebGL.o`

(For C++ you'd replace `-x c -std=c99` with `-x c++ -std=c++11 -fno-rtti`)

Then to link the object files together into one .wasm file, call the linker wasm-ld like this:

`wasm-ld -strip-all -gc-sections -no-entry -allow-undefined ./system/system.bc -export=__wasm_call_ctors -export=main -export=malloc -export=free WebGL.o -o WebGL.wasm`

### Manually Building System Libraries
Prebuilt system libraries and headers are provided as a [download on this repository](https://github.com/schellingb/wajic/releases/tag/bin).

To build them yourself you can use the provided GNU make file.

1. Getting System Library Sources  
The system libraries (libc/libcxx prepared for WASM) are maintained in the [Emscripten project](https://github.com/emscripten-core/emscripten/tree/main/system).  
Just download its [GitHub main archive](https://github.com/emscripten-core/emscripten/archive/refs/heads/main.zip) and extract only the `system` directory from it.

2. Getting GNU Make  
If you're on Windows, GNU Make is a small 180 KB EXE file which you can get [here](https://github.com/schellingb/ZillaLib/raw/master/Tools/make.exe).  
On Linux you can install the Make package and on macOS it comes as part of Xcode.

Next create a file called `LocalConfig.mk` and place it next to `wajic.mk` with the following content:

```make
LLVM_ROOT   = D:/dev/wasm/llvm
SYSTEM_ROOT = D:/dev/wasm/system
```

The variables are:  
  - `LLVM_ROOT`: Path to LLVM with clang and wasm-ld executables (see [Getting LLVM](#getting-llvm))
  - `SYSTEM_ROOT`: Pointing to the path of the `system` directory explained above.

Then you can build the system libraries (contains libc + libcxx + malloc) with the following command (use forward slashes even on Windows):

`make -j 8 -f <path-to-wajic.mk> <path-to-wajic-root>/system/system.bc`

### Directly Compiling with WAjicUp
WAjicUp actually accepts c/cpp files as input.
To use it, the executables of clang, wasm-ld and wasm-opt need to be in the same directory as wajicup.js.
See the section [Using Symbolic Links](#using-symbolic-links) for means to avoid copying these files.

Then it's as easy as running it like this:

`node wajicup.js Samples/Basic.c Basic.wasm`

Just like with a .wasm file as input, all output variations of wasm/js/html are supported.  
To pass additional command line options (like -I or -D) to the compiler, you can use one or more `-cc` switches.  
And similarly with one or more `-ld` switches options can be passed to the linker.
When passing the special `-cc -g` switch, code will be built in debug mode with full DWARF debug information included.
This makes it possible to debug through the native code and have breakpoints in the actual C/CPP files.

It is also possible to output a single c/cpp file into a object .o file. This later can then be linked to a .wasm with `-ld obj.o`.  
Further more, one or more source files can be compiled into a single bitcode archive .bc file. Just like .o this can be linked.  
Example:

```sh
node wajicup.js big.c big.o
node wajicup.js multiple.c source.c files.c multi.bc
node wajicup.js main.cpp -ld big.o -ld multi.bc combined.html
```

### Using Symbolic Links
To compile directly with WAjicUp, the executables clang, wasm-ld and wasm-opt need to be in the same directory as wajicup.js.
If you have them somewhere else on your system then it's easiest to use symbolic links.

On Windows, you can accomplish this with (replace 'D:\dev\wasm\*' with your paths):

```
mklink "<path-to-wajic-root>\clang.exe" "D:\dev\wasm\llvm\clang.exe"
mklink "<path-to-wajic-root>\wasm-ld.exe" "D:\dev\wasm\llvm\wasm-ld.exe"
mklink "<path-to-wajic-root>\wasm-opt.exe" "D:\dev\wasm\wasm-opt.exe"
```

On Linux, this can be done by running (replace '/usr/bin/' with your paths):

```
ln -s /usr/bin/clang <path-to-wajic-root>/clang
ln -s /usr/bin/wasm-ld <path-to-wajic-root>/wasm-ld
ln -s /usr/bin/wasm-opt <path-to-wajic-root>/wasm-opt
```

## Missing Features
At this point in time, WAjic has no support for the following features:
 * Threads (or posix thread emulation) *[Coroutines](#coroutines) might be an alternative
 * setjmp/longjmp *[Coroutines](#coroutines) might be an alternative
 * Full Filesystem emulation *[Embedding Files](#embedding-files) supports file access
 * C++ exceptions
 * TCP socket emulation
 * SIMD

These features are all fully or partially addressed by [Emscripten](https://emscripten.org/).  
If you rely on any of them, you should use Emscripten or try contributing to this project.

## License
WAjic is available under the [zlib license](https://choosealicense.com/licenses/zlib/).  
WAjicUp uses [Terser JavaScript compressor](https://github.com/terser/terser) which is under the [BSD license](https://github.com/terser/terser/blob/master/LICENSE).
