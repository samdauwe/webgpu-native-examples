#
#  WAjic - WebAssembly JavaScript Interface Creator
#  Copyright (C) 2020-2022 Bernhard Schelling
#
#  This software is provided 'as-is', without any express or implied
#  warranty.  In no event will the authors be held liable for any damages
#  arising from the use of this software.
#
#  Permission is granted to anyone to use this software for any purpose,
#  including commercial applications, and to alter it and redistribute it
#  freely, subject to the following restrictions:
#
#  1. The origin of this software must not be misrepresented; you must not
#     claim that you wrote the original software. If you use this software
#     in a product, an acknowledgment in the product documentation would be
#     appreciated but is not required.
#  2. Altered source versions must be plainly marked as such, and must not be
#     misrepresented as being the original software.
#  3. This notice may not be removed or altered from any source distribution.
#

THIS_MAKEFILE := $(patsubst ./%,%,$(subst \,/,$(lastword $(MAKEFILE_LIST))))
WAJIC_ROOT    := $(dir $(THIS_MAKEFILE))
ISWIN         := $(findstring :,$(firstword $(subst \, ,$(subst /, ,$(abspath .)))))
PIPETONULL    := $(if $(ISWIN),>nul 2>nul,>/dev/null 2>/dev/null)
-include $(WAJIC_ROOT)LocalConfig.mk
ifneq ($(if $(LLVM_ROOT),,$(wildcard ./clang.exe)),)
  LLVM_ROOT = .
endif
ifeq ($(LLVM_ROOT),)
  $(info )
  $(info Please create the file $(WAJIC_ROOT)LocalConfig.mk with at least the following definition:)
  $(info )
  $(info LLVM_ROOT = $(if $(ISWIN),d:)/path/to/llvm)
  $(info )
  $(info For building system.bc from source, add this definition:)
  $(info )
  $(info SYSTEM_ROOT = $(if $(ISWIN),d:)/path/to/emscripten-system)
  $(info )
  $(info For optimizing wasm files built with this makefile, add these definitions:)
  $(info WASMOPT = $(if $(ISWIN),d:)/path/to/wasm-opt$(if $(ISWIN),.exe))
  $(info NODE = $(if $(ISWIN),d:)/path/to/node$(if $(ISWIN),.exe))
  $(info )
  $(error Aborting)
endif

SYSTEM_ROOT := $(or $(SYSTEM_ROOT),$(WAJIC_ROOT)system)

#------------------------------------------------------------------------------------------------------

ifeq ($(BUILD),DEBUG)
  OUTDIR    := Debug-wasm
  OFLAGS    := -debug-info-kind=limited -DDEBUG -D_DEBUG
  LDFLAGS   :=
  WOPTFLAGS := -g
else
  OUTDIR    := Release-wasm
  OFLAGS    := -Os -DNDEBUG
  LDFLAGS   := -strip-all -gc-sections
  WOPTFLAGS := -O3 --legalize-js-interface --low-memory-unused --ignore-implicit-traps --converge
endif

# Global compiler flags
CXXFLAGS := -x c++ -std=c++11 -fno-rtti $(OFLAGS)
CFLAGS   := -x c -std=c11 $(OFLAGS)

# Global compiler flags for Wasm targeting
CLANGFLAGS := -triple wasm32-unknown-emscripten -emit-obj -fcolor-diagnostics
CLANGFLAGS += -I${WAJIC_ROOT}
CLANGFLAGS += -isystem$(SYSTEM_ROOT)/$(if $(wildcard $(SYSTEM_ROOT)/include/libcxx),include/libcxx,lib/libcxx/include) # due to change in 2.0.13
CLANGFLAGS += -isystem$(SYSTEM_ROOT)/include/compat
CLANGFLAGS += -isystem$(SYSTEM_ROOT)/include
CLANGFLAGS += -isystem$(SYSTEM_ROOT)/$(if $(wildcard $(SYSTEM_ROOT)/include/libc),include/libc,lib/libc/musl/include) # due to change in 2.0.13
CLANGFLAGS += -isystem$(SYSTEM_ROOT)/lib/libc/musl/arch/emscripten
CLANGFLAGS += $(if $(wildcard $(SYSTEM_ROOT)/lib/libc/musl/arch/generic),-isystem$(SYSTEM_ROOT)/lib/libc/musl/arch/generic) # for 2.0.13 and newer
CLANGFLAGS += -fno-common #required for musl-libc
CLANGFLAGS += -mconstructor-aliases #lower .o file size
CLANGFLAGS += -fvisibility hidden -fno-threadsafe-statics -fgnuc-version=4.2.1
CLANGFLAGS += -D__WAJIC__ -D__EMSCRIPTEN__ -D_LIBCPP_ABI_VERSION=2

# Flags for wasm-ld
LDFLAGS += -no-entry -allow-undefined
LDFLAGS += -export=__wasm_call_ctors -export=main -export=__original_main -export=__main_argc_argv -export=__main_void -export=malloc -export=free

# Project Build flags, add defines from the make command line (e.g. D=MACRO=VALUE)
FLAGS := $(subst \\\, ,$(foreach F,$(subst \ ,\\\,$(D)),"-D$(F)"))

# Check if there are any source files, prefer file passed by the commandline (e.g. SRC=main.c)
SOURCES := $(if $(SRC),$(wildcard $(SRC)),$(wildcard *.c *.cpp *.cc))
-include sources.mk
SOURCES += $(foreach F, $(ADD_SOURCES), $(wildcard $(F)))
ifeq ($(strip $(SOURCES))$(filter %system.bc,$(MAKECMDGOALS)),)
  $(error No source files found for build)
endif
OUTBASE := $(OUTDIR)/$(if $(SRC),$(basename $(notdir $(firstword $(SRC)))),output)

# Compute tool paths
ifeq ($(wildcard $(subst $(strip ) ,\ ,$(LLVM_ROOT))/clang*),)
  $(error clang executables not found in set LLVM_ROOT path ($(LLVM_ROOT)). Set custom path in $(WAJIC_ROOT)LocalConfig.mk with LLVM_ROOT = $(if $(ISWIN),d:)/path/to/clang)
endif
ifeq ($(wildcard $(subst $(strip ) ,\ ,$(WASMOPT))),)
  undefine WASMOPT
endif

# Surround used commands with double quotes
CC := "$(LLVM_ROOT)/clang" -cc1
LD := "$(LLVM_ROOT)/wasm-ld"

all: $(OUTBASE).wasm
.PHONY: clean

clean:
	$(info Removing all build files ...)
	@$(if $(wildcard $(OUTDIR)),$(if $(ISWIN),rmdir /S /Q,rm -rf) "$(OUTDIR)" $(PIPETONULL))

# Generate a list of .o files to build, include dependency rules for source files, then compile files
OBJS := $(addprefix $(OUTDIR)/,$(notdir $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cc,%.o,$(SOURCES))))))
-include $(OBJS:%.o=%.d)
MAKEOBJ = $(OUTDIR)/$(basename $(notdir $(1))).o: $(1) ; $$(call COMPILE,$$@,$$<,$(2),$(3) $$(FLAGS))
$(foreach F,$(filter %.cc ,$(SOURCES)),$(eval $(call MAKEOBJ,$(F),$$(CC),$$(CXXFLAGS))))
$(foreach F,$(filter %.cpp,$(SOURCES)),$(eval $(call MAKEOBJ,$(F),$$(CC),$$(CXXFLAGS))))
$(foreach F,$(filter %.c  ,$(SOURCES)),$(eval $(call MAKEOBJ,$(F),$$(CC),$$(CFLAGS))))

$(OUTBASE).wasm : $(OBJS) $(WAJIC_ROOT)system/system.bc $(THIS_MAKEFILE)
	$(info Linking $@ ...)
	@$(LD) $(LDFLAGS) $(WAJIC_ROOT)system/system.bc $(OBJS) -o $@
	@$(if $(WASMOPT),"$(WASMOPT)" --legalize-js-interface $(WOPTFLAGS) $@ -o $@)
	@$(if $(NODE),"$(NODE)" "$(WAJIC_ROOT)wajicup.js" $(if $(filter $(BUILD),DEBUG),-nominify )$@ $@)

define COMPILE
	$(info $2)
	@$(if $(wildcard $(dir $1)),,$(shell mkdir "$(dir $1)"))
	@$3 $(CLANGFLAGS) $4 -dependency-file $(patsubst %.o,%.d,$1) -MT $1 -MP -o $1 $2
endef

#------------------------------------------------------------------------------------------------------
#if system.bc exists, don't even bother checking sources, build once and forget for now
ifeq ($(if $(wildcard $(WAJIC_ROOT)system/system.bc),1,0),0)
SYS_ADDS := libcxxabi/src/cxa_guard.cpp compiler-rt/lib/builtins/*.c libc/wasi-helpers.c pthread/library_pthread_stub.c
SYS_ADDS += $(if $(wildcard $(SYSTEM_ROOT)/lib/emmalloc.c),emmalloc.c,emmalloc.cpp) # due to change in 2.0.27
SYS_ADDS += $(if $(wildcard $(SYSTEM_ROOT)/lib/pthread/pthread_self_stub.c),pthread/pthread_self_stub.c) # due to change in 2.0.32
SYS_ADDS += $(if $(wildcard $(SYSTEM_ROOT)/lib/libc/emscripten_pthread.c),libc/emscripten_pthread.c) # needed for 2.0.13 until 2.0.25
SYS_ADDS += libcxx/$(if $(wildcard $(SYSTEM_ROOT)/lib/libcxx/src/*.cpp),src/)*.cpp # due to change in 2.0.13
SYS_MUSL := complex crypt ctype dirent errno fcntl fenv internal locale math misc mman multibyte prng regex select stat stdio stdlib string termios unistd
#SYS_MUSL += compat-emscripten time #uncomment if you need time formatting and C++ streams and locale

# Threads and exceptions are not supported, C++ streams and locale are not included on purpose because it can increase the output up to 500kb
SYS_IGNORE := thread.cpp exception.cpp
SYS_IGNORE += iostream.cpp strstream.cpp locale.cpp  #comment out if you need C++ streams and locale
SYS_IGNORE += abs.c acos.c acosf.c acosl.c asin.c asinf.c asinl.c atan.c atan2.c atan2f.c atan2l.c atanf.c atanl.c ceil.c ceilf.c ceill.c cos.c cosf.c cosl.c exp.c expf.c expl.c
SYS_IGNORE += fabs.c fabsf.c fabsl.c floor.c floorf.c floorl.c pow.c powf.c powl.c rintf.c round.c roundf.c sin.c sinf.c sinl.c sqrt.c sqrtf.c sqrtl.c tan.c tanf.c tanl.c
SYS_IGNORE += log.c log_data.c log_small.c logf.c logf_data.c logl.c log10.c log10f.c log10l.c log1p.c log1pf.c log1pl.c log2.c log2_data.c log2_small.c log2f.c log2f_data.c log2l.c
SYS_IGNORE += syscall.c wordexp.c initgroups.c getgrouplist.c popen.c _exit.c alarm.c usleep.c faccessat.c iconv.c
SYS_IGNORE += gcc_personality_v0.c # 1.39.20 and newer only
SYS_IGNORE += progname.c # 3.1.3 and newer only

SYS_SOURCES := $(filter-out $(SYS_IGNORE:%=\%/%),$(wildcard $(addprefix $(SYSTEM_ROOT)/lib/,$(SYS_ADDS) $(SYS_MUSL:%=libc/musl/src/%/*.c))))
SYS_SOURCES := $(subst $(SYSTEM_ROOT)/lib/,,$(SYS_SOURCES))

ifeq ($(findstring !,$(SYS_SOURCES)),!)
  $(error SYS_SOURCES contains a filename with a ! character in it - Unable to continue)
endif

SYS_MISSING := $(filter-out $(SYS_SOURCES) $(dir $(SYS_SOURCES)),$(subst *.c,,$(subst *.cpp,,$(SYS_ADDS))) $(SYS_MUSL:%=libc/musl/src/%/))
ifeq ($(if $(SYS_MISSING),1,0),1)
  $(error SYS_SOURCES missing the following files in $(SYSTEM_ROOT)/lib: $(SYS_MISSING))
endif

SYS_TEMP := $(or $(SYSTEM_TEMP),temp)
SYS_OLDFILES := $(filter-out $(subst /,!,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SYS_SOURCES)))),$(notdir $(wildcard $(SYS_TEMP)/*.o)))
$(foreach F,$(SYS_OLDFILES),$(shell $(if $(ISWIN),del "$(SYS_TEMP)\,rm "$(SYS_TEMP)/)$(F)" $(PIPETONULL)))

SYS_CXXFLAGS := -x c++ -Os -std=c++11 -fno-threadsafe-statics -fno-rtti -I$(SYSTEM_ROOT)/lib/libcxxabi/include
SYS_CXXFLAGS += -DNDEBUG -D_LIBCPP_BUILDING_LIBRARY -D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS

SYS_CFLAGS := -x c -Os -std=gnu11 -fno-threadsafe-statics -fno-builtin
SYS_CFLAGS += -DNDEBUG -Dunix -D__unix -D__unix__ -D_XOPEN_SOURCE
SYS_CFLAGS += -Wno-dangling-else -Wno-ignored-attributes -Wno-bitwise-op-parentheses -Wno-logical-op-parentheses -Wno-shift-op-parentheses
SYS_CFLAGS += -Wno-string-plus-int -Wno-unknown-pragmas -Wno-ignored-pragmas -Wno-shift-count-overflow -Wno-return-type -Wno-macro-redefined
SYS_CFLAGS += -Wno-unused-result -Wno-pointer-sign -Wno-implicit-function-declaration -Wno-int-conversion
SYS_CFLAGS += -isystem$(SYSTEM_ROOT)/lib/libc/musl/src/internal

SYS_MUSLFLAGS := -isystem$(SYSTEM_ROOT)/lib/libc/musl/src/include
SYS_MUSLFLAGS += $(if $(wildcard $(SYSTEM_ROOT)/lib/pthread/threading_internal.h),-I$(SYSTEM_ROOT)/lib/pthread) # for 3.0.1 and newer

SYS_MUSL_OBJS := $(addprefix $(SYS_TEMP)/,$(subst /,!,$(patsubst   %.c,%.o,$(filter libc/musl/%,$(SYS_SOURCES)) $(filter pthread/%,$(SYS_SOURCES)))))
SYS_CC_OBJS   := $(addprefix $(SYS_TEMP)/,$(subst /,!,$(patsubst   %.c,%.o,$(filter   %.c,$(filter-out pthread/%,$(filter-out libc/musl/%,$(SYS_SOURCES)))))))
SYS_CPP_OBJS  := $(addprefix $(SYS_TEMP)/,$(subst /,!,$(patsubst %.cpp,%.o,$(filter %.cpp,$(filter-out pthread/%,$(filter-out libc/musl/%,$(SYS_SOURCES)))))))
$(SYS_MUSL_OBJS) : ; $(call SYS_COMPILE,$@,$(subst !,/,$(patsubst $(SYS_TEMP)/%.o,$(SYSTEM_ROOT)/lib/%.c,$@)),$(CC),$(SYS_CFLAGS) $(SYS_MUSLFLAGS))
$(SYS_CC_OBJS)   : ; $(call SYS_COMPILE,$@,$(subst !,/,$(patsubst $(SYS_TEMP)/%.o,$(SYSTEM_ROOT)/lib/%.c,$@)),$(CC),$(SYS_CFLAGS))
$(SYS_CPP_OBJS)  : ; $(call SYS_COMPILE,$@,$(subst !,/,$(patsubst $(SYS_TEMP)/%.o,$(SYSTEM_ROOT)/lib/%.cpp,$@)),$(CC),$(SYS_CXXFLAGS))

define SYS_COMPILE
	$(info $2)
	@$(if $(wildcard $(dir $1)),,$(shell mkdir "$(dir $1)"))
	@$3 $4 $(CLANGFLAGS) -o $1 $2
endef

$(WAJIC_ROOT)system/system.bc : $(SYS_MUSL_OBJS) $(SYS_CPP_OBJS) $(SYS_CC_OBJS)
	$(info Creating archive $@ ...)
	@$(if $(wildcard $(dir $@)),,$(shell mkdir "$(dir $@)"))
	@$(LD) $(if $(ISWIN),"$(SYS_TEMP)/*.o",$(SYS_TEMP)/*.o) -r -o $@
	@$(if $(ISWIN),rmdir /S /Q,rm -rf) "$(SYS_TEMP)"
endif #need system.bc
#------------------------------------------------------------------------------------------------------
