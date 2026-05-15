
# Resolve paths relative to this toolchain file's location.
# Layout: external/wajic/cmake/clang-wajic.cmake
#   wajic_dir        = external/wajic          (parent of cmake/)
#   emscripten_system = external/emscripten/system (sibling of wajic/)
get_filename_component(wajic_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(emscripten_system "${CMAKE_CURRENT_LIST_DIR}/../../emscripten/system" ABSOLUTE)

set(WAJIC true)
set(WAJIC_DIR "${wajic_dir}" CACHE PATH "WAjic root directory")
set(WAJIC_SYSTEM_DIR "${emscripten_system}" CACHE PATH "Emscripten system directory")
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR wasm32)
set(CMAKE_C_COMPILER clang)
set(CMAKE_C_COMPILER_TARGET wasm32)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_COMPILER_TARGET wasm32)
set(CMAKE_EXECUTABLE_SUFFIX_C .wasm)
set(CMAKE_EXECUTABLE_SUFFIX_CXX .wasm)

add_compile_definitions(
  NDEBUG
  __WAJIC__
)

# Include WAjic headers always
include_directories(SYSTEM ${wajic_dir})

add_compile_options(
  -target wasm32
  -Os
  -fvisibility=hidden
  -fno-rtti
  -fno-exceptions
  -fno-threadsafe-statics
  -Wno-unused-command-line-argument
  -Wno-invalid-unevaluated-string
)

# Base link options (always needed for wasm targets)
add_link_options(
  -target wasm32
  -nostartfiles
  -nodefaultlibs
  -nostdinc
  -nostdinc++
  -fvisibility=hidden
)

# Core linker flags: always set them (override any stale cache value)
set(_wajic_link_flags
  -Xlinker -strip-all
  -Xlinker -gc-sections
  -Xlinker --no-entry
  -Xlinker -allow-undefined
  -Xlinker -export=__wasm_call_ctors
)
list(JOIN _wajic_link_flags " " _wajic_link_flags_str)
# Ensure flags are present regardless of cache state
if(NOT CMAKE_EXE_LINKER_FLAGS MATCHES "--no-entry")
  set(CMAKE_EXE_LINKER_FLAGS "${_wajic_link_flags_str}" CACHE STRING "" FORCE)
endif()

# Helper: link against emscripten system libraries (libc, libcxx, etc.)
# Call this on targets that need stdlib support.
function(wajic_use_system_libs target)
  # Get clang's resource directory for compiler-provided headers (stdint.h etc.)
  execute_process(
    COMMAND ${CMAKE_C_COMPILER} -target wasm32 -print-resource-dir
    OUTPUT_VARIABLE _clang_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  target_compile_definitions(${target} PRIVATE
    __EMSCRIPTEN__
    _LIBCPP_ABI_VERSION=2
  )
  # musl stdint.h uses __UINT*_C / __INT*_C macros normally defined by clang's stdint.h,
  # which isn't included due to -nostdinc. Define them for wasm32 ABI.
  target_compile_options(${target} PRIVATE
    "-D__INT8_C(c)=c"
    "-D__INT16_C(c)=c"
    "-D__INT32_C(c)=c"
    "-D__INT64_C(c)=c##LL"
    "-D__INTMAX_C(c)=c##LL"
    "-D__UINT8_C(c)=c"
    "-D__UINT16_C(c)=c"
    "-D__UINT32_C(c)=c##U"
    "-D__UINT64_C(c)=c##ULL"
    "-D__UINTMAX_C(c)=c##ULL"
  )
  target_include_directories(${target} SYSTEM PRIVATE
    ${emscripten_system}/lib/libcxx/include
    ${emscripten_system}/include/compat
    ${emscripten_system}/include
    ${emscripten_system}/lib/libc/musl/include
    ${emscripten_system}/lib/libc/musl/arch/emscripten
    ${emscripten_system}/lib/libc/musl/arch/generic
    ${_clang_resource_dir}/include
  )
  # Use LINK_FLAGS property — target_link_options doesn't pass -Xlinker per-target
  set_property(TARGET ${target} APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker ${emscripten_system}/system.bc -Xlinker -export=malloc -Xlinker -export=free"
  )
endfunction()

# Helper: export main and all its alternative entry points (per wajic.mk).
# Different LLVM versions emit main under different symbols.
function(wajic_export_main target)
  set_property(TARGET ${target} APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker -export=main -Xlinker -export=__original_main -Xlinker -export=__main_argc_argv -Xlinker -export=__main_void"
  )
endfunction()

# wajicup helper: run wajicup.js as a post-build step
function(wajicup target)
  cmake_parse_arguments(ARG "" "OUTPUT_DIR" "OPTIONS" ${ARGN})
  if(NOT ARG_OUTPUT_DIR)
    set(ARG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND node "${wajic_dir}/wajicup.js" ${ARG_OPTIONS}
      "$<TARGET_FILE:${target}>"
      "${ARG_OUTPUT_DIR}/$<TARGET_FILE_BASE_NAME:${target}>.wasm"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "WAjicUp: processing ${target}"
  )
endfunction()