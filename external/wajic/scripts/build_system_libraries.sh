#!/usr/bin/env bash
#
# build_system_libraries.sh
#
# Downloads the latest Emscripten system library sources, builds system.bc,
# and installs headers + system.bc into the output directory.
#
# This script is self-contained and handles modern Emscripten (3.x/5.x+) source
# layouts. It generates a proper sysroot version.h (which Emscripten's build
# system normally creates) and compiles libc (musl), libcxx, libcxxabi,
# emmalloc, compiler-rt builtins, and pthread stubs into a single system.bc.
#
# Directory layout assumed:
#   <parent>/
#     wajic/                    <- WAJIC_ROOT (this repo)
#       scripts/
#         build_system_libraries.sh   <- this script
#     emscripten/               <- OUTPUT_ROOT
#       system/
#         system.bc             <- installed output
#         include/
#         lib/
#
# Requirements:
#   - clang and wasm-ld (LLVM 11.0+) with wasm32 target support
#   - curl or wget
#   - unzip
#
# Usage:
#   ./scripts/build_system_libraries.sh [OPTIONS]
#
# Options:
#   --llvm-root <path>    Path to LLVM bin directory (containing clang/wasm-ld)
#   --output-dir <path>   Override output directory (default: ../emscripten/system)
#   --jobs <n>            Parallel jobs (default: nproc)
#   --keep-sources        Don't remove downloaded sources after build
#   --help                Show this help
#
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly WAJIC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly EXTERNAL_ROOT="$(cd "${WAJIC_ROOT}/.." && pwd)"

# Defaults (can be overridden via CLI flags or environment)
OUTPUT_DIR="${OUTPUT_DIR:-${EXTERNAL_ROOT}/emscripten/system}"
LLVM_ROOT="${LLVM_ROOT:-}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
KEEP_SOURCES=0

readonly TEMP_DIR="${WAJIC_ROOT}/.build_tmp"
readonly OBJ_DIR="${TEMP_DIR}/obj"
readonly EMSCRIPTEN_ARCHIVE_URL="https://github.com/emscripten-core/emscripten/archive/refs/heads/main.zip"
readonly EMSCRIPTEN_ARCHIVE="${TEMP_DIR}/emscripten-main.zip"
readonly EMSCRIPTEN_ROOT="${TEMP_DIR}/emscripten-main"
readonly SYSTEM_ROOT="${EMSCRIPTEN_ROOT}/system"

# ─── Helpers ─────────────────────────────────────────────────────────────────

log()   { printf '\033[1;32m[INFO]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# Print a distro-specific package install hint for the given tool names.
distro_hint() {
    local pkgs="$*"
    if command -v dnf >/dev/null 2>&1; then
        printf '  Install with: sudo dnf install %s\n' "${pkgs}" >&2
    elif command -v apt-get >/dev/null 2>&1; then
        printf '  Install with: sudo apt-get install %s\n' "${pkgs}" >&2
    elif command -v yum >/dev/null 2>&1; then
        printf '  Install with: sudo yum install %s\n' "${pkgs}" >&2
    fi
}

# Find an LLVM tool by name.  Tries the plain name, then versioned variants
# (18, 19, 17, 16, 15), then common distro-specific LLVM bin directories.
# Prints the resolved path; returns 0 on success, 1 on failure.
find_llvm_tool() {
    local tool="$1"
    local candidate
    # 1. Plain name in PATH
    candidate="$(command -v "${tool}" 2>/dev/null || true)"
    [[ -x "${candidate}" ]] && { printf '%s' "${candidate}"; return 0; }
    # 2. Versioned names in PATH (newest-first so we prefer the latest)
    for ver in 19 18 17 16 15 14; do
        candidate="$(command -v "${tool}-${ver}" 2>/dev/null || true)"
        [[ -x "${candidate}" ]] && { printf '%s' "${candidate}"; return 0; }
    done
    # 3. Common distro LLVM directories (Fedora, RHEL, Ubuntu LLVM PPA)
    for llvm_dir in \
            /usr/lib64/llvm19/bin /usr/lib64/llvm18/bin /usr/lib64/llvm17/bin \
            /usr/lib64/llvm/bin \
            /usr/lib/llvm-19/bin /usr/lib/llvm-18/bin /usr/lib/llvm-17/bin \
            /usr/lib/llvm/bin; do
        candidate="${llvm_dir}/${tool}"
        [[ -x "${candidate}" ]] && { printf '%s' "${candidate}"; return 0; }
    done
    return 1
}

usage() {
    sed -n '/^# Usage:/,/^#$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    sed -n '/^# Options:/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit 0
}

cleanup() {
    if [[ "${KEEP_SOURCES}" -eq 0 && -d "${TEMP_DIR}" ]]; then
        log "Cleaning up temporary files..."
        rm -rf "${TEMP_DIR}"
    fi
}

# Check if a source file should be ignored (pure bash, no subprocess).
# Uses the global IGNORE_RE variable.
should_ignore() {
    [[ "$1" =~ ${IGNORE_RE} ]]
}

# Derive object filename: strip SYSTEM_ROOT/lib/ prefix, replace / with !, change extension.
# Args: $1=source path, $2=source extension (c or cpp)
make_obj_name() {
    local rel="${1#"${SYSTEM_ROOT}/lib/"}"
    printf '%s/%s.o' "${OBJ_DIR}" "${rel//\//'!'}"
}

# ─── Argument Parsing ────────────────────────────────────────────────────────

while (( $# )); do
    case "$1" in
        --llvm-root)
            [[ -n "${2:-}" ]] || die "--llvm-root requires an argument"
            LLVM_ROOT="$2"; shift 2 ;;
        --output-dir)
            [[ -n "${2:-}" ]] || die "--output-dir requires an argument"
            OUTPUT_DIR="$2"; shift 2 ;;
        --jobs)
            [[ -n "${2:-}" ]] || die "--jobs requires an argument"
            JOBS="$2"; shift 2 ;;
        --keep-sources)
            KEEP_SOURCES=1; shift ;;
        --help|-h)
            usage ;;
        *)
            die "Unknown option: $1" ;;
    esac
done

# ─── Prerequisite Checks ────────────────────────────────────────────────────

log "Checking prerequisites..."

# Find LLVM tools
if [[ -n "${LLVM_ROOT}" ]]; then
    CLANG="${LLVM_ROOT}/clang"
    # wasm-ld lives alongside clang in the same LLVM bin dir; also accept ld.lld
    if [[ -x "${LLVM_ROOT}/wasm-ld" ]]; then
        WASM_LD="${LLVM_ROOT}/wasm-ld"
    elif [[ -x "${LLVM_ROOT}/ld.lld" ]]; then
        WASM_LD="${LLVM_ROOT}/ld.lld"
    else
        WASM_LD="${LLVM_ROOT}/wasm-ld"   # will fail the -x check below
    fi
else
    CLANG="$(find_llvm_tool clang || true)"
    # Prefer wasm-ld; fall back to ld.lld (some Fedora lld packages)
    WASM_LD="$(find_llvm_tool wasm-ld || find_llvm_tool ld.lld || true)"
    if [[ -n "${CLANG}" ]]; then
        LLVM_ROOT="$(dirname "${CLANG}")"
    fi
fi

if [[ ! -x "${CLANG}" ]]; then
    printf '\033[1;31m[ERROR]\033[0m clang not found. Set --llvm-root or add to PATH.\n' >&2
    distro_hint "clang"
    exit 1
fi
if [[ ! -x "${WASM_LD}" ]]; then
    printf '\033[1;31m[ERROR]\033[0m wasm-ld not found. Set --llvm-root or add to PATH.\n' >&2
    distro_hint "lld"
    exit 1
fi

# Verify clang supports the wasm32 target.
# --print-targets lists either "wasm32" (target arch) or "WebAssembly" (backend
# name) depending on the LLVM build; accept either.
if ! "${CLANG}" --print-targets 2>/dev/null | grep -qiE '(wasm32|WebAssembly)'; then
    die "clang at '${CLANG}' does not support the wasm32 target."
fi

log "Using clang: ${CLANG} ($(${CLANG} --version 2>&1 | head -1))"
log "Using wasm-ld: ${WASM_LD}"
log "Output directory: ${OUTPUT_DIR}"

if ! command -v unzip >/dev/null 2>&1; then
    printf '\033[1;31m[ERROR]\033[0m unzip not found.\n' >&2
    distro_hint "unzip"
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    DOWNLOAD="curl"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOAD="wget"
else
    die "Neither curl nor wget found. Install one of them."
fi

# ─── Download Emscripten Sources ────────────────────────────────────────────

trap cleanup EXIT

mkdir -p "${TEMP_DIR}"

if [[ -d "${SYSTEM_ROOT}" ]]; then
    log "Emscripten system sources already present, skipping download."
else
    log "Downloading latest Emscripten sources (system/ + version info)..."
    if [[ "${DOWNLOAD}" == "curl" ]]; then
        curl -fSL --retry 3 --retry-delay 5 -o "${EMSCRIPTEN_ARCHIVE}" "${EMSCRIPTEN_ARCHIVE_URL}"
    else
        wget --tries=3 --waitretry=5 -O "${EMSCRIPTEN_ARCHIVE}" "${EMSCRIPTEN_ARCHIVE_URL}"
    fi

    log "Extracting system directory and version info..."
    unzip -q -o "${EMSCRIPTEN_ARCHIVE}" \
        "emscripten-main/system/*" \
        "emscripten-main/emscripten-version.txt" \
        -d "${TEMP_DIR}"

    [[ -d "${SYSTEM_ROOT}" ]] || die "Failed to extract system directory from archive."

    rm -f "${EMSCRIPTEN_ARCHIVE}"
    log "Extraction complete."
fi

# Verify structure
[[ -d "${SYSTEM_ROOT}/lib" ]]     || die "Missing ${SYSTEM_ROOT}/lib"
[[ -d "${SYSTEM_ROOT}/include" ]] || die "Missing ${SYSTEM_ROOT}/include"

# ─── Generate version.h ─────────────────────────────────────────────────────
# Emscripten's build system generates this file in its sysroot cache.
# The source tree only has a guard #error. We replicate what
# tools/system_libs.py does: generate proper version macros.

log "Generating emscripten/version.h..."

VERSION_TXT="${EMSCRIPTEN_ROOT}/emscripten-version.txt"
if [[ -f "${VERSION_TXT}" ]]; then
    VERSION_STRING="$(tr -d '[:space:]' < "${VERSION_TXT}")"
    IFS='.-' read -r VMAJOR VMINOR VTINY _ <<< "${VERSION_STRING}"
else
    warn "emscripten-version.txt not found, using defaults"
    VMAJOR=5; VMINOR=0; VTINY=0
fi

log "  Emscripten version: ${VMAJOR}.${VMINOR}.${VTINY}"

cat > "${SYSTEM_ROOT}/include/emscripten/version.h" <<EOF
/* Generated by build_system_libraries.sh from emscripten-version.txt */
#define __EMSCRIPTEN_MAJOR__ ${VMAJOR}
#define __EMSCRIPTEN_MINOR__ ${VMINOR}
#define __EMSCRIPTEN_TINY__ ${VTINY}

#define __EMSCRIPTEN_major__ __EMSCRIPTEN_MAJOR__
#define __EMSCRIPTEN_minor__ __EMSCRIPTEN_MINOR__
#define __EMSCRIPTEN_tiny__ __EMSCRIPTEN_TINY__
EOF

# ─── Build System Libraries ─────────────────────────────────────────────────

log "Building system libraries with ${JOBS} parallel jobs..."

rm -rf "${OBJ_DIR}"
mkdir -p "${OBJ_DIR}"

# ─── Compiler flags ─────────────────────────────────────────────────────────
# These match wajic.mk optimal settings adapted for modern clang -cc1

CC_BASE="${CLANG} -cc1 -triple wasm32-unknown-emscripten -emit-obj"
CC_BASE+=" -fcolor-diagnostics -fno-common -mconstructor-aliases"
CC_BASE+=" -fvisibility=hidden -fno-threadsafe-statics -fgnuc-version=4.2.1"
CC_BASE+=" -D__WAJIC__ -D__EMSCRIPTEN__ -D_LIBCPP_ABI_VERSION=2 -DNDEBUG"

# Emscripten's musl stdint.h uses GCC-style __UINT*_C / __INT*_C function-like
# macros that clang doesn't predefine. Create a compat header with them.
readonly COMPAT_HDR="${TEMP_DIR}/wajic_compat_macros.h"
cat > "${COMPAT_HDR}" <<'MACROEOF'
#ifndef _WAJIC_COMPAT_MACROS_H
#define _WAJIC_COMPAT_MACROS_H
#define __INT8_C(c) c
#define __INT16_C(c) c
#define __INT32_C(c) c
#define __INT64_C(c) c ## LL
#define __UINT8_C(c) c
#define __UINT16_C(c) c
#define __UINT32_C(c) c ## U
#define __UINT64_C(c) c ## ULL
#define __INTMAX_C(c) c ## LL
#define __UINTMAX_C(c) c ## ULL
#endif
MACROEOF
CC_BASE+=" -include${COMPAT_HDR}"

# System include paths (order matches wajic.mk)
# wajic.mk uses either/or: include/libcxx OR lib/libcxx/include (changed in 2.0.13)
# Similarly: include/libc OR lib/libc/musl/include
INCLUDES=""
if [[ -d "${SYSTEM_ROOT}/include/libcxx" ]]; then
    INCLUDES+=" -isystem${SYSTEM_ROOT}/include/libcxx"
else
    INCLUDES+=" -isystem${SYSTEM_ROOT}/lib/libcxx/include"
fi
[[ -d "${SYSTEM_ROOT}/include/compat" ]] && INCLUDES+=" -isystem${SYSTEM_ROOT}/include/compat"
INCLUDES+=" -isystem${SYSTEM_ROOT}/include"
if [[ -d "${SYSTEM_ROOT}/include/libc" ]]; then
    INCLUDES+=" -isystem${SYSTEM_ROOT}/include/libc"
else
    INCLUDES+=" -isystem${SYSTEM_ROOT}/lib/libc/musl/include"
fi
INCLUDES+=" -isystem${SYSTEM_ROOT}/lib/libc/musl/arch/emscripten"
[[ -d "${SYSTEM_ROOT}/lib/libc/musl/arch/generic" ]] && INCLUDES+=" -isystem${SYSTEM_ROOT}/lib/libc/musl/arch/generic"

# C flags — applied to ALL C files (musl, compiler-rt, emmalloc, wasi-helpers).
# Matches wajic.mk SYS_CFLAGS which includes musl/src/internal for all C files.
CFLAGS="-x c -Os -std=gnu11 -fno-threadsafe-statics -fno-builtin"
CFLAGS+=" -DNDEBUG -Dunix -D__unix -D__unix__ -D_XOPEN_SOURCE"
CFLAGS+=" -Wno-dangling-else -Wno-ignored-attributes -Wno-bitwise-op-parentheses"
CFLAGS+=" -Wno-logical-op-parentheses -Wno-shift-op-parentheses"
CFLAGS+=" -Wno-string-plus-int -Wno-unknown-pragmas -Wno-ignored-pragmas"
CFLAGS+=" -Wno-shift-count-overflow -Wno-return-type -Wno-macro-redefined"
CFLAGS+=" -Wno-unused-result -Wno-pointer-sign -Wno-implicit-function-declaration"
CFLAGS+=" -Wno-int-conversion"
# musl internal headers needed by ALL C files (matches wajic.mk SYS_CFLAGS)
CFLAGS+=" -isystem${SYSTEM_ROOT}/lib/libc/musl/src/internal"

# Additional musl-specific includes — only for musl and pthread sources.
# Matches wajic.mk SYS_MUSLFLAGS.
# -I (not -isystem) so musl/src/include/features.h overrides the public one.
MUSL_EXTRA="-I${SYSTEM_ROOT}/lib/libc/musl/src/include"
# pthread threading_internal.h needed for emscripten 3.0.1+
[[ -f "${SYSTEM_ROOT}/lib/pthread/threading_internal.h" ]] && MUSL_EXTRA+=" -I${SYSTEM_ROOT}/lib/pthread"

# C++ flags for libcxx/libcxxabi
# Emscripten 5.x libcxx uses C++20 features: constinit, std::ranges, std::to_chars.
# c++11 or c++17 produce compile errors in algorithm.cpp, new_handler.cpp, ios.cpp, etc.
CXXFLAGS="-x c++ -Os -std=c++20 -fno-threadsafe-statics -fno-rtti"
CXXFLAGS+=" -DNDEBUG -D_LIBCPP_BUILDING_LIBRARY -D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS"
CXXFLAGS+=" -I${SYSTEM_ROOT}/lib/libcxxabi/include"

# ─── Collect source files ────────────────────────────────────────────────────

# Musl subdirectories to compile (matches wajic.mk SYS_MUSL, minus syscall-heavy dirs).
#
# The following directories are intentionally excluded for Emscripten 5.x / browser targets:
#   dirent, fcntl, mman, select, stat, termios, unistd
#
# In Emscripten 5.x these files no longer compile for wasm32-unknown-emscripten because
# the SYS_* and __syscall_* constants were removed from the WASM sysroot.  More critically,
# 'unistd/readv.c' imports '__wasi_fd_read' from the legacy 'env' module while
# 'stdio/__stdio_read.c' imports it from 'wasi_snapshot_preview1' — wasm-ld treats this
# as a fatal module mismatch.  None of these directories are needed in a WAjic browser
# application: the JavaScript runtime provides all real I/O via its own host functions.
#
# 'time' and 'compat-emscripten' are also excluded (see original wajic.mk comment).
# Uncomment the line below if you need time formatting or C++ streams/locale:
#   MUSL_DIRS+=" compat-emscripten time"
MUSL_DIRS="complex crypt ctype errno fenv internal locale math misc multibyte prng regex stdio stdlib string"

# Ignored files — threads/exceptions unsupported, iostream/locale excluded for size.
# Matches wajic.mk SYS_IGNORE list.
IGNORE_RE='/(thread|exception|iostream|strstream|locale)\.cpp$'
IGNORE_RE+='|/shared_mutex\.cpp$'
IGNORE_RE+='|/barrier\.cpp$'
IGNORE_RE+='|/latch\.cpp$'
IGNORE_RE+='|/support/(ibm|win32)/'
IGNORE_RE+='|/filesystem/int128_builtins\.cpp$'
IGNORE_RE+='|/filesystem/directory_iterator\.cpp$'
IGNORE_RE+='|/filesystem/operations\.cpp$'
IGNORE_RE+='|/experimental/'
# Wasm builtins (provided natively by the WebAssembly VM)
IGNORE_RE+='|/(abs|acos|acosf|acosl|asin|asinf|asinl|atan|atan2|atan2f|atan2l|atanf|atanl)\.c$'
IGNORE_RE+='|/(ceil|ceilf|ceill|cos|cosf|cosl|exp|expf|expl)\.c$'
IGNORE_RE+='|/(fabs|fabsf|fabsl|floor|floorf|floorl|pow|powf|powl)\.c$'
IGNORE_RE+='|/(rintf|round|roundf|sin|sinf|sinl|sqrt|sqrtf|sqrtl|tan|tanf|tanl)\.c$'
IGNORE_RE+='|/(log|log_data|log_small|logf|logf_data|logl)\.c$'
IGNORE_RE+='|/(log10|log10f|log10l|log1p|log1pf|log1pl)\.c$'
IGNORE_RE+='|/(log2|log2_data|log2_small|log2f|log2f_data|log2l)\.c$'
# Problematic/unnecessary system calls
IGNORE_RE+='|/(syscall|wordexp|initgroups|getgrouplist|popen|_exit|alarm|usleep|faccessat|iconv)\.c$'
IGNORE_RE+='|/(gcc_personality_v0|progname)\.c$'

# musl misc: OS-level syscall wrappers removed/unavailable in Emscripten 5.x wasm32 sysroot.
# None of these are callable from a browser WAjic application.
IGNORE_RE+='|/misc/(getauxval|getpriority|setpriority|getresgid|getresuid|setresgid|setresuid)\.c$'
IGNORE_RE+='|/misc/(getrlimit|setrlimit|getrusage|ioctl|pty|setdomainname|uname|prctl)\.c$'

# musl stdio: filesystem operations that use WASI paths unavailable in WAjic.
IGNORE_RE+='|/stdio/(remove|rename|tempnam|tmpnam)\.c$'

# compiler-rt: x87 80-bit float builtins ("xf" = extended float).  wasm32 has
# no 80-bit float type so these will never link and cause spurious errors.
IGNORE_RE+='|/builtins/(divxc3|mulxc3|powixf2|extendhfxf2|truncxfhf2)\.c$'
IGNORE_RE+='|/builtins/fix(uns)?xf[a-z]*\.c$'
IGNORE_RE+='|/builtins/float[a-z]*xf\.c$'
# compiler-rt: bfloat16 truncation builtins — not yet supported by this clang target config.
IGNORE_RE+='|/builtins/trunc[sd]fbf2\.c$'
# compiler-rt: lock-free C11 atomic flag/fence builtins and CRT init — not needed for wasm.
IGNORE_RE+='|/builtins/(atomic_flag_(clear|test_and_set)[^/]*|atomic_(signal|thread)_fence|crtbegin)\.c$'

# libcxx filesystem: require OS file-system access not available in a browser.
# The three files excluded by wajic.mk are already above; add the rest added in libcxx 15+.
IGNORE_RE+='|/libcxx/src/filesystem/(directory_entry|filesystem_error|path)\.cpp$'
# libcxx: future/promise needs full threading; pstl/libdispatch needs Apple GCD.
IGNORE_RE+='|/libcxx/src/future\.cpp$'
IGNORE_RE+='|/pstl/libdispatch\.cpp$'
# libcxx: C++23 std::print — not available until libc++ is built with c++23 support.
IGNORE_RE+='|/libcxx/src/print\.cpp$'
# libcxx: C++23 std::expected — safe to skip; WAjic apps use C99/C++11 error handling.
IGNORE_RE+='|/libcxx/src/expected\.cpp$'

# Write compilation commands to a job file for xargs
readonly JOBFILE="${TEMP_DIR}/compile_jobs.txt"
: > "${JOBFILE}"

# Musl sources
for dir in ${MUSL_DIRS}; do
    srcdir="${SYSTEM_ROOT}/lib/libc/musl/src/${dir}"
    [[ -d "${srcdir}" ]] || continue
    for src in "${srcdir}"/*.c; do
        [[ -f "${src}" ]] || continue
        should_ignore "${src}" && continue
        obj="$(make_obj_name "${src}" c)"
        printf '%s\n' "${CC_BASE} ${MUSL_EXTRA} ${INCLUDES} ${CFLAGS} -o ${obj} ${src}" >> "${JOBFILE}"
    done
done

# Pthread stubs (compiled with musl flags per wajic.mk)
for src in "${SYSTEM_ROOT}"/lib/pthread/*stub*.c; do
    [[ -f "${src}" ]] || continue
    obj="$(make_obj_name "${src}" c)"
    printf '%s\n' "${CC_BASE} ${MUSL_EXTRA} ${INCLUDES} ${CFLAGS} -o ${obj} ${src}" >> "${JOBFILE}"
done

# Pthread data-symbol stubs (Emscripten 5.x defines these in library_pthread.c,
# which we skip; they MUST be DATA definitions because wasm-ld -r cannot resolve
# data relocations (R_WASM_MEMORY_ADDR_*) via --allow-undefined).
readonly PTHREAD_GLOBALS_C="${TEMP_DIR}/wajic_pthread_globals.c"
cat > "${PTHREAD_GLOBALS_C}" <<'PTEOF'
/* WAjic pthread globals stub - Emscripten 5.x single-threaded mode.
 * Provides the data symbols defined in library_pthread.c that some musl/misc
 * files reference even in single-threaded builds. */
#include <stddef.h>
size_t __default_guardsize = 4096;
size_t __default_stacksize = 81920;
PTEOF
printf '%s\n' "${CC_BASE} ${INCLUDES} ${CFLAGS} -o ${OBJ_DIR}/wajic_pthread_globals.o ${PTHREAD_GLOBALS_C}" >> "${JOBFILE}"

# compiler-rt builtins
for src in "${SYSTEM_ROOT}"/lib/compiler-rt/lib/builtins/*.c; do
    [[ -f "${src}" ]] || continue
    should_ignore "${src}" && continue
    obj="$(make_obj_name "${src}" c)"
    printf '%s\n' "${CC_BASE} ${INCLUDES} ${CFLAGS} -o ${obj} ${src}" >> "${JOBFILE}"
done

# wasi-helpers
if [[ -f "${SYSTEM_ROOT}/lib/libc/wasi-helpers.c" ]]; then
    printf '%s\n' "${CC_BASE} ${INCLUDES} ${CFLAGS} -o ${OBJ_DIR}/libc!wasi-helpers.o ${SYSTEM_ROOT}/lib/libc/wasi-helpers.c" >> "${JOBFILE}"
fi

# emscripten_pthread.c (needed for Emscripten 2.0.13–2.0.25)
if [[ -f "${SYSTEM_ROOT}/lib/libc/emscripten_pthread.c" ]]; then
    printf '%s\n' "${CC_BASE} ${MUSL_EXTRA} ${INCLUDES} ${CFLAGS} -o ${OBJ_DIR}/libc!emscripten_pthread.o ${SYSTEM_ROOT}/lib/libc/emscripten_pthread.c" >> "${JOBFILE}"
fi

# emmalloc (changed from .c to .cpp in Emscripten 2.0.27)
if [[ -f "${SYSTEM_ROOT}/lib/emmalloc.c" ]]; then
    printf '%s\n' "${CC_BASE} ${INCLUDES} ${CFLAGS} -o ${OBJ_DIR}/emmalloc.o ${SYSTEM_ROOT}/lib/emmalloc.c" >> "${JOBFILE}"
elif [[ -f "${SYSTEM_ROOT}/lib/emmalloc.cpp" ]]; then
    printf '%s\n' "${CC_BASE} ${INCLUDES} ${CXXFLAGS} -o ${OBJ_DIR}/emmalloc.o ${SYSTEM_ROOT}/lib/emmalloc.cpp" >> "${JOBFILE}"
fi

# libcxxabi/cxa_guard
# cxa_guard_impl.h includes "include/atomic_support.h" relative to libcxx/src/
# (the header lives at libcxx/src/include/atomic_support.h). Add libcxx/src as -I.
if [[ -f "${SYSTEM_ROOT}/lib/libcxxabi/src/cxa_guard.cpp" ]]; then
    CXA_GUARD_INCLUDES="${INCLUDES} -I${SYSTEM_ROOT}/lib/libcxx/src"
    printf '%s\n' "${CC_BASE} ${CXA_GUARD_INCLUDES} ${CXXFLAGS} -o ${OBJ_DIR}/libcxxabi!src!cxa_guard.o ${SYSTEM_ROOT}/lib/libcxxabi/src/cxa_guard.cpp" >> "${JOBFILE}"
fi

# libcxx sources (src/ subdirectory added in Emscripten 2.0.13)
LIBCXX_SRC_DIR=""
if [[ -d "${SYSTEM_ROOT}/lib/libcxx/src" ]]; then
    LIBCXX_SRC_DIR="${SYSTEM_ROOT}/lib/libcxx/src"
elif compgen -G "${SYSTEM_ROOT}/lib/libcxx/*.cpp" >/dev/null 2>&1; then
    LIBCXX_SRC_DIR="${SYSTEM_ROOT}/lib/libcxx"
fi

if [[ -n "${LIBCXX_SRC_DIR}" ]]; then
    # ryu/ and charconv files use relative includes like '#include "include/ryu/common.h"'
    # that resolve from libcxx/src/.  Add LIBCXX_SRC_DIR as an explicit -I include path.
    # Also add LIBCXX_SRC_DIR/include for internal headers like shared/fp_bits.h that are
    # included as "shared/fp_bits.h" from within libcxx/src/include/ implementation headers.
    # fp_bits.h actually lives at llvm-libc/shared/fp_bits.h in Emscripten 5.x — add that too.
    LIBCXX_INCLUDES="${INCLUDES} -I${LIBCXX_SRC_DIR} -I${LIBCXX_SRC_DIR}/include"
    [[ -d "${SYSTEM_ROOT}/lib/llvm-libc" ]] && LIBCXX_INCLUDES+=" -I${SYSTEM_ROOT}/lib/llvm-libc"
    while IFS= read -r -d '' src; do
        should_ignore "${src}" && continue
        obj="$(make_obj_name "${src}" cpp)"
        printf '%s\n' "${CC_BASE} ${LIBCXX_INCLUDES} ${CXXFLAGS} -o ${obj} ${src}" >> "${JOBFILE}"
    done < <(find "${LIBCXX_SRC_DIR}" -name '*.cpp' -print0)
fi

TOTAL_JOBS=$(wc -l < "${JOBFILE}")
log "  Total compilation jobs: ${TOTAL_JOBS}"

if [[ "${TOTAL_JOBS}" -eq 0 ]]; then
    die "No source files found to compile. Check SYSTEM_ROOT: ${SYSTEM_ROOT}"
fi

# ─── Execute compilation in parallel ────────────────────────────────────────

log "Compiling (this may take a few minutes)..."

readonly ERRLOG="${TEMP_DIR}/compile_errors.log"
: > "${ERRLOG}"

# Use xargs for parallel execution.
# -d '\n' ensures lines with spaces are handled correctly.
# '{}' is the standard replacement string (avoids collisions with file paths).
xargs -P "${JOBS}" -d '\n' -I '{}' bash -c '{}' < "${JOBFILE}" 2>>"${ERRLOG}" || true

TOTAL_OBJS=$(find "${OBJ_DIR}" -name '*.o' 2>/dev/null | wc -l)
FAIL_COUNT=$((TOTAL_JOBS - TOTAL_OBJS))

if [[ ${TOTAL_OBJS} -eq 0 ]]; then
    die "All compilations failed. First errors:\n$(head -20 "${ERRLOG}")"
fi

if [[ ${FAIL_COUNT} -gt 0 ]]; then
    warn "${FAIL_COUNT}/${TOTAL_JOBS} files failed to compile (non-critical files may be skipped)."
    warn "First errors:"
    grep "error:" "${ERRLOG}" 2>/dev/null | head -5 || true
fi

log "  Successfully compiled: ${TOTAL_OBJS}/${TOTAL_JOBS} objects"

# ─── Link into system.bc ────────────────────────────────────────────────────

log "Linking ${TOTAL_OBJS} objects into system.bc..."

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Use wasm-ld -r to create a relocatable combined object directly in the output dir
"${WASM_LD}" "${OBJ_DIR}"/*.o -r -o "${OUTPUT_DIR}/system.bc"

[[ -f "${OUTPUT_DIR}/system.bc" ]] || die "Linking failed: system.bc was not created."

log "system.bc created: $(du -h "${OUTPUT_DIR}/system.bc" | cut -f1)"

# ─── Install headers into output directory ──────────────────────────────────

log "Installing headers into ${OUTPUT_DIR}..."

# Copy include/ tree (emscripten headers, GL, compat, etc.)
cp -a "${SYSTEM_ROOT}/include" "${OUTPUT_DIR}/include"

# Copy required lib/ header subdirectories
mkdir -p "${OUTPUT_DIR}/lib"

# libcxx headers
if [[ -d "${SYSTEM_ROOT}/lib/libcxx/include" ]]; then
    mkdir -p "${OUTPUT_DIR}/lib/libcxx"
    cp -a "${SYSTEM_ROOT}/lib/libcxx/include" "${OUTPUT_DIR}/lib/libcxx/include"
fi

# musl headers and arch
mkdir -p "${OUTPUT_DIR}/lib/libc/musl"
cp -a "${SYSTEM_ROOT}/lib/libc/musl/include" "${OUTPUT_DIR}/lib/libc/musl/include"
cp -a "${SYSTEM_ROOT}/lib/libc/musl/arch" "${OUTPUT_DIR}/lib/libc/musl/arch"

# pthread headers
if [[ -d "${SYSTEM_ROOT}/lib/pthread" ]]; then
    mkdir -p "${OUTPUT_DIR}/lib/pthread"
    find "${SYSTEM_ROOT}/lib/pthread" -name '*.h' -exec cp {} "${OUTPUT_DIR}/lib/pthread/" \;
fi

log "Headers installed."

# ─── Verification ───────────────────────────────────────────────────────────

log "Running verification test..."

readonly TEST_SRC="${TEMP_DIR}/test.c"
readonly TEST_OBJ="${TEMP_DIR}/test.o"
readonly TEST_WASM="${TEMP_DIR}/test.wasm"

cat > "${TEST_SRC}" <<'TESTEOF'
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(void) {
    char *p = (char *)malloc(64);
    if (p) {
        memset(p, 0, 64);
        snprintf(p, 64, "WAjic system libraries OK");
        free(p);
    }
    return 0;
}
TESTEOF

# Build test include flags — use same conditional logic as the main build
T_INC=""
if [[ -d "${OUTPUT_DIR}/include/libcxx" ]]; then
    T_INC+=" -isystem${OUTPUT_DIR}/include/libcxx"
elif [[ -d "${OUTPUT_DIR}/lib/libcxx/include" ]]; then
    T_INC+=" -isystem${OUTPUT_DIR}/lib/libcxx/include"
fi
[[ -d "${OUTPUT_DIR}/include/compat" ]]                && T_INC+=" -isystem${OUTPUT_DIR}/include/compat"
T_INC+=" -isystem${OUTPUT_DIR}/include"
if [[ -d "${OUTPUT_DIR}/include/libc" ]]; then
    T_INC+=" -isystem${OUTPUT_DIR}/include/libc"
elif [[ -d "${OUTPUT_DIR}/lib/libc/musl/include" ]]; then
    T_INC+=" -isystem${OUTPUT_DIR}/lib/libc/musl/include"
fi
[[ -d "${OUTPUT_DIR}/lib/libc/musl/arch/emscripten" ]] && T_INC+=" -isystem${OUTPUT_DIR}/lib/libc/musl/arch/emscripten"
[[ -d "${OUTPUT_DIR}/lib/libc/musl/arch/generic" ]]    && T_INC+=" -isystem${OUTPUT_DIR}/lib/libc/musl/arch/generic"

# Use -std=gnu11 to match the main build (not c99)
${CLANG} -cc1 -triple wasm32-unknown-emscripten -emit-obj \
    ${T_INC} \
    -fno-common -fvisibility=hidden -fno-threadsafe-statics -fgnuc-version=4.2.1 \
    -D__WAJIC__ -D__EMSCRIPTEN__ -D_LIBCPP_ABI_VERSION=2 -DNDEBUG \
    -Os -x c -std=gnu11 \
    -o "${TEST_OBJ}" "${TEST_SRC}"

${WASM_LD} --strip-all --gc-sections --no-entry --allow-undefined \
    --export=__wasm_call_ctors --export=main --export=malloc --export=free \
    "${OUTPUT_DIR}/system.bc" "${TEST_OBJ}" \
    -o "${TEST_WASM}"

if [[ -f "${TEST_WASM}" ]]; then
    log "Verification PASSED (test.wasm: $(du -h "${TEST_WASM}" | cut -f1))"
else
    die "Verification FAILED: could not link test program"
fi

# ─── Summary ────────────────────────────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════════════════"
log " Build complete!"
log ""
log " Emscripten version: ${VMAJOR}.${VMINOR}.${VTINY}"
log " system.bc:          ${OUTPUT_DIR}/system.bc ($(du -h "${OUTPUT_DIR}/system.bc" | cut -f1))"
log " Headers:            ${OUTPUT_DIR}/include/"
log "                     ${OUTPUT_DIR}/lib/"
log ""
log " Build settings (matching wajic.mk):"
log "   Optimization: -Os (size + performance)"
log "   C standard:   gnu11"
log "   C++ standard: C++20, -fno-rtti"
log "   Threading:    disabled (-fno-threadsafe-statics)"
log "   Exceptions:   excluded"
log "   iostream/locale: excluded (saves ~500KB)"
log "   Dead code:    stripped (-gc-sections)"
log "═══════════════════════════════════════════════════════════════"
