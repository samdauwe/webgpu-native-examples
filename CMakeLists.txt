cmake_minimum_required(VERSION 3.17)
set(NAME webgpu-native-examples)
project(${NAME} LANGUAGES C CXX)

# Architecture string.
set(ARCH "${CMAKE_SYSTEM_PROCESSOR}" CACHE STRING "Override CPU architecture")
string(REGEX REPLACE "amd64" "x86_64" KERNEL_ARCH "${ARCH}")

if("${KERNEL_ARCH}" STREQUAL "x86_64")
  set(ARCH "x64")
else()
  message(FATAL_ERROR "Unsupported ARCH: ${KERNEL_ARCH}")
endif()

# Set directory locations (allowing us to move directories easily)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}${ARCH})

# ==============================================================================
# Dependencies
# ==============================================================================

# WebGPU native
add_subdirectory(lib/wgpu_native)

# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ktx
set(KTX_SOURCES
    external/ktx/lib/checkheader.c
    external/ktx/lib/filestream.c
    external/ktx/lib/hashlist.c
    external/ktx/lib/hashtable.c
    external/ktx/lib/memstream.c
    external/ktx/lib/swap.c
    external/ktx/lib/texture.c
)

# rply
set(RPLY_SOURCES
    external/rply/rply.c
)

# cimgui
set(CIMGUI_SOURCES
    external/cimgui/cimgui.cpp
    external/cimgui/imgui/imgui.cpp
    external/cimgui/imgui/imgui_draw.cpp
    external/cimgui/imgui/imgui_widgets.cpp
    external/cimgui/imgui/imgui_demo.cpp
)

# ffmpeg
# dnf -y install ffmpeg-devel
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavdevice
    libavfilter
    libavformat
    libavcodec
    libswresample
    libswscale
    libavutil
)

# ==============================================================================
# Headers and sources
# ==============================================================================

set(HEADERS
    src/core/api.h
    src/core/camera.h
    src/core/file.h
    src/core/frustum.h
    src/core/input.h
    src/core/log.h
    src/core/macro.h
    src/core/platform.h
    src/core/video_decode.h
    src/core/resource_pool.h
    src/examples/examples.h
    src/examples/example_base.h
    src/webgpu/api.h
    src/webgpu/buffer.h
    src/webgpu/context.h
    src/webgpu/imgui_overlay.h
    src/webgpu/shader.h
    src/webgpu/texture.h
)

set(SOURCES
    src/main.c
    src/core/camera.c
    src/core/file.c
    src/core/frustum.c
    src/core/log.c
    src/core/video_decode.c
    src/core/resource_pool.c
    src/examples/example_base.c
    src/examples/examples.c
    src/webgpu/buffer.c
    src/webgpu/context.c
    src/webgpu/imgui_overlay.c
    src/webgpu/shader.c
    src/webgpu/texture.c
)

# examples
set(SOURCES
    ${SOURCES}
    src/examples/animometer.c
    src/examples/clear_screen.c
    src/examples/compute_boids.c
    src/examples/compute_n_body.c
    src/examples/compute_particles.c
    src/examples/compute_ray_tracing.c
    src/examples/cube_reflection.c
    src/examples/gears.c
    src/examples/dynamic_uniform_buffer.c
    src/examples/gltf_loading.c
    src/examples/image_blur.c
    src/examples/imgui_overlay.c
    src/examples/instanced_cube.c
    src/examples/msaa_line.c
    src/examples/reversed_z.c
    src/examples/shadertoy.c
    src/examples/shadow_mapping.c
    src/examples/skybox.c
    src/examples/textured_cube.c
    src/examples/textured_quad.c
    src/examples/triangle.c
    src/examples/two_cubes.c
    src/examples/video_uploading.c
)

if(WIN32)
  set(SOURCES ${SOURCES} src/platforms/win32.c)
elseif(APPLE)
  set(SOURCES ${SOURCES} src/platforms/macos.m)
else()
    set(SOURCES ${SOURCES} src/platforms/linux.c)
endif()

# ==============================================================================
# Target definition
# ==============================================================================

set(TARGET wgpu_sample_launcher)

add_executable(${TARGET}
    ${HEADERS}
    ${SOURCES}
    ${KTX_SOURCES}
    ${RPLY_SOURCES}
    ${CIMGUI_SOURCES}
)
set_target_properties(${TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
)

if (WIN32)
    # nothing to do for now
elseif (APPLE)
    # nothing to do for now
elseif (UNIX)
    find_package(Vulkan REQUIRED FATAL_ERROR)
endif()

# ==============================================================================
# Target properties
# ==============================================================================

set_target_properties(${TARGET} PROPERTIES C_STANDARD 99)
set_target_properties(${TARGET} PROPERTIES C_STANDARD_REQUIRED ON)
set_target_properties(${TARGET} PROPERTIES C_EXTENSIONS OFF)
set_target_properties(${TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

# ==============================================================================
# Compile options
# ==============================================================================

if(MSVC)
    target_compile_options(${TARGET} PRIVATE /W4 /D_CRT_SECURE_NO_WARNINGS)
    target_compile_options(${TARGET} PRIVATE /fp:fast)
else()
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(${TARGET} PRIVATE -ffast-math)
endif()

if(UNIX AND NOT APPLE)
    target_compile_options(${TARGET} PRIVATE -D_POSIX_C_SOURCE=200809L)
endif()

# ==============================================================================
# Include directories
# ==============================================================================

if(WIN32)
    # nothing to do for now
elseif(APPLE)
    # nothing to do for now
else()
    target_include_directories(${TARGET}
        PRIVATE "../lib/wgpu_native"
        PRIVATE external/cglm/include
        PRIVATE external/cgltf
        PRIVATE external/cimgui
        PRIVATE external/ktx/include
        PRIVATE external/rply
        PRIVATE external/stb
        PRIVATE /home/sdauwe/Dev/libdawn/out/include
    )
endif()

# ==============================================================================
# Link libraries
# ==============================================================================

if(WIN32)
    # nothing to do for now
elseif(APPLE)
    target_link_libraries(${TARGET} PRIVATE "-framework Cocoa")
else()
    target_link_libraries(${TARGET} PRIVATE
      m
      X11
      ${Vulkan_LIBRARIES}
      Threads::Threads
      PkgConfig::LIBAV
      wgpu_native
    )
endif()

# ==============================================================================
# IDE support
# ==============================================================================

set_directory_properties(PROPERTIES VS_STARTUP_PROJECT ${TARGET})
source_group(TREE "${CMAKE_SOURCE_DIR}/src" FILES ${HEADERS} ${SOURCES})
source_group(external/cimgui FILES ${CIMGUI_SOURCES})
source_group(external/ktx FILES ${KTX_SOURCES})
source_group(external/rply FILES ${RPLY_SOURCES})

# ==============================================================================
# Deployment
# ==============================================================================

# Copy full directories
macro(copy_resource_dirs dirs)
    foreach(dir ${dirs})
        # Replace / at the end of the path (copy dir content VS copy dir)
        string(REGEX REPLACE "/+$" "" dirclean "${dir}")
        message(STATUS "Copying resource ${dirclean} to ${BUILD_DIR}")
        file(COPY ${dirclean} DESTINATION ${BUILD_DIR})
    endforeach()
endmacro()
copy_resource_dirs("${CMAKE_SOURCE_DIR}/assets")
