cmake_minimum_required(VERSION 3.10)
project(wgpu_native LANGUAGES CXX)

# ==============================================================================
# Headers and sources
# ==============================================================================

set(HEADERS
    wgpu_native.h
)

if(WIN32)
    # nothing to do for now
elseif(APPLE)
    # nothing to do for now
else()
    set(SOURCES ${SOURCES} wgpu_native_linux.cpp)
endif()

# ==============================================================================
# Target definition
# ==============================================================================

set(TARGET wgpu_native)

add_library(${TARGET} SHARED ${HEADERS} ${SOURCES})
set_target_properties(${TARGET} PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
    SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX}
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

set_target_properties(${TARGET} PROPERTIES CMAKE_CXX_STANDARD 17)
set_target_properties(${TARGET} PROPERTIES CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# Include directories
# ==============================================================================

if(WIN32)
    # nothing to do for now
elseif(APPLE)
    # nothing to do for now
else()
    target_include_directories(${TARGET}
        PRIVATE "${CMAKE_SOURCE_DIR}/external/dawn/include")
endif()

# ==============================================================================
# Link libraries
# ==============================================================================

if(WIN32)
    # nothing to do for now
elseif(APPLE)
    # nothing to do for now
else()
    target_link_libraries(${TARGET}
        ${Vulkan_LIBRARIES}
        "${CMAKE_SOURCE_DIR}/external/dawn/lib/libdawn_native.so"
        "${CMAKE_SOURCE_DIR}/external/dawn/lib/libdawn_proc.so"
    )
endif()
