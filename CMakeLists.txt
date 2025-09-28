cmake_minimum_required(VERSION 3.22)
set(NAME webgpu-native-examples)
project(${NAME} LANGUAGES C CXX)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

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

# cJSON: Ultralightweight JSON parser in ANSI C.
set(CJSON_SOURCES
    external/cJSON/cJSON.c
)

# Google Dawn
set(DAWN_FETCH_DEPENDENCIES ON)
set(DAWN_BUILD_MONOLITHIC_LIBRARY SHARED)
add_subdirectory(external/dawn EXCLUDE_FROM_ALL)

# rply
set(RPLY_SOURCES
    external/rply/rply.c
)

# ==============================================================================
# Common headers and sources
# ==============================================================================

set(COMMON_HEADERS
    src/core/camera.h
    src/core/input.h
    src/webgpu/wgpu_common.h
    src/examples/common_shaders.h
    src/examples/meshes.h
)

set(COMMON_SOURCES
    src/core/camera.c
    src/webgpu/wgpu_common.c
    src/examples/meshes.c
)

# ==============================================================================
# Function to add example executables
# ==============================================================================
function(add_example_executable EXAMPLE_NAME)
    set(EXAMPLE_SOURCE "src/examples/${EXAMPLE_NAME}.c")
    add_executable(${EXAMPLE_NAME}
        ${COMMON_HEADERS}
        ${COMMON_SOURCES}
        ${EXAMPLE_SOURCE}
        ${CJSON_SOURCES}
        ${RPLY_SOURCES}
    )
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
        INTERPROCEDURAL_OPTIMIZATION ON
    )
    # ==============================================================================
    # Target compilation options
    # ==============================================================================
    if(MSVC)
        target_compile_options(${EXAMPLE_NAME} PRIVATE /W4 /D_CRT_SECURE_NO_WARNINGS)
        target_compile_options(${EXAMPLE_NAME} PRIVATE /fp:fast)
    else()
        target_compile_options(${EXAMPLE_NAME} PRIVATE -Wall -Wextra -pedantic)
        target_compile_options(${EXAMPLE_NAME} PRIVATE -ffast-math)
    endif()
    if(UNIX AND NOT APPLE)
        target_compile_options(${EXAMPLE_NAME} PRIVATE -D_POSIX_C_SOURCE=200809L)
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE _GLFW_X11)
    endif()
    # ==============================================================================
    # Include directories
    # ==============================================================================
    if(WIN32)
        # nothing to do for now
    elseif(APPLE)
        # nothing to do for now
    else()
        target_include_directories(${EXAMPLE_NAME}
            PRIVATE ${PROJECT_SOURCE_DIR}/src
            PRIVATE external/cglm/include
            PRIVATE external/cJSON
            PRIVATE external/rply
            PRIVATE external/sokol
            PRIVATE external/stb
        )
    endif()
    # ==============================================================================
    # Link libraries
    # ==============================================================================
    if(WIN32)
        # nothing to do for now
    else()
        set_target_properties(${EXAMPLE_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,./")
        target_link_libraries(${EXAMPLE_NAME} PRIVATE
          webgpu_dawn
          webgpu_glfw
          glfw
        )
    endif()
    # ==============================================================================
    # Target properties
    # ==============================================================================
    set_target_properties(${EXAMPLE_NAME} PROPERTIES C_STANDARD 99)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES C_STANDARD_REQUIRED ON)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES C_EXTENSIONS OFF)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    # ==============================================================================
    # IDE support
    # ==============================================================================
    # Place the example source under <example>/src/<example>.c in the project tree
    source_group("src" FILES ${EXAMPLE_SOURCE})
endfunction()

# Examples
add_example_executable(animometer)
add_example_executable(clear_screen)
add_example_executable(compute_particles_easing)
add_example_executable(conway_paletted_blurring)
add_example_executable(conway)
add_example_executable(cube_reflection)
add_example_executable(game_of_life)
add_example_executable(gerstner_waves)
add_example_executable(hello_triangle)
add_example_executable(instanced_cube)
add_example_executable(minimal)
add_example_executable(prng)
add_example_executable(square)
add_example_executable(terrain_mesh)
add_example_executable(textured_cube)
add_example_executable(tile_map)
add_example_executable(triangle)
add_example_executable(two_cubes)
add_example_executable(vertex_buffer)
add_example_executable(wireframe_vertex_pulling)

# ==============================================================================
# IDE support
# ==============================================================================

source_group(TREE "${CMAKE_SOURCE_DIR}/src" FILES ${HEADERS} ${SOURCES})

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
