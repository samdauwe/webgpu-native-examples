# WAjic WASM build for webgpu-native-examples.
#
# Usage (with CMake preset):
#   cmake -S wasm --preset wasm
#   cmake --build --preset wasm
#
# Usage (manual):
#   cmake -S wasm -B build/wasm
#   cmake --build build/wasm
#
# Prerequisites:
#   - clang / wasm-ld on PATH
#   - external/wajic/  (WAjic headers + runtime)
#   - external/emscripten/system/  (musl libc, libcxx, system.bc)

cmake_minimum_required(VERSION 3.20)

# WAjic toolchain must be set before project()
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../external/wajic/cmake/clang-wajic.cmake"
    CACHE FILEPATH "WAjic wasm32 toolchain")

project(webgpu-native-examples-wajic LANGUAGES C CXX)

# Set default build type if not specified by the user or IDE
# QtCreator and other IDEs rely on CMAKE_BUILD_TYPE to recognize build configurations
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Root of the webgpu-native-examples tree
set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Output directory for deployable WebAssembly files.
# Processed .wasm / .html / wajic.js land under dist/wasm/ so they are
# separate from native binaries while still sharing dist/assets/.
set(DIST_DIR "${ROOT}/dist/wasm")

# Optional tools
find_program(WASM_OPT wasm-opt)
find_program(NODE_BIN node)

# ==============================================================================
# cimgui (Dear ImGui C bindings) — static library for wasm32
# ==============================================================================
set(CIMGUI_DIR "${ROOT}/external/cimgui")
add_library(cimgui_wajic STATIC
    ${CIMGUI_DIR}/cimgui.cpp
    ${CIMGUI_DIR}/imgui/imgui.cpp
    ${CIMGUI_DIR}/imgui/imgui_demo.cpp
    ${CIMGUI_DIR}/imgui/imgui_draw.cpp
    ${CIMGUI_DIR}/imgui/imgui_widgets.cpp
    ${CIMGUI_DIR}/imgui/imgui_tables.cpp
)
set_target_properties(cimgui_wajic PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)
wajic_use_system_libs(cimgui_wajic)
target_include_directories(cimgui_wajic SYSTEM PUBLIC
    ${CIMGUI_DIR}
    ${CIMGUI_DIR}/imgui
)
target_compile_definitions(cimgui_wajic PUBLIC
    IMGUI_USER_CONFIG="../cimconfig.h"
    IMGUI_DISABLE_OBSOLETE_FUNCTIONS=1
    IMGUI_IMPL_API=extern\ \"C\"
)
target_compile_options(cimgui_wajic PRIVATE -w)

# ==============================================================================
# Helper: add a WAjic WebGPU example
# ==============================================================================
function(add_wajic_example name source)
    # Extra sources and include dirs after the required positional args
    cmake_parse_arguments(EX "USE_CIMGUI" "" "EXTRA_SOURCES;EXTRA_INCLUDES" ${ARGN})

    set(EXAMPLE_SOURCES
        ${ROOT}/${source}
        ${ROOT}/src/webgpu/wgpu_common.c
        ${ROOT}/src/core/camera.c
    )
    if(EX_EXTRA_SOURCES)
        foreach(src ${EX_EXTRA_SOURCES})
            list(APPEND EXAMPLE_SOURCES ${ROOT}/${src})
        endforeach()
    endif()

    # Suppress all warnings from third-party source files (under external/)
    if(EX_EXTRA_SOURCES)
        foreach(src ${EX_EXTRA_SOURCES})
            if("${src}" MATCHES "^external/")
                set_source_files_properties("${ROOT}/${src}"
                    PROPERTIES COMPILE_FLAGS "-w")
            endif()
        endforeach()
    endif()

    if(EX_USE_CIMGUI)
        list(APPEND EXAMPLE_SOURCES ${ROOT}/src/webgpu/imgui_overlay.c)
    endif()

    add_executable(${name} ${EXAMPLE_SOURCES})

    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
    )

    # System libs (libc, libcxx, malloc/free)
    wajic_use_system_libs(${name})
    wajic_export_main(${name})

    # Include directories
    target_include_directories(${name}
        PRIVATE ${ROOT}/src
    )
    target_include_directories(${name} SYSTEM
        PRIVATE ${ROOT}/external/cglm/include
                ${ROOT}/external/wajic
    )
    if(EX_EXTRA_INCLUDES)
        foreach(incdir ${EX_EXTRA_INCLUDES})
            target_include_directories(${name} SYSTEM PRIVATE ${ROOT}/${incdir})
        endforeach()
    endif()

    target_compile_options(${name} PRIVATE
        -Wno-unused-parameter
    )

    if(EX_USE_CIMGUI)
        target_link_libraries(${name} PRIVATE cimgui_wajic)
    endif()

    # --- Post-build: wasm-opt ---
    if(WASM_OPT)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND "${WASM_OPT}"
                --enable-sign-ext
                --enable-mutable-globals
                --enable-nontrapping-float-to-int
                --enable-bulk-memory-opt
                --legalize-js-interface --low-memory-unused --converge -Os
                "$<TARGET_FILE:${name}>" -o "$<TARGET_FILE:${name}>"
            COMMENT "wasm-opt: optimizing ${name}.wasm"
        )
    endif()

    # --- Post-build: wajicup ---
    if(NODE_BIN)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${DIST_DIR}"
            COMMAND "${NODE_BIN}" "${ROOT}/external/wajic/wajicup.js"
                "$<TARGET_FILE:${name}>"
                "${DIST_DIR}/${name}.wasm"
            COMMENT "WAjicUp: processing ${name}.wasm -> dist/wasm/"
        )
    endif()

    # --- Copy wajic.js loader to dist/wasm ---
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DIST_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ROOT}/external/wajic/wajic.js"
            "${DIST_DIR}/wajic.js"
        COMMENT "Copying wajic.js to dist/wasm/"
    )
endfunction()

# ==============================================================================
# Examples
# ==============================================================================
add_wajic_example(clear_screen src/examples/clear_screen.c
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(triangle src/examples/triangle.c)

add_wajic_example(textured_cube src/examples/textured_cube.c
    EXTRA_SOURCES
        src/examples/meshes.c
    EXTRA_INCLUDES
        external/sokol
        external/cJSON
        external/rply
)

add_wajic_example(animometer src/examples/animometer.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(gltf_viewer src/examples/gltf_viewer.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        src/webgpu/pbr.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(a_buffer src/examples/a_buffer.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/examples/meshes.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/cJSON
        external/rply
)

add_wajic_example(aquarium src/examples/aquarium.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/cJSON
        external/stb
)

add_wajic_example(blending src/examples/blending.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(blinn_phong_lighting src/examples/blinn_phong_lighting.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/cJSON
        external/stb
)

add_wajic_example(bloom src/examples/bloom.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(cameras src/examples/cameras.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/examples/meshes.c
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
        external/cJSON
        external/rply
)

add_wajic_example(cascaded_shadow_mapping src/examples/cascaded_shadow_mapping.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

# clustered_shading uses cgltf directly (not via gltf_model API).  In the
# native build, CGLTF_IMPLEMENTATION is provided by gltf_model.c in
# COMMON_SOURCES.  For WAjic we define it as a compile flag so the single
# clustered_shading.c compilation unit pulls in the cgltf implementation.
add_wajic_example(clustered_shading src/examples/clustered_shading.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
target_compile_definitions(clustered_shading PRIVATE CGLTF_IMPLEMENTATION)

add_wajic_example(compute_boids src/examples/compute_boids.c
    USE_CIMGUI
)

add_wajic_example(compute_cloth src/examples/compute_cloth.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
)

add_wajic_example(compute_metaballs src/examples/compute_metaballs.c
    USE_CIMGUI
)
# compute_metaballs has a deep init call chain that needs more than the default
# ~95 KB of WASM shadow stack. Reserve 512 KB for the stack and use 4 MB of
# initial memory so there is also enough heap room.
set_property(TARGET compute_metaballs APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker -z -Xlinker stack-size=524288 -Xlinker --initial-memory=4194304"
)

add_wajic_example(compute_n_body src/examples/compute_n_body.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)
# compute_n_body heap-allocates 768 KB of particle data during init plus ImGui
# and texture decode buffers. Reserve 8 MB of initial WASM memory to prevent
# out-of-bounds traps.
set_property(TARGET compute_n_body APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=8388608"
)

add_wajic_example(compute_particles_easing src/examples/compute_particles_easing.c
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)
# compute_particles_easing heap-allocates ~9.6 MB of particle data (60000 particles
# x 40 floats x 4 bytes) during init. Reserve 16 MB of initial WASM memory to
# ensure sufficient heap for the particle buffer plus runtime overhead.
set_property(TARGET compute_particles_easing APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(compute_particles_webgpu_logo src/examples/compute_particles_webgpu_logo.c
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)
# compute_particles_webgpu_logo uses a 512x512 RGBA texture plus probability-map
# buffers (2 × 1 MB) and a 50000-particle buffer (~2.4 MB). Reserve 16 MB of
# initial WASM memory to ensure enough heap for all allocations.
set_property(TARGET compute_particles_webgpu_logo APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(compute_particles src/examples/compute_particles.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)
# compute_particles heap-allocates 8 MB of particle data during init (freed
# after GPU upload) plus 512 KB texture file buffers and ImGui overhead.
# Reserve 16 MB of initial WASM memory.
set_property(TARGET compute_particles APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(compute_ray_tracing src/examples/compute_ray_tracing.c
    USE_CIMGUI
)

add_wajic_example(compute_shader src/examples/compute_shader.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)

add_wajic_example(conway src/examples/conway.c)

add_wajic_example(conway_paletted_blurring src/examples/conway_paletted_blurring.c)

add_wajic_example(coordinate_system src/examples/coordinate_system.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)

add_wajic_example(cornell_box src/examples/cornell_box.c
    USE_CIMGUI
)

add_wajic_example(cube_reflection src/examples/cube_reflection.c)

add_wajic_example(cubemap src/examples/cubemap.c
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)
# cubemap pre-allocates 6 × 4 MB (FACE_NUM_BYTES) pixel buffers simultaneously
# during init (24 MB peak).  Reserve 32 MB of initial WASM memory so the heap
# does not exhaust during the concurrent sfetch face fetches.
set_property(TARGET cubemap APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=33554432"
)

add_wajic_example(deferred_multi_sampling src/examples/deferred_multi_sampling.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# deferred_multi_sampling pre-allocates 4 × 5 MB texture fetch buffers (20 MB
# peak) plus model vertex/index data and WASM overhead.  Reserve 32 MB of
# initial WASM memory.
set_property(TARGET deferred_multi_sampling APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=33554432"
)

add_wajic_example(deferred_rendering src/examples/deferred_rendering.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/examples/meshes.c
        external/rply/rply.c
    EXTRA_INCLUDES
        external/sokol
        external/cJSON
        external/rply
)
# deferred_rendering fetches a 358 KB ASCII PLY file and holds its raw bytes
# plus the parsed mesh struct in memory simultaneously.  Reserve 16 MB of
# initial WASM memory.
set_property(TARGET deferred_rendering APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(deferred_shading src/examples/deferred_shading.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# deferred_shading pre-allocates 4 × 5 MB texture fetch buffers (20 MB peak)
# plus two gltf model data + WASM overhead.  Reserve 32 MB of initial WASM
# memory.
set_property(TARGET deferred_shading APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=33554432"
)

add_wajic_example(deferred_shadows src/examples/deferred_shadows.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# deferred_shadows pre-allocates 4 × 5 MB texture fetch buffers (20 MB peak)
# plus two gltf models + shadow map data + WASM overhead.  Reserve 32 MB.
set_property(TARGET deferred_shadows APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=33554432"
)

add_wajic_example(dynamic_uniform_buffer src/examples/dynamic_uniform_buffer.c)

# equirectangular_image decodes a 5 MB JPEG into RGBA pixels.  Reserve 16 MB
# of initial WASM memory so the malloc'd file buffer + decoded pixels fit.
add_wajic_example(equirectangular_image src/examples/equirectangular_image.c
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)
set_property(TARGET equirectangular_image APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(fluid_simulation src/examples/fluid_simulation.c
    USE_CIMGUI
)

add_wajic_example(game_of_life src/examples/game_of_life.c
    USE_CIMGUI
)

add_wajic_example(gears src/examples/gears.c)

add_wajic_example(gerstner_waves src/examples/gerstner_waves.c
    EXTRA_SOURCES
        src/examples/meshes.c
        src/core/image_loader.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/cJSON
        external/rply
        external/stb
)

add_wajic_example(hello_triangle src/examples/hello_triangle.c)

add_wajic_example(image_blur src/examples/image_blur.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)

add_wajic_example(imgui_overlay src/examples/imgui_overlay.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(input_attachments src/examples/input_attachments.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(instanced_cube src/examples/instanced_cube.c
    EXTRA_SOURCES
        src/examples/meshes.c
        external/rply/rply.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/rply
        external/cJSON
)

add_wajic_example(instancing src/examples/instancing.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(metaballs src/examples/metaballs.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/sokol
)

add_wajic_example(minimal src/examples/minimal.c)

add_wajic_example(msaa_line src/examples/msaa_line.c)

add_wajic_example(multi_sampling src/examples/multi_sampling.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# multi_sampling loads voyager.gltf (~3.2 MB) with embedded data URIs.
# Reserve 16 MB of initial WASM memory for the fetch buffer + decoded model data.
set_property(TARGET multi_sampling APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(n_body_simulation src/examples/n_body_simulation.c)

add_wajic_example(normal_map src/examples/normal_map.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/examples/meshes.c
        src/core/image_loader.c
        external/rply/rply.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
        external/rply
        external/cJSON
)

add_wajic_example(obj_pbr_ibl src/examples/obj_pbr_ibl.c
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)

add_wajic_example(occlusion_query src/examples/occlusion_query.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(offscreen src/examples/offscreen.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(oit src/examples/oit.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(out_of_bounds_viewport src/examples/out_of_bounds_viewport.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)

add_wajic_example(parallax_mapping src/examples/parallax_mapping.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(particle_system src/examples/particle_system.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# particle_system allocates 4 × 4 MB texture fetch buffers simultaneously (16 MB
# peak) plus decoded pixel data and WASM overhead.  Reserve 32 MB of initial
# WASM memory.
set_property(TARGET particle_system APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=33554432"
)

add_wajic_example(pipeline_constants src/examples/pipeline_constants.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(pipelines src/examples/pipelines.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(point_light_shadows src/examples/point_light_shadows.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(points src/examples/points.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(post_processing src/examples/post_processing.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)

add_wajic_example(primitive_picking src/examples/primitive_picking.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/examples/meshes.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/cJSON
        external/rply
)

add_wajic_example(pristine_grid src/examples/pristine_grid.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(procedural_mesh src/examples/procedural_mesh.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
        external/par
)

add_wajic_example(prng src/examples/prng.c)

add_wajic_example(projected_shadow_mapping src/examples/projected_shadow_mapping.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(quake2_renderer src/examples/quake2_renderer.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
        external/stb
)
# The Quake 2 PAK file (pak0.pak) is ~44 MB loaded entirely into WASM heap.
# Additional heap is needed for BSP geometry, MD2 models, textures, and
# ImGui overhead.  Reserve 128 MB of initial WASM memory.
set_property(TARGET quake2_renderer APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=134217728"
)

add_wajic_example(radial_blur src/examples/radial_blur.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(raytracer src/examples/raytracer.c
    EXTRA_INCLUDES
        external/sokol
)
# raytracer heap-allocates ~32 MB of face arrays (16 models × 23 000 faces)
# during OBJ parsing, plus a 2 MB fetch buffer, the 1280×720 float4 pixel
# storage buffer (~14.7 MB) and the RNG state buffer (~3.7 MB).
# Reserve 64 MB of initial WASM memory.
set_property(TARGET raytracer APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=67108864"
)

add_wajic_example(render_bundles src/examples/render_bundles.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/examples/meshes.c
        external/cJSON/cJSON.c
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/rply
        external/cJSON
        external/stb
        external/sokol
)

add_wajic_example(reversed_z src/examples/reversed_z.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(sampler_parameters src/examples/sampler_parameters.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(shadertoy src/examples/shadertoy.c
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(shadow_mapping src/examples/shadow_mapping.c
    EXTRA_SOURCES
        src/examples/meshes.c
        external/rply/rply.c
    EXTRA_INCLUDES
        external/rply
        external/cJSON
        external/sokol
)

add_wajic_example(shadow src/examples/shadow.c)

add_wajic_example(skinned_mesh src/examples/skinned_mesh.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/cgltf
        external/rply
        external/stb
        external/sokol
)

add_wajic_example(spherical_env_mapping src/examples/spherical_env_mapping.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# spherical_env_mapping has a 1.5 MB static sfetch buffer in BSS plus stbi-
# decoded pixels and gltf model data on the heap.  Reserve 16 MB.
set_property(TARGET spherical_env_mapping APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(square src/examples/square.c)

add_wajic_example(ssao src/examples/ssao.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# ssao loads Sponza.glb (~13 MB) with all textures embedded, plus SSAO kernel
# and noise buffers, plus a full deferred render pipeline.  Reserve 64 MB of
# initial WASM memory so the heap is large enough for the peak allocation.
set_property(TARGET ssao APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=67108864"
)

add_wajic_example(stencil_buffer src/examples/stencil_buffer.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# stencil_buffer loads venus.gltf (~1.5 MB, base64-embedded geometry).
# Reserve 16 MB of initial WASM memory.
set_property(TARGET stencil_buffer APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(subpasses src/examples/subpasses.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# subpasses loads samplebuilding.gltf (~706 KB base64-embedded) and
# samplebuilding_glass.gltf (~34 KB), plus a colored_glass_rgba.png texture
# (~178 KB), plus G-Buffer textures, 64 lights, and ImGui overhead.
# Reserve 32 MB of initial WASM memory.
set_property(TARGET subpasses APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=33554432"
)

add_wajic_example(terrain_mesh src/examples/terrain_mesh.c
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)
# terrain_mesh fetches two PNG textures (color 512x512 and heightmap 256x256)
# via malloc'd buffers plus procedural mesh data.  Reserve 8 MB of initial
# WASM memory.
set_property(TARGET terrain_mesh APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=8388608"
)

add_wajic_example(terrain_tessellation src/examples/terrain_tessellation.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# terrain_tessellation loads a 1024x1024 R16Unorm heightmap (~2 MB decoded),
# a 256x256x6 terrain texture array (~2.5 MB per layer, ~15 MB total decoded),
# a skysphere RGBA texture (~5 MB), a sphere.gltf model, and generates a
# 256x256 terrain mesh (two copies, ~4 MB each).  Reserve 64 MB.
set_property(TARGET terrain_tessellation APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=67108864"
)

add_wajic_example(text_overlay src/examples/text_overlay.c
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        src/webgpu/text_overlay.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)

add_wajic_example(text_rendering_msdf src/examples/text_rendering_msdf.c
    EXTRA_SOURCES
        src/examples/meshes.c
        src/core/image_loader.c
        external/rply/rply.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
        external/cJSON
        external/rply
)

add_wajic_example(texture_3d src/examples/texture_3d.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)

add_wajic_example(texture_array src/examples/texture_array.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)

add_wajic_example(texture_cubemap src/examples/texture_cubemap.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# texture_cubemap loads a 4 MB cross PNG + 5 gltf models; reserve 32 MB.
set_property(TARGET texture_cubemap APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=33554432"
)

add_wajic_example(texture_cubemap_array src/examples/texture_cubemap_array.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/gltf_model.c
        src/core/image_loader.c
        external/mikktspace/mikktspace.c
    EXTRA_INCLUDES
        external/cgltf
        external/stb
        external/mikktspace
        external/sokol
)
# texture_cubemap_array loads 3 cross PNGs (2 MB each) + 5 gltf models; reserve 32 MB.
set_property(TARGET texture_cubemap_array APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=33554432"
)

add_wajic_example(textured_quad src/examples/textured_quad.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/camera.c
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)

add_wajic_example(tile_map src/examples/tile_map.c
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)

add_wajic_example(timestamp_query src/examples/timestamp_query.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/examples/meshes.c
        external/rply/rply.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/rply
        external/cJSON
)

add_wajic_example(two_cubes src/examples/two_cubes.c
    EXTRA_SOURCES
        src/examples/meshes.c
        external/rply/rply.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/rply
        external/cJSON
)

add_wajic_example(vertex_buffer src/examples/vertex_buffer.c)

add_wajic_example(volume_rendering_texture_3d src/examples/volume_rendering_texture_3d.c
    USE_CIMGUI
    EXTRA_INCLUDES
        external/sokol
)
# volume_rendering loads a ~7 MB binary 3D texture; reserve 16 MB.
set_property(TARGET volume_rendering_texture_3d APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(voxel_space src/examples/voxel_space.c
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)
# voxel_space uses two 1024x1024 maps loaded asynchronously; reserve 16 MB.
set_property(TARGET voxel_space APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(water_simulation src/examples/water_simulation.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/core/image_loader.c
    EXTRA_INCLUDES
        external/stb
        external/sokol
)
# water_simulation loads tiles + 6 skybox faces; reserve 16 MB.
set_property(TARGET water_simulation APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=16777216"
)

add_wajic_example(wireframe src/examples/wireframe.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/examples/meshes.c
        external/rply/rply.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/rply
        external/cJSON
)

add_wajic_example(wireframe_vertex_pulling src/examples/wireframe_vertex_pulling.c
    USE_CIMGUI
    EXTRA_SOURCES
        src/examples/meshes.c
        external/rply/rply.c
        external/cJSON/cJSON.c
    EXTRA_INCLUDES
        external/sokol
        external/rply
        external/cJSON
)

# compute_shader malloc-allocates a 16 MB file buffer during init for the
# async texture fetch. Reserve 8 MB of initial WASM memory; the heap grows
# automatically via sbrk when the 16 MB fetch buffer is allocated.
set_property(TARGET compute_shader APPEND_STRING PROPERTY
    LINK_FLAGS " -Xlinker --initial-memory=8388608"
)

# ==============================================================================
# Generate per-example HTML pages + index page for serving from dist/
# ==============================================================================

# Maps each example slug to its human-readable display title.
# Format: "slug|Display Title"  (sorted alphabetically by slug)
set(WASM_EXAMPLE_TITLES
    "a_buffer|A-Buffer (OIT)"
    "animometer|Animometer"
    "aquarium|Aquarium"
    "blending|Blending"
    "blinn_phong_lighting|Blinn-Phong Lighting"
    "bloom|Bloom (Fullscreen Blur)"
    "cameras|Cameras"
    "cascaded_shadow_mapping|Cascaded Shadow Mapping"
    "clear_screen|Clear Screen"
    "clustered_shading|Clustered Shading"
    "compute_boids|Compute Boids"
    "compute_cloth|Compute Cloth"
    "compute_metaballs|Compute Metaballs"
    "compute_n_body|Compute N-Body"
    "compute_particles|Compute Shader Particle System"
    "compute_particles_easing|Compute Particles Easing"
    "compute_particles_webgpu_logo|Compute Particles WebGPU Logo"
    "compute_ray_tracing|Compute Ray Tracing"
    "compute_shader|Compute Shader Image Processing"
    "conway|Conway Game of Life"
    "conway_paletted_blurring|Conway Game of Life (Paletted Blurring)"
    "coordinate_system|Coordinate System"
    "cornell_box|Cornell Box (Lightmap Raytracer)"
    "cube_reflection|Cube Reflection"
    "cubemap|Cubemap"
    "deferred_multi_sampling|Deferred Multi-Sampling"
    "deferred_rendering|Deferred Rendering"
    "deferred_shading|Deferred Shading"
    "deferred_shadows|Deferred Shadows"
    "dynamic_uniform_buffer|Dynamic Uniform Buffer"
    "equirectangular_image|Equirectangular Image"
    "fluid_simulation|Fluid Simulation"
    "game_of_life|Game of Life"
    "gears|Gears"
    "gerstner_waves|Gerstner Waves"
    "gltf_viewer|glTF PBR Viewer"
    "hello_triangle|Hello Triangle"
    "image_blur|Image Blur"
    "imgui_overlay|ImGui Overlay"
    "input_attachments|Input Attachments"
    "instanced_cube|Instanced Cube"
    "instancing|Instanced Mesh Rendering"
    "metaballs|Metaballs"
    "minimal|Minimal"
    "msaa_line|MSAA Line"
    "multi_sampling|Multisampling (MSAA)"
    "n_body_simulation|N-Body Simulation"
    "normal_map|Normal Mapping"
    "obj_pbr_ibl|OBJ PBR with IBL"
    "occlusion_query|Occlusion Query"
    "offscreen|Offscreen Rendering"
    "oit|Order Independent Transparency"
    "out_of_bounds_viewport|Out-of-bounds Viewport"
    "parallax_mapping|Parallax Mapping"
    "particle_system|CPU Particle System"
    "pipeline_constants|Pipeline Constants"
    "pipelines|Pipeline State Objects"
    "point_light_shadows|Point Light Shadows (Cubemap)"
    "points|Points"
    "post_processing|Post-processing"
    "primitive_picking|Primitive Picking"
    "pristine_grid|Pristine Grid"
    "procedural_mesh|Procedural Mesh"
    "projected_shadow_mapping|Projected Shadow Mapping"
    "prng|Pseudorandom Number Generation"
    "quake2_renderer|Quake 2 BSP Renderer"
    "radial_blur|Radial Blur"
    "raytracer|GPU Raytracer"
    "render_bundles|Render Bundles"
    "reversed_z|Reversed Z"
    "sampler_parameters|Sampler Parameters"
    "shadow|Shadow (Depth Array)"
    "shadow_mapping|Shadow Mapping"
    "shadertoy|Shadertoy (Seascape)"
    "skinned_mesh|Skinned Mesh"
    "spherical_env_mapping|Spherical Environment Mapping"
    "square|Square"
    "ssao|Screen Space Ambient Occlusion (SSAO)"
    "stencil_buffer|Stencil Buffer Outlines"
    "subpasses|Subpasses (G-Buffer + Transparency)"
    "terrain_mesh|Terrain Mesh"
    "terrain_tessellation|Terrain Tessellation"
    "text_overlay|Text Overlay"
    "text_rendering_msdf|Text Rendering MSDF"
    "texture_3d|3D Texture"
    "texture_array|Texture Array"
    "texture_cubemap|Texture Cubemap"
    "texture_cubemap_array|Texture Cubemap Array"
    "textured_cube|Textured Cube"
    "textured_quad|Textured Quad"
    "tile_map|Tile Map"
    "timestamp_query|Timestamp Query"
    "triangle|Triangle"
    "two_cubes|Two Cubes"
    "vertex_buffer|Vertex Buffer"
    "volume_rendering_texture_3d|Volume Rendering (3D Texture)"
    "voxel_space|Voxel Space"
    "water_simulation|Water Simulation"
    "wireframe|Wireframe"
    "wireframe_vertex_pulling|Wireframe Vertex Pulling"
)

# Helper: generate per-example HTML from a template file.
# Uses configure_file() so the template lives outside CMakeLists.txt.
function(generate_example_html entry)
    string(REPLACE "|" ";" pair "${entry}")
    list(GET pair 0 EXAMPLE_NAME)
    list(GET pair 1 EXAMPLE_TITLE)
    set(HTML_OUT "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}.html")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/templates/example.html.in"
        "${HTML_OUT}"
        @ONLY
    )
    if(TARGET ${EXAMPLE_NAME})
        add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${HTML_OUT}" "${DIST_DIR}/${EXAMPLE_NAME}.html"
            COMMENT "Copying ${EXAMPLE_NAME}.html to dist/wasm/"
        )
    endif()
endfunction()

foreach(entry ${WASM_EXAMPLE_TITLES})
    generate_example_html("${entry}")
endforeach()

# Build the <a> tile list for the index page, then configure the template.
set(EXAMPLE_TILES "")
foreach(entry ${WASM_EXAMPLE_TITLES})
    string(REPLACE "|" ";" pair "${entry}")
    list(GET pair 0 _slug)
    list(GET pair 1 _title)
    string(APPEND EXAMPLE_TILES
        "  <a class=\"tile\" href=\"${_slug}.html\">${_title}</a>\n")
endforeach()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/templates/index.html.in"
    "${CMAKE_CURRENT_BINARY_DIR}/index.html"
    @ONLY
)

# Always-run target: copy index.html to dist/wasm/ on every build.
add_custom_target(copy_index_html ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DIST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/index.html"
        "${DIST_DIR}/index.html"
    COMMENT "Copying index.html to dist/wasm/"
)

# Symlink assets/ into dist/wasm/ so examples can fetch textures, models, etc.
# The symlink target is the project-level assets/ directory, two levels up from
# dist/wasm/. Examples that use sokol_fetch with relative paths like
# "assets/textures/..." work when the dev server roots at dist/wasm/.
add_custom_command(TARGET textured_cube POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DIST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${ROOT}/assets" "${DIST_DIR}/assets"
    COMMENT "Symlinking assets/ into dist/wasm/"
)
