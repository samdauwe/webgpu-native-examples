#!/usr/bin/env bash
# build.sh — convenience wrapper around CMake presets.
#
# Build layout:
#   build/x86_64/debug/    <- native debug executables
#   build/x86_64/release/  <- native release executables
#   build/wasm/            <- WAjic cmake build directory
#   dist/assets/           <- symlink to assets/ (no duplication)
#   dist/native/x86_64/    <- symlink(s) to native build dirs
#   dist/wasm/             <- deployable .wasm / .html / wajic.js

set -e

DOCKER_DIR="$PWD/docker"
DOCKER_NAME="docker-webgpu-native-examples:latest"

webgpu_native_examples() {
    echo "---------- Building WebGPU Native Examples (x86_64 Release) ----------"
    cmake --preset x64-release
    cmake --build --preset x64-release
}

webgpu_wasm_examples() {
    echo "---------- Building WebAssembly (WAjic) Examples ----------"
    cmake -S wasm -B build/wasm
    cmake --build build/wasm -- -j"$(nproc)"
}

docker_build() {
    WORKING_DIR=`pwd`

    echo "---------- Building Docker image ----------"
    cd "$DOCKER_DIR"
    docker build --network host -t $DOCKER_NAME -f Dockerfile .

    cd "$WORKING_DIR"
}

docker_run() {
    echo "---------- Running Docker container ----------"
    xhost + && \
    docker run -it --rm --privileged \
        --security-opt label=disable \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --network=host \
        --ipc=host \
        -v "$PWD":/webgpu-native-examples:rw \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "$HOME/.Xauthority":/webgpu-native-examples/.Xauthority:rw \
        -v /dev/video0:/dev/video0 \
        -v /dev/video1:/dev/video1 \
        -v /dev/video2:/dev/video2 \
        -v /dev/snd:/dev/snd \
        -w /webgpu-native-examples \
        $DOCKER_NAME /bin/bash && \
    xhost -
}

while [[ $# -gt 0 ]]; do case "$1" in
  -webgpu_native_examples)
    shift
    webgpu_native_examples
    ;;
  -webgpu_wasm_examples)
    shift
    webgpu_wasm_examples
    ;;
  -docker_build)
    shift
    docker_build
    ;;
  -docker_run)
    shift
    docker_run
    ;;
  -h|-help|--help)
    cat << EOF
usage: $0 [options]
options:
  -webgpu_native_examples Build WebGPU native examples (Dawn)
  -webgpu_wasm_examples   Build WebAssembly examples (WAjic)
  -docker_build           Build Docker image for running the examples
  -docker_run             Run the Docker container with the examples
  -help                   Show help on stdout and exit
EOF
    exit 0 ;;
  *) _err "Unexpected argument $1" ;;
esac; done
