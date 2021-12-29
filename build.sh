#!/usr/bin/env bash

set -e

EXTERNAL_DIR="$PWD/external"
DAWN_DIR="$EXTERNAL_DIR/dawn"
BUILD_DIR="$PWD/build"

DOCKER_DIR="$PWD/docker"
DOCKER_NAME="docker-webgpu-native-examples:latest"

update_dawn() {
    WORKING_DIR=`pwd`

    echo "---------- Updating Dawn code ----------"
    cd "$DAWN_DIR"
    /bin/bash download_dawn.sh

    cd "$WORKING_DIR"
}

webgpu_native_examples() {
    WORKING_DIR=`pwd`

    echo "---------- Building WebGPU Native Examples ----------"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make all -j8

    cd "$WORKING_DIR"
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
  -update_dawn)
    shift
    update_dawn
    ;;
  -webgpu_native_examples)
    shift
    webgpu_native_examples
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
  -update_dawn            Update to the latest version of "depot_tools" and "Dawn"
  -webgpu_native_examples Build WebGPU native examples
  -docker_build           Build Docker image for running the examples
  -docker_run             Run the Docker container with the examples
  -help                   Show help on stdout and exit
EOF
    exit 0 ;;
  *) _err "Unexpected argument $1" ;;
esac; done
