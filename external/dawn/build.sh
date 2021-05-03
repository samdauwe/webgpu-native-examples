# !/bin/bash

## ---------------------------------------------------------------------------------------------- #
 # Build script for Dawn, a WebGPU implementation.
 #
 # Uses the Chromium build system and dependency management to build Dawn from source.
 #
 # NOTE: Only works on GNU/Linux
 #
 # Ref:
 # https://dawn.googlesource.com/dawn
 # https://dawn.googlesource.com/dawn/+/HEAD/docs/building.md
 # ---------------------------------------------------------------------------------------------- #/

GIT_DEPOT_TOOLS_URL="https://chromium.googlesource.com/chromium/tools/depot_tools.git"
DAWN_URL="https://dawn.googlesource.com/dawn"

# Get location of the script itself
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
PROJECT_ROOT="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

# Set paths
DEPOT_TOOLS="$PROJECT_ROOT/depot_tools"
PATH="$PATH:$DEPOT_TOOLS"
DAWN_ROOT="$PROJECT_ROOT/dawn"

# Utility method for creating a directory
create_directory_if_not_found() {
    # if we cannot find the directory
    if [ ! -d "$1" ];
        then
        echo "$1 directory not found, creating..."
        mkdir -p "$1"
        echo "directory created at $1"
    fi
}

# Execute Ninja build
exec_ninja() {
  echo "Running ninja"
  ninja -C $1 # $DAWN_TARGET
}

# Update/Get/Ensure the Gclient Depot Tools
# Ref: http://commondatastorage.googleapis.com/chrome-infra-docs/flat/depot_tools/docs/html/depot_tools_tutorial.html#_setting_up
pull_depot_tools() {
    WORKING_DIR=`pwd`

    # Either clone or get latest depot tools
    if [ ! -d "$DEPOT_TOOLS" ]
    then
        echo 'Make directory for gclient called Depot Tools'
        mkdir -p "$DEPOT_TOOLS"

        echo "Pull the depo tools project from chromium source into the depot tools directory"
        git clone $GIT_DEPOT_TOOLS_URL $DEPOT_TOOLS
    else
        echo "Change directory into the depot tools"
        cd "$DEPOT_TOOLS"

        echo "Pull the depot tools down to the latest"
        git pull
    fi

    # Navigate back
    cd "$WORKING_DIR"
}

# Update/Get the dawn code base
pull_dawn() {
    WORKING_DIR=`pwd`

    # Either clone or get latest dawn code
    if [ ! -d "$DAWN_ROOT" ]
    then
        echo "Make dawn directory"
        mkdir -p "$DAWN_ROOT"

        echo "Pull the latest dawn code"
        git clone $DAWN_URL $DAWN_ROOT
    else
        echo "Change directory into dawn"
        cd "$DAWN_ROOT"

        echo "Pull the latest dawn code"
        git pull
    fi

    # Bootstrap the gclient configuration
    echo Bootstrap the gclient configuration
    cd "$DAWN_ROOT"
    cp scripts/standalone.gclient .gclient

    # Fetch external dependencies and toolchains with gclient
    echo "Fetch external dependencies and toolchains with gclient"
    echo "this can take a while..."
    if [ -z $1 ]
    then
        echo "gclient sync with newest"
        gclient sync
    else
        echo "gclient sync with $1"
        gclient sync -r $1
    fi

    # Navigate back
    cd "$WORKING_DIR"
}

execute_build() {
    WORKING_DIR=`pwd`
    cd "$DAWN_ROOT"

    # Set architecture
    if [ "$DAWN_ARCH" = "x86" ] ;
    then
        ARCH="x86"
    elif [ "$DAWN_ARCH" = "x86_64" ] ;
    then
        ARCH="x64"
    fi

    # Set build mode
    if [ "$DAWN_DEBUG" = "true" ] ;
    then
        BUILD_TYPE="Debug"
        DEBUG_ARG='is_debug=true symbol_level=1'
    else
        BUILD_TYPE="Release"
        DEBUG_ARG='is_debug=false symbol_level=0 dcheck_always_on=true'
    fi

    # Generate build files
    ARCH_OUT="out-linux-${DAWN_ARCH}"
    export DAWN_BUILD_DIR="$DAWN_ROOT/$ARCH_OUT/$BUILD_TYPE"
    echo "Generate projects using GN"
    gn gen "$ARCH_OUT/$BUILD_TYPE" --args="$DEBUG_ARG target_os=\"linux\" target_cpu=\"${ARCH}\" is_component_build=true is_clang=true"

    # Build dawn
    echo "Build Dawn in $BUILD_TYPE mode (arch: ${DAWN_ARCH})"
    exec_ninja "$ARCH_OUT/$BUILD_TYPE"

    # Verify the build actually worked
    if [ $? -eq 0 ]; then
        cd "$WORKING_DIR"
        echo "$BUILD_TYPE build for Dawn complete!"
    else
        echo "$BUILD_TYPE build for Dawn failed!"
    fi
}

install_dawn() {
    # Install headers
    echo "Install Dawn headers"
    create_directory_if_not_found "include/dawn"
    cp "$DAWN_BUILD_DIR/gen/src/include/dawn/webgpu.h" "include/dawn"
    cp "$DAWN_BUILD_DIR/gen/src/include/dawn/dawn_proc_table.h" "include/dawn"
    cp "$DAWN_BUILD_DIR/gen/src/include/dawn/webgpu_cpp.h" "include/dawn"
    cp "$DAWN_ROOT/src/include/dawn/EnumClassBitmasks.h" "include/dawn"
    cp "$DAWN_ROOT/src/include/dawn/dawn_proc.h" "include/dawn"
    cp "$DAWN_ROOT/src/include/dawn/dawn_wsi.h" "include/dawn"

    create_directory_if_not_found "include/dawn_native"
    cp "$DAWN_ROOT/src/include/dawn_native/DawnNative.h" "include/dawn_native"
    cp "$DAWN_ROOT/src/include/dawn_native/dawn_native_export.h" "include/dawn_native"
    cp "$DAWN_ROOT/src/include/dawn_native/VulkanBackend.h" "include/dawn_native"
    cp "$DAWN_ROOT/src/include/dawn_native/NullBackend.h" "include/dawn_native"

    # Install shared libraries
    echo "Install Dawn shared libraries"
    create_directory_if_not_found "lib"
    cp "$DAWN_BUILD_DIR/libc++.so" "lib"
    cp "$DAWN_BUILD_DIR/libdawn_native.so" "lib"
    cp "$DAWN_BUILD_DIR/libdawn_platform.so" "lib"
    cp "$DAWN_BUILD_DIR/libdawn_proc.so" "lib"
}

# Build google Dawn
# Ref: https://dawn.googlesource.com/dawn/+/HEAD/docs/building.md
build_dawn() {
    pull_depot_tools
    pull_dawn

    export DAWN_DEBUG=false
    export DAWN_ARCH=x86_64
    execute_build
    install_dawn
}
build_dawn
