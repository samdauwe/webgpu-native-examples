# Fedora (latest minimal)
FROM fedora:latest

# ---------------------------------------------------------------------------
# System packages — native (Dawn/WebGPU) build + WebAssembly (WAjic) build
# ---------------------------------------------------------------------------
RUN dnf -y update && \
    dnf -y install \
        # Core build tools
        sudo wget curl git ca-certificates openssl unzip \
        @development-tools \
        cmake ninja-build pkg-config python3 python3-pip \
        # X11 / XCB (required by Dawn/GLFW)
        libXi-devel libXcursor-devel libXinerama-devel libXrandr-devel \
        libxcb-devel libX11-devel libxkbcommon-devel libxkbcommon-x11-devel \
        mesa-libGL-devel mesa-libEGL-devel \
        # Vulkan (required by Dawn)
        vulkan-headers vulkan-loader-devel mesa-vulkan-drivers vulkan-tools \
        # LLVM / Clang — native compiler and wasm32 cross-compiler
        clang lld \
        # Binaryen — wasm-opt for 10-15% WASM size reduction
        binaryen \
        # Node.js — required by wajicup.js post-processor
        nodejs \
        # Python cryptography for the built-in HTTPS dev server
        python3-cryptography \
    && dnf clean all

# ---------------------------------------------------------------------------
# Remove sudoer password requirement
# ---------------------------------------------------------------------------
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
