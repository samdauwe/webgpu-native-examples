# Ubuntu 20.04 (Focal Fossa)
FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq \
        sudo \
        wget \
        git ca-certificates openssl \
        # Dev
        pkg-config build-essential cmake python3 \
        # X11 / XCB
        libxi-dev libxcursor-dev libxinerama-dev libxrandr-dev mesa-common-dev \
        xcb libxcb-xkb-dev x11-xkb-utils libx11-xcb-dev libxkbcommon-x11-dev \
        # Libav
        libavdevice-dev \
        # Vulkan
        libvulkan1 libvulkan-dev mesa-vulkan-drivers vulkan-utils

# Remove sudoer password
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
