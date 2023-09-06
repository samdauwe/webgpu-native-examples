# Ubuntu 20.04 (Focal Fossa)
FROM ubuntu:20.04

# reference : https://www.softwarepronto.com/2022/09/dockerubuntu-installing-latest-cmake-on.html
RUN apt-get update \
  && apt-get -y install build-essential \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/* \
  && wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-3.24.1 \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
      && rm /tmp/cmake-install.sh \
      && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq \
        sudo \
        wget \
        git ca-certificates openssl \
        # Dev
        pkg-config build-essential python3 \
        # X11 / XCB
        libxi-dev libxcursor-dev libxinerama-dev libxrandr-dev mesa-common-dev \
        xcb libxcb-xkb-dev x11-xkb-utils libx11-xcb-dev libxkbcommon-x11-dev \
        # Libav
        libavdevice-dev \
        # Vulkan
        libvulkan1 libvulkan-dev mesa-vulkan-drivers vulkan-utils

# Remove sudoer password
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
