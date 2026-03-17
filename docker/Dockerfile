# Base image using CUDA-enabled Ubuntu 22.04
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglfw3 \
    libglfw3-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libglew-dev \
    libeigen3-dev \
    libboost-all-dev \
    libssl-dev \
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxxf86vm-dev \
    mesa-utils \
    x11vnc \
    xvfb \
    fluxbox \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Setup Python environment
RUN python3 -m pip install --upgrade pip
RUN pip3 install numpy open3d matplotlib jupyter

# Install Pangolin dependencies
RUN apt-get update && apt-get install -y \
    # OpenGL core libraries
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libglew-dev \
    libglfw3-dev \
    mesa-utils \
    # X11 and display libraries
    libxrandr-dev \
    xorg-dev \
    libx11-dev \
    libxxf86vm-dev \
    libxcursor-dev \
    libxi-dev \
    # Pangolin specific dependencies
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols \
    libegl1-mesa-dev \
    libc++-dev \
    # Epoxy library
    libepoxy-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install Pangolin
WORKDIR /tmp
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_EXAMPLES=OFF \
          -DBUILD_PANGOLIN_PYTHON=OFF \
          -DCMAKE_CXX_STANDARD=20 \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp && rm -rf Pangolin

# Install Python Open3D (includes CUDA support)
RUN pip3 install open3d

# Configure VNC
RUN mkdir ~/.vnc
RUN x11vnc -storepasswd 123456 ~/.vnc/passwd

# Set environment variables
ENV DISPLAY=:1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Create working directory
WORKDIR /

# Create startup script
RUN echo '#!/bin/bash\n\
Xvfb :1 -screen 0 1920x1080x24 &\n\
fluxbox &\n\
x11vnc -display :1 -usepw -forever &\n\
echo "VNC server started at port 5900"\n\
echo "Password: 123456"\n\
exec "$@"' > /startup.sh && chmod +x /startup.sh

# Expose VNC port
EXPOSE 5900

# Set default command
CMD ["/startup.sh", "/bin/bash"]