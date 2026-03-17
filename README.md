# PointcloudProcessingSample_CUDA
A repository for CUDA-based pointcloud processing algorithms.

## 🎯 Overview

This repository provides a collection of CUDA-accelerated pointcloud processing algorithms with practical examples. All algorithms are implemented in C++ with CUDA and Python bindings, running in a fully configured Docker environment with Open3D and Pangolin for visualization.

### Key Features

**🚀 GPU Acceleration:** All algorithms leverage CUDA for massive parallelism

**🔧 Complete Environment:** Docker image with CUDA 12.1, Open3D, Pangolin, and VNC

**📊 Real-time Visualization:** 3D visualization with Pangolin and Open3D

**🔬 Production Ready:** Optimized for large-scale pointcloud processing

## 📦 Included Algorithms

TODO

## 🚀 Quick Start

### Prerequisites

- Docker (version 20.10+)
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with drivers installed

### Installation Steps

1. Clone or create the project files
```bash
git clone https://github.com/lizhunan/PointcloudProcessingSample_CUDA.git
cd PointcloudProcessingSample_CUDA
```
2. Make scripts executable
```bash
chmod +x build.sh run.sh
```
3. Build the Docker image
```bash
./build.sh
```
4. Run the container
```bash
./run.sh
```
Follow the interactive menu to choose your preferred mode.

### 🛠️ Build Script (build.sh)

#### Usage

```bash
./build.sh [IMAGE_NAME] [IMAGE_TAG] [DOCKERFILE_PATH]
```

#### Parameters

| Parameter | Default | Description |
| ----- | ---- | ---- |
| `IMAGE_NAME` | `pointcloud-processing-sample-cuda` | Name of the Docker image |
| `IMAGE_TAG` | `latest` | Tag for the image |
| `DOCKERFILE_PATH` | `.` | Path containing Dockerfile |

#### Example

```bash
# Build with default settings
./build.sh

# Build with custom name and tag
./build.sh my-cuda-image v1.0

# Build from specific path
./build.sh my-cuda-image latest ./docker/
```

### 🏃 Run Script (run.sh)

#### Usage

```bash
./run.sh [IMAGE_NAME] [IMAGE_TAG] [CONTAINER_NAME] [WORKSPACE_DIR]
```

#### Parameters

| Parameter | Default | Description |
| ----- | ---- | ---- |
| `IMAGE_NAME` | `pointcloud-processing-sample-cuda` | Name of the Docker image |
| `IMAGE_TAG` | `latest` | Tag for the image |
| `CONTAINER_NAME` | `cuda-pointcloud-dev` | Container name |
| `WORKSPACE_DIR` | `$(pwd)` | Workspace directory path |

#### Environment Variables

| Variable | Default | Description |
| ----- | ---- | ---- |
| `VNC_PORT` | `5900` | VNC server port |
| `JUPYTER_PORT` | `8888` | Jupyter notebook port |
| `SSH_PORT` | `2222` | SSH server port |

#### Run Modes

The script provides 5 interactive modes:

1. Interactive Mode (suitable for development)
    - Starts container with interactive bash shell
    - Automatically removed when exit
2. Daemon Mode (suitable for long-running processes)
    - Runs container in background
    - Auto-restarts unless stopped
3. One-time Command Mode
    - Executes a single command and exits
    - Example: nvidia-smi, nvcc --version
4. View Container Status
    - Shows container status and resource usage
5. Exit

#### Example

```bash
# Run with default settings
./run.sh

# Run with custom parameters
./run.sh my-cuda-image v1.0 my-container ./my-workspace

# Use custom ports
export VNC_PORT=5901
export JUPYTER_PORT=8889
./run.sh
```
## 🔧 Building Examples

## 💻 Usage Examples

## 🧪 Testing

## 📊 Performance Benchmarks

## 📈 Algorithm Details

## 📚 Documentation

This project is licensed under the MIT License - see the LICENSE file for details.

📧 Contact

For questions or suggestions:

- Open an issue on GitHub
- Contact: zhunan.li@outlook.com

---

Happy Coding! 🚀