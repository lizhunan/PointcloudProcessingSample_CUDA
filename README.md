# PointcloudProcessingSample_CUDA
A repository for CUDA-based pointcloud processing algorithms.

## 🎯 Overview

This repository provides a collection of CUDA-accelerated pointcloud processing algorithms with practical examples. All algorithms are implemented in C++ with CUDA and Python bindings, running in a fully configured Docker environment with Open3D and Pangolin for visualization.

### Repository Structure

```text
PointcloudProcessingSample_CUDA/
├── build/                                      # Build outputs
├── data/                                       # Sample point cloud data
├── doc/                                        # Documentation files
├── docker/                                     # Docker configuration files
│   ├── build.sh                                # Script to build Docker image
│   ├── Dockerfile                              # Docker image definition
│   ├── run.sh                                  # Script to run Docker container
├── script/                                     # Utility scripts
├── src/                                        # Source code
├── CMakeLists.txt                              # CMake build configuration
├── LICENSE                                     # License file
├── README.md                                   # Project documentation
```

### Key Features

**GPU Acceleration:** All algorithms leverage CUDA for massive parallelism

**Complete Environment:** Docker image with CUDA 12.1, Open3D, Pangolin, and VNC

**Real-time Visualization:** 3D visualization with Pangolin and Open3D

**Production Ready:** Optimized for large-scale pointcloud processing

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

### Build Script (build.sh)

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

### Run Script (run.sh)

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

### Dataset

The following dataset formats are currently supported: ModelNet40 (OFF format) and PCD (Point Cloud Data) files.

#### ModelNet40

The dataset used is [ModelNet40](http://modelnet.cs.princeton.edu/), a widely used benchmark dataset for 3D shape classification and segmentation.

**Download and Extraction**

```bash
# Download the ModelNet40 dataset
wget https://modelnet.cs.princeton.edu/ModelNet40.zip

# Extract to the data directory
unzip ModelNet40.zip
```

**Data Format Conversion**

The raw ModelNet40 data is in OFF format. Use the provided `off2txt.py` script to convert it to TXT pointcloud files:

```bash
# Basic usage with default parameters
python script/off2txt.py

# Specify custom source and target directories
python script/off2txt.py --source /workspace/data/ModelNet40 --target /workspace/data/ModelNet40_txt

# Adjust the number of sampled points per point cloud
python script/off2txt.py --num-points 8192
```

| Argument | Short | Default | Description |
| ----- | ---- | ---- | ---- |
| `--source` | `-s` | `/workspace/data/ModelNet40` | ModelNet40 source directory path |
| `--target` | `-t` | `/workspace/data/ModelNet40_txt` | Point cloud output directory path |
| `--num-points` | `-n` | `4096` | Number of points to sample per mesh |

**Visualization Comparison**

Use the `vis_pc_txt.py` script to visualize both OFF files and converted TXT files, allowing visual verification of the conversion quality:

- **Green mesh**: Original OFF file with vertex normals
- **Red point cloud**: Converted TXT point cloud file

```bash
# Visualize default airplane sample
python script/vis.py

# Visualize specific files
python script/vis.py --off /workspace/data/ModelNet40/chair/train/chair_0001.off \
                     --txt /workspace/data/ModelNet40_txt/chair/train/chair_0001.txt
```

- OFF files: Display the original mesh model
- TXT files: Display the sampled point cloud

| Argument | Short | Default | Description |
| ----- | ---- | ---- | ---- |
| `--off` | `-o` | `/workspace/data/ModelNet40/airplane/train/airplane_0001.off` | Path to the OFF mesh file for visualization |
| `--txt` | `-x` | `/workspace/data/ModelNet40_txt/airplane/train/airplane_0001.txt` | Path to the TXT pointcloud file for visualization |

#### PCD

NO IMPLEMENT

## 🛠️ Building Examples

## 💻 Usage Examples

## 🧪 Testing

## 📊 Performance Benchmarks

## 📈 Algorithm Details

## 📚 Documentation

## 🔧 Troubleshooting

**Common Issues**

1. error: XDG_RUNTIME_DIR not set in the environment.

```bash
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
```

2. libEGL warning: egl: failed to create dri2 screen & libEGL warning: DRI2: failed to authenticate

```bash
# Force the use of GLX instead of EGL
export PANGO_USE_EGL=0
export PANGO_USE_GLX=1

# If the above does not work, try forcing software rendering (alternative)
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
```

**Network Issues (China)**

1. If you're in China and experiencing Docker pull timeouts, you can use the image source in China, like [aityp](http://docker.aityp.com).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or suggestions:

- Open an issue on GitHub
- Contact: zhunan.li@outlook.com

---

Happy Coding! 🚀