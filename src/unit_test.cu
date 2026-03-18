// test_cuda.cu
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "common/include/pc.h"

__global__ void hello_cuda() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    std::cout << "CUDA Test Program" << std::endl;
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        
        hello_cuda<<<2, 4>>>();
        cudaDeviceSynchronize();
    }
    
    std::string file_path = "/workspace/data/exp1.pcd";
    std::vector<pc::PointXYZIL> points;
    pc::PCDInfo info;
    
    std::shared_ptr<pc::Pointcloud> pointcloud = std::make_unique<pc::Pointcloud>();
    pointcloud->read_pcd_bin(file_path, points, info);
    
    return 0;
}