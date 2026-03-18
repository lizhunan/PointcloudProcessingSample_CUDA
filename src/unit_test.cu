// test_cuda.cu
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "common/include/pc.h"
#include "common/include/timer.h"

__global__ void hello_cuda() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {

    std::shared_ptr<logger::Logger> logger = std::make_unique<logger::Logger>(logger::Level::VERB);
    std::cout << "CUDA Test Program" << std::endl;
    
    std::shared_ptr<timer::Timer> timer = std::make_unique<timer::Timer>();
    timer->init();
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        
        timer->start_gpu();
        hello_cuda<<<2, 4>>>();
        timer->stop_gpu("hello cuda");
        cudaDeviceSynchronize();
    }
    
    std::string file_path = "/workspace/data/exp1.pcd";
    std::vector<pc::PointXYZIL> points;
    pc::PCDInfo info;
    
    std::shared_ptr<pc::Pointcloud> pointcloud = std::make_unique<pc::Pointcloud>();
    timer->start_cpu();
    pointcloud->read_pcd_bin(file_path, points, info);
    timer->stop_cpu("read pcd");
    timer->show();
    return 0;
}