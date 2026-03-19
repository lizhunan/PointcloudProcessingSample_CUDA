// test_cuda.cu
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "common/include/pc.h"
#include "common/include/timer.h"

int main() {
    std::shared_ptr<pc::Pointcloud> pointcloud  = std::make_unique<pc::Pointcloud>(true);
    std::shared_ptr<logger::Logger> logger      = std::make_unique<logger::Logger>(logger::Level::VERB);
    std::shared_ptr<timer::Timer> timer         = std::make_unique<timer::Timer>();
    timer->init();
    
    
    std::string file_path = "/workspace/data/ModelNet40_txt/airplane/train/airplane_0001.txt";
    pointcloud->read_txt_xyz(file_path);

    // timer->start_cpu();
    // pointcloud->read_pcd_bin(file_path, points, info);
    // timer->stop_cpu("read pcd");
    // timer->show();
    return 0;
}