// test_cuda.cu
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "common/include/pc.h"
#include "common/include/timer.h"
#include "common/include/neighbor.h"
#include "common/include/mat.h"

int main() {
    std::shared_ptr<pc::Pointcloud> pointcloud  = std::make_unique<pc::Pointcloud>(true);
    std::shared_ptr<logger::Logger> logger      = std::make_unique<logger::Logger>(logger::Level::VERB);
    std::shared_ptr<neighbor::Neighbor> neighbor = std::make_unique<neighbor::Neighbor>(true);
    std::shared_ptr<mat::MAT> mat = std::make_unique<mat::MAT>(true);
    std::shared_ptr<timer::Timer> timer         = std::make_unique<timer::Timer>();
    timer->init();
    
    float* d_points;
    int points_num = 0;
    CUDA_CHECK(cudaMalloc((void **)&d_points, sizeof(float) * 1000000*6));

    std::string file_path = "/workspace/data/ModelNet40_txt/airplane/train/airplane_0001.txt";
    timer->start_cpu();
    bool is_read = pointcloud->read_txt_xyz(file_path, d_points, points_num);
    timer->stop_cpu("Read TXT pointcloud");
    timer->show();
    if (is_read)
    {
        LOGV("Read TXT pointcloud file success, pointcloud size: %d", points_num);
        float *h_points=new float [points_num*6];
        CUDA_CHECK(cudaMemcpy(h_points, d_points, sizeof(float)*points_num*6, cudaMemcpyDeviceToHost));
        for(int i=0; i<100; i++)
        {
            LOGV("[%d] point: (%f, %f, %f)", i+1, h_points[i*6+0], h_points[i*6+1], h_points[i*6+2]);
        }

        int* h_neighbor = new int[points_num*(32+1)];
        float* h_normals = new float[points_num*3];
        neighbor->search_bf(h_points, points_num, 32, h_neighbor);
        mat->normals_estimator(h_points, points_num, h_neighbor, 32, h_normals);

        delete[] h_normals;
        delete[] h_neighbor;
        delete[] h_points;
    }


    return 0;
}