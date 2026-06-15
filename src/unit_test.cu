// test_cuda.cu
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "common/include/pc.h"
#include "common/include/timer.h"
#include "common/include/neighbor.h"
#include "common/include/mat.h"
#include "downsampling/include/nss_sampling.h"
#include "feature_detection/include/iss.h"
#include "feature_description/include/shot.h"
#include "feature_description/include/fpfh.h"

int main() {
    std::shared_ptr<pc::Pointcloud> pointcloud  = std::make_unique<pc::Pointcloud>(false);
    std::shared_ptr<logger::Logger> logger      = std::make_unique<logger::Logger>(logger::Level::DEBUG);
    std::shared_ptr<neighbor::Neighbor> neighbor = std::make_unique<neighbor::Neighbor>(false);
    std::shared_ptr<mat::MAT> mat = std::make_unique<mat::MAT>(false);
    std::shared_ptr<downsampling::NSS> nss = std::make_unique<downsampling::NSS>(true);
    // std::shared_ptr<shot::SHOT> shot = std::make_unique<shot::SHOT>(true);
    std::shared_ptr<iss::ISS> iss = std::make_unique<iss::ISS>(true);
    std::shared_ptr<feat_desc::FPFH> fpfh = std::make_unique<feat_desc::FPFH>(true);
    std::shared_ptr<timer::Timer> timer         = std::make_unique<timer::Timer>();
    timer->init();
    
    float* d_points;
    int points_num = 0;
    CUDA_CHECK(cudaMalloc((void **)&d_points, sizeof(float) * 1000000*6));

    // std::string file_path = "/workspace/data/airplane_0001_test.txt";
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
        float* feat_det_output = new float[points_num*6];
        float* h_feat = new float[points_num*33];
        // float* h_lrfs = new float[points_num*9];
        neighbor->search_bf(h_points, points_num, 32, h_neighbor);
        mat->normals_estimator(h_points, points_num, h_neighbor, 32, h_normals);
        LOGV("normal finish.");

        nss->set_normals(h_normals, points_num);
        nss->set_points(h_points, points_num);
        nss->set_sampling_ratio(0.03);
        nss->downsampling(h_points);

        // shot->descriptor(h_points, points_num, h_lrfs, 5);
        // iss->detector(h_points, points_num, h_neighbor, 5, 32, feat_det_output);
        // std::vector<float> keypoints;
        // keypoints.reserve(points_num);
        // for (size_t i = 0; i < points_num; i++)
        // {
        //     if (feat_det_output[i*6+5] == 0)
        //     {
        //         for (int j = 0; j < 6; j++)
        //         {
        //             keypoints.push_back(feat_det_output[i * 6 + j]);
        //         }
        //     }
        // }
        // float* h_keypoints = keypoints.data();
        // int keypoints_num = keypoints.size() / 6;
        // LOGD("Keypoints num: %d", keypoints_num);
        
        // fpfh->descriptor(feat_det_output, points_num, h_neighbor, h_normals, 32, 5, h_feat);
        // for (size_t i = 0; i < points_num; i++)
        // {
        //     for (int j = 0; j < 33; j++)
        //     {
        //         printf("feat[%d] = %.6f\n", j, h_feat[i * 33 + j]);
        //     }
        //     printf("\n");
        // }
        
        delete[] h_feat;
        delete[] h_normals;
        delete[] h_neighbor;
        delete[] h_points;
    }


    return 0;
}