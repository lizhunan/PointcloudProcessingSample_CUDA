// test_cuda.cu
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include "common/include/pc.h"
#include "common/include/timer.h"
#include "denoising/include/stat_filter.h"
#include "cluster/include/dbscan.h"
#include "common/include/neighbor.h"
#include "common/include/mat.h"
#include "feature_description/include/shot.h"
#include "feature_description/include/fpfh.h"
#include "registration/include/icp.h"

int main() {
    std::shared_ptr<pc::Pointcloud>         pointcloud = std::make_unique<pc::Pointcloud>(false);
    std::shared_ptr<logger::Logger>         logger = std::make_unique<logger::Logger>(logger::Level::DEBUG);
    std::shared_ptr<denoising::StatFilter>  stat_filter = std::make_unique<denoising::StatFilter>(false);
    std::shared_ptr<cluster::DBSCAN>        dbscan = std::make_unique<cluster::DBSCAN>(true);
    std::shared_ptr<neighbor::Neighbor>     neighbor = std::make_unique<neighbor::Neighbor>(false);
    std::shared_ptr<mat::MAT>               mat = std::make_unique<mat::MAT>(false);
    std::shared_ptr<reg::ICP>               icp = std::make_unique<reg::ICP>(false);
    std::shared_ptr<timer::Timer>           timer = std::make_unique<timer::Timer>();
    timer->init();

    int k = 32;

    stat_filter->set_k(k);
    stat_filter->set_alpha(3.0f);
    stat_filter->set_radius(20.0f);

    dbscan->set_dist_epsilon(20.0f);
    dbscan->set_min_pts(32);
    dbscan->set_cell_size(2.0f);
    int range[] = {-50, 50, -20, 20, -2, 8};    // x min, x max, y min, y max, z min, z height
        
    icp->set_max_iterations(500);
    icp->set_max_correspondence_distance(0.5f);
    icp->set_transformation_epsilon(0.01);
    icp->set_euclidean_fitness_epsilon(0.01f);
    // Eigen::Matrix4f init_solution = Eigen::Matrix4f::Identity();
    // icp->set_transformation_init(init_solution);
    
    float* d_source; float* d_target;
    int source_num = 0; int target_num=0;
    CUDA_CHECK(cudaMalloc((void **)&d_source, sizeof(float) * 1000000*6));
    CUDA_CHECK(cudaMalloc((void **)&d_target, sizeof(float) * 1000000*6));

    // std::string file_dir = "/workspace/data/reg_pc_txt";
    // std::string file_dir = "/workspace/data/test_reg_txt";
    std::string file_dir = "/workspace/data/test";
    std::string file_path = file_dir + "/0.txt";
    std::string source_path, target_path;
    // for (size_t i = 1; i <= 5; i++)
    for (size_t i = 0; i < 1; i++)
    {
        // Data loading(0.txt)
        // Data loading(1.txt...i.txt)
        source_path = file_dir + "/" + std::to_string(i) + ".txt";
        target_path = file_dir + "/" + std::to_string(i+1) + ".txt";

        bool is_src_read = pointcloud->read_txt_xyz(source_path, d_source, source_num);
        bool is_tar_read = pointcloud->read_txt_xyz(target_path, d_target, target_num);

        float *h_source = new float[source_num*6];
        float *h_target = new float[target_num*6];
        int* h_neighbor = new int[source_num*(k+1)];
        float* h_normals = new float[source_num*3];
        
        CUDA_CHECK(cudaMemcpy(h_source, d_source, sizeof(float)*source_num*6, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_target, d_target, sizeof(float)*target_num*6, cudaMemcpyDeviceToHost));
        LOGV("Read source: %s, status: %d.", source_path.c_str(), is_src_read);
        LOGV("Read source: %s, status: %d.", target_path.c_str(), is_tar_read);
        // file_path = file_dir + "/" + std::to_string(i) + ".txt";

        neighbor->search_bf(h_source, source_num, k, h_neighbor);
        mat->normals_estimator(h_source, source_num, h_neighbor, k, h_normals);

        stat_filter->set_points(h_source, source_num);
        stat_filter->set_neighbors(h_neighbor, k, source_num);
        stat_filter->denoising(h_source);

        dbscan->set_points(h_source, source_num);
        // timer->start_cpu();
        dbscan->clustering(h_source, range);
        // timer->stop_cpu("DBSCAN pointcloud clustering time: ");
        // timer->show();

        Eigen::Matrix4f trans_mat;
        // icp->set_source(h_source, source_num);
        // icp->set_target(h_target, target_num);
        // icp->set_neighbors(h_neighbor, k, source_num);
        // icp->point_to_point(trans_mat, k);

        delete[] h_source;
        delete[] h_target;
        delete[] h_neighbor;
        delete[] h_normals;
    }

    return 0;
}