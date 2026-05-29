#ifndef __ICP_H__
#define __ICP_H__

#include <random>
#include "/workspace/src/common/include/logger.h"
#include "display.h"
#include "/workspace/src/common/include/mat.h"
#include "/workspace/src/common/include/neighbor.h"
#include "/workspace/src/feature_detection/include/iss.h"
#include "/workspace/src/feature_description/include/fpfh.h"

namespace reg {

class ICP {

public:
    ICP();
    ICP(bool vis);
    ~ICP();

public:
    float point_to_point(Eigen::Matrix4f transformation, const int k);
    float point_to_plane(Eigen::Matrix4f transformation);
    float plant_to_plane(Eigen::Matrix4f transformation);

    void set_source(const float* points, const int source_num);
    void set_target(const float* points, const int target_num);
    void set_neighbors(const int* neighbors, const int k, const int points_num);
    void set_transformation_init(Eigen::Matrix4f trans_init);
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    void set_max_correspondence_distance(float dis);
    // Set the maximum number of iterations (criterion 1)
    void set_max_iterations(int n);
    // Set the transformation epsilon (criterion 2)
    void set_transformation_epsilon(double eps);
    // Set the euclidean distance difference epsilon (criterion 3)
    void set_euclidean_fitness_epsilon(float eps);

private:
    void match(const float* src, const float* tar, const int src_num, const int tar_num, 
               const float* src_feat, const float* tar_feat, float* src_out, float* tar_out, int& matched_num);
    bool init_solution_estimator(const float* src, const float* tar, const int src_num,
                                  const int max_iterations, float inlier_threshold, int min_inliers,
                                  float* init_solution);
    bool Rt_3_points(const float* src, const float* tar, const int* indices, float* T);
    void solve_svd(const float* source_centered, const float* target_centered, const float* source_center, const float* target_center,
                   const int num_corrs, float* R, float* t);

private:
    bool vis;
    bool is_init_solution;
    std::mt19937 rng;
    
    int     source_num                  = 0;
    int     target_num                  = 0;
    float   max_correspondence_distance = 0.05f;
    int     max_iterations              = 50;
    double  transformation_epsilon      = 1e-8;
    float   euclidean_fitness_epsilon   = 1.0f;

    std::shared_ptr<logger::Logger> logger;
    std::shared_ptr<reg_display::Display> display;
    std::shared_ptr<neighbor::Neighbor> neighbor;
    std::shared_ptr<mat::MAT> mat;
    std::shared_ptr<iss::ISS> feat_det;
    std::shared_ptr<feat_desc::FPFH> feat_desc;

    float* d_source;
    float* d_target;
    float* d_source_corr;
    float* d_target_corr;
    float* d_src_centered;
    float* d_tar_centered;
    float* d_points;
    int*   d_neighbors;
    float* d_loss;
    int*   d_valid_corr;
    float* d_src_center;
    float* d_tar_center;

};
}

#endif