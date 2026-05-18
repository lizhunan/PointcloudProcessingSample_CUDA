#ifndef __ICP_H__
#define __ICP_H__

#include "/workspace/src/common/include/logger.h"
#include "/workspace/src/display/include/display.h"
#include "/workspace/src/common/include/mat.h"
#include "/workspace/src/common/include/neighbor.h"

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
    void set_transformation_init(Eigen::Matrix4f trans_init= Eigen::Matrix4f::Identity(4, 4));
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    void set_max_correspondence_distance(int dis=0.05);
    // Set the maximum number of iterations (criterion 1)
    void set_max_iterations(int n=50);
    // Set the transformation epsilon (criterion 2)
    void set_transformation_epsilon(int eps=1e-8);
    // Set the euclidean distance difference epsilon (criterion 3)
    void set_euclidean_fitness_epsilon(int eps=1);

private:
    bool vis;
    
    int     source_num                  = 0;
    int     target_num                  = 0;
    float   max_correspondence_distance = 0.05f;
    float   max_iterations              = 50;
    float   transformation_epsilon      = 1e-8;
    float   euclidean_fitness_epsilon   = 1;

    std::shared_ptr<logger::Logger> logger;
    std::shared_ptr<display::Display> display;
    std::shared_ptr<neighbor::Neighbor> neighbor;
    std::shared_ptr<mat::MAT> mat;

    float* d_source;
    float* d_target;
    float* d_points;
    int*   d_neighbors;
    float* d_loss;
    int*   d_valid_crpd;

};
}

#endif