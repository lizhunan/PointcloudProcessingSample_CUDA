#ifndef __NSS_H__
#define __NSS_H__

#include <curand_kernel.h>
#include "/workspace/src/common/include/logger.h"
#include "display.h"
#include "/workspace/src/common/include/mat.h"

namespace downsampling {

class NSS {

public:

    /**
     * @brief Default constructor - creates NSS without visualization.
     */
    NSS();

    /**
     * @brief Constructor with visualization option.
     * @param vis Enable/disable visualization of downsampling results
     */
    NSS(bool vis);

    /**
     * @brief Destructor - cleans up GPU memory and resources.
     */
    ~NSS();

public:

    void downsampling(float* h_points_out);

    /**
     * @brief Set the input pointcloud and allocate GPU memory.
     * 
     * @param points      Host pointcloud array [N x POINT_DIM]
     * @param source_num  Number of pointcloud
     */
    void set_points(const float* points, const int points_num);

    void set_normals(const float* normals, const int normals_num);

    void set_sampling_ratio(float ratio) { this->sampling_ratio = ratio; }

private:
    bool vis;

    int points_num          = 0;        ///< Total number of points in the cloud
    int normals_num         = 0;        ///< Total number of normals in the cloud
    float sampling_ratio    = 0.5f;     ///< Ratio of points to sample (e.g., 0.5 means keep 50% of points)

    std::unique_ptr<downsampling::Display> display;
    std::unique_ptr<mat::MAT> mat;

    float* d_points;                ///< Device pointer to input point cloud [N x POINT_DIM]
    float* d_points_out;            ///< Device pointer to output point cloud [N x POINT_DIM]
    float* d_normals;               ///< Device pointer to input normals [N x 3]
    float* d_normal_space;          ///< Device pointer to normal space values [N x 2]
    int*   d_bucket_counts;         ///< Device pointer to bucket counts for sampling [n_theta x n_phi]
    int*   d_bucket_ids;
    int*   d_samples_per_bucket;
    int*   d_bucket_offsets;
    int*   d_bucket_pos_counter;
    int*   d_sorted_indices;
    int*   d_out_counter;
};
}

#endif