#ifndef __FPFH_H__
#define __FPFH_H__

#define HIST_DIM 33 // [] * 11 Bins - Total histogram dimension (3 features × 11 bins each)
#define BIN_SIZE 11 // bin size - Number of bins per feature (alpha, theta, rho)

#include "/workspace/src/common/include/logger.h"
#include "/workspace/src/display/include/display.h"
#include "/workspace/src/common/include/mat.h"
#include "/workspace/src/common/include/neighbor.h"

namespace feat_desc {

/**
 * @class FPFH
 * @brief Fast Point Feature Histograms (FPFH) descriptor computation for 3D point clouds.
 * 
 * @details
 * The FPFH descriptor captures local geometric characteristics of a 3D point cloud
 * by analyzing relationships between a keypoint and its neighbors using angular features.
 * 
 * Key features:
 * - Computes Simplified Point Feature Histograms (SPFH) for each point
 * - Combines keypoint's SPFH with weighted neighbors' SPFH
 * - Uses three angular features: alpha, theta, and rho
 * - Each feature quantized into 11 bins, total 33 bins per point
 * 
 * Algorithm steps:
 *   1. For each point, compute SPFH using local neighborhood relationships
 *   2. For keypoint, compute weighted sum of neighbors' SPFH features
 *   3. Final FPFH = SPFH(keypoint) + (1/k) * Σ(weighted neighbors' SPFH)
 */
class FPFH {

public:

    /**
     * @brief Default constructor - creates FPFH instance without visualization.
     */
    FPFH();

    /**
     * @brief Constructor with visualization option.
     * @param vis Enable/disable visualization of features (if true, creates display window)
     */
    FPFH(bool vis);

    /**
     * @brief Destructor - cleans up GPU memory and resources.
     */
    ~FPFH();

public:

    /**
     * @brief Compute FPFH descriptors for all points in the pointcloud.
     * 
     * @details
     * This is the main entry point for FPFH descriptor computation. It:
     *   1. Allocates GPU memory for points, neighbors, normals, and histograms
     *   2. Copies input data from host to device
     *   3. Launches CUDA kernel to compute FPFH descriptors in parallel
     *   4. Copies results back to host and cleans up GPU memory
     * 
     * Mathematical formulation:
     *   For each point p with neighbors N(p):
     *     SPFH(p) = histogram of {α, θ, ρ} for all neighbors within radius r
     *     FPFH(p) = SPFH(p) + (1/k) * Σ_{q∈N(p)} (1/||p-q||) * SPFH(q)
     * 
     * Where:
     *   α = v·n_q          (angle between normal of q and vector v)
     *   θ = atan2(w·n_q, u·n_q) (angle in the plane perpendicular to u)
     *   ρ = u·(p-q)/||p-q||     (projection of difference onto u)
     * 
     * @param points         Input point cloud array [N x POINT_DIM] (x,y,z coordinates)
     * @param points_num     Total number of points in the cloud
     * @param h_neighbors    Host neighbor index array [N x (k+1)] (first entry is self)
     * @param h_normals      Host surface normals array [N x 3]
     * @param k              Number of nearest neighbors to consider (excluding self)
     * @param r              Radius search threshold for neighbor selection
     * @param h_histogram    Output FPFH descriptors [N x HIST_DIM] (33 bins per point)
     */
    void descriptor(const float* points, const int points_num, const int* h_neighbors, const float* h_normals,
                    const int k, const float r, float* h_histogram);

private:
    bool vis;               ///< Flag to enable/disable visualization

    std::shared_ptr<logger::Logger> logger;         ///< Logging utility
    std::shared_ptr<display::Display> display;      ///< Visualization utility
    std::shared_ptr<neighbor::Neighbor> neighbor;   ///< Neighborhood search utility
    std::shared_ptr<mat::MAT> mat;                  ///< Mathematical operations utility

    float* d_points;        ///< Device pointer to pointcloud [N x POINT_DIM]
    int*   d_neighbors;     ///< Device pointer to neighbor indices [N x (k+1)]
    float* d_normals;       ///< Device pointer to surface normals [N x 3]
    float* d_histogram;     ///< Device pointer to output FPFH histograms [N x HIST_DIM]

};
}

#endif