#ifndef __STAT_FILTER_H__
#define __STAT_FILTER_H__

#include "/workspace/src/common/include/logger.h"
#include "display.h"
#include "/workspace/src/common/include/mat.h"

namespace denoising {

/**
 * @class StatFilter
 * @brief Statistical outlier removal filter for 3D point cloud denoising.
 * 
 * @details
 * This class implements a statistical outlier removal algorithm that identifies and
 * removes noise points based on the Gaussian distribution of mean distances
 * to neighboring points.
 * 
 * Mathematical formulation:
 *   1. For each point p_i, compute mean distance to its k nearest neighbors:
 *      d̄_i = (1/k) * Σ_{j=1}^{k} ||p_i - p_j||
 * 
 *   2. Assuming d̄_i follows a Gaussian distribution, compute global statistics:
 *      μ = (1/N) * Σ_{i=1}^{N} d̄_i
 *      σ = sqrt((1/N) * Σ_{i=1}^{N} (d̄_i - μ)²)
 * 
 *   3. Point is classified as outlier if:
 *      d̄_i > μ + α·σ
 * 
 * Key features:
 *   - Automatically computes global mean and standard deviation of neighbor distances
 *   - Uses threshold multiplier α to control outlier sensitivity
 *   - Preserves original point attributes (coordinates, intensity, etc.)
 *   - Marks outliers with special flag (-2) in the 6th dimension
 */
class StatFilter {

public:

    /**
     * @brief Default constructor - creates StatFilter without visualization.
     */
    StatFilter();

    /**
     * @brief Constructor with visualization option.
     * @param vis Enable/disable visualization of denoising results
     */
    StatFilter(bool vis);

    /**
     * @brief Destructor - cleans up GPU memory and resources.
     */
    ~StatFilter();

public:

    /**
     * @brief Execute the statistical outlier removal process.
     * 
     * @details
     * This is the main denoising pipeline that:
     *   1. Allocates GPU memory for intermediate results
     *   2. Computes mean distances for all points in parallel
     *   3. Calculates global mean (μ) and standard deviation (σ)
     *   4. Applies outlier classification using threshold μ + α·σ
     *   5. Copies results back to host and optionally visualizes
     * 
     * The output maintains the same point structure as input, with outliers
     * marked by setting the 6th coordinate (index 5) to -2.
     * 
     * @param h_points_out Output point cloud array [N x POINT_DIM]
     *                     Points are preserved, outliers flagged in dimension 5
     */
    void denoising(float* h_points_out);

    /**
     * @brief Set the input point cloud and allocate GPU memory.
     * 
     * @param points      Host point cloud array [N x POINT_DIM]
     * @param source_num  Number of pointcloud
     */
    void set_points(const float* points, const int source_num);

    /**
     * @brief Set the neighbor indices for all points.
     * 
     * @param neighbors   Host neighbor index array [N x (k+1)], first entry is self
     * @param k           Number of nearest neighbors (excluding self)
     * @param points_num  Total number of points in the cloud
     */
    void set_neighbors(const int* neighbors, const int k, const int points_num);

    /**
     * @brief Set the radius threshold for neighbor selection.
     * @param r Maximum distance to consider a point as neighbor
     */
    void set_radius(const float r);

    /**
     * @brief Set the number of nearest neighbors.
     * @param k Number of neighbors to consider for distance calculation
     */
    void set_k(const int k);

    /**
     * @brief Set the outlier detection threshold multiplier.
     * 
     * @param alpha Multiplier for standard deviation in threshold calculation
     *              Default: 1.0
     *              Lower values: more aggressive outlier removal
     *              Higher values: more conservative removal
     */
    void set_alpha(const float alpha);

private:
    bool vis;

    int points_num      = 0;        ///< Total number of points in the cloud
    float r             = 0.0f;     ///< Radius threshold for neighbor inclusion
    int k               = 0;        ///< Number of nearest neighbors to consider
    float alpha         = 1.0f;     ///< Threshold multiplier (μ + α·σ) for outlier detection

    std::unique_ptr<denoising_display::Display> display;
    std::unique_ptr<mat::MAT> mat;

    float* d_points;                ///< Device pointer to input point cloud [N x POINT_DIM]
    int*   d_neighbors;             ///< Device pointer to neighbor indices [N x (k+1)]
    float* d_points_out;            ///< Device pointer to output point cloud [N x POINT_DIM]
    float* d_mean_dists;            ///< Device pointer to mean distances for each point [N]
    
};
}

#endif