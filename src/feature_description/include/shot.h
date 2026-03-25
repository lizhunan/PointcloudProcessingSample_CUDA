#ifndef __SHOT_H__
#define __SHOT_H__

#include "/workspace/src/common/include/logger.h"
#include "/workspace/src/display/include/display.h"
#include "/workspace/src/common/include/mat.h"
#include "/workspace/src/common/include/neighbor.h"

namespace shot {

class SHOT {

public:
    SHOT();
    SHOT(bool vis);
    ~SHOT();

public:

    /**
     * @brief Compute SHOT descriptors for a given pointcloud.
     *
     * @details
     * This function implements the full SHOT (Signature of Histograms of Orientations)
     * pipeline, including:
     *
     *   1. Neighbor search (k-NN or radius-based)
     *   2. Surface normal estimation
     *   3. Local Reference Frame (LRF) construction
     *   4. Spatial + angular histogram encoding
     *
     * The final descriptor is a concatenation of histograms over spatial bins:
     *
     *      feature_dim = bins_radial × bins_azimuth × bins_elevation × bins_size
     *
     * Default configuration:
     *      bins_radial   = 2
     *      bins_azimuth  = 8
     *      bins_elevation= 2
     *      bins_size     = 11
     * → feature_dim = 2 × 8 × 2 × 11 = 352
     *
     * This matches the standard SHOT352 descriptor [Tombari et al., ECCV 2010].
     *
     * @param[in]  points        Input point cloud (flattened array, size = N × 3)
     * @param[in]  points_num    Number of points (N)
     * @param[out] feat          Output descriptor array (size = N × feature_dim)
     *                          Each point corresponds to one descriptor
     *
     * @param[in]  radius        Support radius for neighborhood definition
     *                          (defines the local surface region)
     *
     * @param[in]  bins_azimuth   Number of bins along azimuth angle (α ∈ [0, 2π])
     * @param[in]  bins_elevation Number of bins along elevation angle (β ∈ [0, π])
     * @param[in]  bins_radial    Number of radial bins (ρ ∈ [0, radius])
     * @param[in]  bins_size      Number of bins for cosine similarity (normal angle)
     *
     * @note
     * - The descriptor is invariant to rotation due to LRF normalization.
     * - Robustness comes from histogram aggregation + interpolation.
     * - Requires reliable normal estimation and stable LRF.
     *
     * @warning
     * - `feat` must be pre-allocated with size (points_num × feature_dim)
     * - Memory alignment should be ensured for GPU transfers
     */
    void descriptor(const float* points, const int points_num, float* feat,
                const float radius, const int bins_azimuth=8, const int bins_elevation=2,
                const int bins_radial=2, const int bins_size=11);
private:

    /**
     * @brief Compute Local Reference Frame (LRF) for each point.
     *
     * @details
     * For each keypoint, an orthonormal coordinate system (LRF) is constructed
     * using weighted PCA over its neighborhood:
     *
     *   1. Compute weighted covariance matrix
     *   2. Eigen-decomposition
     *   3. Select principal directions
     *   4. Enforce sign disambiguation for consistency
     *
     * The resulting LRF is:
     *      R = [x_axis, y_axis, z_axis] ∈ ℝ^{3×3}
     *
     * where:
     *   - z-axis: normal-like direction
     *   - x-axis: principal curvature direction
     *   - y-axis: z × x (right-handed system)
     *
     * @param[in]  h_points      Host point cloud (N × 3)
     * @param[in]  points_num    Number of points
     * @param[in]  h_neighbors   Neighbor indices (N × (k+1))
     *                          (first index is usually the point itself)
     * @param[in]  k             Number of neighbors
     * @param[in]  radius        Support radius for weighting
     *
     * @param[out] lrfs          Output LRFs (N × 9)
     *                          Stored as 3 column vectors:
     *                              [x_axis | y_axis | z_axis]
     *
     * @note
     * - LRF is critical for rotation invariance.
     * - Instability may occur in flat or noisy regions.
     *
     * @warning
     * - Neighbor indices must be valid (no out-of-bound access)
     * - Degenerate neighborhoods may lead to unstable eigenvectors
     */
    void local_reference_frame(const float* h_points, const int points_num, const int* h_neighbors, const int k, const float radius, float* lrfs);

    /**
     * @brief Construct SHOT histogram descriptors using LRF and normals.
     *
     * @details
     * For each point:
     *
     *   1. Transform neighbor points into LRF coordinates
     *   2. Convert to spherical coordinates (ρ, α, β)
     *   3. Compute cosine similarity between normals:
     *
     *          cosθ = n_i ⋅ n_k
     *
     *   4. Quantize into histogram bins:
     *          - radial      (ρ)
     *          - azimuth     (α)
     *          - elevation   (β)
     *          - cosine      (cosθ)
     *
     *   5. Apply quadrilinear interpolation to distribute weights
     *   6. Accumulate into descriptor vector
     *   7. Normalize (L2)
     *
     * @param[in]  h_points      Input point cloud (N × 3)
     * @param[in]  points_num    Number of points
     * @param[in]  h_neighbors   Neighbor indices (N × (k+1))
     * @param[in]  h_lrfs        Precomputed LRFs (N × 9)
     * @param[in]  h_normals     Surface normals (N × 3)
     *
     * @param[in]  bins_size      Number of bins for cosine similarity
     * @param[in]  bins_radial    Number of radial bins
     * @param[in]  bins_azimuth   Number of azimuth bins
     * @param[in]  bins_elevation Number of elevation bins
     *
     * @param[in]  k              Number of neighbors
     * @param[in]  radius         Support radius
     *
     * @param[in]  feat_dim       Descriptor dimension
     *                           (= bins_radial × bins_azimuth × bins_elevation × bins_size)
     *
     * @param[out] feat           Output descriptors (N × feat_dim)
     *
     * @note
     * - Histogram interpolation reduces discretization artifacts
     * - L2 normalization ensures scale invariance
     *
     * @warning
     * - Incorrect bin indexing may cause illegal memory access (CUDA)
     * - All bin indices must be clamped within valid range
     * - Neighbor index must be checked before dereferencing
     */
    void histogram(const float* h_points, const int points_num, const int* h_neighbors, const float* h_lrfs, const float* h_normals,
                const int bins_size, const int bins_radial, const int bins_azimuth, const int bins_elevation,
                const float k, const float radius, const int feat_dim, float* feat);

private:
    bool vis;

    std::shared_ptr<logger::Logger> logger;
    std::shared_ptr<display::Display> display;
    std::shared_ptr<neighbor::Neighbor> neighbor;
    std::shared_ptr<mat::MAT> mat;

    /* GPU memory */
    float*      d_points;           // Device point cloud
    int*        d_neighbors;        // Device neighbor indices
    float*      d_lrfs;             // Device Local Reference Frame(LRF)
    float*      d_normals;          // Device normal vectors
    float*      d_feat;             // Device featrue descpription
};

}

#endif