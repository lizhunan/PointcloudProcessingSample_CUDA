#include "shot.h"

namespace shot {

/**
 * @brief Transform a point from Cartesian coordinates to Local Reference Frame (LRF).
 *
 * @details
 *  Given a point p and a keypoint center, this function:
 *   1. Translates the point to local coordinates
 *   2. Projects it onto the LRF (rotation normalization)
 *   3. Converts it into spherical coordinates (ρ, α, β)
 *
 *  Mathematical formulation:
 *
 *      d = p - center
 *      local = R^T * d
 *
 *      ρ     = ||local||
 *      α     = atan2(y, x)
 *      β     = arccos(z / ρ)
 *
 * @param p        Input point (3D)
 * @param center   Keypoint center (3D)
 * @param lrf      Local reference frame (3x3, column = axis)
 * @param rho      Output radial distance
 * @param alpha    Output azimuth angle [-π, π]
 * @param beta     Output elevation angle [0, π]
 */
__device__ void cartesian2lrf(const float* p_k, const float* p_i, const float* lrf, float& rho, float& alpha, float& beta)
{
    // Translation
    float dx = p_i[0] - p_k[0];
    float dy = p_i[1] - p_k[1];
    float dz = p_i[2] - p_k[2];

    // Projection to LRF (R^T * d)
    float x = lrf[0] * dx + lrf[1] * dy + lrf[2] * dz;
    float y = lrf[3] * dx + lrf[4] * dy + lrf[5] * dz;
    float z = lrf[6] * dx + lrf[7] * dy + lrf[8] * dz;

    // Build shperical coordinates
    rho     = mat::distf(p_k, p_i);
    alpha   = atan2f(y, x);
    if (rho > 1e-6f)
    {
        float cos_theta = z / rho;
        cos_theta = fminf(1.0f, fmaxf(-1.0f, cos_theta));
        beta = acosf(cos_theta);
    }
    else
    {
        beta = 0.0f;
    }
}

/**
 * @brief CUDA kernel for computing Local Reference Frames (LRF) for each point.
 *
 * @details
 * This kernel computes a Local Reference Frame (LRF) for every point in the input
 * point cloud. The LRF provides a rotation-invariant coordinate system, which is
 * a critical prerequisite for SHOT descriptor construction.
 *
 * For each point (keypoint), the algorithm performs:
 *
 *   1. Gather k-nearest neighbors
 *   2. Compute a weighted covariance matrix:
 *
 *          M = Σ w_i (p_i - p_k)(p_i - p_k)^T
 *
 *      where:
 *          - p_k : keypoint
 *          - p_i : neighbor point
 *          - w_i : distance-based weight (typically decreases with distance)
 *
 *   3. Perform eigen-decomposition of M:
 *
 *          M v_j = λ_j v_j
 *
 *   4. Construct orthonormal axes:
 *
 *          z-axis → eigenvector with smallest eigenvalue (normal direction)
 *          x-axis → eigenvector with largest eigenvalue (principal direction)
 *          y-axis → z × x (right-handed coordinate system)
 *
 *   5. Apply sign disambiguation:
 *      Eigenvectors are direction-ambiguous (±v), so consistency is enforced
 *      by checking alignment with neighbor distribution.
 *
 * The final LRF is:
 *
 *      R = [x_axis | y_axis | z_axis] ∈ ℝ^{3×3}
 *
 * ------------------------------------------------------------------------
 * CUDA Execution Model:
 * ------------------------------------------------------------------------
 * - 1 thread  → 1 point (keypoint)
 * - Each thread independently computes its LRF
 * - No inter-thread synchronization is required
 *
 * ------------------------------------------------------------------------
 * @param[in]  points
 *      Input point cloud (flattened array, size = points_num × 3)
 *      Layout: [x0,y0,z0, x1,y1,z1, ...]
 *
 * @param[in]  points_num
 *      Number of points (N)
 *
 * @param[in]  neighbors
 *      Neighbor index array (size = N × (k+1))
 *      Each row corresponds to a point:
 *          neighbors[i*(k+1) + 0] → usually the point itself
 *          neighbors[i*(k+1) + j] → j-th neighbor
 *
 * @param[in]  k
 *      Number of neighbors used for LRF estimation
 *
 * @param[in]  radius
 *      Support radius used for weighting neighbors
 *      (in covariance matrix computation)
 *
 * @param[out] lrfs
 *      Output Local Reference Frames (size = N × 9)
 *      Stored in column-major format:
 *
 *          lrfs[i*9 + 0..2] → x-axis
 *          lrfs[i*9 + 3..5] → y-axis
 *          lrfs[i*9 + 6..8] → z-axis
 *
 * ------------------------------------------------------------------------
 * @note
 * - LRF ensures rotation invariance for local descriptors (e.g., SHOT).
 * - Weighting scheme improves robustness against noise and uneven sampling.
 * - Eigen-decomposition is performed per thread (computationally intensive).
 *
 * ------------------------------------------------------------------------
 * @warning
 * - Degenerate neighborhoods (e.g., planar or collinear points) may lead to
 *   unstable eigenvectors and unreliable LRFs.
 * - Neighbor indices must be valid (0 ≤ idx < points_num).
 * - Eigenvector sign ambiguity must be resolved (otherwise descriptor flips).
 *
 * ------------------------------------------------------------------------
 * @performance
 * - Kernel is compute-heavy due to eigen-decomposition.
 * - Potential optimizations:
 *      - Use shared memory for neighbor reuse
 *      - Reduce eigen-solver cost (closed-form or approximation)
 *      - Precompute weights efficiently
 *
 * ------------------------------------------------------------------------
 * @reference
 * Tombari, Federico, et al.
 * "Unique signatures of histograms for local surface description."
 * ECCV 2010.
 */
__global__ void lrf(const float* points, const int points_num, const int* neighbors, const int k, const float radius, float* lrfs)
{   
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid >= points_num) return;

    int stride = k + 1;
    int base = threadid * stride;

    // Build weighted covariance matrix
    float M[3][3] = {0};
    bool conv_rst = mat::conv_weight(points, threadid, points_num, neighbors, k, radius, M);
    if (conv_rst)
    {
        float eigval[3];
        float eigvec[3][3];
        mat::jacobi_3x3(M, eigval, eigvec);

        // Sign disambiguation
        float x[3] = {eigvec[0][0], eigvec[1][0], eigvec[2][0]};
        float z[3] = {eigvec[0][2], eigvec[1][2], eigvec[2][2]};
        float sign_z = 0.0f;
        float sign_x = 0.0f;
        for (size_t i = 1; i <= k; i++)
        {
            int idx = neighbors[base + i];
            if (idx < 0) continue;
            float dx = points[idx*POINT_DIM+0] - points[threadid * POINT_DIM + 0];
            float dy = points[idx*POINT_DIM+1] - points[threadid * POINT_DIM + 1];
            float dz = points[idx*POINT_DIM+2] - points[threadid * POINT_DIM + 2];

            sign_z += dx*z[0] + dy*z[1] + dz*z[2];
            sign_x += dx*x[0] + dy*x[1] + dz*x[2];
        }
        
        if (sign_z < 0)
        {
            z[0]*=-1; z[1]*=-1; z[2]*=-1;
        }

        if (sign_x < 0)
        {
            x[0]*=-1; x[1]*=-1; x[2]*=-1;
        }

        // Construct y axis (right-handed system)
        float y[3];
        y[0] = z[1]*x[2] - z[2]*x[1];
        y[1] = z[2]*x[0] - z[0]*x[2];
        y[2] = z[0]*x[1] - z[1]*x[0];

        // Store LRF (column-wise)
        int offset = threadid*9;
        for(int i=0;i<3;i++)
        {
            lrfs[offset + i*3 + 0] = x[i];
            lrfs[offset + i*3 + 1] = y[i];
            lrfs[offset + i*3 + 2] = z[i];
        }

    }else{}
}

/**
 * @brief CUDA kernel for constructing SHOT descriptors via histogram binning.
 *
 * @details
 * This kernel implements the core step of the SHOT (Signature of Histograms of
 * Orientations) descriptor: histogram construction in a Local Reference Frame (LRF).
 *
 * Each CUDA thread processes one keypoint independently:
 *
 *   1. Retrieve its k-nearest neighbors
 *   2. Transform neighbor points into the local reference frame (LRF)
 *   3. Convert Cartesian coordinates → spherical coordinates (ρ, α, β)
 *   4. Compute angular similarity between normals:
 *
 *          cosθ = n_i ⋅ n_k
 *
 *   5. Quantize (ρ, α, β, cosθ) into histogram bins
 *   6. Apply quadrilinear interpolation (soft binning)
 *   7. Accumulate contributions into the descriptor vector
 *   8. Normalize descriptor (L2 normalization)
 *
 * The descriptor layout follows:
 *
 *      feature_dim = bins_radial × bins_elevation × bins_azimuth × bins_size
 *
 * where:
 *   - bins_radial    : number of radial partitions
 *   - bins_azimuth   : number of azimuth partitions
 *   - bins_elevation : number of elevation partitions
 *   - bins_size      : number of bins for cosine similarity
 *
 * This design matches the SHOT352 descriptor [Tombari et al., ECCV 2010].
 *
 * ------------------------------------------------------------------------
 * CUDA Execution Model:
 * ------------------------------------------------------------------------
 * - 1 thread  → 1 keypoint
 * - Each thread builds its own histogram (no inter-thread communication)
 * - Memory access pattern:
 *      - Global memory reads: points, neighbors, normals, lrfs
 *      - Global memory writes: feat (descriptor output)
 *
 * ------------------------------------------------------------------------
 * @param[in]  points
 *      Flattened point cloud array (size = points_num × POINT_DIM)
 *      Layout: [x0,y0,z0, x1,y1,z1, ...]
 *
 * @param[in]  points_num
 *      Total number of points (N)
 *
 * @param[in]  neighbors
 *      Neighbor index array (size = N × (k+1))
 *      Each row contains:
 *          neighbors[i*(k+1) + 0] → typically the point itself
 *          neighbors[i*(k+1) + j] → j-th neighbor
 *
 * @param[in]  lrfs
 *      Local Reference Frames (size = N × 9)
 *      Stored as 3 orthonormal axes (column-major):
 *          [x_axis | y_axis | z_axis]
 *
 * @param[in]  normals
 *      Surface normals (size = N × 3)
 *      Assumed to be unit vectors
 *
 * @param[in]  bins_azimuth
 *      Number of bins for azimuth angle α ∈ [0, 2π]
 *
 * @param[in]  bins_elevation
 *      Number of bins for elevation angle β ∈ [0, π]
 *
 * @param[in]  feat_dim
 *      Descriptor dimension:
 *          bins_radial × bins_azimuth × bins_elevation × bins_size
 *
 * @param[in]  k
 *      Number of neighbors per point
 *
 * @param[in]  radius
 *      Support radius defining local neighborhood
 *
 * @param[in]  bin_size
 *      Number of bins for cosine similarity (cosθ)
 *
 * @param[out] feat
 *      Output descriptor array (size = N × feat_dim)
 *      Each thread writes one descriptor
 *
 * ------------------------------------------------------------------------
 * @note
 * - Quadrilinear interpolation distributes each sample into up to 11 bins.
 * - LRF ensures rotation invariance.
 * - L2 normalization improves robustness to density variation.
 *
 * ------------------------------------------------------------------------
 * @warning
 * - Neighbor indices must be valid (0 ≤ idx < points_num)
 * - Bin indices must be clamped to avoid out-of-bound writes
 * - Incorrect bin resolution (e.g., misuse of bin_size) can corrupt descriptors
 * - Race conditions are avoided since each thread writes its own descriptor
 *
 * ------------------------------------------------------------------------
 * @performance
 * - Memory-bound kernel due to scattered global reads
 * - Potential optimizations:
 *      - Use float3 for coalesced loads
 *      - Shared memory histogram buffering
 *      - Warp-level reduction
 *
 * ------------------------------------------------------------------------
 * @reference
 * Tombari, Federico, et al.
 * "Unique signatures of histograms for local surface description."
 * ECCV 2010.
 */
__global__ void binning(const float* points, const int points_num, const int* neighbors, const float* lrfs, const float* normals, 
                        const int bins_azimuth, const int bins_elevation, const int feat_dim,
                        const int k, const float radius, const int bin_size, float* feat)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid >= points_num) return;

    // Load keypoint coordinates p_k
    float point[3] = {points[threadid*POINT_DIM+0], points[threadid*POINT_DIM+1], points[threadid*POINT_DIM+2]};    

    // Neighbor indexing
    // Each point has (k+1) neighbors (including itself at index 0)
    int stride = k + 1;
    int base = threadid * stride;

    // Initialize histogram descriptor
    // desc is a slice of global memory (no local malloc!)
    float* desc = feat + threadid * feat_dim;
    for (int i = 0; i < feat_dim; i++) desc[i] = 0.0f;

    // Load Local Reference Frame (LRF)
    // Stored as 3 column vectors (x, y, z) axes
    const float* lrf = &lrfs[threadid * 9];

    for (int i = 1; i <= k; i++)
    {
        int idx = neighbors[base + i];

        // IMPORTANT: must check validity
        if (idx < 0 || idx >= points_num) continue;

        // // Load neighbor point p_i
        float neighbor[3] = {points[idx * POINT_DIM + 0], points[idx * POINT_DIM + 1], points[idx * POINT_DIM + 2]};  
        
        // Transform to LRF and compute spherical coords
        // 
        // (ρ, α, β):
        //   ρ     = radial distance
        //   α     = azimuth angle [-π, π]
        //   β     = elevation angle [0, π]
        float rho           = 0.0f;
        float alpha         = 0.0f;
        float beta          = 0.0f;
        float cos_theta     = 0.0f;
        cartesian2lrf(neighbor, point, lrf, rho, alpha, beta);

        // Ignore points outside support radius
        if (rho > radius) continue;

        // Compute cosine of normal angle
        //
        // cosθ = n_i ⋅ n_k
        // normals are assumed normalized
        cos_theta = mat::dot3f(&normals[idx*3], &normals[threadid*3]);

        // Normalize value ranges for histogram
        //
        // α: [-π, π] → [0, 2π]
        // cosθ: [-1, 1] → [0, 2]
        alpha       += M_PI;
        cos_theta   += 1.0f;

        // Quadrilinear interpolation (4D binning)
        //
        // Each dimension contributes up to 2 bins:
        // → total max combinations = 2^4 = 16
        //
        // ---- radial ----
        int r_bins[2]; float r_weights[2]; int r_count;
        mat::binning_weight(rho, radius/bin_size, radius, r_bins, r_weights, r_count);
        // ---- azimuth ----
        int az_bins[2]; float az_weights[2]; int az_count;
        mat::binning_weight(alpha, 2*M_PI/bin_size, 2*M_PI, az_bins, az_weights, az_count);
        // ---- elevation ----
        int el_bins[2]; float el_weights[2]; int el_count;
        mat::binning_weight(beta, M_PI/bin_size, M_PI, el_bins, el_weights, el_count);
        // ---- cosine ----
        int cos_bins[2]; float cos_weights[2]; int cos_count;
        mat::binning_weight(cos_theta, 2.0f/bin_size, 2.0f, cos_bins, cos_weights, cos_count);

        // Accumulate weighted contributions
        for (int ri = 0; ri < r_count; ri++)
        for (int ai = 0; ai < az_count; ai++)
        for (int ei = 0; ei < el_count; ei++)
        for (int ci = 0; ci < cos_count; ci++)
        {
            // Combined weight (product of 4 dimensions)
            float w =   r_weights[ri] *
                        az_weights[ai] *
                        el_weights[ei] *
                        cos_weights[ci];
            
            // Flatten 4D bin index → 1D descriptor index
            //
            // Layout:
            // [radial][elevation][azimuth][cosine]
            //
            // MUST ensure:
            // ri < bins_radial
            // ai < bins_azimuth
            // ei < bins_elevation
            // ci < bin_size
            int idx = ((ri*bins_elevation+ei)*bins_azimuth+ai)*bin_size+ci;
            
            //CRITICAL: bounds check to avoid illegal memory access
            if (idx >= 0 && idx < feat_dim)
                desc[idx] += w;
        }
    }
    // L2 normalization
    //
    // Ensures descriptor invariance to point density
    float norm = 0.0f;
    for (int i = 0; i < feat_dim; i++) norm += desc[i]*desc[i];
    norm = sqrtf(norm);
    if (norm > 1e-6f)
    {
        for (int i = 0; i < feat_dim; i++)
            desc[i] /= norm;
    }
    // -----------------------------
    // write back
    // -----------------------------
    for (int i = 0; i < feat_dim; i++)
        feat[threadid*feat_dim + i] = desc[i];
}

SHOT::SHOT(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("SHOT");
    mat         = std::make_unique<mat::MAT>(false);
    neighbor    = std::make_unique<neighbor::Neighbor>(false);
}

void SHOT::local_reference_frame(const float* h_points, const int points_num, const int* h_neighbors, const int k, const float radius, float* lrfs)
{
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*points_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_neighbors,    sizeof(int)*points_num*(k+1)));
    CUDA_CHECK(cudaMalloc((void **)&d_lrfs,         sizeof(float)*points_num*9));
    
    // Copy pointcloud to GPU
    CUDA_CHECK(cudaMemcpy(d_points, h_points,       sizeof(float)*points_num*POINT_DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, h_neighbors, sizeof(int)*points_num*(k+1), cudaMemcpyHostToDevice));

    int grid_num    = (points_num+1024-1)/1024;
    int block_num   = 1024;   
    lrf<<<grid_num, block_num>>>(d_points, points_num, d_neighbors, k, radius, d_lrfs);
    CUDA_CHECK(cudaMemcpy(lrfs, d_lrfs, sizeof(float)*points_num*9, cudaMemcpyDeviceToHost));

    if (this->vis)
    {
        float* h_lrfs = new float[points_num*9];
        CUDA_CHECK(cudaMemcpy(h_lrfs, d_lrfs, sizeof(float)*points_num*9, cudaMemcpyDeviceToHost));
        for (int i=0; i< 10; i++)
        {
            float x[3] = {h_lrfs[i*9+0], h_lrfs[i*9+3], h_lrfs[i*9+6]};
            float y[3] = {h_lrfs[i*9+1], h_lrfs[i*9+4], h_lrfs[i*9+7]};
            float z[3] = {h_lrfs[i*9+2], h_lrfs[i*9+5], h_lrfs[i*9+8]};

            LOGV("Point %d:", i);
            LOGV("LRF  x: (%f, %f, %f)", x[0], x[1], x[2]);
            LOGV("LRF  y: (%f, %f, %f)", y[0], y[1], y[2]);
            LOGV("LRF  z: (%f, %f, %f)", z[0], z[1], z[2]);
        }
        display->set_lrfs(h_lrfs, 10, points_num);
        display->set_pointcloud_xyz(h_points, points_num);
        delete[] h_lrfs;
    }

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_lrfs));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_points));
}

void SHOT::histogram(const float* h_points, const int points_num, const int* h_neighbors, const float* h_lrfs, const float* h_normals,
                     const int bins_size, const int bins_radial, const int bins_azimuth, const int bins_elevation,
                     const float k, const float radius, const int feat_dim, float* feat)
{
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*points_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_neighbors,    sizeof(int)*points_num*(32+1)));
    CUDA_CHECK(cudaMalloc((void **)&d_normals,      sizeof(float)*points_num*3));
    CUDA_CHECK(cudaMalloc((void **)&d_lrfs,         sizeof(float)*points_num*9));
    CUDA_CHECK(cudaMalloc((void **)&d_feat,         sizeof(float)*points_num*feat_dim));

    // Copy pointcloud to GPU
    CUDA_CHECK(cudaMemcpy(d_points, h_points,       sizeof(float)*points_num*POINT_DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, h_neighbors, sizeof(int)*points_num*(32+1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normals, h_normals,     sizeof(float)*points_num*3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lrfs, h_lrfs,           sizeof(float)*points_num*9, cudaMemcpyHostToDevice));

    int grid_num    = (points_num+1024-1)/1024;
    int block_num   = 1024;   
    binning<<<grid_num, block_num>>>(d_points, points_num, d_neighbors, d_lrfs, d_normals, bins_azimuth, bins_elevation, 
                                    feat_dim, k, radius, bins_size, d_feat);
    CUDA_CHECK(cudaMemcpy(feat, d_feat, sizeof(float)*points_num*feat_dim, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_feat));
    CUDA_CHECK(cudaFree(d_lrfs));
    CUDA_CHECK(cudaFree(d_normals));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_points));
}

void SHOT::descriptor(const float* points, const int points_num, float* feat,
                    const float radius, const int bins_azimuth, const int bins_elevation,
                    const int bins_radial, const int bins_size)
{
    int volumes_num     = bins_azimuth*bins_elevation*bins_radial;  // 2*2*8=32
    int feat_dim        = volumes_num*bins_size;                    // 32*11=352

    int* h_neighbor = new int[points_num*(32+1)];
    neighbor->search_bf(points, points_num, 32, h_neighbor);

    float* h_normals = new float[points_num*3];
    mat->normals_estimator(points, points_num, h_neighbor, 32, h_normals);

    float* h_lrfs = new float[points_num*9];
    local_reference_frame(points, points_num, h_neighbor, 32, radius, h_lrfs);

    float* h_feat = new float[points_num*feat_dim];
    histogram(points, points_num, h_neighbor, h_lrfs, h_normals, bins_size, bins_radial, bins_azimuth, bins_elevation, 32, radius, feat_dim, h_feat);
    
    delete[] h_feat;
    delete[] h_lrfs;
    delete[] h_normals;
    delete[] h_neighbor;
}

SHOT::~SHOT()
{}

}