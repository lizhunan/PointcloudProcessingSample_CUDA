#include "iss.h"

namespace iss {

/**
 * @brief Determine if a point is a keypoint based on eigenvalue ratios
 * 
 * @details
 *   This function evaluates whether a point's local neighborhood exhibits
 *   distinctive geometric properties suitable for keypoint detection.
 * 
 *   The decision is based on the ratios of eigenvalues from PCA:
 *     - λ₁ ≥ λ₂ ≥ λ₃ (principal components sorted descending)
 *     - λ₂/λ₁ indicates "linearity" (how much variance is in one direction)
 *     - λ₃/λ₂ indicates "planarity" (how much variance is in two directions)
 * 
 *   Keypoint criteria:
 *     - Both eigenvalues must be positive (non-zero to avoid division issues)
 *     - λ₂/λ₁ < lambda21: point lies on a surface/edge (not isotropic)
 *     - λ₃/λ₂ < lambda32: point has dominant planar or linear structure
 * 
 * 
 * @param eigenvalues  Array of 3 eigenvalues (may be unsorted on input)
 * @param lambda21     Threshold for λ₂/λ₁ ratio (controls linearity)
 * @param lambda32     Threshold for λ₃/λ₂ ratio (controls planarity)
 * 
 * @return true   Point is a keypoint candidate
 * @return false  Point does not meet keypoint criteria
 * 
 * @note The function sorts eigenvalues in descending order internally
 *       to ensure consistent behavior regardless of input ordering
 */
__device__ bool is_keypoint(float* eigenvalues, const float lambda21, const float lambda32)
{
    // Extract eigenvalues from input array
    // Note: Input eigenvalues may not be sorted
    float lambda1 = eigenvalues[0];
    float lambda2 = eigenvalues[1];
    float lambda3 = eigenvalues[2];

    // Sort eigenvalues in descending order (λ₁ ≥ λ₂ ≥ λ₃)
    float temp;
    if (lambda1 < lambda2)
    {
        temp = lambda1;
        lambda1 = lambda2;
        lambda2 = temp;
    }
    if (lambda1 < lambda3)
    {
        temp = lambda1;
        lambda1 = lambda3;
        lambda3 = temp;
    }
    if (lambda2 < lambda3)
    {
        temp = lambda2;
        lambda2 = lambda3;
        lambda3 = temp;
    }

    // Avoid division by zero
    if (lambda1 == 0 || lambda2 == 0) {
        return false;
    }

    // Ensures eigenvalues are strictly decreasing (distinct eigenvalues)
    if (lambda1 > lambda2 && lambda2 > lambda3)
    {
        if ((lambda2/lambda1)<lambda21 && (lambda3/lambda2)<lambda32)
        {
            return true;    // keypoint
        }
    }
    return false;           // not keypoint
}

/**
 * @brief Non-Maximum Suppression (NMS) for keypoint detection
 * 
 * @details
 *   This kernel performs non-maximum suppression on detected keypoints based on
 *   their lambda3 values (e.g., eigenvalues from covariance analysis). Points with
 *   lower lambda3 values within a suppression radius are suppressed (marked as -1).
 * 
 *   Mathematical formulation:
 *     For each keypoint p_k, compare its lambda3 value with all neighbors within
 *     radius nms_r. If any neighbor has a higher lambda3 value, p_k is suppressed.
 * 
 * 
 * @param points            Pointer to point cloud array [N x POINT_DIM]
 *                          Modified in-place: points[threadid*POINT_DIM+5] set to -1 if suppressed
 * @param point_num         Total number of points in the cloud
 * @param neighbor_indices  Neighbor index array [N x (k+1)]
 *                          Layout: [self_index, neighbor1, neighbor2, ..., neighbork]
 * @param k                 Number of neighbors per point (excluding self)
 * @param nms_r             Suppression radius threshold
 * @param lambda3           Array of lambda3 values for each point (e.g., smallest eigenvalue)
 */
__device__ inline void nms_lambda(float *points, const int point_num, const int p_id, const int* neighbor_indices, 
                                const int k, const float nms_r, const float* lambda3)
{
    // Load keypoint coordinates p_k and lambda
    float point[3]      = {points[p_id*POINT_DIM+0], points[p_id*POINT_DIM+1], points[p_id*POINT_DIM+2]};    
    float point_lambda3 = lambda3[p_id];                                                

    // Neighbor indexing
    // Layout:
    //   neighbor_indices[p_id * (k+1) + 0]     → self index
    //   neighbor_indices[p_id * (k+1) + i]     → i-th neighbor
    int stride = k + 1;
    int base   = p_id * stride; // Current point index in neighbor_indices

    /**
     * Iterate through all neighbors to check for suppression condition
     * 
     * Algorithm:
     *   1. For each neighbor within radius nms_r
     *   2. Compare lambda3 values
     *   3. If neighbor has higher lambda3, suppress current point
     */
    for (int i=0; i<k; i++)
    {
        int idx                 = neighbor_indices[base + i];
        float neighbor_point[3] = {points[idx * POINT_DIM + 0], points[idx * POINT_DIM + 1], points[idx * POINT_DIM + 2]};
        float neighbor_lambda3  = lambda3[idx];
        float dist              = mat::distf(point, neighbor_point);
        
        /**
         * Skip neighbors outside the suppression radius
         * ONLY consider neighbors within nms_r for suppression
         */
        if (dist > nms_r) continue;
        
        /**
         * Suppression condition:
         * If current point's lambda3 is SMALLER than neighbor's lambda3,
         * then current point is NOT a local maximum and should be suppressed.
         * 
         * In keypoint detection, we typically want points with LARGER lambda3
         * (e.g., more distinctive or salient points)
         */
        if (point_lambda3 < neighbor_lambda3)
        {
            /**
             * Mark current point as suppressed
             * 
             * points[threadid*POINT_DIM+5] is typically used as a flag:
             *   -1 : suppressed/invalid point
             *    0 : valid point
             * 
             * Note: This modifies the pointcloud in-place
             */
            points[p_id*POINT_DIM+5] = -1;
        }
    }
}

/**
 * @brief Keypoint detection kernel using covariance analysis and NMS
 * 
 * @details
 *   This kernel performs keypoint detection by:
 *     1. Computing covariance matrix for each point's local neighborhood
 *     2. Performing PCA via Jacobi method to extract eigenvalues
 *     3. Applying keypoint criteria based on eigenvalue ratios
 *     4. Running non-maximum suppression to filter final keypoints
 * 
 *   The detection process identifies points with distinctive geometric properties,
 *   such as corners, edges, or high surface variation.
 * 
 * @param points        Point cloud array [N x POINT_DIM] (modified in-place)
 *                      - points[threadid*POINT_DIM+5] set to 0 if keypoint candidate
 * @param points_num    Total number of points in the cloud
 * @param lambda3       Output array for smallest eigenvalues (surface variation indicator)
 * @param neighbors     Neighbor index array [N x (k+1)]
 *                      Layout: [self_index, neighbor1, ..., neighbork]
 * @param k             Number of neighbors per point (excluding self)
 * @param lambda21      Threshold ratio for λ₂/λ₁ (second to first eigenvalue)
 * @param lambda32      Threshold ratio for λ₃/λ₂ (third to second eigenvalue)
 * 
 * Geometric interpretation of eigenvalue ratios:
 *   - λ₁ ≥ λ₂ ≥ λ₃ (descending order)
 *   - λ₃/λ₂ close to 0: planar surface (point lies on plane)
 *   - λ₃/λ₂ close to 1: isotropic/volumetric (point in scattered region)
 *   - λ₂/λ₁ determines linearity vs planarity
 */
__global__ void detect_keypoints(float *points, const int points_num, float* lambda3, const int* neighbors, 
                                const int k, const float nms_r, const float lambda21, const float lambda32)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid >= points_num) return;

    float M[3][3]       = {0};  // Zero-initialized covariance matrix
    float eigval[3]     = {0};  // Eigenvalues (will be sorted in is_keypoint)
    float eigvec[3][3]  = {0};  // Eigenvectors

    mat::cov_mat(points, threadid, points_num, neighbors, k, M);    // Compute covariance matrix for local neighborhood
    mat::jacobi_3x3(M, eigval, eigvec);                             // Compute eigenvalues and eigenvectors using Jacobi iteration

    float trace         = M[0][0] + M[1][1] + M[2][2];          // Trace and sum for other keypoint criteria(unused)
    float sum           = eigval[0] + eigval[1] + eigval[2];    // Trace and sum for other keypoint criteria(unused)
    lambda3[threadid]   = eigval[2];                            // Store lambda3 for NMS

    if(is_keypoint(eigval, lambda21, lambda32)) points[threadid*POINT_DIM+5] = 0;       // Apply keypoint criteria based on eigenvalue ratios
    nms_lambda(points, points_num, threadid, neighbors, k, nms_r, lambda3);             // Apply NMS
}

ISS::ISS(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("ISS");
}

void ISS::detector(const float* points, const int points_num, const int* h_neighbors, const float nms_r, const int k, float* output)
{   
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*points_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_neighbors,    sizeof(int)*points_num*(k+1)));
    CUDA_CHECK(cudaMalloc((void **)&d_lambda3,      sizeof(float)*points_num));

    CUDA_CHECK(cudaMemcpy(d_points, points,         sizeof(float)*points_num*POINT_DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, h_neighbors, sizeof(int)*points_num*(k+1), cudaMemcpyHostToDevice));

    int grid_num    = (points_num+1024-1)/1024;
    int block_num   = 1024;
    detect_keypoints<<<grid_num, block_num>>>(d_points, points_num, d_lambda3, d_neighbors, k, nms_r, 0.35, 0.75);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, d_points, sizeof(float)*points_num*POINT_DIM, cudaMemcpyDeviceToHost));

    if (this->vis)
    {
        float* h_points = new float[points_num*POINT_DIM];
        CUDA_CHECK(cudaMemcpy(h_points, d_points, sizeof(float)*points_num*POINT_DIM, cudaMemcpyDeviceToHost));
        display->set_pointcloud_xyz(h_points, points_num);
        delete[] h_points;
    }

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_lambda3));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_points));
}

ISS::~ISS()
{}

}