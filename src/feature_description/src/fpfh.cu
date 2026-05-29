#include "fpfh.h"

namespace feat_desc{

FPFH::FPFH(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("FPFH");
    mat         = std::make_unique<mat::MAT>(false);
    neighbor    = std::make_unique<neighbor::Neighbor>(false);
}

/**
 * @brief Compute Simplified Point Feature Histogram (SPFH) for a single point.
 * 
 * @details
 * SPFH captures local geometry by analyzing relationships between a point and its neighbors.
 * For each neighbor within radius r, computes three angular features:
 * 
 * Feature computation steps:
 *   1. For point p with normal u, compute vector to neighbor q: d = (p-q)/||p-q||
 *   2. Define local coordinate frame:
 *      u = normal at p
 *      v = d × u
 *      w = u × v
 *   3. Compute three angles using neighbor's normal n_q:
 *      α = v·n_q
 *      θ = atan2(w·n_q, u·n_q)
 *      ρ = u·d
 *   4. Quantize each angle into BIN_SIZE (11) bins:
 *      α ∈ [-1,1]  → bin = floor((α+1)/2 * BIN_SIZE)
 *      ρ ∈ [-1,1]  → bin = floor((ρ+1)/2 * BIN_SIZE)  
 *      θ ∈ [-π,π]  → bin = floor((θ+π)/(2π) * BIN_SIZE)
 *   5. Increment corresponding histogram bins
 * 
 * @param points            Input point cloud array [N x POINT_DIM]
 * @param points_num        Total number of points (for bounds checking)
 * @param p_id              Index of current keypoint
 * @param neighbor_indices  Neighbor index array [N x (k+1)], first entry is self
 * @param normals           Surface normals array [N x 3]
 * @param k                 Number of neighbors (excluding self)
 * @param r                 Radius threshold for neighbor filtering
 * @param local_hist        Output SPFH histogram [HIST_DIM] (33 bins)
 */
__device__ inline void spfh(const float* points, const int points_num, const int p_id, 
                            const int* neighbor_indices, const float* normals, 
                            const int k, const float r, int* local_hist)
{
    // Load keypoint coordinates and normal
    float point[3]  = {points[p_id*POINT_DIM+0], points[p_id*POINT_DIM+1], points[p_id*POINT_DIM+2]};
    float u[3]      = {normals[p_id*3+0], normals[p_id*3+1], normals[p_id*3+2]};
    
    // Initialize histogram bins to zero
    for (size_t i = 0; i < HIST_DIM; i++)
    {
        local_hist[i] = 0;
    }

    // Neighbor indexing
    // Layout:
    //   neighbor_indices[p_id * (k+1) + 0]     → self index
    //   neighbor_indices[p_id * (k+1) + i]     → i-th neighbor
    int stride          = k + 1;
    int base            = p_id * stride;    // Current point index in neighbor_indices
    int valid_neighbors = 0;                // Valid neighbors

    // Iterate through all k neighbors
    for (int i = 1; i <= k; i++)
    {
        int idx                 = neighbor_indices[base + i];
        float neighbor_point[3] = {points[idx*POINT_DIM+0], points[idx*POINT_DIM+1], points[idx*POINT_DIM+2]};
        float dist              = mat::distf(point, neighbor_point);
        
        // Skip neighbors outside the specified radius
        if (dist > r) continue;
        
        // Compute normalized direction vector from neighbor to keypoint: d = (p - q)/||p-q||
        float dx            = (point[0] - neighbor_point[0])/dist;
        float dy            = (point[1] - neighbor_point[1])/dist;
        float dz            = (point[2] - neighbor_point[2])/dist;
        float d_p[3]        = {dx, dy, dz};
        float normal_k[3]   = {normals[idx*3+0], normals[idx*3+1], normals[idx*3+2]};

        // Build local coordinate frame
        float v[3]          = {0.0f};
        float w[3]          = {0.0f};
        mat::cross_product3(d_p, u, v);
        mat::cross_product3(u, v, w);

        // Compute three angular features
        float alpha     = mat::dot3f(v, normal_k);                                  // α: angle between v and neighbor's normal
        float rho       = mat::dot3f(u, d_p)/dist;                                  // ρ: projection of d onto u
        float theta     = atan2(mat::dot3f(w, normal_k), mat::dot3f(u, normal_k));  // θ: angle in the plane perpendicular to u

        // Map features to histogram bins (0 to BIN_SIZE-1)
        int alpha_bin   = min(BIN_SIZE - 1, max(0, static_cast<int>((alpha + 1.0f) * (BIN_SIZE / 2.0f))));          // α range: [-1, 1] → map to [0, BIN_SIZE-1]
        int rho_bin     = min(BIN_SIZE - 1, max(0, static_cast<int>((rho + 1.0f) * (BIN_SIZE / 2.0f))));            // ρ range: [-1, 1] → map to [0, BIN_SIZE-1]
        int theta_bin   = min(BIN_SIZE - 1, max(0, static_cast<int>((theta + M_PI) * (BIN_SIZE / (2.0f * M_PI))))); // θ range: [-π, π] → map to [0, BIN_SIZE-1]
        
        // Update histogram bins
        // Layout: bins 0-10: α, bins 11-21: θ, bins 22-32: ρ
        local_hist[alpha_bin]               += 1;   // α feature (first 11 bins)
        local_hist[2*BIN_SIZE + rho_bin]    += 1;   // θ feature (next 11 bins)
        local_hist[BIN_SIZE + theta_bin]    += 1;   // ρ feature (last 11 bins)
        // valid_neighbors++;
    }

}

/**
 * @brief CUDA kernel to compute FPFH descriptors for all points in parallel.
 * 
 * @details
 * For each point (one thread per point):
 *   1. Compute SPFH for the keypoint itself
 *   2. For each neighbor within radius r, compute its SPFH
 *   3. Accumulate weighted SPFH contributions from neighbors
 *   4. Combine keypoint SPFH with weighted neighbor SPFH to get FPFH
 * 
 * Weight calculation: weight = 1/dist (inverse distance weighting)
 * Final FPFH = SPFH(keypoint) + (1/k) * Σ(weight * SPFH(neighbor))
 * 
 * @param points            Input point cloud [N x POINT_DIM]
 * @param points_num        Total number of points
 * @param neighbor_indices  Precomputed neighbor indices [N x (k+1)]
 * @param normals           Surface normals [N x 3]
 * @param k                 Number of neighbors per point
 * @param r                 Radius threshold for neighbor inclusion
 * @param fpfh_features     Output FPFH descriptors [N x HIST_DIM]
 */
__global__ void fpfh(const float* points, const int points_num, const int* neighbor_indices, const float* normals, 
                    const int k, const float r, float* fpfh_features)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid >= points_num) return;

    // ONLY keypoints (points[threadid*POINT_DIM+5] == 0) are processed for FPFH computation
    if (points[threadid*POINT_DIM+5]==0)     
    {
        // Load keypoint coordinates p_k
        float point[3] = {points[threadid*POINT_DIM+0], points[threadid*POINT_DIM+1], points[threadid*POINT_DIM+2]};

        // Compute SPFH for the keypoint itself
        int keypoint_spfh[HIST_DIM] = {0};
        spfh(points, points_num, threadid, neighbor_indices, normals, k, r, keypoint_spfh);

        // Neighbor indexing
        // Layout:
        //   neighbor_indices[p_id * (k+1) + 0]     → self index
        //   neighbor_indices[p_id * (k+1) + i]     → i-th neighbor
        int stride = k + 1;
        int base   = threadid * stride; // Current point index in neighbor_indices

        // Accumulate weighted SPFH from all neighbors within radius
        volatile float neighbors_spfh[HIST_DIM]      = {0.0f};  // Accumulator for neighbor features
        float total_weight                           = 0.0f;    // Sum of all weights (for normalization
        float test = 0.0f;
        for (int i = 1; i <= k; i++)
        {
            int idx                 = neighbor_indices[base + i];
            float neighbor_point[3] = {points[idx*POINT_DIM+0], points[idx*POINT_DIM+1], points[idx*POINT_DIM+2]};
            float dist              = mat::distf(point, neighbor_point);

            // Only include neighbors within radius r
            if (dist > r) continue;

            // Compute SPFH for this neighbor
            int neighbor_hist[HIST_DIM] = {0};
            spfh(points, points_num, idx, neighbor_indices, normals, k, r, neighbor_hist);

            // Weight = inverse distance (closer neighbors have higher influence)
            float weight = 1/dist;

            // Accumulate weighted SPFH
            for (size_t j = 0; j < HIST_DIM; j++)
            {
                neighbors_spfh[j] += weight*(float)neighbor_hist[j];
            }
            total_weight += weight;
        }

        // Compute final FPFH = SPFH(keypoint) + (1/k) * Σ(weighted neighbors' SPFH)
         // Note: Using k (number of neighbors) for normalization, not total_weight
        for (int j = 0; j < HIST_DIM; j++)
        {
            fpfh_features[threadid * HIST_DIM + j] = (float)keypoint_spfh[j] + (neighbors_spfh[j] / k);
        }
    }
}

void FPFH::descriptor(const float* points, const int points_num, const int* h_neighbors, const float* h_normals,
                     const int k, const float r, float* h_histogram)
{
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*points_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_neighbors,    sizeof(int)*points_num*(k+1)));
    CUDA_CHECK(cudaMalloc((void **)&d_normals,      sizeof(float)*points_num*3));
    CUDA_CHECK(cudaMalloc((void **)&d_histogram,    sizeof(float)*points_num*HIST_DIM));

    CUDA_CHECK(cudaMemcpy(d_points, points,         sizeof(float)*points_num*POINT_DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, h_neighbors, sizeof(int)*points_num*(k+1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normals, h_normals,     sizeof(float)*points_num*3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0,           sizeof(float)*points_num*HIST_DIM));
    

    int grid_num    = (points_num+1024-1)/1024;
    int block_num   = 1024;
    fpfh<<<grid_num, block_num>>>(d_points, points_num, d_neighbors, d_normals, k, r, d_histogram);
    CUDA_CHECK(cudaThreadSynchronize());
    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, sizeof(float)*points_num*HIST_DIM, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_histogram));
    CUDA_CHECK(cudaFree(d_normals));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_points));
}

FPFH::~FPFH()
{}

}