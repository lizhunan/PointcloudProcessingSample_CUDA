#include "neighbor.h"

namespace neighbor {

/**
 * @brief Brute-force KNN search
 *
 * @details
 *  - Each thread processes one query point.
 *  - Computes distance to all other points (O(N^2)).
 *  - Maintains a top-K nearest neighbor list using max-distance replacement strategy.
 *  - Output layout:
 *      [self1, nn1, nn2, ..., self2, nn1, nn2, ..., nnK] for each point
 *
 *  Memory layout:
 *      neighbor_indices[i*(k+1) + 0]     = self index
 *      neighbor_indices[i*(k+1) + j+1]   = j-th neighbor index
 *
 * @param points            Input point cloud array [N * POINT_DIM]
 * @param points_num        Number of points
 * @param k                 Number of nearest neighbors
 * @param neighbor_indices  Output neighbor indices [N * (k+1)]
 */
__global__ void bf_knn(const float* points, const int points_num, const int k, int* neighbor_indices)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid >= points_num) return;

    // Load current query point (x, y, z)
    float point[3] = {points[threadid*POINT_DIM+0], points[threadid*POINT_DIM+1], points[threadid*POINT_DIM+2]};
    
    // Initialize Top-K buffers (max-distance priority)
    float   best_dist[MAX_K];
    int     best_idx[MAX_K];
    for (int i = 0; i < k; i++)
    {
        best_dist[i] = FLT_MAX;
        best_idx[i]  = -1;
    }

    // Traverse all points (brute-force)
    for (int i=0; i<points_num; i++)
    {
        if (i == threadid) continue;

        // Search for Neighbor Points
        float neighbor[3]   = {points[i*POINT_DIM+0], points[i*POINT_DIM+1], points[i*POINT_DIM+2]};
        float dist          = mat::distf(point, neighbor);

        // Find current worst (maximum distance) in Top-K
        int     max_id      = 0;
        float   max_dist    = best_dist[0];
        for (int j = 1; j < k; j++)
        {
            if (best_dist[j] > max_dist)
            {
                max_dist    = best_dist[j];
                max_id      = points[j*POINT_DIM+4];
            }
        }

        // Replace if better neighbor found
        if (dist < max_dist)
        {
            best_dist[max_id] = dist;
            best_idx[max_id]  = points[i*POINT_DIM+4];
        }
    }

    // Write output (flattened structure)
    int base = threadid*(k+1);
    // Self index
    neighbor_indices[base + 0] = points[threadid*POINT_DIM+4];
    // Neighbor indices
    for (int j = 0; j < k; j++)
    {
        neighbor_indices[base + 1 + j] = best_idx[j];
    }
}

Neighbor::Neighbor(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("neighbors");
}

void Neighbor::search_bf(const float* h_points, const int points_num, const int k, int* neighbors)
{
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*points_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_neighbors,    sizeof(int)*points_num*(k+1)));
    
    // Copy pointcloud to GPU
    CUDA_CHECK(cudaMemcpy(d_points, h_points, sizeof(float)*points_num*POINT_DIM, cudaMemcpyHostToDevice));

    int grid_num    = (points_num+1024-1)/1024;
    int block_num   = 1024;
    bf_knn<<<grid_num, block_num>>>(d_points, points_num, k, d_neighbors);
    CUDA_CHECK(cudaMemcpy(neighbors, d_neighbors, sizeof(int)*points_num*(k+1), cudaMemcpyDeviceToHost));

    if (this->vis)
    {
        int* h_neighbors = new int[points_num*(k+1)];
        CUDA_CHECK(cudaMemcpy(h_neighbors, d_neighbors, sizeof(int)*points_num*(k+1), cudaMemcpyDeviceToHost));
        for (int i=0; i< 2*k; i++)
        {
            LOGV("[%d], h_neighbors: %d\n", i, h_neighbors[i]);
        }
        display->set_neighbors(h_neighbors, k, 2);
        display->set_pointcloud_xyz(h_points, points_num);
        delete[] h_neighbors;
    }

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_points));
}

Neighbor::~Neighbor()
{
}

}