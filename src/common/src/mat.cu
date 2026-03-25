#include "mat.h"

namespace mat {

/**
 * @brief Compute normals using PCA.
 *
 * @details
 *  - Runs PCA per point
 *  - Selects eigenvector corresponding to smallest eigenvalue
 *  - This eigenvector approximates surface normal
 *
 * @param points            Input points
 * @param points_num        Number of points
 * @param neighbor_indices  Neighbor indices
 * @param k                 Number of neighbors
 * @param eigenvalues       Output eigenvalues
 * @param eigenvectors      Output eigenvectors
 * @param normals           Output normals [N * 3]
 */
__global__ void normals_pca(const float* points, const int points_num, const int* neighbor_indices, const int k, 
                        float* eigenvalues, float* eigenvectors, float* normals)
{
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= points_num) return;

    pca(points, points_num, neighbor_indices, k, eigenvalues, eigenvectors);

    // Find min eigenvalues
    int min_id      = 0;
    float min_val   = eigenvalues[threadid * 3 + 0];
    for (int i = 1; i < 3; i++)
    {
        float v = eigenvalues[threadid * 3 + i];
        if (v < min_val)
        {
            min_val     = v;
            min_id      = i;
        }
    }

    normals[threadid * 3 + 0] = eigenvectors[threadid * 9 + min_id*3 + 0];
    normals[threadid * 3 + 1] = eigenvectors[threadid * 9 + min_id*3 + 1];
    normals[threadid * 3 + 2] = eigenvectors[threadid * 9 + min_id*3 + 2];
}

MAT::MAT()
{}

MAT::MAT(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("mat");
}

void MAT::normals_estimator(const float* h_points, const int points_num, const int* h_neighbors, const int k, float* normals)
{
     // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*points_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_neighbors,    sizeof(int)*points_num*(k+1)));
    CUDA_CHECK(cudaMalloc((void **)&d_eigenvalues,  sizeof(float)*points_num*3));
    CUDA_CHECK(cudaMalloc((void **)&d_eigenvectors, sizeof(float)*points_num*9));
    CUDA_CHECK(cudaMalloc((void **)&d_normals,      sizeof(float)*points_num*3));
    
    // Copy pointcloud/neighbors to GPU
    CUDA_CHECK(cudaMemcpy(d_points, h_points, sizeof(float)*points_num*POINT_DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, h_neighbors, sizeof(int)*points_num*(k+1), cudaMemcpyHostToDevice));

    int grid_num    = (points_num+1024-1)/1024;
    int block_num   = 1024;
    normals_pca<<<grid_num, block_num>>>(d_points, points_num, d_neighbors, k, d_eigenvalues, d_eigenvectors, d_normals);
    CUDA_CHECK(cudaMemcpy(normals, d_normals, sizeof(float)*points_num*3, cudaMemcpyDeviceToHost));

    if (this->vis)
    {
        float* h_normals = new float[points_num*3];
        CUDA_CHECK(cudaMemcpy(h_normals, d_normals, sizeof(int)*points_num*3, cudaMemcpyDeviceToHost));
        for (int i=0; i< 2*k; i++)
        {
            LOGV("[%d], h_normals: (%f, %f, %f)\n", i, h_normals[i*3+0], h_normals[i*3+1], h_normals[i*3+2]);
        }
        display->set_normals(h_normals, points_num);
        display->set_pointcloud_xyz(h_points, points_num);
        delete[] h_normals;
    }

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_normals));
    CUDA_CHECK(cudaFree(d_eigenvectors));
    CUDA_CHECK(cudaFree(d_eigenvalues));
    CUDA_CHECK(cudaFree(d_points));
}

MAT::~MAT()
{}

}