#include "fpfh.h"

namespace feat_desc{

FPFH::FPFH(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("FPFH");
    mat         = std::make_unique<mat::MAT>(false);
    neighbor    = std::make_unique<neighbor::Neighbor>(false);
}

__device__ inline void spfh(const float* points, const int points_num, const int p_id, 
                            const int* neighbor_indices, const float* normals, 
                            const int k, const float r, int* local_hist)
{
    // printf("spfh id: [%d]\n", p_id);
    float point[3]  = {points[p_id*POINT_DIM+0], points[p_id*POINT_DIM+1], points[p_id*POINT_DIM+2]};
    float u[3]      = {normals[p_id*3+0], normals[p_id*3+1], normals[p_id*3+2]};
    
    // 初始化histogram
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

    for (int i = 1; i <= k; i++)
    {
        int idx                 = neighbor_indices[base + i];
        float neighbor_point[3] = {points[idx*POINT_DIM+0], points[idx*POINT_DIM+1], points[idx*POINT_DIM+2]};
        float dist              = mat::distf(point, neighbor_point);
        
        if (dist > r) continue;
        
        float dx            = (point[0] - neighbor_point[0])/dist;
        float dy            = (point[1] - neighbor_point[1])/dist;
        float dz            = (point[2] - neighbor_point[2])/dist;
        float d_p[3]        = {dx, dy, dz};
        float normal_k[3]   = {normals[idx*3+0], normals[idx*3+1], normals[idx*3+2]};

        float v[3]          = {0.0f};
        float w[3]          = {0.0f};
        mat::cross_product3(d_p, u, v);
        mat::cross_product3(u, v, w);

        float alpha     = mat::dot3f(v, normal_k);
        float rho       = mat::dot3f(u, d_p)/dist;
        float theta     = atan2(mat::dot3f(w, normal_k), mat::dot3f(u, normal_k));

        // 将特征值映射到直方图bin
        int alpha_bin   = min(BIN_SIZE - 1, max(0, static_cast<int>((alpha + 1.0f) * (BIN_SIZE / 2.0f))));          // alpha 范围: [-1, 1]
        int rho_bin     = min(BIN_SIZE - 1, max(0, static_cast<int>((rho + 1.0f) * (BIN_SIZE / 2.0f))));            // rho 范围: [-1, 1]（实际上rho = u·d_p，范围在[-1,1]）
        int theta_bin   = min(BIN_SIZE - 1, max(0, static_cast<int>((theta + M_PI) * (BIN_SIZE / (2.0f * M_PI))))); // theta 范围: [-π, π]
        
        // 累计直方图
        local_hist[alpha_bin]               += 1;
        local_hist[2*BIN_SIZE + rho_bin]    += 1;
        local_hist[BIN_SIZE + theta_bin]    += 1;
        // valid_neighbors++;
    }

}

__global__ void fpfh(const float* points, const int points_num, const int* neighbor_indices, const float* normals, 
                    const int k, const float r, float* fpfh_features)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    // if (threadid!=20) return;
    if (threadid >= points_num) return;

    // Load keypoint coordinates p_k
    float point[3] = {points[threadid*POINT_DIM+0], points[threadid*POINT_DIM+1], points[threadid*POINT_DIM+2]};

    // Keypoint SPFH
    int keypoint_spfh[HIST_DIM] = {0};
    spfh(points, points_num, threadid, neighbor_indices, normals, k, r, keypoint_spfh);

    // Neighbor indexing
    // Layout:
    //   neighbor_indices[p_id * (k+1) + 0]     → self index
    //   neighbor_indices[p_id * (k+1) + i]     → i-th neighbor
    int stride = k + 1;
    int base   = threadid * stride; // Current point index in neighbor_indices

    // Neighbors SPFH
    volatile float neighbors_spfh[HIST_DIM]      = {0.0f};
    float total_weight                  = 0.0f;
    float test = 0.0f;
    for (int i = 1; i <= k; i++)
    {
        int idx                 = neighbor_indices[base + i];
        float neighbor_point[3] = {points[idx*POINT_DIM+0], points[idx*POINT_DIM+1], points[idx*POINT_DIM+2]};
        float dist              = mat::distf(point, neighbor_point);

        if (dist > r) continue;

        int neighbor_hist[HIST_DIM] = {0};
        spfh(points, points_num, idx, neighbor_indices, normals, k, r, neighbor_hist);

        float weight    = 1/dist;
        for (size_t j = 0; j < HIST_DIM; j++)
        {
            neighbors_spfh[j] += weight*(float)neighbor_hist[j];
        }
        total_weight    += weight;
    }

    volatile float fpfh[HIST_DIM] = {0.0f}; 
    float t = 0.0f;
    for (int j = 0; j < HIST_DIM; j++)
    {
        fpfh_features[threadid * HIST_DIM + j] = (float)keypoint_spfh[j] + (neighbors_spfh[j] / k);
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