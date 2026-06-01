#include "stat_filter.h"

namespace denoising {

/**
 * @brief CUDA kernel to compute mean distance to neighbors for each point.
 * 
 * @details
 * For each point, this kernel:
 *   1. Loads the point's coordinates
 *   2. Iterates through its k neighbors (excluding self)
 *   3. Filters neighbors by radius threshold r
 *   4. Accumulates Euclidean distances to valid neighbors
 *   5. Computes mean distance = total_distance / k
 * 
 * The computed mean distance represents the local point density - points in
 * sparse regions typically have larger mean distances and may be classified as outliers.
 * 
 * @param points            Input point cloud [N x POINT_DIM]
 * @param points_num        Total number of points
 * @param neighbor_indices  Precomputed neighbor indices [N x (k+1)]
 * @param k                 Number of neighbors per point (excluding self)
 * @param r                 Radius threshold (neighbors beyond r are ignored)
 * @param mean_dists        Output array of mean distances for each point [N]
 */
__global__ void mean_dist_func(const float* points, const int points_num, const int* neighbor_indices, const int k, const int r,
                               float* mean_dists)
{
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= points_num) return;

    float p_s[3] = {points[threadid*POINT_DIM+0], points[threadid*POINT_DIM+1], points[threadid*POINT_DIM+2]};
    float mean_dist = 0.0f;
    
    // Neighbor indexing
    // Layout:
    //   neighbor_indices[p_id * (k+1) + 0]     → self index
    //   neighbor_indices[p_id * (k+1) + i]     → i-th neighbor
    int stride = k + 1;
    int base   = threadid * stride;
    for (int i=1; i<=k; i++)
    {
        int idx                 = neighbor_indices[base + i];
        float p_t[3]            = {points[idx * POINT_DIM + 0], points[idx * POINT_DIM + 1], points[idx * POINT_DIM + 2]};
        float dist              = mat::distf(p_s, p_t);
        if (dist > r) continue; // Skip neighbors outside the specified radius
        mean_dist += dist;
    }
    mean_dist /= k;

    mean_dists[threadid] = mean_dist;
}

/**
 * @brief CUDA kernel to classify and filter outliers based on statistical thresholds.
 * 
 * @details
 * Each point is classified as:
 *   - Inlier: d̄_i ≤ μ + α·σ  (keep original point)
 *   - Outlier: d̄_i > μ + α·σ (mark with flag -2)
 * 
 * The classification is based on the assumption that mean distances follow a
 * Gaussian distribution, where points with abnormally large mean distances
 * (typically noise points in sparse regions) are removed.
 * 
 * @param points        Input point cloud [N x POINT_DIM]
 * @param points_num    Total number of points
 * @param mean_dists    Precomputed mean distances for each point [N]
 * @param mu            Global mean of all mean distances (μ)
 * @param sigma         Global standard deviation of all mean distances (σ)
 * @param alpha         Threshold multiplier for outlier detection
 * @param points_out    Output point cloud [N x POINT_DIM] with outliers flagged
 */
__global__ void state_filter(const float* points, const int points_num, const float* mean_dists,
                             const float mu, const float sigma, const float alpha, float* points_out)
{
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= points_num) return;
    
    // Outlier removal
    float mean_dist = mean_dists[threadid];
    float threshold = mu + alpha * sigma;
    points_out[threadid*POINT_DIM+0] = points[threadid*POINT_DIM+0];
    points_out[threadid*POINT_DIM+1] = points[threadid*POINT_DIM+1];
    points_out[threadid*POINT_DIM+2] = points[threadid*POINT_DIM+2];
    points_out[threadid*POINT_DIM+3] = points[threadid*POINT_DIM+3];
    points_out[threadid*POINT_DIM+4] = points[threadid*POINT_DIM+4];
    if (mean_dist >= threshold)
    {
        
        points_out[threadid*POINT_DIM+5] = -2; // Mark as outlier
    }
    else
    {
        points_out[threadid*POINT_DIM+5] = points[threadid*POINT_DIM+5]; // Keep original classification flag
    }
}

StatFilter::StatFilter(){}

StatFilter::StatFilter(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<denoising_display::Display>("stat_denoising");
    mat          = std::make_unique<mat::MAT>(false);
}

void StatFilter::set_points(const float* points, const int points_num)
{
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*points_num*POINT_DIM));
    CUDA_CHECK(cudaMemcpy(d_points, points,         sizeof(float)*points_num*POINT_DIM, cudaMemcpyHostToDevice));
    this->points_num = points_num;
}

void StatFilter::set_neighbors(const int* neighbors, const int k, const int points_num)
{
    CUDA_CHECK(cudaMalloc((void **)&d_neighbors,    sizeof(int)*points_num*(k+1)));
    CUDA_CHECK(cudaMemcpy(d_neighbors, neighbors,   sizeof(int)*points_num*(k+1), cudaMemcpyHostToDevice));
}

void StatFilter::set_radius(const float r)
{
    this->r = r;
}

void StatFilter::set_k(const int k)
{
    this->k = k;
}

void StatFilter::set_alpha(const float alpha)
{
    this->alpha = alpha;
}

void StatFilter::denoising(float* h_points_out)
{

    CUDA_CHECK(cudaMalloc((void **)&d_mean_dists, sizeof(float)*this->points_num));
    CUDA_CHECK(cudaMalloc((void **)&d_points_out, sizeof(float)*this->points_num*POINT_DIM));

    float* h_mean_dists = new float[this->points_num];

    int grid_num    = (this->points_num+1024-1)/1024;
    int block_num   = 1024;

    // Compute mean distance to neighbors for each point
    mean_dist_func<<<grid_num, block_num>>>(d_points, this->points_num, d_neighbors, this->k, this->r, d_mean_dists);
    CUDA_CHECK(cudaMemcpy(h_mean_dists, d_mean_dists, sizeof(float)*points_num, cudaMemcpyDeviceToHost));
    
    // Gaussian distribution
    // Assuming d̄ follows Gaussian distribution N(μ, σ²)
    // WARNING: computing global μ and σ should be done on GPU for efficiency, but we compute on CPU here for simplicity
    float mu        = 0;    // Global mean
    float sigma     = 0;    // Global standard deviation

    // Compute mean (μ) = (1/N) * Σ d̄_i
    for (size_t i = 0; i < this->points_num; i++)
    {
        mu += h_mean_dists[i];
    }
    mu /= this->points_num;
    
    // Compute standard deviation (σ) = sqrt((1/N) * Σ (d̄_i - μ)²)
    for (size_t i = 0; i < this->points_num; i++)
    {
        sigma += (h_mean_dists[i] - mu) * (h_mean_dists[i] - mu);
    }
    sigma = sqrt(sigma / this->points_num);

    LOGV("mu: %f, sigma: %f", mu, sigma);

    // Classify outliers using threshold μ + α·σ
    state_filter<<<grid_num, block_num>>>(d_points, this->points_num, d_mean_dists, mu, sigma, this->alpha, d_points_out);
    CUDA_CHECK(cudaMemcpy(h_points_out, d_points_out, sizeof(float)*points_num*POINT_DIM, cudaMemcpyDeviceToHost));

    if (this->vis)
    {
        display->set_points(h_points_out, points_num);
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_points_out));
    CUDA_CHECK(cudaFree(d_mean_dists));
}

StatFilter::~StatFilter()
{}

}