#include "nss_sampling.h"

namespace downsampling
{

__global__ void normal_space(const float* normals, const int normal_num, const int n_theta, const int n_phi, 
                             float* normal_space, int* bucket_ids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= normal_num) return;

    float nx = normals[idx*3 + 0];
    float ny = normals[idx*3 + 1];
    float nz = normals[idx*3 + 2];

    // Normalize normal vectors
    float len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (len > 1e-6) 
    {
        nx /= len;
        ny /= len;
        nz /= len;
    }

    float theta = acosf(nz);
    float phi   = atan2f(ny, nx);
    if (phi < 0) 
    {
        phi += 2.0f * M_PI;
    }

    int theta_idx   = min(int(theta / (M_PI / n_theta)), n_theta - 1);
    int phi_idx     = min(int(phi / (2.0f * M_PI / n_phi)), n_phi - 1);
    int bucket_idx  = theta_idx * n_phi + phi_idx;

    normal_space[idx*2 + 0] = theta;
    normal_space[idx*2 + 1] = phi;
    bucket_ids[idx]         = bucket_idx;
}

__global__ void compute_bucket_id(const float* normal_space, const int normal_num, 
                                  const int n_theta, const int n_phi, int* bucket_ids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= normal_num) return;

    float theta = normal_space[idx*2 + 0];
    float phi   = normal_space[idx*2 + 1];

    if (phi < 0) phi += 2.0f * M_PI;
    
    int theta_idx   = min(int(theta / (M_PI / n_theta)), n_theta - 1);
    int phi_idx     = min(int(phi / (2.0f * M_PI / n_phi)), n_phi - 1);
    int bucket_idx  = theta_idx * n_phi + phi_idx;
    
    bucket_ids[idx] = bucket_idx;
}

__global__ void count_bucket(const int* bucket_ids, const int normal_num, int* bucket_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= normal_num) return;
    
    int bid = bucket_ids[idx];
    atomicAdd(&bucket_counts[bid], 1);
}

__global__ void fill_sorted_indices(const int* bucket_ids, const int* bucket_offsets,
                                    int* bucket_pos_counter, int* sorted_indices,
                                    int normals_num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= normals_num) return;
    
    int bid = bucket_ids[idx];
    int pos = atomicAdd(&bucket_pos_counter[bid], 1);
    
    sorted_indices[bucket_offsets[bid] + pos] = idx;
}

__global__ void sampling(const float* points, const int* bucket_offsets,
                         const int* sorted_indices, const int* bucket_counts,
                         const int* samples_per_bucket,
                         float* points_out, int* out_counter,
                         int total_buckets, unsigned int seed)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= total_buckets) return;
    
    int bucket_size     = bucket_counts[bid];
    int samples_needed  = samples_per_bucket[bid];
    
    if (bucket_size == 0 || samples_needed == 0) return;
    
    curandState state;
    curand_init(seed + bid, 0, 0, &state);

    if (samples_needed >= bucket_size) {
        for (int i = 0; i < bucket_size; i++) {
            int out_pos = atomicAdd(out_counter, 1);
            int src_idx = sorted_indices[bucket_offsets[bid] + i];
            
            points_out[out_pos * POINT_DIM + 0] = points[src_idx * POINT_DIM + 0];
            points_out[out_pos * POINT_DIM + 1] = points[src_idx * POINT_DIM + 1];
            points_out[out_pos * POINT_DIM + 2] = points[src_idx * POINT_DIM + 2];
            points_out[out_pos * POINT_DIM + 3] = points[src_idx * POINT_DIM + 3];
            points_out[out_pos * POINT_DIM + 4] = points[src_idx * POINT_DIM + 4];
            points_out[out_pos * POINT_DIM + 5] = points[src_idx * POINT_DIM + 5];
        }
        return;
    }
    
    bool* selected = new bool[bucket_size];
    for (int i = 0; i < bucket_size; i++) selected[i] = false;
    
    int sampled = 0;
    while (sampled < samples_needed) {
        int idx = curand(&state) % bucket_size;
        if (!selected[idx]) {
            selected[idx] = true;
            sampled++;
            
            int out_pos = atomicAdd(out_counter, 1);
            int src_idx = sorted_indices[bucket_offsets[bid] + idx];
            
            points_out[out_pos * POINT_DIM + 0] = points[src_idx * POINT_DIM + 0];
            points_out[out_pos * POINT_DIM + 1] = points[src_idx * POINT_DIM + 1];
            points_out[out_pos * POINT_DIM + 2] = points[src_idx * POINT_DIM + 2];
            points_out[out_pos * POINT_DIM + 3] = points[src_idx * POINT_DIM + 3];
            points_out[out_pos * POINT_DIM + 4] = points[src_idx * POINT_DIM + 4];
            points_out[out_pos * POINT_DIM + 5] = points[src_idx * POINT_DIM + 5];
        }
    }
    
    delete[] selected;
}

NSS::NSS(){}

NSS::NSS(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<downsampling::Display>("NSS Downsampling");
    mat          = std::make_unique<mat::MAT>(false);
}

void NSS::set_points(const float* points, const int points_num)
{
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*points_num*POINT_DIM));
    CUDA_CHECK(cudaMemcpy(d_points, points,         sizeof(float)*points_num*POINT_DIM, cudaMemcpyHostToDevice));
    this->points_num = points_num;
}

void NSS::set_normals(const float* normals, const int normals_num)
{
    CUDA_CHECK(cudaMalloc((void **)&d_normals,  sizeof(float)*normals_num*3));
    CUDA_CHECK(cudaMemcpy(d_normals, normals,   sizeof(float)*normals_num*3, cudaMemcpyHostToDevice));
    this->normals_num = normals_num;
}

void NSS::downsampling(float* h_points_out)
{
    int target_samples  = int(this->points_num * this->sampling_ratio);
    int n_theta         = int(sqrtf(float(target_samples) / 2.0f));  // 2 is an empirical factor to control bucket size
    int n_phi           = int(n_theta * 2);                          // phi has twice the resolution of theta
    int total_buckets   = n_theta * n_phi;
    int total_nonempty  = 0;

    LOGV("Sampling ratio=%f, target samples=%d", this->sampling_ratio, target_samples);
    LOGV("Theta buckets num=%d, phi buckets num=%d, total buckets num=%d", n_theta, n_phi, total_buckets);

    CUDA_CHECK(cudaMalloc((void **)&d_normal_space,     sizeof(float)*this->normals_num*2));
    CUDA_CHECK(cudaMalloc((void **)&d_bucket_counts,    sizeof(int)*total_buckets));
    CUDA_CHECK(cudaMalloc((void **)&d_bucket_ids,       sizeof(int)*normals_num));
    CUDA_CHECK(cudaMalloc((void **)&d_points_out,       sizeof(float)*this->points_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_samples_per_bucket,   sizeof(int)*total_buckets));
    CUDA_CHECK(cudaMalloc((void **)&d_bucket_offsets,       sizeof(int) * (total_buckets + 1)));
    CUDA_CHECK(cudaMalloc((void **)&d_bucket_pos_counter,   sizeof(int) * total_buckets));
    CUDA_CHECK(cudaMalloc((void **)&d_sorted_indices,       sizeof(int) * this->normals_num));
    CUDA_CHECK(cudaMalloc((void **)&d_out_counter,          sizeof(int)));

    int* h_bucket_counts = new int[total_buckets];
    int* h_bucket_ids    = new int[total_buckets];
    float* h_points      = new float[this->points_num];

    LOGV("Memory alloc finish.");

    int grid_num    = (this->points_num+1024-1)/1024;
    int block_num   = 1024;
    normal_space<<<grid_num, block_num>>>(d_normals, this->normals_num, n_theta, n_phi, d_normal_space, d_bucket_ids);
    CUDA_CHECK(cudaDeviceSynchronize());
    LOGV("Normal space constructed, space size=%d.", this->normals_num*2);

    // compute_bucket_id<<<grid_num, block_num>>>(d_normal_space, this->normals_num, n_theta, n_phi, d_bucket_ids);
    // CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_bucket_counts,0,sizeof(int)*total_buckets));
    count_bucket<<<grid_num, block_num>>>(d_bucket_ids, this->normals_num, d_bucket_counts); 
    CUDA_CHECK(cudaMemcpy(h_bucket_counts, d_bucket_counts, sizeof(int)*total_buckets, cudaMemcpyDeviceToHost));
    LOGV("Bucket map constructedm, map size=%d.", total_buckets);

    for (int i = 0; i < total_buckets; i++)
    {
        if (h_bucket_counts[i] > 0) total_nonempty++;
    }
    LOGV("Non-empty buckets = %d / %d.", total_nonempty, total_buckets);

    // Counting sampling points number per bucket
    int* h_samples_per_bucket   = new int[total_buckets];
    int base_per_bucket         = target_samples / total_nonempty;
    int remaining_samples       = target_samples % total_nonempty;
    for (int i = 0; i < total_buckets; i++)
    {
        if (h_bucket_counts[i] > 0) 
        {
            h_samples_per_bucket[i] = base_per_bucket;
            if (remaining_samples > 0)
            {
                h_samples_per_bucket[i]++;
                remaining_samples--;
            }
            // make sure don't above the number of points for each bucket 
            h_samples_per_bucket[i] = min(h_samples_per_bucket[i], h_bucket_counts[i]);
        } else
        {
            h_samples_per_bucket[i] = 0;
        }
    }

    // Recompute actual sampling number
    int actual_samples = 0;
    for (int i = 0; i < total_buckets; i++)
    {
        actual_samples += h_samples_per_bucket[i];
    }
    LOGV("Actual samples to take = %d (target was %d).", actual_samples, target_samples);

    CUDA_CHECK(cudaMemcpy(d_samples_per_bucket, h_samples_per_bucket, sizeof(int) * total_buckets, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_bucket_offsets, 0,      sizeof(int) * (total_buckets + 1)));

    int* h_bucket_offsets = new int[total_buckets + 1];
    h_bucket_offsets[0] = 0;
    for (int i = 0; i < total_buckets; i++)
    {
        h_bucket_offsets[i + 1] = h_bucket_offsets[i] + h_bucket_counts[i];
    }
    CUDA_CHECK(cudaMemcpy(d_bucket_offsets, h_bucket_offsets, sizeof(int) * (total_buckets + 1), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_bucket_pos_counter, 0, sizeof(int) * total_buckets));
    fill_sorted_indices<<<grid_num, block_num>>>(d_bucket_ids, d_bucket_offsets,
                                                 d_bucket_pos_counter, d_sorted_indices,
                                                 this->normals_num);

    CUDA_CHECK(cudaDeviceSynchronize());
    LOGV("Sorted indices filled.");

    // Sampling
    CUDA_CHECK(cudaMemset(d_out_counter, 0, sizeof(int)));
    int bucket_grid = (total_buckets + block_num - 1) / block_num;
    unsigned int seed = time(NULL);
    sampling<<<bucket_grid, block_num>>>(d_points, d_bucket_offsets,
                                         d_sorted_indices, d_bucket_counts,
                                         d_samples_per_bucket,
                                         d_points_out, d_out_counter,
                                         total_buckets, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    int final_count;
    CUDA_CHECK(cudaMemcpy(&final_count, d_out_counter, sizeof(int), cudaMemcpyDeviceToHost));
    LOGV("Sampling completed, final count = %d", final_count);

    CUDA_CHECK(cudaMemcpy(h_points_out, d_points_out, sizeof(float)*final_count*POINT_DIM, cudaMemcpyDeviceToHost));
    if (this->vis)
    {
        display->set_points(h_points_out, final_count);
    }

    delete[] h_bucket_counts;
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_points_out));
    CUDA_CHECK(cudaFree(d_normal_space));
}

NSS::~NSS()
{}

}