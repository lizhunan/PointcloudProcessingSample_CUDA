#include "icp.h"

namespace reg{

__constant__ float d_transform_init[16];

__device__ inline float dist_l2(const float* src_feat, const float* tar_feat, const int dim = 33)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        float diff = src_feat[i] - tar_feat[i];
        sum += diff * diff;
    }
    return sqrtf(sum); 
}

__global__ void transform(const float* points, const int points_num, float* transformed_points)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadid>=points_num) return;

    float x = points[threadid*POINT_DIM+0];
	float y = points[threadid*POINT_DIM+1];
	float z = points[threadid*POINT_DIM+2];

	float new_x=x*d_transform_init[0]+y*d_transform_init[1]+z*d_transform_init[2]+d_transform_init[3];
	float new_y=x*d_transform_init[4]+y*d_transform_init[5]+z*d_transform_init[6]+d_transform_init[7];
	float new_z=x*d_transform_init[8]+y*d_transform_init[9]+z*d_transform_init[10]+d_transform_init[11];
	transformed_points[threadid*POINT_DIM+0]=new_x;
	transformed_points[threadid*POINT_DIM+1]=new_y;
	transformed_points[threadid*POINT_DIM+2]=new_z;
}

__global__ void centered(const float* points, const int points_num, const float* center, float* points_centered)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid>=points_num) return;
    
    float px = points[threadid * POINT_DIM + 0];
    float py = points[threadid * POINT_DIM + 1];
    float pz = points[threadid * POINT_DIM + 2];
    
    float cx = center[0];
    float cy = center[1];
    float cz = center[2];
    
    points_centered[threadid * POINT_DIM + 0] = px - cx;
    points_centered[threadid * POINT_DIM + 1] = py - cy;
    points_centered[threadid * POINT_DIM + 2] = pz - cz;
}

__global__ void loss_func(const float* source, const int source_num, const float* target, const int target_num, 
                          const float max_dist, float* loss, int* valid_correspondences, float* source_corr, float* target_corr)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid >= source_num) return;

    // Load source point
    float p_s[3]                = {source[threadid*POINT_DIM+0], source[threadid*POINT_DIM+1], source[threadid*POINT_DIM+2]};
    int min_idx                 = -1;
    float min_dist              = max_dist;

    for (int i=0; i<target_num; i++)
    {
        float p_t[3]            = {target[i * POINT_DIM + 0], target[i * POINT_DIM + 1], target[i * POINT_DIM + 2]};
        float dist              = mat::distf(p_s, p_t);

        /**
         * Skip neighbors outside the correspondence distance
         */
        if (dist > max_dist) continue;

        // matched
        if (dist < min_dist)
        {
            min_dist = dist;
            min_idx  = i;
        }
    }

    if (min_idx>=0 && min_dist < max_dist)
    {
        int out_idx = atomicAdd(valid_correspondences, 1);
        target_corr[out_idx*POINT_DIM+0] = target[min_idx*POINT_DIM + 0];
        target_corr[out_idx*POINT_DIM+1] = target[min_idx*POINT_DIM + 1];
        target_corr[out_idx*POINT_DIM+2] = target[min_idx*POINT_DIM + 2];
        source_corr[out_idx*POINT_DIM+0] = source[threadid*POINT_DIM + 0];
        source_corr[out_idx*POINT_DIM+1] = source[threadid*POINT_DIM + 1];
        source_corr[out_idx*POINT_DIM+2] = source[threadid*POINT_DIM + 2];
        loss[out_idx] = min_dist;
    }

}

__global__ void loss_func(const float* source, const int source_num, const float* target, const int target_num, const int* neighbor_indices,
                          const float max_dist, const int k, float* loss, int* valid_correspondences, float* source_corr, float* target_corr)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid >= source_num || threadid >= target_num) return;

    // Load source point
    float p_s[3] = {source[threadid*POINT_DIM+0], source[threadid*POINT_DIM+1], source[threadid*POINT_DIM+2]};

    int min_idx     = -1;
    float min_dist  = max_dist;

    // Neighbor indexing
    // Layout:
    //   neighbor_indices[p_id * (k+1) + 0]     → self index
    //   neighbor_indices[p_id * (k+1) + i]     → i-th neighbor
    int stride = k + 1;
    int base   = threadid * stride;
    for (int i=1; i<=k; i++)
    {
        int idx                 = neighbor_indices[base + i];
        float p_t[3]            = {target[idx * POINT_DIM + 0], target[idx * POINT_DIM + 1], target[idx * POINT_DIM + 2]};
        float dist              = mat::distf(p_s, p_t);
        // if (threadid == 5)
        // {
        //     printf("dist: %f\n", dist);
        // }
        
        /**
         * Skip neighbors outside the correspondence distance
         */
        if (dist > max_dist) continue;

        // matched
        if (dist < min_dist)
        {
            min_dist = dist;
            min_idx = idx;
        }

    }
    if (min_idx>=0)
    {
        target_corr[threadid*POINT_DIM+0] = target[min_idx*POINT_DIM + 0];
        target_corr[threadid*POINT_DIM+1] = target[min_idx*POINT_DIM + 1];
        target_corr[threadid*POINT_DIM+2] = target[min_idx*POINT_DIM + 2];
        source_corr[threadid*POINT_DIM+0] = source[threadid*POINT_DIM + 0];
        source_corr[threadid*POINT_DIM+1] = source[threadid*POINT_DIM + 1];
        source_corr[threadid*POINT_DIM+2] = source[threadid*POINT_DIM + 2];
        loss[threadid] = min_dist;
        atomicAdd(valid_correspondences, 1);
    }
}

__global__ void feat_match(const float* src, const float* tar, const int src_num, const int tar_num,
                           const float* src_feat, const float* tar_feat, float* src_out, float* tar_out, int* macthed_num,
                           const int feat_dim=33, const float max_dist=20.0f)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid >= src_num || threadid >= tar_num) return;

    float label = src[threadid*POINT_DIM+5];

    if (label!=0) return;

    const float* src_desc = &src_feat[threadid * feat_dim];
    float best_dist = max_dist;
    int best_tgt    = -1;

    for (int tgt_id = 0; tgt_id < tar_num; tgt_id++)
    {
        const float* tgt_desc = &tar_feat[tgt_id * feat_dim];
        float dist = dist_l2(src_desc, tgt_desc, feat_dim);
        
        if (dist < best_dist)
        {
            best_dist = dist;
            best_tgt  = tgt_id;
        }
    }

    if (best_tgt >= 0) {
        int out_idx = atomicAdd(macthed_num, 1);
        src_out[out_idx*POINT_DIM+0] = src[threadid*POINT_DIM + 0];
        src_out[out_idx*POINT_DIM+1] = src[threadid*POINT_DIM + 1];
        src_out[out_idx*POINT_DIM+2] = src[threadid*POINT_DIM + 2];
        tar_out[out_idx*POINT_DIM+0] = tar[best_tgt*POINT_DIM + 0];
        tar_out[out_idx*POINT_DIM+1] = tar[best_tgt*POINT_DIM + 1];
        tar_out[out_idx*POINT_DIM+2] = tar[best_tgt*POINT_DIM + 2];
    }
}

void ICP::solve_svd(const float* source_centered, const float* target_centered, const float* source_center, const float* target_center,
                   const int num_corrs, float* R, float* t)
{
    // build covariance matrix H = Σ (p_src_centered_i) * (p_tar_centered_i)^T
    Eigen::MatrixXf src_mat(num_corrs, 3);
    Eigen::MatrixXf tar_mat(num_corrs, 3);

    for (int j = 0; j < num_corrs; j++)
    {
        src_mat(j, 0) = source_centered[j * POINT_DIM + 0];
        src_mat(j, 1) = source_centered[j * POINT_DIM + 1];
        src_mat(j, 2) = source_centered[j * POINT_DIM + 2];
        tar_mat(j, 0) = target_centered[j * POINT_DIM + 0];
        tar_mat(j, 1) = target_centered[j * POINT_DIM + 1];
        tar_mat(j, 2) = target_centered[j * POINT_DIM + 2];
    }

    Eigen::Matrix3f H = tar_mat.transpose() * src_mat;

    // SVD decomponent
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    Eigen::Matrix3f R_eigen = U * V.transpose();

    // Handle reflection situations (ensure that the determinant of the rotation matrix is+1)
    if (R_eigen.determinant() < 0)
    {
        V.col(2) = -V.col(2);
        R_eigen = R_eigen = U * V.transpose();
    }

    // solve R and t
    for (int j = 0; j < 3; j++)
    {
        for (int k = 0; k < 3; k++)
        {
            R[j * 3 + k] = R_eigen(j, k);
        }
    }
    t[0] = target_center[0] - (R[0] * source_center[0] + R[1] * source_center[1] + R[2] * source_center[2]);
    t[1] = target_center[1] - (R[3] * source_center[0] + R[4] * source_center[1] + R[5] * source_center[2]);
    t[2] = target_center[2] - (R[6] * source_center[0] + R[7] * source_center[1] + R[8] * source_center[2]);
}

void ICP::match(const float* src, const float* tar, const int src_num, const int tar_num, 
                const float* src_feat, const float* tar_feat, float* src_out, float* tar_out, int& matched_num)
{
    float* d_src_ = nullptr;
    float* d_tar_ = nullptr;
    float* d_src_feat_ = nullptr;
    float* d_tar_feat_ = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src_, src_num * sizeof(float)*POINT_DIM));
    CUDA_CHECK(cudaMalloc(&d_tar_, tar_num * sizeof(float)*POINT_DIM));
    CUDA_CHECK(cudaMalloc(&d_src_feat_, src_num * sizeof(float)*33));
    CUDA_CHECK(cudaMalloc(&d_tar_feat_, tar_num * sizeof(float)*33));
    CUDA_CHECK(cudaMemcpy(d_src_, src,              sizeof(float)*src_num*POINT_DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tar_, tar,              sizeof(float)*tar_num*POINT_DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_feat_, src_feat,    sizeof(float)*src_num*33, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tar_feat_, tar_feat,    sizeof(float)*tar_num*33, cudaMemcpyHostToDevice));
    float* d_src_out = nullptr;
    float* d_tar_out = nullptr;
    int*   d_matched_num;
    CUDA_CHECK(cudaMalloc(&d_src_out, src_num * sizeof(float)*POINT_DIM));
    CUDA_CHECK(cudaMalloc(&d_tar_out, tar_num * sizeof(float)*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_matched_num, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_matched_num, 0, sizeof(int)));
    int grid_num    = (src_num+1024-1)/1024;
    int block_num   = 1024;
    feat_match<<<grid_num, block_num>>>(d_src_, d_tar_, src_num, tar_num, d_src_feat_, d_tar_feat_, d_src_out, d_tar_out, d_matched_num);
    CUDA_CHECK(cudaMemcpy(&matched_num, d_matched_num, sizeof(int), cudaMemcpyDeviceToHost));
    LOGV("Matched pair num: %d", matched_num);
    if (matched_num > 0) {
        CUDA_CHECK(cudaMemcpy(src_out, d_src_out, sizeof(float) * matched_num * POINT_DIM, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tar_out, d_tar_out, sizeof(float) * matched_num * POINT_DIM, cudaMemcpyDeviceToHost));
    }
}

bool ICP::Rt_3_points(const float* src, const float* tar, const int* indices, float* T)
{
    // check if is collinear
    float src_p1[3] = {src[indices[0]*POINT_DIM+0], src[indices[0]*POINT_DIM+1], src[indices[0]*POINT_DIM+2]};
    float src_p2[3] = {src[indices[1]*POINT_DIM+0], src[indices[1]*POINT_DIM+1], src[indices[1]*POINT_DIM+2]};
    float src_p3[3] = {src[indices[2]*POINT_DIM+0], src[indices[2]*POINT_DIM+1], src[indices[2]*POINT_DIM+2]};
    float tar_p1[3] = {tar[indices[0]*POINT_DIM+0], tar[indices[0]*POINT_DIM+1], tar[indices[0]*POINT_DIM+2]};
    float tar_p2[3] = {tar[indices[1]*POINT_DIM+0], tar[indices[1]*POINT_DIM+1], tar[indices[1]*POINT_DIM+2]};
    float tar_p3[3] = {tar[indices[2]*POINT_DIM+0], tar[indices[2]*POINT_DIM+1], tar[indices[2]*POINT_DIM+2]};

    float v1x = src_p2[0] - src_p1[0];
    float v1y = src_p2[1] - src_p1[1];
    float v1z = src_p2[2] - src_p1[2];
    float v2x = src_p3[0] - src_p1[0];
    float v2y = src_p3[1] - src_p1[1];
    float v2z = src_p3[2] - src_p1[2];
    float cx = v1y * v2z - v1z * v2y;
    float cy = v1z * v2x - v1x * v2z;
    float cz = v1x * v2y - v1y * v2x;
    float src_cross_norm = cx*cx + cy*cy + cz*cz;
    
    if (src_cross_norm < 1e-6f) {
        LOGW("RANSAC: Three source points are collinear, retry.");
        return false;
    }

    float w1x = tar_p2[0] - tar_p1[0];
    float w1y = tar_p2[1] - tar_p1[1];
    float w1z = tar_p2[2] - tar_p1[2];
    float w2x = tar_p3[0] - tar_p1[0];
    float w2y = tar_p3[1] - tar_p1[1];
    float w2z = tar_p3[2] - tar_p1[2];
    
    float dx = w1y * w2z - w1z * w2y;
    float dy = w1z * w2x - w1x * w2z;
    float dz = w1x * w2y - w1y * w2x;
    float tar_cross_norm = dx*dx + dy*dy + dz*dz;
    
    if (tar_cross_norm < 1e-6f) {
        LOGW("RANSAC: Three target points are collinear, retry.");
        return false;
    }

    // center points
    float src_center[3] = {0.0f, 0.0f, 0.0f};
    float tar_center[3] = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 3; i++) {
        src_center[0] += src[indices[i]*POINT_DIM + 0];
        src_center[1] += src[indices[i]*POINT_DIM + 1];
        src_center[2] += src[indices[i]*POINT_DIM + 2];
        
        tar_center[0] += tar[indices[i]*POINT_DIM + 0];
        tar_center[1] += tar[indices[i]*POINT_DIM + 1];
        tar_center[2] += tar[indices[i]*POINT_DIM + 2];
    }
    src_center[0] /= 3.0f;
    src_center[1] /= 3.0f;
    src_center[2] /= 3.0f;
    tar_center[0] /= 3.0f;
    tar_center[1] /= 3.0f;
    tar_center[2] /= 3.0f;

    // centered
    float* src_centered = new float[3*POINT_DIM];
    float* tar_centered = new float[3*POINT_DIM];

    for (int i = 0; i < 3; i++)
    {
        src_centered[i * POINT_DIM + 0] = src[indices[i] * POINT_DIM + 0] - src_center[0];
        src_centered[i * POINT_DIM + 1] = src[indices[i] * POINT_DIM + 1] - src_center[1];
        src_centered[i * POINT_DIM + 2] = src[indices[i] * POINT_DIM + 2] - src_center[2];
        
        tar_centered[i * POINT_DIM + 0] = tar[indices[i] * POINT_DIM + 0] - tar_center[0];
        tar_centered[i * POINT_DIM + 1] = tar[indices[i] * POINT_DIM + 1] - tar_center[1];
        tar_centered[i * POINT_DIM + 2] = tar[indices[i] * POINT_DIM + 2] - tar_center[2];
    }

    float R[9], t[3];
    solve_svd(src_centered, tar_centered, src_center, tar_center, 3, R, t);

    for (int j=0; j<16; j++) T[j] = 0;
    T[15] = 1.0f;

    // R
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            T[row*4 + col] = R[row*3 + col];
        }
    }

    // t
    T[3]  = t[0];
    T[7]  = t[1];
    T[11] = t[2];

    return true;
}

bool ICP::init_solution_estimator(const float* src, const float* tar, const int matched_num,
                                  const int max_iterations, float inlier_threshold, int min_inliers,
                                  float* init_solution)
{
    int n_points = matched_num;
    if (n_points<3)
    {
        LOGW("RANSAC: Too few correspondences (%d), stopping RANSAC.", n_points);
        return false;
    }

    int best_inlier_count = 0;
    bool* inliers       = new bool[n_points]; 
    bool* best_inliers  = new bool[n_points];
    for (int i=0; i<n_points; i++) inliers[i] = false;
    float* best_T = new float[16];

    // init random sampling
    std::uniform_int_distribution<int> dist(0, n_points-1);

    LOGV("RANSAC: Starting with %d points, %d iterations.", n_points, max_iterations);

    for (int iter = 0; iter < max_iterations; iter++)
    {   

        // random sampling 3 points
        int* sample_indices = new int[3];
        for (int i = 0; i < 3; i++) sample_indices[i] = dist(rng);

        // compute Rt matrix
        float* current_T = new float[16];
        if(!Rt_3_points(src, tar, sample_indices, current_T))
        {
            continue;
        }

        // compute inliers
        int inlier_count = 0;
        bool* current_inliers = new bool[n_points]; 
        for (int i=0; i<n_points; i++) current_inliers[i] = false;
        for (int i = 0; i < n_points; i++)
        {
            float src_x = src[i * POINT_DIM + 0];
            float src_y = src[i * POINT_DIM + 1];
            float src_z = src[i * POINT_DIM + 2];

            float tar_x = tar[i * POINT_DIM + 0];
            float tar_y = tar[i * POINT_DIM + 1];
            float tar_z = tar[i * POINT_DIM + 2];

            float transformed_x = current_T[0] * src_x + current_T[4] * src_y + current_T[8] * src_z + current_T[12];
            float transformed_y = current_T[1] * src_x + current_T[5] * src_y + current_T[9] * src_z + current_T[13];
            float transformed_z = current_T[2] * src_x + current_T[6] * src_y + current_T[10] * src_z + current_T[14];
            
            float dx = transformed_x - tar_x;
            float dy = transformed_y - tar_y;
            float dz = transformed_z - tar_z;

            float error = sqrt(dx*dx + dy*dy + dz*dz);

            if (error < inlier_threshold)
            {
                current_inliers[i] = true;
                inlier_count++;
            }
        }

        // update best parameters
        if (inlier_count > best_inlier_count)
        {
            best_inlier_count = inlier_count;
            memcpy(best_T, current_T, 16 * sizeof(float));
            memcpy(best_inliers, current_inliers, n_points * sizeof(bool));
            LOGV("RANSAC: Iter %d: Found %d inliers (threshold=%f).", iter+1, inlier_count, inlier_threshold);

            // if (best_inlier_count > (n_points*0.5))
            // {
            //     LOGV("RANSAC: Early termination with %d inliers.", inlier_count);
            //     break;
            // }
        }
    }

    // final Rt matrix by all inliers
    if(best_inlier_count >= min_inliers)
    {
        float* src_inliers = new float[best_inlier_count*POINT_DIM];
        float* tar_inliers = new float[best_inlier_count*POINT_DIM];
        float src_center[3] = {0.0f, 0.0f, 0.0f};
        float tar_center[3] = {0.0f, 0.0f, 0.0f};

        for (int i=0; i<n_points; i++)
        {
            if (best_inliers[i])
            {
                // center points
                src_center[0] += src[i*POINT_DIM + 0];
                src_center[1] += src[i*POINT_DIM + 1];
                src_center[2] += src[i*POINT_DIM + 2];
                    
                tar_center[0] += tar[i*POINT_DIM + 0];
                tar_center[1] += tar[i*POINT_DIM + 1];
                tar_center[2] += tar[i*POINT_DIM + 2];
            }
        }

        src_center[0] /= best_inlier_count;
        src_center[1] /= best_inlier_count;
        src_center[2] /= best_inlier_count;
        tar_center[0] /= best_inlier_count;
        tar_center[1] /= best_inlier_count;
        tar_center[2] /= best_inlier_count;

        // centered
        int inlier_idx = 0;
        for (int i = 0; i < n_points; i++)
        {
            if (best_inliers[i])
            {
                src_inliers[inlier_idx * POINT_DIM + 0] = src[i * POINT_DIM + 0] - src_center[0];
                src_inliers[inlier_idx * POINT_DIM + 1] = src[i * POINT_DIM + 1] - src_center[1];
                src_inliers[inlier_idx * POINT_DIM + 2] = src[i * POINT_DIM + 2] - src_center[2];
                        
                tar_inliers[inlier_idx * POINT_DIM + 0] = tar[i * POINT_DIM + 0] - tar_center[0];
                tar_inliers[inlier_idx * POINT_DIM + 1] = tar[i * POINT_DIM + 1] - tar_center[1];
                tar_inliers[inlier_idx * POINT_DIM + 2] = tar[i * POINT_DIM + 2] - tar_center[2];
                
                inlier_idx++;
            }
        }

        float R[9], t[3];
        solve_svd(src_inliers, tar_inliers, src_center, tar_center, best_inlier_count, R, t);

        for (int j=0; j<16; j++) init_solution[j] = 0;
        init_solution[15] = 1.0f;

        // R
        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                init_solution[row*4 + col] = R[row*3 + col];
            }
        }

        // t
        init_solution[3]  = t[0];
        init_solution[7]  = t[1];
        init_solution[11] = t[2];

        LOGV("RANSAC: Final result --- %d / %d inliers", best_inlier_count, n_points);
        LOGV("RANSAC: Init solution matrix: ");
        LOGV("  [%8.6f %8.6f %8.6f %8.6f]", init_solution[0], init_solution[1], init_solution[2], init_solution[3]);
        LOGV("  [%8.6f %8.6f %8.6f %8.6f]", init_solution[4], init_solution[5], init_solution[6], init_solution[7]);
        LOGV("  [%8.6f %8.6f %8.6f %8.6f]", init_solution[8], init_solution[9], init_solution[10], init_solution[11]);
        LOGV("  [%8.6f %8.6f %8.6f %8.6f]", init_solution[12], init_solution[13], init_solution[14], init_solution[15]);

        return true;
    }

    LOGW("RANSAC: Failed to find enough inliers (found %d, need %d).", best_inlier_count, min_inliers);
    return false;
}

ICP::ICP(): rng(std::random_device{}()) {}

ICP::ICP(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<reg_display::Display>("ICP");
    mat          = std::make_unique<mat::MAT>(false);
    neighbor     = std::make_unique<neighbor::Neighbor>(false);
    feat_det     = std::make_unique<iss::ISS>(false);
    feat_desc    = std::make_unique<feat_desc::FPFH>(false);
}

void ICP::set_source(const float* points, const int source_num)
{
    CUDA_CHECK(cudaMalloc((void **)&d_source,       sizeof(float)*source_num*POINT_DIM));
    CUDA_CHECK(cudaMemcpy(d_source, points,         sizeof(float)*source_num*POINT_DIM, cudaMemcpyHostToDevice));
    this->source_num = source_num;
}

void ICP::set_target(const float* points, const int target_num)
{
    CUDA_CHECK(cudaMalloc((void **)&d_target,       sizeof(float)*target_num*POINT_DIM));
    CUDA_CHECK(cudaMemcpy(d_target, points,         sizeof(float)*target_num*POINT_DIM, cudaMemcpyHostToDevice));
    this->target_num = target_num;
}

void ICP::set_neighbors(const int* neighbors, const int k, const int points_num)
{
    CUDA_CHECK(cudaMalloc((void **)&d_neighbors,    sizeof(int)*points_num*(k+1)));
    CUDA_CHECK(cudaMemcpy(d_neighbors, neighbors,   sizeof(int)*points_num*(k+1), cudaMemcpyHostToDevice));
}

void ICP::set_transformation_init(Eigen::Matrix4f trans_init)
{
    float transform_data[16];
	for (int i=0;i<4;i++)
	{
		for (int j=0;j<4;j++)
		{
			transform_data[i*4+j]=trans_init(i,j);
		}
	}
	cudaMemcpyToSymbol(d_transform_init, transform_data, sizeof(float) * 16 );
    this->is_init_solution = true;
}

void ICP::set_max_correspondence_distance(float dis)
{
    this->max_correspondence_distance = dis;
}

void ICP::set_max_iterations(int n)
{
    this->max_iterations = n;
}

void ICP::set_transformation_epsilon(double eps)
{
    this->transformation_epsilon = eps;
}

void ICP::set_euclidean_fitness_epsilon(float eps)
{
    this->euclidean_fitness_epsilon = eps;
}

float ICP::point_to_point(Eigen::Matrix4f transformation, const int k)
{
    LOGV("Set the max correspondence distance to %f m.",            this->max_correspondence_distance);
    LOGV("Set the maximum number of iterations to %d.",             this->max_iterations);
    LOGV("Set the transformation epsilon to %.10f.",                this->transformation_epsilon);
    LOGV("Set the euclidean distance difference epsilon to %f.",    this->euclidean_fitness_epsilon);
    LOGV("Loading source points size: %d.",                         source_num);
    LOGV("Loading target points size: %d.",                         target_num);

    CUDA_CHECK(cudaMalloc((void **)&d_points,           sizeof(float)*source_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_source_corr,      sizeof(float)*source_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_target_corr,      sizeof(float)*target_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_src_centered,     sizeof(float)*source_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_tar_centered,     sizeof(float)*target_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_loss,             sizeof(float)*source_num));
    CUDA_CHECK(cudaMalloc((void **)&d_valid_corr,       sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_src_center,       sizeof(float)*3));
    CUDA_CHECK(cudaMalloc((void **)&d_tar_center,       sizeof(float)*3));


    int grid_num    = (this->source_num+1024-1)/1024;
    int block_num   = 1024;
    // calculate point pose status by transformation_init
    // transform<<<grid_num, block_num>>>(d_source, source_num, d_points);
    // CUDA_CHECK(cudaMemcpy(d_source, d_points, sizeof(float)*source_num*POINT_DIM, cudaMemcpyDeviceToDevice));

    float  prev_loss            = 1000;
    float* h_source             = new float[source_num * POINT_DIM];
    float* h_target             = new float[target_num * POINT_DIM];
    int*   h_neighbors          = new int[source_num*(32+1)];
    float* h_current_transform  = new float[16];
    float* src_feat_output      = new float[source_num*POINT_DIM];
    float* tar_feat_output      = new float[target_num*POINT_DIM];
    float* init_solution        = new float[16];
    CUDA_CHECK(cudaMemcpy(h_target,     d_target,       sizeof(float)*target_num*POINT_DIM, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_neighbors,  d_neighbors,    sizeof(int)*source_num*(32+1), cudaMemcpyDeviceToHost));

    if (!this->is_init_solution)
    {
        LOGW("No initial solution, start estimating transform matrix.");
        float* src_normals = new float[source_num*3];
        float* tar_normals = new float[target_num*3];
        float* src_feat = new float[source_num*33];
        float* tar_feat = new float[target_num*33];
        float* src_matched = new float[source_num*POINT_DIM];
        float* tar_matched = new float[target_num*POINT_DIM];
        int matched_num = 0;
        CUDA_CHECK(cudaMemcpy(h_source, d_source, sizeof(float)*source_num*POINT_DIM, cudaMemcpyDeviceToHost));
        int* tar_neighbors = new int[target_num*(32+1)];
        neighbor->search_bf(h_target, target_num, k, tar_neighbors);
        feat_det->detector(h_source, source_num, h_neighbors, 5, k, src_feat_output);
        feat_det->detector(h_target, target_num, tar_neighbors, 5, k, tar_feat_output);
        mat->normals_estimator(h_source, source_num, h_neighbors, 32, src_normals);
        mat->normals_estimator(h_target, target_num, tar_neighbors, 32, tar_normals);
        feat_desc->descriptor(src_feat_output, source_num, h_neighbors, src_normals, k, 5, src_feat);
        feat_desc->descriptor(tar_feat_output, target_num, tar_neighbors, tar_normals,k, 5, tar_feat);
        match(src_feat_output, tar_feat_output, source_num, target_num, src_feat, tar_feat, src_matched, tar_matched, matched_num);
        this->is_init_solution = init_solution_estimator(src_matched, tar_matched, matched_num, 500, 15.0f, matched_num*0.3, init_solution);
        cudaMemcpyToSymbol(d_transform_init, init_solution, sizeof(float) * 16);
        transform<<<grid_num, block_num>>>(d_source, source_num, d_points);
        
        if (this->vis)
        {
            // display->set_feat(src_feat_output, tar_feat_output, src_feat, tar_feat, source_num, target_num);
            // display->set_feat_match(src_matched, tar_matched, matched_num);
            
            CUDA_CHECK(cudaMemcpy(h_source, d_points, sizeof(float)*source_num*POINT_DIM, cudaMemcpyDeviceToHost));

            display->set_source(h_source, source_num);
            display->set_target(h_target, target_num);
        }
        LOGV("Solution matrix estimation finish.");
        // return 0; // debug init_solution
    }else
    {
        transform<<<grid_num, block_num>>>(d_source, source_num, d_points);
    }

    CUDA_CHECK(cudaMemcpyFromSymbol(h_current_transform, d_transform_init, sizeof(float) * 16));

    for (int i=0; i<this->max_iterations; i++)
    {
        // reset
        float   avg_loss     = 0;
        int     current_corr = 0;
        CUDA_CHECK(cudaMemset(d_source_corr, 0, sizeof(float) * source_num * POINT_DIM));
        CUDA_CHECK(cudaMemset(d_target_corr, 0, sizeof(float) * target_num * POINT_DIM));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)*source_num));
        CUDA_CHECK(cudaMemset(d_valid_corr, 0, sizeof(int)));
        // loss_func<<<grid_num, block_num>>>(d_source, source_num, d_target, target_num, d_neighbors, 
        //                                    this->max_correspondence_distance, k, d_loss, d_valid_corr, d_source_corr, d_target_corr);
        loss_func<<<grid_num, block_num>>>(d_source, source_num, d_target, target_num, 
                                           this->max_correspondence_distance, d_loss, d_valid_corr, d_source_corr, d_target_corr);
        CUDA_CHECK(cudaMemcpy(&current_corr, d_valid_corr,  sizeof(int),                  cudaMemcpyDeviceToHost));
        float* h_loss = new float[current_corr];
	    CUDA_CHECK(cudaMemcpy(h_loss, d_loss,               sizeof(float)*current_corr,   cudaMemcpyDeviceToHost));
        
        LOGV("Correspondence points number: %d / %d.", current_corr, source_num);

        for (int j=0; j<current_corr; j++) 
        {
            avg_loss += h_loss[j];
        }
        
        if (current_corr < 3)
        {
            LOGE("Too few correspondences (%d), stopping ICP.", current_corr);
            break;
        }

        // check loss coverage
        avg_loss = avg_loss/current_corr;
        LOGV("Loss: %f, Iterations: %d/%d.", avg_loss, i+1, this->max_iterations);
        if (fabs(avg_loss)<this->euclidean_fitness_epsilon)
        {
            LOGW("Converged: Euclidean fitness epsilon reached at iteration %d.", i);
            break;
        }

        float* h_source_corr = new float[current_corr*POINT_DIM];
        float* h_target_corr = new float[current_corr*POINT_DIM];
        CUDA_CHECK(cudaMemcpy(h_source_corr, d_source_corr, sizeof(float)*current_corr*POINT_DIM, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_target_corr, d_target_corr, sizeof(float)*current_corr*POINT_DIM, cudaMemcpyDeviceToHost));

        // compute center and centered
        float h_src_center[] = {0.0f, 0.0f, 0.0f};
        float h_tar_center[] = {0.0f, 0.0f, 0.0f};
        for (int j=0; j<current_corr; j++)
        {
            h_src_center[0] += h_source_corr[j*POINT_DIM + 0];
            h_src_center[1] += h_source_corr[j*POINT_DIM + 1];
            h_src_center[2] += h_source_corr[j*POINT_DIM + 2];
                    
            h_tar_center[0] += h_target_corr[j*POINT_DIM + 0];
            h_tar_center[1] += h_target_corr[j*POINT_DIM + 1];
            h_tar_center[2] += h_target_corr[j*POINT_DIM + 2];
        }

        h_src_center[0] /= current_corr;
        h_src_center[1] /= current_corr;
        h_src_center[2] /= current_corr;
        h_tar_center[0] /= current_corr;
        h_tar_center[1] /= current_corr;
        h_tar_center[2] /= current_corr;

        // mat->compute_center(d_source_corr, current_corr, h_src_center);
        // mat->compute_center(d_target_corr, current_corr, h_tar_center);
        // CUDA_CHECK(cudaMemcpy(d_src_center, h_src_center, sizeof(float)*3, cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_tar_center, h_tar_center, sizeof(float)*3, cudaMemcpyHostToDevice));
        LOGV("Source center: [%f, %f, %f]", h_src_center[0], h_src_center[1], h_src_center[2]);
        LOGV("Target center: [%f, %f, %f]", h_tar_center[0], h_tar_center[1], h_tar_center[2]);

        float* h_src_centered = new float[current_corr*POINT_DIM];
        float* h_tar_centered = new float[current_corr*POINT_DIM];

        for (int j = 0; j < current_corr; j++)
        {
            h_src_centered[j * POINT_DIM + 0] = h_source_corr[j * POINT_DIM + 0] - h_src_center[0];
            h_src_centered[j * POINT_DIM + 1] = h_source_corr[j * POINT_DIM + 1] - h_src_center[1];
            h_src_centered[j * POINT_DIM + 2] = h_source_corr[j * POINT_DIM + 2] - h_src_center[2];
            
            h_tar_centered[j * POINT_DIM + 0] = h_target_corr[j * POINT_DIM + 0] - h_tar_center[0];
            h_tar_centered[j * POINT_DIM + 1] = h_target_corr[j * POINT_DIM + 1] - h_tar_center[1];
            h_tar_centered[j * POINT_DIM + 2] = h_target_corr[j * POINT_DIM + 2] - h_tar_center[2];
        }

        // int centered_grid_num = (current_corr + 1024 - 1) / 1024;
        // centered<<<centered_grid_num, block_num>>>(d_source_corr, current_corr, d_src_center, d_src_centered);
        // centered<<<centered_grid_num, block_num>>>(d_target_corr, current_corr, d_tar_center, d_tar_centered);
        // CUDA_CHECK(cudaMemcpy(h_src_centered, d_src_centered, sizeof(float)*current_corr*POINT_DIM, cudaMemcpyDeviceToHost));
        // CUDA_CHECK(cudaMemcpy(h_tar_centered, d_tar_centered, sizeof(float)*current_corr*POINT_DIM, cudaMemcpyDeviceToHost));
        LOGV("Centered finish.");

        float R[9], t[3];
        solve_svd(h_src_centered, h_tar_centered, h_src_center, h_tar_center, current_corr, R, t);

        // update delta transform 
        float T_delta[16];
        for (int j=0; j<16; j++) T_delta[j] = 0;
        T_delta[15] = 1.0f;

        // R_delta
        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                T_delta[row*4 + col] = R[row*3 + col];
            }
        }

        // t_delta
        T_delta[3]  = t[0];
        T_delta[7]  = t[1];
        T_delta[11] = t[2];

        LOGV("SVD result: ");
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", R[0], R[1], R[2], t[0]);
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", R[3], R[4], R[5], t[1]);
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", R[6], R[7], R[8], t[2]);

        LOGV("T_delta result:");
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", T_delta[0], T_delta[1], T_delta[2], T_delta[3]);
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", T_delta[4], T_delta[5], T_delta[6], T_delta[7]);
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", T_delta[8], T_delta[9], T_delta[10], T_delta[11]);
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", T_delta[12], T_delta[13], T_delta[14], T_delta[15]);

        // final R and t, T_new = T_delta * T_current
        float combined_transform[16];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                combined_transform[row*4 + col] = 0;
                for (int k = 0; k < 4; k++) {
                    combined_transform[row*4 + col] += T_delta[row*4 + k] * h_current_transform[k*4 + col];
                }
            }
        }

        LOGV("combined_transform result:");
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", combined_transform[0], combined_transform[1], combined_transform[2], combined_transform[3]);
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", combined_transform[4], combined_transform[5], combined_transform[6], combined_transform[7]);
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", combined_transform[8], combined_transform[9], combined_transform[10], combined_transform[11]);
        LOGV("    [%.6f, %.6f, %.6f, %.6f]", combined_transform[12], combined_transform[13], combined_transform[14], combined_transform[15]);
        
        // check error of transform
        float transform_diff = 0.0f;
        for (int i = 0; i < 16; i++)
        {
            float diff = combined_transform[i] - h_current_transform[i];
            transform_diff += diff * diff;
        }

        transform_diff = sqrtf(transform_diff);
        LOGV("Transform diff: %f --> %f.", transform_diff, this->transformation_epsilon);
        if (transform_diff < this->transformation_epsilon)
        {
            LOGW("Converged: Transformation epsilon reached at iteration %d", i);
            memcpy(h_current_transform, combined_transform, sizeof(float) * 16);
            break;
        }

        // update transform
        memcpy(h_current_transform, combined_transform, sizeof(float) * 16);
        CUDA_CHECK(cudaMemcpyToSymbol(d_transform_init, h_current_transform, sizeof(float) * 16));
        
        // apply new transform mat
        transform<<<grid_num, block_num>>>(d_source, source_num, d_points);
        CUDA_CHECK(cudaMemcpy(d_source, d_points, sizeof(float)*source_num*POINT_DIM, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        // vislization
        if (this->vis)
        {
            CUDA_CHECK(cudaMemcpy(h_source, d_source, sizeof(float)*source_num*POINT_DIM, cudaMemcpyDeviceToHost));

            // display->set_source(h_source_corr, current_corr);
            // display->set_target(h_target_corr, current_corr);
            display->set_corr(h_source_corr, h_target_corr, current_corr);
            display->set_source(h_source, source_num);
            display->set_target(h_target, target_num);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // update loss
        prev_loss = avg_loss;
        
        delete[] h_source_corr;
        delete[] h_target_corr;
        // delete[] h_src_center;
        // delete[] h_tar_center;
        delete[] h_src_centered;
        delete[] h_tar_centered;
        delete[] h_loss;
    }
    delete[] h_source;
    delete[] h_target;
    delete[] h_current_transform;
    return 0.0f;
}


ICP::~ICP()
{}

}