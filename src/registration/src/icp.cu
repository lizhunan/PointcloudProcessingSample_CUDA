#include "icp.h"

namespace reg{

__constant__ float d_transform_init[16];

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

__global__ void solve_Rt()
{

}

__global__ void loss_func(const float* source, const int source_num, const float* target, const int target_num, const int* neighbor_indices,
                     const float max_dist, const int k, float* loss, int* valid_correspondences)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadid >= source_num || threadid >= target_num) return;

    // Load source point
    float p_s[3]      = {source[threadid*POINT_DIM+0], source[threadid*POINT_DIM+1], source[threadid*POINT_DIM+2]};

    int min_idx                 = -1;
    float min_dist              = max_dist;

    // Neighbor indexing
    // Layout:
    //   neighbor_indices[p_id * (k+1) + 0]     → self index
    //   neighbor_indices[p_id * (k+1) + i]     → i-th neighbor
    int stride = k + 1;
    int base   = threadid * stride;
    for (int i=0; i<k; i++)
    {
        int idx                 = neighbor_indices[base + i];
        float p_t[3]            = {target[idx * POINT_DIM + 0], target[idx * POINT_DIM + 1], target[idx * POINT_DIM + 2]};
        float dist              = mat::distf(p_s, p_t);
        
        /**
         * Skip neighbors outside the correspondence distance
         */
        if (dist > max_dist) continue;

        if (dist < min_dist)
        {
            min_dist = dist;
            min_idx = idx;
        }

    }
    loss[threadid] = min_dist;
    atomicAdd(valid_correspondences, 1);
}



ICP::ICP(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("ICP");
    mat         = std::make_unique<mat::MAT>(false);
    neighbor    = std::make_unique<neighbor::Neighbor>(false);
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
}

void ICP::set_max_correspondence_distance(int dis)
{
    this->max_correspondence_distance = dis;
}

void ICP::set_max_iterations(int n)
{
    this->max_iterations = n;
}

void ICP::set_transformation_epsilon(int eps)
{
    this->transformation_epsilon = 1e-8;
}

void ICP::set_euclidean_fitness_epsilon(int eps)
{
    this->euclidean_fitness_epsilon = eps;
}

float ICP::point_to_point(Eigen::Matrix4f transformation, const int k)
{
    LOGV("Set the max correspondence distance to %d m.",            this->max_correspondence_distance);
    LOGV("Set the maximum number of iterations to %d.",             this->max_iterations);
    LOGV("Set the transformation epsilon to %d.",                   this->transformation_epsilon);
    LOGV("Set the euclidean distance difference epsilon to %d.",    this->euclidean_fitness_epsilon);
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*source_num*POINT_DIM));
    CUDA_CHECK(cudaMalloc((void **)&d_loss,         sizeof(float)*source_num));
    CUDA_CHECK(cudaMalloc((void **)&d_valid_crpd,   sizeof(int)));


    int grid_num    = (this->source_num+1024-1)/1024;
    int block_num   = 1024;
    // calculate point pose status by transformation_init
    transform<<<grid_num, block_num>>>(d_source, source_num, d_points);
    CUDA_CHECK(cudaMemcpy(d_source, d_points, sizeof(float)*source_num*POINT_DIM, cudaMemcpyDeviceToDevice));

    float prev_loss = INFINITY;
    float* h_loss = new float [source_num];
    for (int i=0; i<this->max_iterations; i++)
    {
        // calculate loss bewteen target and source(current trans) and output new trans
        float   current_loss = INFINITY;
        int     current_crpd = 0;
        CUDA_CHECK(cudaMemset(d_valid_crpd, 0, sizeof(int)));
        loss_func<<<grid_num, block_num>>>(d_source, source_num, d_target, target_num, d_neighbors, 
                                           this->max_correspondence_distance, k, d_loss, d_valid_crpd);
	    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, sizeof(float)*source_num, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&current_crpd, d_valid_crpd, sizeof(int), cudaMemcpyDeviceToHost));
        for (int j=0; j<source_num; j++) current_loss+= h_loss[j];
        
        if (current_crpd < 3) {
            LOGV("Too few correspondences (%d), stopping ICP", current_crpd);
            break;
        }

        // check loss coverage
        if (fabs(current_loss-prev_loss)<this->euclidean_fitness_epsilon)
        {
            LOGV("Converged: Euclidean fitness epsilon reached at iteration %d.", i);
            break;
        }
        
        // solve R and t
        float R[9], t[3];
        
        // check error of transform

        // update transform

        // vislization

        prev_loss = current_loss;

        // return transform matrix

        // calculate point pose status by each current trans
        transform<<<grid_num, block_num>>>(d_source, source_num, d_points);
        CUDA_CHECK(cudaMemcpy(d_source, d_points, sizeof(float)*source_num*POINT_DIM, cudaMemcpyDeviceToDevice));
    }

    delete[] h_loss;
    return true;
}


ICP::~ICP()
{}

}