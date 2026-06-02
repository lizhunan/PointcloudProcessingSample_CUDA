#ifndef _CLUSTER_H__
#define _CLUSTER_H__

#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <time.h>
#include <mutex>
#include <cmath>
#include <Eigen/Dense>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/binary_search.h> 
#include "/workspace/src/common/include/logger.h"
#include "display.h"
#include "/workspace/src/common/include/neighbor.h"

namespace cluster
{

class DBSCAN
{
public:
    
    DBSCAN(); 
    DBSCAN(bool vis);
    ~DBSCAN();

public:
   
    void clustering(float* cluster_points, const int *range);
    void set_points(const float* points, const int point_num);
    void set_cell_size(float cell_size);
    void set_dist_epsilon(float eps);
    void set_min_pts(int min_pts);

private:

    void exclusive_scan(const std::vector<int>& in, std::vector<int>& out);
    void sparse_voxel_compress(const int point_num, 
                                thrust::device_vector<int64_t>& unique_voxels, 
                                int &unique_voxels_size);
    void compressed_spare_row(const int point_num, 
                                const int unique_voxels_size, 
                                const thrust::device_vector<int64_t>& d_unique_voxels, 
                                thrust::device_vector<int>& d_voxel_count);
    void voxel_neighbor(const int unique_voxels_size, 
                        const thrust::device_vector<int64_t>& d_unique_voxels, 
                        const int grid_x, const int grid_y, const int grid_z,
                        thrust::device_vector<int>& d_neighbor_offset, 
                        thrust::device_vector<int>& d_neighbors, 
                        int& total_neighbors);
    void union_neighbor(const int unique_voxels_size, 
                        const thrust::device_vector<int>& d_voxel_count, 
                        const thrust::device_vector<int>& d_neighbor_offset, 
                        const thrust::device_vector<int>& d_neighbors,
                        thrust::device_vector<int>& d_root_out, 
                        thrust::device_vector<uint8_t>& d_is_core);
    void assgin_cluster(const int point_num, 
                        const int unique_voxels_size,
                        const thrust::device_vector<int64_t>& d_unique_voxels,  
                        const thrust::device_vector<int>& d_root_out, 
                        const thrust::device_vector<uint8_t>& d_is_core,
                        const thrust::device_vector<int>& d_neighbor_offset, 
                        const thrust::device_vector<int>& d_neighbors,
                        thrust::device_vector<int> &d_point_labels);
    void cluster_distribution(const std::string name, const std::vector<int> labels, const int point_num);

private:

    bool vis;

    std::shared_ptr<logger::Logger> logger;
    std::shared_ptr<cluster_display::Display> display;

    float     dbscan_cell_size            = 0.8f;     /// Size (meters) of each BEV grid cell.
    float     dbscan_epsilon              = 0.5f;     /// DBSCAN distance threshold.
    int       dbscan_min_pts              = 10;       /// Minimum points per core voxel.

    float       *d_points;
    int         *d_labels;
    int         *d_counts;              /// Accumulator for number of points per cluster (size = KEAMS_K).
    float       *d_cluster_points;      /// Device buffer storing clustered points.
    int         *d_cluster_points_num;  /// Device-side counter for the number of clustered points.
    int64_t     *d_voxel_idx;           /// Device buffer storing voxel index for each point.
    
    float       *h_points;                  /// Host-side buffer storing input point cloud.
    float       *h_keams_centroids;         /// Host-side buffer storing K-Means centroids.
    float       *h_new_keams_centroids;     /// Host-side buffer storing updated K-Means centroids.
    int         *h_counts;                  /// Accumulator for number of points per cluster (size = KEAMS_K).

   
    int     point_num;
    float   pointcloud_range[6]           = {0.0f};
    int     grid_x;     /// Number of grid cells along X axis.
    int     grid_y;     /// Number of grid cells along Y axis.
    int     grid_z;     /// Number of grid cells along Z axis.
    int     grid_num;   /// Total number of BEV grid cells = grid_x * grid_y.

}; 
} 

#endif