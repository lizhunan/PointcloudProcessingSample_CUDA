#ifndef __NEIGHBOR_H__
#define __NEIGHBOR_H__

#include <cuda_runtime.h>
#include <float.h>
#include "mat.h"
#include "cuda_base.h"
#include "logger.h"
#include "/workspace/src/display/include/display.h"

#define POINT_DIM   6       // Point dimension: [x, y, z, intensity, index, label]
#define MAX_K       32      // Maximum supported KNN neighbors per thread

namespace neighbor {

/**
 * @class Neighbor
 * @brief GPU-based neighbor search module for point cloud processing.
 *
 * @details
 *  - Supports brute-force KNN search (CUDA)
 *  - Provides extensible interface for KD-Tree / Octree (future)
 *  - Optional visualization for debugging
 *
 *  Output format:
 *      For each point:
 *          [self1, nn1, nn2, ..., self2, nn1, nn2, ..., nnK] for each point
 */
class Neighbor {

public:
    /**
     * @brief Default constructor.
     */
    Neighbor();
    
    /**
     * @brief Construct Neighbor with visualization option.
     *
     * @param vis Enable visualization
     */
    Neighbor(bool vis);

    /**
     * @brief Destructor.
     */
    ~Neighbor();


public:

    /**
     * @brief Perform brute-force KNN search on GPU.
     *
     * @details
     *  - Allocates GPU memory
     *  - Copies input points to device
     *  - Launches CUDA kernel
     *  - Optionally visualizes results
     *
     * @param h_points      Host input point cloud [N * POINT_DIM]
     * @param points_num    Number of points
     * @param k             Number of nearest neighbors
     * @param neighbors     Output buffer (host, optional use)
     */
    void search_bf(const float* h_points, const int points_num, const int k, int* neighbors);

    /**
     * @brief KD-Tree based neighbor search (TODO).
     */
    void search_kdtree();

    /**
     * @brief Octree based neighbor search (TODO).
     */
    void search_octree();

private:
    std::shared_ptr<display::Display>   display;        // Visualization module

private:
    bool vis = false;   // Enable visualization

    /* GPU memory */
    float*      d_points;           // Device point cloud
    int*        d_neighbors;        // Device neighbor indices

};  

}   // namespace neighbor

#endif