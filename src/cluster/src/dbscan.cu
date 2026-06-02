#include "dbscan.h"

namespace cluster
{

/* ============================================================
*              Utils for CUDA Compute
* ============================================================ */

/**
 * @brief   Convert 3D voxel indices to a linear voxel ID.
 *
 * @details
 *          Maps a 3D voxel coordinate (ix, iy, iz) into a unique
 *          linear index using row-major order:
 *
 *              id = iz * (gx * gy) + iy * gx + ix
 *
 *          This linearization guarantees a one-to-one mapping
 *          between 3D voxel coordinates and a compact integer ID,
 *          which is suitable for sorting, binary search, and
 *          sparse voxel compression.
 *
 * @param ix   Voxel index along x-axis.
 * @param iy   Voxel index along y-axis.
 * @param iz   Voxel index along z-axis.
 * @param gx   Number of voxels along x-axis.
 * @param gy   Number of voxels along y-axis.
 *
 * @return     Linear voxel ID.
 */
__device__ __host__ inline int64_t linear_voxel_id(int ix, int iy, int iz, int gx, int gy) {
    // linear index: iz * (gx*gy) + iy*gx + ix
    return (int64_t)iz * ( (int64_t)gx * (int64_t)gy ) + (int64_t)iy * (int64_t)gx + (int64_t)ix;
}

/**
 * @brief   Device-side binary search for int64 voxel IDs.
 *
 * @details
 *          Performs a standard binary search on a sorted array
 *          of int64 voxel IDs. This function is used extensively
 *          during voxel neighbor construction to determine whether
 *          a candidate neighboring voxel exists in the sparse
 *          voxel set.
 *
 *          Time complexity: O(log V), where V is the number of
 *          non-empty voxels.
 *
 * @param arr  Pointer to sorted array of voxel IDs.
 * @param n    Number of elements in the array.
 * @param key  Query voxel ID.
 *
 * @return     Index of the voxel ID in arr if found, otherwise -1.
 */

__device__ int device_binary_search_int64(const int64_t* arr, int n, int64_t key) {
    int l = 0, r = n - 1;
    while (l <= r) {
        int m = (l + r) >> 1;
        int64_t v = arr[m];
        if (v == key) return m;
        if (v < key) l = m + 1;
        else r = m - 1;
    }
    return -1;
}

/**
 * @brief   Squared Euclidean distance between two fixed-point points.
 *
 * @details
 *          Computes squared distance in metric space by converting
 *          fixed-point coordinates (/256) into floating-point values.
 *          The squared distance is returned to avoid unnecessary
 *          square root operations during DBSCAN neighbor checks.
 *
 * @param ax,ay,az  Coordinates of point A (fixed-point).
 * @param bx,by,bz  Coordinates of point B (fixed-point).
 *
 * @return          Squared Euclidean distance.
 */
__device__ float dist(float ax, float ay, float az,
                    float bx, float by, float bz)
{
    float x1 = ax*1.0/256;
	float y1 = ay*1.0/256;
    float z1 = az*1.0/256;

    float x2 = bx*1.0/256;
	float y2 = by*1.0/256;
    float z2 = bz*1.0/256;

    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;

    return dx*dx + dy*dy + dz*dz;
}

/* ============================================================
*              DBSCAN: Voxelization & Sparse Representation
* ============================================================ */

// 1) Compute voxel id per point (grid_size = 1 / cell_size passed for numeric stability)
/**
 * @brief   Assign each point to a voxel via spatial discretization.
 *
 * @details
 *          This kernel converts each input point into a 3D voxel index
 *          based on a fixed grid resolution. Points outside the
 *          predefined grid bounds are marked with voxel ID = -1.
 *
 *          The output voxel IDs are later used to:
 *            1) Identify non-empty voxels.
 *            2) Build sparse voxel representations.
 *            3) Perform voxel-level DBSCAN clustering.
 *
 * @param points        Input point cloud stored in fixed-point format (/256).
 * @param point_num     Total number of points.
 * @param x_min         Minimum coordinate of the voxel grid along x-axis.
 * @param y_min         Minimum coordinate of the voxel grid along y-axis.
 * @param z_min         Minimum coordinate of the voxel grid along z-axis.
 * @param GRID_SIZE     Reciprocal of voxel size (1 / cell_size).
 * @param grid_x        Grid resolution along x-axis.
 * @param grid_y        Grid resolution along y-axis.
 * @param grid_z        Grid resolution along z-axis.
 * @param out_voxel_id  Output linear voxel ID per point (-1 if out-of-range).
 */
__global__ void voxelization(const float* points, int point_num, 
                            float x_min, float y_min, float z_min, 
                            float GRID_SIZE, int grid_x, int grid_y, int grid_z,
                            int64_t* out_voxel_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= point_num) return;

    float x = points[tid * POINT_DIM + 0];
    float y = points[tid * POINT_DIM + 1];
    float z = points[tid * POINT_DIM + 2];

    int ix = (int)floorf((x - x_min) * GRID_SIZE);
    int iy = (int)floorf((y - y_min) * GRID_SIZE);
    int iz = (int)floorf((z - z_min) * GRID_SIZE);

    if (ix < 0 || ix >= grid_x || iy < 0 || iy >= grid_y || iz < 0 || iz >= grid_z) {
        out_voxel_id[tid] = -1;
    } else {
        out_voxel_id[tid] = linear_voxel_id(ix, iy, iz, grid_x, grid_y);
    }
}

/* ============================================================
*              DBSCAN: Voxel Neighbor Construction (CSR)
* ============================================================ */

/**
 * @brief   Count upper-bound voxel neighbors for each voxel.
 *
 * @details
 *          For each non-empty voxel, this kernel counts the number of
 *          spatially adjacent voxel positions within a 3×3×3 cube.
 *          The count is an upper bound (≤27), as actual existence of
 *          neighboring voxels is not checked at this stage.
 *
 *          This kernel is used to preallocate memory for voxel
 *          neighbor CSR construction.
 *
 * @param unique_voxels   Sorted list of non-empty voxel IDs.
 * @param voxel_size      Number of non-empty voxels.
 * @param grid_x          Grid resolution along x-axis.
 * @param grid_y          Grid resolution along y-axis.
 * @param grid_z          Grid resolution along z-axis.
 * @param neighbor_count  Output upper-bound neighbor count per voxel.
 */
__global__ void count_voxel_neighbors(const int64_t* unique_voxels, int voxel_size, 
                                    int grid_x, int grid_y, int grid_z, 
                                    int* neighbor_count) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= voxel_size) return;
    int64_t vid_id = unique_voxels[vid];
    int ix = (int)(vid_id % grid_x);
    int iy = (int)((vid_id / grid_x) % grid_y);
    int iz = (int)(vid_id / ( (int64_t)grid_x * grid_y ));
    int cnt = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        int nz = iz + dz; if (nz < 0 || nz >= grid_z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = iy + dy; if (ny < 0 || ny >= grid_y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = ix + dx; if (nx < 0 || nx >= grid_x) continue;
                // Neighbor id existence will be checked in fill kernel (this kernel just count a potential max).
                ++cnt;
            }
        }
    }
    neighbor_count[vid] = cnt; // Upper bound (<=27).
}

/**
 * @brief   Construct voxel adjacency list in CSR format.
 *
 * @details
 *          For each non-empty voxel, this kernel enumerates its 26
 *          neighboring voxel coordinates (plus itself), checks whether
 *          each neighbor exists in the sparse voxel set using binary
 *          search, and writes valid neighbors contiguously into a
 *          CSR-formatted adjacency array.
 *
 *          Each thread processes one voxel and writes into a
 *          precomputed, exclusive segment of the neighbors array,
 *          ensuring race-free execution without atomic operations.
 *
 * @param unique_voxels    Sorted array of non-empty voxel IDs.
 * @param voxel_size       Number of non-empty voxels.
 * @param grid_x           Grid resolution along x-axis.
 * @param grid_y           Grid resolution along y-axis.
 * @param grid_z           Grid resolution along z-axis.
 * @param neighbor_offset  CSR offset array of size V+1.
 * @param neighbors_out    CSR neighbor index array.
 */
__global__ void fill_voxel_neighbors(const int64_t* unique_voxels, int voxel_size, 
                                    int grid_x, int grid_y, int grid_z,
                                    const int* neighbor_offset, 
                                    int* neighbors_out) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= voxel_size) return;

    int64_t vid_id = unique_voxels[vid];
    int ix = (int)(vid_id % grid_x);
    int iy = (int)((vid_id / grid_x) % grid_y);
    int iz = (int)(vid_id / ( (int64_t)grid_x * grid_y ));

    int write_pos = neighbor_offset[vid];
    
    for (int dz = -1; dz <= 1; ++dz) {
        int nz = iz + dz; if (nz < 0 || nz >= grid_z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = iy + dy; if (ny < 0 || ny >= grid_y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = ix + dx; if (nx < 0 || nx >= grid_x) continue;
                int64_t nvid    = linear_voxel_id(nx, ny, nz, grid_x, grid_y);
                int idx         = device_binary_search_int64(unique_voxels, voxel_size, nvid);   // Binary search in unique_voxels (device)
                if (idx >= 0) {
                    neighbors_out[write_pos++] = idx;   // Store neighbor as index into unique_voxels
                }
            }
        }
    }
    // NOTE: write_pos should not overflow if neighbor_offset was computed correctly.
}

/**
 * @brief   Determine core voxels based on point density.
 *
 * @details
 *          A voxel is marked as a core voxel if the number of points
 *          contained within it is greater than or equal to minPts.
 *          This is the voxel-level analog of the core-point condition
 *          in classical DBSCAN.
 *
 * @param voxel_count   Number of points per voxel.
 * @param voxel_size    Number of non-empty voxels.
 * @param minPts        Minimum points threshold.
 * @param is_core       Output binary flag per voxel (1 = core).
 */
__global__ void voxel_core_mark(const int* voxel_count, 
                                int voxel_size, 
                                int minPts, 
                                uint8_t* is_core) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= voxel_size) return;
    is_core[vid] = (voxel_count[vid] >= minPts) ? 1 : 0;
}

/**
 * @brief   Device-side union-find find operation with path halving.
 *
 * @details
 *          Finds the root of a voxel in the disjoint-set forest.
 *
 * @param parent  Union-find parent array.
 * @param x       Query voxel index.
 *
 * @return        Root index of the set containing x.
 */
__device__ __forceinline__ int find_root(int* parent, int x) {
    int p = parent[x];
    while (p != parent[p]) p = parent[p]; // Loop until p is a root(p == parent[p]).
    return p;
}

/**
 * @brief   Atomic union operation for two voxel sets.
 *
 * @details
 *          Merges the sets containing voxels a and b using an atomic
 *          compare-and-swap operation to ensure thread safety.
 *          The union is performed by attaching the higher index
 *          root to the lower index root.
 *
 * @param parent  Union-find parent array.
 * @param a,b     Voxel indices to be unified.
 */
__device__ void union_set(int* parent, int a, int b) {
    while (true) {
        int ra = find_root(parent, a);
        int rb = find_root(parent, b);
        if (ra == rb) return;   // Within the same set.
        int high = ra > rb ? ra : rb;
        int low  = ra < rb ? ra : rb;

        /* ------------------------------------------------------------
         * Thread safety union opreation.
         * if parent[high] == high then parent[high] <- low return parent[hight]
         * ONLY when high is still root, allow the current thread(higher index) to hang it under low index.
         * ------------------------------------------------------------*/
        int old = atomicCAS(&parent[high], high, low);
        if (old == high) return;
    }
}

/**
 * @brief   Union neighboring core voxels.
 *
 * @details
 *          For each core voxel, this kernel iterates over its voxel
 *          neighbors and unions them if both voxels are core voxels.
 *          This step constructs connected components of core voxels,
 *          corresponding to DBSCAN clusters.
 *
 * @param is_core          Core voxel flags.
 * @param voxel_size       Number of non-empty voxels.
 * @param neighbor_offset  CSR offset array.
 * @param neighbors        CSR neighbor list.
 * @param parent           Union-find parent array.
 */
__global__ void union_core_neighbors(const uint8_t* is_core, int voxel_size,
                                    const int* neighbor_offset,
                                    const int* neighbors,
                                    int* parent) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= voxel_size) return;
    if (!is_core[vid]) return;
    int start   = neighbor_offset[vid];         // Obtain neighbor(first) vexel for current voxel.
    int end     = neighbor_offset[vid + 1];     // Obtain neighbor(last) vexel for current voxel.
    for (int i = start; i < end; ++i) {         // Iterates all neighbors.
        int nidx = neighbors[i];                // Neighbor index.
        if (nidx == vid) continue;              // Exclude self-voxel.
        if (is_core[nidx])                      // Neighbor voxel must be core-voxel.
            union_set(parent, vid, nidx);
    }
}

/**
 * @brief   Compress union-find trees and extract final roots.
 *
 * @details
 *          Applies path compression to obtain the final root
 *          (representative) for each voxel in the union-find structure.
 *
 * @param voxel_size        Number of voxels.
 * @param parent            Union-find parent array.
 * @param root_out          Output root index per voxel.
 */
__global__ void compress_roots(int voxel_size, int* parent, int* root_out) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= voxel_size) return;
    int x = vid;
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    root_out[vid] = x;
}

/**
 * @brief   Assign cluster IDs to union-find roots.
 *
 * @details
 *          Each unique root corresponding to a core voxel is assigned
 *          a unique cluster ID using atomic operations. Non-core voxels
 *          are ignored at this stage.
 *
 * @param voxel_size           Number of voxels.
 * @param root_out             Root index per voxel.
 * @param is_core              Core voxel flags.
 * @param cluster_id_of_root   Output cluster ID per root.
 * @param next_cluster_id      Global atomic counter.
 */
__global__ void map_root_to_clusterid(int voxel_size, 
                                    const int* root_out,
                                    const uint8_t* is_core,
                                    int* cluster_id_of_root,
                                    int* next_cluster_id) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= voxel_size) return;
    if (!is_core[vid]) return;
    int root = root_out[vid];
    
    /* ------------------------------------------------------------
     * For thread safety
     * ------------------------------------------------------------
     * cluster_id_of_root:
     *      -1  : Cluster not assigned.
     *      -2  : Occupied by other thread, cluster id generating.
     *      >=0 : Cluster has assigned.
     * ------------------------------------------------------------ */
    int old  = atomicCAS(&cluster_id_of_root[root], -1, -2);
    /* ONLY the first thread that successfully seizes root(old==-1) will enter here. */
    if (old == -1) {
        int cid = atomicAdd(next_cluster_id, 1);    // Atomically allocate a new cluster ID from the global counter.
        cluster_id_of_root[root] = cid;
    }
}

/**
 * @brief   Assign cluster ID to each voxel.
 *
 * @details
 *          Maps the cluster ID of each root voxel to all voxels
 *          belonging to that root. Voxels not associated with any
 *          core cluster are labeled as noise (-1).
 *
 * @param voxel_size           Number of voxels.
 * @param root_out             Root index per voxel.
 * @param cluster_id_of_root   Cluster ID per root.
 * @param voxel_cluster_id     Output cluster ID per voxel.
 */
__global__ void assign_voxel_cluster(int voxel_size, 
                                    const int* root_out, 
                                    const int* cluster_id_of_root, 
                                    int* voxel_cluster_id) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= voxel_size) return;
    int root = root_out[vid];
    int cid = cluster_id_of_root[root];
    if (cid >= 0) voxel_cluster_id[vid] = cid;
    else voxel_cluster_id[vid] = -1; // Not assigned (non-core singleton).
}

/**
 * @brief Attach border voxels to neighboring core voxel clusters (DBSCAN semantics).
 *
 * @details
 *          This kernel processes all voxels in parallel. For each non-core voxel, it
 *          checks whether it has at least one neighboring core voxel. If so, the voxel
 *          is assigned to the same cluster as that core voxel. Otherwise, it remains
 *          unassigned and is treated as noise.
 *
 *          Neighbor relationships are stored in CSR (Compressed Sparse Row) format.
 *
 * @param voxel_size         Number of unique voxels
 * @param is_core            Flag array indicating whether each voxel is a core voxel
 * @param neighbor_offset    CSR row pointer array (size = voxel_size + 1)
 * @param neighbor_list      CSR adjacency list storing neighboring voxel indices
 * @param voxel_cluster_id   Output array: cluster id per voxel (-1 if unassigned)
 */
__global__ void attach_border_voxels(int voxel_size,
                                    const uint8_t* is_core,
                                    const int* neighbor_offset,
                                    const int* neighbor_list,
                                    int* voxel_cluster_id)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= voxel_size) return;

    
    if (is_core[vid]) return;                   // Skip core voxels (already assigned)
    if (voxel_cluster_id[vid] >= 0) return;     // If already assigned (defensive check)

    int begin = neighbor_offset[vid];
    int end   = neighbor_offset[vid + 1];

    /* Search for any neighboring core voxel. */
    for (int i = begin; i < end; ++i)
    {
        int nb = neighbor_list[i];

        if (is_core[nb])
        {
            int cid = voxel_cluster_id[nb];
            if (cid >= 0)
            {
                voxel_cluster_id[vid] = cid;    // Attach to the first found core neighbor
                return;
            }
        }
    }

    // NOTE: If no core neighbor found: voxel_cluster_id[vid] remains -1 (noise)
}

__global__ void map_voxel_cluster_to_points(const int64_t* voxel_id_of_point,int point_num,
                                            const int64_t* unique_voxels, int voxel_size,
                                            const int* voxel_cluster_id, int* point_labels) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= point_num) return;
    int64_t vid = voxel_id_of_point[pid];
    if (vid < 0) { point_labels[pid] = -1; return; }
    // find index in unique_voxels
    int idx = device_binary_search_int64(unique_voxels, voxel_size, vid);
    if (idx < 0) { point_labels[pid] = -1; return; }
    point_labels[pid] = voxel_cluster_id[idx];
}

__global__ void collect_cluster(const float *points, const int point_num, 
                                const int *labels, 
                                float *cluster_points, 
                                int *cluster_points_num)
{
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= point_num) return;

    int label = labels[threadid];
    if (label < 0) return; 

    int id = atomicAdd(cluster_points_num, 1);
    cluster_points[id*POINT_DIM + 0] = points[threadid*POINT_DIM + 0];
    cluster_points[id*POINT_DIM + 1] = points[threadid*POINT_DIM + 1];
    cluster_points[id*POINT_DIM + 2] = points[threadid*POINT_DIM + 2];
    cluster_points[id*POINT_DIM + 3] = points[threadid*POINT_DIM + 3];
    cluster_points[id*POINT_DIM + 4] = points[threadid*POINT_DIM + 4];
    cluster_points[id*POINT_DIM + 5] = label;
}


DBSCAN::DBSCAN()
{
}

DBSCAN::DBSCAN(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<cluster_display::Display>("DBSCAN");
}

void DBSCAN::set_points(const float* points, const int point_num)
{
    CUDA_CHECK(cudaMalloc((void **)&d_points,       sizeof(float)*point_num*POINT_DIM));
    CUDA_CHECK(cudaMemcpy(d_points, points,         sizeof(float)*point_num*POINT_DIM, cudaMemcpyHostToDevice));
    this->point_num = point_num;
}

void DBSCAN::set_cell_size(float cell_size)
{
    this->dbscan_cell_size = cell_size;
}

void DBSCAN::set_dist_epsilon(float epsilon)
{
    this->dbscan_epsilon = epsilon;
}

void DBSCAN::set_min_pts(int min_pts)
{
    this->dbscan_min_pts = min_pts;
}

void DBSCAN::clustering(float* cluster_points, const int *range)
{
    CUDA_CHECK(cudaMalloc((void **)&d_labels,                 this->point_num * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_cluster_points,         this->point_num * POINT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_cluster_points_num,     sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_voxel_idx,              this->point_num * sizeof(int64_t)));

    h_points = (float*) malloc(this->point_num * POINT_DIM * sizeof(float));

    pointcloud_range[0] = range[0]; // x_min
    pointcloud_range[1] = range[1]; // x_max
    pointcloud_range[2] = range[2]; // y_min
    pointcloud_range[3] = range[3]; // y_max
    pointcloud_range[4] = range[4]; // z_min
    pointcloud_range[5] = range[5]; // z_max

    grid_x      = (int)ceilf((pointcloud_range[1] - pointcloud_range[0]) / this->dbscan_cell_size);
    grid_y      = (int)ceilf((pointcloud_range[3] - pointcloud_range[2]) / this->dbscan_cell_size);
    grid_z      = (int)ceilf((pointcloud_range[5] - pointcloud_range[4]) / this->dbscan_cell_size);
    grid_num    = grid_x * grid_y * grid_z;

    LOGV("DBSCAN parameter initialization:");
    LOGV("DBSCAN_EPS: %f", this->dbscan_epsilon);
    LOGV("DBSCAN_CELL_SIZE: %f", this->dbscan_cell_size);
    LOGV("Grid X: %d, Grid Y: %d, Grid Z: %d, number of grid: %d.", grid_x, grid_y, grid_z, grid_num);

    /* Voxelization: map each point to a linear voxel index */
    int grid_num    = (this->point_num+1024-1)/1024;
    int block_num   = 1024;
    voxelization<<<grid_num, block_num>>>(d_points, this->point_num, 
                                          pointcloud_range[0], pointcloud_range[2], pointcloud_range[4], 
                                          1.0f / this->dbscan_cell_size, 
                                          grid_x, grid_y, grid_z,
                                          d_voxel_idx);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Compress sparse voxels (remove empty voxels),  result: sorted list of unique non-empty voxel IDs. */
    int unique_voxels_size = 0;                         // Number of non-empty voxel list.
    thrust::device_vector<int64_t> d_unique_voxels;     // Non-empty voxel list(sorted).
    sparse_voxel_compress(point_num, d_unique_voxels, unique_voxels_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    LOGV("Unique non-empty voxels: %d", unique_voxels_size);

    /* Build voxel-level CSR structure(voxel -> list of point indices). */
    thrust::device_vector<int> d_voxel_count(unique_voxels_size);
    compressed_spare_row(point_num, unique_voxels_size, d_unique_voxels, d_voxel_count);
    
    /* Construct voxel adjacency graph (CSR), each voxel connects to its 26 neighboring voxels if present. */
    int total_neighbors = 0;
    thrust::device_vector<int> d_neighbor_offset(unique_voxels_size+1);
    thrust::device_vector<int> d_neighbors(static_cast<int>(total_neighbors));
    voxel_neighbor(unique_voxels_size,
                    d_unique_voxels,
                    grid_x, grid_y, grid_z, 
                    d_neighbor_offset, 
                    d_neighbors, 
                    total_neighbors);

    /* Identify core voxels and union connected core voxels. */
    thrust::device_vector<int>      d_root_out(unique_voxels_size);
    thrust::device_vector<uint8_t>  d_is_core(unique_voxels_size);
    union_neighbor(unique_voxels_size, 
                    d_voxel_count, 
                    d_neighbor_offset, 
                    d_neighbors, 
                    d_root_out, 
                    d_is_core);

    /* Assign cluster IDs and propagate to point level(includes border voxel attachment). */
    thrust::device_vector<int> d_point_labels(point_num);
    assgin_cluster(point_num, unique_voxels_size, 
                    d_unique_voxels,
                    d_root_out, 
                    d_is_core, 
                    d_neighbor_offset, 
                    d_neighbors, 
                    d_point_labels);

    /* Collect clustered points. */
    int* d_labels_ptr = thrust::raw_pointer_cast(d_point_labels.data());
    CUDA_CHECK(cudaMemset(d_cluster_points_num,
                                            0, 
                                            sizeof(int)));
    collect_cluster<<<grid_num, block_num>>>(d_points, point_num, 
                                                d_labels_ptr, 
                                                d_cluster_points, 
                                                d_cluster_points_num);
    CUDA_CHECK(cudaMemcpy(cluster_points, 
                          d_cluster_points, 
                          point_num * POINT_DIM * sizeof(float), 
                          cudaMemcpyDeviceToHost));

    if (this->vis)
    {
        display->set_points(cluster_points, point_num);
    }

}

void DBSCAN::exclusive_scan(const std::vector<int>& in, std::vector<int>& out)
{
    int n = (int)in.size();
    out.resize(n+1);
    out[0] = 0;
    for (int i=0;i<n;i++) out[i+1] = out[i] + in[i];
}

void DBSCAN::sparse_voxel_compress(const int point_num, 
                                    thrust::device_vector<int64_t>& unique_voxels, 
                                    int &unique_voxels_size)
{
    /* Copy per-point voxel indices to a temporary array. */
    thrust::device_vector<int64_t> d_voxel_idx_list(point_num);
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_voxel_idx_list.data()), 
                                            d_voxel_idx, 
                                            point_num * sizeof(int64_t), 
                                            cudaMemcpyDeviceToDevice
                                            ));

    /* Sort voxel IDs to group identical voxels. */
    thrust::sort(d_voxel_idx_list.begin(), d_voxel_idx_list.end());

    /* Remove duplicate voxel IDs. */
    auto new_end = thrust::unique(d_voxel_idx_list.begin(), d_voxel_idx_list.end());
    int64_t non_empty_voxel_size = new_end - d_voxel_idx_list.begin(); // Number of unique voxel IDs (including possible -1).
    
    /* Copy unique voxel IDs to host (for filtering -1). */
    std::vector<int64_t> h_unique_tmp(non_empty_voxel_size);
    CUDA_CHECK(cudaMemcpy(h_unique_tmp.data(), 
                                            thrust::raw_pointer_cast(d_voxel_idx_list.data()), 
                                            non_empty_voxel_size * sizeof(int64_t), 
                                            cudaMemcpyDeviceToHost
                                            ));

    /* ------------------------------------------------------------
     * Remove sentinel voxel ID (-1).
     *
     * Voxel ID -1 corresponds to points outside the valid grid.
     * Since the array is sorted, all -1 entries appear at the front.
     * ------------------------------------------------------------ */
    int valid_start = 0;
    while (valid_start < non_empty_voxel_size && h_unique_tmp[valid_start] == -1) ++valid_start;
    int valid_size = non_empty_voxel_size - valid_start;
    if (valid_size <= 0) LOGE("No valid voxels found.");

    /* Allocate and copy valid unique voxel IDs to output. */
    thrust::device_vector<int64_t> d_unique_voxels(valid_size);
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_unique_voxels.data()), 
                                            thrust::raw_pointer_cast(d_voxel_idx_list.data()) + valid_start, 
                                            valid_size * sizeof(int64_t), 
                                            cudaMemcpyDeviceToDevice
                                            ));

    /* Output results. */
    unique_voxels       = d_unique_voxels;
    unique_voxels_size  = valid_size;
}

void DBSCAN::compressed_spare_row(const int point_num, 
                                    const int unique_voxels_size, 
                                    const thrust::device_vector<int64_t>& d_unique_voxels, 
                                    thrust::device_vector<int>& d_voxel_count)
{   
    /* Temporary device buffers for key-value sorting. */
    thrust::device_vector<int>      d_point_idx_list(point_num);            // Original point indices.
    thrust::device_vector<int64_t>  d_key_list(point_num);                  // Voxel IDs (sort key).
    thrust::device_vector<int>      d_value_list(point_num);                // Point indices (sort value).
    std::vector<int64_t>            h_voxel_list(point_num);                // Sorted voxel IDs.
    std::vector<int>                h_point_list(point_num);                // Sorted point indices.
    std::vector<int64_t>            h_unique_voxels(unique_voxels_size);    // Unique voxel ID list.

    /* Buffers for CSR construction. */
    std::vector<int>            h_voxel_count(unique_voxels_size, 0);        // Number of points per voxel(host-side).
    std::vector<int>            h_voxel_offset(unique_voxels_size+1, 0);     // CSR row offsets(host-side).
    thrust::device_vector<int>  d_voxel_offset(unique_voxels_size+1);        // CSR row offsets(GPU-side).

    /* ------------------------------------------------------------
     * Prepare key-value pairs for sorting.
     *
     *   key   : voxel ID
     *   value : point index
     * ------------------------------------------------------------ */
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_key_list.data()),
                          d_voxel_idx, 
                          point_num * sizeof(int64_t), 
                          cudaMemcpyDeviceToDevice
                          ));
    thrust::sequence(d_point_idx_list.begin(), d_point_idx_list.end());
    thrust::copy(d_point_idx_list.begin(), d_point_idx_list.end(), d_value_list.begin());

    /* ------------------------------------------------------------
     * Sort points by voxel ID.
     *
     * After sorting:
     *   points belonging to the same voxel are stored contiguously.
     * ------------------------------------------------------------ */
    thrust::sort_by_key(d_key_list.begin(), d_key_list.end(), d_value_list.begin());

    /* ------------------------------------------------------------
     * Copy sorted results to host for CSR construction.
     *
     * This host round-trip is acceptable since:
     *   number of voxels V << number of points N.
     * ------------------------------------------------------------ */
    CUDA_CHECK(cudaMemcpy(h_voxel_list.data(), 
                                            thrust::raw_pointer_cast(d_key_list.data()), 
                                            point_num * sizeof(int64_t), 
                                            cudaMemcpyDeviceToHost
                                            ));
    CUDA_CHECK(cudaMemcpy(h_point_list.data(), 
                                            thrust::raw_pointer_cast(d_value_list.data()), 
                                            point_num * sizeof(int), 
                                            cudaMemcpyDeviceToHost
                                            ));

    /* Copy unique voxel IDs to host. */
    CUDA_CHECK(cudaMemcpy(h_unique_voxels.data(), 
                                            thrust::raw_pointer_cast(d_unique_voxels.data()), 
                                            sizeof(int64_t)*unique_voxels_size,
                                            cudaMemcpyDeviceToHost
                                            ));
    
    /* ------------------------------------------------------------
     * Host-side mapping from voxel ID -> compact index.
     *
     * unique_voxels is sorted, enabling binary search.
     * ------------------------------------------------------------ */
    auto find_unique_index = [&](int64_t vid)->int {
        int l=0, r=unique_voxels_size-1;
        while (l <= r) {
            int m = (l+r)>>1;
            if (h_unique_voxels[m] == vid) return m;
            if (h_unique_voxels[m] < vid) l = m + 1;
            else r = m - 1;
        }
        return -1;
    };

     /* ------------------------------------------------------------
     * Count number of points per voxel.
     *
     * Skip invalid voxel ID (-1).
     * ------------------------------------------------------------ */
    for (int i = 0; i < point_num; ++i) {
        int64_t vid = h_voxel_list[i];      // Global voxel IDs.
        if (vid == -1) continue;            // Remove invalid voxel.
        int idx = find_unique_index(vid);   // Compact voxel IDs.
        if (idx < 0) continue;              // Shouldn't happen.
        h_voxel_count[idx] += 1;            // Unique voxel point number+1.
    }

    /* Compute CSR offsets (exclusive scan). */
    h_voxel_offset[0] = 0;
    for (int i = 0; i < unique_voxels_size; ++i) h_voxel_offset[i+1] = h_voxel_offset[i] + h_voxel_count[i];
    int total_points_in_voxels = h_voxel_offset[unique_voxels_size];
    LOGV("Total points inside mapped voxels: %d (of %d total points).", total_points_in_voxels, point_num);

    /* Copy voxel_count and voxel_offset to device. */
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_voxel_count.data()),
                                            h_voxel_count.data(), 
                                            unique_voxels_size * sizeof(int), 
                                            cudaMemcpyHostToDevice
                                            ));
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_voxel_offset.data()), 
                                            h_voxel_offset.data(), 
                                            (unique_voxels_size+1) * sizeof(int),
                                            cudaMemcpyHostToDevice
                                            ));


    /* ------------------------------------------------------------
     * Deprecated
     * 
     * Build compact point list (CSR column indices).
     *
     * Each voxel occupies a contiguous segment in point_list
     * ------------------------------------------------------------ */
    // thrust::device_vector<int>  d_pointlist_list(total_points_in_voxels);
    // std::vector<int>            h_pointlist_list(total_points_in_voxels);
    // std::vector<int>            cursor(unique_voxels_size,0);

    // for (int i=0;i<point_num;i++){
    //     int64_t vid = h_voxel_list[i];                  // Global voxel IDs.
    //     if (vid == -1) continue;                        // Remove invalid voxel.
    //     int idx = find_unique_index(vid);               // Compact voxel IDs.
    //     if (idx < 0) continue;                          // Shouldn't happen.
    //     int pos = h_voxel_offset[idx] + cursor[idx];    // final_pos = start_pos + cursor
    //     h_pointlist_list[pos] = h_pointlist_list[i];
    //     cursor[idx] ++;
    // }

    // /* Copy CSR point list to device. */
    // CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_pointlist_list.data()), 
    //                                         h_pointlist_list.data(), 
    //                                         total_points_in_voxels * sizeof(int), 
    //                                         cudaMemcpyHostToDevice));

}

void DBSCAN::voxel_neighbor(const int unique_voxels_size, 
                            const thrust::device_vector<int64_t>& d_unique_voxels, 
                            const int grid_x, const int grid_y, const int grid_z,
                            thrust::device_vector<int>& d_neighbor_offset, 
                            thrust::device_vector<int>& d_neighbors, 
                            int& total_neighbors)
{
    int bthreads    = 256;
    int bblocks     = (unique_voxels_size + bthreads - 1) / bthreads;
    
    /* ------------------------------------------------------------
     *  Upper-bound neighbor count per voxel
     * ------------------------------------------------------------
     *  Each voxel can have at most 27 neighbors (3×3×3).
     *  This kernel only counts geometric possibilities and
     *  does NOT check whether neighbors actually exist.
     */
    thrust::device_vector<int> d_neighbor_upper(unique_voxels_size);
    count_voxel_neighbors<<<bblocks,bthreads>>>(
        thrust::raw_pointer_cast(d_unique_voxels.data()),
        unique_voxels_size,
        grid_x, grid_y, grid_z,
        thrust::raw_pointer_cast(d_neighbor_upper.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ------------------------------------------------------------
     *  Temporary fixed-size neighbor buffer (V × 27)
     * ------------------------------------------------------------
     *  We conservatively allocate space for all possible neighbors.
     *  Each voxel is assigned a fixed chunk of 27 entries.
     */
    int upper_total = unique_voxels_size * 27;                                  // Upper number of neighbor of all non-empty voxels (V × 27).
    thrust::device_vector<int>  d_neighbors_tmp(upper_total);                   // Store neighbor indices (may have gaps).
    thrust::device_vector<int>  d_neighbor_offset_tmp(unique_voxels_size+1);    // Device-side neighbor offset, fill offsets tmp: each vid has 27 slots.
    std::vector<int>            h_neighbor_offset_tmp(unique_voxels_size+1);    // Host-side neighbor offset, fill offsets tmp: each vid has 27 slots.
    for (int i=0;i<unique_voxels_size;i++) h_neighbor_offset_tmp[i] = i * 27;   // Initialize neighbor offset: [0, 27, 54, 81, ..., (unique_voxels_size-1)*27].
    h_neighbor_offset_tmp[unique_voxels_size] = upper_total;                    // Initialize neighbor offset: [0, 27, 54, 81, ..., (unique_voxels_size-1)*27, upper_total].
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_neighbor_offset_tmp.data()), 
                                            h_neighbor_offset_tmp.data(), 
                                            (unique_voxels_size+1) * sizeof(int), 
                                            cudaMemcpyHostToDevice
                                            ));

    /* ------------------------------------------------------------
     *  Fill temporary neighbor list (with gaps)
     * ------------------------------------------------------------
     *  For each voxel:
     *      - enumerate its 27 neighboring grid locations
     *      - check whether that voxel exists via binary search
     *      - write valid neighbors into its 27-slot chunk
     *
     *  Invalid entries remain as garbage / unused.
     */
    fill_voxel_neighbors<<<bblocks,bthreads>>>(
        thrust::raw_pointer_cast(d_unique_voxels.data()),
        unique_voxels_size,
        grid_x, grid_y, grid_z,
        thrust::raw_pointer_cast(d_neighbor_offset_tmp.data()),
        thrust::raw_pointer_cast(d_neighbors_tmp.data()));
    CUDA_CHECK(cudaDeviceSynchronize());


    
    /* ------------------------------------------------------------
     *  Compute exact neighbor count per voxel (host)
     * ------------------------------------------------------------
     *  We scan each 27-slot chunk and count only valid neighbors.
     *  The result overwrites d_neighbor_upper.
     *
     *  NOTE:
     *      This is done on the host for simplicity and debugging
     *      clarity. A GPU kernel can replace this in production.
     */
    auto compute_exact_neighbor_counts = [&](){
        std::vector<int> h_tmp_neighbors(upper_total);      // Host-side store neighbor indices (may have gaps).
        std::vector<int> h_exact_cnt(unique_voxels_size);   // Host-side 
        CUDA_CHECK(cudaMemcpy(h_tmp_neighbors.data(), 
                                                thrust::raw_pointer_cast(d_neighbors_tmp.data()), 
                                                upper_total * sizeof(int), 
                                                cudaMemcpyDeviceToHost
                                                ));
        for (int vid=0; vid<unique_voxels_size; ++vid) {
            int st = h_neighbor_offset_tmp[vid];
            int cnt = 0;
            for (int j=0;j<27;++j) {
                int v = h_tmp_neighbors[st+j];
                if (v >= 0 && v < unique_voxels_size) cnt++;
            }
            h_exact_cnt[vid] = cnt;
        }
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_neighbor_upper.data()),
                                                h_exact_cnt.data(), 
                                                unique_voxels_size * sizeof(int), 
                                                cudaMemcpyHostToDevice
                                                ));
    };

    compute_exact_neighbor_counts();

    /* ------------------------------------------------------------
     *  Exclusive scan → CSR neighbor offsets
     * ------------------------------------------------------------
     *  neighbor_offset[v+1] - neighbor_offset[v]
     *      = number of neighbors of voxel v
     */
    thrust::fill(d_neighbor_offset.begin(), d_neighbor_offset.begin() + 1, 0);
    
    thrust::exclusive_scan(
        d_neighbor_upper.begin(),
        d_neighbor_upper.end(), 
        d_neighbor_offset.begin()+1
    );
    
    /* ------------------------------------------------------------
     *  Fetch total number of neighbors
     * ------------------------------------------------------------ */
    CUDA_CHECK(cudaMemcpy(&total_neighbors, 
                                            thrust::raw_pointer_cast(d_neighbor_offset.data()) + unique_voxels_size, 
                                            sizeof(int), 
                                            cudaMemcpyDeviceToHost));
    LOGV("DBSCAN voxel neighbor construction: exact = %d.", total_neighbors);

    /* ------------------------------------------------------------
     *  Allocate final compact neighbor list
     * ------------------------------------------------------------ */
    d_neighbors.resize(total_neighbors);

    /* ------------------------------------------------------------
     *  Final CSR neighbor fill (compact)
     * ------------------------------------------------------------
     *  Same kernel as before, but now:
     *      - neighbor_offset is exact
     *      - neighbors are written contiguously
     */
    fill_voxel_neighbors<<<bblocks,bthreads>>>(
        thrust::raw_pointer_cast(d_unique_voxels.data()),
        unique_voxels_size,
        grid_x, grid_y, grid_z,
        thrust::raw_pointer_cast(d_neighbor_offset.data()),
        thrust::raw_pointer_cast(d_neighbors.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DBSCAN::union_neighbor(const int unique_voxels_size, 
                            const thrust::device_vector<int>& d_voxel_count, 
                            const thrust::device_vector<int>& d_neighbor_offset, 
                            const thrust::device_vector<int>& d_neighbors,
                            thrust::device_vector<int>& d_root_out, 
                            thrust::device_vector<uint8_t>& d_is_core)
{
    /* ------------------------------------------------------------
     * Voxel-level DBSCAN using Union-Find (Disjoint Set Union, DSU)
     *
     * Each voxel is treated as a node in a graph.
     * Edges are defined between spatially adjacent voxels
     * (constructed previously in CSR format).
     *
     * Core voxels (voxel_count >= minPts) are connected via union operations.
     * After union, connected components correspond to DBSCAN clusters
     * at the voxel level.
     * ------------------------------------------------------------ */
    int bthreads    = 256;
    int bblocks     = (unique_voxels_size + bthreads - 1) / bthreads;
    
    /* ------------------------------------------------------------
     * Core voxel identification
     * ------------------------------------------------------------
     * 
     * d_voxel_count[v]  : number of points inside voxel v
     * DBSCAN_MIN_PTS    : minPts parameter of DBSCAN
     * 
     * A voxel is marked as a "core voxel" if it contains at least DBSCAN_MIN_PTS points.
     * d_is_core[v] = 1 if voxel v is core, otherwise 0.
     *
     * This corresponds to the core-point definition in classical DBSCAN.
     * ------------------------------------------------------------ */
    voxel_core_mark<<<bblocks,bthreads>>>(
        thrust::raw_pointer_cast(d_voxel_count.data()), 
        unique_voxels_size, 
        this->dbscan_min_pts, 
        thrust::raw_pointer_cast(d_is_core.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ------------------------------------------------------------
     * Initialize Union-Find parent array
     * ------------------------------------------------------------
     * 
     * Initially, each voxel is its own parent (singleton set).
     *
     * parent[v] = v  for all v
     * ------------------------------------------------------------ */
    thrust::device_vector<int>  d_parent(unique_voxels_size);       // Device-side stores the parent of voxel v in the disjoint-set structure.
    std::vector<int>            h_parent(unique_voxels_size);       // Host-side stores the parent of voxel v in the disjoint-set structure.
    for (int i=0;i<unique_voxels_size;i++) h_parent[i] = i;         // Initialize parent as [0, 1, 2, ..., unique_voxels_size-1].
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_parent.data()), 
                                            h_parent.data(), 
                                            unique_voxels_size * sizeof(int), 
                                            cudaMemcpyHostToDevice));

    /* ------------------------------------------------------------
     * Union core voxel neighbors
     * ------------------------------------------------------------
     * 
     * For each voxel v:
     *      - If v is not a core voxel, it is skipped.
     *      - Otherwise, iterate over its neighbors using CSR:
     *
     *          neighbors are in:
     *          d_neighbors[d_neighbor_offset[v] ... d_neighbor_offset[v+1] - 1]
     *
     *      - If neighbor u is also a core voxel, perform union(v, u).
     *
     *
     *  This step merges spatially adjacent core voxels into the same
     *  connected component, corresponding to density-reachable regions
     *  in DBSCAN.
     *
     *  Only core-core connections are unioned at this stage.
     *  Border voxels will be attached later during point-level expansion.
     * ------------------------------------------------------------ */
    union_core_neighbors<<<bblocks,bthreads>>>(
        thrust::raw_pointer_cast(d_is_core.data()), 
        unique_voxels_size,
        thrust::raw_pointer_cast(d_neighbor_offset.data()),
        thrust::raw_pointer_cast(d_neighbors.data()),
        thrust::raw_pointer_cast(d_parent.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    
    /* ------------------------------------------------------------
     *  Path compression (find root of each voxel)
     * ------------------------------------------------------------
     * 
     * For each voxel v, we find its representative root in the
     * union-find structure and apply path compression.
     *      - Otherwise, iterate over its neighbors using CSR:
     *
     * d_root_out[v] = root index of voxel v
     *
     * After this step, all voxels belonging to the same connected
     * component share the same root.
     * ------------------------------------------------------------ */
    compress_roots<<<bblocks,bthreads>>>(
        unique_voxels_size, 
        thrust::raw_pointer_cast(d_parent.data()), 
        thrust::raw_pointer_cast(d_root_out.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DBSCAN::assgin_cluster(const int point_num, const int unique_voxels_size,
                            const thrust::device_vector<int64_t>& d_unique_voxels, 
                            const thrust::device_vector<int>& d_root_out, 
                            const thrust::device_vector<uint8_t>& d_is_core,
                            const thrust::device_vector<int>& d_neighbor_offset, 
                            const thrust::device_vector<int>& d_neighbors,
                            thrust::device_vector<int> &d_point_labels)
{   
    int threads         = 1024;
    int blocks_points   = (point_num + threads - 1) / threads;
    int bthreads        = 256;
    int bblocks         = (unique_voxels_size + bthreads - 1) / bthreads;

    thrust::device_vector<int>      d_cluster_id_of_root(unique_voxels_size);
    thrust::device_vector<int>      d_next_cluster_id(1);
    thrust::fill(d_cluster_id_of_root.begin(), d_cluster_id_of_root.end(), -1);
    thrust::fill(d_next_cluster_id.begin(), d_next_cluster_id.end(), 0);

    /**
     * Assign cluster ids to union-find roots(root -> cluster id).
     *
     * Only roots corresponding to core voxels receive a new cluster id.
     * Non-core roots (noise) keep cluster_id = -1.
     *
     * This ensures that:
     *   - Each cluster corresponds to one connected component of core voxels.
     *   - Border voxels inherit the cluster id of their connected core root.
     *   - Noise voxels never form clusters.
     */
    map_root_to_clusterid<<<bblocks,bthreads>>>(unique_voxels_size, 
                                                thrust::raw_pointer_cast(d_root_out.data()),
                                                thrust::raw_pointer_cast(d_is_core.data()),
                                                thrust::raw_pointer_cast(d_cluster_id_of_root.data()),
                                                thrust::raw_pointer_cast(d_next_cluster_id.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

     /**
     * Assign a cluster id to each voxel based on its union-find root(voxel -> cluster id).
     *
     * For voxel v:
     *   voxel_cluster_id[v] = cluster_id_of_root[ root_out[v] ]
     *
     * Result:
     *   - Core voxels      -> valid cluster id (>= 0)
     *   - Border voxels    -> valid cluster id (inherited)
     *   - Noise voxels     -> -1
     */
    thrust::device_vector<int> d_voxel_cluster_id(unique_voxels_size);
    assign_voxel_cluster<<<bblocks,bthreads>>>(unique_voxels_size, 
                                                thrust::raw_pointer_cast(d_root_out.data()),
                                                thrust::raw_pointer_cast(d_cluster_id_of_root.data()),
                                                thrust::raw_pointer_cast(d_voxel_cluster_id.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    /**
    * Attach border voxels to neighboring core voxel clusters (DBSCAN border handling).
    *
    * Semantic:
    *  - core voxel: already has voxel_cluster_id >= 0
    *  - border voxel: is_core == 0 AND has at least one core neighbor
    *  - noise voxel: is_core == 0 AND no core neighbors
    *
    * After this kernel:
    *  - border voxels will have voxel_cluster_id >= 0
    *  - noise voxels remain -1
    */
    attach_border_voxels<<<bblocks, bthreads>>>(unique_voxels_size,
                                                thrust::raw_pointer_cast(d_is_core.data()),
                                                thrust::raw_pointer_cast(d_neighbor_offset.data()),
                                                thrust::raw_pointer_cast(d_neighbors.data()),
                                                thrust::raw_pointer_cast(d_voxel_cluster_id.data()));

    /**
     * Map voxel-level cluster ids back to point-level labels(voxel -> point).
     *
     * For each point i:
     *   1. Obtain its voxel id from d_voxel_idx[i].
     *   2. Locate the voxel index in d_unique_voxels.
     *   3. Assign the corresponding voxel cluster id to the point.
     *
     * Final labeling:
     *   - Core and border points -> valid cluster id
     *   - Noise points           -> -1
     */
    map_voxel_cluster_to_points<<<blocks_points, threads>>>(d_voxel_idx, point_num,
                                                            thrust::raw_pointer_cast(d_unique_voxels.data()), unique_voxels_size,
                                                            thrust::raw_pointer_cast(d_voxel_cluster_id.data()),
                                                            thrust::raw_pointer_cast(d_point_labels.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DBSCAN::cluster_distribution(const std::string name, const std::vector<int> labels, const int point_num)
{
    /*
     * Count the number of points for each cluster label.
     *
     * The label value corresponds to:
     *   - cluster ID (>= 0) for valid clusters
     *   - -1 for noise / unassigned points
     */
    std::unordered_map<int, int> label_count;
    /* Reserve initial capacity to reduce rehash overhead.
     * The value can be tuned depending on expected number of clusters.
     */
    label_count.reserve(128);

     /* Accumulate point counts per label. */
    for (int i = 0; i < point_num; ++i) {
        int lb = labels[i];
        label_count[lb] += 1;
    }
    
    /*
     * Convert the hash map to a vector of (label, count) pairs
     * for sorting and ordered printing.
     */
    std::vector<std::pair<int,int>> stats;
    stats.reserve(label_count.size());

    for (auto &kv : label_count) {
        stats.emplace_back(kv.first, kv.second);
    }

    /*
     * Sort the statistics by cluster label in ascending order.
     *
     * Alternative strategies:
     *   - sort by kv.second (descending) to show largest clusters first
     *   - filter out label == -1 to ignore noise points
     */
    std::sort(stats.begin(), stats.end(),
        [](const std::pair<int,int> &a, const std::pair<int,int> &b){
            return a.first < b.first;   // Sort by cluster ID.
        });
        
    LOGV("%s cluster distribution:", name.c_str());
    for (auto &kv : stats) {
        LOGV("Label %d : %d points.", kv.first, kv.second);
    }
}

DBSCAN::~DBSCAN()
{
}

}