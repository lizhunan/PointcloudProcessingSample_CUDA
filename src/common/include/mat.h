#ifndef __MAT_H__
#define __MAT_H__

#include <math.h>
#include "logger.h"
#include "/workspace/src/display/include/display.h"
#include "cuda_base.h"

#define POINT_DIM 6     // Point dimension: [x, y, z, intensity, index, label]

namespace mat {

/**
 * @brief Compute Euclidean distance between two 3D points.
 *
 * @param p1 Pointer to first point (size >= 3)
 * @param p2 Pointer to second point (size >= 3)
 * @return Distance between p1 and p2
 */
__device__ inline float distf(const float* p1, const float* p2)
{
    return sqrtf(powf(p1[0] - p2[0], 2) + powf(p1[1] - p2[1], 2) + powf(p1[2] - p2[2], 2));
}

/**
 * @brief Compute dot product of two 3D vectors.
 *
 * @details
 *  This function calculates the inner product between two 3D vectors:
 *
 *      dot(a, b) = a_x * b_x + a_y * b_y + a_z * b_z
 *
 *  Mathematical formulation:
 *
 *      Given:
 *          a = (a_x, a_y, a_z)
 *          b = (b_x, b_y, b_z)
 *
 *      Then:
 *          dot(a, b) = a_x b_x + a_y b_y + a_z b_z
 *
 *  Geometric interpretation:
 *      - Measures similarity between two vectors
 *      - If vectors are normalized:
 *
 *            dot(a, b) = cos(θ)
 *
 *        where θ is the angle between a and b
 *
 *      - Range (for unit vectors):
 *            [-1, 1]
 *
 * @param a Pointer to first 3D vector (size = 3)
 * @param b Pointer to second 3D vector (size = 3)
 *
 * @return Dot product (scalar)
 */
__device__ inline float dot3f(const float* a, const float* b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * @brief Compute cross product of two 3D vectors
 * 
 * @details
 *   Computes the cross product c = a × b for 3D vectors.
 * 
 *   Mathematical formulation:
 *     c = a × b = |i   j   k  |
 *                 |a.x a.y a.z|
 *                 |b.x b.y b.z|
 *   
 *     c.x = a.y * b.z - a.z * b.y
 *     c.y = a.z * b.x - a.x * b.z
 *     c.z = a.x * b.y - a.y * b.x
 * 
 * 
 * @param a       Input vector a (3 components)
 * @param b       Input vector b (3 components)
 * @param result  Output vector c = a × b (3 components)
 */
__device__ inline void cross_product3(const float* a, const float* b, float* result)
{

    // Compute cross product components
    result[0] = a[1] * b[2] - a[2] * b[1];  // c.x = a.y * b.z - a.z * b.y
    result[1] = a[2] * b[0] - a[0] * b[2];  // c.y = a.z * b.x - a.x * b.z
    result[2] = a[0] * b[1] - a[1] * b[0];  // c.z = a.x * b.y - a.y * b.x
}

/**
 * @brief Compute weighted covariance matrix for a given keypoint.
 *
 * @details
 *  This function constructs a **weighted covariance matrix** of the local neighborhood
 *  for a given point p_k. It is commonly used in:
 *
 *  - Local Reference Frame (LRF) estimation
 *  - SHOT / ISS feature computation
 *  - PCA-based geometric analysis
 *
 *  Mathematical formulation:
 *
 *      Let p_k be the keypoint and p_i its neighbors.
 *
 *      d_i = p_i - p_k
 *      w_i = | ||p_i - p_k|| - R |
 *
 *      Then the covariance matrix is:
 *
 *          M = (1 / Σ w_i) Σ w_i * (d_i * d_i^T)
 *
 *  where:
 *      - R is the support radius
 *      - w_i is the radial weight
 *
 *  Geometric interpretation:
 *      - Captures local surface structure
 *      - Eigenvectors → principal directions
 *      - Smallest eigenvalue → surface normal
 *
 *  Notes:
 *      - Neighbor index layout: [self, n1, n2, ..., nk]
 *      - The first entry (self) is skipped
 *      - Weight emphasizes points near the boundary (depending on formulation)
 *
 * @param points            Pointer to input pointcloud array [N x POINT_DIM]
 * @param p_id              Index of the current keypoint
 * @param points_num        Total number of points
 * @param neighbor_indices  Neighbor index array [N x (k+1)]
 * @param k                 Number of neighbors
 * @param radius            Support radius for weighting
 * @param M                 Output 3x3 covariance matrix (accumulated)
 *
 * @return true  If covariance matrix is valid (norm > threshold)
 * @return false If degenerate (insufficient or zero weights)
 */
__device__ inline bool conv_weight(const float* points, const int p_id, const int points_num, 
                                    const int* neighbor_indices, const int k, const int radius, float M[3][3])
{
    // Load keypoint coordinates p_k
    float point[3] = {points[p_id*POINT_DIM+0], points[p_id*POINT_DIM+1], points[p_id*POINT_DIM+2]};

    // Neighbor indexing
    // Layout:
    //   neighbor_indices[p_id * (k+1) + 0]     → self index
    //   neighbor_indices[p_id * (k+1) + i]     → i-th neighbor
    int stride = k + 1;
    int base   = p_id * stride; // Current point index in neighbor_indices

    // Iterate over k neighbors (skip self at index 0)
    float norm = 0.0f;  // Normalization term Σ w_i
    for (int i = 1; i <= k; i++)
    {
        int idx = neighbor_indices[base + i];

        float neighbor[3]   = {points[idx * POINT_DIM + 0], points[idx * POINT_DIM + 1], points[idx * POINT_DIM + 2]};  // Load neighbor point p_i
        float dist          = mat::distf(point, neighbor);                                                              // Euclidean distance: dist = ||p_i - p_k||
        float weight        = fabs(dist - radius);                                                                      // Radial weight function: w_i = |dist - radius|
        norm                += weight;

        // Relative coordinate d_i = p_i - p_k
        float dx = points[idx * POINT_DIM + 0] - points[p_id*POINT_DIM+0];
        float dy = points[idx * POINT_DIM + 1] - points[p_id*POINT_DIM+1];
        float dz = points[idx * POINT_DIM + 2] - points[p_id*POINT_DIM+2];

        /**
         * Accumulate weighted outer product
         *
         * M += w_i * (d_i * d_i^T)
         *
         * Expanded form:
         *  M_xx += w_i * dx * dx
         *  M_xy += w_i * dx * dy
         *  ...
         */
        M[0][0] += weight*dx*dx; M[0][1] += weight*dx*dy; M[0][2] += weight*dx*dz;
        M[1][0] += weight*dy*dx; M[1][1] += weight*dy*dy; M[1][2] += weight*dy*dz;
        M[2][0] += weight*dz*dx; M[2][1] += weight*dz*dy; M[2][2] += weight*dz*dz;
    }
   
    /**
     * Normalize covariance matrix
     *
     * M = M / Σ w_i
     *
     * Prevent division by zero (degenerate neighborhood)
     */
    if (norm > 1e-6f)
    {
        M[0][0] /= norm; M[0][1] /= norm; M[0][2] /= norm; 
        M[1][0] /= norm; M[1][1] /= norm; M[1][2] /= norm; 
        M[2][0] /= norm; M[2][1] /= norm; M[2][2] /= norm; 
        return true;
    }else
    {
        /**
         * Degenerate case:
         *  - No valid neighbors
         *  - All weights are zero
         */
        return false;
    }
}

/**
 * @brief Compute covariance matrix for a given keypoint (unweighted version).
 *
 * @details
 *  This function constructs a **standard covariance matrix** of the local neighborhood
 *  for a given point p_k. Unlike conv_weight(), this version does NOT use radial weights.
 *
 *  Mathematical formulation:
 *
 *      1. Compute centroid C = (1/(k+1)) * (p_k + Σ p_i)
 *
 *      2. Compute covariance matrix M = (1/(k+1)) * Σ (p_j - C) * (p_j - C)^T
 *         where j runs over all points (keypoint + k neighbors)
 *
 *
 *  Notes:
 *      - Neighbor index layout: [self, n1, n2, ..., nk]
 *      - The first entry (self) is skipped in neighbor loop but added separately
 *      - Covariance is normalized by (k+1) points total
 *
 * @param points            Pointer to input pointcloud array [N x POINT_DIM]
 * @param p_id              Index of the current keypoint
 * @param points_num        Total number of points (unused in this version)
 * @param neighbor_indices  Neighbor index array [N x (k+1)]
 * @param k                 Number of neighbors (excluding self)
 * @param M                 Output 3x3 covariance matrix (accumulated)
 */
__device__ inline void cov_mat(const float* points, const int p_id, const int points_num, 
                                const int* neighbor_indices, const int k, float M[3][3])
{
    // Load keypoint coordinates p_k
    float point[3] = {points[p_id*POINT_DIM+0], points[p_id*POINT_DIM+1], points[p_id*POINT_DIM+2]};

    // Neighbor indexing
    // Layout:
    //   neighbor_indices[p_id * (k+1) + 0]     → self index
    //   neighbor_indices[p_id * (k+1) + i]     → i-th neighbor
    int stride = k + 1;
    int base   = p_id * stride; // Current point index in neighbor_indices

    // Centroid
    // Sum all neighbor coordinates (skip self at index 0)
    float cx = 0.f, cy = 0.f, cz = 0.f;
    for (int i = 1; i <= k; i++)
    {
        int idx = neighbor_indices[base + i];

        cx += points[idx * POINT_DIM + 0];
        cy += points[idx * POINT_DIM + 1];
        cz += points[idx * POINT_DIM + 2];
    }

    cx += point[0]; cy += point[1]; cz += point[2]; // Add keypoint itself to the sum
    cx /= (k+1); cy /= (k+1); cz /= (k+1);          // Divide by total number of points (k+1) to get centroid

    // Covariance
    // Accumulate contributions from all neighbor points (i = 1 to k)
    for (int i = 1; i <= k; i++)
    {
        int idx = neighbor_indices[base + i];

        // Relative coordinates from centroid
        float x = points[idx * POINT_DIM + 0] - cx;
        float y = points[idx * POINT_DIM + 1] - cy;
        float z = points[idx * POINT_DIM + 2] - cz;

        /**
         * Accumulate outer product (x,y,z) * (x,y,z)^T
         * 
         * Symmetric matrix, only upper triangular shown:
         *   M[0][0] = Σ x*x    (variance along X)
         *   M[0][1] = Σ x*y    (covariance XY)
         *   M[0][2] = Σ x*z    (covariance XZ)
         *   M[1][1] = Σ y*y    (variance along Y)
         *   M[1][2] = Σ y*z    (covariance YZ)
         *   M[2][2] = Σ z*z    (variance along Z)
         */
        M[0][0] += x*x; M[0][1] += x*y; M[0][2] += x*z;
        M[1][0] += y*x; M[1][1] += y*y; M[1][2] += y*z;
        M[2][0] += z*x; M[2][1] += z*y; M[2][2] += z*z;
    }

    // Add contribution from the keypoint itself
    float x_p = point[0] - cx;
    float y_p = point[1] - cy;
    float z_p = point[2] - cz; 
    M[0][0] += x_p*x_p; M[0][1] += x_p*y_p; M[0][2] += x_p*z_p;
    M[1][0] += y_p*x_p; M[1][1] += y_p*y_p; M[1][2] += y_p*z_p;
    M[2][0] += z_p*x_p; M[2][1] += z_p*y_p; M[2][2] += z_p*z_p;

    // Final covariance matrix = (1/(k+1)) * Σ (p_j - C)(p_j - C)^T
    M[0][0] = M[0][0]/(k+1); M[0][1] = M[0][1]/(k+1); M[0][2] = M[0][2]/(k+1);
    M[1][0] = M[1][0]/(k+1); M[1][1] = M[1][1]/(k+1); M[1][2] = M[1][2]/(k+1);
    M[2][0] = M[2][0]/(k+1); M[2][1] = M[2][1]/(k+1); M[2][2] = M[2][2]/(k+1);
}

/**
 * @brief Compute bin index and linear interpolation weights.
 *
 * @details
 *  This function maps a scalar value into histogram bins with linear interpolation.
 *
 *  Given:
 *      bin_value   = value / step
 *      bin_idx     = floor(v)
 *
 *  Then:
 *      lower bin: i       with weight (1 - (v - i))
 *      upper bin: i + 1   with weight (v - i)
 *
 *  Boundary handling:
 *      - If value < 0        → assign to bin 0
 *      - If value >= max_val → assign to last bin
 *
 *  Output:
 *      Up to 2 bins with corresponding weights
 *
 * @param value     Input value
 * @param step      Bin width
 * @param max_val   Maximum value range
 * @param bins      Output bin indices (size 2)
 * @param weights   Output weights (size 2)
 * @param count     Number of valid bins (1 or 2)
 */
__device__ inline void binning_weight(const float value, const float step, const float max_val, int bins[2], float weights[2], int& count)
{
    float bin_value = value/step;
    int bin_idx     = (int)floorf(bin_value);
    int max_bin     = (int)(max_val/step);
    float w_upper   = bin_value - bin_idx;
    float w_lower   = 1.0f - w_upper;

    // Boundary: below range
    if (bin_idx < 0)
    {
        bins[0]     = 0;
        weights[0]  = 1.0f;
        count       = 1;
        return;
    }

    // Boundary: above range
    if (bin_idx >= max_bin)
    {
        bins[0]     = max_bin - 1;
        weights[0]  = 1.0f;
        count       = 1;
        return;
    }

    // Linear interpolation
    bins[0]     = bin_idx;
    weights[0]  = w_lower;
    bins[1]     = bin_idx+1;
    weights[1]  = w_upper;
    count       = 2;          
}

/**
 * @brief Jacobi eigen decomposition for 3x3 symmetric matrix (CUDA device).
 *
 * @details
 *  - Input: symmetric matrix A
 *  - Output:
 *      eigenvalues  -> diagonal elements
 *      eigenvectors -> column vectors
 *
 *  - Iterative orthogonal rotation
 *  - Typically converges in < 10 iterations for 3x3
 *
 * @param A         Input covariance matrix (modified in-place)
 * @param eigval    Output eigenvalues (size 3)
 * @param eigvec    Output eigenvectors (3x3, column-major)
 */
__device__ inline void jacobi_3x3(float A[3][3], float eigval[3], float eigvec[3][3])
{
    // Initialize the eigenvector matrix to an identity matrix
    eigvec[0][0]=1; eigvec[0][1]=0; eigvec[0][2]=0;
    eigvec[1][0]=0; eigvec[1][1]=1; eigvec[1][2]=0;
    eigvec[2][0]=0; eigvec[2][1]=0; eigvec[2][2]=1;

    // Jacobi iteration
    for (int iter = 0; iter < 10; iter++)
    {
        // Find the largest non diagonal element
        int p = 0, q = 1;
        float max = fabsf(A[0][1]);

        if (fabsf(A[0][2]) > max) {
            p = 0; q = 2;
            max = fabsf(A[0][2]);
        }

        if (fabsf(A[1][2]) > max) {
            p = 1; q = 2;
            max = fabsf(A[1][2]);
        }

        // Check converges
        if (max < 1e-6f) break;

        float app = A[p][p];
        float aqq = A[q][q];
        float apq = A[p][q];

        // Compute rotation
        float phi = 0.5f * atan2f(2.0f * apq, (aqq - app));

        float c = cosf(phi);
        float s = sinf(phi);

        // Update matrix A
        for (int i = 0; i < 3; i++)
        {
            float aip = A[i][p];
            float aiq = A[i][q];

            A[i][p] = c * aip - s * aiq;
            A[i][q] = s * aip + c * aiq;
        }

        for (int i = 0; i < 3; i++)
        {
            float api = A[p][i];
            float aqi = A[q][i];

            A[p][i] = c * api - s * aqi;
            A[q][i] = s * api + c * aqi;
        }

        // Enforced symmetry (numerical correction), to prevent floating-point errors from disrupting symmetry
        A[p][q] = 0.0f;
        A[q][p] = 0.0f;

        // Update eigvec
        for (int i = 0; i < 3; i++)
        {
            float vip = eigvec[i][p];
            float viq = eigvec[i][q];

            eigvec[i][p] = c * vip - s * viq;
            eigvec[i][q] = s * vip + c * viq;
        }
    }

    eigval[0] = A[0][0];
    eigval[1] = A[1][1];
    eigval[2] = A[2][2];
}

/**
 * @brief Compute PCA (covariance + eigen decomposition).
 *
 * @details
 *  - Each thread processes one point
 *  - Computes covariance matrix from neighbors
 *  - Outputs eigenvalues and eigenvectors
 *
 * @param points            Input points
 * @param neighbor_indices  Neighbors indices
 * @param N                 Number of points
 * @param k                 Number of neighbors
 * @param eigenvalues       Output eigenvalues [N * 3]
 * @param eigenvectors      Output eigenvectors [N * 9]
 */
__device__ inline void pca(const float* points, const int points_num, const int* neighbor_indices, const int k,
                            float* eigenvalues, float* eigenvectors)
{
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= points_num) return;

    int stride = k + 1;
    int base   = threadid * stride; // Current point index in neighbor_indices

    // Centroid
    float cx = 0.f, cy = 0.f, cz = 0.f;
    for (int i = 1; i <= k; i++)
    {
        int idx = neighbor_indices[base + i];

        cx += points[idx * POINT_DIM + 0];
        cy += points[idx * POINT_DIM + 1];
        cz += points[idx * POINT_DIM + 2];
    }
    cx /= k; cy /= k; cz /= k;

    // Covariance
    float C[3][3] = {0};
    for (int i = 1; i <= k; i++)
    {
        int idx = neighbor_indices[base + i];

        float x = points[idx * POINT_DIM + 0] - cx;
        float y = points[idx * POINT_DIM + 1] - cy;
        float z = points[idx * POINT_DIM + 2] - cz;

        C[0][0] += x*x; C[0][1] += x*y; C[0][2] += x*z;
        C[1][0] += y*x; C[1][1] += y*y; C[1][2] += y*z;
        C[2][0] += z*x; C[2][1] += z*y; C[2][2] += z*z;
    }

    // Eigen decomposition
    float eigval[3];
    float eigvec[3][3];

    jacobi_3x3(C, eigval, eigvec);

    // Store
    for (int i = 0; i < 3; i++)
    {
        eigenvalues[threadid * 3 + i] = eigval[i];

        eigenvectors[threadid * 9 + i*3 + 0] = eigvec[i][0];
        eigenvectors[threadid * 9 + i*3 + 1] = eigvec[i][1];
        eigenvectors[threadid * 9 + i*3 + 2] = eigvec[i][2];
    }
}

class MAT {

public:

    /**
     * @class MAT
     * @brief GPU-based math utilities for point cloud processing.
     *
     * @details
     *  - Provides PCA computation on GPU
     *  - Computes normals using eigen decomposition
     *  - Supports optional visualization
     */
    MAT();

    /**
     * @class MAT
     * @brief GPU-based math utilities for point cloud processing with visualization option.
     *
     * @details
     *  - Provides PCA computation on GPU
     *  - Computes normals using eigen decomposition
     *  - Supports optional visualization
     *  @param vis Enable visualization
     */
    MAT(bool vis);

    /**
     * @brief Destructor.
     */
    ~MAT();

public:

    /**
     * @brief Estimate normals for input point cloud.
     *
     * @details
     *  - Uploads data to GPU
     *  - Launches PCA + normal extraction kernel
     *  - Downloads normals to host
     *  - Optionally visualizes results
     *
     * @param h_points      Host point cloud
     * @param points_num    Number of points
     * @param h_neighbors   Neighbor indices
     * @param k             Number of neighbors
     * @param normals       Output normals (host)
     */
    void normals_estimator(const float* h_points, const int points_num, const int* h_neighbors, const int k, float* normals);

    private:
    std::shared_ptr<display::Display>   display;        // Visualization module

private:
    bool vis = false;   // Enable visualization

    /* GPU memory */
    float*      d_points;           // Device point cloud
    int*        d_neighbors;        // Device neighbor indices
    float*      d_eigenvalues;      // Eigenvalues [N * 3]
    float*      d_eigenvectors;     // Eigenvectors [N * 9]
    float*      d_normals;          // Normals [N * 3]

};

}

#endif