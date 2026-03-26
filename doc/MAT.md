# MAT: GPU-based Mathematical Library for PointCloud Processing

## 🎯 Overview

This library provides GPU-accelerated mathematical utilities for pointcloud processing, with a primary focus on Principal Component Analysis (PCA) and surface normal estimation. Designed for real-time applications in robotics, 3D perception, and geometric feature extraction, the implementation leverages CUDA parallelism to achieve high-throughput processing of large-scale pointcloud.

## 📦 Mathematical Foundations

### Point Representation

Let the point loud be represented as a set of $N$ points in a 6-dimensional feature space:

$$
\mathbf{P} = \{\mathbf{p}_i\}_{i=0}^{N-1}, \quad \mathbf{p}_i = [x_i, y_i, z_i, I_i, \text{idx}_i, l_i]^\top
$$

where:
- $(x_i, y_i, z_i) \in \mathbb{R}^3$ are spatial coordinates
- $I_i \in \mathbb{R}$ is intensity
- $\text{idx}_i \in \mathbb{Z}$ is the point index
- $l_i \in \mathbb{R}$ is the label

For geometric computations, only the first three coordinates are used.

---

### Euclidean Distance

For two points $\mathbf{p}_i, \mathbf{p}_j \in \mathbb{R}^3$:

$$
d(\mathbf{p}_i, \mathbf{p}_j) = \|\mathbf{p}_i - \mathbf{p}_j\|_2 = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}
$$

**Implementation:** 

```c++
/**
* @param p1 Pointer to first point (size >= 3)
* @param p2 Pointer to second point (size >= 3)
* @return Distance between p1 and p2
*/
distf(p1, p2)
```

---

### Dot Product

For two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^3$:

$$
\mathbf{a} \cdot \mathbf{b} = a_x b_x + a_y b_y + a_z b_z
$$

**Geometric interpretation:** For unit vectors, $\mathbf{a} \cdot \mathbf{b} = \cos\theta$, where $\theta$ is the angle between them.

**Implementation:**

```c++
/**
 * @param a Pointer to first 3D vector (size = 3)
 * @param b Pointer to second 3D vector (size = 3)
 *
 * @return Dot product (scalar)
 */
dot3f(a, b)
```

---

### Weighted Covariance Matrix

For a keypoint $\mathbf{p}_k$ with neighbor set $\mathcal{N}_k = \{\mathbf{p}_{i_1}, \mathbf{p}_{i_2}, \ldots, \mathbf{p}_{i_k}\}$, define the relative displacement vectors:

$$
\mathbf{d}_i = \mathbf{p}_i - \mathbf{p}_k, \quad \forall \mathbf{p}_i \in \mathcal{N}_k
$$

The radial weight function with support radius $R \in \mathbb{R}^+$:

$$
w_i = |\|\mathbf{p}_i - \mathbf{p}_k\| - R|
$$

This weighting emphasizes points near the neighborhood boundary.

The **weighted covariance matrix** $\mathbf{M} \in \mathbb{R}^{3 \times 3}$ is defined as:

$$
M = \frac{1}{\sum_{i: d_i \leq R} (R - d_i)} \sum_{i: d_i \leq R} (R - d_i) (\mathbf{p}_i - \mathbf{p})(\mathbf{p}_i - \mathbf{p})^\top
$$

In expanded form:

$$
\mathbf{M} = \frac{1}{\sum w_i} \sum w_i \begin{bmatrix}
d_x^2 & d_x d_y & d_x d_z \\
d_y d_x & d_y^2 & d_y d_z \\
d_z d_x & d_z d_y & d_z^2
\end{bmatrix}
$$

**Properties:**
- $\mathbf{M}$ is symmetric positive semi-definite
- Eigenvectors of $\mathbf{M}$ represent principal directions of the local neighborhood
- The eigenvector corresponding to the smallest eigenvalue approximates the surface normal

**Implementation:**
```c++
/**
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
conv_weight(points, p_id, points_num, neighbor_indices, k, radius, M)
```

---

### Histogram Binning with Linear Interpolation

Given a scalar value $v \in \mathbb{R}$, bin width $\Delta > 0$, and maximum value $V_{\max}$, define:

$$
v_{\text{bin}} = \frac{v}{\Delta}, \quad i = \lfloor v_{\text{bin}} \rfloor
$$

The linear interpolation weights for adjacent bins:

$$
w_{\text{lower}} = 1 - (v_{\text{bin}} - i), \quad w_{\text{upper}} = v_{\text{bin}} - i
$$

Boundary conditions:
- If $v < 0$: assign to bin $0$ with weight $1$
- If $v \ge V_{\max}$: assign to bin $\lfloor V_{\max}/\Delta \rfloor - 1$ with weight $1$
- Otherwise: contribute to bins $i$ and $i+1$ with weights $w_{\text{lower}}$ and $w_{\text{upper}}$ respectively

**Implementation:**

```c++
/**
 * @param value     Input value
 * @param step      Bin width
 * @param max_val   Maximum value range
 * @param bins      Output bin indices (size 2)
 * @param weights   Output weights (size 2)
 * @param count     Number of valid bins (1 or 2)
 */
binning_weight(value, step, max_val, bins, weights, count)
```

---

### Jacobi Eigenvalue Decomposition

For a symmetric matrix $\mathbf{A} \in \mathbb{R}^{3 \times 3}$, the Jacobi method iteratively applies orthogonal rotations to diagonalize $\mathbf{A}$.

#### STEP 1: Rotation Matrix

For off-diagonal element $A_{pq}$ with $p \neq q$, the rotation angle $\phi$ is computed as:

$$
\phi = \frac{1}{2} \arctan\left(\frac{2 A_{pq}}{A_{qq} - A_{pp}}\right)
$$

The rotation matrix $\mathbf{J}(p, q, \phi) \in \mathbb{R}^{3 \times 3}$ has:

$$
J_{pp} = J_{qq} = \cos\phi, \quad J_{pq} = -\sin\phi, \quad J_{qp} = \sin\phi
$$

with all other diagonal entries $1$ and off-diagonal entries $0$.

#### STEP 2: Similarity Transformation

Each iteration updates:

$$
\mathbf{A} \leftarrow \mathbf{J}^\top \mathbf{A} \mathbf{J}
$$

The transformation zeros out $A_{pq}$ while preserving symmetry and eigenvalues.

#### STEP 3: Convergence Criterion

Iteration terminates when:

$$
\max_{i \neq j} |A_{ij}| < \varepsilon
$$

where $\varepsilon = 10^{-6}$ in implementation.

**Implementation:**

```c++
/**
 * @param A         Input covariance matrix (modified in-place)
 * @param eigval    Output eigenvalues (size 3)
 * @param eigvec    Output eigenvectors (3x3, column-major)
 */
jacobi_3x3(A, eigval, eigvec)
```

---

### Principal Component Analysis (PCA)

#### STEP 1: Centroid Computation

For point $\mathbf{p}_k$ with $k$ neighbors $\{\mathbf{p}_{i_1}, \ldots, \mathbf{p}_{i_k}\}$:

$$
\bar{\mathbf{p}} = \frac{1}{k} \sum_{j=1}^{k} \mathbf{p}_{i_j}
$$

#### STEP 2: Centered Covariance Matrix

Define centered coordinates:

$$
\mathbf{u}_j = \mathbf{p}_{i_j} - \bar{\mathbf{p}} = \begin{bmatrix} u_x \\ u_y \\ u_z \end{bmatrix}
$$

The unbiased covariance matrix:

$$
\mathbf{C} = \frac{1}{k} \sum_{j=1}^{k} \mathbf{u}_j \mathbf{u}_j^\top = \frac{1}{k} \sum_{j=1}^{k} \begin{bmatrix}
u_x^2 & u_x u_y & u_x u_z \\
u_y u_x & u_y^2 & u_y u_z \\
u_z u_x & u_z u_y & u_z^2
\end{bmatrix}
$$

#### STEP 3: Eigendecomposition

Solve the eigenvalue problem:

$$
\mathbf{C} \mathbf{v}_m = \lambda_m \mathbf{v}_m, \quad m \in \{0,1,2\}
$$

with eigenvalues $\lambda_0 \ge \lambda_1 \ge \lambda_2 \ge 0$ and corresponding orthonormal eigenvectors $\{\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2\}$.

#### STEP 4: Geometric Interpretation

- $\lambda_0$: variance along principal direction $\mathbf{v}_0$
- $\lambda_1$: variance along secondary direction $\mathbf{v}_1$
- $\lambda_2$: variance along normal direction $\mathbf{v}_2$ (smallest eigenvalue)

The surface normal is approximated as $\mathbf{n} = \mathbf{v}_2$ (or $-\mathbf{v}_2$, up to sign ambiguity).


**Implementation:**

```c++
/*
 * @param points            Input points
 * @param neighbor_indices  Neighbors indices
 * @param N                 Number of points
 * @param k                 Number of neighbors
 * @param eigenvalues       Output eigenvalues [N * 3]
 * @param eigenvectors      Output eigenvectors [N * 9]
 */
pca(points, points_num, neighbor_indices, k, eigenvalues, eigenvectors)
```

---

### Normal Estimation

For each point $\mathbf{p}_i$, compute PCA and extract the eigenvector corresponding to the smallest eigenvalue:

$$
\lambda_{\min} = \min\{\lambda_0, \lambda_1, \lambda_2\}
$$

$$
\mathbf{n}_i = \mathbf{v}_{\arg\min_j \lambda_j}
$$

The output normal vector is stored as:

$$
\mathbf{n}_i = [n_x^{(i)}, n_y^{(i)}, n_z^{(i)}]^\top
$$

**Implementation:**

```c++
/*
 * @param h_points      Host point cloud
 * @param points_num    Number of points
 * @param h_neighbors   Neighbor indices
 * @param k             Number of neighbors
 * @param normals       Output normals (host)
 */
void normals_estimator(const float* h_points, const int points_num, const int* h_neighbors, const int k, float* normals);
```

---

## 🚀 Quick Usage

```c++
#include "mat.h"

// Host data
float* h_points;      // N × 6 point cloud
int* h_neighbors;     // N × (k+1) neighbor indices
float* h_normals;     // Output: N × 3 normals
int N = 10000;        // Number of points
int k = 20;           // Number of neighbors

// Create MAT instance with visualization
mat::MAT processor(true);

// Compute normals
processor.normals_estimator(h_points, N, h_neighbors, k, h_normals);
```
