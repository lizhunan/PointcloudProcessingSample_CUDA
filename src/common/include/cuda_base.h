#ifndef __CUDA_BASE_HPP__
#define __CUDA_BASE_HPP__

#include <cuda_runtime.h>
#include <system_error>
#include <string>
#include <vector>
#include <memory>

/**
 * @brief CUDA API error checking macro.
 *
 * @param call CUDA runtime API call
 *
 * @details
 *  This macro wraps a CUDA API call and performs error checking:
 *
 *      CUDA_CHECK(cudaMalloc(...));
 *
 *  Expands to:
 *
 *      __cudaCheck(call, __FILE__, __LINE__)
 *
 *  Behavior:
 *      - Executes the CUDA API call
 *      - Checks return status
 *      - If failure:
 *          → prints file and line info
 *          → prints CUDA error name and description
 *          → terminates program
 *
 *  Example:
 *
 *      CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
 *
 *  Reference:
 *      CUDA Runtime API:
 *      https://docs.nvidia.com/cuda/cuda-runtime-api/
 */
#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)

/**
 * @brief CUDA kernel launch error checking macro.
 *
 * @details
 *  Used to check for errors after kernel launch:
 *
 *      kernel<<<...>>>();
 *      LAST_KERNEL_CHECK();
 *
 *  Internally:
 *
 *      cudaPeekAtLastError()
 *
 *  Notes:
 *      - Detects launch configuration errors
 *      - Does NOT guarantee kernel execution success
 *      - Non-blocking (does not synchronize device)
 *
 *  For full correctness:
 *
 *      cudaDeviceSynchronize()
 *
 *  should be used when necessary.
 */
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)

/**
 * @brief Check CUDA runtime API return status.
 *
 * @param err  CUDA error code
 * @param file Source file name
 * @param line Line number
 *
 * @details
 *  This function validates the return value of CUDA API calls.
 *
 *  If an error occurs:
 *
 *      cudaSuccess != err
 *
 *  then:
 *
 *      - Prints error location (file:line)
 *      - Prints error name:
 *
 *          cudaGetErrorName(err)
 *
 *      - Prints error description:
 *
 *          cudaGetErrorString(err)
 *
 *      - Terminates program
 *
 *  Example:
 *
 *      cudaError_t err = cudaMalloc(...);
 *      __cudaCheck(err, __FILE__, __LINE__);
 *
 *  Output format:
 *
 *      ERROR: file:line, code:<name>, reason:<description>
 */
static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

/**
 * @brief Check last CUDA kernel launch error.
 *
 * @param file Source file name
 * @param line Line number
 *
 * @details
 *  This function checks for errors from the most recent kernel launch.
 *
 *  Internally:
 *
 *      cudaPeekAtLastError()
 *
 *  Behavior:
 *
 *      if error exists:
 *          → print error information
 *          → terminate program
 */
static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}


#endif