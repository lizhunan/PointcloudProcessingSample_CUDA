#ifndef __TIMER_H__
#define __TIMER_H__

#include <chrono>
#include <ratio>
#include <vector>
#include <string>
#include "cuda_runtime.h"
#include "logger.h"
#include "cuda_base.h"

namespace timer {

/**
 * @brief High-resolution CPU/GPU hybrid timer utility.
 *
 * @details
 *  This class provides a unified interface for measuring execution time on:
 *
 *  - CPU (based on std::chrono)
 *  - GPU (based on CUDA events)
 *
 *  It is designed for:
 *
 *  - Performance benchmarking of CUDA kernels
 *  - Algorithm profiling (e.g., KNN, PCA, clustering)
 *  - Fine-grained stage timing in pipelines
 *
 *  Timing mechanism:
 *
 *  CPU timing:
 *      Uses std::chrono::high_resolution_clock
 *
 *          Δt = t_stop - t_start
 *
 *  GPU timing:
 *      Uses CUDA event API:
 *
 *          cudaEventRecord(start)
 *          cudaEventRecord(stop)
 *          cudaEventElapsedTime(...)
 *
 *      Internally:
 *          Δt (ms) = elapsed time between two events on GPU stream
 *
 *  Notes:
 *      - GPU timing is asynchronous by default → requires synchronization
 *      - CPU timing is synchronous
 *
 *  Reference:
 *      NVIDIA CUDA Programming Guide:
 *      https://docs.nvidia.com/cuda/cuda-c-programming-guide/
 */
class Timer
{

public:
    
    /**
     * @brief Time unit definitions (compile-time ratio types).
     *
     * @details
     *  These types are used as template parameters to control
     *  the output unit of CPU timing.
     *
     *      s  → seconds
     *      ms → milliseconds
     *      us → microseconds
     *      ns → nanoseconds
     */
    using s  = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

public:
    
    /**
     * @brief Constructor.
     *
     * @details
     *  - Initializes CPU timestamps
     *  - Creates CUDA events for GPU timing
     *  - Initializes internal buffers
     */
    Timer();

    /**
     * @brief Destructor.
     *
     * @details
     *  - Releases CUDA event resources
     *
     *  Important:
     *      cudaEventDestroy must be called to avoid GPU resource leaks
     */
    ~Timer();

public:

    /**
     * @brief Start CPU timing.
     *
     * @details
     *  Records current time as start point:
     *
     *      t_start = now()
     */
    void start_cpu();

    /**
     * @brief Start GPU timing.
     *
     * @details
     *  Records CUDA event on default stream:
     *
     *      cudaEventRecord(start)
     *
     *  Notes:
     *      - Non-blocking
     *      - Associated with stream 0
     */
    void start_gpu();

    /**
     * @brief Stop CPU timing.
     *
     * @details
     *  Records end time:
     *
     *      t_stop = now()
     */
    void stop_cpu();

    /**
     * @brief Stop GPU timing (without logging).
     *
     * @details
     *  Records stop CUDA event:
     *
     *      cudaEventRecord(stop)
     */
    void stop_gpu();

    /**
     * @brief Stop CPU timing and log result with unit.
     *
     * @tparam span Time unit (default = milliseconds)
     *
     * @param msg Description message for this timing block
     *
     * @details
     *  Computes:
     *
     *      Δt = t_stop - t_start
     *
     *  Then formats output:
     *
     *      "<msg> uses <time> <unit>"
     *
     *  Example:
     *
     *      timer.stop_cpu<timer::us>("PCA computation");
     *
     *  Notes:
     *      - Uses template-based unit selection
     *      - Stores result internally (not printed immediately)
     */
    template <typename span = ms>
    void stop_cpu(std::string msg);

    /**
     * @brief Stop GPU timing and log result.
     *
     * @param msg Description message
     *
     * @details
     *  Workflow:
     *
     *      1. Record stop event
     *      2. Synchronize start & stop events
     *      3. Compute elapsed time:
     *
     *          Δt = cudaEventElapsedTime(...)
     *
     *  Output unit:
     *      milliseconds (ms)
     *
     *  Notes:
     *      - Synchronization ensures correctness
     *      - Blocking operation
     */
    void stop_gpu(std::string msg);

    /**
     * @brief Print all recorded timing results.
     *
     * @details
     *  Iterates through internal message buffer and prints:
     *
     *      LOGV(...)
     *
     *  Typical usage:
     *
     *      timer.show();
     */
    void show();

    /**
     * @brief Reset timer state.
     *
     * @details
     *  Clears all stored timing messages:
     *
     *      _timeMsgs.clear()
     */
    void init();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;    /// CPU start timestamp
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;     /// CPU stop timestamp
    cudaEvent_t _gStart;                                                    /// CUDA event: start
    cudaEvent_t _gStop;                                                     /// CUDA event: stop
    float _timeElasped;                                                     /// GPU elapsed time (milliseconds)
    std::vector<std::string> _timeMsgs;                                     /// Stored formatted timing messages
};

/**
 * @brief Template implementation of CPU timing with unit control.
 *
 * @tparam span Time unit ratio (s/ms/us/ns)
 *
 * @param msg Description string
 *
 * @details
 *  Computes:
 *
 *      Δt = t_stop - t_start
 *
 *  Converts duration into specified unit:
 *
 *      duration<double, span>
 *
 *  Unit selection:
 *      - Compile-time via std::ratio
 *      - Runtime string mapping for display
 *
 *  Output format:
 *
 *      "\t<msg> uses <time> <unit>"
 */
template <typename span>
void Timer::stop_cpu(std::string msg) {
    _cStop = std::chrono::high_resolution_clock::now();

    char buff[100];
    std::string str;

    if(std::is_same<span, s>::value) { str = "s"; }
    else if(std::is_same<span, ms>::value) { str = "ms"; }
    else if(std::is_same<span, us>::value) { str = "us"; }
    else if(std::is_same<span, ns>::value) { str = "ns"; }

    std::chrono::duration<double, span> time = _cStop - _cStart;
    sprintf(buff, "\t%-60s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
    _timeMsgs.emplace_back(buff);
}

}

#endif