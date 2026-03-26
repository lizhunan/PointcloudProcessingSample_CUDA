# Timer Module (CPU/GPU Profiling Utility)

## 🎯 Overview

The `timer` module provides a **high-resolution hybrid timing utility** for both CPU and GPU execution. It is designed for:

- CUDA kernel performance benchmarking
- Algorithm profiling (e.g., KNN, PCA, clustering)
- Pipeline-level latency analysis in pointcloud processing systems

This module supports:

- **CPU timing** via `std::chrono`
- **GPU timing** via CUDA event API
- **Flexible time units** (s / ms / us / ns)
- **Buffered logging system**

## 📚 Design Principles

### CPU Timing

CPU timing is based on:

$$
\Delta t = t_{stop} - t_{start}
$$

Implemented using:

```cpp
std::chrono::high_resolution_clock
```

### GPU Timing

GPU timing is implemented using CUDA events:

```cpp
cudaEventRecord(start);
cudaEventRecord(stop);
cudaEventElapsedTime(...)
```

Mathematically:

$$
\Delta t = t_{event}^{stop} - t_{event}^{start}
$$

**Key properties:**

* Unit: milliseconds (ms)
* Asynchronous by default
* Requires explicit synchronization

## 📦 Class Definition

```cpp
class Timer
```

### Time Units

```cpp
using s  = std::ratio<1, 1>;
using ms = std::ratio<1, 1000>;
using us = std::ratio<1, 1000000>;
using ns = std::ratio<1, 1000000000>;
```

| Unit | Description  |
| ---- | ------------ |
| s    | seconds      |
| ms   | milliseconds |
| us   | microseconds |
| ns   | nanoseconds  |


## 📊 API Reference

### Constructor & Destructor

```cpp
Timer();
~Timer();
```

**Description:**

* Initializes CPU timestamps
* Creates CUDA events
* Releases CUDA resources in destructor

### CPU Timing

#### Start CPU Timer

```cpp
void start_cpu();
```

#### Stop CPU Timer

```cpp
void stop_cpu();
```

#### Stop + Log (Recommended)

```cpp
template <typename span = ms>
void stop_cpu(std::string msg);
```

**Example:**

```cpp
timer.start_cpu();
// code block
timer.stop_cpu<timer::us>("PCA computation");
```

### GPU Timing

#### Start GPU Timer

```cpp
void start_gpu();
```

#### Stop GPU Timer

```cpp
void stop_gpu();
```

#### Stop + Log

```cpp
void stop_gpu(std::string msg);
```

**Example:**

```cpp
timer.start_gpu();
// CUDA kernel
timer.stop_gpu("KNN kernel");
```

---

### Utility Functions

#### Reset Timer

```cpp
void init();
```

* Clears all stored messages

#### Show Results

```cpp
void show();
```

* Outputs all timing logs via `LOG`

## 🚀 Usage Example

### CPU Profiling

```cpp
timer::Timer timer;

timer.start_cpu();
// Algorithm
timer.stop_cpu<timer::ms>("Total CPU Time");

timer.show();
```

### GPU Profiling

```cpp
timer::Timer timer;

timer.start_gpu();

kernel<<<grid, block>>>(...);

timer.stop_gpu("CUDA Kernel");

timer.show();
```

---

### Mixed Profiling (Recommended)

```cpp
timer::Timer timer;

timer.start_cpu();
timer.start_gpu();

kernel<<<...>>>();

timer.stop_gpu("Kernel");
timer.stop_cpu<timer::ms>("Total Pipeline");

timer.show();
```