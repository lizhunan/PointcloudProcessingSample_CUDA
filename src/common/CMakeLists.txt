cmake_minimum_required(VERSION 3.20)
set(MODULE_NAME common)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

cuda_add_library(${MODULE_NAME} STATIC 
    src/pc.cu
    src/timer.cpp
    src/logger.cpp
    src/mat.cu
    src/neighbor.cu
)

include_directories(
    include
    ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(${MODULE_NAME}
    display
    ${Pangolin_LIBRARIES}
    Eigen3::Eigen
)