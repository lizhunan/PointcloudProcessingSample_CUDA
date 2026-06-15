cmake_minimum_required(VERSION 3.20)
set(MODULE_NAME downsampling)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

cuda_add_library(${MODULE_NAME} STATIC 
    src/nss.cu
    src/display.cpp
)

include_directories(
    include
)

target_link_libraries(${MODULE_NAME}
    common
    Eigen3::Eigen
)