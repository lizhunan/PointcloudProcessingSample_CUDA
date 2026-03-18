cmake_minimum_required(VERSION 3.20)
project(Sample)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
  include
  ${Pangolin_INCLUDE_DIRS}
)

add_subdirectory(src/common)

cuda_add_executable(Sample 
    src/main.cpp
    src/unit_test.cu)
    
target_link_libraries(Sample
    common
    ${Pangolin_LIBRARIES}
    Eigen3::Eigen
)