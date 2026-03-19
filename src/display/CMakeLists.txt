cmake_minimum_required(VERSION 3.20)
set(MODULE_NAME display)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Pangolin REQUIRED)

add_library(${MODULE_NAME} STATIC 
    src/display.cpp
)

find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    include
    ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(${MODULE_NAME}
    Eigen3::Eigen
    ${Pangolin_LIBRARIES}
)