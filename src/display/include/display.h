#ifndef __DISPLAY_H__
#define __DISPLAY_H__

#include <Eigen/Dense>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#define POINT_DIM 6

namespace display {

class Display {

public:
    Display(const std::string win_name);
    ~Display();

public:
    void set_pointcloud_xyz(const float* points, const int points_num);
    void set_neighbors(const int* neighbors, const int k, const int point_idx=0);
    void set_normals(const float* normals, const int points_num);

private:
    void show();

private:
    std::thread thread;
    std::string win_name;

    float* points;
    int points_num;
    int* neighbors;
    int neighbors_k;
    float* normals;
};

}

#endif