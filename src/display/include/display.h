#ifndef __DISPLAY_H__
#define __DISPLAY_H__

#include <Eigen/Dense>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

namespace display {

class Display {

public:
    Display(const std::string win_name);
    ~Display();

public:
    void set_pointcloud_xyz(const float* points, const int points_num);
    void set_normal_vector();

private:
    void show();

private:
    std::thread thread;
    std::string win_name;

    float* points;
    int points_num;
};

}

#endif