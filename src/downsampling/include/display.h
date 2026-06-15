#ifndef __DOWNSAMPLING_DISPLAY_H__
#define __DOWNSAMPLING_DISPLAY_H__

#include <Eigen/Dense>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#define POINT_DIM 6

namespace downsampling {

class Display {

public:
    Display(const std::string win_name);
    ~Display();

public:
    void set_points(const float* points, const int points_num);

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