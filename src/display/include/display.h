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
    Display(int point_num, std::string vehicle_type, int ratio);
    ~Display();

public:
    void set_pointcloud();
    void set_normal_vector();

};

}

#endif