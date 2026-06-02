#ifndef __CLUSTER_DISPLAY_H__
#define __CLUSTER_DISPLAY_H__

#include <Eigen/Dense>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#define POINT_DIM 6

namespace cluster_display {

class Display {

public:
    Display(const std::string win_name);
    ~Display();

public:
    void set_points(const float* points, const int points_num);

private:
    void show();

    inline void label_to_color(int label, float &r, float &g, float &b)
    {
        if (label == 0) {
            r = g = b = 1.0f; // Defualt white
            return;
        }

        unsigned int x = static_cast<unsigned int>(label * 2654435761u);

        x = (x ^ 61u) ^ (x >> 16);
        x *= 9u;
        x = x ^ (x >> 4);
        x *= 0x27d4eb2du;
        x = x ^ (x >> 15);

        r = (x & 0x0000FFu) / 255.0f;
        g = ((x & 0x00FF00u) >> 8) / 255.0f;
        b = ((x & 0xFF0000u) >> 16) / 255.0f;
    }

private:
    std::thread thread;
    std::string win_name;

    float* points;
    int points_num;
};

}

#endif