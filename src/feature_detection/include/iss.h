#ifndef __ISS_H__
#define __ISS_H__

#include "/workspace/src/common/include/logger.h"
#include "/workspace/src/display/include/display.h"
#include "/workspace/src/common/include/mat.h"
#include "/workspace/src/common/include/neighbor.h"

namespace iss {

class ISS {

public:
    ISS();
    ISS(bool vis);
    ~ISS();

public:
    void detector(const float* points, const int points_num, const int* h_neighbors, const float nms_r, const int k, float* output);

private:
    bool vis;
    std::shared_ptr<logger::Logger> logger;
    std::shared_ptr<display::Display> display;

    float* d_points;
    int*   d_neighbors;
    float* d_lambda3;
};

}

#endif