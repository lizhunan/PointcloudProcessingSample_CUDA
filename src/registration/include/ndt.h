#ifndef __NDT_H__
#define __NDT_H__

#include "/workspace/src/common/include/logger.h"
#include "/workspace/src/display/include/display.h"
#include "/workspace/src/common/include/mat.h"
#include "/workspace/src/common/include/neighbor.h"

namespace reg {

class NDT {

public:
    NDT();
    NDT(bool vis);
    ~NDT();

public:

private:
    bool vis;

    std::shared_ptr<logger::Logger> logger;
    std::shared_ptr<display::Display> display;
    std::shared_ptr<neighbor::Neighbor> neighbor;
    std::shared_ptr<mat::MAT> mat;

    float* d_points;
    int*   d_neighbors;
    float* d_normals;
    float* d_histogram;

};
}

#endif