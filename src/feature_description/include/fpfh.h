#ifndef __FPFH_H__
#define __FPFH_H__

#define HIST_DIM 33 // [] * 11 Bins
#define BIN_SIZE 11 // bin size

#include "/workspace/src/common/include/logger.h"
#include "/workspace/src/display/include/display.h"
#include "/workspace/src/common/include/mat.h"
#include "/workspace/src/common/include/neighbor.h"

namespace feat_desc {

class FPFH {

public:
    FPFH();
    FPFH(bool vis);
    ~FPFH();

public:
    void descriptor(const float* points, const int points_num, const int* h_neighbors, const float* h_normals,
                    const int k, const float r, float* h_histogram);

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