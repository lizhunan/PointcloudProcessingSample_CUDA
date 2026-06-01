#ifndef __STAT_FILTER_H__
#define __STAT_FILTER_H__

#include "/workspace/src/common/include/logger.h"
#include "display.h"
#include "/workspace/src/common/include/mat.h"

namespace denoising {

class StatFilter {

public:
    StatFilter();
    StatFilter(bool vis);
    ~StatFilter();

private:
    bool vis;
    
    std::unique_ptr<denoising_display::Display> display;
    std::unique_ptr<mat::MAT> mat;
};
}

#endif