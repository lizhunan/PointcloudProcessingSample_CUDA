#ifndef __SHOT_H__
#define __SHOT_H__

#include "/workspace/src/common/include/logger.h"
#include "/workspace/src/display/include/display.h"

namespace shot {

class SHOT {

public:
    SHOT();
    SHOT(bool vis);
    ~SHOT();

public:
    void detector(const float* points, const int points_num, const float r, bool *is_keypoints);

private:
    bool vis;
    std::shared_ptr<logger::Logger> logger;
    std::shared_ptr<display::Display> display;

};

}

#endif