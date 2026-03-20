#include "shot.h"

namespace shot {

SHOT::SHOT(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("SHOT");
}

void SHOT::detector(const float* points, const int points_num, const float r, bool *is_keypoints)
{
    
}

SHOT::~SHOT()
{}

}