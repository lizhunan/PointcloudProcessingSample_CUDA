#include "iss.h"

namespace iss {

ISS::ISS(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("ISS");
}

void ISS::detector(const float* points, const int points_num, const float r, bool *is_keypoints)
{
    
}

ISS::~ISS()
{}

}