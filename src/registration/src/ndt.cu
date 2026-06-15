#include "ndt.h"

namespace reg{

NDT::NDT(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("NDT");
    mat         = std::make_unique<mat::MAT>(false);
    neighbor    = std::make_unique<neighbor::Neighbor>(false);
}



NDT::~NDT()
{}

}