#include "icp.h"

namespace reg{

ICP::ICP(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("ICP");
    mat         = std::make_unique<mat::MAT>(false);
    neighbor    = std::make_unique<neighbor::Neighbor>(false);
}



ICP::~ICP()
{}

}