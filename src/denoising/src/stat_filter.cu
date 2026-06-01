#include "stat_filter.h"

namespace denoising {

StatFilter::StatFilter(){}

StatFilter::StatFilter(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<denoising_display::Display>("stat_denoising");
    mat          = std::make_unique<mat::MAT>(false);
}

StatFilter::~StatFilter()
{}

}