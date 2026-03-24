#include "shot.h"

namespace shot {

SHOT::SHOT(bool vis)
{
    this->vis = vis;
    if (this->vis) display = std::make_unique<display::Display>("SHOT");
}

void SHOT::local_reference_frame(const float* points)
{

}

void SHOT::detector(const float* points, const int points_num, 
                    const float radius, const float bins_azimuth, const float bins_elevation,
                    const float bins_radial, const int bins_hist,
                    float* feat)
{
    // 计算每个点法向量
    // normals = estimate_normals()

    // 查询每个点的邻居
    // neighbors = kdtree.query_neighbors()

    // 计算局部坐标参考系
    // lrf = local_reference_frame()

    // 初始化直方图
    // histogram
    
}

SHOT::~SHOT()
{}

}