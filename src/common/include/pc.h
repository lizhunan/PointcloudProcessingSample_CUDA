#ifndef __PC_H__
#define __PC_H__

#include <iostream>
#include <string>
#include <sstream> 
#include <fstream>
#include <vector>
#include "logger.h"
#include "cuda_base.h"
#include "/workspace/src/display/include/display.h"

namespace pc {

struct PointXYZIL
{
    float   X, Y, Z, I; // x, y, z, intensity
    uint8_t L;          // label
};

struct PCDInfo {
    std::string                 version;
    std::vector<std::string>    fields;
    std::vector<int>            sizes;
    std::vector<std::string>    types;
    std::vector<int>            counts;
    int                         width   = 0;
    int                         height  = 0;
    int                         points  = 0;
    std::string                 data_type;  // "ascii" 或 "binary"
    bool                        has_rgb = false;
};

class Pointcloud {

public:
    Pointcloud();
    Pointcloud(bool vis);
    ~Pointcloud();

public:
    bool read_pcd_ascii();
    bool read_pcd_bin(const std::string& filename, std::vector<PointXYZIL>& points,PCDInfo& info);
    bool read_bin();
    bool read_txt_xyz(const std::string& filename, float* output_points, int& points_num);

private:
    bool parse_pcd_header(std::ifstream& file, PCDInfo& info);

private:
    bool vis;

private:
    std::shared_ptr<logger::Logger> logger;
    std::shared_ptr<display::Display> display;

};

}

#endif