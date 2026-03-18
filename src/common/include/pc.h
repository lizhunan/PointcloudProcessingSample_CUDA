#ifndef _PC_H
#define _PC_H

#include <iostream>
#include <string>
#include <sstream> 
#include <fstream>
#include <vector>

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
    ~Pointcloud();

public:
    bool read_pcd_ascii();
    bool read_pcd_bin(const std::string& filename, std::vector<PointXYZIL>& points,PCDInfo& info);
    bool read_bin();
    bool read_txt();

private:
    bool parse_pcd_header(std::ifstream& file, PCDInfo& info);

};

}

#endif