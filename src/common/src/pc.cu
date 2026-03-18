#include "pc.h"

namespace pc {

Pointcloud::Pointcloud()
{
    logger = std::make_unique<logger::Logger>(logger::Level::DEBUG);
}

bool Pointcloud::read_pcd_bin(const std::string& filename, std::vector<PointXYZIL>& points, PCDInfo& info)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        LOGE("Loading pcd binary file, %s is not founded!", filename.c_str());
        return false;
    }

    if (!parse_pcd_header(file, info))
    {
        LOGE("Loading pcd binary file, file header parsing error!");
        return false;
    }

    return true;
}

bool Pointcloud::parse_pcd_header(std::ifstream& file, PCDInfo& info)
{
    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
            
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;
            
        if (token == "VERSION") {
            iss >> info.version;
        }
        else if (token == "FIELDS") {
            info.fields.clear();
            std::string field;
            while (iss >> field) {
                info.fields.push_back(field);
                if (field == "rgb" || field == "rgba") {
                    info.has_rgb = true;
                }
            }
        }
        else if (token == "SIZE") {
            info.sizes.clear();
            int size;
            while (iss >> size) {
                info.sizes.push_back(size);
            }
        }
        else if (token == "TYPE") {
            info.types.clear();
            std::string type;
            while (iss >> type) {
                info.types.push_back(type);
            }
        }
        else if (token == "COUNT") {
            info.counts.clear();
            int count;
            while (iss >> count) {
                info.counts.push_back(count);
            }
        }
        else if (token == "WIDTH") {
            iss >> info.width;
        }
        else if (token == "HEIGHT") {
            iss >> info.height;
        }
        else if (token == "POINTS") {
            iss >> info.points;
        }
        else if (token == "DATA") {
            iss >> info.data_type;
            break;  // DATA之后是点云数据
        }
    }
        
    // 验证必要信息
    if (info.points == 0) {
        info.points = info.width * info.height;
    }
        
    return info.points > 0;
}

Pointcloud::~Pointcloud()
{

}

};