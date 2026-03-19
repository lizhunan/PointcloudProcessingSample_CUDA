#include "pc.h"

namespace pc {

__global__ void __to_device(const float* points, const int point_num, float* d_points)
{
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadid >= point_num) return;


}

Pointcloud::Pointcloud()
{}

bool Pointcloud::read_pcd_bin(const std::string& filename, std::vector<PointXYZIL>& points, PCDInfo& info)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        LOGE("Loading PCD binary file, %s is not founded!", filename.c_str());
        return false;
    }

    if (!parse_pcd_header(file, info))
    {
        LOGE("Loading PCD binary file, file header parsing error!");
        return false;
    }

    points.resize(info.points);

    size_t row_size = 0;
    for (size_t i = 0; i < info.sizes.size(); ++i) {
        row_size += info.sizes[i] * info.counts[i];
    }

    std::vector<char> buffer(row_size);

    return true;
}

bool Pointcloud::read_txt_xyz(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        LOGE("Loading TXT pointcloud file, %s is not founded!", filename.c_str());
        return false;
    }

    std::string line;
    int line_count = 0;
    while (std::getline(file, line))
    {
        if (!line.empty()) line_count++;
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    int points_num = 0;
    float* h_points = (float*)malloc(line_count*3*sizeof(float));
    if (!h_points)
    {
        LOGE("Failed to allocate host memory!");
        file.close();
        return false;
    }

    int point_idx = 0;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        
        for (char& c : line)
        {
            if (c == ',') c = ' ';
        }
        
        std::stringstream ss(line);
        float x, y, z;
        ss >> x >> y >> z;
        
        h_points[point_idx * 3 + 0] = x;
        h_points[point_idx * 3 + 1] = y;
        h_points[point_idx * 3 + 2] = z;
        
        point_idx++;
    }

    file.close();

    points_num = line_count;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(float)*points_num));
    to_device(h_points, points_num, d_points);
}

void Pointcloud::to_device(const float* h_points, const int points_num, float* d_points)
{
    int block_size  = 256;
    int grid_size   = (points_num + block_size - 1) / block_size; 
    __to_device<<<grid_size, block_size>>>(h_points, points_num, d_points);
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
            LOGV("Parsing PCD header, Version: %s", info.version.c_str());
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
            break;
        }
    }
        
    if (info.points == 0) {
        info.points = info.width * info.height;
    }
    LOGV("Parsing PCD header, Pointcloud size: %d", info.points);
    return info.points > 0;
}

Pointcloud::~Pointcloud()
{

}

};