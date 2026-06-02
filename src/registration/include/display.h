#ifndef __REG_DISPLAY_H__
#define __REG_DISPLAY_H__

#include <Eigen/Dense>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#define POINT_DIM 6

namespace reg_display {

class Display {

public:
    Display(const std::string win_name);
    ~Display();

public:
    void set_source(const float* points, const int points_num);
    void set_target(const float* points, const int points_num);
    void set_corr(const float* source, const float* target, const int current_corr);
    void set_feat(const float* src, const float* tar, const float* src_feat, const float* tar_feat, 
                  const int src_feat_num, const int tar_feat_num);
    void set_feat_match(const float* src_matched, const float* tar_matched, const int matched_num);

private:
    void show();

private:
    std::thread thread;
    std::string win_name;

    float* source;
    int source_num;
    float* target;
    int target_num;
    float* source_corr;
    float* target_corr;
    int corr_num;
    float* src_feat;
    float* tar_feat;
    float* src_matched;
    float* tar_matched;
    int matched_num;
};

}

#endif