#ifndef HUNGARIANOPER_H
#define HUNGARIANOPER_H
#include "munkres.h"
#include <opencv2/opencv.hpp>

class HungarianOper {
public:
    static cv::Mat Solve(const cv::Mat &cost_matrix);
};

#endif // HUNGARIANOPER_H
