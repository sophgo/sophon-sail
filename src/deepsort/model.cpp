#include "model.h"
#include <algorithm>

const float kRatio=0.5;
enum DETECTBOX_IDX {IDX_X = 0, IDX_Y, IDX_W, IDX_H};

cv::Mat DETECTION_ROW::to_xyah() const {
    cv::Mat ret = tlwh.clone();
    ret.at<float>(0, IDX_X) += ret.at<float>(0, IDX_W) * kRatio;
    ret.at<float>(0, IDX_Y) += ret.at<float>(0, IDX_H) * kRatio;
    ret.at<float>(0, IDX_W) /= ret.at<float>(0, IDX_H);
    return ret;
}

cv::Mat DETECTION_ROW::to_tlbr() const {
    cv::Mat ret = tlwh.clone();
	ret.at<float>(0, IDX_X) += ret.at<float>(0, IDX_W);
	ret.at<float>(0, IDX_Y) += ret.at<float>(0, IDX_H);
    return ret;
}

