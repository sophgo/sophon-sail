#ifndef MODEL_H
#define MODEL_H
#include "dataType.h"
#include <opencv2/opencv.hpp>

// * Each rect's data structure.
// * tlwh: topleft point & (w,h)
// * confidence: detection confidence.
// * feature: the rect's 128d feature.
// */
class DETECTION_ROW {
   public:
    cv::Mat tlwh;
    float confidence;
    int class_id;
    cv::Mat feature;
    cv::Mat to_xyah() const;
    cv::Mat to_tlbr() const;
};

typedef std::vector<DETECTION_ROW> DETECTIONS;

#endif  // MODEL_H
