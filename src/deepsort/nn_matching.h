#ifndef NN_MATCHING_H
#define NN_MATCHING_H

#include <opencv2/opencv.hpp>
#include <map>

// A tool to calculate distance;
class NearNeighborDisMetric {
   public:
    enum METRIC_TYPE { euclidean = 1, cosine };
    NearNeighborDisMetric(METRIC_TYPE metric, float matching_threshold, int budget, int k_feature_dim);
    cv::Mat distance(const cv::Mat& features, const std::vector<int>& targets);
    void partial_fit(std::vector<std::pair<int, cv::Mat>>& tid_feats, std::vector<int>& active_targets);
    float mating_threshold;
    int k_feature_dim;
   private:
    typedef cv::Mat(NearNeighborDisMetric::*PTRFUN)(const cv::Mat&, const cv::Mat&);
    cv::Mat _nncosine_distance(const cv::Mat& x, const cv::Mat& y);
    cv::Mat _cosine_distance(const cv::Mat& x, const cv::Mat& y);

   private:
    PTRFUN _metric;
    int budget;
    std::map<int, cv::Mat> samples;
};

#endif  // NN_MATCHING_H
