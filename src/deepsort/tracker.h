#ifndef TRACKER_H
#define TRACKER_H
#include <vector>

#include "../kalmanfilter/kalmanfilter.h"
#include "model.h"
#include "track.h"

class NearNeighborDisMetric;

class tracker {
   public:
    NearNeighborDisMetric* metric;
    float max_iou_distance;
    int max_age;
    int n_init;

    KalmanFilter* kf;

    int _next_idx;
    int k_feature_dim;
   public:
    std::vector<Track> tracks;
    tracker(float max_cosine_distance,
            int nn_budget,
            int k_feature_dim, 
            float max_iou_distance = 0.7,
            int max_age = 30,
            int n_init = 3);
    virtual ~tracker();
    void predict();
    void update(const DETECTIONS& detections);
    typedef cv::Mat (tracker::*GATED_METRIC_FUNC)(std::vector<Track>& tracks,
                                                   const DETECTIONS& dets,
                                                   const std::vector<int>& track_indices,
                                                   const std::vector<int>& detection_indices);

   private:
    void _match(const DETECTIONS& detections, TRACKER_MATCHD& res);
    void _initiate_track(const DETECTION_ROW& detection);

   public:
    cv::Mat gated_matric(std::vector<Track>& tracks,
                          const DETECTIONS& dets,
                          const std::vector<int>& track_indices,
                          const std::vector<int>& detection_indices);
    cv::Mat iou_cost(std::vector<Track>& tracks,
                      const DETECTIONS& dets,
                      const std::vector<int>& track_indices,
                      const std::vector<int>& detection_indices);
    cv::Mat iou(cv::Mat& bbox, cv::Mat& candidates);
};

#endif  // TRACKER_H
