#ifndef SORT_TRACKER_H
#define SORT_TRACKER_H
#include <vector>

#include "../kalmanfilter/kalmanfilter.h"
#include "../deepsort/model.h"
#include "sort_track.h"

#define SORT_INFTY_COST 1e5

class NearNeighborDisMetric;

class sort_tracker {
   public:
//     NearNeighborDisMetric* metric;
    float max_iou_distance;
    int max_age;
    int n_init;

    KalmanFilter* kf;

    int _next_idx;
//     int k_feature_dim;
   public:
    std::vector<SortTrack> tracks;
    sort_tracker(float max_iou_distance = 0.7,
            int max_age = 30,
            int n_init = 3);
    virtual ~sort_tracker();
    void predict();
    void update(const DETECTIONS& detections);

//     typedef cv::Mat (sort_tracker::*GATED_METRIC_FUNC)(std::vector<SortTrack>& tracks,
//                                                    const DETECTIONS& dets,
//                                                    const std::vector<int>& track_indices,
//                                                    const std::vector<int>& detection_indices);

   private:
    void _match(const DETECTIONS& detections, TRACKER_MATCHD& res);
    void _initiate_track(const DETECTION_ROW& detection);
    TRACKER_MATCHD min_cost_matching(float max_distance,std::vector<SortTrack> &tracks,const DETECTIONS &detections);

   public:
    cv::Mat gated_matric(std::vector<SortTrack>& tracks,
                          const DETECTIONS& dets,
                          const std::vector<int>& track_indices,
                          const std::vector<int>& detection_indices);
    cv::Mat iou_cost(std::vector<SortTrack>& tracks,
                      const DETECTIONS& dets,
                      const std::vector<int>& track_indices,
                      const std::vector<int>& detection_indices);
    cv::Mat iou(cv::Mat& bbox, cv::Mat& candidates);

    
};

#endif  // SORT_TRACKER_H
