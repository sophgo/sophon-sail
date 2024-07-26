#ifndef LINEAR_ASSIGNMENT_H
#define LINEAR_ASSIGNMENT_H
#include "tracker.h"

#define INFTY_COST 1e5

class linear_assignment
{
    linear_assignment();
    linear_assignment(const linear_assignment& );
    virtual ~linear_assignment();
    linear_assignment& operator=(const linear_assignment&);
    static linear_assignment* instance;

public:
    static linear_assignment* getInstance();
    TRACKER_MATCHD matching_cascade(tracker* distance_metric,
            tracker::GATED_METRIC_FUNC distance_metric_func,
            float max_distance,
            int cascade_depth,
            std::vector<Track>& tracks,
            const DETECTIONS& detections,
            std::vector<int> &track_indices,
            std::vector<int> detection_indices = std::vector<int>());
    TRACKER_MATCHD min_cost_matching(
            tracker* distance_metric,
            tracker::GATED_METRIC_FUNC distance_metric_func,
            float max_distance,
            std::vector<Track>& tracks,
            const DETECTIONS& detections,
            std::vector<int>& track_indices,
            std::vector<int>& detection_indices);
    cv::Mat gate_cost_matrix(
            KalmanFilter* kf,
            cv::Mat& cost_matrix,
            std::vector<Track>& tracks,
            const DETECTIONS& detections,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices,
            float gated_cost = INFTY_COST,
            bool only_position = false);
};

#endif // LINEAR_ASSIGNMENT_H
