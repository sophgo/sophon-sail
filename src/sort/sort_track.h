#ifndef SORT_TRACK_H
#define SORT_TRACK_H

#include <opencv2/opencv.hpp>
#include "../kalmanfilter/kalmanfilter.h"
#include "../deepsort/model.h"

class SortTrack {
    /*"""
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.

    """*/
    enum TrackState { Tentative = 1, Confirmed, Deleted };

   public:
    SortTrack(const cv::Mat& mean,
          const cv::Mat& covariance,
          int track_id,
          int class_id,
          int n_init,
          int max_age);

    void predit(KalmanFilter* kf);
    void update(KalmanFilter* const kf, const DETECTION_ROW& detection);
    
    void mark_missed();
    bool is_confirmed();
    bool is_deleted();
    bool is_tentative();

    cv::Mat to_tlwh();
    int time_since_update;

    int track_id;
    int class_id;

    cv::Mat mean;
    cv::Mat covariance;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;

   private:
    void featuresAppendOne(const cv::Mat& f);
};

#endif  // SORT_TRACK_H
