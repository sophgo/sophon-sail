#include "track.h"


Track::Track(const cv::Mat& mean,
             const cv::Mat& covariance,
             int track_id,
             int class_id,
             int n_init,
             int max_age,
             const cv::Mat& features,
             int k_feature_dim) {
    this->mean = mean.clone();
    this->covariance = covariance.clone();
    this->track_id = track_id;
    this->class_id = class_id;
    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    this->state = TrackState::Tentative;
    this->features = features.clone();
    this->k_feature_dim = k_feature_dim;
    this->_n_init = n_init;
    this->_max_age = max_age;
}

void Track::predit(KalmanFilter* kf) {
    /*Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        */  
    auto pa = kf->predict(mean, covariance);

    mean = pa.first.clone();
    covariance = pa.second.clone();

    this->age += 1;
    this->time_since_update += 1;
}

void Track::update(KalmanFilter* const kf, const DETECTION_ROW& detection) {
    auto pa = kf->update(mean, covariance, detection.to_xyah());
    mean = pa.first.clone();
    covariance = pa.second.clone();    

    featuresAppendOne(detection.feature);
    this->hits += 1;
    this->time_since_update = 0;
    if (this->state == TrackState::Tentative && this->hits >= this->_n_init) {
        this->state = TrackState::Confirmed;
    }
}

void Track::mark_missed() {
    if (this->state == TrackState::Tentative) {
        this->state = TrackState::Deleted;
    } else if (this->time_since_update > this->_max_age) {
        this->state = TrackState::Deleted;
    }
}

bool Track::is_confirmed() {
    return this->state == TrackState::Confirmed;
}

bool Track::is_deleted() {
    return this->state == TrackState::Deleted;
}

bool Track::is_tentative() {
    return this->state == TrackState::Tentative;
}

cv::Mat Track::to_tlwh() {
    cv::Mat ret = mean.clone();
    ret.at<float>(2) *= ret.at<float>(3);
    ret.at<float>(0) -= (ret.at<float>(2) / 2);
    ret.at<float>(1) -= (ret.at<float>(3) / 2);
    return ret;
}

void Track::featuresAppendOne(const cv::Mat& f) {
    int size = features.rows;
    cv::Mat newFeatures(size + 1, k_feature_dim, f.type());
    cv::Mat roiNewFeatures(newFeatures, cv::Rect(0, 0, k_feature_dim, size));
    features.copyTo(roiNewFeatures);
    cv::Mat roiNewFeaturesLastRow(newFeatures, cv::Rect(0, size, k_feature_dim, 1));
    f.copyTo(roiNewFeaturesLastRow);
    features = newFeatures;
}