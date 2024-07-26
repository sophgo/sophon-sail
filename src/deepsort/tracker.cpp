#include "tracker.h"
#include "linear_assignment.h"
#include "model.h"
#include "nn_matching.h"
#include <opencv2/opencv.hpp>
using namespace std;

#ifdef MY_inner_DEBUG
#include <iostream>
#include <string>
#endif


tracker::tracker(float max_cosine_distance,
                 int nn_budget,
                 int k_feature_dim,
                 float max_iou_distance,
                 int max_age,
                 int n_init) {
    this->metric =
        new NearNeighborDisMetric(NearNeighborDisMetric::METRIC_TYPE::cosine, max_cosine_distance, nn_budget, k_feature_dim);
    this->max_iou_distance = max_iou_distance;
    this->max_age = max_age;
    this->n_init = n_init;
    this->kf = new KalmanFilter();
    this->tracks.clear();
    this->_next_idx = 1;
    this->k_feature_dim = k_feature_dim;
}

tracker::~tracker() {
    delete this->metric;
    delete this->kf;
}

void tracker::predict() {
    for (Track& track : tracks) {
        track.predit(kf);
    }
}

void tracker::update(const DETECTIONS& detections) {
    TRACKER_MATCHD res;
    _match(detections, res);

    vector<MATCH_DATA>& matches = res.matches;
    for (MATCH_DATA& data : matches) {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx]);
    }
    vector<int>& unmatched_tracks = res.unmatched_tracks;
    for (int& track_idx : unmatched_tracks) {
        this->tracks[track_idx].mark_missed();
    }
    vector<int>& unmatched_detections = res.unmatched_detections;
    for (int& detection_idx : unmatched_detections) {
        this->_initiate_track(detections[detection_idx]);
    }
    vector<Track>::iterator it;
    for (it = tracks.begin(); it != tracks.end();) {
        if ((*it).is_deleted())
            it = tracks.erase(it);
        else
            ++it;
    }

    vector<int> active_targets;
    vector<pair<int, cv::Mat>> tid_features;
    for (Track& track : tracks) {
        if (track.is_confirmed() == false)
            continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        cv::Mat t(0, k_feature_dim, CV_32F);
        track.features = t.clone();
    }
    this->metric->partial_fit(tid_features, active_targets);
}

void tracker::_match(const DETECTIONS& detections, TRACKER_MATCHD& res) {
    vector<int> confirmed_tracks;
    vector<int> unconfirmed_tracks;
    int idx = 0;
    for (Track& t : tracks) {
        if (t.is_confirmed())
            confirmed_tracks.push_back(idx);
        else
            unconfirmed_tracks.push_back(idx);
        idx++;
    }

    TRACKER_MATCHD matcha =
        linear_assignment::getInstance()->matching_cascade(this, &tracker::gated_matric, this->metric->mating_threshold,
                                                           this->max_age, this->tracks, detections, confirmed_tracks);
    vector<int> iou_track_candidates;
    iou_track_candidates.assign(unconfirmed_tracks.begin(), unconfirmed_tracks.end());
    vector<int>::iterator it;
    for (it = matcha.unmatched_tracks.begin(); it != matcha.unmatched_tracks.end();) {
        int idx = *it;
        if (tracks[idx].time_since_update == 1) {  // push into unconfirmed
            iou_track_candidates.push_back(idx);
            it = matcha.unmatched_tracks.erase(it);
            continue;
        }
        ++it;
    }
    TRACKER_MATCHD matchb = linear_assignment::getInstance()->min_cost_matching(
        this, &tracker::iou_cost, this->max_iou_distance, this->tracks, detections, iou_track_candidates,
        matcha.unmatched_detections);
    // get result:
    res.matches.assign(matcha.matches.begin(), matcha.matches.end());
    res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());
    // unmatched_tracks;
    res.unmatched_tracks.assign(matcha.unmatched_tracks.begin(), matcha.unmatched_tracks.end());
    res.unmatched_tracks.insert(res.unmatched_tracks.end(), matchb.unmatched_tracks.begin(),
                                matchb.unmatched_tracks.end());
    res.unmatched_detections.assign(matchb.unmatched_detections.begin(), matchb.unmatched_detections.end());
}

void tracker::_initiate_track(const DETECTION_ROW& detection) {
    auto data = kf->initiate(detection.to_xyah());
    auto mean = data.first.clone();
    auto covariance = data.second.clone();

    this->tracks.push_back(
        Track(mean, covariance, this->_next_idx, detection.class_id, this->n_init, this->max_age, detection.feature, k_feature_dim));
    _next_idx += 1;
}

cv::Mat tracker::gated_matric(std::vector<Track>& tracks,
                               const DETECTIONS& dets,
                               const std::vector<int>& track_indices,
                               const std::vector<int>& detection_indices) {
    cv::Mat features(detection_indices.size(), k_feature_dim, CV_32F);
    int pos = 0;
    for (int i : detection_indices) {
        dets[i].feature.copyTo(features.row(pos++));
    }
    vector<int> targets;
    for (int i : track_indices) {
        targets.push_back(tracks[i].track_id);
    }

    cv::Mat cost_matrix = this->metric->distance(features, targets).clone();
    cv::Mat res = linear_assignment::getInstance()->gate_cost_matrix(this->kf, cost_matrix, tracks, dets,
                                                                      track_indices, detection_indices);
    return res;
}

cv::Mat tracker::iou_cost(std::vector<Track>& tracks,
                  const DETECTIONS& dets,
                  const std::vector<int>& track_indices,
                  const std::vector<int>& detection_indices) {
    int rows = track_indices.size();
    int cols = detection_indices.size();
    cv::Mat _cost_matrix(rows, cols, CV_32F);
    for (int i = 0; i < rows; i++) {
        int track_idx = track_indices[i];
        if (tracks[track_idx].time_since_update > 1) {
            for (int j = 0; j < cols; j++) {
                _cost_matrix.at<float>(i, j) = INFTY_COST;
            }
            continue;
        }
        cv::Mat bbox = tracks[track_idx].to_tlwh();
        int csize = detection_indices.size();
        cv::Mat candidates(csize, 4, CV_32F);
        for (int k = 0; k < csize; k++) {
            dets[detection_indices[k]].tlwh.copyTo(candidates.row(k));
        }

        cv::Mat iouMat = iou(bbox, candidates);
        cv::Mat rowV = cv::Mat::ones(1, iouMat.rows, CV_32FC1) - iouMat.t();
        for (int j = 0; j < cols; j++) {
            _cost_matrix.at<float>(i, j) = rowV.at<float>(j);
        }
    }
    return _cost_matrix;
}

cv::Mat tracker::iou(cv::Mat& bbox, cv::Mat& candidates) {
    float bbox_tl_1 = bbox.at<float>(0);
    float bbox_tl_2 = bbox.at<float>(1);
    float bbox_br_1 = bbox.at<float>(0) + bbox.at<float>(2);
    float bbox_br_2 = bbox.at<float>(1) + bbox.at<float>(3);
    float area_bbox = bbox.at<float>(2) * bbox.at<float>(3);

    int size = candidates.rows;
    cv::Mat res(size, 1, CV_32FC1);
    for (int i = 0; i < size; i++) {
        float tl_1 = std::max(bbox_tl_1, candidates.at<float>(i, 0));
        float tl_2 = std::max(bbox_tl_2, candidates.at<float>(i, 1));
        float br_1 = std::min(bbox_br_1, candidates.at<float>(i, 0) + candidates.at<float>(i, 2));
        float br_2 = std::min(bbox_br_2, candidates.at<float>(i, 1) + candidates.at<float>(i, 3));

        float w = br_1 - tl_1;
        w = (w < 0 ? 0 : w);
        float h = br_2 - tl_2;
        h = (h < 0 ? 0 : h);
        float area_intersection = w * h;
        float area_candidates = candidates.at<float>(i, 2) * candidates.at<float>(i, 3);
        res.at<float>(i, 0) = area_intersection / (area_bbox + area_candidates - area_intersection);
    }
    return res;
}


