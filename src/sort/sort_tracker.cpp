#include "../deepsort/hungarianoper.h"
#include "sort_tracker.h"
#include "../deepsort/model.h"
#include "../deepsort/nn_matching.h"
#include <opencv2/opencv.hpp>
using namespace std;

#ifdef MY_inner_DEBUG
#include <iostream>
#include <string>
#endif


sort_tracker::sort_tracker(float max_iou_distance,
                            int max_age,
                            int n_init) {
    // this->metric =
    //     new NearNeighborDisMetric(NearNeighborDisMetric::METRIC_TYPE::cosine, max_cosine_distance, nn_budget, k_feature_dim);
    this->max_iou_distance = max_iou_distance;
    this->max_age = max_age;
    this->n_init = n_init;
    this->kf = new KalmanFilter();
    this->tracks.clear();
    this->_next_idx = 1;
    // this->k_feature_dim = k_feature_dim;
}

sort_tracker::~sort_tracker() {
    // delete this->metric;
    delete this->kf;
}

void sort_tracker::predict() {
    for (SortTrack& track : tracks) {
        track.predit(kf);
    }
}

void sort_tracker::update(const DETECTIONS& detections) {

    // 基于匈牙利算法对目标进行匹配
    TRACKER_MATCHD res;
    _match(detections, res);

    // 匹配更新卡尔曼滤波器
    vector<MATCH_DATA>& matches = res.matches;
    for (MATCH_DATA& data : matches) {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx]);
    }

    // 追踪目标丢失，更新目标丢失记录
    vector<int>& unmatched_tracks = res.unmatched_tracks;
    for (int& track_idx : unmatched_tracks) {
        this->tracks[track_idx].mark_missed();
    }

    // 检测到新的目标，初始化目标跟踪器
    vector<int>& unmatched_detections = res.unmatched_detections;
    for (int& detection_idx : unmatched_detections) {
        this->_initiate_track(detections[detection_idx]);
    }

    // 删除已完全丢失目标
    vector<SortTrack>::iterator it;
    for (it = tracks.begin(); it != tracks.end();) {
        if ((*it).is_deleted())
            it = tracks.erase(it);
        else
            ++it;
    }

    // vector<int> active_targets;
    // vector<pair<int, cv::Mat>> tid_features;
    // for (SortTrack& track : tracks) {
    //     if (track.is_confirmed() == false)
    //         continue;
    //     active_targets.push_back(track.track_id);
    //     tid_features.push_back(std::make_pair(track.track_id, track.features));
    //     cv::Mat t(0, k_feature_dim, CV_32F);
    //     track.features = t.clone();
    // }
    // this->metric->partial_fit(tid_features, active_targets);
}

void sort_tracker::_match(const DETECTIONS& detections, TRACKER_MATCHD& res) {

    //  IOU cost计算匹配
    res = min_cost_matching(max_iou_distance, tracks, detections);



    // vector<int> confirmed_tracks;
    // vector<int> unconfirmed_tracks;
    // int idx = 0;
    // for (SortTrack& t : tracks) {
    //     if (t.is_confirmed())
    //         confirmed_tracks.push_back(idx);
    //     else
    //         unconfirmed_tracks.push_back(idx);
    //     idx++;
    // }


    // // 匈牙利算法匹配 之前成功追踪的目标
    // TRACKER_MATCHD matcha =
    //     linear_assignment::getInstance()->matching_cascade(this, &sort_tracker::gated_matric, this->metric->mating_threshold,
    //                                                        this->max_age, this->tracks, detections, confirmed_tracks);
    
    // // 针对未匹配成功的追踪框
    // vector<int> iou_track_candidates;
    // iou_track_candidates.assign(unconfirmed_tracks.begin(), unconfirmed_tracks.end());
    // vector<int>::iterator it;
    // for (it = matcha.unmatched_tracks.begin(); it != matcha.unmatched_tracks.end();) {
    //     int idx = *it;
    //     if (tracks[idx].time_since_update == 1) {  // push into unconfirmed
    //         iou_track_candidates.push_back(idx);
    //         it = matcha.unmatched_tracks.erase(it);
    //         continue;
    //     }
    //     ++it;
    // }

    // // 计算最小的代价矩阵
    // TRACKER_MATCHD matchb = linear_assignment::getInstance()->min_cost_matching(
    //     this, &sort_tracker::iou_cost, this->max_iou_distance, this->tracks, detections, iou_track_candidates,
    //     matcha.unmatched_detections);

    // // get result:
    // res.matches.assign(matcha.matches.begin(), matcha.matches.end());
    // res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());
    // // unmatched_tracks;
    // res.unmatched_tracks.assign(matcha.unmatched_tracks.begin(), matcha.unmatched_tracks.end());
    // res.unmatched_tracks.insert(res.unmatched_tracks.end(), matchb.unmatched_tracks.begin(),
    //                             matchb.unmatched_tracks.end());
    // res.unmatched_detections.assign(matchb.unmatched_detections.begin(), matchb.unmatched_detections.end());
}

void sort_tracker::_initiate_track(const DETECTION_ROW& detection) {
    auto data = kf->initiate(detection.to_xyah());
    auto mean = data.first.clone();
    auto covariance = data.second.clone();

    this->tracks.push_back(
        SortTrack(mean, covariance, this->_next_idx, detection.class_id, this->n_init, this->max_age));
    _next_idx += 1;
}

// cv::Mat sort_tracker::gated_matric(std::vector<Track>& tracks,
//                                const DETECTIONS& dets,
//                                const std::vector<int>& track_indices,
//                                const std::vector<int>& detection_indices) {
//     cv::Mat features(detection_indices.size(), k_feature_dim, CV_32F);
//     int pos = 0;
//     for (int i : detection_indices) {
//         dets[i].feature.copyTo(features.row(pos++));
//     }
//     vector<int> targets;
//     for (int i : track_indices) {
//         targets.push_back(tracks[i].track_id);
//     }

//     cv::Mat cost_matrix = this->metric->distance(features, targets).clone();
//     cv::Mat res = linear_assignment::getInstance()->gate_cost_matrix(this->kf, cost_matrix, tracks, dets,
//                                                                       track_indices, detection_indices);
//     return res;
// }

cv::Mat sort_tracker::iou_cost(std::vector<SortTrack>& tracks,
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
                _cost_matrix.at<float>(i, j) = SORT_INFTY_COST;
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

cv::Mat sort_tracker::iou(cv::Mat& bbox, cv::Mat& candidates) {
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

// 最小代价矩阵计算
TRACKER_MATCHD sort_tracker::min_cost_matching(float max_distance,std::vector<SortTrack> &tracks,const DETECTIONS &detections)
{
    TRACKER_MATCHD res;

    std::vector<int>track_indices(tracks.size());
    for (auto i = 0; i < track_indices.size(); ++i){
        track_indices[i] = i;
    }

    std::vector<int>detection_indices(detections.size());
    for (auto i = 0; i < detection_indices.size(); ++i){
        detection_indices[i] = i;
    }

    if((track_indices.size() == 0) || (detection_indices.size() == 0)) {
        res.matches.clear();
        res.unmatched_tracks.assign(track_indices.begin(), track_indices.end());
        res.unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
        return res;
    }

    // 计算代价矩阵
    cv::Mat cost_matrix = iou_cost(tracks, detections, track_indices, detection_indices);
    for(int i = 0; i < cost_matrix.rows; i++) {
        for(int j = 0; j < cost_matrix.cols; j++) {
            float tmp = cost_matrix.at<float>(i,j);
            if(tmp > max_distance) cost_matrix.at<float>(i,j) = max_distance + 1e-5;
        }
    }
    
    cv::Mat indices = HungarianOper::Solve(cost_matrix);

    res.matches.clear();
    res.unmatched_tracks.clear();
    res.unmatched_detections.clear();
    for(size_t col = 0; col < detection_indices.size(); col++) {
        bool flag = false;
        for(int i = 0; i < indices.rows; i++)
            if(indices.at<float>(i, 1) == col) { flag = true; break;}
        if(flag == false)res.unmatched_detections.push_back(detection_indices[col]);
    }
    for(size_t row = 0; row < track_indices.size(); row++) {
        bool flag = false;
        for(int i = 0; i < indices.rows; i++)
            if(indices.at<float>(i, 0) == row) { flag = true; break; }
        if(flag == false) res.unmatched_tracks.push_back(track_indices[row]);
    }
    for(int i = 0; i < indices.rows; i++) {
        int row = indices.at<float>(i, 0);
        int col = indices.at<float>(i, 1);

        int track_idx = track_indices[row];
        int detection_idx = detection_indices[col];
        if(cost_matrix.at<float>(row, col) > max_distance) {
            res.unmatched_tracks.push_back(track_idx);
            res.unmatched_detections.push_back(detection_idx);
        } else res.matches.push_back(std::make_pair(track_idx, detection_idx));
    }
    return res;
}


