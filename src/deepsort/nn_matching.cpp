#include "nn_matching.h"
NearNeighborDisMetric::NearNeighborDisMetric(NearNeighborDisMetric::METRIC_TYPE metric,
                                             float matching_threshold,
                                             int budget, int k_feature_dim) {
    _metric = &NearNeighborDisMetric::_nncosine_distance;

    this->mating_threshold = matching_threshold;
    this->budget = budget;
    this->samples.clear();
    this->k_feature_dim = k_feature_dim;
}

cv::Mat NearNeighborDisMetric::distance(const cv::Mat& features, const std::vector<int>& targets) {
    cv::Mat cost_matrix(targets.size(), features.rows, CV_32F);
    int idx = 0;
    for (int target : targets) {        
        cv::Mat vec = (this->*_metric)(this->samples[target], features);
        for(int i = 0; i < cost_matrix.cols; i++) {
            cost_matrix.at<float>(idx, i) = vec.at<float>(i);
        }

        idx++;
    }
    return cost_matrix;
}

void NearNeighborDisMetric::partial_fit(std::vector<std::pair<int, cv::Mat>>& tid_feats, std::vector<int>& active_targets) {
    for (auto& data : tid_feats) {
        int track_id = data.first;
        cv::Mat newFeatOne = data.second.clone();

        if (samples.find(track_id) != samples.end()) {  // append
            int oldSize = samples[track_id].rows;
            int addSize = newFeatOne.rows;
            int newSize = oldSize + addSize;

            if (newSize <= this->budget) {
                cv::Mat newSampleFeatures(newSize, k_feature_dim, CV_32F);
                cv::Mat oldSamples = samples[track_id];

                cv::Mat oldRegion = newSampleFeatures(cv::Rect(0, 0, k_feature_dim, oldSize));
                oldSamples.copyTo(oldRegion);

                cv::Mat newRegion = newSampleFeatures(cv::Rect(0, oldSize, k_feature_dim, addSize));
                newFeatOne.copyTo(newRegion);
                
                samples[track_id] = newSampleFeatures;
            } else {
                if (oldSize < this->budget) {  // original space is not enough
                    cv::Mat newSampleFeatures(this->budget, k_feature_dim, CV_32F);
                    if (addSize >= this->budget) {
                        newFeatOne(cv::Range(0, this->budget), cv::Range::all()).copyTo(newSampleFeatures);
                    } else {
                        cv::Mat dstRegion1 = newSampleFeatures(cv::Range(0, this->budget - addSize), cv::Range::all());
                        cv::Mat srcRegion1 = samples[track_id](cv::Range(addSize - 1, this->budget - 1), cv::Range::all());
                        srcRegion1.copyTo(dstRegion1);

                        cv::Mat dstRegion2 = newSampleFeatures(cv::Range(this->budget - addSize, this->budget), cv::Range::all());
                        newFeatOne(cv::Range(0, addSize), cv::Range::all()).copyTo(dstRegion2);
                    }
                    samples[track_id] = newSampleFeatures;
                } else {  // original space is ok
                    if (addSize >= this->budget) {
                        newFeatOne(cv::Range(0, this->budget), cv::Range::all()).copyTo(samples[track_id]);
                    } else {
                        cv::Mat dstRegion1 = samples[track_id](cv::Range(0, this->budget - addSize), cv::Range::all());
                        cv::Mat srcRegion1 = samples[track_id](cv::Range(addSize - 1, this->budget - 1), cv::Range::all());
                        srcRegion1.copyTo(dstRegion1);

                        cv::Mat dstRegion2 = samples[track_id](cv::Range(this->budget - addSize, this->budget), cv::Range::all());
                        newFeatOne(cv::Range(0, addSize), cv::Range::all()).copyTo(dstRegion2);
                    }
                }
            }
        } else {  // not exit, create new one;
            samples[track_id] = newFeatOne;
        }
    }  // add features;

    // erase the samples which not in active_targets;
    for (auto i = samples.begin(); i != samples.end();) {
        bool flag = false;
        for (int j : active_targets)
            if (j == i->first) {
                flag = true;
                break;
            }
        if (flag == false)
            samples.erase(i++);
        else
            i++;
    }
}

cv::Mat NearNeighborDisMetric::_nncosine_distance(const cv::Mat& x, const cv::Mat& y) {
    cv::Mat distances = _cosine_distance(x, y);
    // 计算每一列的最小值
    cv::Mat minValues;
    cv::reduce(distances, minValues, 0, cv::REDUCE_MIN);

    cv::Mat res = minValues.t();
    return res;
}

cv::Mat NearNeighborDisMetric::_cosine_distance(const cv::Mat& x, const cv::Mat& y) {
    cv::Mat res;
    cv::gemm(x, y.t(), 1, cv::Mat(), 0, res);
    cv::subtract(cv::Scalar::all(1), res, res);
    return res;
}