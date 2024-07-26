#pragma once
#include <cstddef>
#include <vector>

// track_idx, detection_idx
using MATCH_DATA = std::pair<int, int>;
typedef struct t {
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
} TRACKER_MATCHD;
