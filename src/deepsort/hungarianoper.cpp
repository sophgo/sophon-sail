#include "hungarianoper.h"
//sisyphus
cv::Mat HungarianOper::Solve(const cv::Mat &cost_matrix) {
    int rows = cost_matrix.rows;
    int cols = cost_matrix.cols;
    Matrix<double> matrix(rows, cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            matrix(row, col) = cost_matrix.at<float>(row, col);
        }
    }
    //Munkres get matrix;
    Munkres<double> m;
    m.solve(matrix);


    std::vector<std::pair<int, int>> pairs;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int tmp = (int)matrix(row, col);
            if (tmp == 0) pairs.push_back(std::make_pair(row, col));
        }
    }

    int count = pairs.size();
    cv::Mat re(count, 2, CV_32F);
    for (int i = 0; i < count; i++) {
        re.at<float>(i, 0) = pairs[i].first;
        re.at<float>(i, 1) = pairs[i].second;
    }
    return re;
}//end Solve;
