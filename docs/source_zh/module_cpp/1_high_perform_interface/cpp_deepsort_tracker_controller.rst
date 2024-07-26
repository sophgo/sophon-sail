deepsort_tracker_controller
____________________________________________

针对DeepSORT算法，通过处理检测的结果和提取的特征，实现对目标的跟踪。

构造函数
>>>>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        deepsort_tracker_controller(float max_cosine_distance, 
                                int nn_budget, 
                                int k_feature_dim, 
                                float max_iou_distance = 0.7, 
                                int max_age = 30, 
                                int n_init = 3);

**参数说明:**

* max_cosine_distance: float

输入参数。用于相似度计算的最大余弦距离阈值。

* nn_budget: int

输入参数。用于最近邻搜索的最大数量限制。

* k_feature_dim: int

输入参数。被检测的目标的特征维度。

* max_iou_distance: float

输入参数。模用于跟踪器中的最大交并比（IoU）距离阈值。

* max_age: int

输入参数。跟踪目标在跟踪器中存在的最大帧数。

* n_init: int

输入参数。跟踪器中的初始化帧数阈值。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: c

        int process(const vector<DeteObjRect>& detected_objects, 
                vector<Tensor>& feature, 
                vector<TrackObjRect>& tracked_objects);

**参数说明1:**

* detected_objects: vector<DeteObjRect>

输入参数。检测出的物体框。

* feature: vector<Tensor>

输入参数。检测出的物体的特征。

* tracked_objects: vector<TrackObjRect>

输出参数。被跟踪的物体。

**返回值说明1:**

int

成功返回0，失败返回其他。


**接口形式2:**
    .. code-block:: c

        int process(const vector<DeteObjRect>& detected_objects, 
                vector<vector<float>>& feature, 
                vector<TrackObjRect>& tracked_objects);

**参数说明2:**

* detected_objects: vector<DeteObjRect>

输入参数。检测出的物体框。

* feature: vector<float>

输入参数。检测出的物体的特征。

* tracked_objects: vector<TrackObjRect>

输出参数。被跟踪的物体。

**返回值说明2:**

int

成功返回0，失败返回其他。

**示例代码:**
    .. code-block:: c
        
        // The example code relies on sophon-demo/sample/YOLOv5/cpp/yolov5_bmcv/yolov5.h and sophon-demo/sample/DeepSORT/cpp/deepsort_bmcv/FeatureExtractor.h
        #include <sail/cvwrapper.h>
        #include "yolov5.h"
        #include "FeatureExtractor.h"
        #include <opencv2/opencv.hpp>  
        #include <vector>  
        #include <string>  
        
        using namespace std;  
        
        class YOLOv5Arg {  
        public:  
            string bmodel;  
            int dev_id;  
            float conf_thresh;  
            float nms_thresh;  
            
            YOLOv5Arg(string bmodel, int dev_id, float conf_thresh, float nms_thresh) {  
                this->bmodel = bmodel;  
                this->dev_id = dev_id;  
                this->conf_thresh = conf_thresh;  
                this->nms_thresh = nms_thresh;  
            }  
        };  
        
        int main() {  
            string input = "data/test_car_person_1080P.mp4";  
            string bmodel_detector = "models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel";  
            string bmodel_extractor = "models/BM1684X/extractor_int8_1b.bmodel";  
            int dev_id = 0;  
            float conf = 0.4;  
            float nms = 0.7;  
            
            YOLOv5Arg yolov5_args(bmodel_detector, dev_id, conf, nms);  
            YOLOv5 yolov5(yolov5_args);  
            Extractor extractor(bmodel_extractor, dev_id);  
            
            cv::VideoCapture cap(input);  
            vector<cv::Mat> img_batch;  
            
            sail::deepsort_tracker_controller dstc(0.2, 100, extractor.output_shape[1], 0.7, 70, 3);  
            
            vector<vector<float>> track_res_all_numpy;  
            
            for (int i = 0; i < 15; i++) {  
                cv::Mat img;  
                cap.read(img);  
                if (img.empty()) {  
                    break;  
                }  
                img_batch.push_back(img);  
                vector<vector<float>> det = yolov5(img_batch);  
                vector<cv::Rect> dets;  
                for (auto& item : det) {  
                    int x1 = static_cast<int>(item[0]);  
                    int y1 = static_cast<int>(item[1]);  
                    int x2 = static_cast<int>(item[2]);  
                    int y2 = static_cast<int>(item[3]);  
                    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);  
                    dets.push_back(roi);  
                }  
                vector<cv::Mat> im_crops;  
                for (auto& roi : dets) {  
                    cv::Mat img_crop = img(roi);  
                    im_crops.push_back(img_crop);  
                }     
                vector<vector<float>> ext_results = extractor(im_crops);   
          
                // The order of this API and the demo is inconsistent, and the class_id and score are reversed 
                for (auto& row : det) {  
                    swap(row[4], row[5]);  
                }  
                img_batch.clear();  
                
                vector<tuple<int, int, int, int, int, float, int>> det_tuple;  
                for (auto& row : det) {  
                    det_tuple.push_back(make_tuple(static_cast<int>(row[0]), static_cast<int>(row[1]), static_cast<int>(row[2]), static_cast<int>(row[3]), static_cast<int>(row[4]), row[5], static_cast<int>(row[6])));  
                }   
           
            }  
            return 0; 
        }  