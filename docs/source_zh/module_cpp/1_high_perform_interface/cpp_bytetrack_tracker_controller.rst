bytetrack_tracker_controller
____________________________________________

针对ByteTrack算法，通过处理检测的结果，实现对目标的跟踪。

\_\_init\_\_
>>>>>>>>>>>>

**接口形式:**
    .. code-block:: c
          
        bytetrack_tracker_controller(int frame_rate = 30, 
                                int track_buffer = 30);

**参数说明:**

* frame_rate: int

输入参数。用于控制被追踪物体允许消失的最大帧数，数值越大则被追踪物体允许消失的最大帧数越大。

* track_buffer: int

输入参数。用于控制被追踪物体允许消失的最大帧数，数值越大则被追踪物体允许消失的最大帧数越大。


process
>>>>>>>>>>>>>

处理接口。

**接口形式1:**
    .. code-block:: c

        int process(const vector<DeteObjRect>& detected_objects, 
                vector<TrackObjRect>& tracked_objects);

**参数说明1:**

* detected_objects: vector<DeteObjRect>

输入参数。检测出的物体框。

* tracked_objects: vector<TrackObjRect>

输出参数。被跟踪的物体。

**返回值说明:**

int

成功返回0，失败返回其他。

**示例代码:**
    .. code-block:: c
        
        #include <sail/cvwrapper.h>
        #include "yolov5.h"// The example code relies on sophon-demo/sample/YOLOv5/cpp/yolov5_bmcv/yolov5.h
        #include <opencv2/opencv.hpp>  
        #include <vector>  
        #include <string>  
        
        using namespace std;  
        using namespace cv;  
        
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
            string input = "datasets/test_car_person_1080P.mp4";  
            string bmodel = "models/BM1684X/yolov5s_v6.1_3output_int8_1b.bmodel";  
            int dev_id = 0;  
            float conf = 0.4;  
            float nms = 0.7;  
            YOLOv5Arg yolov5_args(bmodel, dev_id, conf, nms);  
            YOLOv5 yolov5(yolov5_args);  
            
            VideoCapture cap(input);  
            vector<Mat> img_batch;  
            BytetrackTrackerController btt;  
            vector<vector<float>> track_res_all;  
            
            for (int i = 0; i < 50; i++) {  
                Mat img;  
                cap.read(img);  
                if (img.empty()) {  
                    break;  
                }  
                img_batch.push_back(img);  
                vector<vector<float>> results = yolov5.process(img_batch);  
                vector<vector<float>> det = results[0];  
                for (auto& row : det) {  
                    swap(row[4], row[5]);  
                }  
                img_batch.clear();  
                vector<tuple<int, int, int, int, int, float, int>> det_tuple;  
                for (auto& row : det) {  
                    det_tuple.push_back(make_tuple(static_cast<int>(row[0]), static_cast<int>(row[1]), static_cast<int>(row[2]), static_cast<int>(row[3]), static_cast<int>(row[4]), row[5], static_cast<int>(row[6])));  
                }  
                vector<vector<float>> track_res = btt.process(det_tuple);  
                track_res_all.push_back(track_res);  
            }   
            cap.release();    
            return 0;  
        }  