#define USE_FFMPEG  1
#define USE_OPENCV  1
#define USE_BMCV    1

#include <stdio.h>
#include <sail/cvwrapper.h>
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

using namespace std;


void get_frame(int device_id, std::string video_path) 
{
    sail::Decoder decoder(video_path,true,device_id);
    if(!decoder.is_opened()){
        printf("Video[%s] read failed!\n",video_path.c_str());
        exit(1) ;
    }
    printf("Video[%s] read success!\n",video_path.c_str());
    sail::Handle handle(device_id);
    sail::Bmcv bmcv(handle);
    
    while(true){
        sail::BMImage ost_image;
        int ret = decoder.read(handle,ost_image);
        if (ret != 0){
            printf("Video[%s] read fail!\n",video_path.c_str());
            break;
        }
        printf("Video[%s] read one img success!\n",video_path.c_str());
        bmcv.imwrite("test.jpg", ost_image);
        break;
    }

    return;
}


void get_batch_frame(int device_id, std::string video_path) 
{
    sail::Decoder decoder(video_path,true,device_id);
    if(!decoder.is_opened()){
        printf("Video[%s] read failed!\n",video_path.c_str());
        exit(1) ;
    }
    
    sail::Handle handle(device_id);
    sail::Bmcv bmcv(handle);
    
    sail::BMImageArray<4> img_array;
    int idx = 0;
    while(true){
        for (int i = 0; i < 4; i++) {
            decoder.read_(handle, img_array[i]);
        }
        break;
    }

    bmcv.imwrite_("batch_0.jpg",img_array[0]);
    bmcv.imwrite_("batch_1.jpg",img_array[1]);
    bmcv.imwrite_("batch_2.jpg",img_array[2]);
    bmcv.imwrite_("batch_3.jpg",img_array[3]);
    return ;
}


int main(int argc, char* argv[])
{
    cout.setf(ios::fixed);
    // get params
    const char* keys =
        "{dev_id | 0 | device id}"
        "{input  | ../datasets/test_car_person_1080P.mp4 | input video path}"
        "{get_4batch | 0 | get 4 batch frame}";
    cv::CommandLineParser parser(argc, argv, keys);

    std::string video_path = parser.get<string>("input");
    int device_id = parser.get<int>("dev_id");
    bool get_4batch = parser.get<bool>("get_4batch");

    if (get_4batch){
        get_batch_frame(device_id, video_path);
    }
    else{
        get_frame(device_id, video_path);
    }
    return 0;
}


