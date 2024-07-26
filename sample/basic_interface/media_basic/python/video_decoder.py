import sophon.sail as sail
import argparse

def get_frame(device_id:int, video_path:str):
    handle = sail.Handle(device_id)
    bmcv = sail.Bmcv(handle)
    decoder = sail.Decoder(video_path,True,device_id)
    while True:
        image = decoder.read(handle)
        bmcv.imwrite("test.jpg",image)
        break


def get_batch_frame(device_id:int, video_path:str):
    handle = sail.Handle(device_id)
    bmcv = sail.Bmcv(handle)
    decoder = sail.Decoder(video_path,True,device_id)
    img_array = sail.BMImageArray4D()
    idx = 0
    while True:
        decoder.read_(handle,img_array[idx])
        idx += 1
        if idx >= 4:
            break
    for idx in range(4):
        bmcv.imwrite_("batch_{}.jpg".format(idx),img_array[idx]) # save bm_image 


if __name__ == "__main__":
    # 参数解析
    parse = argparse.ArgumentParser(description="Demo for media basic interface/video_decoder.py")
    parse.add_argument('--input_file_path', default="../datasets/test_car_person_1080P.mp4", type=str, help="Path or rtsp url to the video file.") 
    parse.add_argument('--device_id', default=0, type=int, help="Device id.") 
    parse.add_argument('--get_4batch', default=False, type=bool, help="Get 4 batch frame.")
    opt = parse.parse_args()
    video_path = opt.input_file_path
    device_id = opt.device_id

    if opt.get_4batch:
        get_batch_frame(device_id, video_path)
    else:
        get_frame(device_id, video_path)
    
    