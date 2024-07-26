import time
import os
import sophon.sail as sail
import argparse

"""
多路视频解码演示程序。
使用sail.MultiDecoder类，向MultiDecoder类添加channel，每个channel独立解码一路视频。
在MultiDecoder类内部，每一路视频分别使用一个C++线程进行解码任务的调度，每一路分别维护一个缓存队列，线程将VPU硬解码的结果存放在缓存队列中，每次read()是直接从队列中获取。
"""

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Demo for multi-channel decode")
    parse.add_argument('--dev_id', type=int, default=0, help='device id')
    parse.add_argument('--video_url', type=str, default="../datasets/test_car_person_1080P.mp4", help='video url, can be rtsp, file path.')
    parse.add_argument('--is_local', type=bool, default=True, help='is local video')
    parse.add_argument('--read_timeout', type=int, default=5, help='read timeout')
    parse.add_argument('--test_duration', type = int, default = 20, help = 'test time')
    parse.add_argument('--channel_num', type = int, default = 10, help = 'channel number')
    parse.add_argument('--is_save_jpeg', type = bool, default = True, help = 'is save jpeg')
    args = parse.parse_args()

    # 指定要用的设备号
    dev_id:int = args.dev_id
    # 初始化MultiDecoder
    md = sail.MultiDecoder(queue_size=10, tpu_id=dev_id, discard_mode=0)
    md.set_read_timeout(args.read_timeout)

    # 在这里指定测试码流地址，或者本地视频文件路径
    video_url:str = args.video_url
    # 设置视频是否为本地视频。默认为False，即默认解码网络视频流。如果解码本地视频，需要设置为True，每路视频每秒固定解码25帧
    md.set_local_flag(args.is_local)
    # 本次测试的时长
    test_duration:int = args.test_duration
    # 本次测试的路数
    channel_num:int = args.channel_num
    # 是否将解码结果保存成本地图片
    is_save_jpeg:bool = args.is_save_jpeg


    if(is_save_jpeg):
        save_interval:int = 100
        save_jpeg_path:str = "./multi-channel_decoded_images"
        if not os.path.exists(save_jpeg_path):
            os.mkdir(save_jpeg_path)
        handle = sail.Handle(dev_id)
        bmcv = sail.Bmcv(handle)

    # 向MultiDecoder添加channel
    ch_idx_list:list = []
    for i in range(channel_num):
        # 如果添加成功，则返回该通道的编号，从0开始
        ch_idx:int = md.add_channel(video_url, 0) # zero means no skip frame
        if (ch_idx >= 0):
            ch_idx_list.append(ch_idx)
            print(f"channel {ch_idx} is added")
        # 如果返回-1，说明添加失败
        else:
            print("add_channel failed")
            exit()

    # 初始化一个列表，用于统计解码帧数
    frame_cnt:list = [0]*channel_num
    ts_start = time.time()
    # 开始read获取解码结果
    while (True):
        for i, ch_idx in enumerate(ch_idx_list):
            img:sail.BMImage = md.read(ch_idx)
            # 此时可以根据需要处理获取的BMImage
            # ...
            frame_cnt[i] += 1
            # 根据前面的设置，保存图片
            if is_save_jpeg and not (frame_cnt[i] % save_interval):
                bmcv.imwrite(os.path.join(save_jpeg_path, f"chn_{ch_idx}_cnt_{frame_cnt[i]}.jpg"), img)
        if(time.time() - ts_start >= test_duration):
            break

    # 统计解码过程的FPS
    for i, ch_idx in enumerate(ch_idx_list):
        del_ret:int = md.del_channel(ch_idx)
        if (del_ret != 0):
            print("del_channel failed")
            exit()
    print("\n\n")
    print(frame_cnt)
    print(f"\n\n========== avg FPS in {test_duration}s : {sum(frame_cnt) / channel_num / test_duration} ==========\n\n")