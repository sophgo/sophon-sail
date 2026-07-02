import time
import os
import sophon.sail as sail
import argparse
"""
Multi channel video decoding demo program
Use the sail.MultiDecoder class to add channels to the MultiDecoder class, and each channel decodes a video independently.
In the MultiDecoder class, each video stream uses a separate C++ thread for scheduling decoding tasks, and each stream maintains its own cache queue. 
The thread stores the results of VPU hardware decoding in the cache queue, and each read() directly retrieves from the queue.
"""

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Demo for multi-channel decode")
    parse.add_argument('--dev_id', type=int, default=0, help='device id')
    parse.add_argument('--video_url',
                       type=str,
                       default="../datasets/test_car_person_1080P.mp4",
                       help='video url, can be rtsp, file path.')
    parse.add_argument('--is_local', type=bool, default=True, help='is local video')
    parse.add_argument('--read_timeout', type=int, default=5, help='read timeout')
    parse.add_argument('--test_duration', type=int, default=20, help='test time')
    parse.add_argument('--channel_num', type=int, default=10, help='channel number')
    parse.add_argument('--is_save_jpeg', type=bool, default=True, help='is save jpeg')
    args = parse.parse_args()

    # Specify the device number to use
    dev_id: int = args.dev_id
    # Initialize MultiDecoder
    md = sail.MultiDecoder(queue_size=10, tpu_id=dev_id, discard_mode=0)
    md.set_read_timeout(args.read_timeout)

    # Specify here the test stream address or local video file path
    video_url: str = args.video_url
    # Sets whether the video is local or not. The default is False, which means that the network video stream is decoded by default.
    # If decoding local video, it needs to be set to True, and each video is fixed to decode 25 frames per second
    md.set_local_flag(args.is_local)
    # Duration of this test
    test_duration: int = args.test_duration
    # Number of routes in this test
    channel_num: int = args.channel_num
    # Whether to save the decoded local image
    is_save_jpeg: bool = args.is_save_jpeg

    if (is_save_jpeg):
        save_interval: int = 100
        save_jpeg_path: str = "./multi-channel_decoded_images"
        if not os.path.exists(save_jpeg_path):
            os.mkdir(save_jpeg_path)
        handle = sail.Handle(dev_id)
        bmcv = sail.Bmcv(handle)

    # Add channel to MultiDecoder
    ch_idx_list: list = []
    for i in range(channel_num):
        # If added successfully, the number of the channel is returned, starting from 0
        ch_idx: int = md.add_channel(video_url, 0)  # zero means no skip frame
        if (ch_idx >= 0):
            ch_idx_list.append(ch_idx)
            print(f"channel {ch_idx} is added")
        # If -1 is returned, add failed
        else:
            print("add_channel failed")
            exit()

    # Initialize a list to count the number of decoded frames
    frame_cnt: list = [0] * channel_num
    ts_start = time.time()
    # Start reading for decoding results
    while (True):
        for i, ch_idx in enumerate(ch_idx_list):
            img: sail.BMImage = md.read(ch_idx)
            # At this point, the acquired BMImage can be processed as needed
            # ...
            frame_cnt[i] += 1
            # Save the image based on the previous settings
            if is_save_jpeg and not (frame_cnt[i] % save_interval):
                bmcv.imwrite(os.path.join(save_jpeg_path, f"chn_{ch_idx}_cnt_{frame_cnt[i]}.jpg"),
                             img)
        if (time.time() - ts_start >= test_duration):
            break

    # Calculate FPS for decoding process
    for i, ch_idx in enumerate(ch_idx_list):
        del_ret: int = md.del_channel(ch_idx)
        if (del_ret != 0):
            print("del_channel failed")
            exit()
    print("\n\n")
    print(frame_cnt)
    print(
        f"\n\n========== avg FPS in {test_duration}s : {sum(frame_cnt) / channel_num / test_duration} ==========\n\n"
    )
