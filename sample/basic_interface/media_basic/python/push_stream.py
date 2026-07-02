# Full process demo of streaming media decoding, encoding, and pushing
import sophon.sail as sail
import argparse
import time


# Picture coding and decoding routines
def picture_decode_encode(handle: sail.Handle, decoder: sail.Decoder, encoder: sail.Encoder,
                          bmcv: sail.Bmcv):
    # Initialize bm_image
    img = sail.BMImage()
    # decode
    decode_ret = decoder.read(handle, img)
    if decode_ret != 0:
        print("image decode failed!")
        return -1

    ## you can use bmcv to do something here
    time.sleep(0.01)

    # encode
    img_data = encoder.pic_encode(".jpg", img)

    # imwrite
    with open('picture_encode.jpg', mode='wb+') as f:
        f.writelines(img_data)

    return 0


# Video Push
def video_push_stream(handle: sail.Handle, decoder: sail.Decoder, encoder: sail.Encoder,
                      bmcv: sail.Bmcv):
    # Cycle Detection
    count = 0
    while (True):
        # Initialize bm_image
        img = sail.BMImage()
        # decode
        decode_ret = decoder.read(handle, img)
        count += 1
        if (decode_ret != 0):
            break

        ## you can use bmcv to do something here
        time.sleep(0.01)

        ## encode   (push rtsp/rtmp stream or write local video file)
        encode_ret = encoder.video_write(img)
        while (encode_ret != 0):
            time.sleep(0.01)
            encode_ret = encoder.video_write(img)


def is_image(filename):
    image_extensions = ['.jpg', '.png', '.bmp']
    return any(filename.lower().endswith(ext) for ext in image_extensions)


if __name__ == "__main__":
    # Parameter resolution
    parse = argparse.ArgumentParser(description="Demo for media basic interface/push_stream.py")
    parse.add_argument('--input_file_path',
                       default="../datasets/test_car_person_1080P.mp4",
                       type=str,
                       help="Path or rtsp url to the video/image file.")  #Folder directory
    parse.add_argument('--output_path', type=str, help="Local file path or stream url")
    parse.add_argument('--device_id', default=0, type=int, help="Device id")
    parse.add_argument('--compressed_nv12',
                       default=True,
                       type=bool,
                       help="Whether the format of decoded output is compressed NV12.")
    parse.add_argument('--height', default=1080, type=int, help="The height of the encoded video")
    parse.add_argument('--width', default=1920, type=int, help="The width of the encoded video")
    parse.add_argument('--enc_fmt',
                       default='h264_bm',
                       type=str,
                       help="encoded video format, h264_bm/hevc_bm")
    parse.add_argument('--bitrate', default=2000, type=int, help="encoded bitrate")
    parse.add_argument('--pix_fmt', default='NV12', type=str, help="encoded pixel format")
    parse.add_argument('--gop', default=32, type=int, help="gop size")
    parse.add_argument('--gop_preset', default=2, type=int, help="gop_preset")
    parse.add_argument('--framerate', default=25, type=int, help="encode frame rate")
    opt = parse.parse_args()

    # Initialize Hardware
    handle = sail.Handle(opt.device_id)

    # Initialize control instructions for bmcv
    bmcv = sail.Bmcv(handle)

    # Initialize decoder
    decoder = sail.Decoder(opt.input_file_path, opt.compressed_nv12, opt.device_id)

    # Initialize encoder and start streaming test
    if is_image(opt.input_file_path):
        encoder = sail.Encoder()
        picture_decode_encode(handle, decoder, encoder, bmcv)

    else:
        enc_params = f"width={opt.width}:height={opt.height}:bitrate={opt.bitrate}:gop={opt.gop}:gop_preset={opt.gop_preset}:framerate={opt.framerate}"
        encoder = sail.Encoder(opt.output_path, opt.device_id, opt.enc_fmt, opt.pix_fmt, enc_params,
                               10)
        video_push_stream(handle, decoder, encoder, bmcv)

    # Resource Release
    decoder.release()
    encoder.release()
