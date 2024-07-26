import sophon.sail as sail
import time
"""
测试进程间通信的代码。开启两个终端, 一个运行python3 IPC_sender.py, 一个运行python3 IPC_receiver.py
发送方会给接收方传递一个BMImage, 接收方收到之后,会将它保存成本地文件

IPC的构造函数参数:
    bool isSender_ 是否为发送方;
    string image_pipe 用于传递图片或tensor的管道名, 管道不需要提前创建，但父目录必须存在;
    string final_pipe 用于接收返回信息的管道名;
    usec2c 接收方是否使用c2c接口拷贝内存。默认为false;
        从单纯通信上来看, c2c比d2d更慢, 但由于d2d使用gdma, 所以可能c2c对推理影响更小。
        发送方不需要设置。

运行本例程前, 请函数中的文件路径
"""

def test_send_bmi():
    file_path = '/home/xyz/projects/videos/monitor.mp4'
    
    handle = sail.Handle(0)
    decoder = sail.Decoder(file_path)
    ipc = sail.IPC(True, "/tmp/img", "/tmp/final")
    idx = 0
    
    while True:
        time.sleep(0.04)
        bmi = decoder.read(handle)
        # 发送方
        ipc.sendBMImage(bmi, 0, idx)
        idx += 1

def test_send_tensor():
    file_path = '/home/xyz/projects/videos/monitor.mp4'
    
    handle = sail.Handle(0)
    decoder = sail.Decoder(file_path)
    bmcv = sail.Bmcv(handle)
    ipc = sail.IPC(True, "/tmp/img", "/tmp/final")
    idx = 0
    
    while True:
        time.sleep(0.04)
        bmi = decoder.read(handle)
        tensor = bmcv.bm_image_to_tensor(bmi)
        # 发送方
        ipc.sendTensor(tensor, 0, idx)
        idx += 1

if __name__ == "__main__":
    test_send_bmi()