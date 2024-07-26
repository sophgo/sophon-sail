import sophon.sail as sail

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
"""

def test_recv_bmi():
    handle = sail.Handle(0)
    bmcv = sail.Bmcv(handle)
    
    # 接收方
    ipc = sail.IPC(False, "/tmp/img", "/tmp/final", True, 10)
    
    while True:
        bmi_recv, channel_id, frame_id = ipc.receiveBMImage()
        save_name = "./recv" + str(frame_id) + ".jpg"
        bmcv.imwrite(save_name, bmi_recv)

def test_recv_tensor():
    handle = sail.Handle(0)
    bmcv = sail.Bmcv(handle)
    
    # 接收方
    ipc = sail.IPC(False, "/tmp/img", "/tmp/final", True, 10)
    
    while True:
        tensor, channel_id, frame_id = ipc.receiveTensor()
        bmi = bmcv.tensor_to_bm_image(tensor)
        save_name = "./recv" + str(frame_id) + ".jpg"
        bmcv.imwrite(save_name, bmi)

if __name__ == "__main__":
    test_recv_bmi()