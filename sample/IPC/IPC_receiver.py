import sophon.sail as sail
"""
Code for testing interprocess communication.
Open two terminals, one running python3 IPC_sender.py, one running python3 IPC_receiver.py
The sender will pass a BMImage to the receiver, and when the receiver receives it, it will save the local file.

Constructor parameters for IPC:
    bool isSender_: Is Sender;
    string image_pipe: The pipe name used to pass the picture or tensor, the pipe does not need to be created in advance, but the parent directory must exist;
    string final_pipe: the pipe name used to receive the returned information;
    usec2c: Whether the receiver copies memory using the c2c interface. Defaults to false;
        From the perspective of pure communication, c2c is slower than d2d, but since d2d uses gdma, it is possible that c2c has less influence on inference.
        Sender does not need to be set up.
"""


def test_recv_bmi():
    handle = sail.Handle(0)
    bmcv = sail.Bmcv(handle)

    # Recipient
    ipc = sail.IPC(False, "/tmp/img", "/tmp/final", True, 10)

    while True:
        bmi_recv, channel_id, frame_id = ipc.receiveBMImage()
        save_name = "./recv" + str(frame_id) + ".jpg"
        bmcv.imwrite(save_name, bmi_recv)


def test_recv_tensor():
    handle = sail.Handle(0)
    bmcv = sail.Bmcv(handle)

    # Recipient
    ipc = sail.IPC(False, "/tmp/img", "/tmp/final", True, 10)

    while True:
        tensor, channel_id, frame_id = ipc.receiveTensor()
        bmi = bmcv.tensor_to_bm_image(tensor)
        save_name = "./recv" + str(frame_id) + ".jpg"
        bmcv.imwrite(save_name, bmi)


if __name__ == "__main__":
    test_recv_bmi()
