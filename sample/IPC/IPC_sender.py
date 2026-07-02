import sophon.sail as sail
import time
"""
Code for testing interprocess communication. Turn on two terminals, one running python3 IPC_sender.py, one running python3 IPC_receiver.py
The sender will pass a BMImage to the receiver, and when the receiver receives it, it will save the local file

Constructor parameters for IPC:
    bool isSender_: Is the sender or not;
    string image_pipe: Whether used to pass the picture or tensor pipe name, the pipe does not need to be created in advance, but the parent directory must exist;
    string final_pipe: The pipe name used to receive the returned information;
    usec2c: Whether the receiver copies memory using the c2c interface. Defaults to false;
        From the perspective of pure communication, c2c is slower than d2d, but since d2d uses gdma, it is possible that c2c has less influence on inference.
        Sender does not need to be set up.

Before running this routine, please ask the file path in the function
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
        # Sender
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
        # Sender
        ipc.sendTensor(tensor, 0, idx)
        idx += 1


if __name__ == "__main__":
    test_send_bmi()
