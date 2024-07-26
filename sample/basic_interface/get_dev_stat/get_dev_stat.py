import sophon.sail as sail

if __name__ == "__main__":
    device_id = 0

    board_temp=sail.get_board_temp(device_id)
    chip_temp=sail.get_chip_temp(device_id)
    dev_stat=sail.get_dev_stat(device_id)

    print("board_temp:",board_temp,"摄氏度")
    print("chip_temp:",chip_temp,"摄氏度")
    print("total mem (MB):",dev_stat[0],"used mem (MB):",dev_stat[1],"tpu_util (%):",dev_stat[2])