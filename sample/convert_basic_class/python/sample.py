import sophon.sail as sail
import numpy as np
import cv2

from PIL import Image

if __name__ == '__main__':
    file_path = './dog.jpg' # 请替换为您的文件路径
    dev_id = 0
    handle = sail.Handle(dev_id)
    decoder = sail.Decoder(file_path, False, dev_id)
    BMimg = sail.BMImage()
    ret = decoder.read(handle, BMimg)
    bmcv = sail.Bmcv(handle)

    # get bm_image
    bm_image = BMimg.data()

    # get BMimg width,height,dtype,format
    width = BMimg.width()
    height = BMimg.height()
    format = BMimg.format()



    # BMimage to MAT
    mat_from_bmimg = BMimg.asmat()
    cv2.imwrite('mat_from_bmimg.jpg', mat_from_bmimg)
    # BMimage to tensor 
    tensor = bmcv.bm_image_to_tensor(BMimg)



    # Mat to tensor
    tensor_from_mat = sail.Tensor(handle, mat_from_bmimg)
    # Mat to BMimage
    bmimg_from_mat = bmcv.mat_to_bm_image(mat_from_bmimg)
    bmcv.imwrite('bmimg_from_mat.jpg', bmimg_from_mat)



    # tensor to MAT
    mat_from_tensor = tensor.asnumpy()
    cv2.imwrite('mat_from_tensor.jpg', mat_from_tensor[0])
    # tensor to BMimage
    bmimg_from_tensor = bmcv.tensor_to_bm_image(tensor, False, 'nhwc')
    bmcv.imwrite('bmimg_from_tensor.jpg', bmimg_from_tensor)

