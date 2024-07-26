import numpy as np
import sophon.sail as sail
import argparse
import os
import time

class Classifier(object):
    def __init__(self, bmodel_path, dev_id):
        #讲sail函数写入类
        self.engine = sail.Engine(bmodel_path,dev_id,sail.IOMode.SYSO)
        #获取句柄
        self.handle = self.engine.get_handle()
        self.bmcv = sail.Bmcv(self.handle)

        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]
        #获取数据类型，输入大小，规模
        self.input_dtype = self.engine.get_input_dtype(self.graph_name,self.input_name)
        self.input_shape = self.engine.get_input_shape(self.graph_name,self.input_name)
        self.input_scale = self.engine.get_input_scale(self.graph_name,self.input_name)
        self.dtype = sail.DATA_TYPE_EXT_1N_BYTE
        if self.input_dtype == sail.BM_FLOAT32:
            self.dtype = sail.DATA_TYPE_EXT_FLOAT32
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        #获取数据类型，输出大小，规模
        self.output_dtype = self.engine.get_output_dtype(self.graph_name,self.output_name)
        self.output_shape = self.engine.get_output_shape(self.graph_name,self.output_name)
        self.output_scale = self.engine.get_output_scale(self.graph_name,self.output_name)
        self.batch_size, self.c, self.net_h, self.net_w,  = self.input_shape
        #构造函数分配设备内存的张量。
        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True, True)

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]        
        self.a = [1/(255.*x) for x in self.std]
        self.b = [-x/y for x,y in zip(self.mean, self.std)]

        self.ab = []
        for i in range(3):
            self.ab.append(self.a[i]*self.input_scale)
            self.ab.append(self.b[i]*self.input_scale)

    def preprocess_bmcv(self, input_bmimg):
        output_bmimg = sail.BMImage(self.handle,self.net_h,self.net_w,sail.Format.FORMAT_RGB_PLANAR,self.img_dtype)
        if input_bmimg.format()==sail.Format.FORMAT_YUV420P:
            input_bmimg_bgr = self.bmcv.yuv2bgr(input_bmimg)
        else:
            input_bmimg_bgr = input_bmimg

        resize_bmimg = self.bmcv.resize(input_bmimg_bgr, self.net_w, self.net_h, sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        resize_bmimg_rgb = sail.BMImage(self.handle, self.net_h, self.net_w, \
                        sail.Format.FORMAT_RGB_PLANAR, resize_bmimg.dtype())
        self.bmcv.convert_format(resize_bmimg, resize_bmimg_rgb)
        self.bmcv.convert_to(resize_bmimg_rgb, output_bmimg, ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))
        return output_bmimg

    def inference(self,input_tensor):
        #进行张量推理
        self.engine.process(self.graph_name, {self.input_name: input_tensor},  {self.output_name:self.output_tensor})
        return self.output_tensor.asnumpy()

    def soft_max(self, z):
        t = np.exp(z)
        a = np.exp(z) / np.sum(t, axis=1).reshape(-1,1)
        return a

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Demo for classifier")
    parse.add_argument('--jpg_dir', default="../datasets/imagenet_val_1k/img", type=str, help="jpg path directory")#文件夹所在目录
    parse.add_argument('--bmodel_path', default="../models/BM1684X/resnet50_int8_1b.bmodel", type=str)
    parse.add_argument('--device_id', default=0, type=int)   
    opt= parse.parse_args()
    
    # 导入bmodel, dev_id
    instance = Classifier(opt.bmodel_path, opt.device_id)
    handle = instance.handle
    # 获取 Handle 中标识设备的序列码
    print(handle.get_sn())

    # 导入bmcv
    bmcv = sail.Bmcv(handle)

    # 获取模型信息
    batch_size = instance.batch_size
    print("dev_id: {}".format(opt.device_id))
    print("Batch Size: {}".format(batch_size))
    print("Network Input width: {}".format(instance.net_w))
    print("Network Input height: {}".format(instance.net_h))

    # 文件路径
    image_path = opt.jpg_dir
    file_list = os.listdir(image_path)
    image_list = []
    for file_name in file_list:
        #切割文件和文件名
        ext_name = os.path.splitext(file_name)[-1]
        if ext_name in ['.jpg','.png','.jpeg','.bmp','.JPEG','.JPG','.BMP']:
            #添加路径名
            image_list.append(os.path.join(image_path,file_name))
    if len(image_list) == 0:
        print("Can not find any pictures!")
        exit(1)

    if batch_size == 1:
        for image_name in image_list:
            decoder = sail.Decoder(image_name,True,opt.device_id)
            #读图
            img = decoder.read(handle)
            print(img.format(),'width:',img.width(), 'hight:',img.height() )               

            #preprocess_bmcv预处理
            output_image = instance.preprocess_bmcv(img)
            input_tensor = sail.Tensor(handle, instance.input_shape, instance.input_dtype, False, True)
            bmcv.bm_image_to_tensor(output_image,input_tensor)

            # 使用tensor推理
            start_time = time.time()
            output_npy = instance.inference(input_tensor)
            
            # 进行后处理
            end_time = time.time()
            output_npy = instance.soft_max(output_npy)
            prediction = output_npy[0]
            print('Maximum value:',prediction[np.argmax(prediction)] )
            print("top1:",np.argmax(prediction))
            print("top5:",np.argsort(-prediction)[:5])

            print("Decoder and Inference time use:{:.2f} ms, Batch size : 1".format((end_time-start_time)*1000))
    else:
        print("Error batch size: {}".format(batch_size))
        exit(1)