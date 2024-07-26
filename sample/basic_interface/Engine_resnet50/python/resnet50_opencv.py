import numpy as np
import sophon.sail as sail
import cv2
import argparse
import os
import time


save_path = "./"
# 定义一个类，主要功能是导入数据
class Classifier(object):
    def __init__(self, bmodel_path, tpu_id):
        #将sail函数写入类
        self.engine = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
        #获取句柄
        self.handle = self.engine.get_handle()
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]
        self.input_shape = self.engine.get_input_shape(self.graph_name,self.input_name)
        self.batch_size,  self.c, self.net_h, self.net_w = self.input_shape

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

    def preprocess_center(self, img):
        h, w, _ = img.shape
        if h != self.net_h or w != self.net_w:
            img = cv2.resize(img, (self.net_w, self.net_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img = (img/255-self.mean)/self.std
        img = np.transpose(img, (2, 0, 1))

        return np.expand_dims(img, axis=0)

    def inference(self, input_data):
        input = {self.input_name: input_data} 
        #进行张量推理
        output = np.array(list(self.engine.process(self.graph_name, input).values()))[0]
        # print(output)
        return output

    def soft_max(self, z):
        t = np.exp(z)
        a = t / np.sum(t, axis=1).reshape(-1,1)
        return a

    def postprocess(self, outputs, correct_labs):
        # 分类里模型一般无特殊后处理
        # 直接输出结果
        print('RESULT')
        # 是否需要soft_max取决网络中是否已经包含
        # 比如mobilenetv2的bmodel包含了soft_max层，则这里不再需要
        # 比如wsdan的bmodel不包含soft_max层
        outputs = self.soft_max(outputs)
        for i in range(outputs.shape[0]):
            test = outputs[i]
            print('correct_label',correct_labs)
            print('Maximum value:',test[np.argmax(test)] )
            print ("top1:",np.argmax(test),correct_labs[i])
            print("top5:",np.argsort(test)[-5:])
        return topk

if __name__ == "__main__":
    # argparse是python用于解析命令行参数和选项的标准模块，
    # argparse模块的作用是用于解析命令行参数。
    parse = argparse.ArgumentParser(description="Demo for classifier")
    parse.add_argument('--jpg_dir', default="../datasets/imagenet_val_1k/img", type=str, help="jpg path directory")#文件夹所在目录
    parse.add_argument('--bmodel_path', default="../models/BM1684X/resnet50_int8_1b.bmodel", type=str)
    parse.add_argument('--device_id', default=0, type=int)
    # 解析参数
    opt= parse.parse_args()
    #导入bmodel路径，id
    instance = Classifier(opt.bmodel_path, opt.device_id)
    handle = instance.handle
    batch_size = instance.batch_size


    topk=[0,0]
    all_time = 0.0
    print("TPU: {}".format(opt.device_id))
    print("Batch Size: {}".format(batch_size))
    print("Network Input width: {}".format(instance.net_w))
    print("Network Input height: {}".format(instance.net_h))
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
        img_nums = 0
        for image_name in image_list:
            print(image_name)
            correct_labs = []            
            img = cv2.imread(image_name)
            img_nums = img_nums + 1          
            preprocessed_image  = instance.preprocess_center(img)
                        
            start_time = time.time()
            output_npy = instance.inference(preprocessed_image)
            end_time = time.time()

            prediction = output_npy[0]
            print('Maximum value:',prediction[np.argmax(prediction)])
            print("top1:",np.argmax(prediction))
            print("top5:",np.argsort(-prediction)[:5])

            all_time = (end_time-start_time)*1000 +all_time
            print("Inference time use:{:.2f} ms, Batch size : 1".format((end_time-start_time)*1000))
        print('done!')

    else:
        print("Error batch size: {}".format(batch_size))
        exit(1)

