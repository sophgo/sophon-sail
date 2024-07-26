Tensor
______________


Tensor是模型推理的输入输出类型，包含了数据信息，实现内存管理。


构造函数
>>>>>>>>>>>>>>>>>>>>>

初始化Tensor,并为Tensor分配内存,如果需要进行系统内存和设备内存的同步,需要执行sync_d2s或sync_s2d


**接口形式:**
    .. code-block:: c

        Tensor(
            const std::vector<int>& shape={},
            bm_data_type_t          dtype=BM_FLOAT32);

        Tensor(
            Handle                  handle,
            const std::vector<int>& shape,
            bm_data_type_t          dtype=BM_FLOAT32,
            bool                    own_sys_data,
            bool                    own_dev_data);


**参数说明:**

* handle: Handle

设备标识Handle

* shape: std::vector<int>

设置Tensor的shape

* dtype: Dtype

Tensor的数据类型

* own_sys_data: bool

指示Tensor是否拥有system memory

* own_dev_data: bool

指示Tensor是否拥有device memory

**示例代码:**
    .. code-block:: c
    
        #include "tensor.h"
        
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor1,input_tensor2;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; // dtype can choose  BM_FLOAT32, BM_INT8, BM_UINT8, BM_INT32, BM_UINT32

            // init tensor
            input_tensor1 = std::make_shared<sail::Tensor>(input_shape, input_dtype);
            input_tensor2 = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true);
            
            return 0;  
        }



shape
>>>>>>>>>>>>>>>>>>>>>

获取Tensor的shape

**接口形式:**
    .. code-block:: c

        const std::vector<int>& shape() const;

**返回值说明:**

* tensor_shape : std::vector<int>

返回Tensor的shape的vector。

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor1;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor1 = std::make_shared<sail::Tensor>(input_shape, input_dtype);  

            // get shape
            std::vector<int> tensor_shape;
            tensor_shape = input_tensor1->shape();
            std::cout << "tensor shape: ";
            for(int i = 0; i < tensor_shape.size(); i++) {
                std::cout << tensor_shape[i] << " ";
            }
            std::cout << std::endl;
            return 0;  
        }

dtype
>>>>>>>>>>>>>>>>>>>>>

获取Tensor的数据类型

**接口形式:**
    .. code-block:: cpp

        bm_data_type_t dtype() const;

**返回值说明:**

* data_type : bm_data_type_t

返回Tensor的数据类型。

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor1;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor1 = std::make_shared<sail::Tensor>(input_shape, input_dtype);  
            
            // get dtype
            bm_data_type_t input_dtype_;
            input_dtype_ = input_tensor1->dtype();
            return 0;  
        }


scale_from
>>>>>>>>>>>>>>>>>>>>>

先对data按比例缩放，再将数据更新到Tensor的系统内存。
    
**接口形式:**
    .. code-block:: c

        void scale_from(float* src, float scale, int size);

**参数说明:**

* src: float*

数据的起始地址

* scale: float32

等比例缩放时的尺度。

* size: int

数据的长度

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor1,input_tensor2;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor1 = std::make_shared<sail::Tensor>(input_shape, input_dtype);
            
            // prepare data
            std::shared_ptr<float> src_ptr(
                    new float[10 * 10],
                    std::default_delete<float[]>());
            float * src_data = src_ptr.get();
            for(int i = 0; i < 10 * 10; i++) {
                src_data[i] = rand() % 255;
            }

            // scale data len is 99
            input_tensor1->scale_from(src_data, 0.1, 99); 

            return 0;
        }

scale_to
>>>>>>>>>>>>>>>>>>>>>

先对Tensor进行等比例缩放，再将数据返回到系统内存。
    
**接口形式:**
    .. code-block:: c

        void scale_to(float* dst, float scale);

        void scale_to(float* dst, float scale, int size);

**参数说明:**

* dst: float*

数据的起始地址。

* scale: float32

等比例缩放时的尺度。

* size: int

数据的长度。

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor1;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor1 = std::make_shared<sail::Tensor>(input_shape, input_dtype);  

            // prepare dst 
            float* dst = new float[100];

            // scale data len is 99
            input_tensor1->scale_to(dst, 0.1, 99); 

            // print scaled data
            for (int i = 0; i < size; ++i) {
                std::cout << dst[i] << " ";
            }
            std::cout << std::endl;
            delete[] dst; 

            return 0;  
        }

reshape
>>>>>>>>>>>>>>>>>>>>>

对Tensor进行reshape
    
**接口形式:**
    .. code-block:: c

        void reshape(const std::vector<int>& shape);

**参数说明:**

* shape: std::vector<int>

设置期望得到的新shape。

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor1;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor1 = std::make_shared<sail::Tensor>(input_shape, input_dtype);  

            // reshape from 10x10 to 2x50
            input_tensor1->reshape({2,50}); 

            // get shape
            std::vector<int> tensor_shape;
            tensor_shape = input_tensor1->shape();
            std::cout << "tensor new shape: ";
            for(int i = 0; i < tensor_shape.size(); i++) {
                std::cout << tensor_shape[i] << " ";
            }
            std::cout << std::endl;
            return 0;  
        }


own_sys_data
>>>>>>>>>>>>>>>>>>>>>

查询该Tensor是否拥有系统内存的数据指针。

**接口形式:**
    .. code-block:: c

        bool& own_sys_data();

**返回值说明:**

* judge_ret: bool

如果拥有系统内存的数据指针则返回True，否则False。

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true); // own sys mem:true, own dev mem:true
            // input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, false, true); // own sys mem:true, own dev mem:false

            // input_tensor: own sys or dev data 
            bool _own_sys_data = input_tensor->own_sys_data();
            std::cout << "input_tensor own_sys_data:" << _own_sys_data << std::endl;
            return 0;  
        }


own_dev_data
>>>>>>>>>>>>>>>>>>>>>

查询该Tensor是否拥有设备内存的数据

**接口形式:**
    .. code-block:: c

        bool& own_dev_data();

**返回值说明:**

* judge_ret : bool

如果拥有设备内存中的数据则返回True，否则False。

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true); // own sys mem:true, own dev mem:true
            // input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, false); // own sys mem:true, own dev mem:false

            // input_tensor: own sys or dev data 
            bool _own_dev_data = input_tensor->own_dev_data();
            std::cout << "input_tensor own_dev_data:" << _own_dev_data << std::endl;

            return 0;  
        }

sync_s2d
>>>>>>>>>>>>>>>>>>>>>

将Tensor中的数据从系统内存拷贝到设备内存。

**接口形式:**
    .. code-block:: c

        void sync_s2d();

        void sync_s2d(int size);

**参数说明:**

* size: int

将特定size字节的数据从系统内存拷贝到设备内存。

**接口形式:**
    .. code-block:: c

        void sync_s2d(Tensor* src, int offset_src, int offset_dst, int len);

**参数说明:**

* Tensor*: src

指定被拷贝的Tensor。

* offset_src: int

指定被拷贝Tensor上的数据偏移几个元素后开始拷贝。

* offset_dst: int

指定拷贝目标Tensor上的数据偏移几个元素后开始拷贝。

* len: int

指定拷贝长度，既拷贝的元素个数。

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true); // own sys mem:true, own dev mem:true
            // prepare data
            input_tensor->ones();

            // input_tensor -> sync_s2d(); // copy all data
            input_tensor -> sync_s2d(99); // copy part data

            // prepare another data: output_tensor, which is on sys mem, and don't have data
            // copy input_tensor to output_tensor
            std::shared_ptr<sail::Tensor> output_tensor;
            output_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true); 

            sail::Tensor& input_ref = *input_tensor;
            output_tensor -> sync_s2d(input_ref,2,3,10);

            // test if copy success
            // must copy to system memory and save to dst
            output_tensor -> sync_d2s(); 
            int size = 100;
            float* dst = new float[size];
            output_tensor->scale_to(dst, 1, size); 
            for (int i = 0; i < size; ++i) {
                std::cout << dst[i] << " ";
            }
            std::cout << std::endl;
            delete[] dst; 
            return 0;  
        }

sync_d2s
>>>>>>>>>>>>>>>>>>>>>

将Tensor中的数据从设备内存拷贝到系统内存。

**接口形式:**
    .. code-block:: c

        void sync_d2s();
          
        void sync_d2s(int size);

**参数说明:**

* size: int

将特定size字节的数据从设备内存拷贝到系统内存。

**接口形式:**
    .. code-block:: c

        void sync_d2s(Tensor* src, int offset_src, int offset_dst, int len);

**参数说明:**

* Tensor*: src

指定被拷贝的Tensor。

* offset_src: int

指定被拷贝Tensor上的数据偏移几个元素后开始拷贝。

* offset_dst: int

指定拷贝目标Tensor上的数据偏移几个元素后开始拷贝。

* len: int

指定拷贝长度，既拷贝的元素个数。

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, false, true); // own sys mem:false, own dev mem:true

            // prepare data
            input_tensor->ones();

            input_tensor -> sync_d2s(); // copy all data
            // input_tensor -> sync_d2s(99); // copy part data

            // prepare another data: output_tensor, which is on sys mem, and don't have data
            // copy input_tensor to output_tensor
            std::shared_ptr<sail::Tensor> output_tensor;
            output_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true); 

            sail::Tensor& input_ref = *input_tensor;
            output_tensor -> sync_d2s(input_ref,2,3,10);
            
            // test if copy success
            int size = 100;
            float* dst = new float[size];
            output_tensor->scale_to(dst, 1, size); 
            for (int i = 0; i < size; ++i) {
                std::cout << dst[i] << " ";
            }
            std::cout << std::endl;
            delete[] dst; 
            return 0;  
        }

sync_d2d
>>>>>>>>>>>>>>>>>>>>>

将另外一个Tensor设备内存上的数据拷贝到本Tensor的设备内存中。

**接口形式:**
    .. code-block:: c

        void sync_d2d(Tensor* src, int offset_src, int offset_dst, int len);

**参数说明:**

* Tensor*: src

指定被拷贝的Tensor。

* offset_src: int

指定被拷贝Tensor上的数据偏移几个元素后开始拷贝。

* offset_dst: int

指定拷贝目标Tensor上的数据偏移几个元素后开始拷贝。

* len: int

指定拷贝长度，既拷贝的元素个数。

**示例代码:**
    .. code-block:: c
    
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            sail::Handle handle_(dev_id+1);
            std::shared_ptr<sail::Tensor> input_tensor,output_tensor;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, false, true); // on dev0
            output_tensor = std::make_shared<sail::Tensor>(handle_, input_shape, input_dtype, false, true); // on dev1
            // prepare data
            input_tensor -> ones();
            
            // d2d
            sail::Tensor& input_ref = *input_tensor;
            output_tensor -> sync_d2d(input_ref,1,1,10); 

            return 0;  
        }


sync_d2d_stride
>>>>>>>>>>>>>>>>>>>>>

将另外一个Tensor设备内存上的数据拷贝到本Tensor的设备内存中。

**接口形式:**
    .. code-block:: c

        void sync_d2d_stride(Tensor* src, int stride_src, int stride_dst, int count);

**参数说明:**

* Tensor*: src

指定被拷贝的Tensor。

* stride_src: int

指定被拷贝Tensor上数据的stride。

* stride_dst: int

指定拷贝目标Tensor上数据的stride。stride_dst必须是1，除了stride_dst为4且stride_src为1且tensor数据类型大小为1字节的情况。

* count: int

指定拷贝的元素个数。需要保证count * stride_src <= tensor_src_size, count * stride_dst <= tensor_dst_size。


dump_data
>>>>>>>>>>>>>>>>>>>>>

将Tensor中的数据写入到指定文件中,如果需要进行系统内存和设备内存的同步,需要执行sync_d2s

**接口形式:**
    .. code-block:: c
          
        void dump_data(std::string file_name, bool bin = false);

**参数说明:**

* file_name: string 

写入文件的路径

* bin: bool

是否采用二进制的形式存储Tensor,默认false.

**示例代码:**
    .. code-block:: c

        int main() {  
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 
            // init tensor
            input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true); // own sys mem:true, own dev mem:true
            // prepare data
            input_tensor->ones();

            input_tensor->dump_data("dumped_tensor.txt",false);
            input_tensor->dump_data("dumped_tensor_bin.bin",true);
        
            return 0;  
        }

memory_set
>>>>>>>>>>>>>>>>>>>>>

用value的前N个字节填充Tensor的内存，N可以是1、2、4，取决于Tensor的dtype。

**接口形式:**
    .. code-block:: c

        void memory_set(void* value);


**参数说明:**

* value: void*

需要填充的值。

**示例代码:**
    .. code-block:: c

        void test_if_success(int size, std::shared_ptr<sail::Tensor> output_tensor){
            float* dst = new float[size];
            output_tensor->scale_to(dst, 1); 
            for (int i = 0; i < 100; ++i) {
                std::cout << dst[i] << " ";
            }
            std::cout << std::endl;
            delete[] dst; 
        }
        
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor;
            std::vector<int> input_shape = {3, 1920, 1080};
            bm_data_type_t input_dtype = BM_FLOAT32;
            input_tensor = std::make_shared<sail::Tensor>(handle,input_shape, input_dtype,true,true);

            float src_data = 1.1;
            // memory set to tensor
            input_tensor->memory_set(src_data);
            test_if_success(3 * 1920 * 1080,input_tensor); 

            return 0;
        }

memory_set
>>>>>>>>>>>>>>>>>>>>>

将本Tensor的数据全部置为c，在接口内部根据本Tensor的dtype对c做相应的类型转换，本接口可能会因为数据类型转换而带来精度损失，建议用上面的memory_set接口。

**接口形式:**
    .. code-block:: c

        void memory_set(float c);


**参数说明:**

* c: float

需要填充的值。

**示例代码:**
    .. code-block:: c

        void test_if_success(int size, std::shared_ptr<sail::Tensor> output_tensor){
            float* dst = new float[size];
            output_tensor->scale_to(dst, 1); 
            for (int i = 0; i < size; ++i) {
                std::cout << dst[i] << " ";
            }
            std::cout << std::endl;
            delete[] dst; 
        }
        
        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor;
            std::vector<int> input_shape = {1};
            bm_data_type_t input_dtype = BM_FLOAT32;
            input_tensor = std::make_shared<sail::Tensor>(handle,input_shape, input_dtype,true,true);

            float value_ = 1.1;
            input_tensor->memory_set(value_);
            test_if_success(1,input_tensor);

            return 0;
        }

zeros
>>>>>>>>>>>>>>>>>>>>>

将本Tensor的数据全部置为0。

**接口形式:**
    .. code-block:: c

        void zeros();

**示例代码:**
    .. code-block:: c

        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true); 
            // prepare data
            input_tensor->zeros();

            return 0;
        }


ones
>>>>>>>>>>>>>>>>>>>>>

将本Tensor的数据全部置为1。

**接口形式:**
    .. code-block:: c

        void ones();

**示例代码:**
    .. code-block:: c

        int main() {
            int dev_id = 0;
            int ret;
            sail::Handle handle(dev_id);
            std::shared_ptr<sail::Tensor> input_tensor;
            std::vector<int> input_shape = {10,10};
            bm_data_type_t input_dtype = BM_FLOAT32; 

            // init tensor
            input_tensor = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true); 
            // prepare data
            input_tensor->ones();

            return 0;
        }