Tensor
______________


Tensor is the input and output type of model inference, which contains data information and implements memory management.


Constructor
>>>>>>>>>>>>>>>>>>>>>

Initialize Tensor and allocate memory for Tensor. If synchronization between system memory and device memory is required, sync_d2s or sync_s2d needs to be executed.


**Interface:**
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


**Parameters:**

* handle: Handle

The device identification Handle.

* shape: std::vector<int>

Set the shape of Tensor.

* dtype: Dtype

The data type of Tensor.

* own_sys_data: bool

Indicates whether the Tensor has system memory.

* own_dev_data: bool

Indicates whether the Tensor has device memory



**Sample:**
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

Get the shape of Tensor

**Interface:**
    .. code-block:: c

        const std::vector<int>& shape() const;

**Parameters:**

* tensor_shape : std::vector<int>

Returns a vector containing the shape of the Tensor.

**Sample:**
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

Get the dtype of Tensor

**Interface:**
    .. code-block:: cpp

        bm_data_type_t dtype() const;

**Return:**

* data_type : bm_data_type_t

dtype of Tensor

**Sample:**
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

First scale the data proportionally, and then update the data to the system memory of Tensor.
    
**Interface:**
    .. code-block:: c

        void scale_from(float* src, float scale, int size);

**Parameters:**

* src: float*

The starting address of the data.

* scale: float32

The scale when scaling proportionally.

* size: int

The length of the data.

**Sample:**
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

First scale the Tensor proportionally and then return the data to the system memory.
    
**Interface:**
    .. code-block:: c

        void scale_to(float* dst, float scale);

        void scale_to(float* dst, float scale, int size);

**Parameters:**

* dst: float*

The starting address of the data.

* scale: float32

The scale when scaling proportionally.

* size: int

The length of the data.


**Sample:**
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

Reshape Tensor
    
**Interface:**
    .. code-block:: c

        void reshape(const std::vector<int>& shape);

**Parameters:**

* shape: std::vector<int>

Set the desired new shape.

**Sample:**
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

Query whether the Tensor has a data pointer in system memory.

**Interface:**
    .. code-block:: c

        bool& own_sys_data();

**Returns:**

* judge_ret: bool

Returns True if it owns the data pointer of system memory, otherwise False.

**Sample:**
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

Query whether the Tensor has data in the device memory.

**Interface:**
    .. code-block:: c

        bool& own_dev_data();

**Returns:**

* judge_ret : bool

Returns True if the Tensor owns the data in device memory, False otherwise.

**Sample:**
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

Copy the data in Tensor from system memory to device memory.

**Interface:**
    .. code-block:: c

        void sync_s2d();

        void sync_s2d(int size);

**Parameters:**

* size: int

Copy data of a specific size bytes from system memory to device memory.

**Interface:**
    .. code-block:: c

        void sync_s2d(Tensor* src, int offset_src, int offset_dst, int len);

**Parameters:**

* Tensor*: src

Specifies the Tensor to be copied from.

* offset_src: int

Specifies the number of elements to offset in the source Tensor from where to start copying.

* offset_dst: int

Specifies the number of elements to offset in the destination Tensor from where to start copying.

* len: int

Specifies the length of the copy, i.e., the number of elements to copy.

**Sample:**
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

Copy the data in Tensor from device memory to system memory.

**Interface:**
    .. code-block:: c

        void sync_d2s();
          
        void sync_d2s(int size);

**Parameters:**

* size: int

Copies data of a specific size bytes from device memory to system memory.

**Interface:**
    .. code-block:: c

        void sync_d2s(Tensor* src, int offset_src, int offset_dst, int len);

**Parameters:**

* Tensor*: src

Specifies the Tensor to be copied from.

* offset_src: int

Specifies the number of elements to offset in the source Tensor from where to start copying.

* offset_dst: int

Specifies the number of elements to offset in the destination Tensor from where to start copying.

* len: int

Specifies the length of the copy, i.e., the number of elements to copy.


**Sample:**
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

Copies the data from another Tensor's device memory to this Tensor's device memory.

**Interface:**
    .. code-block:: c

        void sync_d2d(Tensor* src, int offset_src, int offset_dst, int len);

**Parameters:**

* Tensor*: src

Specifies the Tensor to be copied from.

* offset_src: int

Specifies the number of elements to offset in the source Tensor from where to start copying.

* offset_dst: int

Specifies the number of elements to offset in the destination Tensor from where to start copying.

* len: int

Specifies the length of the copy, i.e., the number of elements to copy.


**Sample:**
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

Copies the data from another Tensor's device memory to this Tensor's device memory in stride.

**Interface:**
    .. code-block:: c

        void sync_d2d_stride(Tensor* src, int stride_src, int stride_dst, int count);


**Parameters:**

* Tensor*: src

Specifies the Tensor to be copied from.

* stride_src: int

Specifies the stride of the the source Tensor.

* stride_dst: int

Specifies the stride of the destination Tensor.stride_dst must be 1, EXCEPT: stride_dst == 4 && stride_src == 1 && Tensor_type_size == 1

* count: int

Specifies the count of elements to copy.Ensure count * stride_src <= tensor_src_size, count * stride_dst <= tensor_dst_size.

dump_data
>>>>>>>>>>>>>>>>>>>>>

Write the data in Tensor to the specified file. If synchronization between system memory and device memory is required, sync_d2s needs to be executed.

**Interface:**
    .. code-block:: c
          
        void dump_data(std::string file_name, bool bin = false);

**Parameters:**

* file_name: string 

The path to the file to write to.

* bin: bool

Whether to store Tensor in binary form, default false.

**Sample:**
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

Fill the memory of the Tensor with the first N bytes of value, 
N can be 1, 2, 4, depending on the dtype of the Tensor.

**Interface:**
    .. code-block:: c

        void memory_set(void* value);


**Parameters:**

* value: void*

the value to fill.

**Sample:**
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

            // set data
            std::shared_ptr<float> src_ptr(
                    new float[3 * 1920 * 1080],
                    std::default_delete<float[]>());
            float * src_data = src_ptr.get();
            for(int i = 0; i < 3 * 1920 * 1080; i++) {
                src_data[i] = rand() % 255;
            }
            // print src_data
            for (int i = 0; i < 100; ++i) {
                std::cout << src_data[i] << " ";
            }
            std::cout << std::endl;

            // memory set to tensor
            input_tensor->memory_set(src_data);
            test_if_success(3 * 1920 * 1080,input_tensor); 

            return 0;
        }


memory_set
>>>>>>>>>>>>>>>>>>>>>

Fill memory with a scalar, it will be automatically converted to tensor's dtype. This interface may has precision loss due to data type conversion, It is recommended to use the interface above. 

**Interface:**
    .. code-block:: c

        void memory_set(float c);


**Parameters:**

* c: float

the value to fill.

**Sample:**
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

fill memory with zeros.

**Interface:**
    .. code-block:: c

        void zeros();

**Sample:**
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

fill memory with ones.

**Interface:**
    .. code-block:: c

        void ones();

**Sample:**
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