Engine
___________

Engine可以实现bmodel的加载与管理，是实现模型推理的主要模块。

构造函数
>>>>>>>>>>>>>>>>>>>>>

初始化Engine

**接口形式1:**

创建Engine实例，并不加载bmodel

    .. code-block:: c

        Engine(int tpu_id);
            
        Engine(const Handle&   handle);  

**参数说明1:**

* tpu_id: int

指定Engine实例使用的智能视觉深度学习处理器的id

* handle: Handle

指定Engine实例使用的设备标识Handle


**接口形式2:**

创建Engine实例并加载bmodel，需指定bmodel路径或内存中的位置。

    .. code-block:: c

        Engine(
           const std::string& bmodel_path,
           int                tpu_id,
           IOMode             mode);

        Engine(
           const std::string& bmodel_path,
           const Handle&      handle,
           IOMode             mode);

        Engine(
            const void* bmodel_ptr,
            size_t      bmodel_size,
            int         tpu_id,
            IOMode      mode);

        Engine(
           const void*        bmodel_ptr,
           size_t             bmodel_size,
           const Handle&      handle,
           IOMode             mode);

        

**参数说明2:**

* bmodel_path: string

指定bmodel文件的路径

* tpu_id: int

指定Engine实例使用的智能视觉深度学习处理器的id

* mode: IOMode

指定输入/输出Tensor所在的内存位置：系统内存或设备内存。

* bmodel_ptr: void*

bmodel在系统内存中的起始地址。

* bmodel_size: size_t

bmodel在内存中的字节数

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id);
            sail::Engine engine1(dev_id);
            sail::Engine engine2(handle);
            std::string bmodel_path = "your_bmodel.bmodel";
            sail::Engine engine3(bmodel_path, dev_id, sail::SYSI);
            sail::Engine engine4(bmodel_path, handle, sail::SYSI);

            // 打开文件输入流
            std::ifstream file(bmodel_path, std::ios::binary);
            // 获取文件大小
            file.seekg(0, std::ios::end);
            size_t bmodel_size = file.tellg();
            file.seekg(0, std::ios::beg);
            // 分配内存来存储模型数据
            char* bmodel_ptr = new char[bmodel_size];
            // 读取文件内容到内存中
            file.read(bmodel_ptr, bmodel_size);
            // 关闭文件输入流
            file.close();
            sail::Engine engine5(bmodel_ptr, bmodel_size, dev_id, sail::SYSI);
            delete [] bmodel_ptr;
            return 0;  
        }


get_handle
>>>>>>>>>>>>>>>>>>>>>

获取Engine中使用的设备句柄sail.Handle

**接口形式:**
    .. code-block:: c

        Handle get_handle();

**返回值说明:**

* handle: Handle

返回Engine中的设备句柄。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine1(dev_id);
            sail::Handle handle = engine1.get_handle();
            return 0;  
        }

load
>>>>>>>>>>>>>>>>>>>>>

将bmodel载入Engine中。

**接口形式1:**

指定bmodel路径，从文件中载入bmodel。

    .. code-block:: c

        bool load(const std::string& bmodel_path);

**参数说明1:**

* bmodel_path: string

bmodel的文件路径

**接口形式2:**

从系统内存中载入bmodel。

    .. code-block:: c

        bool load(const void* bmodel_ptr, size_t bmodel_size);

**参数说明2:**

* bmodel_ptr: void*

bmodel在系统内存中的起始地址。

* bmodel_size: size_t

bmodel在内存中的字节数。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            if (!engine.load(bmodel_path)) {
                // load failed
                std::cout << "Engine load bmodel "<< bmodel_file << "failed" << "\n";
                exit(0);
            }
            return 0;  
        }

get_graph_names
>>>>>>>>>>>>>>>>>>>>>

获取Engine中所有载入的计算图的名称。

**接口形式:**
    .. code-block:: c

        std::vector<std::string> get_graph_names();

**返回值说明:**

* graph_names: std::vector<std::string>

Engine中所有计算图的name的数组。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            return 0;  
        }


set_io_mode
>>>>>>>>>>>>>>>>>>>>>

设置Engine的输入/输出Tensor所在的内存位置：系统内存或设备内存。

**接口形式:**
    .. code-block:: c

        void set_io_mode(
            const std::string& graph_name,
            IOMode             mode);

**参数说明:**

* graph_name: string

需要配置的计算图的name。

* mode: IOMode

设置Engine的输入/输出Tensor所在的内存位置：系统内存或设备内存。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            engine.set_io_mode(bmodel_names[0], sail::SYSI);
            return 0;  
        }

get_input_names
>>>>>>>>>>>>>>>>>>>>>

获取选定计算图中所有输入Tensor的name

**接口形式:**
    .. code-block:: c

        std::vector<std::string> get_input_names(const std::string& graph_name);

**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

**返回值说明:**

* input_names: std::vector<std::string>

返回选定计算图中所有输入Tensor的name的列表。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            std::vector<std::string> input_names = engine.get_input_names(bmodel_names[0]);
            return 0;  
        }


get_output_names
>>>>>>>>>>>>>>>>>>>>>

获取选定计算图中所有输出Tensor的name。

**接口形式:**
    .. code-block:: c

        std::vector<std::string> get_output_names(const std::string& graph_name);

**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

**返回值说明:**

* output_names: std::vector<std::string>

返回选定计算图中所有输出Tensor的name的列表。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            std::vector<std::string> output_names = engine.get_output_names(bmodel_names[0]);
            return 0;  
        }

get_max_input_shapes
>>>>>>>>>>>>>>>>>>>>>

查询选定计算图中所有输入Tensor对应的最大shape。

在静态模型中，输入Tensor的shape是固定的，应等于最大shape。

在动态模型中，输入Tensor的shape应小于等于最大shape。

**接口形式:**
    .. code-block:: c

        std::map<std::string, std::vector<int>> get_max_input_shapes(
            const std::string& graph_name);

**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

**返回值说明:**

* max_shapes: std::map<std::string, std::vector<int> >

返回输入Tensor中的最大shape。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            std::map<std::string, std::vector<int>> input_max_shapes = engine.get_max_input_shapes(bmodel_names[0]);
            return 0;  
        }


get_input_shape
>>>>>>>>>>>>>>>>>>>>>

查询选定计算图中特定输入Tensor的shape。

**接口形式:**
    .. code-block:: c

        std::vector<int> get_input_shape(
            const std::string& graph_name,
            const std::string& tensor_name);
            
**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

* tensor_name: string

需要查询的Tensor的name。

**返回值说明:**

* tensor_shape: std::vector<int>

该name下的输入Tensor中的最大维度的shape。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            std::vector<std::string> input_names = engine.get_input_names(bmodel_names[0]);
            std::vector<int> input_shape_0 = engine.get_input_shape(bmodel_names[0],input_names[0]);
            return 0;  
        }

get_max_output_shapes
>>>>>>>>>>>>>>>>>>>>>>>

查询选定计算图中所有输出Tensor对应的最大shape。

在静态模型中，输出Tensor的shape是固定的，应等于最大shape。

在动态模型中，输出Tensor的shape应小于等于最大shape。

**接口形式:**
    .. code-block:: c

        std::map<std::string, std::vector<int>> get_max_output_shapes(
            const std::string& graph_name);

**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

**返回值说明:**

* std::map<std::string, std::vector<int> >

返回输出Tensor中的最大shape。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            std::map<std::string, std::vector<int>> output_max_shapes = engine.get_max_output_shapes(bmodel_names[0]);
            return 0;  
        }

get_output_shape
>>>>>>>>>>>>>>>>>>>>>

查询选定计算图中特定输出Tensor的shape。

**接口形式:**
    .. code-block:: c

        std::vector<int> get_output_shape(
            const std::string& graph_name,
            const std::string& tensor_name);

**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

* tensor_name: string

需要查询的Tensor的name。

**返回值说明:**

* tensor_shape: std::vector<int>

该name下的输出Tensor的shape。

**示例代码:**
    .. code-block:: c

        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            std::vector<std::string> output_names = engine.get_output_names(bmodel_names[0]);
            std::vector<int> output_shape_0 = engine.get_output_shape(bmodel_names[0],output_names[0]);
            return 0;  
        }

get_input_dtype
>>>>>>>>>>>>>>>>>>>>>

获取特定计算图的特定输入Tensor的数据类型。

**接口形式:**
    .. code-block:: c

        bm_data_type_t get_input_dtype(
            const std::string& graph_name,
            const std::string& tensor_name);

**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

* tensor_name: string

需要查询的Tensor的name。

**返回值说明:**

* datatype: bm_data_type_t

返回Tensor中数据的数据类型。

**示例代码:**
    .. code-block:: c

        
        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            std::vector<std::string> input_names = engine.get_input_names(bmodel_names[0]);
            std::vector<int> input_dype_0 = engine.get_input_dtype(bmodel_names[0],input_names[0]);
            return 0;  
        }

get_output_dtype
>>>>>>>>>>>>>>>>>>>>>

获取特定计算图的特定输出Tensor的数据类型。

**接口形式:**
    .. code-block:: c

        bm_data_type_t get_output_dtype(
            const std::string& graph_name,
            const std::string& tensor_name);

**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

* tensor_name: string

需要查询的Tensor的name。

**返回值说明:**

* datatype: bm_data_type_t

返回Tensor中数据的数据类型。

**示例代码:**
    .. code-block:: c

        
        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            std::vector<std::string> input_names = engine.get_input_names(bmodel_names[0]);
            std::vector<int> output_dype_0 = engine.get_output_dtype(bmodel_names[0],input_names[0]);
            return 0;  
        }

get_input_scale
>>>>>>>>>>>>>>>>>>>>>

获取特定计算图的特定输入Tensor的scale，只在int8模型中有效。

**接口形式:**
    .. code-block:: c

        float get_input_scale(
            const std::string& graph_name,
            const std::string& tensor_name);

**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

* tensor_name: string

需要查询的Tensor的name。

**返回值说明:**

* scale: float32

返回Tensor数据的scale。

**示例代码:**
    .. code-block:: c

        
        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            std::vector<std::string> input_names = engine.get_input_names(bmodel_names[0]);
            float input_scale_0 = engine.get_input_scale(bmodel_names[0],input_names[0]);
            return 0;  
        }

get_output_scale
>>>>>>>>>>>>>>>>>>>>>

获取特定计算图的特定输出Tensor的scale，只在int8模型中有效。

**接口形式:**
    .. code-block:: c

        float get_output_scale(
            const std::string& graph_name,
            const std::string& tensor_name);

**参数说明:**

* graph_name: string

设定需要查询的计算图的name。

* tensor_name: string

需要查询的Tensor的name。

**返回值说明:**

* scale: float32

返回Tensor数据的scale。


**示例代码:**
    .. code-block:: c

        
        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_name = engine.get_graph_names()[0];
            std::vector<std::string> input_names = engine.get_input_names(bmodel_name);
            float output_scale_0 = engine.get_output_scale(bmodel_names[0],input_names[0]);
            return 0;  
        }

process
>>>>>>>>>>>>>>>>>>>>>

在特定的计算图上进行前向推理。


**接口形式:**
    .. code-block:: c

        void process(
           const std::string&              graph_name,
           std::map<std::string, Tensor*>& input,
           std::map<std::string, Tensor*>& output,
           std::vector<int>                core_list = {});
        
        void process(
           const std::string&                       graph_name,
           std::map<std::string, Tensor*>&          input,
           std::map<std::string, std::vector<int>>& input_shapes,
           std::map<std::string, Tensor*>&          output,
           std::vector<int>                         core_list = {});

**参数说明:**

* graph_name: string

输入参数。特定的计算图name。

* input: std::map<std::string, Tensor*>

输入参数。所有的输入Tensor的数据。

* input_shapes : std::map<std::string, std::vector<int> >

输入参数。所有传入Tensor的shape。

* output: std::map<std::string, Tensor*>

输出参数。所有的输出Tensor的数据。

* core_list: std::vector<int>

输入参数。该参数仅对支持多核推理的处理器有效，可以选择推理时使用的core。设bmodel为对应的核数为N，若core_list为空则使用从core0开始的N个core做推理；若core_list的长度大于N，则使用core_list中对应的前N个core做推理。对于仅支持单核推理的处理器可忽略此参数。


**示例代码:**
    .. code-block:: c

        
        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> graph_names = engine.get_graph_names();
            std::vector<std::string> input_names = engine.get_input_names(graph_names[0]);
            std::vector<std::string> output_names = engine.get_input_names(graph_names[0]);

            std::vector<int> input_shape, output_shape;
            bm_data_type_t input_dtype, output_dtype;
            // allocate input and output tensors with both system and device memory
            // or you can use engine.create_input\output_tensors_map to create
            for (int i = 0; i < input_names.size(); i++) {
                input_shape = engine.get_input_shape(graph_names[0], input_names[i]);
                input_dtype = engine.get_input_dtype(graph_names[0], input_names[i]);
                input_tensor[i] = std::make_shared<sail::Tensor>(handle, input_shape, input_dtype, true, true);
                input_tensors[input_names[i]] = input_tensor[i].get();
            }
            for (int i = 0; i < output_names.size(); i++) {
                output_shape = engine.get_output_shape(graph_names[0], output_names[i]);
                output_dtype = engine.get_output_dtype(graph_names[0], output_names[i]);
                output_tensor[i] = std::make_shared<sail::Tensor>(handle, output_shape[i], output_dtype, true, true);
                output_tensors[output_names[i]] = output_tensor[i].get();
            }

            // process1
            engine.process(graph_names[0], input_tensors, output_tensors);  
            
            // process2
            std::map<std::string, std::vector<int>> input_shapes;
            for (const auto& input_name : input_names) {
                input_shape = engine.get_input_shape(bmodel_names, input_name);
                input_shapes[input_name] = input_shape;
            }

            engine.process(graph_names[0], input_tensors, input_shapes, output_tensors);
            return 0;  
        }


get_device_id
>>>>>>>>>>>>>>>>>>>>>

获取Engine中的设备id号

**接口形式:**
    .. code-block:: c

        int get_device_id() const;

**返回值说明:**

* tpu_id : int

返回Engine中的设备id号。

**示例代码:**
    .. code-block:: c

        
        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            int dev = engine.get_device_id();
            return 0;  
        }

create_input_tensors_map
>>>>>>>>>>>>>>>>>>>>>>>>>>>

创建输入Tensor的映射

**接口形式:**
    .. code-block:: c

        std::map<std::string, Tensor*> create_input_tensors_map(
            const std::string& graph_name, 
            int create_mode = -1);

**参数说明:**

* graph_name: string

特定的计算图name。

* create_mode: int

创建Tensor分配内存的模式。为0时只分配系统内存，为1时只分配设备内存，其他时则根据Engine中IOMode的配置分配。

**返回值说明:**

input: std::map<std::string, Tensor*>

返回字符串到tensor的映射。

**示例代码:**
    .. code-block:: c

        #include "engine.h"
        
        
        int main() {  
            int dev_id = 0;  
            sail::Handle handle(dev_id);  
            std::string bmodel_path = "your_bmodel.bmodel";  
            sail::Engine engine(bmodel_path, dev_id, sail::IOMode::SYSIO);  
            std::string graph_name = engine.get_graph_names()[0];  
            std::map<std::string, sail::Tensor> input_tensors_map = engine.create_input_tensors_map(graph_name);  
            return 0;  
        }


create_output_tensors_map
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

创建输入Tensor的映射，在python接口中为字典dict{string : Tensor}

**接口形式:**
    .. code-block:: c

        std::map<std::string, Tensor*> create_output_tensors_map(
            const std::string& graph_name, 
            int create_mode = -1);

**参数说明:**

* graph_name: string

特定的计算图name。

* create_mode: int

创建Tensor分配内存的模式。为0时只分配系统内存，为1时只分配设备内存，其他时则根据Engine中IOMode的配置分配。

**返回值说明:**

output: std::map<std::string, Tensor*>

返回字符串到tensor的映射。

**示例代码:**
    .. code-block:: c

        #include "engine.h"
        
        int main() {  
            int dev_id = 0;  
            sail::Handle handle(dev_id);  
            std::string bmodel_path = "your_bmodel.bmodel";  
            sail::Engine engine(bmodel_path, dev_id, sail::IOMode::SYSIO);  
            std::string graph_name = engine.get_graph_names()[0];  
            std::map<std::string, sail::Tensor> output_tensors_map = engine.create_output_tensors_map(graph_name);        
            return 0;  
        }