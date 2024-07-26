MultiEngine
________________

多线程的推理引擎，实现特定计算图的多线程推理。

MultiEngine
>>>>>>>>>>>>>>>

初始化MutiEngine。

**接口形式:**
    .. code-block:: c

        MultiEngine(const std::string& bmodel_path,
                std::vector<int>   tpu_ids,
                bool               sys_out=true,
                int                graph_idx=0);
                
**参数说明:**

* bmodel_path: string

bmodel所在的文件路径。

* tpu_ids: std::vector<int>

该MultiEngine可见的智能视觉深度学习处理器的ID。

* sys_out: bool

表示是否将结果拷贝到系统内存，默认为True

* graph_idx : int

特定的计算图的index。

**示例代码:**
    .. code-block:: c

        #include <sail/engine_multi.h>

        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            return 0;  
        }


set_print_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

设置是否打印调试信息。

**接口形式:**
    .. code-block:: c

        void set_print_flag(bool print_flag);

**参数说明:**

* print_flag: bool

为True时，打印调试信息，否则不打印。

**示例代码:**
    .. code-block:: c

        #include <sail/engine_multi.h>
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            engine.set_print_flag(true);
            return 0;  
        }

set_print_time
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

设置是否打印主要处理耗时。

**接口形式:**
    .. code-block:: c
        
        void set_print_time(bool print_flag);

**参数说明:**

* print_flag: bool

为True时，打印主要耗时，否则不打印。

**示例代码:**
    .. code-block:: c

        #include <sail/engine_multi.h>
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            engine.set_print_time(true);
            return 0;  
        }


get_device_ids
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取MultiEngine中所有可用的智能视觉深度学习处理器的id。

**接口形式:**
    .. code-block:: c

        std::vector<int> get_device_ids();

**返回值说明:**

* device_ids: std::vector<int>

返回可见的智能视觉深度学习处理器的ids

**示例代码:**
    .. code-block:: c

        #include <sail/engine_multi.h>
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            std::vector<int> device_ids = engine.get_device_ids();
            return 0;  
        }


get_graph_names
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

获取MultiEngine中所有载入的计算图的名称。

**接口形式:**
    .. code-block:: c

        std::vector<std::string> get_graph_names();

**返回值说明:**

* graph_names: std::vector<std::string>

MultiEngine中所有计算图的name的列表。

**示例代码:**
    .. code-block:: c

        #include <sail/engine_multi.h>
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            std::vector<std::string> graph_names = engine.get_graph_names();
            return 0;  
        }

get_input_names
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

        #include <sail/engine_multi.h>
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            std::vector<std::string> graph_names = engine.get_graph_names();
            std::vector<std::string> input_names = engine.get_input_names(graph_names[0]);
            return 0;  
        }


get_output_names
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

        #include <sail/engine_multi.h>
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            std::vector<std::string> graph_names = engine.get_graph_names();
            std::vector<std::string> output_name = engine.get_output_names(graph_names[0]);
            return 0;  
        }

get_input_shape
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

        #include <sail/engine_multi.h>
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            std::vector<std::string> graph_names = engine.get_graph_names();
            std::vector<std::string> input_names = engine.get_input_names(graph_names[0]);
            std::vector<int> input_shape = engine.get_input_shape(graph_names[0],input_names[0]);
            return 0;  
        }

get_output_shape
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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

        #include <sail/engine_multi.h>
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            std::vector<std::string> graph_names = engine.get_graph_names();
            std::vector<std::string> output_names = engine.get_output_names(graph_names[0]);
            std::vector<int> output_shape = engine.get_output_shape(graph_names[0],output_names[0]);
            return 0;  
        }


process
>>>>>>>>>>>>>>>>>>>

在特定的计算图上进行推理，需要提供系统内存的输入数据。

**接口形式:**
    .. code-block:: c

        std::vector<std::map<std::string, Tensor*>> process(std::vector<std::map<std::string, Tensor*>>& input_tensors);

**参数说明:**

* input_tensors: std::vector<std::map<std::string, Tensor*> >

输入的Tensors。

**返回值说明:**

* output_tensors: std::vector<std::map<std::string, Tensor*> >

返回推理之后的结果。

**示例代码:**
    .. code-block:: c
        
        #include <sail/engine_multi.h>

        int main() {  
            std::vector<int> dev_id = {0, 1};  

            std::string bmodel_path = "/home/jingyu/SAM-ViT-B_embedding_fp16_1b.bmodel";  
            sail::MultiEngine engine(bmodel_path, dev_id);  


            std::vector<std::string> graph_names = engine.get_graph_names();  
            std::vector<std::string> input_names = engine.get_input_names(graph_names[0]);  

            std::vector<int> input_shape = engine.get_input_shape(graph_names[0], input_names[0]);  

            // prepare one input tensor
            std::map<std::string, sail::Tensor*> input_tensors_map1;  
            for (const auto& input_name : input_names) {  
                sail::Tensor* input_tensor = new sail::Tensor(input_shape);
                input_tensors_map1[input_name] = input_tensor;  
            }  
            // prepare multi input...
            std::vector<std::map<std::string, sail::Tensor*>> input_tensors_vector;
            input_tensors_vector.push_back(input_tensors_map1);
            
            // get multi output
            auto output_tensors_vector = engine.process(input_tensors_vector);  

            for(auto& pair : input_tensors_map1) {  
                delete pair.second;  
            }  
            return 0;  
        }
