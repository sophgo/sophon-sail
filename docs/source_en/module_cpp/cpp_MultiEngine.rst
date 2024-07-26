MultiEngine
________________

A multi-threaded inference engine that implements multi-threaded inference for specific calculation graphs.

MultiEngine
>>>>>>>>>>>>>>>

Initialize MutiEngine。

**Interface:**
    .. code-block:: c

        MultiEngine(const std::string& bmodel_path,
                std::vector<int>   tpu_ids,
                bool               sys_out=true,
                int                graph_idx=0);
                
**Parameters:**

* bmodel_path: string

The file path where bmodel is located.

* tpu_ids: std::vector<int>

The ID of the Tensor Computing Processor visible in this MultiEngine.

* sys_out: bool

Whether to copy the results to system memory, the default is True

* graph_idx : int

The index of a specific computational graph.


**Sample:**
    .. code-block:: c

        #include "engine_multi.h"

        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            return 0;  
        }


set_print_flag
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Set whether to print debugging information.

**Interface:**
    .. code-block:: c

        void set_print_flag(bool print_flag);

**Parameters:**

* print_flag: bool

When True, debugging information is printed, otherwise it is not printed.

**Sample:**
    .. code-block:: c

        #include "engine_multi.h"
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            engine.set_print_flag(True);
            return 0;  
        }

set_print_time
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Setting whether to print is mainly time-consuming.

**Interface:**
    .. code-block:: c
        
        void set_print_time(bool print_flag);

**Parameters:**

* print_flag: bool

When it is True, printing is mainly time-consuming, otherwise it will not print.

**Sample:**
    .. code-block:: c

        #include "engine_multi.h"
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            engine.set_print_time(True);
            return 0;  
        }


get_device_ids
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the IDs of all available Tensor Computing Processors in MultiEngine.

**Interface:**
    .. code-block:: c

        std::vector<int> get_device_ids();

**Returns:**

* device_ids: std::vector<int>

Returns the IDs of visible Tensor Computing Processors

**Sample:**
    .. code-block:: c

        #include "engine_multi.h"
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            std::vector<int> device_ids = engine.get_device_ids();
            return 0;  
        }


get_graph_names
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the names of all loaded calculation graphs in MultiEngine.

**Interface:**
    .. code-block:: c

        std::vector<std::string> get_graph_names();

**Returns:**

* graph_names: std::vector<std::string>

The list of names of all calculation graphs in MultiEngine.

**Sample:**
    .. code-block:: c

        #include "engine_multi.h"
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            std::vector<std::string> graph_names = engine.get_graph_names();
            return 0;  
        }


get_input_names
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Get the names of all input Tensors in the selected calculation graph

**Interface:**
    .. code-block:: c

        std::vector<std::string> get_input_names(const std::string& graph_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

**Returns:**

* input_names: std::vector<std::string>

Returns a list of the names of all input Tensors in the selected computation graph.

**Sample:**
    .. code-block:: c

        #include "engine_multi.h"
        
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

Get the names of all output Tensors in the selected calculation graph.

**Interface:**
    .. code-block:: c

        std::vector<std::string> get_output_names(const std::string& graph_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

**Returns:**

* output_names: std::vector<std::string>

Returns a list of the names of all output Tensors in the selected calculation graph.

**Sample:**
    .. code-block:: c

        #include "engine_multi.h"
        
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

Query the shape of a specific input Tensor in the selected computational graph.

**Interface:**
    .. code-block:: c

        std::vector<int> get_input_shape(
            const std::string& graph_name,
            const std::string& tensor_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

* tensor_name: string

The name of the Tensor to be queried.

**Returns:**

* tensor_shape: std::vector<int>

The shape of the largest dimension in the input Tensor under this name.

**Sample:**
    .. code-block:: c

        #include "engine_multi.h"
        
        int main() {  
            std::vector<int> dev_id = {0, 1};  
            std::string bmodel_path = "your_bmodel.bmodel"
            sail::MultiEngine engine(bmodel_path, dev_id);  
            std::vector<std::string> graph_names = engine.get_graph_names();
            std::vector<std::string> input_names = engine.get_input_names(graph_names[0]);
            std::vector<int>  = engine.get_input_shape(graph_names[0],input_names[0]);
            return 0;  
        }

get_output_shape
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Query the shape of a specific output Tensor in the selected calculation graph.

**Interface:**
    .. code-block:: c

        std::vector<int> get_output_shape(
            const std::string& graph_name,
            const std::string& tensor_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

* tensor_name: string

The name of the Tensor to be queried.

**Returns:**

* tensor_shape: std::vector<int>

The shape of the output Tensor under this name.

**Sample:**
    .. code-block:: c

        #include "engine_multi.h"
        
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

Performing inference on a specific computational graph requires input data from system memory.

**Interface:**
    .. code-block:: c

        std::vector<std::map<std::string, Tensor*>> process(std::vector<std::map<std::string, Tensor*>>& input_tensors);

**Parameters:**

* input_tensors: std::vector<std::map<std::string, Tensor*> >

The input Tensors.

**Returns:**

* output_tensors: std::vector<std::map<std::string, Tensor*> >

Returns the result after inference.

**Sample:**
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