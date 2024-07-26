Engine
___________

Engine can load and manage bmodel and is the main module for model reasoning.

Constructor
>>>>>>>>>>>>>>>>>>>>>

Initialize Engine

**Interface 1:**

Create an Engine instance and do not load bmodel

    .. code-block:: c

        Engine(int tpu_id);
            
        Engine(const Handle&   handle);  

**Parameters 1:**

* tpu_id: int

Specify the Tensor Computing Processor id used by the Engine instance

* handle: Handle

Specify the device identification Handle used by the Engine instance


**Interface 2:**

To create an Engine instance and load bmodel, you need to specify the bmodel path or location in memory.

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

        

**Parameters 2:**

* bmodel_path: string

Specify the path to the bmodel file

* tpu_id: int

Specify the Tensor Computing Processor id used by the Engine instance

* mode: IOMode

Specify the memory location where the input/output Tensor is located: system memory or device memory.

* bmodel_ptr: void*

The starting address of bmodel in system memory.

* bmodel_size: size_t

The number of bytes in memory of bmodel

**Sample:**
    .. code-block:: c

        
        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Handle handle(dev_id);
            sail::Engine engine1(dev_id);
            sail::Engine engine2(handle);
            std::string bmodel_path = "your_bmodel.bmodel";
            sail::Engine engine3(bmodel_path, dev_id, SYSI);
            sail::Engine engine4(bmodel_path, handle, SYSI);

            // Open file input stream
            std::ifstream file(bmodel_path, std::ios::binary);
            // Get file size
            file.seekg(0, std::ios::end);
            size_t bmodel_size = file.tellg();
            file.seekg(0, std::ios::beg);
            // Allocate memory to store model data
            char* bmodel_ptr = new char[bmodel_size];
            // Read file content into memory
            file.read(bmodel_ptr, bmodel_size);
            // Close file input stream
            file.close();
            sail::Engine engine5(bmodel_ptr, bmodel_size, dev_id, sail::SYSI);
            delete [] bmodel_ptr;
            return 0;  
        }


get_handle
>>>>>>>>>>>>>>>>>>>>>

Get the device handle sail.Handle used in Engine

**Interface:**
    .. code-block:: c

        Handle get_handle();

**Returns:**

* handle: Handle

Returns the device handle in Engine.

**Sample:**
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

Load bmodel into Engine.

**Interface 1:**

Specify the bmodel path and load bmodel from the file.

    .. code-block:: c

        bool load(const std::string& bmodel_path);

**Parameters 1:**

* bmodel_path: string

The file path of bmodel.

**Interface 2:**

Load bmodel from system memory.

    .. code-block:: c

        bool load(const void* bmodel_ptr, size_t bmodel_size);

**Parameters 2:**

* bmodel_ptr: void*

The starting address of bmodel in system memory.

* bmodel_size: size_t

The number of bytes in memory of bmodel.

**Sample:**
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

Get the names of all calculation graphs loaded in Engine.

**Interface:**
    .. code-block:: c

        std::vector<std::string> get_graph_names();

**Returns:**

* graph_names: std::vector<std::string>

An array of names of all calculation graphs in Engine.

**Sample:**
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

Set the memory location where the input/output Tensor of Engine is located: system memory or device memory.

**Interface:**
    .. code-block:: c

        void set_io_mode(
            const std::string& graph_name,
            IOMode             mode);

**Parameters:**

* graph_name: string

The name of the calculation graph that needs to be configured.

* mode: IOMode

Set the memory location where the input/output Tensor of Engine is located: system memory or device memory.

**Sample:**
    .. code-block:: c

        
        #include "engine.h"

        int main() {  
            int dev_id = 0;
            sail::Engine engine(dev_id);
            std::string bmodel_path = "your_bmodel.bmodel";
            engine.load(bmodel_path);
            std::vector<std::string> bmodel_names = engine.get_graph_names();
            engine.set_io_mode(bmodel_names[0], SYSI);
            return 0;  
        }


get_input_names
>>>>>>>>>>>>>>>>>>>>>

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

Query the maximum shape corresponding to all input Tensors in the selected calculation graph.

In the static model, the shape of the input Tensor is fixed and should be equal to the maximum shape.

In the dynamic model, the shape of the input Tensor should be less than or equal to the maximum shape.

**Interface:**
    .. code-block:: c

        std::map<std::string, std::vector<int>> get_max_input_shapes(
            const std::string& graph_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

**Returns:**

* max_shapes: std::map<std::string, std::vector<int> >

Returns the largest shape in the input Tensor.

**Sample:**
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

Query the maximum shape corresponding to all output Tensors in the selected calculation graph.

In the static model, the shape of the output Tensor is fixed and should be equal to the maximum shape.

In the dynamic model, the shape of the output Tensor should be less than or equal to the maximum shape.

**Interface:**
    .. code-block:: c

        std::map<std::string, std::vector<int>> get_max_output_shapes(
            const std::string& graph_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

**Returns:**

* std::map<std::string, std::vector<int> >

Returns the largest shape in the output Tensor.

**Sample:**
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

Get the data type of a specific input Tensor for a specific computational graph.

**Interface:**
    .. code-block:: c

        bm_data_type_t get_input_dtype(
            const std::string& graph_name,
            const std::string& tensor_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

* tensor_name: string

The name of the Tensor to be queried.

**Returns:**

* datatype: bm_data_type_t

Returns the data type of the data in the Tensor.


**Sample:**
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

Get the data type of a specific output Tensor for a specific computational graph.

**Interface:**
    .. code-block:: c

        bm_data_type_t get_output_dtype(
            const std::string& graph_name,
            const std::string& tensor_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

* tensor_name: string

The name of the Tensor to be queried.

**Returns:**

* datatype: bm_data_type_t

Returns the data type of the data in the Tensor.

**Sample:**
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

Get the scale of a specific input Tensor of a specific calculation graph, only valid in the int8 model.

**Interface:**
    .. code-block:: c

        float get_input_scale(
            const std::string& graph_name,
            const std::string& tensor_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

* tensor_name: string

The name of the Tensor to be queried.

**Returns:**

* scale: float32

Returns the scale of the Tensor data.

**Sample:**
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

Get the scale of a specific output Tensor of a specific calculation graph, only valid in the int8 model.

**Interface:**
    .. code-block:: c

        float get_output_scale(
            const std::string& graph_name,
            const std::string& tensor_name);

**Parameters:**

* graph_name: string

Set the name of the calculation graph to be queried.

* tensor_name: string

The name of the Tensor to be queried.

**Returns:**

* scale: float32

Returns the scale of the Tensor data.

**Sample:**
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

Perform forward inference on a specific computational graph.


**Interface:**
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

**Parameters:**

* graph_name: string

Input parameter. A specific computational graph name.

* input: std::map<std::string, Tensor*>

Input parameter. All input Tensor data.

* input_shapes : std::map<std::string, std::vector<int> >

Input parameter. All shapes passed into Tensor.

* output: std::map<std::string, Tensor*>

Output parameter. All output Tensor data.

* core_list: std::vector<int> 

Input parameter. This parameter is only valid for processors that support multi-core inference, and the core used for inference can be selected. Set bmodel as the corresponding kernel number N, and if corelist is empty, use N cores starting from core0 for inference; If the length of corelist is greater than N, use the corresponding top N cores in corelist for inference. This parameter can be ignored for processors that only support single core inference.

**Sample:**
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

Get the device ID number in Engine

**Interface:**
    .. code-block:: c

        int get_device_id() const;

**Parameters:**

* tpu_id : int

Returns the device ID number in Engine.

**Sample:**
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

Create a map of the input Tensor

**Interface:**
    .. code-block:: c

        std::map<std::string, Tensor*> create_input_tensors_map(
            const std::string& graph_name, 
            int create_mode = -1);

**Parameters:**

* graph_name: string

The name of specific computational graph.

* create_mode: int

Create a pattern for Tensor to allocate memory. When it is 0, only system memory is allocated. When it is 1, only device memory is allocated. Otherwise, it is allocated according to the IOMode configuration in Engine.

**Returns:**

input: std::map<std::string, Tensor*>

Returns the mapping of strings to tensors.

**Sample:**
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

Create a mapping of the input Tensor, which is a dictionary dict{string : Tensor} in the python interface

**Interface:**
    .. code-block:: c

        std::map<std::string, Tensor*> create_output_tensors_map(
            const std::string& graph_name, 
            int create_mode = -1);

**Parameters:**

* graph_name: string

The name of specific computational graph.

* create_mode: int

Create a pattern for Tensor to allocate memory. When it is 0, only system memory is allocated. When it is 1, only device memory is allocated. Otherwise, it is allocated according to the IOMode configuration in Engine.

**Returns:**

output: std::map<std::string, Tensor*>

Returns a mapping of strings to tensors.

**Sample:**
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