SAIL Python API
===============

SAIL use "pybind11" to wrap python interfaces, support python3.5, python3.6, python3.7, python3.8

Basic function
______________

.. code-block:: python

        def get_available_tpu_num():
            """ Get the number of available Tensor Computing Processors.

            Returns
            -------
            tpu_num : int
                Number of available Tensor Computing Processors
            """

        def set_print_flag(print_flag):
            """ Print main process time use.

            Parameters
            ----------
            print_flag : bool
                if print_flag is true, print main process time use, Otherwise not print.
            """
        
        def set_dump_io_flag(dump_io_flag):
            """ Dump input data and output data.

            Parameters
            ----------
            dump_io_flag : bool
                if dump_io_flag is true, dump input data and output data, Otherwise not dump.
            """
            
        def set_decoder_env(env_name, env_value):
            """ Set Decoder environment, must set befor Decoder Constructor, else use default values
            
            Parameters
            ----------
            env_name: str
                Environment name,
                name list: refcounted_frames, extra_frame_buffer_num, rtsp_transport, stimeout, \
                    rtsp_flags, buffer_size, max_delay, probesize, analyzeduration.
            env_value: str
                Environment value.
            """

        def base64_encode(handle: Handle, input_bytes: bytes) -> bytes:
            """ Encode byte data into base64 and return the encoded data as bytes
                       
            Parameters
            ----------
            handle: Handle
                The handle of the device, created using sail.Handle(dev_id).

            input_bytes: bytes
                The byte data to be encoded.

            Returns
            --------
            bytes
                The byte data encoded in base64.
            """
        
        def base64_decode(handle: Handle, encode_bytes: bytes) -> bytes:
            """ Decode the byte-encoded data in base64 and return the decoded byte data.
                       
            Parameters
            ----------
            handle: Handle
                The handle of the device, created using sail.Handle(dev_id).

            encode_bytes: bytes
                The byte-encoded data in base64.

            Returns
            --------
            bytes
                The byte data decoded from base64.
            """
        
        def base64_encode_array(handle: Handle, input_arr: numpy.ndarray) -> bytes:
            """ Encode a numpy.array into base64, generating byte-encoded data.
                       
            Parameters
            ----------
            handle: Handle
                The handle of the device, created using sail.Handle(dev_id).

            input_arr: numpy.ndarray
                The numpy.ndarray data to be encoded.

            Returns
            --------
            bytes
                The byte data encoded in base64.
            """
        
        def base64_decode_asarray(handle: Handle, encode_arr_bytes: bytes, array_type:str = "uint8") -> numpy.ndarray:
            """ Decode base64 to generate numpy.array data..
                       
            Parameters
            --------------
            handle: Handle
                The handle of the device, created using sail.Handle(dev_id).

            encode_arr_bytes: bytes
                The byte data of the numpy.ndarray encoded in base64.
            
            array_type: str
                The data type of numpy.ndarray, default uint8, supports float, uint8, int8, int16, int32, int64.
            
            Returns
            ----------
            numpy.array
                The one-dimensional numpy.array array decoded from base64.
            """
        
        def get_tpu_util(dev_id):
            """ Get the processor percent utilization of the specified device
            
            Parameters
            ----------
            dev_id: int
                Device id
            """

        def get_vpu_util(dev_id):
            """ Get the VPU percent utilization of the specified device
            
            Parameters
            ----------
            dev_id: int
                Device id
            """

        def get_vpp_util(dev_id):
            """ Get the VPP percent utilization of the specified device
            
            Parameters
            ----------
            dev_id: int
                Device id
            """
        def get_board_temp(dev_id)
            """ Get the temperature of the specified device
            
            Parameters
            ----------
            dev_id: int
                Device id
            """

        def get_chip_temp(dev_id)
            """ Get the temperature of the specified device
            
            Parameters
            ----------
            dev_id: int
                Device id
            """

        def get_dev_stat(dev_id)
            """ Get the device status of the specified device
            
            Parameters
            ----------
            dev_id: int
                Device id
            """

sail.Data type
_______________

.. code-block:: python

        # Data type for float32
        sail.Dtype.BM_FLOAT32
        # Data type for int8
        sail.Dtype.BM_INT8
        # Data type for uint8
        sail.Dtype.BM_UINT8
        # Data type for int32
        sail.Dtype.BM_INT32
        # Data type for uint32
        sail.Dtype.BM_UINT32

sail.PaddingAtrr
___________________

.. code-block:: python

        def __init__():
            """ Constructor with no parameters. """

        def __init__(stx, sty, width, height, r, g, b):
            """ Constructor PaddingAtrr. 
            
            Parameters
            ----------
            stx : int
                Offset x information relative to the origin of dst image
            sty : int
                Offset y information relative to the origin of dst image
            width : int
                The width after resize
            height : int
                The height after resize
            r : int
                Pixel value information of R channel
            g : int
                Pixel value information of G channel
            b : int
                Pixel value information of B channel
            """

        def set_stx(stx):
            """ set offset stx.

            Parameters
            ----------
            stx : int
                Offset x information relative to the origin of dst image
            """

        def set_sty(sty):
            """ set offset sty.

            Parameters
            ----------
            sty : int
                Offset y information relative to the origin of dst image
            """

        def set_w(width):
            """ set widht.

            Parameters
            ----------
            width : int
                The width after resize
            """

        def set_h(height):
            """ set height.

            Parameters
            ----------
            height : int
                The height after resize
            """

        def set_r(r):
            """ set R.

            Parameters
            ----------
            r : int
                Pixel value information of R channel
            """

        def set_g(g):
            """ set G.

            Parameters
            ----------
            g : int
                Pixel value information of G channel
            """

        def set_g(b):
            """ set B.

            Parameters
            ----------
            b : int
                Pixel value information of B channel
            """

sail.Handle
___________

.. code-block:: python

    def __init__(tpu_id):
        """ Constructor handle instance

        Parameters
        ----------
        tpu_id : int
            create handle with tpu Id
        """

    def get_device_id():
        """ Get tpu id of this handle. 
        
        Returns
        -------
        tpu_id : int
            tpu id of this handle.
        """

    def get_sn():
        """ Get serial number of this handle.

        Returns
        -------
        serial_number : str
            serial number of this handle.
        """
    
    def get_target():
        """ Get Tensor Computing Processor type of this handle.

        Returns
        -------
        tpu_chip_type : str
            Tensor Computing Processor type of this handle.
        """

sail.IOMode
___________

.. code-block:: python

        # Input tensors are in system memory while output tensors are in device memory
        sail.IOMode.SYSI
        # Input tensors are in device memory while output tensors are in system memory.
        sail.IOMode.SYSO
        # Both input and output tensors are in system memory.
        sail.IOMode.SYSIO
        # Both input and output tensors are in device memory.
        sail.IOMode.DEVIO


sail.bmcv_resize_algorithm
___________________________

.. code-block:: python

        sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST
        sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR
        sail.bmcv_resize_algorithm.BMCV_INTER_BICUBIC

sail.Format
______________

.. code-block:: python

        sail.Format.FORMAT_YUV420P
        sail.Format.FORMAT_YUV422P
        sail.Format.FORMAT_YUV444P
        sail.Format.FORMAT_NV12
        sail.Format.FORMAT_NV21
        sail.Format.FORMAT_NV16
        sail.Format.FORMAT_NV61
        sail.Format.FORMAT_NV24
        sail.Format.FORMAT_RGB_PLANAR
        sail.Format.FORMAT_BGR_PLANAR
        sail.Format.FORMAT_RGB_PACKED
        sail.Format.FORMAT_BGR_PACKED
        sail.Format.FORMAT_RGBP_SEPARATE
        sail.Format.FORMAT_BGRP_SEPARATE
        sail.Format.FORMAT_GRAY
        sail.Format.FORMAT_COMPRESSED

sail.ImgDtype
______________

.. code-block:: python

        sail.ImgDtype.DATA_TYPE_EXT_FLOAT32
        sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE
        sail.ImgDtype.DATA_TYPE_EXT_4N_BYTE
        sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE_SIGNED
        sail.ImgDtype.DATA_TYPE_EXT_4N_BYTE_SIGNED


sail.Dtype
______________

.. code-block:: python

        sail.Dtype.BM_FLOAT32
        sail.Dtype.BM_INT8
        sail.Dtype.BM_UINT8
        sail.Dtype.BM_INT32
        sail.Dtype.BM_UINT32    


sail.Tensor
___________

**1). Tensor**

    .. code-block:: python

        def __init__(handle, data, own_sys_data=True):
            """ Constructor allocates device memory of the tensor.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            array_data : numpy.array
                Tensor ndarray data, dtype can be np.float32, np.int8 or np.uint8
            own_sys_data : bool, default: True
                Indicator of whether own system memory, If false, the memory will be copied to device directly  
            """

        def __init__(handle, shape, dtype, own_sys_data, own_dev_data):
            """ Constructor allocates system memory and device memory of the tensor.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            shape : tuple
                Tensor shape
            dytpe : sail.Dtype
                Data type
            own_sys_data : bool
                Indicator of whether own system memory
            own_dev_data : bool
                Indicator of whether own device memory
            """

**2). shape**

    .. code-block:: python

        def shape():
            """ Get shape of the tensor.

            Returns
            -------
            tensor_shape : list
                Shape of the tensor
            """

**3). asnumpy**

    .. code-block:: python

        def asnumpy():
            """ Get system data of the tensor.

            Returns
            -------
            data : numpy.array
                System data of the tensor, dtype can be np.float32, np.int8
                or np.uint8 with respective to the dtype of the tensor.
            """

        def asnumpy(shape):
            """ Get system data of the tensor.

            Parameters
            ----------
            shape : tuple
                Tensor shape want to get

            Returns
            -------
            data : numpy.array
                System data of the tensor, dtype can be np.float32, np.int8
                or np.uint8 with respective to the dtype of the tensor.
            """

**4). update_data**

    .. code-block:: python

        def update_data(data):
            """ Update system data of the tensor. The data size should not exceed
                the tensor size, and the tensor shape will not be changed.

            Parameters
            -------
            data : numpy.array
                Data.
            """

**5). scale_from**

    .. code-block:: python

        def scale_from(data, scale):
            """ Scale data to tensor in system memory.

            Parameters
            -------
            data : numpy.array with dtype of float32
                Data.
            scale : float32
                Scale value.
            """

**6). scale_to**

    .. code-block:: python

        def scale_to(scale):
            """ Scale tensor to data in system memory.

            Parameters
            -------
            scale : float32
                Scale value.

            Returns
            -------
            data : numpy.array with dtype of float32
                Data.
            """

        def scale_to(scale, shape):
            """ Scale tensor to data in system memory.

            Parameters
            -------
            scale : float32
                Scale value.
            shape : tuple
                Tensor shape want to get

            Returns
            -------
            data : numpy.array with dtype of float32
                Data.
            """

**7). reshape**

    .. code-block:: python

        def reshape(shape):
            """ Reset shape of the tensor.

            Parameters
            -------
            shape : list
                New shape of the tensor
            """

**8). own_sys_data**

    .. code-block:: python

        def own_sys_data():
            """ Judge if the tensor owns data pointer in system memory.

            Returns
            -------
            judge_ret : bool
                True for owns data pointer in system memory.
            """

**9). own_dev_data**

    .. code-block:: python

        def own_dev_data():
            """ Judge if the tensor owns data in device memory.

            Returns
            -------
            judge_ret : bool
                True for owns data in device memory.
            """

**10). sync_s2d**

    .. code-block:: python

        def sync_s2d():
            """ Copy data from system memory to device memory.
            """

        def sync_s2d(size):
            """ Copy data from system memory to device memory with specified size.

            Parameters
            ----------
            size : int
                Byte size to be copied
            """

**11). sync_d2s**

    .. code-block:: python

        def sync_d2s():
            """ Copy data from device memory to system memory.
            """

        def sync_d2s(size):
            """ Copy data from device memory to system memory with specified size.

            Parameters
            ----------
            size : int
                Byte size to be copied
            """

**12). dump_data**

    .. code-block:: python

        def dump_data(file_name,bin=False)
            """ Dump Tensor data to file.

            Parameters
            ----------
            file_name : str
                file path to dump tensor
            
            bin : bool
                binary format, default False.
            """

**13). dtype**

    .. code-block:: python

        def dtype()
            """ return tensor's data type
            Returns
            -------
            scale: sail.Dtype
                Data type of tensor
            """

sail.Engine
___________

**1). Engine**

    .. code-block:: python

        def __init__(tpu_id):
            """ Constructor does not load bmodel.

            Parameters
            ----------
            tpu_id : int
                Tensor Computing Processor ID. You can use bm-smi to see available IDs
            """

        def __init__(handle):
            """ Constructor does not load bmodel.

            Parameters
            ----------
            hanle : Handle
               A Handle instance
            """

        def __init__(bmodel_path, tpu_id, mode):
            """ Constructor loads bmodel from file.

            Parameters
            ----------
            bmodel_path : str
                Path to bmodel
            tpu_id : int
                Tensor Computing Processor ID. You can use bm-smi to see available IDs
            mode : sail.IOMode
                Specify the input/output tensors are in system memory
                or device memory
            """

        def __init__(bmodel_bytes, bmodel_size, tpu_id, mode):
            """ Constructor using default input shapes with bmodel which
            loaded in memory

            Parameters
            ----------
            bmodel_bytes : bytes
                Bytes of  bmodel in system memory
            bmodel_size : int
                Bmodel byte size
            tpu_id : int
                Tensor Computing Processor ID. You can use bm-smi to see available IDs
            mode : sail.IOMode
                Specify the input/output tensors are in system memory
                or device memory
            """

**2). get_handle**

    .. code-block:: python

        def get_handle():
            """ Get Handle instance.

            Returns
            -------
            handle: sail.Handle
               Handle instance
            """

**3). load**

    .. code-block:: python

        def load(bmodel_path):
            """ Load bmodel from file.

            Parameters
            ----------
            bmodel_path : str
                Path to bmodel
            """

        def load(bmodel_bytes, bmodel_size):
            """ Load bmodel from file.

            Parameters
            ----------
            bmodel_bytes : bytes
                Bytes of  bmodel in system memory
            bmodel_size : int
                Bmodel byte size
            """

**4). get_graph_names**

    .. code-block:: python

        def get_graph_names():
            """ Get all graph names in the loaded bmodels.

            Returns
            -------
            graph_names : list
                Graph names list in loaded context
            """

**5). set_io_mode**

    .. code-block:: python

        def set_io_mode(graph_name, mode):
            """ Set IOMode for a graph.

            Parameters
            ----------
            graph_name: str
                The specified graph name
            mode : sail.IOMode
                Specified io mode
            """

**6). get_input_names**

    .. code-block:: python

        def get_input_names(graph_name):
            """ Get all input tensor names of the specified graph.

            Parameters
            ----------
            graph_name : str
                Specified graph name

            Returns
            -------
            input_names : list
                All the input tensor names of the graph
            """

**7). get_output_names**

    .. code-block:: python

        def get_output_names(graph_name):
            """ Get all output tensor names of the specified graph.

            Parameters
            ----------
            graph_name : str
                Specified graph name

            Returns
            -------
            input_names : list
                All the output tensor names of the graph
            """

**8). get_max_input_shapes**

    .. code-block:: python

        def get_max_input_shapes(graph_name):
            """ Get max shapes of input tensors in a graph.
                For static models, the max shape is fixed and it should not be changed.
                For dynamic models, the tensor shape should be smaller than or equal to
                the max shape.

            Parameters
            ----------
            graph_name : str
                The specified graph name

            Returns
            -------
            max_shapes : dict {str : list}
                Max shape of the input tensors
            """

**9). get_input_shape**

    .. code-block:: python

        def get_input_shape(graph_name, tensor_name):
            """ Get the maximum dimension shape of an input tensor in a graph.
                There are cases that there are multiple input shapes in one input name, 
                This API only returns the maximum dimension one for the memory allocation 
                in order to get the best performance.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified input tensor name

            Returns
            -------
            tensor_shape : list
                The maxmim dimension shape of the tensor
            """

**10). get_max_output_shapes**

    .. code-block:: python

        def get_max_output_shapes(graph_name):
            """ Get max shapes of input tensors in a graph.
                For static models, the max shape is fixed and it should not be changed.
                For dynamic models, the tensor shape should be smaller than or equal to
                the max shape.

            Parameters
            ----------
            graph_name : str
                The specified graph name

            Returns
            -------
            max_shapes : dict {str : list}
                Max shape of the output tensors
            """

**11). get_output_shape**

    .. code-block:: python

        def get_output_shape(graph_name, tensor_name):
            """ Get the shape of an output tensor in a graph.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            tensor_shape : list
                The shape of the tensor
            """

**12). get_input_dtype**

    .. code-block:: python

        def get_input_dtype(graph_name, tensor_name)
            """ Get scale of an input tensor. Only used for int8 models.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            scale: sail.Dtype
                Data type of the input tensor
            """

**13). get_output_dtype**

    .. code-block:: python

        def get_output_dtype(graph_name, tensor_name)
            """ Get scale of an output tensor. Only used for int8 models.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            scale: sail.Dtype
                Data type of the output tensor
            """

**14). get_input_scale**

    .. code-block:: python

        def get_input_scale(graph_name, tensor_name)
            """ Get scale of an input tensor. Only used for int8 models.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            scale: float32
                Scale of the input tensor
            """

**15). get_output_scale**

    .. code-block:: python

        def get_output_scale(graph_name, tensor_name)
            """ Get scale of an output tensor. Only used for int8 models.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            scale: float32
                Scale of the output tensor
            """

**16). process**

    .. code-block:: python

        def process(graph_name, input_tensors):
            """ Inference with provided system data of input tensors.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            input_tensors : dict {str : numpy.array}
                Data of all input tensors in system memory

            Returns
            -------
            output_tensors : dict {str : numpy.array}
                Data of all output tensors in system memory
            """

        def process(graph_name, input_tensors, output_tensors):
            """ Inference with provided input and output tensors.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            input_tensors : dict {str : sail.Tensor}
                Input tensors managed by user
            output_tensors : dict {str : sail.Tensor}
                Output tensors managed by user
            """

        def process(graph_name, input_tensors, input_shapes, output_tensors):
            """ Inference with provided input tensors, input shapes and output tensors.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            input_tensors : dict {str : sail.Tensor}
                Input tensors managed by user
            input_shapes : dict {str : list}
                Shapes of all input tensors
            output_tensors : dict {str : sail.Tensor}
                Output tensors managed by user
            """

**17). get_device_id**

    .. code-block:: python

        def get_device_id():
            """ Get device id of this engine 
            
            Returns
            ----------
            tpu_id : int
                Tensor Computing Processor id of this engine 
   
            """

**18). create_input_tensors_map**

    .. code-block:: python

        def create_input_tensors_map(graph_name, create_mode):
            """ Create input tensors map, according to and bmodel.

            Parameters:
            ----------
            graph_name : str
                The specified graph name.
            create_mode: int
                Tensor Create mode,
                case 0: only allocate system memory; 
                case 1: only allocate device memory;
                case other: according to engine IOMode.

            Returns
            -------
            output: dict[str,Tensor]
                Output result.
            """

**19). create_output_tensors_map**

    .. code-block:: python

        def create_output_tensors_map(graph_name, create_mode):
            """ Create output tensors map, according to and bmodel.

            Parameters:
            ----------
            graph_name : str
                The specified graph name.
            create_mode: int
                Tensor Create mode,
                case 0: only allocate system memory; 
                case 1: only allocate device memory;
                case other: according to engine IOMode.

            Returns
            -------
            output: dict[str,Tensor]
                Output result.
            """

sail.MultiEngine
________________
**1). MultiEngine**

    .. code-block:: python

        def __init__(bmodel_path, device_ids, sys_out, graph_idx):
            """ Constructor load bmodel.

            Parameters
            ----------
            bmodel_path : str
                Path to bmodel
            device_ids : list[int]    
                Tensor Computing Processor ID. You can use bm-smi to see available IDs
            sys_out : bool, default: True
                The flag of copy result to system memory.
            graph_idx : int, default: 0
                The specified graph index
            """

**2). set_print_flag**

    .. code-block:: python

        def set_print_flag(print_flag):
            """ Print debug messages.

            Parameters
            ----------
            print_flag : bool
                if print_flag is true, print debug messages
            """

**3). set_print_time**

    .. code-block:: python
        
        def set_print_time(print_flag):
            """ Print main process time use.

            Parameters
            ----------
            print_flag : bool
                if print_flag is true, print main process time use, Otherwise not print.
            """

**4). get_device_ids**

    .. code-block:: python

        def get_device_ids():
            """ Get device ids of this MultiEngine.
            
            Returns
            -------
            device_ids : list[int]    
                Tensor Computing Processor ids of this MultiEngine.
            """

**5). get_graph_names**

    .. code-block:: python

        def get_graph_names()
            """ Get all graph names in the loaded bmodels.

            Returns
            -------
            graph_names : list
                Graph names list in loaded context
            """

**6). get_input_names**

    .. code-block:: python

        def get_input_names(graph_name):
            """ Get all input tensor names of the specified graph.

            Parameters
            ----------
            graph_name : str
                Specified graph name

            Returns
            -------
            input_names : list
                All the input tensor names of the graph
            """

**7). get_output_names**

    .. code-block:: python

        def get_output_names(graph_name):
            """ Get all output tensor names of the specified graph.

            Parameters
            ----------
            graph_name : str
                Specified graph name

            Returns
            -------
            input_names : list
                All the output tensor names of the graph
            """

**8). get_input_shape**

    .. code-block:: python

        def get_input_shape(graph_name, tensor_name):
            """ Get the maximum dimension shape of an input tensor in a graph.
                There are cases that there are multiple input shapes in one input name, 
                This API only returns the maximum dimension one for the memory allocation 
                in order to get the best performance.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified input tensor name

            Returns
            -------
            tensor_shape : list
                The maxmim dimension shape of the tensor
            """

**9). get_output_shape**

    .. code-block:: python

        def get_output_shape(graph_name, tensor_name):
            """ Get the shape of an output tensor in a graph.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            tensor_shape : list
                The shape of the tensor
            """

**10). process**

    .. code-block:: python

        def process(input_tensors):
            """ Inference with provided system data of input tensors.

            Parameters
            ----------
            input_tensors : dict {str : numpy.array}
                Data of all input tensors in system memory

            Returns
            -------
            output_tensors : dict {str : numpy.array}
                Data of all output tensors in system memory
            """

sail.bm_image
______________

    .. code-block:: python

        def width():
            """ Get width of img.

            Returns
            ----------
            width : int
                width of img
            """

        def height():
            """ Get height of img.

            Returns
            ----------
            height : int
                height of img
            """

        def format():
            """ Get format of img.

            Returns
            ----------
            format : bm_image_format_ext
                format of img
            """

        def dtype():
            """ Get dtype of img.

            Returns
            ----------
            dtype : bm_image_data_format_ext
                dtype of img
            """

sail.BMImage
____________

**1). BMImage**

    .. code-block:: python

        def __init__():
            """ Constructor.
            """

        def __init__(handle, h, w, format, dtype):
            """ Constructor.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            h: int
                The height of img
            w: int
                The width of img
            format : bm_image_format_ext
                The format of img
            dtype: sail.bm_image_data_format_ext
                The data type of img
            """

**2). width**

    .. code-block:: python

        def width():
            """ Get the img width.

            Returns
            ----------
            width : int
               The width of img
            """

**3). height**

    .. code-block:: python

        def height():
            """ Get the img height.

            Returns
            ----------
            height : int
               The height of img
            """

**4). format**

    .. code-block:: python

        def format():
            """ Get the img format.

            Returns
            ----------
            format : bm_image_format_ext
               The format of img
            """

**5). dtype**

    .. code-block:: python

        def dtype():
            """ Get the img dtype.

            Returns
            ----------
            dtype: bm_image_data_format_ext
                The data type of img
            """

**6). data**

    .. code-block:: python
        
        def data(): 
            """ Get inner bm_image.  

            Returns
            ----------
            img : bm_image
                the data of img
            """

**7). get_device_id**

    .. code-block:: python

        def get_device_id():
            """ Get device id of this image.
            
            Returns
            -------
            device_id : int    
                Tensor Computing Processor ids of this image.
            """

**8). asmat**

    .. code-block:: python

        def asmat():
            """ Convert to cv Mat
            
            Returns
            -------
                image : numpy.ndarray[numpy.uint8]    
                    only support uint8
            """

**9). get_plane_num**

    .. code-block:: python

        def get_plane_num() -> int:
            """ Get plane number of this image
            """

**10). align**

    .. code-block:: python

        def align() -> int:
            """ Align the bm_image to 64 bytes
            """

**11). check_align**

    .. code-block:: python

        def check_align() -> bool:
            """ Check if the bm_image aligned 
            """

**12). unalign**

    .. code-block:: python

        def unalign() -> int:
            """ Unalign the bm_image to source bm_image
            """
**13). check_contiguous_memory**

    .. code-block:: python

        def check_contiguous_memory() -> bool:
            """ Check if the bm_image's memory contiguous
            """         
            
**14). get_handle**

    .. code-block:: python

        def get_handle():
        """ Get Handle of this BMImage.
            
            Returns
            -------
            Handle: sail.Handle    
                Handle of this BMImage.
            """

sail.BMImageArray
__________________

**1). BMImageArray**

    .. code-block:: python

        def __init__():
            """ Constructor.
            """
        
        def __init__(handle, h, w, format, dtype):
            """ Constructor.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            h : int
                Height instance
            w : int
                Width instance
            format : bm_image_format_ext
                Format instance
            dtype : bm_image_data_format_ext
                Dtype instance
            """

**2). __getitem__**

    .. code-block:: python

        def __getitem__(i):
            """ Get the bm_image from index i.

            Parameters
            ----------
            i : int
                Index of the specified location.

            Returns
            ----------
            img : sail.bm_image
                result bm_image
            """
            
**3). __setitem__**

    .. code-block:: python

        def __setitem__(i, data): 
            """ Copy the image to the specified index.

            Parameters
            ----------
            i: int
                Index of the specified location.
            data: sail.bm_image
                Input image
            """

**4). copy_from**

    .. code-block:: python

        def copy_from(i, data): 
            """ Copy the image to the specified index.
            
            Parameters
            ----------
            i: int
                Index of the specified location.
            data: sail.BMImage
                Input image
            """

**5). attach_from**

    .. code-block:: python

        def attach_from(i, data): 
            """  Attach the image to the specified index.(Because there is no memory copy, the original data needs to be cached)
       
            Parameters:
            ----------
            i: int
                Index of the specified location.
            data: BMImage
                Input image.
            """

**6). get_device_id**

    .. code-block:: python

        def get_device_id() -> int:
            """ Get device id of this BMImageArray. """
            pass

sail.Decoder
____________

**1). Decoder**

    .. code-block:: python

        def __init__(file_path, compressed=True, tpu_id=0):
            """ Constructor.

            Parameters
            ----------
            file_path : str
               Path or rtsp url to the video/image file
            compressed : bool, default: True
               Whether the format of decoded output is compressed NV12.
            tpu_id: int, default: 0
               ID of Tensor Computing Processor, there may be more than one Tensor Computing Processor for PCIE mode.
            """

**2). is_opened**

    .. code-block:: python

        def is_opened():
            """ Judge if the source is opened successfully.

            Returns
            ----------
            judge_ret : bool
                True for success and False for failure
            """

**3). read**

    .. code-block:: python

        def read(handle, image):
            """ Read an image from the Decoder.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            image : sail.BMImage
                BMImage instance

            Returns
            ----------
            judge_ret : int
                0 for success and others for failure
            """
        
        def read(handle):
            """ Read an image from the Decoder. 

            Parameters
            ----------
            handle : sail.Handle
                Handle instance

            Returns
            ----------
            image : sail.BMImage
                BMImage instance
            """

**4). read_**

    .. code-block:: python

        def read_(handle, image):
            """ Read an image from the Decoder.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            image : sail.bm_image
                bm_image instance

            Returns
            ----------
            judge_ret : int
                0 for success and others for failure
            """

**5). get_frame_shape**

    .. code-block:: python

        def get_frame_shape():
            """ Get frame shape in the Decoder.

            Returns
            ----------
            frame_shape : list
                The shape of the frame
            """

**5). release**
    
    .. code-block:: python
    
        def release():
            """ Release the Decoder.
            """

**6). reconnect**

    .. code-block:: python

        def reconnect():
            """ Reconnect the Decoder.
            """

**7). enable_dump**
    
    .. code-block:: python
    
        def enable_dump():
            """ enable input video dump without encode.
            """

**8). enable_dump**
    
    .. code-block:: python
    
        def enable_dump():
            """ enable input video dump without encode.
            """

**9). dump**
    
    .. code-block:: python
    
        def dump(dump_pre_seconds, dump_post_seconds, file_path)
            """ dump input video without encode.
        
            Parameters:
            ----------
            dump_pre_seconds : int
                dump video length(seconds) before dump moment
            dump_post_seconds : int
                dump video length(seconds) after dump moment
            file_path : str
                output path
                
            Returns
            -------
            int, 0 for success
            """
**10). get_pts_dts**
    
    .. code-block:: python
    
        def get_pts_dts()
            """ get pts and dts.
   
            Returns
            -------
            float, the value of pts and dts.
            
sail.Bmcv
_________
**1). Bmcv**

    .. code-block:: python

        def __init__(handle):
            """ Constructor.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            ""

**2). bm_image_to_tensor**

    .. code-block:: python

        def bm_image_to_tensor(image):
            """ Convert image to tensor.

            Parameters
            ----------
            image : sail.BMImage | sail.BMImageArray
                BMImage/BMImageArray instance

            Returns
            -------
            tensor : sail.Tensor
                Tensor instance
            """

        def bm_image_to_tensor(image, tensor):
            """ Convert image to tensor.

            Parameters
            ----------
            image : sail.BMImage | sail.BMImageArray
                BMImage/BMImageArray instance

            tensor : sail.Tensor
                Tensor instance
            """
            
**3). tensor_to_bm_image**

    .. code-block:: python

        def tensor_to_bm_image(tensor, bgr2rgb=False, layout='nchw'):
            """ Convert tensor to image.

            Parameters
            ----------
            tensor : sail.Tensor
                Tensor instance
            bgr2rgb : bool, default: False
                Swap color channel
            layout : str, default: 'nchw'
                Layout of the input tensor

            Returns
            -------
            image : sail.BMImage
                BMImage instance
            """

        def tensor_to_bm_image(tensor, format):
            """ Convert tensor to image.

            Parameters
            ----------
            tensor : sail.Tensor
                Tensor instance
            format : sail.Format
                Format of the BMImage

            Returns
            -------
            image : sail.BMImage
                BMImage instance
            """
        
        def tensor_to_bm_image(tensor, img, bgr2rgb=False, layout='nchw'):
            """ Convert tensor to image.

            Parameters
            ----------
            tensor : sail.Tensor
                Tensor instance
            img : sail.BMImage | sail.BMImageArray
                BMImage/BMImageArray instance
            bgr2rgb : bool, default: False
                Swap color channel
            layout : str, default: 'nchw'
                Layout of the input tensor

            Returns
            -------
            image : sail.BMImage
                BMImage instance
            """
        
        def tensor_to_bm_image(tensor, img, format):
            """ Convert tensor to image.

            Parameters
            ----------
            tensor : sail.Tensor
                Tensor instance
            img : sail.BMImage | sail.BMImageArray
                BMImage/BMImageArray instance
            format : sail.Format
                Format of the BMImage

            Returns
            -------
            image : sail.BMImage
                BMImage instance
            """

**4). crop_and_resize**

    .. code-block:: python

        def crop_and_resize(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Crop then resize an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

        def crop_and_resize(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Crop then resize an image array.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

**5). crop**

    .. code-block:: python

        def crop(input, crop_x0, crop_y0, crop_w, crop_h):
            """ Crop an image with given window.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

        def crop(input, crop_x0, crop_y0, crop_w, crop_h):
            """ Crop an image array with given window.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

**6). resize**

    .. code-block:: python

        def resize(input, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Resize an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

        def resize(input, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Resize an image array.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

**7). vpp_crop_and_resize**

    .. code-block:: python

        def vpp_crop_and_resize(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Crop then resize an image using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST
                
            Returns
            ----------
            output : sail.BMImage
                Output image
            """
        
        def vpp_crop_and_resize(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Crop then resize an image array using vpp.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

**8). vpp_crop_and_resize_padding**

    .. code-block:: python

        def vpp_crop_and_resize_padding(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, padding, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Crop then resize an image using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height
            padding : PaddingAtrr
                padding info
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

        def vpp_crop_and_resize_padding(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, padding, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Crop then resize an image array using vpp.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height
            padding : PaddingAtrr
                padding info
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

**9). vpp_crop**

    .. code-block:: python

        def vpp_crop(input, crop_x0, crop_y0, crop_w, crop_h):
            """ Crop an image with given window using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

        def vpp_crop(input, crop_x0, crop_y0, crop_w, crop_h):
            """ Crop an image array with given window using vpp.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """


**10). vpp_resize**

    .. code-block:: python

        def vpp_resize(input, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Resize an image with interpolation of INTER_NEAREST using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

        def vpp_resize(input, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Resize an image array with interpolation of INTER_NEAREST using vpp.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

         def vpp_resize(input, output, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Resize an image with interpolation of INTER_NEAREST using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            output : sail.BMImage
                Output image
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST
            """

        def vpp_resize(input, output, resize_w, resize_h, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Resize an image array with interpolation of INTER_NEAREST using vpp.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            output : sail.BMImageArray
                Output image array
            resize_w : int
                Target width
            resize_h : int
                Target height
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST
            """

**11). vpp_resize_padding**

    .. code-block:: python

        def vpp_resize_padding(input, resize_w, resize_h, padding, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Resize an image with interpolation of INTER_NEAREST using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            resize_w : int
                Target width
            resize_h : int
                Target height
            padding : PaddingAtrr
                padding info
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

        def vpp_resize_padding(input, resize_w, resize_h, padding, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Resize an image array with interpolation of INTER_NEAREST using vpp.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            resize_w : int
                Target width
            resize_h : int
                Target height
            padding : PaddingAtrr
                padding info
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

**12). warp**

    .. code-block:: python

        def warp(input, matrix, use_bilinear, similar_to_opencv):
            """ Applies an affine transformation to an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            matrix: 2d list
                2x3 transformation matrix
            use_bilinear: int
                Whether to use bilinear interpolation, default to 0 using nearest neighbor interpolation, 1 being bilinear interpolation
            similar_to_opencv: bool
                Whether to use the interface aligning the affine transformation interface of OpenCV

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

        def warp(input, matrix, use_bilinear, similar_to_opencv):
            """ Applies an affine transformation to an image array. 

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            matrix: 2d list
                2x3 transformation matrix
            use_bilinear: int
                Whether to use bilinear interpolation, default to 0 using nearest neighbor interpolation, 1 being bilinear interpolation
            similar_to_opencv: bool
                Whether to use the interface aligning the affine transformation interface of OpenCV

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

        def warp(input, output, matrix, use_bilinear, similar_to_opencv):
            """ Applies an affine transformation to an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            output : sail.BMImage
                Output image
            matrix: 2d list
                2x3 transformation matrix
            use_bilinear: int
                Whether to use bilinear interpolation, default to 0 using nearest neighbor interpolation, 1 being bilinear interpolation
            similar_to_opencv: bool
                Whether to use the interface aligning the affine transformation interface of OpenCV

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

        def warp(input, output, matrix, use_bilinear, similar_to_opencv):
            """ Applies an affine transformation to an image array. 

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            output : sail.BMImageArray
                Output image array
            matrix: 2d list
                2x3 transformation matrix
            use_bilinear: int
                Whether to use bilinear interpolation, default to 0 using nearest neighbor interpolation, 1 being bilinear interpolation
            similar_to_opencv: bool
                Whether to use the interface aligning the affine transformation interface of OpenCV

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

**13). convert_to**

    .. code-block:: python

        def convert_to(input, alpha_beta):
            """ Applies a linear transformation to an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            alpha_beta: tuple
                (a0, b0), (a1, b1), (a2, b2) factors

            Returns
            ----------
            output : sail.BMImage
                Output image
            """
    
        def convert_to(input, alpha_beta):
            """ Applies a linear transformation to an image array.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            alpha_beta: tuple
                (a0, b0), (a1, b1), (a2, b2) factors

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

        def convert_to(input, output, alpha_beta):
            """ Applies a linear transformation to an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            output : sail.BMImage
                Output image
            alpha_beta: tuple
                (a0, b0), (a1, b1), (a2, b2) factors
            """
    
        def convert_to(input, output, alpha_beta):
            """ Applies a linear transformation to an image array.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array
            alpha_beta: tuple
                (a0, b0), (a1, b1), (a2, b2) factors
            output : sail.BMImageArray
                Output image array
            """

**14). yuv2bgr**

    .. code-block:: python

        def yuv2bgr(input):
            """ Convert an image from YUV to BGR.

            Parameters
            ----------
            input : sail.BMImage
                Input image

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

        def yuv2bgr(input):
            """ Convert an image array from YUV to BGR.

            Parameters
            ----------
            input : sail.BMImageArray
                Input image array

            Returns
            ----------
            output : sail.BMImageArray
                Output image array
            """

**15). rectangle**

    .. code-block:: python

        def rectangle(image, x0, y0, w, h, color, thickness=1):
            """ Draw a rectangle on input image.

            Parameters
            ----------
            image : sail.BMImage
                Input image
            x0 : int
                Start point x of rectangle
            y0 : int
                Start point y of rectangle
            w : int
                Width of rectangle
            h : int
                Height of rectangle
            color : tuple
                Color of rectangle
            thickness : int
                Thickness of rectangle

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

**16). imwrite**

    .. code-block:: python

        def imwrite(file_name, image):
            """ Save the image to the specified file.

            Parameters
            ----------
            file_name : str
                Name of the file
            output : sail.BMImage
                Image to be saved

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

**17). get_handle**

    .. code-block:: python

        def get_handle():
            """ Get Handle instance.

            Returns
            -------
            handle: sail.Handle
               Handle instance
        """

**18). crop_and_resize_padding**

    .. code-block:: python

        def crop_and_resize_padding(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, padding, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST):
            """ Crop then resize an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height
            padding : PaddingAtrr
                padding info
            resize_alg : bmcv_resize_algorithm
                Resize algorithm, default is BMCV_INTER_NEAREST

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**19). rectangle_**

    .. code-block:: python

        def rectangle_(image, x0, y0, w, h, color, thickness=1):
            """ Draw a rectangle on input image.

            Parameters
            ----------
            image : sail.bm_image
                Input image
            x0 : int
                Start point x of rectangle
            y0 : int
                Start point y of rectangle
            w : int
                Width of rectangle
            h : int
                Height of rectangle
            color : tuple
                Color of rectangle
            thickness : int
                Thickness of rectangle

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

**20). imwrite_**

    .. code-block:: python

        def imwrite_(file_name, image):
            """ Save the image to the specified file.

            Parameters
            ----------
            file_name : str
                Name of the file
            output : sail.bm_image
                Image to be saved

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

**21). convert_format**

    .. code-block:: python

        def convert_format(input, output): 
            """Convert input to output format. 
            
            Parameters
            ----------
            input : sail.BMImage
                BMimage instance
            output : sail.BMImage
                output image
            """

        def convert_format(input, image_format): 
            """Convert input to output format. 
            
            Parameters
            ----------
            input : sail.BMImage
                BMimage instance
            image_format : sail.bm_image_format_ext
                output format

            Returns
            ----------
            output : sail.BMImage
                output image
            """

**22). vpp_convert_format**

    .. code-block:: python

        def vpp_convert_format(input, output): 
            """Convert input to output format using vpp. 
            
            Parameters
            ----------
            input : sail.BMImage
                BMimage instance
            output : sail.BMImage
                output image
            """

        def vpp_convert_format(input, image_format): 
            """Convert an image to BGR PLANAR format using vpp. 
            
            Parameters
            ----------
            input : sail.BMImage
                BMimage instance
            image_format : sail.bm_image_format_ext
                output format

            Returns
            ----------
            output : sail.BMImage
                output image
            """

**23). putText**

    .. code-block:: python
        
        def putText(input, text, x, y, color, fontScale, thickness):
            """ Draws a text on the image
            
            Parameters
            ----------
            input : sail.BMImage
                BMimage instance
            text: str
                Text to write on an image.
            x: int
                Start point x
            y: int
                Start point y
            color : tuple
                Color of text
            thickness : int
                Thickness of text

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

**24). putText_**

    .. code-block:: python

        def putText_(input, text, x, y, color, fontScale, thickness):
            """ Draws a text on the image
            
            Parameters
            ----------
            input : sail.bm_image
                bm_image instance
            text: str
                Text to write on an image.
            x: int
                Start point x
            y: int
                Start point y
            color : tuple
                Color of text
            thickness : int
                Thickness of text

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

**25). image_add_weighted**

    .. code-block:: python
        
        def image_add_weighted(input0, alpha, input1, beta, gamma, output):
            """ Calculates the weighted sum of two images

            Parameters
            ----------
            input0 : sail.BMImage
                BMimage instance.
            alpha : float
                alpha instance.
            input1 : sail.BMImage
                BMImage instance.
            beta: float
                beta instance.
            gamma: float
                gamma instance.
            output: BMImage
                result BMImage, output = input1 * alpha + input2 * beta + gamma.
            """

        def image_add_weighted(input0, alpha, input1, beta, gamma):
            """ Calculates the weighted sum of two images

            Parameters
            ----------
            input0 : sail.BMImage
                BMimage instance.
            alpha : float
                alpha instance.
            input1 : sail.BMImage
                BMImage instance.
            beta: float
                beta instance.
            gamma: float
                gamma instance.
            
            Returns
            -------
            output: BMImage
                result BMImage, output = input1 * alpha + input2 * beta + gamma.
            """

**26). image_copy_to**

    .. code-block:: python

        def image_copy_to(input, output, start_x, start_y):
            """ Copy the input to the output.

            Parameters:
            ----------
            input: BMImage|BMImageArray
                Input image or image array.
            output: BMImage|BMImageArray
                Output image or image array.
            start_x: int
                Point start x.
            start_y: int
                Point start y.
            """

**27). image_copy_to_padding**

    .. code-block:: python
    
        def image_copy_to_padding(input, output, padding_r, padding_g, padding_b, start_x, start_y):
            """ Copy the input to the output width padding.

            Parameters:
            ----------
            input: BMImage|BMImageArray
                Input image or image array.
            output: BMImage|BMImageArray
                Output image or image array.
            padding_r: int
                r value for padding.
            padding_g: int
                g value for padding.
            padding_b: int
                b value for padding.
            start_x: int
                point start x.
            start_y: int
                point start y.
            """

**28). nms**

    .. code-block:: python

        def nms(input, threshold) :
            """ Do nms use tpu.

            Parameters:
            ----------
            input: float
                input proposal array, shape must be (n,5) n<56000,
                proposal:[left,top,right,bottom,score].
            threshold: float
                nms threshold.

            Returns:
            ----------
            return nms result, numpy.ndarray[Any, numpy.dtype[numpy.float32]]
            """

**29). drawPoint**

    .. code-block:: python

        def drawPoint(image: BMImage, center: Tuple[int, int], 
            color: Tuple[int, int, int], radius: int) -> int:
            """  Draw Point on input image.
            Parameters:
            ----------
            image: BMImage, Input image
            center: Tuple[int, int], center of point, (point_x, point_y)
            color: Tuple[int, int, int], color of drawn, (b,g,r)
            radius: Radius of drawn
            """

**30). drawPoint_**

    .. code-block:: python

        def drawPoint_(image: bm_image, center: Tuple[int, int], 
            color: Tuple[int, int, int], radius: int) -> int:
            """
            Draw Point on input image.

            Parameters:
            ----------
            image: bm_image, Input image
            center: Tuple[int, int], center of point, (point_x, point_y)
            color: Tuple[int, int, int], color of drawn, (b,g,r)
            radius: Radius of drawn
            """

**31). warp_perspective**

    .. code-block:: python

        def warp_perspective(input: BMImage, coordinate, output_width: int,  output_height: int, \
            format: bm_image_format_ext = FORMAT_BGR_PLANAR,  dtype: bm_image_data_format_ext = DATA_TYPE_EXT_1N_BYTE, \
            use_bilinear: int = 0 ) -> BMImage:
            """
            Applies a perspective transformation to an image.
            
            Parameters:
            ----------
            input: BMImage
            coordinate: Original coordinate, like(left_top.x, left_top.y), (right_top.x, right_top.y), \
                (left_bottom.x, left_bottom.y), (right_bottom.x, right_bottom.y)
            output_width: Output width
            output_height: Output height
            bm_image_format_ext: Output image format, Only FORMAT_BGR_PLANAR,FORMAT_RGB_PLANAR 
            dtype: Output image dtype, Only DATA_TYPE_EXT_1N_BYTE,DATA_TYPE_EXT_4N_BYTE
            use_bilinear: Bilinear use flag.

            Returns:
            -------
            Output image
            """
      
**32). get_bm_data_type**

    .. code-block:: python

        def get_bm_data_type(format: bm_image_data_format_ext) -> bm_data_type_t:
            """ Convert bm_image_data_format_ext to bm_data_type_t
            """

**33). get_bm_data_type**

    .. code-block:: python

        def get_bm_image_data_format(dtype: bm_data_type_t) -> bm_image_data_format_ext:
            """ Convert bm_data_type_t to bm_image_data_format_ext
            """

**34). imdecode**

    .. code-block:: python

        def imdecode(data_bytes: bytes) -> BMImage:
            """ Load image from system memory

            Parameters:
            ----------
            data_bytes: image data bytes in system memory
            
            Returns:
            ----------
            return decoded image
            """

**35). fft**

    .. code-block:: python

        fft(self, forward: bool, input_real: Tensor) -> list[Tensor]:
            """
            1d or 2d fft (only real part)
            
            Parameters:
            ----------
            forward: bool, positive transfer
            input_real: Tensor, input tensor
            
            
            Returns:
            ----------
            return list[Tensor], The real and imaginary part of output
            """
    
        fft(self, forward: bool, input_real: Tensor, input_imag: Tensor) -> list[Tensor]:
            """
            1d or 2d fft (real part and imaginary part)
            
            Parameters:
            ----------
            forward: bool, positive transfer
            input_real: Tensor, input tensor real part
            input_imag: Tensor, input tensor imaginary part

            
            Returns:
            ----------
            return list[Tensor], The real and imaginary part of output
            """

**36). convert_yuv420p_to_gray**

    .. code-block:: python

        def convert_yuv420p_to_gray(input, output): 
            """Convert yuv420p to gray. 
            
            Parameters
            ----------
            input : sail.BMImage
                BMimage instance
            output : sail.BMImage
                output image
            """

**36). convert_yuv420p_to_gray_**

    .. code-block:: python

        def convert_yuv420p_to_gray_(input, output): 
            """Convert yuv420p to gray. 
            
            Parameters
            ----------
            input : sail.bm_image
                bm_image instance
            output : sail.bm_image
                output image
            """

**37). mat_to_bm_image**

    .. code-block:: python
        
        def mat_to_bm_image(self, mat: numpy.ndarray[numpy.uint8]) -> BMImage:
            """
            Convert cv mat to BMImage
            
            Parameters:
            ----------
            mat: input cv mat, rgb_packed.
            
            Returns
            -------
            img: BMImage
                Output image
            """

        def mat_to_bm_image(self, mat: numpy.ndarray[numpy.uint8], img: BMImage) -> int:
            """
            Convert cv mat to BMImage
            
            Parameters:
            ----------
            mat: input cv mat, rgb_packed.
            img: output BMImage
            """
            
**38). imencode**

    .. code-block:: python

        def imencode(self, ext: str, img: BMImage) -> numpy.ndarray:
        """
        Compresses the BMImage and stores it in the memory

        Parameters:
        ----------
        ext: File extension that defines the output format.
        img: BMImage to be written
        buf: Output buffer resized to fit the compressed BMImage

        Returns:
        ----------
        return encoded array
        """

**39). stft**

    .. code-block:: python

        def stft(self, input_real: numpy.ndarray, input_imag: numpy.ndarray, real_input: bool, normalize: bool, n_fft: int, hop_len: int,
                pad_mode: int, win_mode: int) -> tuple[numpy.ndarray, numpy.ndarray]:
            """
            Short-Time Fourier Transform (STFT) for NumPy arrays.

            Parameters:
            ----------
            input_real: numpy.ndarray
                The real part of the input signal as a 1D array.
            input_imag: numpy.ndarray
                The imaginary part of the input signal as a 1D array.
            real_input: bool
                Indicates whether the input is purely real. If true, the imaginary part is ignored.
            normalize: bool
                Indicates whether to normalize the output.
            n_fft: int
                The number of points in the FFT. This defines the resolution of the frequency bins.
            hop_len: int
                The number of samples to hop between successive frames. This controls the overlap.
            pad_mode: int
                An integer indicating the padding mode to use when the input signal is shorter than the expected length:
                - 0: Constant padding (pads with zeros).
                - 1: Reflective padding (pads by reflecting the signal).
            win_mode: int
                An integer specifying the window function to apply to each segment:
                - 0: Hann window.
                - 1: Hamming window.

            Returns:
            ----------
            tuple[numpy.ndarray, numpy.ndarray]
                A tuple containing two NumPy arrays representing the STFT output:
                - The first array is the real part of the STFT.
                - The second array is the imaginary part of the STFT.
            """
            pass

        def stft(self, input_real: Tensor, input_imag: Tensor, real_input: bool, normalize: bool, n_fft: int, hop_len: int,
                pad_mode: int, win_mode: int) -> tuple[Tensor, Tensor]:
            pass 

**40). istft**

    .. code-block:: python

        def istft(self, input_real: numpy.ndarray, input_imag: numpy.ndarray, real_input: bool, normalize: bool, L: int, hop_len: int,
                pad_mode: int, win_mode: int) -> tuple[numpy.ndarray, numpy.ndarray]:
            """
            Inverse Short-Time Fourier Transform (ISTFT) for NumPy arrays.

            Parameters:
            ----------
            input_real: numpy.ndarray
                The real part of the STFT output as a 1D array.
            input_imag: numpy.ndarray
                The imaginary part of the STFT output as a 1D array.
            real_input: bool
                Indicates whether the input STFT is purely real. If true, the imaginary part is ignored.
            normalize: bool
                Indicates whether to normalize the output.
            L: int
                The length of the original time-domain signal to reconstruct.
            hop_len: int
                The number of samples to hop between successive frames. This controls the overlap.
            pad_mode: int
                An integer indicating the padding mode to use when the input signal is shorter than the expected length:
                - 0: Constant padding (pads with zeros).
                - 1: Reflective padding (pads by reflecting the signal).
            win_mode: int
                An integer specifying the window function to apply to each segment when reconstructing the signal:
                - 0: Hann window.
                - 1: Hamming window.

            Returns:
            ----------
            tuple[numpy.ndarray, numpy.ndarray]
                A tuple containing two NumPy arrays representing the reconstructed time-domain signal:
                - The first array is the reconstructed signal.
                - The second array is the corresponding phase information (if applicable).
            """
            pass

        def istft(self, input_real: Tensor, input_imag: Tensor, real_input: bool, normalize: bool, L: int, hop_len: int,
                pad_mode: int, win_mode: int) -> tuple[Tensor, Tensor]:
            pass 

            
sail.MultiDecoder
__________________

**1). MultiDecoder**

    .. code-block:: python

        def __init__(queue_size: int = 10, tpu_id: int = 0, discard_mode: int = 0):
            """ Constructor.

            Parameters
            ----------
            queue_size: int
                Max queue size,default is 10.
            tpu_id: int
                ID of Tensor Computing Processor, default is 0.
            discard_mode: int
                Data discard policy when the queue is full. Default is 0.
                If 0, do not push the data to queue, else pop the data from queue and push new data to queue.
            """

**2). set_read_timeout**

    .. code-block:: python

        def set_read_timeout(timeout: int):
            """ Set read frame timeout waiting time 
            
            Parameters:
            ----------
            timeout: int
                Set read frame timeout waiting time in seconds.
            """

**3). add_channel**

    .. code-block:: python

        def add_channel(file_path: str, frame_skip_num: int = 0) -> int:
            
            """ Add a channel to decode

            Parameters:
            ----------
            file_path : str
                file path
            frame_skip_num : int
                frame skip number, default is 0

            Returns:
            ----------
            return channel index number, int
            """

**4). del_channel**

    .. code-block:: python

        def del_channel(channel_idx: int) -> int : 
            """ Delete channel
            
            Parameters:
            ----------
            channel_idx : int
                channel index number

            Returns:
            -------
            0 for success and other for failure.
            """

**5). clear_queue**

    .. code-block:: python

        def clear_queue(channel_idx: int) -> int : 
            """ Clear data cache queue 
            
            Parameters:
            ----------
            channel_idx : int
                channel index number

            Returns:
            -------
            0 for success and other for failure.
            """

**6). read**

    .. code-block:: python

        def read(channel_idx: int, image: BMImage, read_mode: int = 0) -> int : 
            """ Read a BMImage from the MultiDecoder with a given channel.

            Parameters:
            ----------
            channel_idx : int
                channel index number
            image: BMImage
                BMImage instance to be read to
            read_mode: int
                Read data mode, 0 for not waiting data and other waiting data.

            Returns:
            -------
            0 for successed get data.
            """

        def read(channel_idx: int) -> int :
            """ Read a BMImage from the MultiDecoder with a given channel.

            Parameters:
            ----------
            channel_idx : int
                channel index number

            Returns:
            -------
                BMImage instance to be read to
            """


**7). read_**

    .. code-block:: python

        def read_(channel_idx: int, image: bm_image, read_mode: int=0) -> int :
            """ Read a bm_image from the MultiDecoder with a given channel.

            Parameters:
            ----------
            channel_idx : int
                channel index number
            image: bm_image
                bm_image instance to be read to
            read_mode: int
                read data mode, default 0, if 0, not waiting data, else waiting data.

            Returns:
            -------
            return 0 if get data.
            """

        def read_(channel_idx: int) -> int :
            """ Read a bm_image from the MultiDecoder with a given channel.

            Parameters:
            ----------
            channel_idx : int
                channel index number

            Returns:
            -------
                bm_image instance to be read to
            """

**8). reconnect**

    .. code-block:: python
        
        def reconnect(channel_idx: int) -> int :
            """ Reconnect Decoder for instance channel
                
            Parameters:
            ----------
            channel_idx : int
                channel index number

            Returns:
            -------
            0 for success and other for failure.
                
            """


**9). get_frame_shape**

    .. code-block:: python

        def get_frame_shape() -> list[int]:
            """ Get frame for instance channel
        
            Returns
            -------
            list[int], [1, C, H, W]
            """

**10). set_local_flag**

    .. code-block:: python

        def set_local_flag(flag: bool) -> None: 
            """ Set local video flag

            Parameters:
            ----------
            flag : bool
                If flag is True, Decode up to 25 frames per second

            """

**11). get_channel_fps**

    .. code-block:: python

        def get_channel_fps(self, channel_idx: int) -> float:
            """ Get the fps of the video stream in a specified channel
            
            Parameters:
            ----------
            channel_idx : int
                channel index number

            Returns:
            -------
            Returns the fps of the video stream in the specified channel

**12). get_drop_num**

    .. code-block:: python

        def get_drop_num(channel_idx: int) -> int: 
            """ Get drop num for instance channel

            Parameters:
            ----------
            channel_idx : int
                channel index number

            Returns:
            -------
            drop num for instance channel

            """

**13). reset_drop_num**

    .. code-block:: python

        def reset_drop_num(channel_idx: int) -> int: 
            """ Set drop num  for instance channel

            Parameters:
            ----------
            channel_idx : int
                channel index number

            """

sail.sail_resize_type
______________________

.. code-block:: python

        sail.sail_resize_type.BM_RESIZE_VPP_NEAREST
        sail.sail_resize_type.BM_RESIZE_TPU_NEAREST
        sail.sail_resize_type.BM_RESIZE_TPU_LINEAR
        sail.sail_resize_type.BM_RESIZE_TPU_BICUBIC
        sail.sail_resize_type.BM_PADDING_VPP_NEAREST
        sail.sail_resize_type.BM_PADDING_TPU_NEAREST
        sail.sail_resize_type.BM_PADDING_TPU_LINEAR
        sail.sail_resize_type.BM_PADDING_TPU_BICUBIC


sail.ImagePreProcess
______________________

**1). ImagePreProcess**

    .. code-block:: python

        def __init__(batch_size: int, input_width: int, input_height: int, resize_mode:sail_resize_type,
            tpu_id: int=0, queue_in_size: int=20, queue_out_size: int=20, use_mat_flag: bool=False):
            """ ImagePreProcess Constructor.

            Parameters
            ----------
            batch_size: int
                Output batch size.
            input_width: int
                Input image width.
            input_height: int
                Input image height
            resize_mode: sail_resize_type
                Resize Methods
            tpu_id: int
                ID of Tensor Computing Processor, there may be more than one Tensor Computing Processor for PCIE mode,default is 0.
            queue_in_size: int
                Max input image data queue size, default is 20.
            queue_out_size: int
                Max output tensor data queue size, default is 20.
            use_mat_flag:bool
                Use cv Mat for output, default is false.
            """


**2). SetResizeImageAtrr**

    .. code-block:: python

        def SetResizeImageAtrr(output_width: int, output_height: int ,bgr2rgb : bool, dtype: ImgDtype): 

            """ Set the Resize Image attribute 
            
            Parameters:
            ----------
            output_width: int
                The width of resized image.
            output_height: int
                The height of resized image.
            bgr2rgb: bool
                The flag of convert BGR image to RGB.
            dtype: ImgDtype  
                The data type of resized image,Only supported BM_FLOAT32,BM_INT8,BM_UINT8   
            """


**3). SetPaddingAtrr**

    .. code-block:: python

        def SetPaddingAtrr(padding_b:int=114,padding_g:int=114,padding_r:int=114,align:int=0) -> list[int] :
            """ Set the padding attribute object.

            Parameters:
            ----------
            padding_b: int
                padding value of b channel, dafault 114
            padding_g: int
                padding value of g channel, dafault 114
            padding_r: int
                padding value of r channel, dafault 114
            align: int
                padding position, default 0: start left top, 1 for center

            Returns
            -------
            list[int], [start_x, start_y, resize_w, resize_h]
            """


**4). SetConvertAtrr**

    .. code-block:: python

        def SetConvertAtrr(alpha_beta) -> int :
            """ Set the linear transformation attribute 

            Parameters:
            ----------
            alpha_beta:like (a0, b0), (a1, b1), (a2, b2) factors

            Returns
            -------
            0 for success and other for failure 
            """


**5). PushImage**

    .. code-block:: python

        def PushImage(channel_idx : int, image_idx : int, image: BMImage) -> int: 
            """ Push Image
            Parameters:
            ----------
            channel_idx : int
                Channel index number of the image
            image_idx: int
                Image index number of the image
            image: BMImage
                Input image
            
            Returns
            -------
            0 for success and other for failure 
            """


**6). GetBatchData**

    .. code-block:: python

        def GetBatchData() -> tuple:
            """ Get the Batch Data object
            
            Returns
            -------
            [Tensor, list[BMImage],list[int],list[int],list[list[int]]]
                Output Tensor, Original Images, Original Channel index, Original Index, Padding Atrr(start_x, start_y, width, height)
            """


**7). set_print_flag**

    .. code-block:: python

        def set_print_flag(flag : bool) -> None:
            """ Set the print flag
            
            Parameters:
            ----------
            flag : int
            """

sail.EngineImagePreProcess
___________________________

**1). EngineImagePreProcess**

    .. code-block:: python

        def __init__(bmodel_path: str, tpu_id: int): 
            """ EngineImagePreProcess Constructor.

            Parameters:
            ----------
            bmodel_path: str
            tpu_id: int
                ID of Tensor Computing Processor, there may be more than one Tensor Computing Processor for PCIE mode,default is 0.
            """

**2). InitImagePreProcess**

    .. code-block:: python

        def InitImagePreProcess(resize_mode: sail_resize_type, bgr2rgb: bool = False,
            queue_in_size: int = 20, queue_out_size: int = 20) -> int:
            """ initialize ImagePreProcess.

            Parameters:
            ----------
            resize_mode: sail_resize_type
                Resize Methods
            bgr2rgb: bool
                The flag of convert BGR image to RGB, default is False.
            queue_in_size: int
                Max input image data queue size, default is 20.
            queue_out_size: int
                Max output tensor data queue size, default is 20.
            
            Returns
            -------
            0 for success and other for failure 
            """
            
**3). SetPaddingAtrr**

    .. code-block:: python

        def SetPaddingAtrr(padding_b:int=114,padding_g:int=114,padding_r:int=114,align:int=0) -> int :
            """ Set the padding attribute object.

            Parameters:
            ----------
            padding_b: int
                padding value of b channel, dafault 114
            padding_g: int
                padding value of g channel, dafault 114
            padding_r: int
                padding value of r channel, dafault 114
            align: int
                padding position, default 0: start left top, 1 for center
            
            Returns
            -------
            0 for success and other for failure 
            """


**4). SetConvertAtrr**

    .. code-block:: python

        def SetConvertAtrr(alpha_beta) -> int :
            """ Set the linear transformation attribute 

            Parameters:
            ----------
            alpha_beta:like (a0, b0), (a1, b1), (a2, b2) factors

            Returns
            -------
            0 for success and other for failure 
            """
            
**5). PushImage**

    .. code-block:: python

        def PushImage(channel_idx : int, image_idx : int, image: BMImage) -> int: 
            """ Push Image
            Parameters:
            ----------
            channel_idx : int
                Channel index number of the image
            image_idx: int
                Image index number of the image
            image: BMImage
                Input image
            
            Returns
            -------
            0 for success and other for failure 
            """
            

**6). GetBatchData_Npy**

    .. code-block:: python

        def GetBatchData_Npy() -> tuple:
            """ Get the Batch Data object
            
            Returns
            -------
            [dict[str, ndarray], list[BMImage],list[int],list[int],list[list[int]]]
                Output ndarray map, Original Images, Original Channel index, Original Indexm, Padding Atrr(start_x, start_y, width, height)
            """


**7). GetBatchData_Npy2**
            
    .. code-block:: python

        def GetBatchData_Npy2() -> tuple:
            """ Get the Batch Data object
            
            Returns
            -------
            [dict[str, ndarray], list[numpy.ndarray[numpy.uint8]],list[int],list[int],list[list[int]]]
                Output ndarray map, Original Images, Original Channel index, Original Indexm, Padding Atrr(start_x, start_y, width, height)
            """
            
**8). get_graph_name**

    .. code-block:: python

        def get_graph_name() -> str:
            """ Get first graph name in the loaded bomodel
            
            Returns
            -------
            First graph name
            """
            
**9). get_input_width**

    .. code-block:: python

        def get_input_width() -> int:
            """ Get model input width

            Returns
            -------
            Model input width
            """
            
**10). get_input_height**

    .. code-block:: python

        def get_input_height() -> int:
            """ Get model input height

            Returns
            -------
            Model input height
            """
            
**11). get_output_names**

    .. code-block:: python

        def get_output_names() -> list[str]:
            """ Get all output tensor names of the first graph

            Returns
            -------
            All the output tensor names of the graph
            """
            
**12). get_output_shape**

    .. code-block:: python
        
        def get_output_shape(tensor_name: str) -> list[int]:
            """ Get the shape of an output tensor in frist graph
            
            Parameters:
            ----------
            tensor_name : str
                The specified tensor name

            Returns
            -------
            The shape of the tensor
            """

sail.algo_yolov5_post_1output
_________________________________

**1). algo_yolov5_post_1output**

    .. code-block:: python
          
        def __init__(shape: list[int], network_w:int = 640, network_h:int = 640, max_queue_size: int=20, input_use_multiclass_nms: bool=True, agnostic: bool=False): 
            """ algo_yolov5_post_1output Constructor.

            Parameters:
            ----------
            shape: list[int]
            network_w: int, network input width 
            network_h: int, network input height 
            max_queue_size: int, default 20
            input_use_multiclass_nms: bool,default True
            agnostic: bool,default False
            """

**2). push_npy**

    .. code-block:: python

        def push_npy(self, 
                channel_idx : int, 
                image_idx : int, 
                data : numpy.ndarray[Any, numpy.dtype[numpy.float_]], 
                dete_threshold : float, 
                nms_threshold : float,
                ost_w : int, 
                ost_h : int,
                padding_left : int,
                padding_top : int,
                padding_width : int,
                padding_height : int) -> int:
                """ algo_yolov5_post_1output Constructor.

                Parameters:
                ----------
                
                channel_idx : Channel index number of the image.
                image_idx : Image index number of the image.
                data : Input Data ptr
                dete_threshold : Detection threshold
                nms_threshold : NMS threshold
                ost_w : Original image width
                ost_h : Original image height
                padding_left : Padding left
                padding_top : Padding top
                padding_width : Padding width
                padding_height : Padding height

                Returns
                -------
                return 0 for success and other for failure
                """

**3). push_data**

    .. code-block:: python

        def push_data(self, 
                channel_idx : list[int], 
                image_idx : list[int], 
                input_data : TensorPTRWithName, 
                dete_threshold : list[float],
                nms_threshold : list[float],
                ost_w : list[int],
                ost_h : list[int],
                padding_attr : list[list[int]]) -> int:
                """ algo_yolov5_post_1output Constructor.

                Parameters:
                ----------
                
                channel_idx : Channel index number of the image.
                image_idx : Image index number of the image.
                data : Input Data
                dete_threshold : Detection threshold
                nms_threshold : NMS threshold
                ost_w : Original image width
                ost_h : Original image height
                padding_attr : Padding Attribute(start_x, start_y, width, height)

                Returns
                -------
                return 0 for success and other for failure
                """

**4). get_result_npy**

    .. code-block:: python

        def get_result_npy() -> tuple[tuple[int, int, int, int, int, float],int, int]:
            """ Get the PostProcess result
            
            Returns
            -------
            tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
                
            """

sail.algo_yolov5_post_3output
_________________________________

**1). algo_yolov5_post_3output**

    .. code-block:: python
          
        def __init__(shape: list[list[int]], network_w:int = 640, network_h:int = 640, max_queue_size: int=20, input_use_multiclass_nms: bool=True, agnostic: bool=False): 
            """ algo_yolov5_post_3output Constructor.

            Parameters:
            ----------
            shape: list[list[int]]
            network_w: int, network input width 
            network_h: int, network input height 
            max_queue_size: int, default 20
            input_use_multiclass_nms: bool, default True
            agnostic: bool, default False
            """
            
**2). reset_anchors**

    .. code-block:: python

        def reset_anchors(self, anchors_new: list[list[list[int]]]) -> int:
            """ Reset Anchors
            
            Parameters:
            ----------
            anchors_new: list[list[list[int]]]
            
            Returns
            -------
            return 0 for success and other for failure
            """
            pass
            
**3). push_data**

    .. code-block:: python

        def push_data(self, 
                channel_idx : list[int], 
                image_idx : list[int], 
                input_data : list[TensorPTRWithName], 
                dete_threshold : list[float],
                nms_threshold : list[float],
                ost_w : list[int],
                ost_h : list[int],
                padding_attr : list[list[int]]) -> int:
                """ algo_yolov5_post_3output Constructor.

                Parameters:
                ----------
                
                channel_idx : Channel index number of the image.
                image_idx : Image index number of the image.
                data : Input Data
                dete_threshold : Detection threshold
                nms_threshold : NMS threshold
                ost_w : Original image width
                ost_h : Original image height
                padding_attr : Padding Attribute(start_x, start_y, width, height)

                Returns
                -------
                return 0 for success and other for failure
                """

**4). get_result_npy**

    .. code-block:: python

        def get_result_npy() -> tuple[tuple[int, int, int, int, int, float],int, int]:
            """ Get the PostProcess result
            
            Returns
            -------
            tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
                
            """

sail.algo_yolov5_post_cpu_opt_async
_________________________________

**1). algo_yolov5_post_cpu_opt_async**

    .. code-block:: python
          
        def __init__(shape: list[list[int]], network_w:int = 640, network_h:int = 640, max_queue_size: int=20, use_multiclass_nms: bool=True): 
            """ algo_yolov5_post_cpu_opt_async Constructor.

            Parameters:
            ----------
            shape: list[list[int]]
            network_w: int, network input width 
            network_h: int, network input height 
            max_queue_size: int, default 20
            use_multiclass_nms: bool, whether to use multi-class NMS, default True
            """
            
**2). reset_anchors**

    .. code-block:: python

        def reset_anchors(self, anchors_new: list[list[list[int]]]) -> int:
            """ Reset Anchors
            
            Parameters:
            ----------
            anchors_new: list[list[list[int]]]
            
            Returns
            -------
            return 0 for success and other for failure
            """
            pass
            
**3). push_data**

    .. code-block:: python

        def push_data(self, 
                channel_idx : list[int], 
                image_idx : list[int], 
                input_data : list[TensorPTRWithName], 
                dete_threshold : list[float],
                nms_threshold : list[float],
                ost_w : list[int],
                ost_h : list[int],
                padding_attr : list[list[int]]) -> int:
                """ algo_yolov5_post_cpu_opt_async Constructor.

                Parameters:
                ----------
                
                channel_idx : Channel index number of the image.
                image_idx : Image index number of the image.
                input_data : Input Data
                dete_threshold : Detection threshold
                nms_threshold : NMS threshold
                ost_w : Original image width
                ost_h : Original image height
                padding_attr : Padding Attribute(start_x, start_y, width, height)

                Returns
                -------
                return 0 for success and other for failure
                """

**4). get_result_npy**

    .. code-block:: python

        def get_result_npy() -> tuple[tuple[int, int, int, int, int, float],int, int]:
            """ Get the PostProcess result
            
            Returns
            -------
            tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
                
            """

sail.algo_yolox_post
_________________________________

**1). algo_yolox_post**

    .. code-block:: python
          
        def __init__(shape: list[int], network_w:int = 640, network_h:int = 640, max_queue_size: int=20): 
            """ algo_yolov5_post_1output Constructor.

            Parameters:
            ----------
            shape: list[int]
            network_w: int, network input width 
            network_h: int, network input height 
            max_queue_size: int, default 20
            """

**2). push_npy**

    .. code-block:: python

        def push_npy(self, 
                channel_idx : int, 
                image_idx : int, 
                data : numpy.ndarray[Any, numpy.dtype[numpy.float_]], 
                dete_threshold : float, 
                nms_threshold : float,
                ost_w : int, 
                ost_h : int,
                padding_left : int,
                padding_top : int,
                padding_width : int,
                padding_height : int) -> int:
                """ algo_yolox_post Constructor.

                Parameters:
                ----------
                
                channel_idx : Channel index number of the image.
                image_idx : Image index number of the image.
                data : Input Data ptr
                dete_threshold : Detection threshold
                nms_threshold : NMS threshold
                ost_w : Original image width
                ost_h : Original image height
                padding_left : Padding left
                padding_top : Padding top
                padding_width : Padding width
                padding_height : Padding height

                Returns
                -------
                return 0 for success and other for failure
                """

**3). push_data**

    .. code-block:: python

        def push_data(self, 
                channel_idx : list[int], 
                image_idx : list[int], 
                input_data : TensorPTRWithName, 
                dete_threshold : list[float],
                nms_threshold : list[float],
                ost_w : list[int],
                ost_h : list[int],
                padding_attr : list[list[int]]) -> int:
                """ algo_yolox_post Constructor.

                Parameters:
                ----------
                
                channel_idx : Channel index number of the image.
                image_idx : Image index number of the image.
                data : Input Data
                dete_threshold : Detection threshold
                nms_threshold : NMS threshold
                ost_w : Original image width
                ost_h : Original image height
                padding_attr : Padding Attribute(start_x, start_y, width, height)

                Returns
                -------
                return 0 for success and other for failure
                """

**4). get_result_npy**

    .. code-block:: python

        def get_result_npy() -> tuple[tuple[int, int, int, int, int, float],int, int]:
            """ Get the PostProcess result
            
            Returns
            -------
            tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
                
            """

sail.algo_yolov5_post_cpu_opt
_________________________________

**1). algo_yolov5_post_cpu_opt**

    .. code-block:: python
          
        def __init__(self, shapes: list[list[int]], network_w:int = 640, network_h:int = 640):
            """ algo_yolov5_post_cpu_opt Constructor.

            Parameters:
            ----------
            shapes: list[list[int]],Input Data shape
            network_w: int, Network input width 
            network_h: int, Network input height 
            """

**2). process**

    .. code-block:: python

        def process(self,input_data:list[TensorPTRWithName], ost_w: list[int], ost_h: list[int], dete_threshold: list[float], nms_threshold: list[float], input_keep_aspect_ratio: bool, input_use_multiclass_nms: bool) -> list[list[tuple[int, int, int, int, int, float]]]:
            """ Process

            Parameters:
            ----------
            input_data: list[TensorPTRWithName], Input Data 
            ost_w: list[int], Original image width
            ost_h: list[int], Original image height
            dete_threshold: list[float], Detection threshold
            nms_threshold: list[float], NMS threshold
            input_keep_aspect_ratio: bool, Input keeping aspect ratio  
            input_use_multiclass_nms: bool, Input with multiclass

            Returns
            -------
            list[list[tuple[left, top, right, bottom, class_id, score]]]
                
            """
        
        def process(self,input:dict[str, Tensor], ost_w: list[int], ost_h: list[int], dete_threshold: list[float], nms_threshold: list[float], input_keep_aspect_ratio: bool, input_use_multiclass_nms: bool) -> list[list[tuple[int, int, int, int, int, float]]]:
            """ Process

            Parameters:
            ----------
            input: dict[str, Tensor], Input Data 
            ost_w: list[int], Original image width
            ost_h: list[int], Original image height
            dete_threshold: list[float], Detection threshold
            nms_threshold: list[float], NMS threshold
            input_keep_aspect_ratio: bool, Input keeping aspect ratio  
            input_use_multiclass_nms: bool, Input with multiclass

            Returns
            -------
            list[list[tuple[left, top, right, bottom, class_id, score]]]
                
            """

sail.sort_tracker_controller
_________________________________

**1). sort_tracker_controller**

    .. code-block:: python
          
        def __init__(self,max_iou_distance:float=0.7, max_age:int=30, n_init:int=3):
            """ sort_tracker_controller Constructor.

            Parameters:
            ----------
            max_iou_distance:float, maximum Intersection over Union (IoU) distance threshold used in the tracker, default 0.7
            max_age:int, the maximum number of frames that a tracked object can exist in the tracke, default 30
            n_init:int, the threshold for the number of initialization frames in the tracker, default 3
            """

**2). process**

    .. code-block:: python
        
        def process(detected_objects:list[tuple[int, int, int, int, int, float]])-> list[tuple[int, int, int, int, int, float, int]]:
            """ Track the objects based on the detected objects and their features
            
            Parameters:
            ----------
            detected_objects: list(tuple(left, top, right, bottom, class_id, score))
            
            Returns
            -------
            tracked_objects: tracked_objects:list(tuple(left, top, right, bottom, class_id, score, track_id))
            """

sail.sort_tracker_controller_async
_________________________________

**1). sort_tracker_controller_async**

    .. code-block:: python
          
        def __init__(self,  max_iou_distance:float=0.7, max_age:int=30, n_init:int=3, input_queue_size:int=10, result_queue_size:int=10):
            """ sort_tracker_controller_async Constructor.

            Parameters:
            ----------
            max_iou_distance:float, maximum Intersection over Union (IoU) distance threshold used in the tracker, default 0.7
            max_age:int, the maximum number of frames that a tracked object can exist in the tracke, default 30
            n_init:int, the threshold for the number of initialization frames in the tracker, default 3
            input_queue_size: buffer size of the input data queue, default 10
            result_queue_size: buffer size of the result queue, default 10
            """

**2). push_data**

    .. code-block:: python

        def push_data(detected_objects:list[tuple[int, int, int, int, int, float]]) -> int:
            """ Asynchronous processing interface. Use with get_result_npy()!
                Track the objects based on the detected objects and their features
            
            Parameters:
            ----------
            detected_objects: list(tuple(left, top, right, bottom, class_id, score))
            
            Returns
            -------
            int:Returns 0 on success and others on failure.
            """


**3). get_result_npy**

    .. code-block:: python

        def get_result_npy() -> list[tuple[int, int, int, int, int, float, int]]:
            """ Asynchronous processing interface. Use with push_data()!
                Track the objects based on the detected objects and their features
            
            Returns
            -------
            tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id))
            """

sail.deepsort_tracker_controller
_________________________________

**1). deepsort_tracker_controller**

    .. code-block:: python
          
        def __init__(max_cosine_distance:float, nn_budget:int, k_feature_dim:int, max_iou_distance:float = 0.7, max_age:int = 30, n_init:int = 3): 
            """ deepsort_tracker_controller Constructor.

            Parameters:
            ----------
            max_cosine_distance:float, maximum threshold for cosine distance used in similarity calculation
            nn_budget:int, maximum number for nearest neighbor search
            k_feature_dim:int, the feature dimension of the detected objects
            max_iou_distance:float, maximum Intersection over Union (IoU) distance threshold used in the tracker, default 0.7
            max_age:int, the maximum number of frames that a tracked object can exist in the tracke, default 30
            n_init:int, the threshold for the number of initialization frames in the tracker, default 3
            """

**2). process**

    .. code-block:: python
        
        def process(detected_objects:list[tuple[int, int, int, int, int, float]], feature:sail.Tensor) -> list[tuple[int, int, int, int, int, float, int]]:
            """ Track the objects based on the detected objects and their features
            
            Parameters:
            ----------
            detected_objects: list(tuple(left, top, right, bottom, class_id, score))
            feature: sail.Tensor, the features of the detected objects
            
            Returns
            -------
            tracked_objects: tracked_objects:list(tuple(left, top, right, bottom, class_id, score, track_id))
            """

        def process(detected_objects:list[tuple[int, int, int, int, int, float]], feature:list[numpy.array]) -> list[tuple[int, int, int, int, int, float, int]]:
            """ Track the objects based on the detected objects and their features
            
            Parameters:
            ----------
            detected_objects: list(tuple(left, top, right, bottom, class_id, score))
            feature: list[float], the features of the detected objects
            
            Returns
            -------
            tracked_objects: tracked_objects:list(tuple(left, top, right, bottom, class_id, score, track_id))
            """



sail.deepsort_tracker_controller_async
_________________________________

**1). deepsort_tracker_controller_async**

    .. code-block:: python
          
        def __init__(max_cosine_distance:float, nn_budget:int, k_feature_dim:int, max_iou_distance:float = 0.7, max_age:int = 30, n_init:int = 3, queue_size:int = 10): 
            """ deepsort_tracker_controller_async Constructor.

            Parameters:
            ----------
            max_cosine_distance:float, maximum threshold for cosine distance used in similarity calculation
            nn_budget:int, maximum number for nearest neighbor search
            k_feature_dim:int, the feature dimension of the detected objects
            max_iou_distance:float, maximum Intersection over Union (IoU) distance threshold used in the tracker, default 0.7
            max_age:int, the maximum number of frames that a tracked object can exist in the tracke, default 30
            n_init:int, the threshold for the number of initialization frames in the tracker, default 3
            queue_size:buffer size of the result queue, default 10
            """

**2). push_data**

    .. code-block:: python

        def push_data(detected_objects:list[tuple[int, int, int, int, int, float]], feature:sail.Tensor) -> int:
            """ Asynchronous processing interface. Use with get_result_npy()!
                Track the objects based on the detected objects and their features
            
            Parameters:
            ----------
            detected_objects: list(tuple(left, top, right, bottom, class_id, score))
            feature: sail.Tensor, the features of the detected objects
            
            Returns
            -------
            int:Returns 0 on success and others on failure.
            """
        
        def push_data(detected_objects:list[tuple[int, int, int, int, int, float]], feature:list[float]) -> int:
            """ Asynchronous processing interface. Use with get_result_npy()!
                Track the objects based on the detected objects and their features
            
            Parameters:
            ----------
            detected_objects: list(tuple(left, top, right, bottom, class_id, score))
            feature: list[float], the features of the detected objects
            
            Returns
            -------
            int:Returns 0 on success and others on failure.
            """

**3). get_result_npy**

    .. code-block:: python

        def get_result_npy() -> list[tuple[int, int, int, int, int, float, int]]:
            """ Asynchronous processing interface. Use with push_data()!
                Track the objects based on the detected objects and their features
            
            Returns
            -------
            tracked_objects: list(tuple(left, top, right, bottom, class_id, score, track_id))
            """

sail.bytetrack_tracker_controller
_________________________________

**1). bytetrack_tracker_controller**

    .. code-block:: python
          
        def __init__(frame_rate:int = 30, track_buffer:int = 30): 
            """ bytetrack_tracker_controller Constructor.

            Parameters:
            ----------
            frame_rate: int, used to control the maximum number of frames allowed to disappear for tracked objects 
            track_buffer: int, used to control the maximum number of frames allowed to disappear for tracked objects
            """

**2). process**

    .. code-block:: python

        def process(detected_objects:list[list[int, float, float, float, float, float, float, float]], tracked_objects:list[list[int, float, float, float, float, float, float, float, int]]) -> int:
            """ Track the objects based on the detected objects and their features
            
            Parameters:
            ----------
            detected_objects: list[list[int, float, float, float, float, float, float, float]]
            tracked_objects: tracked_objects:list[list[int, float, float, float, float, float, float, float, int]]
            
            Returns
            -------
            return 0 for success and other for failure
            """

sail.Encoder
_________________________________

**1). Encoder**

    .. code-block:: python

        def __init__():
            """ Encoder Constructor for picture.
            """

        def __init__(output_path: str, handle: sail.Handle, enc_fmt: str, pix_fmt: str, enc_params: str, cache_buffer_length: int=5, abort_policy: int=0):
            """ Encoder Constructor for video or rtsp/rtmp stream.

                Parameters:
                ----------
                output_path: local video file path or rtsp/rtmp address
                handle: sail.Handle, encoder handle instance
                enc_fmt: encoder format, support h264_bm and h265_bm/hevc_bm
                pix_fmt: encoder pixel format, support NV12 and I420
                enc_params: encoder enc_params, "width=1902:height=1080:gop=32:gop_preset=3:framerate=25:bitrate=2000", width and height are necessary.
                cache_buffer_length: The internal cache queue length defaults to 5.
                abort_policy: The reject policy for video_write. 0 for returns -1 immediately. 1 for pop queue header. 2 for clear the queue. 3 for blocking.
            """

        def __init__(output_path: str, device_id: int, enc_fmt: str, pix_fmt: str, enc_params: str, cache_buffer_length: int=5, abort_policy: int=0):
            """ Encoder Constructor for video or rtsp/rtmp stream.

                Parameters:
                ----------
                output_path: local video file path or rtsp/rtmp address
                device_id: encoder device id, encoder will create a Handle when specify device id.
                enc_fmt: encoder format, support h264_bm and h265_bm/hevc_bm
                pix_fmt: encoder pixel format, support NV12 and I420
                enc_params: encoder enc_params, "width=1920:height=1080:gop=32:gop_preset=3:framerate=25:bitrate=2000", width and height are necessary.
                cache_buffer_length: The internal cache queue length defaults to 5.
                abort_policy: The reject policy for video_write. 0 for returns -1 immediately. 1 for pop queue header. 2 for clear the queue. 3 for blocking.
            """

**2). is_opened**

    .. code-block:: python

        def is_opened(self) -> bool:
            """ get encoder status

            Returns
            -------
            return true for opened
            """

**3). pic_encode**

    .. code-block:: python

        def pic_encode(ext, image) -> numpy.array:
            """ encoder picture.

            Parameters:
            ----------
            ext : encode format, ".jpg" , ".png" etc. .
            image : reference for input BMImage, only support FORMAT_BGR_PACKED, 1N byte.

            Returns
            -------
            return encoded data in system memory.
            """

**4). video_write**

    .. code-block:: python

        def video_write(image) -> int:
            """ Send a frame of image to the video encoder. Asynchronous interface, after format conversion, is placed in the internal cache queue.

            Parameters:
            ----------
            image : On BM1684, it is required that the image shape be consistent with the width and height specified by the encoder, and use bmcv_image_storage_convert to perform format conversion.
                    On BM1684X, use bmcv_image_vpp_convert to resize and format conversion.

            Returns
            -------
            Successfully returned 0, internal cache queue full returned -1. encode failed returns -2. push stream failed returns -3. unknown abort policy returns -4.

            """

**5). release**

    .. code-block:: python

        def release() -> None:
            """ release encoder
            """