SAIL
====

SAIL(SOPHON Artificial Intelligent Library) is the core module of SOPHON-SAIL.
SAIL wraps BMLib, sophon-mw(For BM1688 or CV186AH, named sophon-media), BMCV, and BMRuntime in SOPHONSDK, 
The original functions in SOPHONSDK such as "load bmodel and drive Tensor Computing Processor inference", "drive Tensor Computing Processor for image processing" and "drive VPU for image and video decoding" are abstracted into simpler C++ interfaces and provided;
and wrapped again with pybind11 to provide a clean and easy to use python interface.

Currently, all classes, enumerations and functions in the SAIL module are under the "sail" namespace.
The documentation in this unit will give you an in-depth introduction to the modules and classes in SAIL that you may use.
The core classes are as follows:

* Handle: 

Wrapper class for bm_handle_t of BMLib in SDK (device handle, contextual information), used to interact with kernel driver information.

* Tensor:

Wrapper class for BMLib in SDK, encapsulating device memory management and synchronization with system memory.

* Engine:

Wrapper class for the BMRuntime in the SDK that loads the bmodel and drives the Tensor Computing Processor for reasoning.
An Engine instance can load an arbitrary bmodel that
automatically manages the memory corresponding to the input tensor and the output tensor.

* Decoder:

Use VPU to decode video and JPU to decode images, both in hardware.

* Bmcv:

Wrapper class for BMCV in the SDK, encapsulating a series of image processing functions that can drive the Tensor Computing Processor for image processing.

