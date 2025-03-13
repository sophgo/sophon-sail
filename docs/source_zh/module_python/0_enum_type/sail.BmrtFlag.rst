sail.BmrtFlag
_____________


使用EngineLLM加载bmodel时的BmrtFlag。
详细信息请参考《BMRuntime 开发参考手册》的 ``bm_runtime_flag_t`` 。


**接口形式:**
    .. code-block:: python

        sail.BmrtFlag.BM_RUNTIME_AUTO
        sail.BmrtFlag.BM_RUNTIME_SHARE_MEM

**参数说明:**

* BM_RUNTIME_AUTO

加载bmodel的默认flag。加载自动处理。

* BM_RUNTIME_SHARE_MEM

不同net之间共享内存。