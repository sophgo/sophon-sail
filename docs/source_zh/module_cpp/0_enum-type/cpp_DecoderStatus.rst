DecoderStatus
___________________

DecoderStatus用于定义MultiDecoder中解码通道的状态。

**接口形式:**

    .. code-block:: cpp

        enum class DecoderStatus: int
        {
            OPENED = 0,
            CLOSED = 1
        };

**参数说明:**

* OPENED

表示该通道处于打开状态。

* CLOSED

表示该通道处于关闭状态。