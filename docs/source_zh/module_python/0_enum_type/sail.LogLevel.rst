sail.LogLevel
_____________


LogLevel用于定义日志级别。最高为 ``TRACE`` ，最低为 ``OFF`` ，默认为 ``INFO`` 。


**接口形式:**
    .. code-block:: python

        sail.LogLevel.TRACE
        sail.LogLevel.DEBUG
        sail.LogLevel.INFO
        sail.LogLevel.WARN
        sail.LogLevel.ERR
        sail.LogLevel.CRITICAL
        sail.LogLevel.OFF

**参数说明:**

* TRACE

打印 ``TRACE`` 级别和更低级别的日志。

* DEBUG

打印 ``DEBUG`` 级别和更低级别的日志。

* INFO

打印 ``INFO`` 级别和更低级别的日志。

* WARN

打印 ``WARN`` 级别和更低级别的日志。

* ERR

打印 ``ERR`` 级别和更低级别的日志。

* CRITICAL

打印 ``CRITICAL`` 级别和更低级别的日志。

* OFF

关闭各个级别的日志打印。