sail.PaddingAtrr
___________________


\_\_init\_\_
>>>>>>>>>>>>>>>

**Interface:**
    .. code-block:: python

        def __init__(self):
            

**Interface:**
    .. code-block:: python

        def __init__(self, 
                    stx: int, 
                    sty: int, 
                    width: int, 
                    height: int, 
                    r: int, 
                    g: int, 
                    b: int)

            
**Parameters**

* stx : int

Offset x information relative to the origin of dst image

* sty : int

Offset y information relative to the origin of dst image

* width : int

The width after resize

* height : int

The height after resize

* r : int

Pixel value information of R channel

* g : int

Pixel value information of G channel

* b : int

Pixel value information of B channel



set_stx
>>>>>>>>>>>>>>>

set offset stx.

**Interface:**
    .. code-block:: python

        def set_stx(self, stx: int)
 

**Parameters**

* stx : int

Offset x information relative to the origin of dst image


set_sty
>>>>>>>>>>>>>>>

set offset sty.

**Interface:**
    .. code-block:: python

        def set_sty(self, sty: int)
 

**Parameters**

* sty : int

Offset y information relative to the origin of dst image


set_w
>>>>>>>>>>>>>>>

set width.

**Interface:**
    .. code-block:: python

        def set_w(self, width: int)
 

**Parameters**

* width : int

The width after resize


set_h
>>>>>>>>>>>>>>>

set height.

**Interface:**
    .. code-block:: python

        def set_h(self, height: int)
 

**Parameters**

* height : int

The height after resize



set_r
>>>>>>>>>>>>>>>

set R.

**Interface:**
    .. code-block:: python

        def set_r(self, r: int)
 

**Parameters**

* r : int

Pixel value information of R channel



set_g
>>>>>>>>>>>>>>>

set G.

**Interface:**
    .. code-block:: python

        def set_g(self, g: int):
 

**Parameters**

* g : int

Pixel value information of G channel


            
set_b
>>>>>>>>>>>>>>>

set B

**Interface:**
    .. code-block:: python

        def set_b(self, b: int)


**Parameters**

b : int

Pixel value information of B channel
