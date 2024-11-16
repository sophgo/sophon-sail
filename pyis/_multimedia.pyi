"""
SAIL multimedia
===============

Provides
  1.  Codec
  2.  Bmcv

"""

import enum
from typing import Any
import numpy
from ._basic import *

class Format(enum.Enum):
    FORMAT_YUV420P = 0
    FORMAT_YUV422P = 1
    FORMAT_YUV444P = 2
    FORMAT_NV12 = 3
    FORMAT_NV21 = 4
    FORMAT_NV16 = 5
    FORMAT_NV61 = 6
    FORMAT_NV24 = 7
    FORMAT_RGB_PLANAR = 8
    FORMAT_BGR_PLANAR = 9
    FORMAT_RGB_PACKED = 10
    FORMAT_BGR_PACKED = 11
    FORMAT_RGBP_SEPARATE = 12
    FORMAT_BGRP_SEPARATE = 13
    FORMAT_GRAY = 14
    FORMAT_COMPRESSED = 15
    FORMAT_ARGB_PACKED = 16

class ImgDtype(enum.Enum):
    DATA_TYPE_EXT_FLOAT32 = 0
    DATA_TYPE_EXT_1N_BYTE = 1
    DATA_TYPE_EXT_4N_BYTE = 2
    DATA_TYPE_EXT_1N_BYTE_SIGNED = 3
    DATA_TYPE_EXT_4N_BYTE_SIGNED = 4

class bmcv_resize_algorithm(enum.Enum):
    BMCV_INTER_NEAREST = 0
    BMCV_INTER_LINEAR = 1
    BMCV_INTER_BICUBIC = 2

class DecoderStatus(enum.Enum):
    NONE = -1
    OPENED = 0
    CLOSED = 1
    STATUS_MAX = 2

def set_decoder_env(env_name: str, env_value: str) -> None:
    """ Set Decoder environment, must set befor Decoder Constructor, else use default values
    
    refcounted_frames, extra_frame_buffer_num, rtsp_transport, stimeout, rtsp_flags, buffer_size, max_delay, probesize, analyzeduration
    """
    pass

def base64_encode(handle: Handle, input_bytes: bytes) -> bytes:
    """ Encode a base64 string """
    pass

def base64_decode(handle: Handle, encode_bytes: bytes) -> bytes:
    """ Decoder a base64 string """
    pass

def base64_encode_array(handle: Handle, input_arr: numpy.ndarray) -> bytes:
    """ Encode a base64 string """
    pass

def base64_decode_asarray(handle: Handle, encode_arr_bytes: bytes, array_type:str = "uint8") -> numpy.ndarray:
    """ Decoder a base64 string """
    pass

class bm_image:
    def width(self) -> int: pass
    def height(self) -> int: pass
    def format(self) -> Format: pass
    def dtype(self) -> ImgDtype: pass
        
class BMImage:
    def __init__(self) -> BMImage: pass

    def __init__(self, handle: Handle, 
        h: int, w: int, 
        format: Format, 
        dtype: ImgDtype) -> BMImage: pass

    def __init__(self, handle: Handle, 
        buffer: bytes | numpy.ndarray, 
        h: int, w: int, 
        format: Format, 
        dtype: ImgDtype = ImgDtype.DATA_TYPE_EXT_1N_BYTE, 
        strides: list = [], 
        offset: int = 0
        ) -> BMImage: pass

    def width(self) -> int: pass

    def height(self) -> int: pass

    def format(self) -> int: pass

    def dtype(self) -> int: pass

    def data(self) -> bm_image: 
        """ Get inner bm_image  """
        pass

    def get_plane_num(self) -> int: pass

    def need_to_free(self) -> bool: pass

    def empty_check(self) -> int: pass

    def get_device_id(self) -> int: pass

    def get_handle(self) -> Handle:
        """ Get Handle of BMImage """
        pass

    def asmat(self) -> numpy.ndarray[numpy.uint8]: pass

    def asnumpy(self) -> numpy.ndarray: 
        """
        Convert BMImage to numpy.ndarray containing raw data, without color 
        space convert.
        """
        pass


class Decoder:
    def __init__(self, file_path: str, compressed: bool =  True, dev_id: int = 0) -> Decoder:
        """ 
        Decoder by VPU

        Parameters:
        ----------
        file_path : str
            Path or rtsp url to the video/image file.
        compressed : bool, optional
            Whether the format of decoded output is compressed NV12.
            Defaults is True.
        dev_id : int, optional
            ID of TPU. 
            Defaults to 0.
        """
        pass

    def is_opened(self) -> bool:
       """ Judge if the source is opened successfully. """
       pass
    
    def get_frame_shape(self) -> list[int]:
        """ Get frame shape in the Decoder.
       
        Returns
        -------
        list[int], [1, C, H, W]
        """
        pass

    def read(self, handle: Handle) -> BMImage: pass
    # def read(self, handle: Handle, image: BMImage) -> int: pass
    def read_(self, handle: Handle, image: bm_image) -> None: pass    
    def get_fps(self) -> float: pass
    def release(self) -> None: pass
    def reconnect(self) -> int: pass
    
    def enable_dump(self, dump_max_seconds: int) -> None: pass
    def disable_dump(self) -> None: pass
    def dump(self, dump_pre_seconds: int, dump_post_seconds: int, file_path: str) -> int:
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
        pass

    def get_pts_dts(self) -> list:
        """ Get the pts and dts
        
        Returns
        -------
            lists, pts and dts
        """
        pass

class Decoder_RawStream:
    def __init__(self, tpu_id: int = 0 , decformat: str = None ) -> Decoder_RawStream:
        pass

    def read_(self, data_bytes: bytes, image: bm_image, continue_frame: bool = False) -> int:
        """
        Read from the decoder.
        Parameters:
        - data_bytes: Input h264 or h265 data as bytes.
        - image: Output bm_image.
        - continue_frame: Flag to indicate if it's a continuous frame.
        Returns:
        - int: 0 for success, other values for failure.
        """
        pass

    def read(self, data: bytes, image: BMImage, continue_frame: bool = False) -> int:
        pass

    def release(self) -> None: pass


class PaddingAtrr:
    def __init__(self): pass
    def __init__(self,stx: int, sty: int, w: int, h: int, \
        r: int, g: int, b: int): pass
    def set_stx(self, stx: int) -> None : pass
    def set_sty(self, sty: int) -> None : pass
    def set_w(self, w: int) -> None : pass
    def set_h(self, h: int) -> None : pass
    def set_r(self, r: int) -> None : pass
    def set_g(self, g: int) -> None : pass
    def set_b(self, b: int) -> None : pass


class Bmcv:
    def __init__(self, handle: Handle) -> None: pass

    def bm_image_to_tensor(self, img: BMImage|BMImageArray, tensor: Tensor) -> None:
        """
        Convert BMImage|BMImageArray to tensor

        Parameters:
        ----------
        img : BMImage|BMImageArray(Input image)

        Returns
        -------
        tensor: Tensor(Output tensor)
        """
        pass

    def bm_image_to_tensor(self, img: BMImage|BMImageArray) -> Tensor:
        """
        Convert BMImage|BMImageArray to tensor

        Parameters:
        ----------
        img : BMImage|BMImageArray(Input image)

        Returns
        -------
        tensor: Tensor(Output tensor)
        """
        pass

    def tensor_to_bm_image(self, tensor: Tensor, img: BMImage|BMImageArray, bgr2rgb: bool = False, layout: str = 'nchw') -> None:
        """
        Convert tensor to BMImage|BMImageArray

        Parameters:
        ----------
        tensor : Tensor
            Input tensor.
        bgr2rgb : bool, optional
            Swap color channel, default is False.
        layout : str, optional
            Layout of the input tensor: 'nchw' or 'nhwc', default is 'nchw'.

        Returns
        -------
        img: BMImage|BMImageArray
            Output image
        """
        pass
    
    def tensor_to_bm_image(self, tensor: Tensor, img: BMImage|BMImageArray, format: Format) -> None:
        """
        Convert tensor to BMImage|BMImageArray

        Parameters:
        ----------
        tensor : Tensor
            Input tensor.
        format : Format
            Format of the BMImage.

        Returns
        -------
        img: BMImage|BMImageArray
            Output image
        """
        pass

    def tensor_to_bm_image(self, tensor: Tensor, bgr2rgb: bool = False, layout: str = 'nchw') -> BMImage:
        """
        Convert tensor to BMImage

        Parameters:
        ----------
        tensor : Tensor
            Input tensor.
        bgr2rgb : bool, optional
            Swap color channel, default is False.
        layout : str, optional
            Layout of the input tensor: 'nchw' or 'nhwc', default is 'nchw'.

        Returns
        -------
        img: BMImage
            Output image
        """
        pass

    def tensor_to_bm_image(self, tensor: Tensor, format: Format) -> BMImage:
        """
        Convert tensor to BMImage

        Parameters:
        ----------
        tensor : Tensor
            Input tensor.
        format : Format
            Format of the BMImage.

        Returns
        -------
        img: BMImage
            Output image
        """
        pass
    
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
        pass

    def mat_to_bm_image(self, mat: numpy.ndarray[numpy.uint8], img: BMImage) -> int:
        """
        Convert cv mat to BMImage
        
        Parameters:
        ----------
        mat: input cv mat, rgb_packed.
        img: output BMImage
        """
        pass
    
    def crop_and_resize(self, input: BMImage|BMImageArray,
        crop_x0: int, crop_y0: int, crop_w: int, crop_h: int, resize_w: int, resize_h: int, 
        resize_alg: bmcv_resize_algorithm = bmcv_resize_algorithm.BMCV_INTER_NEAREST) -> BMImage|BMImageArray:
        """Crop then resize an image. """
        pass
    
    def crop(self, input: BMImage|BMImageArray, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int) -> BMImage|BMImageArray:
        """Crop an image with given window. """
        pass

    def resize(self, input: BMImage|BMImageArray, resize_w: int, resize_h: int, 
               resize_alg: bmcv_resize_algorithm = bmcv_resize_algorithm.BMCV_INTER_NEAREST) -> BMImage|BMImageArray:
        """Resize an image with interpolation of INTER_NEAREST. """
        pass

    def vpp_crop_and_resize(self, input: BMImage|BMImageArray,
        crop_x0: int, crop_y0: int, crop_w: int, crop_h: int, resize_w: int, resize_h: int, 
        bmcv_resize_algorithm = bmcv_resize_algorithm.BMCV_INTER_NEAREST) -> BMImage|BMImageArray:
        """Crop then resize an image using vpp. """
        pass

    def vpp_crop(self, input: BMImage|BMImageArray, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int) -> BMImage|BMImageArray:
        """Crop an image with given window using vpp. """
        pass

    def vpp_resize(self, input: BMImage|BMImageArray, output: BMImage|BMImageArray, resize_w: int, resize_h: int, 
                   bmcv_resize_algorithm = bmcv_resize_algorithm.BMCV_INTER_NEAREST) -> None:
        """Resize an image with interpolation of INTER_NEAREST using vpp. """
        pass
    
    def vpp_resize(self, input: BMImage|BMImageArray, resize_w: int, resize_h: int, 
                   bmcv_resize_algorithm = bmcv_resize_algorithm.BMCV_INTER_NEAREST) -> BMImage|BMImageArray:
        """Resize an image with interpolation of INTER_NEAREST using vpp. """
        pass

    def vpp_crop_and_resize_padding(self, input: BMImage|BMImageArray, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int, 
        resize_w: int, resize_h: int, padding_in: PaddingAtrr, 
        resize_alg: bmcv_resize_algorithm = bmcv_resize_algorithm.BMCV_INTER_NEAREST) -> BMImage|BMImageArray:
        """Crop then resize an image using vpp. """
        pass
    
    def crop_and_resize_padding(self, input: BMImage, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int, 
        resize_w: int, resize_h: int, padding_in: PaddingAtrr) -> BMImage:
        """Crop then resize an image. """
        pass

    def vpp_crop_padding(self, input: BMImage|BMImageArray, resize_w: int, resize_h: int, padding_in: PaddingAtrr) -> BMImage|BMImageArray:
        """Crop an image with given window using vpp. """
        pass

    def vpp_resize_padding(self, input: BMImage|BMImageArray, resize_w: int, resize_h: int, padding_in: PaddingAtrr,
                           bmcv_resize_algorithm = bmcv_resize_algorithm.BMCV_INTER_NEAREST) -> BMImage|BMImageArray:
        """Resize an image with interpolation of INTER_NEAREST using vpp. """
        pass

    def yuv2bgr(self, input: BMImage|BMImageArray) -> BMImage|BMImageArray: 
        """Convert an image from YUV to BGR."""
        pass

    def warp(self, input: BMImage|BMImageArray, matrix, use_bilinear: int = 0, similar_to_opencv: bool = False) -> BMImage|BMImageArray:
        """
        Applies an affine transformation to an image.
        
        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        matrix: 2x3 transformation matrix
        use_bilinear: Bilinear use flag, 0 for nearest interpolation and 1 for bilinear interpolation
        similar_to_opencv: Whether to use the interface aligning the affine transformation interface of OpenCV

        Returns:
        -------
        Output image
        """
        pass
    def warp(self, input: BMImage|BMImageArray, output: BMImage|BMImageArray, matrix, use_bilinear: int = 0, similar_to_opencv: bool = False) -> int:
        """
        Applies an affine transformation to an image.
        
        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        output: BMImage|BMImageArray(Output image)
        matrix: 2x3 transformation matrix
        use_bilinear: Bilinear use flag, 0 for nearest interpolation and 1 for bilinear interpolation
        similar_to_opencv: Whether to use the interface aligning the affine transformation interface of OpenCV

        Returns:
        -------
        int, 0 for success
        """
        pass
    
    def warp_perspective(self, input: BMImage, coordinate, output_width: int,  output_height: int, \
        format: Format = Format.FORMAT_BGR_PLANAR,  dtype: ImgDtype = ImgDtype.DATA_TYPE_EXT_1N_BYTE, \
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
        Format: Output image format, Only FORMAT_BGR_PLANAR,FORMAT_RGB_PLANAR 
        dtype: Output image dtype, Only DATA_TYPE_EXT_1N_BYTE,DATA_TYPE_EXT_4N_BYTE
        use_bilinear: Bilinear use flag.

        Returns:
        -------
        Output image
        """
        pass

    def convert_to(self, input: BMImage|BMImageArray, output: BMImage|BMImageArray, alpha_beta) -> None:
        """
        Applies a linear transformation to an image.

        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        alpha_beta:like (a0, b0), (a1, b1), (a2, b2) factors

        Returns:
        output: BMImage|BMImageArray(Output image)
        """
        pass

    def convert_to(self, input: BMImage|BMImageArray, alpha_beta) -> BMImage|BMImageArray:
        """
        Applies a linear transformation to an image.

        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        alpha_beta:  (a0, b0), (a1, b1), (a2, b2) factors

        Returns:
        output: BMImage|BMImageArray(Output image)
        """
        pass

    def rectangle(self, image: BMImage, x: int, y: int, w: int, h: int, color, thickness: int = 1) -> None:
        """
        Draw a rectangle on input image.

        Parameters:
        ----------
        image: BMImage, Input image
        x: int, Start point x of rectangle
        y: int, Start point y of rectangle
        w: int, Width of rectangle
        h: int, Height of rectangle
        color: Color of rectangle, like (0, 0, 255)
        thickness: int, optional, default is 1
        """
        pass

    def rectangle(self, image: bm_image, x: int, y: int, w: int, h: int, color, thickness: int = 1) -> None:
        """
        Draw a rectangle on input image.

        Parameters:
        ----------
        image: bm_image, Input image
        x: int, Start point x of rectangle
        y: int, Start point y of rectangle
        w: int, Width of rectangle
        h: int, Height of rectangle
        color: Color of rectangle, like (0, 0, 255)
        thickness: int, optional, default is 1
        """
        pass
    def fillRectangle(self, image: BMImage, x: int, y: int, w: int, h: int, color) -> None:
        """
        Fill a rectangle on input image.

        Parameters:
        ----------
        image: BMImage, Input image
        x: int, Start point x of rectangle
        y: int, Start point y of rectangle
        w: int, Width of rectangle
        h: int, Height of rectangle
        color: Color of rectangle, like (0, 0, 255)
        """
        pass
    def fillRectangle(self, image: bm_image, x: int, y: int, w: int, h: int, color) -> None:
        """
        Fill a rectangle on input image.

        Parameters:
        ----------
        image: bm_image, Input image
        x: int, Start point x of rectangle
        y: int, Start point y of rectangle
        w: int, Width of rectangle
        h: int, Height of rectangle
        color: Color of rectangle, like (0, 0, 255)
        """
        pass
    def putText(self, image: BMImage, text: str, x: int, y: int, color, fontScale: float, thickness: int=1) -> None:
        """
        Draw Text on input image.

        Parameters:
        ----------
        image: BMImage, Input image
        text: str, Text string to be drawn
        x: int, Start point x
        y: int, Start point y
        color: color of text, like(0, 0, 255)
        fontScale: float, Font scale factor that is multiplied by the font-specific base size
        thickness: int, optional, default is 1
        """
        pass

    def putText(self, image: bm_image, text: str, x: int, y: int, color, fontScale: float, thickness: int=1) -> None:
        """
        Draw Text on input image.

        Parameters:
        ----------
        image: bm_image, Input image
        text: str, Text string to be drawn
        x: int, Start point x
        y: int, Start point y
        color: color of text, like(0, 0, 255)
        fontScale: float, Font scale factor that is multiplied by the font-specific base size
        thickness: int, optional, default is 1
        """
        pass


    def imwrite(self, filename: str, image: BMImage) -> None:
        """
        Save the image to the specified file.

        Parameters:
        ----------
        filename: str
            Name of the save file.
        image: BMImage
            Image to be saved.
        """
        pass

    def imwrite(self, filename: str, image: bm_image) -> None:
        """
        Save the image to the specified file.
        
        Parameters:
        ----------
        filename: str
            Name of the save file.
        image: bm_image
            Image to be saved.
        """
        pass

    def get_handle(self) -> Handle: 
        """Get Handle instance. """
        pass

    def convert_format(self, input: BMImage, output: BMImage) -> None: 
        """Convert input to output format. """
        pass

    def convert_format(self, input: BMImage) -> BMImage: 
        """Convert an image to BGR PLANAR format. """
        pass

    def vpp_convert_format(self, input: BMImage, output: BMImage) -> None: 
        """Convert input to output format using vpp. """
        pass

    def vpp_convert_format(self, input: BMImage) -> BMImage: 
        """Convert an image to BGR PLANAR format using vpp. """
        pass

    def get_bm_data_type(self, format: ImgDtype) -> Dtype: pass

    def get_bm_image_data_format(self, dtype: Dtype) -> ImgDtype: pass

    def image_add_weighted(self, input1: BMImage, alpha: float, input2: BMImage, \
        beta: float, gamma: float, output: BMImage) -> int:  
        """output = input1 * alpha + input2 * beta + gamma."""
        pass

    def image_add_weighted(self, input1: BMImage, alpha: float, input2: BMImage, \
        beta: float, gamma: float) -> BMImage:
        """output = input1 * alpha + input2 * beta + gamma."""
        pass

    def image_copy_to(self, input: BMImage|BMImageArray, output: BMImage|BMImageArray, \
        start_x: int, start_y: int) -> None:
        """
        Copy the input to the output.

        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        output: BMImage|BMImageArray(Output image)
        start_x: point start x
        start_y: point start y
        """
        pass

    def image_copy_to_padding(self, input: BMImage|BMImageArray, output: BMImage|BMImageArray, \
        padding_r: numpy.uint8, padding_g: numpy.uint8, padding_b: numpy.uint8, start_x: int, start_y: int) -> None:
        """
        Copy the input to the output width padding.

        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        output: BMImage|BMImageArray(Output image)
        padding_r: r value for padding
        padding_g: g value for padding
        padding_b: b value for padding
        start_x: point start x
        start_y: point start y
        """
        pass

    def convert_yuv420p_to_gray(self, input:BMImage, output:BMImage) -> int:
        """
        Convert YUV420P to gray image.

        Parameters:
        ----------
        input: BMImage(Input image)
        output: BMImage(Output image)
        """
        pass

    def convert_yuv420p_to_gray(self, input:bm_image, output:bm_image) -> int:
        """
        Convert YUV420P to gray image.

        Parameters:
        ----------
        input: bm_image(Input image)
        output: bm_image(Output image)
        """
        pass

    def nms(self, input: numpy.ndarray[Any, numpy.dtype[numpy.float32]], threshold: float) -> numpy.ndarray[Any, numpy.dtype[numpy.float32]]:
        """
        Do nms use tpu.

        Parameters:
        ----------
        input: input proposal array
            shape must be (n,5) n<56000, proposal:[left,top,right,bottom,score]
        threshold: nms threshold

        Returns:
        ----------
        return nms result, numpy.ndarray[Any, numpy.dtype[numpy.float32]]
        """
        pass

    def drawPoint(self, image: BMImage, center: tuple[int, int], color: tuple[int, int, int], radius: int) -> int:
        """
        Draw Point on input image.

        Parameters:
        ----------
        image: BMImage, Input image
        center: Tuple[int, int], center of point, (point_x, point_y)
        color: Tuple[int, int, int], color of drawn, (b,g,r)
        radius: Radius of drawn
        """
        pass

    def drawPoint(self, image: bm_image, center: tuple[int, int], color: tuple[int, int, int], radius: int) -> int:
        """
        Draw Point on input image.

        Parameters:
        ----------
        image: bm_image, Input image
        center: Tuple[int, int], center of point, (point_x, point_y)
        color: Tuple[int, int, int], color of drawn, (b,g,r)
        radius: Radius of drawn
        """
        pass

    def polylines(self, image: BMImage, pts: list[list[tuple[int, int]]], isClosed: bool, color: tuple[int, int, int], thickness: int = 1, shift: int = 0) -> int:
        """
        Draw one or more line segments on an image and polygonal curves.

        Parameters:
        ----------
        image: input image
        pts: The starting point and end point coordinates of the line segment, multiple coordinate points can be entered. 
            The upper left corner of the image is the origin, 
            extending to the right in the x direction and extending down in the y direction.
        isClosed: Whether the graph is closed.
        color: The color of the line is the value of the three RGB channels.
        thickness: The width of the lines is recommended to be even for YUV format images.
        shift: Polygon scaling multiple. Default is not scaling. The scaling factor is(1/2)^shift。

        Returns:
        ----------
        returns 0 if success. 
        """
        pass
        
    def mosaic(self, mosaic_num: int, img: BMImage, rects: list[list[int,int,int,int]], is_expand:int) -> int:
        """
        Print one or more mosaics on an image.

        Parameters:
        ----------
        mosaic_num: Number of mosaics, length of list in rects.
        img: Input BMImage.
        rects: Multiple Mosaic positions, the parameters in each element in the list are 
            [Mosaic at X-axis start point, Mosaic at Y-axis start point, Mosaic width, Mosaic height].
        is_expand:A value of 0 means that the column is not expanded, 
            and a value of 1 means that a macro block (8 pixels)

        Returns:
        ----------
        returns 0 if success. 
        """
        pass

    def transpose(self, src: BMImage, dst: BMImage) -> int:
        """
        Transpose of image width and height.

        Parameters:
        ----------
        src: Input BMImage
        dst: Output BMImage

        Returns:
        ----------
        returns 0 if success. 
        """
        pass    

    def transpose(self, src: BMImage) -> BMImage:
        """
        Transpose of image width and height.

        Parameters:
        ----------
        src: Input BMImage

        Returns:
        ----------
        Output BMImage 
        """
        pass

    def gaussian_blur(self, input: BMImage, kw: int, kh : int, sigmaX : float, sigmaY : float = 0.0) -> BMImage: 
        """
        Gaussian blur
        
        Parameters:
        ----------
        input: BMImage
        kw: int, size of kernel in the width direction.
        kh: int, size of kernel in the height direction.
        sigmaX : float, gaussian kernel standard deviation in the X direction.
        sigmaY : float, gaussian kernel standard deviation in the Y direction.Default is 0.0, 
                which means that it is the same standard deviation as the Gaussian kernel in the X direction.

        Returns:
        ----------
        return gaussian blur image
        """
        pass

    def Sobel(self, input: BMImage, output: BMImage, dx: int, dy: int, ksize: int = 3, scale: float = 1, delta: float = 0) -> int:
        """
        Edge detection Sobel Operator.

        Parameters:
        ----------
        input         Input BMImage 
        output        Output BMImage
        dx            Order of the derivative x.
        dy            Order of the derivative y
        ksize         ize of the extended Sobel kernel; it must be -1, 1, 3, 5, or 7.
        scale         Optional scale factor for the computed derivative values; by default, no scaling is applied
        delta         Optional delta value that is added to the results prior to storing them in dst.
        
        Returns:
        ----------
        return 0 if success
        """
        pass

    def Sobel(self, input: BMImage, dx: int, dy: int, ksize: int = 3, scale: float = 1, delta: float = 0) -> BMImage:
        """
        Edge detection Sobel Operator.

        Parameters:
        ----------
        input         Input BMImage 
        dx            Order of the derivative x.
        dy            Order of the derivative y
        ksize         ize of the extended Sobel kernel; it must be -1, 1, 3, 5, or 7.
        scale         Optional scale factor for the computed derivative values; by default, no scaling is applied
        delta         Optional delta value that is added to the results prior to storing them in dst.
        
        Returns:
        ----------
        return processed BMImage
        """
        pass

    def imdecode(self, data_bytes: bytes) -> BMImage:
        """
        Load Jpeg image from system memory.

        Parameters:
        ----------
        data_bytes: image data bytes in system memory


        Returns:
        ----------
        return decoded image
        """
        pass

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
        pass

    def imread(self, filename: str) -> BMImage:
        """
        Read a jpeg image from file. Only support jpeg baseline image. The returned BMImage keeps YUV color space.

        Parameters:
        ----------
        filename: Name of file to be read.

        Returns:
        ----------
        Return decoded BMImage, whose pixel format is in YUV color space.
        """
        pass

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

    def fft(self, forward: bool, input_real: Tensor) -> list[Tensor]:
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
        pass
    
    def fft(self, forward: bool, input_real: Tensor, input_imag: Tensor) -> list[Tensor]:
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
        pass
    
    def bmcv_overlay(self, image: BMImage, overlay_info: list[list[int]], overlay_image: list[BMImage]) -> int:
        """
        Add a watermark with a transparent channel to the image

        Parameters:
        -----------
        image: BMImage, the base image
        overlay_info: a list of rect (x,y,w,h)
        overlay_image: a list of watermark image with a transparent channel

        Returns:
        return ret, 0 for success and others for failed
        """
class BMImageArray():
    def __init__(self) -> None: ...

    def __init__(self, handle: Handle, h: int, w: int, format: Format, dtype: ImgDtype ) -> None: ...

    def __len__(self) -> int: pass

    def __getitem__(self, i: int) -> bm_image: pass

    def __setitem__(self, i: int, data: bm_image) -> None: 
        """
        Copy the image to the specified index.

        Parameters:
        ----------
        i: int
            Index of the specified location.
        data: bm_image
            Input image.
        """
        pass

    def copy_from(self, i: int, data:BMImage) -> None: 
        """
        Copy the image to the specified index.

        Parameters:
        ----------
        i: int
            Index of the specified location.
        data: bm_image
            Input image.
        """
        pass

    def attach_from(self, i: int, data: BMImage) -> None: 
        """
        Attach the image to the specified index.(Because there is no memory copy, the original data needs to be cached)

        Parameters:
        ----------
        i: int
            Index of the specified location.
        data: BMImage
            Input image.
        """
        pass
    
    def watermark_superpose(self,
        image: BMImage,
        water_name:str,
        bitmap_type: int,
        pitch: int,
        rects: list[list[int]],
       color: tuple
                ) -> int:
        """
        Add a watermark_superpose to BMImage.

        Parameters:
        ----------
        image: BMImage
            input BMImage
        water_name:str
            input watermark_superpose
        bitmap_type: int
            bitmap_type of input watermark_superpose,0/1,0 8byte,1 1byte
        pitch: int
            the width of input watermark_superpose
        rects: list[list[int]]
            the rects of input watermark_superpose
        color: tuple
            the color of input watermark_superpose
        """
        pass
    
    def get_device_id(self) -> int: pass


class BMImageArray2D(BMImageArray): pass
class BMImageArray3D(BMImageArray): pass
class BMImageArray4D(BMImageArray): pass
class BMImageArray8D(BMImageArray): pass
class BMImageArray16D(BMImageArray): pass
class BMImageArray32D(BMImageArray): pass
class BMImageArray64D(BMImageArray): pass
class BMImageArray128D(BMImageArray): pass
class BMImageArray256D(BMImageArray): pass
    
class MultiDecoder:

    def __init__(self, queue_size: int = 10, tpu_id: int = 0, discard_mode: int = 0): 
        """ MultiDecoder Constructor.

        Parameters:
        ----------
        queue_size: int
            Max queue size,default is 10.
        tpu_id: int
            ID of TPU, default is 0.
        discard_mode: int
            Data discard policy when the queue is full. Default is 0.
            If 0, do not push the data to queue, else pop the data from queue and push new data to queue.
        """
        pass


    def set_read_timeout(self, timeout: int) -> None : 

        """ Set read frame timeout waiting time 
        
        Parameters:
        ----------
        timeout: int
            Set read frame timeout waiting time in seconds.
        """
        pass


    def add_channel(self, file_path: str, frame_skip_num: int = 0) -> int:
        
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
        pass

    def del_channel(self, channel_idx: int) -> int : 
        """ Delete channel
        
        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
        0 for success and other for failure.
        """
        pass

    def get_channel_fps(self, channel_idx: int) -> float:
        """ Get the fps of the video stream in a specified channel
        
        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
        Returns the fps of the video stream in the specified channel
        """
        pass

    def clear_queue(self, channel_idx: int) -> int : 
        """ Clear data cache queue 
        
        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
        0 for success and other for failure.
        """
        pass

    def read(self, channel_idx: int, image: BMImage, read_mode: int = 0) -> int : 
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
        pass

    def read(self,channel_idx: int) -> BMImage :
        """ Read a BMImage from the MultiDecoder with a given channel.

        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
            BMImage instance to be read to
        """
        pass


    def read_(self, channel_idx: int, image: bm_image, read_mode: int=0) -> int :
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
        pass

    def read_(self,channel_idx: int) -> int :
        """ Read a bm_image from the MultiDecoder with a given channel.

        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
            bm_image instance to be read to
        """
        pass

        
    def reconnect(self, channel_idx: int) -> int :
        """ Reconnect Decoder for instance channel
            
        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
        0 for success and other for failure.
            
        """
        pass


    def get_frame_shape(self, channel_idx: int) -> list[int]:
        """ Get frame for instance channel
        
        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns
        -------
        list[int], [1, C, H, W]
        """
        pass

    def set_local_flag(self, flag: bool) -> None: 
        """ Set local video flag

        Parameters:
        ----------
        flag : bool
            If flag is True, Decode up to 25 frames per second

        """
        pass

    def exhausted(self) -> None:
        """ Set a flag.
        just a status flag. Always uesd when image dataset has been traversed.

        """
        pass

    def get_exhausted_flag(self) -> bool:
        """ get exhausted flag

        """
        pass

    def get_channel_status(self, channel_idx: int) -> DecoderStatus:
        """ Get the status of the specific channel.
        
        Parameters:
        ----------
        channel_idx : int
            channel index

        Returns:
        -------
        Returns the status of the the specified channel
        """
        pass

class sail_resize_type(enum.Enum):
    BM_RESIZE_VPP_NEAREST = 0
    BM_RESIZE_TPU_NEAREST = 1
    BM_RESIZE_TPU_LINEAR = 2
    BM_RESIZE_TPU_BICUBIC = 3
    BM_PADDING_VPP_NEAREST = 4
    BM_PADDING_TPU_NEAREST = 5
    BM_PADDING_TPU_LINEAR = 6
    BM_PADDING_TPU_BICUBIC = 7


class Encoder:

    def __init__(self):
        """
        encoder constructor for pic
        """
        pass

    def __init__(self, output_path: str, handle: Handle, enc_fmt: str, pix_fmt: str, enc_params: str, cache_buffer_length: int=5, abort_policy: int=0):
        """
        encoder constructor for video

        Parameters:
        ----------
        output_path: local file path or stream url
        handle: sail.Handle
        enc_fmt: encoder name, support h264_bm and h265_bm
        pix_fmt: encoder pixel format, support I420 and NV12
        enc_params: encoder params, width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25
        cache_buffer_length: The internal cache queue length defaults to 6.
        abort_policy:The reject policy for video_write. 0 for returns -1 immediately. 1 for pop queue header. 2 for clear the queue. 3 for blocking.
        """
        pass

    def __init__(self, output_path: str, device_id: int, enc_fmt: str, pix_fmt: str, enc_params: str, cache_buffer_length: int=5, abort_policy: int=0):
        """
        encoder constructor for video

        Parameters:
        ----------
        output_path: local file path or stream url
        device_id: device id
        enc_fmt: encoder name, support h264_bm and h265_bm
        pix_fmt: encoder pixel format, support I420 and NV12
        enc_params: encoder params, width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25
        cache_buffer_length: The internal cache queue length defaults to 6.
        abort_policy:The reject policy for video_write. 0 for returns -1 immediately. 1 for pop queue header. 2 for clear the queue. 3 for blocking.
        """
        pass

    def is_opened(self) -> bool:

        pass

    def pic_encode(self, ext: str, image: BMImage) -> numpy.ndarray[Any, numpy.dtype[Any]]:
        """
        pic encode

        Parameters:
        ----------
        ext: encode format, .jpg .png ...
        image: input image

        Returns:
        ----------
        return pic data numpy.ndarray[Any, numpy.dtype[numpy.uint8_t]]
        """
        pass

    def pic_encode(self, ext: str, image: bm_image) -> numpy.ndarray[Any, numpy.dtype[Any]]:

        pass

    def video_write(self, image: BMImage) -> int:
        """
        On BM1684, it is required that the image shape be consistent with the width and height specified by the encoder, \
            video_write use bmcv_image_storage_convert to perform format conversion.
        On BM1684X, video_write use bmcv_image_vpp_convert to resize and format conversion.

        Parameters:
        ----------
        image: BMImage

        Returns:
        ----------
        Successfully returned 0, internal cache queue full returned -1. encode failed returns -2. push stream failed returns -3. unknown abort policy returns -4.
        """
        pass

    def video_write(self, image: bm_image) -> int:

        pass

    def release(self) -> None:
        """
        release resources
        only invoked when video encode finished
        """
        pass