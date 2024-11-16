.. _Get weight files of images:

Get weight files of images
===========================================

**1. Open the splicing parameter tuning tool in the Windows system, and the download link is as follows:**

    .. parsed-literal::

        python3 -m dfss --url=open@sophgo.com:/sophon-stream/dwa_blend_encode/stitchtool_circular_fisheye_v240103.7.zip

**2. Stitch two pictures together to get a stitched picture**
    The command description is as follows：

    .. parsed-literal::

        gen_gridinfo.exe mode result_path left_image_path right_image_path fusion_start_position fusion_width

    Take a picture with a width of 2240 as an example：

    .. parsed-literal::

        gen_gridinfo.exe 1 ./result_path ./L/left.png ./R/right.png 2144 32

    *Note: The last two values ​​need to be divisible by 32, and their sum does not need to be equal to 2240. When merging, the right edge of the left image will directly take the right image. For example, in the example, 2240 – 2144 – 32 = 64, that is, the right edge of the left image with 64 pixels will directly take the right image.*

**3. Adjust stitching parameters according to the stitching diagram**

    .. parsed-literal::

        gen_gridinfo.exe 0 ./L ./L/left.bmp x_shift y_shift theta_x theta_y theta_z

* x_shift: Positive numbers move the image to the right, and negative numbers move the image to the left.

* y_shift: Positive numbers move the image downward, and negative numbers move the image upward.

* theta_x: rotation around the horizontal axis

* theta_y: rotation around the vertical axis

* theta_z: rotation perpendicular to the paper

*Note: These three parameters are generally not used and are set to 0 by default.*

Generally, the right image is fixed and the left image is adjusted to align. The size of the X direction during stitching is adjusted with 32 as the minimum adjustment unit, and the best stitching image is selected as the final result.
    
    .. parsed-literal::
        
        gen_gridinfo.exe 1 ./result_path ./L/left.png ./R/right.png 2144 32

**4. After the splicing is completed, the weight file is saved in result_path**

