.. _获取图片的权重文件:

获取图片的权重文件
===========================================

**1. 在windows系统中打开拼接调参工具**
    下载链接如下：

    .. parsed-literal::

        python3 -m dfss --url=open@sophgo.com:/sophon-stream/dwa_blend_encode/stitchtool_circular_fisheye_v240103.7.zip

**2. 将两张图片拼接得到拼接图**
    命令说明如下：

    .. parsed-literal::

        gen_gridinfo.exe mode result_path left_image_path right_image_path 融合区起始位置 融合区宽度

    以宽为2240的图片为例：

    .. parsed-literal::

        gen_gridinfo.exe 1 ./result_path ./L/left.png ./R/right.png 2144 32

    *注意：最后两个数值需要被32整除，相加不需要等于2240，融合时，左图右边缘会直接取右图，如示例中，2240 – 2144 – 32 = 64，即左图右边缘64pixel直接取右图。*

**3. 根据拼接图来调整拼接参数**

    .. parsed-literal::

        gen_gridinfo.exe 0 ./L ./L/left.bmp x_shift y_shift theta_x theta_y theta_z

* x_shift: 正数代表向右移动图片，负数代表向左移动图片。

* y_shift: 正数代表向下移动图片，负数代表向上移动图片。

* theta_x: 表示绕横轴旋转。

* theta_y: 表示绕纵轴旋转。

* theta_z: 表示垂直纸面旋转。

*注意: 一般不使用这三个参数，默认设置为0。*

一般固定右图不动，通过调节左图位置来对齐。以32为最小调节单位来调整拼接时X方向的大小，选择效果最好的拼接图作为最终结果。
    
    .. parsed-literal::
        gen_gridinfo.exe 1 ./result_path ./L/left.png ./R/right.png 2144 32

**4. 拼接完成后,权重文件保存在result_path中**

