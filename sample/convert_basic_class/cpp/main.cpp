#include "cvwrapper.h"
#include "opencv2/core.hpp"
#include "string"

int main() {
  int dev_id = 0;
  sail::Handle handle(dev_id);
  std::string image_name = "../dog.jpg";
  sail::Decoder decoder(image_name, true, dev_id);
  sail::BMImage BMimg = decoder.read(handle);
  sail::Bmcv bmcv(handle);
  // Get the image information
  int width = BMimg.width();
  int height = BMimg.height();
  bm_image_format_ext format = BMimg.format();
  bm_image_data_format_ext dtype = BMimg.dtype();
  // Get the device id and handle
  int device_id = BMimg.get_device_id();
  sail::Handle handle_ = BMimg.get_handle();
  int plane_num = BMimg.get_plane_num();
  std::cout << "Width: " << width << ", Height: " << height
            << ", Format: " << format << ", Data Type: " << dtype
            << ", Device ID: " << device_id << ", Plane Num: " << plane_num
            << std::endl;
  int ret;
  // bm_image -> tensor
  sail::Tensor tensor_from_bm_image(handle, {width, height}, BM_FLOAT32, true, true);
  bmcv.bm_image_to_tensor(BMimg, tensor_from_bm_image);

  // tensor -> bm_image
  sail::BMImage bm_image_from_tensor(handle, height, width, BMimg.format(), BMimg.dtype());
  bmcv.tensor_to_bm_image(tensor_from_bm_image, bm_image_from_tensor, std::string("nhwc"));
  bmcv.imwrite("./bmimg_from_tensor.jpg", bm_image_from_tensor);

  // bm_image -> mat
  cv::Mat mat_from_bm_image;
  ret = bmcv.bm_image_to_mat(BMimg, mat_from_bm_image);
  

  // mat -> bm_image
  sail::BMImage bmimg_from_mat(handle, height, width, BMimg.format(),
                               BMimg.dtype());
  ret = bmcv.mat_to_bm_image(mat_from_bm_image, bmimg_from_mat);
  bmcv.imwrite("./bmimg_from_mat.jpg", bmimg_from_mat);

  // mat -> tensor
  sail::Tensor tensor_from_mat(handle, {width, height}, BM_FLOAT32, true, true);
  sail::mat_to_tensor(mat_from_bm_image, tensor_from_mat);
  return 0;
}
