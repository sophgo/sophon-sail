/* Copyright 2016-2022 by SOPHGO Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/

#include "ipc.h"
#include "cvwrapper.h"
#include <bmlib_runtime.h>
#include <bmruntime_interface.h>
#include <mutex>
#include <queue>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace sail {

class IPC::IPC_CC {
public:
  IPC_CC() = delete;
  IPC_CC(bool isSender_, const std::string &image_pipe_,
         const std::string &final_pipe_, bool usec2c, int queue_len);
  ~IPC_CC();

  void sendBMImage(BMImage &img, int channel_id, int frame_id);
  void sendTensor(Tensor &t, int channel_id, int frame_id);

  IPC::RecvTypeBMImage popImage();
  IPC::RecvTypeTensor popTensor();

private:
  bool usec2c = false;
  enum MessageType : int { SAIL_TENSOR, SAIL_BMIMAGE };
  std::unordered_map<int, std::shared_ptr<bm_handle_t>> handle_map;
  using RecvTypeBMImage_CC = std::tuple<bm_image, int, int>;

  void runLoop();
  std::thread t;
  bool isRunning = true;
  std::mutex mtx;

  std::queue<IPC::RecvTypeBMImage> image_queue;
  std::queue<IPC::RecvTypeTensor> tensor_queue;
  void pushImageQueue(IPC::RecvTypeBMImage &rhs);
  void pushTensorQueue(IPC::RecvTypeTensor &rhs);

  int send_fd;
  int final_fd;
  const std::string image_pipe;
  const std::string final_pipe;
  bool isSender = false;

  int IPC_QUEUE_MAX_LEN = 20;
  static constexpr int IPC_BMI_LEN = 128;
  static constexpr int IPC_TSR_LEN = 128;
  static constexpr int IPC_FINAL_LEN = 16;

  typedef union ipc_bmi_ {
    struct {
      int message_type;
      unsigned long long addr0; // 8 * 3 = 24 bytes
      unsigned long long addr1;
      unsigned long long addr2;
      int size0; // 4 * 14 = 56 bytes
      int size1;
      int size2;
      int format;
      int dtype;
      int width;
      int height;
      int dev_id;
      int stride0;
      int stride1;
      int stride2;
      int channel_id;
      int frame_id;
    } message;
    unsigned char data[IPC_BMI_LEN];
  } ipc_bmi;

  typedef union ipc_tensor_ {
    struct {
      int message_type;
      unsigned long long addr;
      int dtype;
      int size;
      int dev_id;
      int ndims;
      int shapes[6];
      int channel_id;
      int frame_id;
    } message;
    unsigned char data[IPC_TSR_LEN];
  } ipc_tensor;

  typedef union ipc_final_ {
    struct {
      int recv_len;
    } message;
    unsigned char data[IPC_FINAL_LEN];
  } ipc_final;

  RecvTypeBMImage_CC receiveBMImage(ipc_bmi &signal, int place_holder);
  void send(bm_image &img, int channel_id, int frame_id, bool need_to_free_);

  IPC::RecvTypeTensor receiveTensor(ipc_tensor &signal);
  IPC::RecvTypeBMImage receiveBMImage(ipc_bmi &signal);
};

IPC::IPC_CC::IPC_CC(bool isSender_, const std::string &image_pipe_,
                    const std::string &final_pipe_, bool usec2c, int queue_len)
    : isSender(isSender_), image_pipe(image_pipe_), final_pipe(final_pipe_),
      usec2c(usec2c), IPC_QUEUE_MAX_LEN(queue_len) {
  if (image_pipe == final_pipe) {
    SPDLOG_ERROR("[IPC] Image_pipe and final_pipe should not be same! Please "
                 "check your params");
    abort();
  }
  if ((mkfifo(image_pipe.c_str(), 0600) == -1) && errno != EEXIST) {
    printf("mkfifo failed\n");
    perror("reason");
  }
  if ((mkfifo(final_pipe.c_str(), 0600) == -1) && errno != EEXIST) {
    printf("mkfifo failed\n");
    perror("reason");
  }
  if (isSender) {
    send_fd = open(image_pipe.c_str(), O_WRONLY);
    if (send_fd == -1) {
      perror("open");
    }
    final_fd = open(final_pipe.c_str(), O_RDONLY);
    if (final_fd == -1) {
      perror("open");
    }
  } else {
    send_fd = open(image_pipe.c_str(), O_RDONLY);
    if (send_fd == -1) {
      perror("open");
    }
    final_fd = open(final_pipe.c_str(), O_WRONLY);
    if (final_fd == -1) {
      perror("open");
    }
    t = std::thread(&IPC_CC::runLoop, this);
  }
}

void IPC::IPC_CC::runLoop() {
  while (isRunning) {
    // Since you don't know what type to accept at first, try picking up the bmi
    // and checking the MessageType. If not bmi, memcpy to tensor signal
    ipc_bmi signal_bmi;
    int cnt = 0;
    while (cnt < IPC_BMI_LEN) {
      cnt += read(send_fd, signal_bmi.data + cnt, IPC_BMI_LEN - cnt);
    }
    if (signal_bmi.message.message_type == IPC_CC::MessageType::SAIL_BMIMAGE) {
      // The received is indeed bmi, operate directly
      IPC::RecvTypeBMImage res = receiveBMImage(signal_bmi);
      pushImageQueue(res);
    } else if (signal_bmi.message.message_type ==
               IPC_CC::MessageType::SAIL_TENSOR) {
      // Received tensor, forward type first
      ipc_tensor signal_tsr;
      memcpy((void *)(&signal_tsr), (void *)(&signal_bmi), IPC_BMI_LEN);
      IPC::RecvTypeTensor res = receiveTensor(signal_tsr);
      pushTensorQueue(res);
    }
  }
  return;
}

void IPC::IPC_CC::pushImageQueue(IPC::RecvTypeBMImage &rhs) {
  while (1) {
    std::unique_lock<std::mutex> lk(mtx);
    if (image_queue.size() < IPC_QUEUE_MAX_LEN) {
      image_queue.push(std::move(rhs));
      break;
    }
    lk.unlock();
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  return;
}

void IPC::IPC_CC::pushTensorQueue(IPC::RecvTypeTensor &rhs) {
  while (1) {
    std::unique_lock<std::mutex> lk(mtx);
    if (tensor_queue.size() < IPC_QUEUE_MAX_LEN) {
      tensor_queue.push(std::move(rhs));
      break;
    }
    lk.unlock();
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  return;
}

IPC::RecvTypeBMImage IPC::IPC_CC::popImage() {
  IPC::RecvTypeBMImage res;
  while (1) {
    std::unique_lock<std::mutex> lk(mtx);
    if (image_queue.size() <= 0) {
      lk.unlock();
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    } else {
      res = std::move(image_queue.front());
      image_queue.pop();
      lk.unlock();
      break;
    }
  }
  return res;
}

IPC::RecvTypeTensor IPC::IPC_CC::popTensor() {
  IPC::RecvTypeTensor res;
  while (1) {
    std::unique_lock<std::mutex> lk(mtx);
    if (tensor_queue.size() <= 0) {
      lk.unlock();
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    } else {
      res = std::move(tensor_queue.front());
      tensor_queue.pop();
      lk.unlock();
      break;
    }
  }
  return res;
}

IPC::IPC_CC::RecvTypeBMImage_CC IPC::IPC_CC::receiveBMImage(ipc_bmi &signal,
                                                            int place_holder) {
  if (signal.message.message_type != MessageType::SAIL_BMIMAGE) {
    SPDLOG_ERROR("Message Type SAIL_BMIMAGE is expected, but got a wrong type. "
                 "Please check your interface!");
    abort();
  }

  bm_device_mem_t mem[3];
  mem[0] = bm_mem_from_device(signal.message.addr0, signal.message.size0);
  mem[1] = bm_mem_from_device(signal.message.addr1, signal.message.size1);
  mem[2] = bm_mem_from_device(signal.message.addr2, signal.message.size2);

  int stride[3] = {signal.message.stride0, signal.message.stride1,
                   signal.message.stride2};

  int format = signal.message.format;
  int dtype = signal.message.dtype;

  int width = signal.message.width;
  int height = signal.message.height;

  int dev_id = signal.message.dev_id;

  int channel_id = signal.message.channel_id;
  int frame_id = signal.message.frame_id;

  bm_image img_recv;

  if (handle_map.find(dev_id) == handle_map.end()) {
    bm_handle_t h;
    bm_dev_request(&h, dev_id);
    handle_map[dev_id] = std::make_shared<bm_handle_t>(h);
    bm_image_create(h, height, width, (bm_image_format_ext)format,
                    (bm_image_data_format_ext)dtype, &img_recv, stride);
  } else {
    bm_image_create(*handle_map[dev_id], height, width,
                    (bm_image_format_ext)format,
                    (bm_image_data_format_ext)dtype, &img_recv, stride);
  }

  // Recipient needs d2d to construct new device memory
  int planes = bm_image_get_plane_num(img_recv);

  bm_device_mem_t mem_d2d[3];
  int bytes[3];
  bm_image_get_byte_size(img_recv, bytes);

  for (int i = 0; i < planes; ++i) {
    bm_malloc_device_byte_heap(*handle_map[dev_id], &mem_d2d[i], 1, bytes[i]);
    if (!usec2c)
      bm_memcpy_d2d_byte(*handle_map[dev_id], mem_d2d[i], 0, mem[i], 0,
                         bytes[i]);
    else
      bm_memcpy_c2c(*handle_map[dev_id], *handle_map[dev_id], mem[i],
                    mem_d2d[i], false);
  }

  bm_image_attach(img_recv, mem_d2d);
  // Send a message to the sender notifying the sender to release the original
  // device memory
  ipc_final final_signal;
  memset(final_signal.data, 0, sizeof(final_signal));
  final_signal.message.recv_len = IPC_BMI_LEN;
  write(final_fd, final_signal.data, IPC_FINAL_LEN);
  return std::make_tuple(img_recv, channel_id, frame_id);
}

void IPC::IPC_CC::send(bm_image &img, int channel_id, int frame_id,
                       bool need_to_free_) {
  bm_device_mem_t mem[3];
  bm_image_get_device_mem(img, mem);
  unsigned long long addr[3];
  unsigned int sz[3];
  int stride[3];
  bm_image_get_stride(img, stride);
  for (int i = 0; i < 3; ++i) {
    addr[i] = bm_mem_get_device_addr(mem[i]);
    sz[i] = bm_mem_get_device_size(mem[i]);
  }

  int planes = bm_image_get_plane_num(img);

  bm_handle_t handle_ = bm_image_get_handle(&img);
  int dev_id = bm_get_devid(handle_);

  ipc_bmi signal;
  memset(signal.data, 0, sizeof(signal));
  signal.message.message_type = MessageType::SAIL_BMIMAGE;

  signal.message.addr0 = addr[0];
  signal.message.addr1 = addr[1];
  signal.message.addr2 = addr[2];

  signal.message.size0 = sz[0];
  signal.message.size1 = sz[1];
  signal.message.size2 = sz[2];

  signal.message.stride0 = stride[0];
  signal.message.stride1 = stride[1];
  signal.message.stride2 = stride[2];

  signal.message.format = img.image_format;
  signal.message.dtype = img.data_type;

  signal.message.width = img.width;
  signal.message.height = img.height;

  signal.message.dev_id = dev_id;

  signal.message.channel_id = channel_id;
  signal.message.frame_id = frame_id;

  ssize_t sz2 = 0;
  {
    std::lock_guard<std::mutex> lk(mtx);
    sz2 = write(send_fd, signal.data, IPC_BMI_LEN);
  }

  ipc_final final_signal;
  int cnt = 0;
  while (cnt < IPC_FINAL_LEN) {
    cnt += read(final_fd, final_signal.data + cnt, IPC_FINAL_LEN - cnt);
  }

  return;
}

IPC::RecvTypeBMImage IPC::IPC_CC::receiveBMImage(ipc_bmi &signal) {
  if (isSender) {
    SPDLOG_ERROR(
        "This Object is Sender! Calling for receive method is invalid!");
    abort();
  }
  IPC_CC::RecvTypeBMImage_CC bmi = receiveBMImage(signal, 1);
  BMImage tmp(get<0>(bmi));
  tmp.set_ipc_flag(true);
  int channel_id = get<1>(bmi);
  int frame_id = get<2>(bmi);
  return std::make_tuple(std::move(tmp), channel_id, frame_id);
}

void IPC::IPC_CC::sendBMImage(BMImage &img, int channel_id, int frame_id) {
  if (!isSender) {
    SPDLOG_ERROR(
        "This Object is Receiver! Calling for send method is invalid!");
    abort();
  }
  send(img.data(), channel_id, frame_id, img.need_to_free());
}

IPC::IPC_CC::~IPC_CC() {
  close(send_fd);
  close(final_fd);
  isRunning = false;
  t.join();
  if (!isSender) {
    if (std::remove(image_pipe.c_str()) != 0) {
      perror("Error deleting file");
    }
    if (std::remove(final_pipe.c_str()) != 0) {
      perror("Error deleting file");
    }
  }
}

IPC::IPC(bool isSender_, const std::string &pipe_path,
         const std::string &final_path, bool usec2c, int queue_len) {
  impl_ = new IPC_CC(isSender_, pipe_path, final_path, usec2c, queue_len);
}

IPC::~IPC() { delete impl_; }

IPC::RecvTypeBMImage IPC::receiveBMImage() { return impl_->popImage(); }

void IPC::sendBMImage(BMImage &img, int channel_id, int frame_id) {
  return impl_->sendBMImage(img, channel_id, frame_id);
}

IPC::RecvTypeTensor IPC::IPC_CC::receiveTensor(ipc_tensor &signal) {
  if (isSender) {
    SPDLOG_ERROR(
        "This Object is Sender! Calling for receive method is invalid!");
    abort();
  }
  if (signal.message.message_type != MessageType::SAIL_TENSOR) {
    SPDLOG_ERROR("Message Type SAIL_TENSOR is expected, but got a wrong type. "
                 "Please check your interface!");
    abort();
  }
  bm_device_mem_t mem =
      bm_mem_from_device(signal.message.addr, signal.message.size);
  int ndims = signal.message.ndims;
  std::vector<int> shapes;
  for (int i = 0; i < ndims; ++i) {
    shapes.push_back(signal.message.shapes[i]);
  }
  bm_data_type_t dtype = (bm_data_type_t)signal.message.dtype;
  int dev_id = signal.message.dev_id;
  Handle h(dev_id);

  bm_device_mem_t mem_d2d;
  bm_malloc_device_byte_heap(h.data(), &mem_d2d, 0, signal.message.size);
  if (!usec2c)
    bm_memcpy_d2d_byte(h.data(), mem_d2d, 0, mem, 0, signal.message.size);
  else
    bm_memcpy_c2c(h.data(), h.data(), mem, mem_d2d, false);

  Tensor t(h, shapes, dtype, false, false);
  t.reset_dev_data(mem_d2d);
  t.own_dev_data() = true;

  ipc_final final_signal;
  memset(final_signal.data, 0, sizeof(final_signal));
  final_signal.message.recv_len = IPC_TSR_LEN;
  write(final_fd, final_signal.data, IPC_FINAL_LEN);

  return std::make_tuple(std::move(t), signal.message.channel_id,
                         signal.message.frame_id);
}

void IPC::IPC_CC::sendTensor(Tensor &t, int channel_id, int frame_id) {
  if (!isSender) {
    SPDLOG_ERROR(
        "This Object is Receiver! Calling for send method is invalid!");
    abort();
  }
  bm_device_mem_t mem = t.dev_data();
  int dev_id = t.get_handle().get_device_id();
  const std::vector<int> shapes = t.shape();

  ipc_tensor signal;
  memset(signal.data, 0, sizeof(signal));
  signal.message.message_type = MessageType::SAIL_TENSOR;

  if (shapes.size() > 6) {
    SPDLOG_ERROR("Now sendTensor only supports Tensor within 6 dims!");
    abort();
  }

  signal.message.ndims = shapes.size();
  for (int i = 0; i < shapes.size(); ++i) {
    signal.message.shapes[i] = shapes[i];
  }

  signal.message.channel_id = channel_id;
  signal.message.frame_id = frame_id;

  signal.message.dev_id = dev_id;

  signal.message.dtype = int(t.dtype());

  signal.message.addr = bm_mem_get_device_addr(mem);
  signal.message.size = bm_mem_get_device_size(mem);

  ssize_t sz2 = 0;
  {
    std::lock_guard<std::mutex> lk(mtx);
    sz2 = write(send_fd, signal.data, IPC_TSR_LEN);
  }

  ipc_final final_signal;
  int cnt = 0;
  while (cnt < IPC_FINAL_LEN) {
    cnt += read(final_fd, final_signal.data + cnt, IPC_FINAL_LEN - cnt);
  }
  return;
}

void IPC::sendTensor(Tensor &t, int channel_id, int frame_id) {
  return impl_->sendTensor(t, channel_id, frame_id);
}

IPC::RecvTypeTensor IPC::receiveTensor() { return impl_->popTensor(); }

} // namespace sail
