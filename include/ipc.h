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

/** @file     ipc.h
 *  @brief    Header file of IPC
 *  @author   yizhou.xu
 *  @version  3.8.0
 *  @date     2024-07-31
 */

#pragma once
#include "cvwrapper.h"
#include <string>

namespace sail
{
  class DECL_EXPORT IPC
  {
  public:
    IPC() = delete;
    /**
     * @brief Construct a new IPC object
     *
     * @param isSender_ This object is Sender of Receiver
     * @param pipe_path The name of the pipe used to transmit data from the sender to the receiver.
     * @param final_path The name of the pipe used to transmit signal from the receiver to the sender.
     * @param usec2c The flag of using c2c or d2d while copy device mem.
     * @param queue_len The length of receiver queue.
     */
    IPC(bool isSender_, const std::string &pipe_path, const std::string &final_path, bool usec2c = false, int queue_len = 20);
    ~IPC();

    using RecvTypeBMImage = std::tuple<BMImage, int, int>;
    using RecvTypeTensor = std::tuple<Tensor, int, int>;

    /**
     * @brief send a BMImage to pipe with channel_id & frame_id, return after the img is received.
     *
     * @param img BMImage
     * @param channel_id channel_id of the BMImage
     * @param frame_id frame_id of the BMImage
     */
    void sendBMImage(BMImage &img, int channel_id, int frame_id);
    /**
     * @brief receive a BMImage from pipe with channel_id & frame_id.
     *
     * @return RecvTypeBMImage a tuple of BMImage, channel_id and frame_id
     */
    RecvTypeBMImage receiveBMImage();

    /**
     * @brief send a Tensor to pipe with channel_id & frame_id, return after the tensor is received.
     *
     * @param t Tensor
     * @param channel_id channel_id of the Tensor
     * @param frame_id frame_id of the Tensor
     */
    void sendTensor(Tensor &t, int channel_id, int frame_id);
    /**
     * @brief receive a Tensor from pipe with channel_id & frame_id
     *
     * @return RecvTypeTensor a tuple of Tensor, channel_id and frame_id
     */
    RecvTypeTensor receiveTensor();

  private:
    class IPC_CC;
    IPC_CC *impl_;
  };
} // namespace sail