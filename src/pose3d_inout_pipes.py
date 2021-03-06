# Copyright (c) 2021 Kourosh T. Baghaei
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections

import torch


class Pose3DInputPipe:
    def __init__(self, queue_size=243):
        self.__qsize = queue_size
        self.__queue = collections.deque(maxlen=queue_size)

    def put(self, x):
        self.__queue.append(x)

    def get(self):
        return torch.as_tensor(self.__queue)

    def __len__(self):
        return self.__qsize

    def __sizeof__(self):
        return len(self.__queue)


class Pose3DOutputPipe:
    def __init__(self):
        pass

    def extract(self, x):
        return x[-1,:,:]
