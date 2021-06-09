# Author : Kourosh T. Baghaei
# June 9, 2021

import numpy as np
import torch
import cv2


class InputPipeLine:
    """
    InputPipeLine : this class starts capturing video from camera and resizes the frames
    to 'target_height' and 'target_width'. Then, returns the resulting image as a torch.Tensor.
    """
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width
        self.__cam = None
        self.__cam_running = False

    def start(self):
        """
        Start capturing video from the first available camera.
        """
        self.__cam = cv2.VideoCapture(0)
        self.__cam_running = True

    def stop(self):
        """
        Stop capturing video from camera device. And release the camera to OS.
        """
        if self.__cam is None:
            raise RuntimeError('Camera is not started yet! So, it is not defined.')
        self.__cam_running = False
        self.__cam.release()

    def frames(self, as_tensor=True):
        """
        This function returns a generator that iterates over images of any available camera.
        Resizes frames to target sizes and converts to torch.Tensor if as_tensor is True.
        If camera is not started, it will raise RuntimeError.
        :param as_tensor: If True, this generator yields a tensor of shape: (3, H, W),
        Otherwise, it yields a numpy array of shape: (3, H, W).
        :return:  torch.Tensor, or a Numpy array.
        """
        if self.__cam is None:
            raise RuntimeError('Camera is not started yet! So, it is not defined.')
        cam = self.__cam
        while cam.isOpened():
            if not self.__cam_running:
                break
            success, frame = cam.read()
            if success:
                resized = cv2.resize(frame,(self.target_width, self.target_height))
                img_out = np.transpose(resized, [2, 0, 1])
                if as_tensor:
                    img_out = torch.as_tensor(img_out)
                yield img_out
            else:
                break