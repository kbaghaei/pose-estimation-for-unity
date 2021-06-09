# Author : Kourosh T. Baghaei
# June 9, 2021


import numpy as np
import torch
from .videopose.model import *


class Pose3DExtractor:
    def __init__(self, path_to_checkpoint=None, use_cuda=True):
        if path_to_checkpoint is None:
            raise TypeError('Set the path to the checkpoint! It cannot be None.')
        # Model params specific to this project:
        filter_widths = [3, 3, 3, 3, 3]
        dropout = 0.25
        channels = 1024
        keypoints_count = 17
        output_joints = 17
        input_features = 2

        self.__model = TemporalModel(keypoints_count, input_features, output_joints,
            filter_widths=filter_widths, causal=True, dropout=dropout, channels=channels, dense=False)

        checkpoint = torch.load(path_to_checkpoint, map_location=lambda storage, loc: storage)
        self.__model.load_state_dict(checkpoint['model_pos'])
        self.__use_cuda = use_cuda
        if use_cuda:
            self.__model = self.__model.cuda()
        self.__model.eval()
        self.__causal_shift = 0

        receptive_field = self.__model.receptive_field()
        print('INFO: Receptive field: {} frames'.format(receptive_field))
        self.__pad = (receptive_field - 1) // 2  # Padding on each side

    def __call__(self, keypoints_seq):
        batch_2d = np.expand_dims(np.pad(keypoints_seq,
                                         ((self.__pad + self.__causal_shift, self.__pad - self.__causal_shift), (0, 0), (0, 0)),
                                         'edge'), axis=0)
        if not isinstance(batch_2d, torch.Tensor):
            batch_2d = torch.from_numpy(batch_2d.astype('float32'))
        if self.__use_cuda:
            batch_2d = batch_2d.cuda()
        with torch.no_grad():
            pred3d = self.__model(batch_2d)
            return pred3d.squeeze(0).cpu().numpy()
