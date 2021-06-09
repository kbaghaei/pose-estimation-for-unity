# Copyright (c) 2021 Kourosh T. Baghaei
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from detectron2.config import get_cfg
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


class Pose2DModel:
    """
    Pose2DModel : loads and prepares a model of type 'torch.nn.Module'.
    The model is implementation of detectron2's keypoints model.
    Inputs: a tensor of type torch.float32 and shape: (3, H, W)
    Output: A list of shape: [{'instances' : <Instances> }]
    """
    def __init__(self, path_to_model_files='model_files', confidence_threshold=0.6, input_height=720, input_width=1280):
        self.__path_to_model_files = path_to_model_files
        self.__device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.input_height = input_height
        self.input_width = input_width
        self.confidence_threshold = confidence_threshold
        self.__config()

    def __config(self):
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(self.__path_to_model_files,'keypoint_rcnn_R_50_FPN_3x.yaml'))
        cfg.merge_from_list(['MODEL.WEIGHTS', os.path.join(self.__path_to_model_files, 'model_final_a6e10b.pkl')])
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_threshold
        cfg.freeze()

        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        model.to(self.__device)
        self.__model = model
        assert isinstance(self.__model, torch.nn.Module), '(x) Pose2D Error: Model has not been loaded correctly.'

    def __call__(self, image):
        """
        Takes in an image and returns the COCO key points of humans detected in the image.

        Example:
        ::
            preds = net(img)
            kp = preds[0]['instances'].get_fields()['pred_keypoints']
            print(kp.shape) # [1,17,3]
            # the first dimension of kp is the number of detected humans

        :param image: An UNSCALED tensor of type torch.float32 and shape: (3, H, W).
        :return: A list of shape: [{'instances' : <Instances> }]
        """
        x = [{'image': image, 'height': self.input_height, 'width': self.input_width }]
        return self.__model(x)
