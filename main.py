# Copyright (c) 2021 Kourosh T. Baghaei
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from src.pose2d_estimation import Pose2DModel
from src.pose3d_estimation import Pose3DExtractor
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.input_pipeline import InputPipeLine
import cv2
import collections
from src.pose3d_inout_pipes import Pose3DInputPipe, Pose3DOutputPipe

import json
import sys
from src.network import ServerSample, UnityPortal
import time
import numpy as np


def main_as_ext_proc():
    input_height = 144
    input_width = 256
    meow = Pose2DModel(input_height=input_height, input_width=input_width)
    ipl = InputPipeLine(144, 256)
    ipl.start()
    frame_idx_in_seq = 0
    for frame in ipl.frames():
        preds = meow(frame)
        if len(preds) > 0:
            kp = preds[0]['instances'].get_fields()['pred_keypoints']
            if kp.shape[0] > 0:
                frame_idx_in_seq = frame_idx_in_seq + 1
                kp = kp.cpu()[0]
                kp = np.transpose(kp, [1, 0])
                kp[0] = 1.0 - (kp[0] / input_width)
                kp[1] = (kp[1] / input_height)
                kp = kp.tolist()
                jdat = str(frame_idx_in_seq) + ',' + ','.join([str(n) for n in kp[0]]) + ',' + ','.join([str(n) for n in kp[1]])
                print(jdat)
                sys.stdout.flush()

def main_send_to_unity_via_udp():
    up = UnityPortal(port_number=64695)
    input_height=144
    input_width=256
    meow = Pose2DModel(input_height=input_height,input_width=input_width)
    ipl = InputPipeLine(144,256)
    ipl.start()
    frame_idx_in_seq = 0
    for frame in ipl.frames():
        preds = meow(frame)
        if len(preds) > 0:
            kp = preds[0]['instances'].get_fields()['pred_keypoints']
            if kp.shape[0] > 0:
                frame_idx_in_seq = frame_idx_in_seq + 1
                kp = kp.cpu()[0]
                kp = np.transpose(kp, [1, 0])
                kp[0] = 1.0 - (kp[0] / input_width)
                kp[1] = (kp[1] / input_height)
                kp = kp.tolist()
                jdat = json.dumps(
                    {'x_poses': kp[0], 'y_poses': kp[1], 'frame_in_seq' : frame_idx_in_seq})
                jstr = str.encode(jdat)
                up.send(jstr)

        if cv2.waitKey(1) == 27:
            ipl.stop()
            break
    print('K!')

def main_network_portal():
    if 'server' in sys.argv:
        s = ServerSample()
        s.start()
    elif 'client' in sys.argv:
        up = UnityPortal()
        for i in range(10):
            saz = 1 # 120
            a = np.arange(saz * 17 * 2).reshape(saz, 2, 17).tolist()
            jdat = json.dumps({'x_poses' : a[0][0],'y_poses': a[0][1]})
            jstr = str.encode(jdat)
            up.send(jstr)
            time.sleep(0.2)

def main_pose3D_needs_test():
    pos3d = Pose3DExtractor('model_files/pretrained_h36m_detectron_coco.bin')
    keypoints = np.load('data_2d_custom_ma-vid.npz', allow_pickle=True)
    kp_arr = keypoints['positions_2d'].item()['cyrus_640.mp4']['custom'][0]
    pype = Pose3DInputPipe()
    outpype = Pose3DOutputPipe()
    for i in range(kp_arr.shape[0]):
        pype.put(kp_arr[i])
        res = pos3d(pype.get())
        res0 = outpype.extract(res)

        print(res0.shape)
        if i > 300:
            break



def main_pose2d():
    meow = Pose2DModel(input_height=144,input_width=256)
    ipl = InputPipeLine(144,256)
    ipl.start()
    for frame in ipl.frames():
        cv2.namedWindow('miaw', cv2.WINDOW_NORMAL)
        preds = meow(frame)
        img = np.zeros((144,256,3),dtype=np.uint8)
        if len(preds) > 0:
            kp = preds[0]['instances'].get_fields()['pred_keypoints']
            if kp.shape[0] > 0:
                kp = kp.cpu()
                for kp_idx in range(17):
                    kpx = int(kp[0, kp_idx, 0].item())
                    kpy = int(kp[0, kp_idx, 1].item())
                    img[kpy, kpx, 0] = 255
                    img[kpy, kpx, 1] = 255

                print('{:2.3f} {:2.3f}'.format(kp[0,9,0] - kp[0,10,0], kp[0,9,1] - kp[0,10,1]))
        cv2.imshow('miaw', img)
        if cv2.waitKey(1) == 27:
            ipl.stop()
            break
    cv2.destroyAllWindows()
    print('K!')


if __name__ == '__main__':
    main_as_ext_proc()