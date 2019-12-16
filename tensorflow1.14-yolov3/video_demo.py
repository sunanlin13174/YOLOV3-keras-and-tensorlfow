#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import os
import time
import numpy as np
import core.utils as utils
import tensorflow as  tf
from timeit import default_timer as timer
from PIL import Image
graph = tf.Graph()
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]

pb_file         = "/home/sal/h5_to_mlmodel/trans_model/demo.pb"
video_path      = "/home/sal/Videos/11.mp4"
# video_path      = 0
num_classes     = 4
input_size      = 608

return_tensors = utils.read_pb_return_tensors(graph, pb_file,return_elements=return_elements)

vid = cv2.VideoCapture(video_path)
# value, pic = vid.read()
# pic = cv2.resize(pic,(1280,720))
# cv2.imshow('image', pic)
# coor = utils.obtain_coordation(pic)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#show fps in real-time
accum_time = 0
curr_fps = 0
fps = "FPS: ??"

i =0
b= 0
frames = []
prev_time = timer()
with tf.Session(graph =graph,config=config) as sess:

    while True:
        i+=1
        return_value, frame = vid.read()
        # if (i+1)%2==0:
        #     frames.append(frame)
        # if i%2==0:
        # utils.draw_four_lines(frame,coor)
        if return_value:
            # img = utils.self_absdiff(frames[b],frame,5)
            # img = utils.diff_img2mask(img)
            # b+=1
            # p_frame0 = np.multiply(img,frame[:,:,0])
            # p_frame1 = np.multiply(img,frame[:,:,1])
            # p_frame2 = np.multiply(img,frame[:,:,2])
            # frame = cv2.merge((p_frame0,p_frame1,p_frame2))
            frame=cv2.resize(frame,(704,480))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        #prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)

        #curr_time = time.time()
        #exec_time = curr_time - prev_time
        result = np.asarray(image)
        #info = "time: %.2f ms" %(1000*exec_time)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time +exec_time
        curr_fps = curr_fps +1
        if accum_time>1:
            accum_time = accum_time -1
            fps = "FPS:"+str(curr_fps)
            curr_fps = 0
        # if i%2==0:
        # result = cv2.resize(result, (1280, 720))
        cv2.putText(result,text=fps,org=(100,100),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2.0,color=(0,0,255),thickness=2)
        cv2.imshow("result", result)

        # cv2.putText(frame, text=fps, org=(422, 138), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0,
        #             color=(0, 0, 255), thickness=2)
        # cv2.imshow("result", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break




