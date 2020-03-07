import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from bike_helmet_detection_utils import infer_image, show_image
from fastai.vision import*
import PIL

confidence=0.5
threshold=0.3
config = './yolov3-coco/yolov3.cfg'
weights = './yolov3-coco/yolov3.weights'
labels = './yolov3-coco/coco-labels'
labels = open(labels).read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


video_path = "D:\\Programs\\test_data\\shrey_1.mp4"
cap = cv.VideoCapture(video_path)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


video_path = "D:\\Programs\\test_data\\shrey_1.mp4"

if video_path:
    # Read the video
    try:
        vid = cv.VideoCapture(video_path)
        height, width = None, None
        writer = None
    except:
        raise 'Video cannot be loaded!\n\
                           Please check the path provided!'

    finally:
        while True:
            grabbed, frame = vid.read()

            # Checking if the complete video is read
            if not grabbed:
                break

            if width is None or height is None:
                height, width = frame.shape[:2]

            frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, confidence,threshold)
            cv.imshow('vid', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv.destroyAllWindows()

