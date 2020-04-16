import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from utils import infer_image, show_image,crop_img
from fastai.vision import*
import PIL

confidence=0.5
threshold=0.3
config = './yolov3-coco/yolov3-personbikehead.cfg'
weights = './yolov3-coco/yolov3-personbikehead.weights'
labels = './yolov3-coco/pbh-labels.txt'
labels = open(labels).read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


video_path = "D:\\Programs\\Final Project\\test_data\\test_1.mp4"
cap = cv.VideoCapture(video_path)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]




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
            try:
                frame1,bike_np_roi, _, _, _, _ = infer_image(net, layer_names, height, width, frame.copy(), colors, labels, confidence,threshold)
            except:
                print("Skipping.................................................")
                continue
            if bike_np_roi != []:
                for k in bike_np_roi:
                    print(f'Bike co-ordinates {k[0]} , number plate co-ordinates {k[1]}')
                    bike=k[0]
                    bike_img = crop_img(frame.copy(), bike)
                    np=k[1]
                    if np!=():
                        np_img = crop_img(frame.copy(), np)
                        cv.imshow("numberplate", np_img)
                        cv.waitKey(5)
                    #cv.imshow("bike", bike_img)
                    #cv.waitKey(5)



            cv.imshow('vid', frame1)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv.destroyAllWindows()

