import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from fastai.vision import*
import PIL

path=Path('.')
learn = load_learner(path)

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def helmet_check(img):
    img = Image(pil2tensor(img, np.float32).div_(255))
    pred_class, pred_idx, outputs = learn.predict(img)
    return pred_class

def overlap(b1,b2):
    x1=b1[0]
    y1=b1[1]
    w1=b1[2]
    h1=b1[3]
    x2=b2[0]
    y2=b2[1]
    h2=b2[2]
    w2=b2[3]
    if (x1>x2+w2 or x2>x1+w1) :
        return False
    if (y2>y1+h1 or y1>y2+h2) :
        return False
    return True




def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        bike_boxes = []
        people_boxes = []
        b_overlap = []
        p_overlap = []
        for i in idxs.flatten():
            j=len(idxs.flatten())-1
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            if classids[i] == 3:
                bike_boxes.append((x,y,w,h))
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
                cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if classids[i] == 0:
                people_boxes.append((x,y,w,h))
                cv.rectangle(img, (x, y), (x + w, y + h), color, 1)
                head_area=(w*(((y+h)//2)-(y-10)))
                text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
                cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            for b_box in range(len(bike_boxes)):
                for p_box in range(len(people_boxes)):
                    if overlap(bike_boxes[b_box],people_boxes[p_box]):
                        b_overlap.append(bike_boxes[b_box])
                        p_overlap.append(people_boxes[p_box])
                        #print("overlap")
            for k in p_overlap:
                x = k[0]
                y = k[1]
                w = k[2]
                h = k[3]
                if (((y+h)//2)-(y-10))>100 and (((y+h)//2)-(y-10))<300 :
                    #print((((y+h)//2)-(y-10)))
                    head_roi=img[(y-10):((y + h) //2),(x):(x+w)]
                    #helmet_flag=helmet_check(head_roi)
                    #print(helmet_flag)

                    cv.rectangle(img, (x, y - 10), (x + w, (y + h) // 2), [0, 0, 0], 2)
                    #cv.putText(img,helmet_flag , (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img





def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            # print (detection)
            # a = input('GO!')

            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf and (classid==0 or classid==3):
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids


def infer_image(net, layer_names, height, width, img, colors, labels, confidence,threshold,
                boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()


        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, confidence)

        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'

    # Draw labels and boxes on the image
    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs
