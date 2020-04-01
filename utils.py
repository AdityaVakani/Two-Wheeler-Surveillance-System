import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from fastai.vision import*
from numberplate import numberplate_detect
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
    w2=b2[2]
    h2=b2[3]
    if (x1>x2+w2 or x2>x1+w1) :
        return False
    if (y2>y1+h1 or y1>y2+h2) :
        return False
    return True



def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        personbike_boxes = []
        head_boxes = []
        pb_overlap = []
        h_overlap = []
        helmet_array=[]
        for i in idxs.flatten():
            j=len(idxs.flatten())-1
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            if classids[i] == 0:
                personbike_boxes.append((x,y,w,h))
                cv.rectangle(img, (x, y), (x + w, y + h), [255,0,0], 1)
                text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
                #cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if classids[i] ==1:
                if h/w>0.8 and h/w<1.4:
                    cv.rectangle(img, (x, y), (x + w, y + h), [0,0,255], 1)
                    #text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
                    head_roi=img[(y):((y + h)),(x):(x+w)]
                    helmet_flag = helmet_check(head_roi)
                    helmet_array.append(str(helmet_flag))
                    head_boxes.append((x, y, w, h))
                    cv.putText(img, " "+str(helmet_flag), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,0], 1)
        #print(len(personbike_boxes),len(head_boxes))
        for pb_box in range(len(personbike_boxes)):
            for h_box in range(len(head_boxes)):
                x1=personbike_boxes[pb_box][0]
                w1=personbike_boxes[pb_box][2]
                x2=head_boxes[h_box][0]
                w2=head_boxes[h_box][2]
                centre_distance = abs((2*x1+w1)/ 2 - (2*x2+w2) / 2)
                # centre_distance=abs((2*personbike_boxes[pb_box][0]+personbike_boxes[pb_box][2])/2 - (2*head_boxes[h_box][0]+ head_boxes[h_box][2])/2)

                if overlap(personbike_boxes[pb_box], head_boxes[h_box]) and centre_distance<40:
                    print("overlap:"+str(centre_distance)+ "  " +str(helmet_array[h_box]))


                if overlap(personbike_boxes[pb_box], head_boxes[h_box]) and centre_distance < 40 and helmet_array[h_box]=="nohelmet":
                    pb_overlap.append(personbike_boxes[pb_box])
                    h_overlap.append(head_boxes[h_box])
                    print("no helmet added")
        for k in(pb_overlap):
            print(k)
            x=k[0]
            y=k[1]
            w=k[2]
            h=k[3]
            test_img=img[(y):((y + h)), (x):(x + w)]
            test_img=numberplate_detect(test_img)
            cv.imshow("bike",test_img)
            cv.waitKey(0)

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
            if confidence > tconf and (classid==0 or classid==1):
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
