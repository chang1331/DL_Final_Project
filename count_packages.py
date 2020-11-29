import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import model_urls

if not os.path.exists('sort_output'):
    os.makedirs('sort_output')
files = glob.glob('sort_output/*.png')
for f in files:
   os.remove(f)

from sort import *
tracker = Sort()
memory = {}
line = [(10, 130), (180, 130)]
counter = 0

try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
except NameError:
    current_file_dir = os.getcwd().replace('\\', '/')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def get_model_instance(num_classes):
    model_urls['fasterrcnn_resnet50_fpn_coco'] = model_urls['fasterrcnn_resnet50_fpn_coco'].replace('https://', 'http://')
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_prediction(model, img_array, threshold):
    # img_array is numpy array of shape (H,W,3)
    img_tensor = torchvision.transforms.ToTensor()(img_array)
    img_tensor = torch.unsqueeze(img_tensor, 0).to(device)
    # print(img_tensor.shape)
    with torch.no_grad():
        pred = model(img_tensor)  # Pass the image to the model
    pred_class = ['package' for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    # Get list of index with score greater than threshold.
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_t) > 0:
        pred_boxes = list(map(lambda i: pred_boxes[i], pred_t))
        pred_class = list(map(lambda i: pred_class[i], pred_t))
        pred_score = list(map(lambda i: round(pred_score[i], 2), pred_t))
    else:
        pred_boxes = []
        pred_class = []
        pred_score = []
    return pred_boxes, pred_class, pred_score

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance(num_classes=2)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(current_file_dir + '/model.pt'))
else:
    model.load_state_dict(torch.load(current_file_dir + '/model.pt', map_location=torch.device('cpu')))
model = model.eval()
model.to(device)

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    #Pass frame through our network. 
    start = time.time()
    pred_boxes, pred_class, pred_score = get_prediction(model, frame, 0.5)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    numBoxes = len(pred_score)
    # loop over each of the detections
    for i in range(numBoxes):
        # extract the class ID and confidence (i.e., probability)
        # of the current object detection
        con = pred_score[i]
        b = pred_boxes[i]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if con > args["confidence"]:
            (x0, y0) = b[0] #upper left corner
            (x1, y1) = b[1] #bottom right corner

            # update our list of bounding box coordinates,
            # confidences, and class IDs
            boxes.append([int(x0), int(y0), int(x1-x0), int(y1-y0)])
            confidences.append(float(con))
            classIDs.append(1)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3)

                if intersect(p0, p1, line[0], line[1]):
                    counter += 1

            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    # draw line
    cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

    # draw counter
    cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 5)
    # counter += 1

    # saves image file
    cv2.imwrite("sort_output/frame-{}.png".format(frameIndex), frame)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

    if frameIndex >= 1000:
        print("[INFO] cleaning up...")
        writer.release()
        vs.release()
        exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()