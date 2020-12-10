import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import re
import csv

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import model_urls

from sort import *

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="input video path or all for all videos")
ap.add_argument("-b", "--batch", type=int, default=4,
	help="batch size")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="min confidence for detection")
ap.add_argument("-l", "--line", type=int, default=1,
	help="virtual line: 0 for top diagonal, 1 for middle horizontal, 2 for bottom diagonal")
ap.add_argument("-a", "--age", type=int, default=1,
	help="max age for keeping trackers")

args = vars(ap.parse_args())

lines = [[(0, 0), (160, 120)],
        [(0, 120), (160, 120)],
        [(0, 120), (160, 230)]]
line = lines[args["line"]]

if not os.path.exists('sort_output'):
    os.makedirs('sort_output')
try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
except NameError:
    current_file_dir = os.getcwd().replace('\\', '/')

def intersect(A,B,C,D):
	return (int_helper(A,C,D) != int_helper(B,C,D)) and (int_helper(A,B,C) != int_helper(A,B,D))

def int_helper(A,B,C):
	return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def get_model_instance(num_classes):
    model_urls['fasterrcnn_resnet50_fpn_coco'] = model_urls['fasterrcnn_resnet50_fpn_coco'].replace('https://', 'http://')
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_batch_prediction(model, batch_frames, threshold):
    batch_pred_boxes = []
    batch_pred_class = []
    batch_pred_score = []
    batch_img_tensors = []
    for frame in batch_frames:
        img_tensor = torchvision.transforms.ToTensor()(frame).to(device)
        # img_tensor = torch.unsqueeze(img_tensor, 0).to(device)
        batch_img_tensors.append(img_tensor)
    with torch.no_grad():
        pred = model(batch_img_tensors)
    for output in pred:
        pred_boxes, pred_class, pred_score = unpack_prediction(output, threshold)
        batch_pred_boxes.append(pred_boxes)
        batch_pred_class.append(pred_class)
        batch_pred_score.append(pred_score)
    return batch_pred_boxes, batch_pred_class, batch_pred_score


def unpack_prediction(pred, threshold):
    pred_class = ['package' for i in list(pred['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred['scores'].detach().cpu().numpy())
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

def update_counter(b, ids, prev_bb, counter, curr_frame):
    if len(b) <= 0:
        return counter, curr_frame
    for i in range(len(b)):
        width = int(b[i][2])
        height = int(b[i][3])
        if ids[i] in prev_bb:
            prev = prev_bb[ids[i]]
            x1 = int(b[i][0])
            y1 = int(b[i][1])
            x2 = int(prev[0])
            y2 = int(prev[1])
            width2 = int(prev[2])
            height2 = int(prev[3])
            com_line = [((int(x1 + (width - x1) / 2), int(y1 + (height - y1) / 2))),
                        ((int(x2 + (width2 - x2) / 2), int(y2 + (height2 - y2) / 2)))]
            cv2.line(curr_frame, com_line[0], com_line[1], (255, 0, 255), 3)
            #Check if virtual line and com line intersects
            #Where count increses
            if intersect(com_line[0], com_line[1], line[0], line[1]):
                counter += 1
        cv2.rectangle(curr_frame, (int(b[i][0]), int(b[i][1])), (width, height), (255, 0, 255), 2)
    return counter, curr_frame

def do_prediction(model, input_path):
    memory = {}
    tracker = Sort(max_age=args["age"])
    counter = 0
    vs = cv2.VideoCapture(input_path)
    writer = None
    (W, H) = (None, None)

    frameIndex = 0

    # count total frames
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video ".format(total) + input_path)
    except:
        print("[INFO] could not determine # of frames in video")
        total = -1

    while True:
        frame_batch = []
        for i in range(args["batch"]):
            (grabbed, frame) = vs.read()
            if not grabbed:
                writer.release()
                vs.release()
                print("Final count for ", input_path, ": ", counter)
                return counter
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            frame_batch.append(frame)

        #Pass batch of frames through our network. 
        start = time.time()
        pred_boxes, pred_class, pred_score = get_batch_prediction(model, frame_batch, 0.5)
        end = time.time()

        for j in range(len(frame_batch)):
            confidences = []
            classIDs = []
            curr_frame = frame_batch[j]
            boxes = []
            numBoxes = len(pred_score[j])
            for i in range(numBoxes):
                con = pred_score[j][i]
                b = pred_boxes[j][i]

                if con > args["confidence"]:
                    (x0, y0) = b[0] #upper left corner
                    (x1, y1) = b[1] #bottom right corner
                    confidences.append(float(con))
                    #calculate left x, left y, width, height
                    boxes.append([int(x0), int(y0), int(x1-x0), int(y1-y0)])
                    # all predictions will be 1
                    classIDs.append(1)

            suppression = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)#0.4, 0.2

            dets = []
            if len(suppression) > 0:
                suppression = suppression.flatten()
                for i in suppression:
                    #Keep box that passes suprresion theshhold
                    dets.append([boxes[i][0], boxes[i][1], boxes[i][0]+boxes[i][2], boxes[i][1]+boxes[i][3], confidences[i]])
            dets = np.asarray(dets)
            tracks = tracker.update(dets)

            b = []
            ids = []
            prev_bb = memory.copy()
            memory = {}

            for k in range(len(tracks)):
                ids.append(int(tracks[k][4]))
                extracted_box = [tracks[k][0], tracks[k][1], tracks[k][2], tracks[k][3]]
                b.append(extracted_box)
                memory[ids[-1]] = b[-1]

            #Check if packages cross virtual line and update counter
            counter, curr_frame = update_counter(b, ids, prev_bb, counter, curr_frame)
            #write line to frame
            cv2.line(curr_frame, line[0], line[1], (255, 0, 255), 5)
            #write counter to frame
            cv2.putText(curr_frame, str(counter), (350,250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 5)

            if writer == None:
                fps = vs.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                output_name = "out_" + input_path.split("/")[-1]
                writer = cv2.VideoWriter("sort_output/" + output_name, fourcc, fps,
                    (curr_frame.shape[1], curr_frame.shape[0]), True)
                print("single batch: ", (end-start), " total estimate: ", (end - start)*total / args["batch"])

            writer.write(curr_frame)
            frameIndex += 1

            # Uncomment this to return at specific frame
            # if frameIndex >= 240:
            #    writer.release()
            #    vs.release()
            #    print("Final count for ", input_path, ": ", counter)
            #    return counter
    writer.release()
    vs.release()
    print("Final count for ", input_path, ": ", counter)
    return counter

#Get model and load weights
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance(num_classes=2)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(current_file_dir + '/full_model.pt'))
else:
    model.load_state_dict(torch.load(current_file_dir + '/full_model.pt', map_location=torch.device('cpu')))
model = model.eval()
model.to(device)

input_filepaths = glob.glob('videos/**/*.mp4', recursive=True)
predicted_files = []

#Creates and inserts header of counter csv file if it does not exist
if not os.path.exists('predicted_count.csv'):
    with open('predicted_count.csv', mode='w') as pred_csv:
        csv_writer = csv.writer(pred_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["date", "video", 'count'])

#Load in file names already predicted from predicted_count.csv
with open('predicted_count.csv', mode='r') as pred_csv:
    csv_reader = csv.reader(pred_csv, delimiter=',')
    read_header = False
    for row in csv_reader:
        if not read_header:
            read_header = True
            continue
        if len(row) == 3:
            predicted_files.append(row[0] + "_" + row[1] + ".mp4")

#Run package count prediction on input files
with open('predicted_count.csv', mode='a') as pred_csv:
    csv_writer = csv.writer(pred_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for filepath in input_filepaths:
        video_name = filepath.split("/")[-1]
        if args["input"] != "all" and args["input"] != video_name:
            continue
        video_done = False
        for f in predicted_files:
            if(f == video_name):
                video_done = True
                break
        if(not video_done):
            count = do_prediction(model, filepath)
            video_name_split = re.split('[_.]', video_name)
            csv_writer.writerow([video_name_split[0], video_name_split[1], count])
            pred_csv.flush()