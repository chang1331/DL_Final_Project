import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import cv2
import imutils
import time

try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
except NameError:
    current_file_dir = os.getcwd().replace('\\', '/')


def get_model_instance(num_classes):
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


def object_detection_api(model, img_array, threshold=0.5, rect_th=1, text_size=0.5, text_th=1):
    boxes, pred_cls, pred_score = get_prediction(model, img_array, threshold)  # Get predictions
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    for i in range(len(boxes)):
        # Draw Rectangle with the coordinates
        cv2.rectangle(img_array, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        # Write the prediction class
        cv2.putText(img_array, str(pred_score[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                    thickness=text_th)
    return img_array


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance(num_classes=2)
model.load_state_dict(torch.load(current_file_dir + '/model.pt'))
model = model.eval()
model.to(device)

video820Path = current_file_dir + '/820_1.mp4'
vs = cv2.VideoCapture(video820Path)
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

frameIndex = 1
writer = None
while True:
    # print(frameIndex)
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    start = time.time()
    frame = object_detection_api(model, frame, threshold=0.8)
    end = time.time()

    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(current_file_dir + '/result_video.avi', fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write the output frame to disk
    writer.write(frame)
    frameIndex += 1

print("[INFO] cleaning up...")
writer.release()
vs.release()
