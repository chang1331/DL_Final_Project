import torchvision.transforms.functional as TF
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import pandas as pd
import numpy as np
import math
import sys


class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            height, width = img.shape[-2:]
            img = img.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return img, target


class ToTensor:
    def __call__(self, img, target):
        img = TF.to_tensor(img)
        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for transform in self.transforms:
            img, target = transform(img, target)
        return img, target


def get_transform(train):
    # convert PIL image to tensor
    transforms = [ToTensor()]
    if train:
        # randomly flip the image and bbox
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


class PackageDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, label_dir, transform=None):
        self.frame_dir = frame_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_dir + 'training_labels.csv',
                                     names=['x1', 'y1', 'x2', 'y2', 'frame_name', 'label'])
        self.frames_list = list(sorted(os.listdir(frame_dir)))

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frame = Image.open(self.frame_dir + self.frames_list[idx])

        frame_name = self.frames_list[idx][:self.frames_list[idx].find('.')]
        label_df = self.labels_df[self.labels_df['frame_name'] == frame_name]

        assert np.unique(label_df['label'].values).size == 1, 'same image should not have different labels'

        class_label = label_df['label'].values[0]  # either 0 (background) or 1 (package)
        num_objs = 0 if class_label == 0 else label_df.shape[0]  # number of objects in this frame

        if num_objs > 0:
            # this is a positive sample (with objects and corresponding bounding boxes)
            boxes = label_df[['x1', 'y1', 'x2', 'y2']].values.tolist()
            boxes = torch.as_tensor(boxes, dtype=torch.float32)  # shape=(num_objects,4)
            class_labels = torch.ones((num_objs,), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            frame_id = torch.tensor([idx])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        else:
            # this is a negative sample (no object, no bounding box)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)
            frame_id = torch.tensor([idx])
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': class_labels, 'image_id': frame_id, 'area': area, 'iscrowd': iscrowd}

        if self.transform is not None:
            frame, target = self.transform(frame, target)

        return frame, target


def get_model_instance(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train(model, optimizer, data_loader):
    model.train()
    device = next(model.parameters()).device
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def collate_fn(batch):
    return tuple(zip(*batch))
