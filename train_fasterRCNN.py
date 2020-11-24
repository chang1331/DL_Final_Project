import torch
import torchvision
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pytorch_vision_detection.engine import train_one_epoch, evaluate
import pytorch_vision_detection.utils as utils
import pytorch_vision_detection.transforms as T


class PackageDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, label_dir, transform=None):
        self.frame_dir = frame_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_dir + 'labels.csv', names=['x1', 'y1', 'x2', 'y2', 'frame_name', 'label'])
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


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
except NameError:
    current_file_dir = os.getcwd().replace('\\', '/')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

frame_dir = current_file_dir + '/data/images/'
label_dir = current_file_dir + '/data/'
package_dataset = PackageDataset(frame_dir=frame_dir, label_dir=label_dir, transform=get_transform(train=True))
package_dataset_test = PackageDataset(frame_dir=frame_dir, label_dir=label_dir, transform=get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(666)
train_fraction = 0.8
indices = torch.randperm(len(package_dataset)).tolist()
dataset = torch.utils.data.Subset(package_dataset, indices[:int(len(indices) * train_fraction)])
dataset_test = torch.utils.data.Subset(package_dataset_test, indices[int(len(indices) * train_fraction):])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False, collate_fn=utils.collate_fn)

model = get_model_instance(num_classes=2)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

testImgPath = current_file_dir + '/frame_51.jpeg'
testImgPath = current_file_dir + '/test_img.png'
img1 = Image.open(testImgPath)
img2 = Image.open(testImgPath)
img2 = img2.convert('RGB')
img1 = torchvision.transforms.functional.to_tensor(img1)
img2 = torchvision.transforms.functional.to_tensor(img2)
images = torch.unsqueeze(img, 0)
images = list(image for image in images)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.eval()
pred = model(images)
boxes = pred

frame = cv2.imread(testImgPath)
cv2.imshow('image', frame)
cv2.waitKey()
cv2.rectangle(frame, (71, 27), (159, 93), color=(0, 255, 0), thickness=1)
