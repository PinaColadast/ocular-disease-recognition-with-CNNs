## Please install pytorch-gradcam as dependency to run gradcam
#pip install pytorch-gradcam
# pip install pytorch-gradcam

import os
import sys

import PIL
import torch
import torchvision
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

import torch.nn as nn
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""please pip install pytorch-gradcam to run gradcam 
   the whole dataset used for reproduce the result can be downloaded via Googledrive: 
   for images(please download in a folder): https://drive.google.com/drive/folders/1cuCfvs5qR4BXxOk-_-ALzbZr5DdxsgXG?usp=sharing
   for csv full_df.csv describing metadata: https://drive.google.com/file/d/1-XEKHT-EVWx2M-PmVYV1m4buBLW-QGqa/view?usp=sharing
"""

#set global variables
#please fill in path of the cnn model named multiclass_augmented_model.pth including model name, e.g. "/user/model.pth"
path_of_cnn_model = ''

#Please fill in the complete path of the csv full_df.csv inluding file name, e.g. "/user/document/full_df.csv"
full_df_path = ""
#
#Please fill in the absolute path where the image dataset is stored(downloaded via the google drive)
#e.g. "/users/data/images_directory"
image_path = ""

#check our GPU availability 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
PATH_cnn = path_of_cnn_model
#CNN definition
class Net2(nn.Module):
    def __init__(self): #Define the layers of the CNN
        super(Net2, self).__init__()

        # Convolutions and Batch-Normalization
        self.conv1 = nn.Conv2d(3, 12, 13)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(12,36, 5, padding = (2,2))
        self.bn2 = nn.BatchNorm2d(36)

        self.conv3 = nn.Conv2d(36, 72, 6)
        self.bn3 = nn.BatchNorm2d(72)

        self.conv4 = nn.Conv2d(72,144, 5, padding = (2,2))
        self.bn4 = nn.BatchNorm2d(144)

        # Max-pooling and Dropout
        self.pool = nn.MaxPool2d(4, 4)
        self.dropout = nn.Dropout(0.25)

        # Normal NN at the end

        self.fc1 = nn.Linear(144 * 30 * 30, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 8)

    def forward(self, x): #Define the forward computing of the result

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(self.pool(x))

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(self.pool(x))

        x = x.view(-1, 144 * 30 * 30)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = Net2()
model.load_state_dict(torch.load(PATH_cnn, map_location=device))

image_path = image_path
def get_image(filename):
  im_path = os.path.join(image_path, filename)
  img = Image.open(im_path)
  return img

def non_norm_tensor(img):
    torch_img = transforms.Compose([
    transforms.Resize((512)),
    transforms.ToTensor()
  ])
    return torch_img(img)

def normed_torch_image(torch_img):
    normed_torch_img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return normed_torch_img(torch_img)[None]


df = pd.read_csv(full_df_path)
filenames = []
label_name =["['N']", "['D']", "['G']", "['C']", "['A']", "['H']", "['M']", "['O']"]
for i in label_name:
  filenames.append(df.loc[df["labels"]== i]["filename"].tolist()[1])
# filenames = [j for i in filenames for j in i]


images = []
# class_idx = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
class_idx = [0,1,2,3,4,5,6,7]

for i,j in enumerate(filenames):
  target_layer = model.conv4
  gradcampp = GradCAMpp(model,target_layer)
  torch_img =  non_norm_tensor(get_image(j))
  normed_torch_img = normed_torch_image(torch_img)

  mask_pp, _ = gradcampp(normed_torch_img, class_idx = class_idx[i])
  

  heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
  images.extend([torch_img.cpu(), 
                #  heatmap_pp,
                 result_pp])

grid_images = make_grid(images, nrow = 6)
transforms.ToPILImage()(grid_images)

from torchvision.utils import save_image

save_image(grid_images, "{}/grad-cam-cnn.jpg".format(os.getcwd()))

print("grad-cam generated images are saved to {}/ as grad-cam-cnn.jpg".format(os.getcwd()))
transforms.ToPILImage()(grid_images)
