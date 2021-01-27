import os

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from PIL import Image

import os
import time
import copy
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

#set global environments#
""" 
the whole dataset used for reproduce the result can be downloaded via Googledrive: 
for images(please download in a folder): https://drive.google.com/drive/folders/1cuCfvs5qR4BXxOk-_-ALzbZr5DdxsgXG?usp=sharing
for csv full_df.csv describing metadata: https://drive.google.com/file/d/1-XEKHT-EVWx2M-PmVYV1m4buBLW-QGqa/view?usp=sharing 

"""

#Please fill in the full absolute path of the full_df.csv file (including the filename), e.g. "/users/data/full_df.csv"
metadata_csv_path =  ""

#Please fill in the absolute path where the image dataset is stored(shared via google drive)
#e.g. /users/data/images_directory
image_path = " "

#num of classes to be predicted 
num_classes = 8

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 50


#helper functions#
#to undersample the data
def undersample(dataframe, column= (), label = (), number=()):
    df_undersampled = ""
    df_m = dataframe
    if isinstance(label, str):
      df_presele = df_m.loc[df_m[column] == "{}".format(label)]
      if len(df_presele) < number:
        replace = True
      else:
        replace = False
      df_sele = df_presele.sample(n=number, random_state = np.random.RandomState(),replace = replace)
      df_non = df_m.loc[df_m[column] !=  "{}".format(label)]
      df_undersampled = pd.concat([df_sele, df_non])
    
    if isinstance(label, list):
        df_non = df_m.loc[~df_m[column].isin(label)]
#     print(df_non)
        for i,j in enumerate(label):
            df_presele = df_m.loc[df_m[column] == j]
            if len(df_presele)<number[i]:
                replace = True
            else:
                replace = False
#             print (number[i], j)
        df_sele = df_presele.sample(n=number[i], random_state = np.random.RandomState(), replace = replace)
        df_non = df_non.append(df_sele)
        df_undersampled = df_non
            
    return df_undersampled

#load the data#    
label_aug = ["red_contr", "rotated", "noised"]
df_path = metadata_csv_path #change the path to the dataframe that describes the metadata#
df = pd.read_csv(df_path)
df_gen = pd.get_dummies(df['Patient Sex'], prefix=None)
df["female"] = df_gen.Female
df["male"] = df_gen.Male

df_data = df
df_train, df_test = train_test_split(df_data, test_size=0.15, random_state=42, stratify = df_data.target)
df_train.reset_index(drop = True)
df_train = df_train.copy().reset_index(drop = True)

df_train_4aug = df_train.loc[df_train["target"]!='[1, 0, 0, 0, 0, 0, 0, 0]']
df_train_norm = df_train.loc[df_train["target"]=='[1, 0, 0, 0, 0, 0, 0, 0]']

df_aug_info = df_train.copy()

for i in label_aug:
    df_aug_i = df_train_4aug.copy()
    df_aug_i["filename"] = df_aug_i.apply(lambda x: i+"_"+df_aug_i["filename"] , axis = 1)
    df_aug_info = pd.concat([df_aug_info, df_aug_i])


df_train_aug = undersample(df_aug_info, column="target", label= ["[0, 1, 0, 0, 0, 0, 0, 0]", "[0, 0, 0, 0, 0, 0, 0, 1]",
                                                                "[0, 0, 0, 1, 0, 0, 0, 0]", "[0, 0, 1, 0, 0, 0, 0, 0]", "[0, 0, 0, 0, 1, 0, 0, 0]",
                                                                "[0, 0, 0, 0, 0, 0, 1, 0]", "[0, 0, 0, 0, 0, 1, 0, 0]"],
                         number = [2000]*7)
df_data_all  = pd.concat([df_train_aug, df_test])

partition={}
partition["train"] = df_train_aug.filename.tolist()
partition["val"]= df_test.filename.tolist()

labels, meta_data = {}, {}
for index, row in df_data_all.iterrows():
    filename = row["filename"]
    labels[filename] = np.array(ast.literal_eval(row["target"]))
    meta_data[filename] = np.array(row[["Patient Age", "female", "male"]].tolist())




class CustomDataSet(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
    def __init__(self, main_dir, transform, list_IDs, labels, meta_data):
        'Initialization'
        self.main_dir = main_dir
        self.transform = transform

        self.labels = labels
        self.meta_data= meta_data
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        img_loc = os.path.join(self.main_dir, ID)
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        label = self.labels[ID]
        meta_data = self.meta_data[ID]
        return tensor_image, meta_data, label

class Net2(nn.Module):
    def __init__(self): #Define the layers of the CNN
        super(Net2, self).__init__()

        # Convolutions and Batch-Normalization
        self.conv1 = nn.Conv2d(3, 12, 13)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(12, 36, 5, padding = (2,2))
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
        
        
        # self.fc4 = nn.Linear(3) 
        self.bn_fc4 = torch.nn.BatchNorm1d(3, momentum=0) # for tabular data normalization
        self.fc4 = nn.Linear(64 + 3, 128 )
        self.bn_fc5 = torch.nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 8)


    def forward(self, x1, x2): #Define the forward computing of the result

        x1 = F.relu(self.bn1(self.conv1(x1)))

        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = self.dropout(self.pool(x1))

        x1 = F.relu(self.bn3(self.conv3(x1)))

        x1 = F.relu(self.bn4(self.conv4(x1)))
        x1 = self.dropout(self.pool(x1))

        x1 = x1.view(-1, 144 * 30 * 30)

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
      # tabular data
        x2 = F.relu(self.bn_fc4(x2.float()))
        x3 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.fc4(x3))
        x3 = F.relu(self.bn_fc5(x3))
        x3 = F.relu(self.fc5(x3))
        x3 = F.relu(self.fc6(x3))

       
        return x3


cnn_net = Net2()
# cnn_net.to(device)

def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_name, is_inception=False):
    since = time.time()

    val_acc_history   = []
    train_acc_history = []

    train_loss, val_loss = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.5

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, meta_data, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
                input_meta = meta_data.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs, input_meta)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs, input_meta)
                        labels  = labels.type_as(outputs)
                        loss = criterion(outputs, labels.type_as(outputs))

                    _, preds = torch.max(outputs, dim=1)
                    _, trues = torch.max(labels, dim =1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # print('Loss: {}'.format(loss)) 
                # statistics
                running_loss += loss.item() * inputs.size(0)

                

                running_corrects += (preds == trues).sum()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = np.float(running_corrects) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, 
                                                       epoch_loss,
                                                       epoch_acc
                                                       ))

            # deep copy the model
            if phase == "train":
              train_acc_history.append(epoch_acc)
              train_loss.append(epoch_loss)
            if phase == 'val' and epoch_acc > best_acc:
              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())
              

            if phase == 'val':
              val_acc_history.append(epoch_acc)
              val_loss.append(epoch_loss)


        #     if epoch%5 == 0:
        #       PATH = '/content/gdrive/My Drive/Deepl_project/dp_model/{}_epoch{}'.format(model_name, epoch)
        #       torch.save({
        #       'epoch': epoch,
        #       'model_state_dict': model.state_dict(),
        #       'optimizer_state_dict': optimizer.state_dict(),
        #       'loss': loss,
        #       }, PATH)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # df_acc.to_csv("{}/{}.txt".format(os.getcwd(), "model_performance"), sep = "\t")
    
    return model, train_acc_history, val_acc_history, train_loss, val_loss



# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_ft = cnn_net.to(device)
# Print the model we just instantiated
# print(model_ft)

image_path = image_path
input_size = 512
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

data_dir = image_path
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()

# print("Params to learn:")

# Observe that all parameters are being optimized
optimizer = optim.SGD(params_to_update, lr=0.001, weight_decay=0.005)
criterion = nn.BCEWithLogitsLoss()

image_datasets = {x: CustomDataSet(data_dir, data_transforms[x], partition[x], labels, meta_data) for x in ["train", "val"]}
# meta_datasets = {x: for x in ["train", "val"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}


# Train and evaluate
model_ft, hist_train, hist_val, train_loss, val_loss = train_model(model_ft, dataloaders, 
                                                                    criterion, optimizer, num_epochs, "cnn_meta", is_inception=False)

df_acc_meta = pd.DataFrame({"train_acc": hist_train, "val_acc":hist_val,
                           "train_loss" : train_loss, "val_loss": val_loss})
df_acc_meta.to_csv("{}/{}.txt".format(os.getcwd(), "cnn_meta_model_performance"), sep = "\t")
print("training history is exported to {}".format(os.getcwd()))

# preds = np.array([])
# vals  = np.array([])
# for input_bt, meta_data ,label_bt in dataloaders["val"]:
#   # if a >=2: break
#     input_bt = input_bt.to(device)
#     label_bt = label_bt.to(device)
#     meta_data = meta_data.to(device)
#     y_preds = model_ft(input_bt, meta_data)

#     _, pred_val = torch.max(y_preds, dim = 1)
#     _,true_val  = torch.max(label_bt, dim =1)

#     preds = np.append(preds, pred_val.cpu().numpy())
#     vals  = np.append(vals, true_val.cpu().numpy())


# classes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
# label_num = [0,1,2,3,4,5,6,7,]
# print("accuracy score of the model is: {}".format(accuracy_score(vals, preds)))
# print("f1_score of the model is: {}".format(f1_score(vals, preds, average = "micro")))
# confusion_matrix(vals, preds, labels = label_num)
