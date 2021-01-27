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
""" the whole dataset used for reproduce the result can be downloaded via Googledrive: 
    for images(please download in a folder): https://drive.google.com/drive/folders/1cuCfvs5qR4BXxOk-_-ALzbZr5DdxsgXG?usp=sharing
    for csv full_df.csv describing metadata: https://drive.google.com/file/d/1-XEKHT-EVWx2M-PmVYV1m4buBLW-QGqa/view?usp=sharing"""

#Please fill in the absolute path of the full_df.csv file, including filename, e.g. "/users/data/full_df.csv"
metadata_csv_path =  ""

#Please fill in the absolute path where the image dataset is stored(downloaded via link shared from googledrive)
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

labels = {}
for index, row in df_data_all.iterrows():
    filename = row["filename"]
    labels[filename] = torch.tensor(np.array(ast.literal_eval(row["target"])))



class CustomDataSet(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
    def __init__(self, main_dir, transform, list_IDs, labels):
        'Initialization'
        self.main_dir = main_dir
        self.transform = transform

        self.labels = labels
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

        return tensor_image, label



def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG13_bn
        """
        model_ft = models.vgg13_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_name, is_inception=False):
    since = time.time()

    val_acc_history   = []
    train_acc_history = []
    val_loss = []
    train_loss = []

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
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
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
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels.type_as(outputs))
                        loss2 = criterion(aux_outputs, labels.type_as(aux_outputs))
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        labels  = labels.type_as(outputs)
                        loss = criterion(outputs, labels.type_as(outputs))

                    _, preds = torch.max(outputs, dim = 1)
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

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    
    return model, train_acc_history, val_acc_history, train_loss, val_loss

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_name = "vgg"
feature_extract = False
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft.to(device)
# Print the model we just instantiated
# print(model_ft)

image_path = image_path

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer = optim.SGD(params_to_update, lr=0.001, weight_decay=0.005)
criterion = nn.BCEWithLogitsLoss()

image_datasets = {x: CustomDataSet(data_dir, data_transforms[x], partition[x],labels) for x in ["train", "val"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}



# Train and evaluate
model_ft, hist_train, hist_val, train_loss, val_loss = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs, "vgg", is_inception=False)
df_acc_vgg = pd.DataFrame({"train_acc": hist_train, "val_acc":hist_val,
                           "train_loss" : train_loss, "val_loss": val_loss})
df_acc_vgg.to_csv("{}/{}.txt".format(os.getcwd(), "vgg_model_performance"), sep = "\t")
print("training history is exported to {}".format(os.getcwd()))
preds = np.array([])
vals  = np.array([])
for input_bt, label_bt in dataloaders["val"]:
    input_bt = input_bt.to(device)
    label_bt = label_bt.to(device)
    y_preds = model_ft(input_bt)

    _, pred_val = torch.max(y_preds, dim = 1)
    _,true_val  = torch.max(label_bt, dim =1)

    preds = np.append(preds, pred_val.cpu().numpy())
    vals  = np.append(vals, true_val.cpu().numpy())


classes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
label_num = [0,1,2,3,4,5,6,7,]
print("accuracy score for model is: {}".format(accuracy_score(vals, preds)))
print("f1_score for model is: {}".format(f1_score(vals, preds, average = "micro")))
confusion_matrix(vals, preds, labels = label_num)
