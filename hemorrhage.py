
#############
# Libraries #
#############

import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
from torch.nn import Linear, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage, RandomApply, RandomHorizontalFlip, RandomAffine, RandomErasing

#########################
# Prepare training data #
#########################

train_df = pd.read_csv("./input/stage_1_train.csv")

# --------- One-hot-encode labels ---------
train_df[["ID", "Image", "Diagnosis"]] = train_df["ID"].str.split("_", expand = True) # 6 rows per image
train_df = train_df.drop("ID", axis = 1)
train_df = train_df.drop_duplicates()
train_df = train_df.pivot(index = "Image", columns = "Diagnosis", values = "Label") # one-hot-encode the label
train_df = train_df.reset_index()
train_df = train_df.rename_axis(None, axis = 1)
train_df["Image"] = "ID_" + train_df["Image"]

# --------- Remove IDs with no image ---------
imgs_found = glob.glob("./input/train/*.png") # get list of img paths
imgs_found = [os.path.basename(img_path)[:-4] for img_path in imgs_found] # convert to list of IDs

train_df = train_df[train_df["Image"].isin(imgs_found)] # filter train_df with list

train_df, val_df = train_test_split(
        train_df,
        shuffle = True,
        test_size = 0.15,
        random_state = 2019
)

###################
# Data Generators #
###################

class HemorrhageDataset(Dataset):
    
    def __init__(self, df, img_path, labels = True, augmentations = None):    
        self.df = df
        self.img_path = img_path
        self.augmentations = augmentations
        self.labels = labels

    def __len__(self):        
        return len(self.df)

    def __getitem__(self, index):        
        img_id = self.df.iloc[index, 0]
        img = cv2.imread(self.img_path + img_id + ".png")   
        
        if self.augmentations is not None:                  
            img = self.augmentations(img)
            
        if self.labels == True:     
            label = torch.tensor(self.df.iloc[index, [2, 3, 4, 5, 6, 1]]) # column order is relevant
            return img, label       
        else:                  
            return img
  
augs_train = Compose([
        ToPILImage(),
        RandomHorizontalFlip(p = 0.4),
        RandomApply([RandomAffine(degrees = 20, scale = (0.75, 0.9), shear = 20)], p = 0.4),
        ToTensor(), # also scales to 0...1
        RandomErasing(scale = (0.01, 0.10), ratio = (0.5, 1.5), p = 0.3)
])

augs_val = Compose([
        ToPILImage(),       
        ToTensor(),
])

train_dataset = HemorrhageDataset(train_df, "./input/train/", augmentations = augs_train)
val_dataset = HemorrhageDataset(val_df, "./input/train/", augmentations = augs_val)

batch_size = 32

train_gen = DataLoader(train_dataset, batch_size)
val_gen = DataLoader(val_dataset, batch_size)

##################
# Visualize data #
##################

imgs, _ = next(iter(val_gen))
imgs = imgs.numpy().transpose(0, 2, 3, 1) # reshape (n, ch, h, w) to (n, h, w, ch)

fig = plt.figure(figsize = (11, 5))
for i in range(18):
    plt.subplot(3, 6, i + 1)
    img = imgs[i, :, :, 0] # select first channel
    plt.imshow(img, cmap = "bone")
    plt.axis("off")
plt.tight_layout() 

#################
# Specify model #
#################

device = torch.device("cuda:0")

n_classes = 6
 
class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b0") 
        self.model._fc = Linear(1280, n_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

model = EfficientNetB0().to(device)

criterion = BCEWithLogitsLoss() # sigmoid layer + BCELoss
optimizer = Adam(model.parameters(), lr = 1e-4)

#############
# Fit model #
#############

num_epochs = 5

for epoch_i in range(num_epochs):
    model.train() # set train mode
    train_loss = []
    val_loss = []
    
    for batch_i, (images, labels) in enumerate(tqdm(train_gen)):
        images = images.to(device, dtype = torch.float)
        labels = labels.to(device, dtype = torch.float)  
        optimizer.zero_grad() # clear gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss.append(loss.item())
        loss.backward() # compute gradient
        optimizer.step() # update parameters
    
    # ------------- Evaluation on validation data -------------
    model.eval() # set evaluation mode
    with torch.no_grad():
        for batch_i, (images, labels) in enumerate(val_gen):
            images = images.to(device, dtype = torch.float)
            labels = labels.to(device, dtype = torch.float)
            output = model(images)
            loss = criterion(output, labels)
            val_loss.append(loss.item()) 
        
    print(f"\n{epoch_i + 1} | Train loss: {np.mean(train_loss):.4f} | Val loss: {np.mean(val_loss):.4f}")

#####################
# Prepare test data #
#####################

test_df = pd.read_csv("./input/stage_1_sample_submission.csv")

test_df[["ID", "Image", "Diagnosis"]] = test_df["ID"].str.split("_", expand = True)
test_df["Image"] = "ID_" + test_df["Image"]
test_df = test_df[["Image", "Label"]]
test_df = test_df.drop_duplicates()

augs_test = Compose([
        ToPILImage(),
        RandomHorizontalFlip(p = 0.4),
        RandomApply([RandomAffine(degrees = 20, scale = (0.75, 0.9), shear = 20)], p = 0.4),
        ToTensor(),
])

test_dataset = HemorrhageDataset(test_df, "./input/test/", augmentations = augs_test, labels = False)
test_gen = DataLoader(test_dataset, batch_size)

##############
# Prediction #
##############

# --------- Test Time Augmentation ---------
model.eval()

preds = np.zeros((len(test_dataset) * n_classes, 5))

with torch.no_grad():
    for fold_i in range(5):
        preds_fold = np.zeros((len(test_dataset) * n_classes, 1))
        
        for batch_i, images in enumerate(test_gen):        
            images = images.to(device, dtype = torch.float)                   
            preds_batch = model(images) 
            preds_batch = torch.sigmoid(preds_batch).cpu() # scale to 0...1 and copy to host memory
            preds_batch = preds_batch.reshape((len(images) * n_classes, 1)) # reshape (32, 6) to (32*6, 1)
            preds_fold[(batch_i * batch_size * n_classes):((batch_i + 1) * batch_size * n_classes)] = preds_batch
                
        preds[:, fold_i] = preds_fold.squeeze() # reshape (n, 1) to (n, ) first

preds = np.mean(preds, axis = 1) # take average per row

##############
# Submission #
##############
        
subm_df = pd.read_csv("./input/stage_1_sample_submission.csv")
subm_df["Label"] = preds
subm_df.to_csv("submission.csv", index = False)
