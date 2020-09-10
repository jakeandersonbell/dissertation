"""This file holds the ablation stage"""

import pickle
import numpy as np
from sklearn import metrics
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from modelling.model_class import vecNet2, vecNet3
from modelling.training_func import *


os.chdir('/home/anderson/project')
print(os.getcwd())

REBUILD_DATA = False

print(torch.cuda.device_count())

print(torch.cuda.is_available())

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

print(device)

# Individual features in dataset
features = ['Area', 'Month', 'Amusement', 'Auto', 'Contractor', 'Drinking', 'Emergency Services', 'Food Vendor',
            'Leisure', 'Lodging', 'Medical', 'Office', 'Public Building', 'Retail', 'Service', 'Storage', 'Transport',
            'Age 0-16 %', 'Age 16-35 %', 'Age 35-65 %', 'Age 65+ %', 'Uneployed %', 'Single Occupant %',
            'Population Density', 'No Qualification %', 'Level 1 Qualification %', 'Level 2 Qualification %',
            'Level 3 Qualification %', 'Level 4 Qualification %', 'Social Grade A/B %', 'Social Grade C1 %',
            'Social Grade C2 %', 'Social Grade D/E %', 'Owned Residential %', 'Social Rented %', 'Private Rented %',
            'White %', 'Mixed %', 'Asian %', 'Black/African %', 'Other Ethnicity %', 'Burglary Crime', 'Violent Crime']

# Sub groups for features
area = [0]
month = [1]
places = list(range(2, 17))
demographic = list(range(17, 41))
age = list(range(17, 21))
qualification = list(range(24, 29))
social_g = list(range(29, 33))
residential = list(range(33, 36))
residential.append(22)
ethnic = list(range(36, 41))
crime = list(range(41, 43))

groups = {'all': [], 'area': area, 'month': month, 'places': places, 'demo': demographic,
          'age': age, 'qual': qualification, 'soc': social_g, 'res': residential,
          'eth': ethnic, 'crime': crime, "img": [], "dsm": []}

training_data = np.load("ensemble_data.npy", allow_pickle=True)

X = [torch.Tensor([i[0].astype(np.float32) for i in training_data]).view(-1,128,128),
     torch.Tensor([i[1].astype(np.float32) for i in training_data]).view(-1,64,64),
     torch.Tensor(np.array([np.array(xi) for xi in training_data[:, 2]]))]

X[0] = X[0]/255.0
X[1] = X[1]/255.0
y = torch.Tensor([i[3] for i in training_data])

VAL_PCT = 0.1  # lets reserve 15% of our data for validation
val_size = int(len(X[0])*VAL_PCT)

weight = torch.tensor([0.688, 0.312])
weight_ = weight[y.data.view(-1).long()].view_as(y).to(device)

loss_function = nn.BCELoss(reduce=False)

with open('models/new/ensemble_param_test_final.pickle', 'rb') as src:
    models = pickle.load(src)

data_perm = models[0]

train_X1, train_X2, train_X3, train_y, val_X1, val_X2, val_X3, val_y, perm = shuffle_data(X, y, data_perm)

xy = [train_X1, train_X2, train_X3, train_y, val_X1, val_X2, val_X3, val_y]


"""Scalar ablation"""
# This loops though the feature groups. With each iteration it removes the feature/s, trains a model 10 times, recording
# auc for each one

# Begin a pickle to hold results - ablation takes a while and it is likely connection could be lost while using ssh
with open('ablation.pickle', 'wb') as dest:
    pickle.dump([], dest)

with open('ablation.pickle', 'rb') as src:
    models = pickle.load(src)

for key, group in groups.items():
    print(key)
    # remove the feature
    new_xy = get_select_features(group, xy)

    # train 10 times for each model
    for i in range(10):
        net = vecNet2(256, 1, 43 - len(group), 128, 128, 3, 3).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=7)

        res = list(train(net, 200, 128, 0.0005, 1, 128, 128, 3, new_xy))
        res.append(key)

        # Batch the validation prediction so it fits
        batch_size = xy[4].view(-1, 1, 128, 128).shape[0] // 5  # Into 5 batches

        step = 0

        y_pred = net(new_xy[4].view(-1, 1, 128, 128)[:batch_size].to(device),
                     new_xy[5].view(-1, 1, 64, 64)[:batch_size].to(device),
                     new_xy[6].view(-1, 43 - len(groups[key]))[:batch_size].to(device)).detach().cpu().numpy()
        for i in range(4):
            step += batch_size
            y_pred_new = net(new_xy[4].view(-1, 1, 128, 128)[step:step + batch_size].to(device),
                             new_xy[5].view(-1, 1, 64, 64)[step:step + batch_size].to(device),
                             new_xy[6].view(-1, 43 - len(groups[key]))[step:step + batch_size].to(
                                 device)).detach().cpu().numpy()
            y_pred = np.append(y_pred, y_pred_new)

        fpr, tpr, thresholds = metrics.roc_curve(new_xy[7][:len(y_pred)], y_pred)
        auc = metrics.auc(fpr, tpr)

        res.append([fpr, tpr, thresholds, auc])
        models.append(res)

        with open('ablation.pickle', 'wb') as dest:
            pickle.dump(models, dest)


"""CNN ablation"""
# Similar to the scalar loop above except the single CNN branch model is used and the CNN input dimension varies for
# each feature
# A CNN branch is removed for each feature removed
models = []

# the Image and DSM branches take different sized inputs
to_train = {'img': 128,
            'dsm': 64}

for i in to_train.items():
    for j in range(10):
        net = vecNet3(256, 1, 43, 128, 3, i[1]).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0005)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=7)
        # _, _, _, _, _, _, _, val_loss = train(net, 40, 256, 0.0001, 1, 128)
        # print(val_loss)
        res = list(train(net, 200, 128, 0.0005, 1, 128, 128, 3, xy, all_x=i[0]))

        # This portion of the loop gets the auc
        # Batch the validation prediction so it fits
        batch_size = xy[4].view(-1, 1, 128, 128).shape[0] // 5  # Into 5 batches

        step = 0
        if i == 'img':
            y_pred = net(new_xy[4].view(-1, 1, 128, 128)[:batch_size].to(device),
                         new_xy[6].view(-1, 43)[:batch_size].to(device)).detach().cpu().numpy()
            for i in range(4):
                step += batch_size
                y_pred_new = net(new_xy[4].view(-1, 1, 128, 128)[step:step + batch_size].to(device),
                                 new_xy[6].view(-1, 43)[step:step + batch_size].to(device)).detach().cpu().numpy()
                y_pred = np.append(y_pred, y_pred_new)

        else:
            y_pred = net(new_xy[5].view(-1, 1, 64, 64)[:batch_size].to(device),
                         new_xy[6].view(-1, 43)[:batch_size].to(device)).detach().cpu().numpy()
            for i in range(4):
                step += batch_size
                y_pred_new = net(new_xy[5].view(-1, 1, 64, 64)[step:step + batch_size].to(device),
                                 new_xy[6].view(-1, 43)[step:step + batch_size].to(device)).detach().cpu().numpy()
                y_pred = np.append(y_pred, y_pred_new)

        fpr, tpr, thresholds = metrics.roc_curve(new_xy[7][:len(y_pred)], y_pred)
        auc = metrics.auc(fpr, tpr)

        print(auc)

        res.append([fpr, tpr, thresholds, auc])

        res.append(i[0])
        models.append(res)

        with open('cnn_ablation.pickle', 'wb') as dest:
            pickle.dump(models, dest)