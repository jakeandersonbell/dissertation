"""This file holds the hyperparameter testing stage"""

import pandas as pd
import pickle
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from modelling.model_class import vecNet2
from modelling.training_func import *


os.chdir('/home/anderson/project')
print(os.getcwd())

REBUILD_DATA = False

print(torch.cuda.device_count())

print(torch.cuda.is_available())

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

print(device)

training_data = np.load("ensemble_data.npy", allow_pickle=True)

X = [torch.Tensor([i[0].astype(np.float32) for i in training_data]).view(-1, 128, 128),
     torch.Tensor([i[1].astype(np.float32) for i in training_data]).view(-1, 64, 64),
     torch.Tensor(np.array([np.array(xi) for xi in training_data[:, 2]]))]

X[0] = X[0]/255.0
X[1] = X[1]/255.0
y = torch.Tensor([i[3] for i in training_data])

VAL_PCT = 0.1  # lets reserve 10% of our data for validation/test
val_size = int(len(X[0])*VAL_PCT)

weight = torch.tensor([0.688, 0.312])
weight_ = weight[y.data.view(-1).long()].view_as(y).to(device)

EPOCHS = 200
BATCH_SIZE = 128
os.listdir('models/new')
len([]) == False


train_X1, train_X2, train_X3, train_y, val_X1, val_X2, val_X3, val_y, perm = shuffle_data(X, y)

xy = [train_X1, train_X2, train_X3, train_y, val_X1, val_X2, val_X3, val_y]

# Hyperparameter testing loop
vec_nodes = [64, 128, 256]
cnn_nodes = [64, 128, 256]
vec_layers = [3, 2, 1]
cnn_layers = [3, 2, 1]
batch_sizes = [128, 64]
models = []
# with open('models/new/ensemble_param_test2.pickle', 'rb') as src:
#     models = pickle.load(src)

with open('models/new/rand_perm4.pickle', 'wb') as dest:
    pickle.dump(perm, dest)

# Append the random perm used to order the data before val split
models.append(perm)

for vec_node in vec_nodes:
    for cnn_node in cnn_nodes:
        for vec_layer in vec_layers:
            for cnn_layer in cnn_layers:
                for batch_size in batch_sizes:
                    if 'net' in locals() or 'net' in globals():
                        del net
                    net = vecNet2(vec_node, vec_layer, 43, cnn_node, cnn_node, cnn_layer, cnn_layer).to(device)
                    optimizer = optim.Adam(net.parameters(), lr=0.0005)
                    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=7)
                    models.append(train(net, 200, batch_size, 0.0005, vec_layer, vec_node, cnn_node, cnn_layer, xy,
                                        optimizer, weight_, scheduler))
                    with open('models/new/ensemble_param_test4.pickle', 'wb') as dest:
                        pickle.dump(models, dest)

with open('models/new/ensemble_param_test4.pickle', 'rb') as src:
    models = pickle.load(src)

data_perm = models[0]  # This can be reused to get the correct split

model_df = pd.DataFrame(models[1:], columns=['MODEL_NAME', 'EPOCHS', 'BATCH_SIZE', 'LR', 'LYR', 'START_NODES',
                                             'cnn_node', 'cnn_layer', 'loss', 'val_loss'])

model_df['loss'] = model_df['loss'].apply(lambda x: float(x))
model_df['val_loss'] = model_df['val_loss'].apply(lambda x: float(x))
