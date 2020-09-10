"""This file holds the classes for the models used"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # CNN network
    def __init__(self, start_nodes, layers, in_dim):
        super().__init__()
        self.start_nodes = start_nodes
        self.layers = layers

        self.conv1 = nn.Conv2d(1, int(start_nodes), 11)
        self.conv2 = nn.Conv2d(int(start_nodes), int(start_nodes/2), 5)
        self.conv3 = nn.Conv2d(int(start_nodes/2), int(start_nodes/4), 5)


        x = torch.randn(in_dim,in_dim).view(-1,1,in_dim,in_dim)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 1)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        if self.layers == 3:
            x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))


        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class vecNet2(nn.Module):
    # 2 CNN branch hybrid
    def __init__(self, vec_start, vec_layers, in_dim, img_start, dsm_start, img_layers, dsm_layers):
        super().__init__()  # just run the init of parent class (nn.Module)
        self.cnn1 = Net(img_start, img_layers, 128)
        self.cnn2 = Net(dsm_start, dsm_layers, 64)

        self.layers = vec_layers
        self.fc1 = nn.Linear(in_dim, int(vec_start))
        self.fc2 = nn.Linear(int(vec_start), int(vec_start / 2))
        self.fc3 = nn.Linear(int(vec_start / 2), int(vec_start / 4))

        x = torch.randn(1, in_dim).view(-1, 1, in_dim)
        self._to_linear = None
        self.fc_layers(x)

        self.fc4 = nn.Linear(self._to_linear + 2, 1)

    def fc_layers(self, x):
        x = F.relu(self.fc1(x))
        if self.layers > 1:
            x = F.relu(self.fc2(x))
        if self.layers > 2:
            x = F.relu(self.fc3(x))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]
        return x

    def forward(self, img, dsm, vec):
        x1 = self.cnn1(img)
        x2 = self.cnn2(dsm)
        x3 = self.fc_layers(vec)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(-1, self._to_linear + 2)  # .view is reshape ... this flattens X before
        x = self.fc4(x)
        return torch.sigmoid(x)


class vecNet3(nn.Module):
    # Single CNN branch hybrid
    def __init__(self, vec_start, vec_layers, in_dim, img_start, img_layers, img_dim):
        super().__init__()  # just run the init of parent class (nn.Module)
        self.cnn1 = Net(img_start, img_layers, img_dim)

        self.layers = vec_layers
        self.fc1 = nn.Linear(in_dim, int(vec_start))
        self.fc2 = nn.Linear(int(vec_start), int(vec_start / 2))
        self.fc3 = nn.Linear(int(vec_start / 2), int(vec_start / 4))

        x = torch.randn(1, in_dim).view(-1, 1, in_dim)
        self._to_linear = None
        self.fc_layers(x)

        self.fc4 = nn.Linear(self._to_linear + 1, 1)

    def fc_layers(self, x):
        x = F.relu(self.fc1(x))
        if self.layers > 1:
            x = F.relu(self.fc2(x))
        if self.layers > 2:
            x = F.relu(self.fc3(x))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1]
        return x

    def forward(self, img, vec):
        x1 = self.cnn1(img)
        x2 = self.fc_layers(vec)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(-1, self._to_linear + 1)  # .view is reshape ... this flattens X before
        x = self.fc4(x)
        return torch.sigmoid(x)