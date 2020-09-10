"""This file holds the functions for generating feature tensors"""

import pandas as pd
import os
import cv2
import numpy as np
import torch

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


class make_data():
    IMG_SIZE = 128
    DSM_SIZE = 64
    POS = 'positive'
    NEG = 'negative'
    LABELS = {POS: 1, NEG: 0}
    training_data = []
    #     TABLES = {POS: pd.read_csv('data/tables/8_positive_final.csv'),
    #               NEG: pd.read_csv('data/tables/8_negative_final.csv')}
    TABLES = {POS: pd.read_csv('data/tables/9_positive_with_point.csv'),
              NEG: pd.read_csv('data/tables/9_negative_with_point.csv')}

    def make_training_data(self):
        poscount = 0
        negcount = 0
        for label in self.LABELS:
            table = self.TABLES[label]
            print(table.columns)
            for f in os.listdir(os.path.join('data/building_mask', label)):
                try:
                    impath = os.path.join('data/building_mask', label, f)
                    dsmpath = os.path.join('data/DSM', label, f)

                    img = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                    dsm = cv2.imread(dsmpath)
                    dsm = np.array(cv2.resize(img, (self.DSM_SIZE, self.DSM_SIZE)))
                    dsm[dsm == -9999.0] = 0  # replace nodata with 0
                    dsm[dsm != 0] = dsm[dsm != 0] - min(dsm[dsm != 0])  # Remove ground level
                    tabular = torch.tensor(np.array(table.loc[table['1'] == int(f.split('.')[0])])[:, 2:][0])
                    self.training_data.append([np.array(img), dsm, tabular, self.LABELS[label]])
                    # print(np.eye(2)[self.LABELS[label]])

                except Exception as e:
                    pass
        #                     print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("ensemble_data.npy", self.training_data)