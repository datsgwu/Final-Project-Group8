#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:53:58 2020

@author: thomaskeeley
"""
import sys
import os

os.system('sudo pip install torch')
os.system('sudo pip install torchvision')
os.system('sudo pip install pycocotools')
os.system('sudo pip install scikit-image')
os.system('sudo pip install Pillow')
os.system('sudo pip install tifffile')
import numpy as np
import pandas as pd
import torch
import torch.nn
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

wd = '/home/ubuntu/project/'

os.chdir(wd + '/pytorch-retinanet/retinanet/')
sys.path.append('/home/ubuntu/final_project/Project/pytorch-retinanet/')

from retinanet.dataloader import *
from retinanet import csv_eval

#%%
model = torch.load(wd + 'Final-Project-Group8/model_final.pt')

os.chdir(cwd)
dataset_val = CSVDataset(train_file='/home/ubuntu/data/test.csv',
                             class_list='/home/ubuntu/data/classes.csv',
                             transform=transforms.Compose([Normalizer(), Resizer()]))

dets = csv_eval._get_detections(dataset_val, model, score_threshold=0.0, max_detections=1000, save_path=None)

test_df = pd.read_csv('/home/ubuntu/data/test.csv', header=None)

img_path_list = (test_df[0].unique()).tolist()

#%%
for i in range(len(img_path_list)):
    path = img_path_list[i]
    detections = dets[i][0]
    img = cv2.imread(path)

    plt.figure()
    plt.imshow(img)
    for bbox in detections:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        box = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
        plt.axes().add_patch(box)

    plt.show()
