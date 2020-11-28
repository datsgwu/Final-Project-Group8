#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 08:37:48 2020

@author: thomaskeeley
"""
import os
import glob
import pandas as pd
import random
from PIL import Image
# %% -------------------------------------------------------------------------------------------------------------------
os.getcwd()
wd = '/home/ubuntu/data/'
os.chdir(wd)

os.chdir(wd + 'images/')
img_file_list = [i for i in glob.iglob('*.png')]

os.chdir(wd + 'annotations/')
annot_file_list = [i for i in glob.iglob('*.txt')]

# %% -------------------------------------------------------------------------------------------------------------------
vehicle_images = []
annot_list = []
for file in annot_file_list:
    img_file = file.replace('txt', 'png')
    if img_file in img_file_list:
        with open(file, "r") as text:
            lines = text.readlines()
            for line in lines[2:]:
                split = line.split(' ')

                if split[-2] == 'small-vehicle':
                    filename = '/home/ubuntu/data/images/{}'.format(img_file)
                    vehicle_images.append(filename)
                    x_min = min(split[:-2][0::2])
                    x_max = max(split[:-2][0::2])
                    y_min = min(split[:-2][1::2])
                    y_max = max(split[:-2][1::2])

                    item = ",".join((filename, x_min, y_min, x_max, y_max, split[-2]))
                    annot_list.append(item)

# %% -------------------------------------------------------------------------------------------------------------------
vehicle_images = set(vehicle_images)
vehicle_images = random.sample(vehicle_images, 1000)
n_test = int(len(vehicle_images) * 0.1)

random.seed(42)
test_images = random.sample(vehicle_images, n_test)

vehicle_images = [i for i in vehicle_images if i not in test_images]

train = pd.DataFrame(columns=['path', 'x_min', 'y_min', 'x_max', 'y_max', 'object'])
test = pd.DataFrame(columns=['path', 'x_min', 'y_min', 'x_max', 'y_max', 'object'])
# %% -------------------------------------------------------------------------------------------------------------------
for item in annot_list:
    split = item.split(',')
    if split[0] in test_images:
        test = test.append({'path': split[0], 'x_min': int(float(split[1])), 'y_min': int(float(split[2])),
                            'x_max': int(float(split[3])), 'y_max': int(float(split[4])), 'object': split[5]},ignore_index=True)
    elif split[0] in vehicle_images:
        train = train.append({'path': split[0], 'x_min': int(float(split[1])), 'y_min': int(float(split[2])),
                            'x_max': int(float(split[3])), 'y_max': int(float(split[4])), 'object': split[5]},ignore_index=True)

# %% -------------------------------------------------------------------------------------------------------------------
os.chdir(wd + 'images/')
for file in train['path'].unique():
    im = Image.open(file)
    x_max, y_max = im.size
    for idx in train.loc[train['path'] == file].index:
        if train.iloc[idx]['x_max'] > x_max:
            train.iloc[idx]['x_max'] = x_max
        if train.iloc[idx]['y_max'] > y_max:
            train.iloc[idx]['y_max'] = y_max

train = (train[(train['x_max'] > train['x_min']) & (train['y_max'] > train['y_min'])]).reset_index(drop=True)
test = (test[(test['x_max'] > test['x_min']) & (test['y_max'] > test['y_min'])]).reset_index(drop=True)

# %% -------------------------------------------------------------------------------------------------------------------
class_df = pd.DataFrame(columns=['class', 'id']).append({'class': 'small-vehicle', 'id':0}, ignore_index=True)

class_df.to_csv(wd + 'classes.csv', header=False, index=False)
train.to_csv(wd + 'train.csv', header=False, index=False)
test.to_csv(wd + 'test.csv', header=False, index=False)

# %% -------------------------------------------------------------------------------------------------------------------
