
#!/usr/bin/env python3.6

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

## The goal of the code is to plot the noise probality density curve for
## the tracked output of different trackers w.r.t. the ground truth boxes
## The meaning of noise here is to be treated as measurement noise in kalman 
## filter where, DCF filter output is taken as the measurements for position
## and fed to Kalman filter as measurement input.

def np_mat_from_txt(text_file):
    file_h = open(text_file)
    lines  = file_h.readlines()
    file_h.close()

    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", '')
        lines[i] = lines[i].replace("\r", '')
        lines[i] = lines[i].split(",")
    lines = np.array(lines, dtype=np.float)
    return lines

def get_tracking_error_dict(dataset, seq_list, tracker):
    error_dict = {}
    for seq in seq_list:
        if dataset == 'uav123':
            tracked_out_file = "/media/dtu-project2/2GB_HDD/object_tracker/UAV123/CFtracker_output_opencv2/not_raw/"+tracker+'/'+ seq + ".txt"
            gt_box_file = "/media/dtu-project2/2GB_HDD/UAV123/annotations/"+seq+'.txt'
        if dataset == "dtb70":
            tracked_out_file = "/media/dtu-project2/2GB_HDD/object_tracker/DTB70/CFtracker_output_opencv/not_raw/"+tracker+"/" + seq + ".txt"
            gt_box_file = "/media/dtu-project2/2GB_HDD/DTB70/dataset/" + seq + "/groundtruth_rect.txt"

        gt_boxes = np_mat_from_txt(gt_box_file)
        gt_boxes_x = np.expand_dims(gt_boxes[:,0], axis=1)
        track_boxes = np_mat_from_txt(tracked_out_file)
        track_boxes_x = np.expand_dims(track_boxes[:,0], axis=1)
        if len(gt_boxes_x) != len(track_boxes_x):
            if (len(gt_boxes_x) - len(track_boxes_x) > 0):
                track_boxes_x = np.pad(track_boxes_x, (0,(len(gt_boxes_x)-len(track_boxes_x))), 'constant', constant_values = (0,0))
                track_boxes_x = track_boxes_x[:,0]
                track_boxes_x = np.expand_dims(track_boxes_x, axis=1)
            else:
                gt_boxes_x = np.pad(gt_boxes_x, (0,(len(track_boxes_x)-len(gt_boxes_x))), 'constant', constant_values = (0,0))
                gt_boxes_x = gt_boxes_x[:,0]
                gt_boxes_x = np.expand_dims(gt_boxes_x, axis=1)
        
        error = gt_boxes_x - track_boxes_x
        error_dict[seq] = error
    return error_dict

seq_list = [
            'person1',
            'person2_1', 
            'person2_2', 
            'person3',
            'person4_1',
            'person4_2',
            'person5_1',
            'person6',
            'person7_1',
            'person7_2',
            'person8_1',
            'person8_2',
            'person9',
            'person10',
            'person11',
            'person12_1',
            'person12_2',
            'person13',
            'person14_1',
            'person14_2',
            'person15',
            'person16',
            'person17_1',
            'person17_2',
            'person18',
            'person19_1',
            'person19_2',
            'person19_3',
            'person20',
            'person21',
            'person22',
            'person23'
            ]
dataset = 'uav123'
tracker = 'CSRDCF-CH'
error_dict = get_tracking_error_dict(dataset, seq_list, tracker)

## vstack all the errors ##
complete_error_vect = np.ones((1,1))
count = 0
for key in error_dict.keys():
    complete_error_vect = np.vstack((complete_error_vect, error_dict[key]))
    count += error_dict[key].shape[0]

## filter
# rand_num = np.random.Generator.integers(200, 300, size=(1,1), dtype=np.int64, endpoint=False)
# print(rand_num)

condition1 = complete_error_vect < 324
complete_error_vect = complete_error_vect[condition1]

condition2 = complete_error_vect > -232
complete_error_vect = complete_error_vect[condition2]

## std deviation and variance ##
print("variance:\t",np.nanvar(complete_error_vect))
print("std_dev:\t", np.sqrt(np.nanvar(complete_error_vect)))
print("mean:\t\t", np.nanmean(complete_error_vect))

## plot ##
ax = sns.histplot(complete_error_vect, bins=500, stat='density', kde=True)
ax.set(xlabel='Number of Pixels', ylabel='Frequency', title='Centre tracking error PDF for CSRDCF')
plt.show()