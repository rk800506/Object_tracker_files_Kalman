import cv2
import sys
import math as m
import numpy as np
import os
import time

root = "/media/dtu-project2/2GB_HDD/object_tracker/CFtracker_output_opencv/CSRT-detector"
root2 = "/media/dtu-project2/2GB_HDD/object_tracker/CFtracker_output_opencv/CSRT-detector_corr"
folder = os.listdir(root)

for tracked_file_bbox in folder:
    tracker_anno_file = os.path.join(root, tracked_file_bbox)
    tracker_anno_file_corr = os.path.join(root2, tracked_file_bbox)

    f_trac = open(tracker_anno_file, 'r')
    f_trac2 = open(tracker_anno_file_corr, 'w')
    trac_lines = f_trac.readlines()
    #print(trac_lines[i])

    for i in range(len(trac_lines)):
        org_line = trac_lines[i]
        #trac_lines[i] = trac_lines[i].replace('\n','')
        trac_lines[i] = trac_lines[i].replace('[','')
        trac_lines[i] = trac_lines[i].replace(']','')
        #print(trac_lines[i])
        f_trac2.write(trac_lines[i])

    print(tracked_file_bbox, ".. done")
