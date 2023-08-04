#!/usr/bin/env python3.6

import json
import os

json_file = "/media/dtu-project2/2GB_HDD/UAV123/UAV123.json"
dir_to_store_annots = "/media/dtu-project2/2GB_HDD/UAV123/annotations"

if not os.path.isdir(dir_to_store_annots):
    os.mkdir(dir_to_store_annots)

f = open(json_file, 'r')
data = json.load(f)
f.close()

for key in data.keys():
    gt_rect = data[key]['gt_rect']
    anno_file = os.path.join(dir_to_store_annots, key+'.txt')
    annot_f = open(anno_file, 'w')
    for coordinate in gt_rect:
        string = str(coordinate[0])+','+str(coordinate[1])+','+str(coordinate[2])+','+str(coordinate[3])
        annot_f.write(string+'\n')
    annot_f.close()