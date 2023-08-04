import numpy as np
import os
import cv2

def get_seq_hw (seq_img_path):
    img_list = os.listdir(seq_img_path)
    img_path = os.path.join(seq_img_path,img_list[0] )
    img = cv2.imread(img_path)
    (h, w, c) = img.shape
    return (h,w)

def get_data_with_x_percent(img_seq, ds_seqs_dir, ds_annots_dir, dataset, p = 0.8):
    overall_data_percent_within_p = 0
    
        
    for seq in img_seq:
        
        count = 0
        seq = seq.replace('\n','')

        if dataset == 'dtb70':
            img_dir = os.path.join(ds_seqs_dir, seq, 'img')
            (h,w) = get_seq_hw(os.path.join(img_dir))
            img_list = os.listdir(img_dir)
            img_list.sort()
            bbox_list_fh = open(os.path.join(ds_annots_dir, seq, 'groundtruth_rect.txt'))
            bbox_lines = bbox_list_fh.readlines()
            exclude_seqs = ['ManRunning1', 'Surfing12']
            
        if dataset == 'uav123':
            (h,w) = get_seq_hw(os.path.join(ds_seqs_dir, seq))
            img_dir = os.path.join(ds_seqs_dir, seq)
            img_list = os.listdir(img_dir)
            img_list.sort()
            bbox_list_fh = open(os.path.join(ds_annots_dir, seq+'.txt'))
            bbox_lines = bbox_list_fh.readlines()
            exclude_seqs = ["person22", "person20", 'person14']
            
        if dataset == 'visdrone':
            (h,w) = get_seq_hw(os.path.join(ds_seqs_dir, seq))
            img_dir = os.path.join(ds_seqs_dir, seq)
            img_list = os.listdir(img_dir)
            img_list.sort()
            bbox_list_fh = open(os.path.join(ds_annots_dir, seq+'.txt'))
            bbox_lines = bbox_list_fh.readlines()
            exclude_seqs = ['uav0000304_00253_s', 'uav0000080_01680_s','uav0000071_01536_s',\
                            'uav0000070_04877_s','uav0000071_00816_s']
        
        for i in range(len(bbox_lines)):
            bbox_lines[i] = bbox_lines[i].replace('\n','')
            bbox_lines[i] = bbox_lines[i].split(',')
            for j in range(len(bbox_lines[i])):
                bbox_lines[i][j] = float(bbox_lines[i][j])

            ## see if the bbox cneter is inside a region
            center = ((bbox_lines[i][0]+bbox_lines[i][2])/2, (bbox_lines[i][1]+bbox_lines[i][3])/2)

            ### For X direction #### Width
            if (w*(1-p)/2)<= center[0] and \
               center[0]<=(w*(1+p)/2):
                count += 1
            elif np.isnan(center[0]):
                count+=1
                
        seq_Percent = count*100/len(img_list)
##        print(str(seq)+'\t\t', seq_Percent)

##        exclude_seqs = []
        if not seq in exclude_seqs:
            overall_data_percent_within_p += seq_Percent
    return(overall_data_percent_within_p/(len(img_seq)-len(exclude_seqs)))

##### --------- MAIN --------########################################

uav123_ds_seqs_dir = "/media/dtu-project2/2GB_HDD/UAV123"
uav123_ds_annots_dir = "/media/dtu-project2/2GB_HDD/UAV123/tracker_benchmark_v1.1/anno"

visdrone_ds_seqs_dir = "/media/dtu-project2/2GB_HDD/visdrone/Visdrone_combined/sequences"
visdrone_ds_annots_dir = "/media/dtu-project2/2GB_HDD/visdrone/Visdrone_combined/annotations"

dtb70_db = "/media/dtu-project2/2GB_HDD/DTB70/dataset"
#dtb70_img = os.path.join(dtb70_db, "img")
#dtb70_anno = os.path.join(dtb70_db, 'groundtruth_rect.txt')

uav123_seq_file = "/media/dtu-project2/2GB_HDD/object_tracker/uav123_seq.txt"
visdrone_seq_file = "/media/dtu-project2/2GB_HDD/object_tracker/visdrone/person_seqs_2.txt"
dtb70_seq_file = "/media/dtu-project2/2GB_HDD/object_tracker/DTB70/relevant_seqs.txt"

datasets = ["uav123", "visdrone", 'dtb70']

dataset = datasets[1] #---------- chose dataset ------#
p = 0.60

for p in np.linspace(p,0.95,int((0.95-p)/0.04)):
    if dataset == datasets[0]:## UAV123
        (h,w) = (720,1280)
        uav123_seq = open(uav123_seq_file, 'r')
        uav123_seq = uav123_seq.readlines()
        del uav123_seq[-1]
        overall_data_percent_within_p =  get_data_with_x_percent(\
            uav123_seq, \
            uav123_ds_seqs_dir, \
            uav123_ds_annots_dir, \
            dataset, \
            p)
        
    if dataset == datasets[1]:## visdrone
        (h,w) = (720,1280)  # this is variable and change in the function for this dataset: just a placeholder
        visdrone_seq = open(visdrone_seq_file, 'r')
        visdrone_seq = visdrone_seq.readlines()
        del visdrone_seq[-1]
        overall_data_percent_within_p =  get_data_with_x_percent(\
            visdrone_seq, \
            visdrone_ds_seqs_dir, \
            visdrone_ds_annots_dir, \
            dataset, \
            p)
        
    if dataset == datasets[2]:## DTB70
        (h,w) = (720,1280)
        dtb70_seq = open(dtb70_seq_file, 'r')
        dtb70_seq = dtb70_seq.readlines()
        del dtb70_seq[-1]
        overall_data_percent_within_p =  get_data_with_x_percent(\
            dtb70_seq, \
            dtb70_db, \
            dtb70_db, \
            dataset, \
            p)
    print('overall_data_percent_within_p\t'+dataset +'  p=' +str(p*100)+'\t', overall_data_percent_within_p)




        
