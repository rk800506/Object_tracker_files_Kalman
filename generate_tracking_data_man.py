import numpy as np
import os

def store_generated_track_bbox(trac_array, file_name):
    fh = open(file_name, 'w')
    for i in range(trac_array.shape[0]):
        #print(trac_array[i])
        bbox_str = str(trac_array[i][0])+','+str(trac_array[i][1])+','+str(trac_array[i][2])+','+str(trac_array[i][3])
        fh.write(bbox_str+'\n')
        #print(bbox_str)
    fh.close()
    '''if fh is not open:
        print('wrote successfully')
    else:
        print('some error')'''


def get_gt_annots(seq_name, dataset):
    ## visdrone ------------
    #gt_anno_file_path = '/media/dtu-project2/2GB_HDD/visdrone/VisDrone2019-SOT-train_part2/VisDrone2019-SOT-train/annotations/'+ seq_name + '.txt'
    #gt_anno_file_path = "/media/dtu-project2/2GB_HDD/DTB70/dataset/" + seq_name + '/groundtruth_rect.txt'

    # uav123 ------------------
    gt_anno_file_path = '/media/dtu-project2/2GB_HDD/UAV123/tracker_benchmark_v1.1/anno/'+ seq_name + '.txt'

    gt_annots_fh = open(gt_anno_file_path, 'r')
    gt_annots = gt_annots_fh.readlines()

    # read gt and store in nd array----##
    for i in range(len(gt_annots)):
        gt_annots[i] = gt_annots[i].replace('\n', '')
        gt_annots[i] = gt_annots[i].split(",")
        for j in range(len(gt_annots[i])):
            gt_annots[i][j] = float(gt_annots[i][j])
    gt_annots = np.array(gt_annots)
    return gt_annots

def append_track_outs(gt_annots, track_outs):
    if gt_annots.shape[0]>track_outs.shape[0]:
        append_array_size = (gt_annots.shape[0]-track_outs.shape[0], gt_annots.shape[1])
        track_outs = np.vstack((track_outs, np.zeros(append_array_size)))
    else:
        track_outs = track_outs[0:gt_annots.shape[0], :]
    return track_outs

datasets = ['uav123', 'visdrone', 'DTB70']
dataset = datasets[0]

## ------------- Visdrone ------------##
#seq_name = 'person1'

#print(gt_annots)

# read the tracked output #
tracker_names = ['KCF-CH', 'KCF-new', 'KCF', "CSRDCF", "CSRDCF-new", 'CSRDCF-CH']
traccker_with_kalman_detector = ['KCF-kalman', 'KCF-kalman-detector', 'CSRDCF-kalman', 'CSRDCF-kalman-detector']
#tracker_name = traccker_with_kalman_detector[3]
tracker_names2 = ['KCF','KCF-kalman', 'KCF-kalman-detector',"CSRDCF",'CSRDCF-kalman','CSRDCF-kalman-detector']

## for UAV123 only 33
seq_file_fh = open('/media/dtu-project2/2GB_HDD/object_tracker/UAV123/CFtracker_output_opencv2/relevant_seqs.txt', 'r')
seq_lists = seq_file_fh.readlines()
for i in range(len(seq_lists)):
    seq_lists[i] = seq_lists[i].replace('\n','')
print(seq_lists)

#for seq_name in seq_lists:

for tracker_name in tracker_names2:
    for seq_name in seq_lists:
        gt_annots = get_gt_annots(seq_name, dataset) 

        if dataset == 'visdrone':
            tracker_out_folder = '/media/dtu-project2/2GB_HDD/object_tracker/visdrone/CFtracker_output_opencv/R/' + tracker_name
            tracker_out_file = os.path.join(tracker_out_folder,seq_name + '.txt')
        if dataset == 'DTB70':
            tracker_out_folder = '/media/dtu-project2/2GB_HDD/object_tracker/DTB70/CFtracker_output_opencv/' + tracker_name
            tracker_out_file = os.path.join(tracker_out_folder,seq_name + '.txt')
        if dataset == 'uav123':
            tracker_out_folder = '/media/dtu-project2/2GB_HDD/object_tracker/UAV123/CFtracker_output_opencv2/R/' + tracker_name
            tracker_out_file = os.path.join(tracker_out_folder,seq_name + '.txt')

        trac_out_fh = open(tracker_out_file, 'r')
        trac_outs = trac_out_fh.readlines()
        
        # read gt and store in nd array----##
        for i in range(len(trac_outs)):
            trac_outs[i] = trac_outs[i].replace('\n', '')
            trac_outs[i] = trac_outs[i].split(",")
            for j in range(len(trac_outs[i])):
                trac_outs[i][j] = int(float(trac_outs[i][j]))
        trac_outs = np.array(trac_outs)

        if gt_annots.shape[0] != trac_outs.shape[0]:
            trac_outs = append_track_outs(gt_annots, trac_outs)
    
        if gt_annots.shape != trac_outs.shape:
           print(seq_name,tracker_name,'\tshapes:\t',gt_annots.shape, trac_outs.shape)
        #print(trac_outs)
        
        # diff between the vectors ##
        avg0_50 =  (gt_annots + trac_outs)//2
        avg0_75  = (trac_outs+avg0_50)//2
        #avg3 = (avg1 + avg2)//2
        avg0_875 = (trac_outs+avg0_75)//2
        avg0_625 = (avg0_50+avg0_75)//2
        #print(avg1)
        #print(avg2)

        avg_out_file_name = os.path.join(tracker_out_folder+'avg0.75', seq_name + '.txt')
        if not os.path.isdir(tracker_out_folder+'avg0.75'):
            os.mkdir(tracker_out_folder+'avg0.75')
        store_generated_track_bbox(avg0_75, avg_out_file_name)

        avg_out_file_name2 = os.path.join(tracker_out_folder+'avg0.875', seq_name + '.txt')
        if not os.path.isdir(tracker_out_folder+'avg0.875'):
            os.mkdir(tracker_out_folder+'avg0.875')
        store_generated_track_bbox(avg0_875, avg_out_file_name2)

        avg_out_file_name3 = os.path.join(tracker_out_folder+'avg0.50', seq_name + '.txt')
        if not os.path.isdir(tracker_out_folder+'avg0.50'):
            os.mkdir(tracker_out_folder+'avg0.50')
        store_generated_track_bbox(avg0_50, avg_out_file_name3)

        avg_out_file_name4 = os.path.join(tracker_out_folder+'avg0.625', seq_name + '.txt')
        if not os.path.isdir(tracker_out_folder+'avg0.625'):
            os.mkdir(tracker_out_folder+'avg0.625')
        store_generated_track_bbox(avg0_625, avg_out_file_name4)

    print('Done:\t',tracker_name)
