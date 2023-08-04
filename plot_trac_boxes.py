import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def modify_gt_bbox(gt_bbox):
    # the gt bbox for VOT2016_dataset has eight values : format [x1,y1, x2, y2]
    for i in range(len(gt_bbox)):
        gt_bbox[i] = gt_bbox[i].replace('\n', '')
        gt_bbox[i] = gt_bbox[i].split(',')
        for j in range(len(gt_bbox[i])):
            gt_bbox[i][j] = float(gt_bbox[i][j])
        gt_bbox[i] = [gt_bbox[i][0], gt_bbox[i][1], gt_bbox[i][4]-gt_bbox[i][0], gt_bbox[i][5]-gt_bbox[i][1]]
        #gt_bbox[i] = gt_bbox[i][0:4]
        
    return gt_bbox


datasets = ["UAV123", "DTB70", "visdrone", "VOT2016"]
dataset = datasets[3]

if dataset == datasets[0]:
    track_output_base_path = "UAV123/CFtracker_output_opencv2/not_raw"
    seq = "person15"
    img_seq_path = os.path.join("/media/dtu-project2/2GB_HDD/UAV123", seq)
    ground_truth_path = os.path.join("/media/dtu-project2/2GB_HDD/UAV123/tracker_benchmark_v1.1/anno", seq+".txt")

if dataset == "VOT2016":
    track_output_base_path = "VOT2016/CFtracker_output_opencv/"
    seq = "bag"
    img_seq_path = os.path.join('/media/dtu-project2/2GB_HDD/object_tracker/vot-workspace/sequences', seq)
    ground_truth_path = os.path.join(img_seq_path,"groundtruth.txt")
    img_seq_path = os.path.join(img_seq_path, 'color')

elif dataset == datasets[1]:
    track_output_base_path = "DTB70/CFtracker_output_opencv/not_raw"
    seq = "Walking"
    img_seq_path = os.path.join("/media/dtu-project2/2GB_HDD/DTB70/dataset", seq, "img")
    ground_truth_path = os.path.join("/media/dtu-project2/2GB_HDD/DTB70/dataset", seq, "groundtruth_rect.txt")

elif dataset == datasets[2]:
    track_output_base_path = "visdrone/CFtracker_output_opencv/not_raw"
    seq = "uav0000068_01488_s"
    img_seq_path = os.path.join("/media/dtu-project2/2GB_HDD/visdrone/Visdrone_combined/sequences", seq)
    ground_truth_path = os.path.join("/media/dtu-project2/2GB_HDD/visdrone/Visdrone_combined/annotations", seq+".txt" )

trackers = ["KCF-CH", "KCF-kalman-CH", "KCF-kalman-detector-CH", "CSRDCF-CH", "CSRDCF-kalman-CH",\
     "CSRDCF-kalman-detector-CH"]
trackers = ["KCF-CH", "KCF-kalman-CH", "KCF-kalman-detector-CH"]
trackers = ["CSRDCF-CH", "CSRDCF-kalman-CH","CSRDCF-kalman-detector-CH"]
trackers = []

tracker_colors = {"KCF-CH":"yellow",
                "KCF-kalman-CH": "green",
                "KCF-kalman-detector-CH":"blue",
                "CSRDCF-CH":"violet",
                "CSRDCF-kalman-CH":"orange",
                "CSRDCF-kalman-detector-CH":"red"
                }

'''
tracker_colors = {"KCF-CH":"yellow",
                "KCF-kalman-CH": "green",
                "KCF-kalman-detector-CH":"blue",
                "CSRDCF-CH":"violet",
                "CSRDCF-kalman-CH":"orange",
                "CSRDCF-kalman-detector-CH":"red"
                }'''

box_color = {"yellow":(0,255,255), 
            "green":(0,255,0), 
            "blue":(255,0,0), 
            "violet":(255, 0, 127), 
            "orange":(0, 165, 255), 
            "red":(0,0,255)
            }

test_out_dir = os.path.join(track_output_base_path,seq)
print(test_out_dir)
if not os.path.isdir(test_out_dir):
    os.mkdir(test_out_dir)
    print(os.path.isdir(test_out_dir))

img_list = os.listdir(img_seq_path)
img_list.sort()

## GT Annotations
annotations_file_h = open(ground_truth_path, 'r')
annotations = annotations_file_h.readlines()
annotations_file_h.close()
#print(annotations)
if dataset == "VOT2016":
    annotations = modify_gt_bbox(annotations)
#print(annotations)

#####
tracker_boxes = {}
for tracker in trackers:
    tracked_bbox_path = os.path.join(track_output_base_path, tracker, seq+".txt")
    tracked_bbox_fh = open(tracked_bbox_path, "r")
    tracked_bbox = tracked_bbox_fh.readlines()
    tracked_bbox_fh.close()
    tracker_boxes[tracker] = tracked_bbox
    #print(tracker, ':\t',len(tracked_bbox))


for i in range(len(img_list)):
    image = cv2.imread(os.path.join(img_seq_path, img_list[i]))
    anno = annotations[i]
    if not dataset == "VOT2016":
        anno = anno.replace("\n", "")
        anno = anno.split(",")
    if anno[0] != "NaN":
        anno = [int(float(anno[0])), int(float(anno[1])), int(float(anno[2])), int(float(anno[3]))]
        ############### draw ground truth rectangular boxes #########################
        cv2.rectangle(image, (anno[0], anno[1]), (anno[0]+anno[2], anno[1]+anno[3]), color=(255,255,255),\
            thickness=1)
    gt_center = (int(anno[0]+anno[2]/2), int(anno[1]+anno[3]/2))
    
    ############## draw rectangular boxes of trackers ###########################
    #print(image.shape)
    for tracker in trackers:
        boxes = tracker_boxes[tracker]
        bbox = boxes[i]
        bbox = bbox.replace("\n", "")
        bbox = bbox.split(",")
        if bbox[0] != "NaN":
            bbox = [int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))]
            color_key = tracker_colors[tracker]                    
            color = box_color[color_key]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color=color,\
                thickness=2)
    #w = 100
    #image = image[max(gt_center[1]-w, 0):min(gt_center[1]+w, image.shape[0]), \
    #    max(gt_center[0]-w, 0):min(gt_center[0]+w, image.shape[1])]
    cv2.imshow("frame", image)

    # save image
    #if i%10 == 0:
    #    cv2.imwrite(os.path.join(test_out_dir, img_list[i]), image)

    key  = cv2.waitKey(1) or 0xff                      
    if key == 27:
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()