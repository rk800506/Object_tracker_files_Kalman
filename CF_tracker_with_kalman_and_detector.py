#! /usr/bin/env python3

import cv2
import sys
import os
import kalman.kalman as kalm
import numpy as np
from pyCFTrackers.cftracker.kcf import *
from pyCFTrackers.cftracker.csrdcf import *
#from skimage.feature import hog
import matplotlib.pyplot as plt
import time

# detector related imports
import torch
import torchvision
from torchvision import transforms
from torch.utils import tensorboard
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import vision.box_utils as box_utils


# detector initial setup
checkpoint_path = "/media/dtu-project2/2GB_HDD/jetson-inference/python/training/detection/ssd/models/uav123_person_mb1_ssd/mb1-ssd-Epoch-49-Loss-1.3428684696555138.pth"
class_file_path = "/media/dtu-project2/2GB_HDD/jetson-inference/python/training/detection/ssd/models/uav123_person_mb1_ssd/labels.txt"
model_arg = 'mb1-ssd'

class_names = [name.strip() for name in open(class_file_path).readlines()]
net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(checkpoint_path)
DEVICE = "cuda" if torch.cuda.is_available() else "CPU"
net.to(DEVICE)
net.eval()

transf = transforms.ToTensor()
prob_threshold = 0.15
cpu_device = torch.device("cpu")

img_width = 300
img_height = 300


## -------------------------- HOMOGRAPHy -----------------------###
## camera homography initiation
orb = cv2.ORB_create(nfeatures=20)
brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

def get_cam_homography(img1, img2, orb):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = brute_force.match(des1,des2)
    # finding the humming distance of the matches and sorting them
    matches = sorted(matches,key=lambda x:x.distance)
    #matches = BF_FeatureMatcher(des1, des2)
    matches = matches[0:10]
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10.0)
    return M

## ------------------- ------------- ---------------------------###

def process_boxes(boxes, scores, prob_threshold):
    global img_width, img_height
    width = img_width
    height = img_height
    boxes = boxes[0]
    scores = scores[0]
    boxes = boxes.to(cpu_device)
    scores = scores.to(cpu_device)
    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, scores.size(1)):
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs = box_utils.nms(box_probs, "hard",
                                    score_threshold=prob_threshold,
                                    iou_threshold=0.45,
                                    sigma=0.5,
                                    top_k=-1,
                                    candidate_size=200)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.size(0))
    if not picked_box_probs:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    picked_box_probs = torch.cat(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)
print('major_ver', major_ver, 'minor_ver', int(minor_ver))

def get_frame_vel(frame, prev_frame):
    if frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    if prev_frame.shape[2] == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,\
        0.5, 3, 15, 3, 5, 1.2, 0)
    x_mean = np.mean(flow[... , 0])
    y_mean = np.mean(flow[... , 1])
    return [x_mean, y_mean]


def crop_win_for_OF(curr_frame, prev_frame, win_size =64):
    [a, b, c] = curr_frame.shape
    curr_frame_corners = np.zeros((2*win_size, 2*win_size))
    prev_frame_corners = np.zeros_like(curr_frame)
    
    temp1 = curr_frame[0:win_size, 0:win_size, ...]
    temp2 = curr_frame[0:win_size, b-win_size:b, ...]
    temp4 = curr_frame[a-win_size:a, b-win_size:b, ...]
    temp3 = curr_frame[a-win_size:a, 0:win_size, ...]
   
    curr_frame_corners = cv2.hconcat([temp1, temp2, temp3, temp4])
    return curr_frame_corners

def detector_detect(frame):
    [h, w, c] =  frame.shape
    [sh, sv] = [h/img_height, w/img_width]
    frame = cv2.resize(frame, (img_height, img_width))
    tensor_img = transf(frame)
    tensor_img = tensor_img[None,:]
    tensor_img = tensor_img.to(DEVICE)
    with torch.no_grad():
        scores, boxes = net.forward(tensor_img)
    boxes, labels, probabilties = process_boxes(boxes, scores, prob_threshold)
    if (boxes.numel()):
        box = (boxes.numpy()[0])
        # resize box
        box = [int(box[0]*sv), int(box[1]*sh), int(box[2]*sv), int(box[3]*sh)]
        box = [box[0], box[1], box[2]-box[0],box[3]-box[1]]
        return box, True
    return [0.0, 0.0, 0.0, 0.0], False

detector_init = False
if __name__ == '__main__' :
    # Set up tracker
    f_resp =  open("trac_resp_conf.txt", 'w')

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
    tracker_type = tracker_types[6]

    if int(major_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
        tracker2 = cv2.Tracker_create(tracker_type)
        print('i am here')
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
            #tracker2 = cv2.TrackerKCF_create()
            #tracker = KCF(features='gray')
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == "CSRT":
            #tracker = CSRDCF()
            tracker = cv2.TrackerCSRT_create()
            #tracker2 = cv2.TrackerCSRT_create()
            
    #dd_p = "/media/dtu-project2/2GB_HDD/UAV123"
    #dd = os.listdir(dd_p)
    #dd.sort()
    #for item in dd:
    #    if item.startswith("person"):
    item = "person1"
    img_dir = "/media/dtu-project2/2GB_HDD/UAV123/"+item
    img_list = os.listdir(img_dir)
    img_list.sort()

    frame = cv2.imread(img_dir+'/'+img_list[0])
##    frame = cv2.resize(frame, (640,360))
    
    prev_frame = frame.copy()

    cv2.namedWindow("frame")
    
    if detector_init:
        init_bbox, box_detected = detector_detect(frame)
        init_bbox = [int(init_bbox[0]-init_bbox[2]*0.4), int(init_bbox[1]-init_bbox[3]*0.25), \
                    int(init_bbox[2]*1.8), int(init_bbox[3]*1.5)]
        if not box_detected:
            print("no object detcted by detector in 1st frame")
            try:
                init_bbox = cv2.selectROI("frame", frame)
            except:
                exit()
    else:
        try:
            init_bbox = cv2.selectROI("frame", frame)
        except:
            exit()

    

    # Initialize tracker with first frame and bounding box
    init_resp = tracker.init(frame, init_bbox)
    

    #Initialize KALMAN parameter
    h,w = 360, 640
    dt = 0.2
    [res_h, res_v] = kalm.map_res_to_cam(uav_height=8, \
                                    cam_orient=45, fov_h=72, fov_v=60, img_h=h, img_w=w)
    [sd_pos_mes, sd_vel, sd_acc] = kalm.init_kalman_params(res_h, res_v, update_time=dt, acc_max=4.536)
    [state_est, process_noise_cov_est] = kalm.initialize_state_cov_first(sd_vel, init_bbox)
    [A, C, Q, R] = kalm.initilize_state_mat(sd_acc, sd_pos_mes, update_time=dt)

    #print(state_est.shape, process_noise_cov_est.shape)

##    ### ....... file to record tracking bounding box .......... ####
    tracker_type = 'CSRT-detector'
    f_name = '/media/dtu-project2/2GB_HDD/object_tracker/CFtracker_output_opencv/'+tracker_type + '/' + os.path.basename(img_dir)+'.txt'
    
    trackbox_writer = open(f_name, 'w')

    resp = []
    dur = 0.04
    count = 1
    ROI = (0, 0, 0, 0)
    win_size_of = 32
    prev_cropped_frame = np.zeros((win_size_of, 4*win_size_of, 1))
    max_resp_prev = init_resp

    detector_st_time = time.time()
    Dbox = [0, 0, 0, 0]
    for img in img_list:
        
        # Read a new frame
        img_path = os.path.join(img_dir, img)

##        #ok, frame = video.read()
        frame = cv2.imread(img_path)

        #detector get detection box
##        Dbox, box_detected = detector_detect(frame)

        # Start timer
        timer = cv2.getTickCount()

        #### kalman state estimate correction using optical flow velocity ####
        #state_est = state_est - np.array([[0], [0], [], []])

        #### integrating kalman predicts.. PREDICT
        #print('?????????????????????? ', state_est.shape, '?????????????????????????')
        [pos_pred, state_pred, process_noise_cov_pred] = \
                kalm.kalman_predict(A, C, Q, state_est, process_noise_cov_est)
        #print(pos_pred, state_pred, '...................')
        init_scale = 1
        
        #print('roi \t', ROI)
        #print('pred pos: \t', pos_pred, '\n')
        #print('pred state: \t', state_pred, '\n')
        
        
        ## ROI calculation
        ROI = (int(pos_pred[0][0]- init_bbox[2]*init_scale/2), int(pos_pred[1][0]-init_bbox[3]*init_scale/2), \
            int(init_scale*init_bbox[2]), int(init_scale*init_bbox[3]))

        arrow_magni = 1
        #frame = cv2.arrowedLine(frame, (int(state_pred[0][0]), int(state_pred[1][0])),\
        #     (int(state_pred[0][0]+state_pred[2][0]*arrow_magni), \
        #      int(state_pred[1][0]+state_pred[3][0]*arrow_magni)), \
        #         color=(0, 255, 0), thickness=1)
        


        #bbox = ROI
        #roi_pose = [state_pred[0][0] + state_pred[2][0]*dur*10, state_pred[1][0] + state_pred[3][0]*dur*10]
            
        #ROI = (int(roi_pose[0]- init_bbox[2]*init_scale/2), int(roi_pose[1]-init_bbox[3]*init_scale/2), \
        #       int(init_scale*init_bbox[2]), int(init_scale*init_bbox[3]))
        
        # Update tracker
        #bbox, max_resp, ok = tracker.update(frame, manipulate_roi = [False, ROI])
        
        start_time = time.time()
        ok, bbox = tracker.update(frame)
##        bbox, max_resp = tracker.update(frame)
        end_time = time.time()
        dur = end_time - start_time
        fps = 1/dur
        #print(bbox)
##        resp.append(max_resp)
##
##        if abs(max_resp - max_resp_prev) > 0.25:
##            ok = False
##        else:
##            ok = True
##        print(abs(max_resp - max_resp_prev))    
##        max_resp_prev = max_resp
        
        #print('max -reso ', max_resp, '\t', img)



        print('ok: ', ok)

        if time.time()-detector_st_time > 5.0:  # re initiate tracker with kalam roi
            
            init_bbox, box_detected = detector_detect(frame)
            Dbox = [int(init_bbox[0]-init_bbox[2]*0.35), int(init_bbox[1]-init_bbox[3]*0.15), \
                    int(init_bbox[2]*1.7), int(init_bbox[3]*1.3)]
            if box_detected:
                tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(frame, Dbox)
                print('tracker reinitiated')
            detector_st_time = time.time()
            print(ok)
            
        if ok :
            ## update/correct kalman
            #print('inside okay')
            pos_mes = np.array([[int(bbox[0] + bbox[2]/2)],[int(bbox[1] + bbox[3]/2)]])

            # correct measured position
            #pos_mes = pos_mes - np.array([[x_mean_of], [y_mean_of]])
            
            [state_est, process_noise_cov_est] = \
                    kalm.kalman_correct(pos_mes, pos_pred, state_pred, process_noise_cov_pred, C, R)
            

            pos_est = []
            
        if ok== True or ok == False:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            pk1  = (int(pos_pred[0][0]-bbox[2]/2), int(pos_pred[1][0]-bbox[2]/2))
            pk2  = (int(pos_pred[0][0]+bbox[2]/2), int(pos_pred[1][0]+bbox[2]/2))

            #pk3  = (int(state_est[0][0]-bbox[2]/2), int(state_est[1][0]-bbox[2]/2))
            #pk4  = (int(state_est[0][0]+bbox[2]/2), int(state_est[1][0]+bbox[2]/2))

            #print(pk1)

            #print('...............................', pos_pred.shape, '\n')
            #print('...............................', state_pred.shape, '\n')
            #print(pk3[0]-pk1[0], pk3[1]-pk1[1])
            
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.rectangle(frame, pk1, pk2, (255,255,100), 3, 1)
            #cv2.rectangle(frame, pk3, pk4, (0,0,200), 3, 1)
            
            cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (255,255,255), 4, 1)
            if box_detected:
                cv2.rectangle(frame, (Dbox[0], Dbox[1]), (Dbox[0]+Dbox[2], Dbox[1]+Dbox[3]), (0,0,255), 2, 1)
            
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            count += 1
            bbox = (0.0, 0.0, 0.0, 0.0)
            
        
        bb = str(bbox)
        bb = bb.replace('(','')
        bb = bb.replace(')','')
        trackbox_writer.write(str(bb)+'\n')
        
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,250),2);
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,250), 2);

        # Display result
        cv2.imshow("frame", frame)

        ####
        prev_frame = frame.copy()
        
        

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 :
            cv2.destroyAllWindows()
            print(count)
            break
    cv2.destroyAllWindows()
    print(count)
    
    trackbox_writer.close()
    #plt.plot(resp)
    #plt.show()
