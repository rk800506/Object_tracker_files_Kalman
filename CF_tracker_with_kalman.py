#! /usr/bin/env python3
import cv2
import sys
import os
import kalman.kalman as kalm
import numpy as np
from pyCFTrackers.cftracker.kcf import *
from pyCFTrackers.cftracker.csrdcf import *
from pyCFTrackers.cftracker.mccth_staple import *
from pyCFTrackers.cftracker.config.mccth_staple_config import *
##from pyCFTrackers.cftracker.eco import *
from pyCFTrackers.cftracker.strcf import *

from pyCFTrackers.cftracker.staple import *
from pyCFTrackers.cftracker.config.staple_config import *
#from skimage.feature import hog
import matplotlib.pyplot as plt
import time


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)
print('major_ver', major_ver, 'minor_ver', int(minor_ver))


#def extract_hog_feature(img, cell_size=4):
    #fhog_feature=fhog(img.astype(np.float32),cell_size,num_orients=9,clip=0.2)[:,:,:-1]
    #cells_per_block = 4
    #orientations = 9
#    fhog_feature = hog(img, orientations=9, pixels_per_cell=cell_size, cells_per_block=4, block_norm='L2')
#    return fhog_feature


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
    x_mean = np.median(flow[... , 0])
    y_mean = np.median(flow[... , 1])
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


if __name__ == '__main__' :

    
    # Set up tracker
    f_resp =  open("trac_resp_conf.txt", 'w')

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', \
                     'CSRT', 'MCCH_staple', 'staple', 'strcf-hc']
    tracker_type = tracker_types[2]

    if int(major_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
        #tracker2 = cv2.Tracker_create(tracker_type)
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
            tracker = cv2.TrackerCSRT_create()
        if tracker_type == 'MCCH_staple':
            config = MCCTHOTBConfig()
            tracker = MCCTHStaple(config)
        if tracker_type == 'staple':
            config = StapleConfig()
            tracker = Staple(config)
        if tracker_type == 'strcf-hc':
            tracker = STRCF()
        
        # to use kalman and detector
        use_kalman = False
        use_kalam_detector= False

        if use_kalman:
            tracker_type = tracker_type+'Kalman'
        if use_kalam_detector:
            tracker_type = tracker_type+'KalmanDetector'
           
    # choose which dataset to use and the first frame
    datasets = ["UAV123", "VisDrone", "DTB70"]
    dataset = datasets[2]
    if dataset == "UAV123":
        tracking_save_pt = "CFtracker_output_opencv/"
        first_frame = "000001.jpg"
    elif dataset == "VisDrone":
        tracking_save_pt = "visdrone/CFtracker_output_opencv/"
        first_frame = "img0000001.jpg"
    elif dataset == "DTB70":
        tracking_save_pt = "DTB70/CFtracker_output_opencv/"
        first_frame = "00001.jpg"

    #img_dir = "/media/dtu-project2/2GB_HDD/UAV123/person23"
    if dataset == "VisDrone":
        seq = "uav0000349_02668_s"
        annot = seq+".txt"
        img_dir = "/media/dtu-project2/2GB_HDD/visdrone/VisDrone2019-SOT-train_part2/VisDrone2019-SOT-train/sequences/"+seq
        annotation_path = "/media/dtu-project2/2GB_HDD/visdrone/Visdrone_combined/annotations/"+annot
    if dataset == "DTB70":
        base_path= "/media/dtu-project2/2GB_HDD/DTB70/dataset"
        relevant_fold_names = "/media/dtu-project2/2GB_HDD/object_tracker/DTB70/relevant_seqs.txt"
    
    pers_seq_fh = open(relevant_fold_names, 'r')
    pers = pers_seq_fh.readlines()

    pers =["ManRunning2"]
    #person = pers[0]
    k_filt_wait = 3
    k_st = time.time()
    
    for person in pers:
        print(person)
        person = person.replace("\n","")
        img_dir = os.path.join(base_path, person, "img")
        annotation_path = os.path.join(base_path, person, "groundtruth_rect.txt")
        print(annotation_path)

        ########## ############ ################# ############
        #loop for one person
        img_list = os.listdir(img_dir)
        img_list.sort()
        annotation_fh = open(annotation_path, 'r')
        gt_annots = annotation_fh.readlines()
        #print(gt_annots)
        #print(len(gt_annots))
        annotation_fh.close()
        scale_read_gt = 1.3
        for i in range(len(gt_annots)):
            #print(len(gt_annots))
            gt_annots[i] = gt_annots[i].replace("\n", "")
            gt_annots[i] = gt_annots[i].split(",")
            #print(gt_annots[i])
            gt_annots[i] = [int(float(gt_annots[i][0]) - 0.5*(scale_read_gt-1.0)*float(gt_annots[i][2])), \
                int(float(gt_annots[i][1]) - 0.5*(scale_read_gt-1.0)*float(gt_annots[i][3])), \
                int(float(gt_annots[i][2])*scale_read_gt),\
                int(float(gt_annots[i][3])*scale_read_gt)]
            gt_annots[i]
        
        frame = cv2.imread(img_dir+'/'+img_list[0])
        #frame = cv2.resize(frame, (640,360))
        init_bbox = tuple(gt_annots[0])
        #print(init_bbox)
        
##        cv2.namedWindow("frame")
##        try:
##            init_bbox = cv2.selectROI("frame", frame)
##        except:
##            exit()

        # Initialize tracker with first frame and bounding box
        init_resp = tracker.init(frame, init_bbox)

        (h,w, c)  = frame.shape
        dt = 0.5
        uav_h = 8
        fov_h = 72
        [res_h, res_v] = kalm.map_res_to_cam(uav_height=uav_h, \
                                        cam_orient=45, fov_h=fov_h, fov_v=fov_h*w/h, img_h=h, img_w=w)
        [sd_pos_mes, sd_vel, sd_acc] = kalm.init_kalman_params(uav_h, res_h, res_v, \
            update_time=dt, acc_max=4.536, cam_orient=45)
        [state_est, process_noise_cov_est] = kalm.initialize_state_cov_first(sd_vel, init_bbox)
        [A, C, Q, R] = kalm.initilize_state_mat(sd_acc, sd_pos_mes, update_time=dt)


    ##    ### ....... file to record tracking bounding box .......... ####
        #tracker_type = 'strcf-hc-kalman'
        #f_name = '/media/dtu-project2/2GB_HDD/object_tracker/CFtracker_output_opencv/'+tracker_type + '/' + os.path.basename(img_dir)+'.txt
        f_name = tracking_save_pt + tracker_type + '/' + person +'.txt'
        trackbox_writer = open(f_name, 'w')

        resp = []
        dur = 0.04
        count = 1
        ROI = (0, 0, 0, 0)
        win_size_of = 32
        prev_cropped_frame = np.zeros((win_size_of, 4*win_size_of, 1))
        max_resp_prev = init_resp
        
        tracker_st = time.time()
        detector_init_frame = 150    #every 100 frames
        frame_count = 0
        for i in range(len(img_list)):
            frame_count += 1
            
            # Read a new frame
            img_path = os.path.join(img_dir, img_list[i])

    ##        #ok, frame = video.read()
            frame = cv2.imread(img_path)
    ##        frame = cv2.resize(frame, (640,360))
    ##
    ##        ## get cropped window
    ##        cropped_frm = crop_win_for_OF(frame, prev_frame, win_size_of)
    ##        #print(cropped_frm.shape)
    ##        # get frame velocity
    ##        [x_mean_of, y_mean_of] = get_frame_vel(cropped_frm, prev_cropped_frame)
    ##        prev_cropped_frame = cropped_frm
    ##
    ##
    ##        #of_p1 = (320, 180 + int(x_mean_of*10))
    ##        #of_p2 = (320 + , 180 + int(y_mean_of*10))
    ##        print(x_mean_of)
    ##
    ##        i = 5
    ##        of_p1 = (320, 180)
    ##        of_p2 = (320+int(x_mean_of*i), 180+int(y_mean_of*i))
    ##        cv2.arrowedLine(frame, of_p1, of_p2, (0,0,255),  thickness = 3)

            #cv2.imshow('cropped frae', cropped_frm)
            
            
            '''
            if not ok:
                cv2.destroyAllWindows()
                break
            '''
            # Start timer
            timer = cv2.getTickCount()

            #### kalman state estimate correction using optical flow velocity ####
            #state_est = state_est - np.array([[0], [0], [], []])

            #### integrating kalman predicts.. PREDICT
            #print('?????????????????????? ', state_est.shape, '?????????????????????????')
            [pos_pred, state_pred, process_noise_cov_pred] = \
                    kalm.kalman_predict(A, C, Q, state_est, process_noise_cov_est)
            #print(pos_pred,'\n', state_pred, '...................')
            init_scale = 1.15
                
            ## ROI calculation
            ROI = (int(pos_pred[0][0]- init_bbox[2]*init_scale/2), int(pos_pred[1][0]-init_bbox[3]*init_scale/2), \
                int(init_scale*init_bbox[2]), int(init_scale*init_bbox[3]))

            arrow_magni = 1
    ##        frame = cv2.arrowedLine(frame, (int(state_pred[0][0]), int(state_pred[1][0])),\
    ##             (int(state_pred[0][0]+state_pred[2][0]*arrow_magni), \
    ##              int(state_pred[1][0]+state_pred[3][0]*arrow_magni)), \
    ##                 color=(0, 255, 0), thickness=1)

            #bbox = ROI
    ##        roi_pose = [state_pred[0][0] + state_pred[2][0]*dur*10, state_pred[1][0] + state_pred[3][0]*dur*10]
    ##            
    ##        ROI = (int(roi_pose[0]- init_bbox[2]*init_scale/2), int(roi_pose[1]-init_bbox[3]*init_scale/2), \
    ##               int(init_scale*init_bbox[2]), int(init_scale*init_bbox[3]))

    ##        frame = cv2.arrowedLine(frame, (int(state_pred[0][0]), int(state_pred[1][0])),\
    ##             (int(state_pred[0][0]+state_pred[2][0]*arrow_magni), \
    ##              int(state_pred[1][0]+state_pred[3][0]*arrow_magni)), \
    ##                 color=(0, 255, 0), thickness=1)
            
            # Update tracker
            #bbox, max_resp, ok = tracker.update(frame, manipulate_roi = [False, ROI])
            
            start_time = time.time()
    ##        ok, bbox = tracker.update(frame)
    ##        bbox, max_resp = tracker.update(frame)
            kalman_center = (int(pos_pred[0][0]- init_bbox[2]*init_scale/2), int(pos_pred[1][0]-init_bbox[3]*init_scale/2))
    ##        bbox = tracker.update(frame, kalman_center)
            ok, bbox = tracker.update(frame)
            #print(bbox)
            end_time = time.time()
            dur = end_time - start_time
            fps = 1/dur

            # tracker reinitilization by detector
            #print("tracker_type:\t", tracker_type)
            if frame_count >= detector_init_frame and use_kalam_detector:
                if tracker_type == "KCFKalmanDetector" :
                    #tracker.clear()
                    tracker = cv2.TrackerKCF_create()
                    print("tracker reinitiated by detector")
                if tracker_type == "CSRTKalmanDetector":
                    #tracker.clear()
                    tracker = cv2.TrackerCSRT_create()
                    print("tracker reinitiated by detector")
                
                if gt_annots[i][0] != "NaN":
                    #print(i, gt_annots[i], type(gt_annots[i]), tuple(gt_annots[i]), type(gt_annots[i][0]))
                    ok = tracker.init(frame, tuple(gt_annots[i]))
                    frame_count = 0


            ## optical flow
            '''
            curr_frame = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

            #print(curr_frame.shape, prev_frame.shape)
            curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
            #print(curr_frame.shape, prev_frame.shape)
            
            
            [x_mean, y_mean] = get_frame_vel(curr_frame, prev_frame)
            
            ofp1 = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))
            ofp2 = (int(bbox[0]+bbox[2]/2 + x_mean*100), int(bbox[1]+bbox[3]/2 + y_mean*100))
            
            frame = cv2.arrowedLine(frame, ofp1, ofp2, color=(0, 255, 0), thickness=1)
            '''
            
            print('ok: ', ok)
            
            if ok == False and (use_kalman or use_kalam_detector):  # re initiate tracker with kalam roi
            #if (time.time() - k_st)>k_filt_wait and (use_kalman or use_kalam_detector):
                if tracker_type == 'KCFKalman' or "KCFKalmanDetector":
                #    #tracker.clear()
                    tracker = cv2.TrackerKCF_create()
                    #print(tracker)
                    print("tracker reinitiated by KCF kalman")
                    print(ROI)
                    if ROI[0] <0 or ROI[1]<0 or ROI[2]<0 or ROI[3]<0:
                        ROI = (100, 100, 20, 20)
                    ok = tracker.init(frame, ROI)
                '''
                if tracker_type == 'CSRTKalman' or "CSRTKalmanDetector":
                #    print(tracker_type == 'CSRTKalman' or "CSRTKalmanDetector")
                #    print(tracker_type)
                #    print("I am inside CSRT kalman and det")
                #    #tracker.clear()
                    tracker = cv2.TrackerCSRT_create()
                    print("tracker reinitiated by CSRT kalman")'''
                    
                
                #ok = tracker.update(frame, [True, ROI])
                '''
                print('count...', count)
                
                if count%3 == 0:
                    print('tracker reinitiated')
                    init_resp = tracker.init(frame, ROI)
                    max_resp_prev = init_resp
                #print('tracker updated')
                print(ok)
            print('........', bbox)
            '''

    ##        bbox = bbox[0]
                
            if ok :
                ## update/correct kalman
                #print('inside okay')
                pos_mes = np.array([[int(bbox[0] + bbox[2]/2)],[int(bbox[1] + bbox[3]/2)]])
                #print(pos_mes, '...................')

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
            #prev_frame = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
            #cv2.imshow('prev_frame', prev_frame)
            
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 :
                cv2.destroyAllWindows()
                #print(count)
                break
        cv2.destroyAllWindows()
        #print(count)
        
        trackbox_writer.close()
        print("at the end of ist fold")
    ##    plt.plot(resp)
    ##    plt.show()
    
