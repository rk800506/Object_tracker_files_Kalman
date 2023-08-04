import cv2
import sys
import os
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)
print('major_ver', major_ver, 'minor_ver', int(minor_ver))

## -------------------------- HOMOGRAPHy -----------------------###
## camera homography initiation
orb = cv2.ORB_create(nfeatures=200)
brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

def get_cam_homography(img1, img2, orb):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = brute_force.match(des1,des2)
    # finding the humming distance of the matches and sorting them
    matches = sorted(matches,key=lambda x:x.distance)
    #matches = BF_FeatureMatcher(des1, des2)
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    #M,_ = cv2.findHomography(src_pts, dst_pts)
    if len(matches) < 8:
        print('length_matches:\t',len(matches))
        M = np.array([[1,0,0], [0,1,0], [0,0,1]])
        return M
    return M

'''
def limit_bbox_size(bbox, img_h, img_w):
    bbox = list(bbox)
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[2] > img_w:
        bbox[2] = img_w
    if bbox[3] > img_h:
        bbox[3] = img_h
    return tuple(bbox)
'''
## ------------------- ------------- ---------------------------###

if __name__ == '__main__' :
    print('haha')

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
    tracker_type = tracker_types[5]

    #tracker = cv2.Tracker_create(tracker_type)
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
        print('i am here')
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    
    use_cam_homography = False
    if use_cam_homography:
        tracker_type_new = tracker_type + '_new'
        tracker_type = tracker_type + '_CH'

    # choose which dataset to use and the first frame
    datasets = ["UAV123", "VisDrone", "DTB70", 'VOT2016']
    dataset = datasets[2]
    if dataset == "UAV123":
        tracking_save_pt = "UAV123/CFtracker_output_opencv2"
        first_frame = "000001.jpg"
    elif dataset == "VisDrone":
        tracking_save_pt = "visdrone/CFtracker_output_opencv"
        first_frame = "img0000001.jpg"
    elif dataset == "DTB70":
        tracking_save_pt = "DTB70/CFtracker_output_opencv"
        first_frame = "00001.jpg"
        img_dataset_base_path  = "/media/dtu-project2/2GB_HDD/DTB70/dataset"
    elif dataset == "VOT2016":
        tracking_save_pt = "VOT2016/CFtracker_output_opencv"
        first_frame = "00000001.jpg"
        img_dataset_base_path = '/media/dtu-project2/2GB_HDD/object_tracker/vot-workspace/sequences'
        gt_bbox_base_path = img_dataset_base_path

    # ------------UNCOMMENT the line below to work with UAV123 dataset--------------------------#
    # seq = 'person19_1'
    # img_dir = "/media/dtu-project2/2GB_HDD/UAV123/"+seq

    #------------ UNCOMMENT the following code part to work with visdrone dataset----------------#
    # seq = "sequences/uav0000016_00000_s"
    # img_dir = "/media/dtu-project2/2GB_HDD/visdrone/VisDrone2019-SOT-train_part1/VisDrone2019-SOT-train/"+seq
    
    
    # ------------UNCOMMENT below section to work with DTB-70 and VOT2016 datasets---------------#
    seq = 'Wakeboarding2'
    if dataset == "DTB70":
        img_dir = os.path.join(img_dataset_base_path, seq, 'img')
    elif dataset == "VOT2016":
        img_dir = os.path.join(img_dataset_base_path, seq, 'color')
        gt_file = os.path.join(gt_bbox_base_path, 'groundtruth.txt')

    img_list = os.listdir(img_dir)
    img_list.sort()

    frame = cv2.imread(os.path.join(img_dir,first_frame))
    print(os.path.join(img_dir,first_frame))
    previous_frame = frame.copy()
    #frame = cv2.resize(frame, (640,360))

    ## initiate CSRT tracker
##    tracker2 = cv2.TrackerCSRT_create()

    cv2.namedWindow("frame")
##    cv2.namedWindow("CSRT")

    try:
        init_bbox = cv2.selectROI("frame", frame)
##        init_bbox = cv2.selectROI("CSRT", frame)
    except:
        exit()

    # Initialize tracker with first frame and bounding box
    
    ok = tracker.init(frame, init_bbox)
    #--------------------------------------------------------------------
    previous_trac_center = np.expand_dims(np.array([init_bbox[0]+init_bbox[2]//2, init_bbox[1]+init_bbox[3]//2, 1]), axis=1)
    #previous_trac_center = np.expand_dims(previous_trac_center, axis=1)
    #print('prev_trac_shaepe:\t', previous_trac_center.shape)
    #print(previous_trac_center)
##    print(init_bbox)
##    ok2 = tracker2.init(frame, init_bbox)
    print(ok)


    ### ....... file to record tracking bounding box .......... ####
    
    if use_cam_homography:
        track_fold_new = os.path.join(tracking_save_pt, tracker_type_new)
        track_fold_ch = os.path.join(tracking_save_pt, tracker_type)

        if not os.path.isdir(track_fold_new):
            os.mkdir(track_fold_new)
        if not os.path.isdir(track_fold_ch):
            os.mkdir(track_fold_ch)

        f_name = os.path.join(track_fold_new, seq+'.txt')
        f_name_2 = os.path.join(track_fold_ch, seq+'.txt')
        trackbox_writer_new = open(f_name_2, 'w')
        trackbox_writer = open(f_name, 'w')
    else:
        f_name = tracking_save_pt + '/'+tracker_type + '/' + seq+'.txt'
        if not os.path.isdir(tracking_save_pt + '/'+tracker_type):
            os.mkdir(tracking_save_pt + '/'+tracker_type)
        trackbox_writer_new = open(f_name, 'w')

    #while True:
    for img in img_list:
        # Read a new frame
        img_path = os.path.join(img_dir, img)

        #ok, frame = video.read()
        frame = cv2.imread(img_path)
        (h, w, c) = frame.shape
        #frame = cv2.resize(frame, (640,360))
        '''
        if not ok:
            cv2.destroyAllWindows()
            break
        '''
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)
        # print('bbox \n', bbox)

        #--- a problem encountered with goturn tracker is that, the bbox is dynamically becoming very
        # very larger and causing low memory error in the system. This is unrealistic situation and
        # thus limiting the size of bounding box.
        # if tracker_type == "GOTURN":
        #    bbox = limit_bbox_size(bbox, h, w)

        #-------- Store the original tacker reponse without homography--#
        bb2 = str(bbox)
        bb2 = bb2.replace('(','')
        bb2 = bb2.replace(')','')
        trackbox_writer_new.write(str(bb2)+'\n')

        
        # ---------------------------------------------------------- homography ------------------------------------------------------------- #
        if use_cam_homography:
            current_trac_center = np.expand_dims(np.array([bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2, 1], dtype=np.int16), axis=1)
            cv2.circle(frame, center=(int(bbox[0]+bbox[2]//2), int(bbox[1]+bbox[3]//2)), radius=10, color=(255,0,0), thickness=5)
            print('current_center:\n', current_trac_center)
            print('Previous_center:\n', previous_trac_center)
            delta_center = current_trac_center - previous_trac_center
            #print('delta_center:\n', delta_center)
            # calc homography
            Homography_matrix = get_cam_homography(previous_frame, frame, orb)
            #print('homography_matrix:\n', Homography_matrix)
            #print('src_pts:\t',src_pts)
            #print('dst_pts:\t', dst_pts)
            # update the current track center
            #print('homo*previous:\t', np.matmul(Homography_matrix, previous_trac_center))
            Pre_trac_center_with_homo = np.matmul(Homography_matrix, previous_trac_center)
            Pre_trac_center_with_homo = Pre_trac_center_with_homo/Pre_trac_center_with_homo[2]
            Pre_trac_center_with_homo[2] = np.ceil(Pre_trac_center_with_homo[2])
            #print('homo*previous:\t',Pre_trac_center_with_homo)
            """ if Pre_trac_center_with_homo[2][0] != 1.0:
                print('lal lallallal')
                break """
            current_trac_center = Pre_trac_center_with_homo + delta_center
            #current_trac_center = current_trac_center/current_trac_center[2]
            current_trac_center = current_trac_center.astype('int32')
            #print('updated_current_center:\n', current_trac_center)

            previous_trac_center = current_trac_center
            previous_frame = frame.copy()

            bbox = (float(int(current_trac_center[0][0]-bbox[2]//2)), float(int(current_trac_center[1][0]-bbox[3]//2)), bbox[2], bbox[3])
            if bbox[2] == 0.0 or bbox[3] == 0:
                bbox = (0.0,0.0,0.0,0.0)
            cv2.circle(frame, center=(int(bbox[0]+bbox[2]//2), int(bbox[1]+bbox[3]//2)), radius=10, color=(0,0,255), thickness=5)
            print('Final bbox:\n', bbox)
            print('\n\n')

            #------ modify and store tracked boxes --------#
            bb = str(bbox)
            bb = bb.replace('(','')
            bb = bb.replace(')','')
            trackbox_writer.write(str(bb)+'\n')
        # ----------------------------------------------------------------------------------- #

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        #print(fps)

        # Draw bounding box

##        if ok2:
##            # Tracking success
##            p1 = (int(bbox_CSRT[0]), int(bbox_CSRT[1]))
##            p2 = (int(bbox_CSRT[0] + bbox_CSRT[2]), int(bbox_CSRT[1] + bbox_CSRT[3]))
##            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
##        else :
##            # Tracking failure
##            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
##
##        # Display tracker type on frame
##        cv2.putText(frame, "CSRT" + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
##    
##        # Display FPS on frame
##        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
##
##
##        # Display result
##        cv2.imshow("CSRT", frame)
        
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv2.imshow("frame", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 :
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    # trackbox_writer.close()
    trackbox_writer_new.close()
