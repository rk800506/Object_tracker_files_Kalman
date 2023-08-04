import cv2
import sys
import os
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(cv2.__version__)
print('major_ver', major_ver, 'minor_ver', int(minor_ver))

def convert_bbox_to_4_point(bbox):
    # bbox format [x,y,h,w] to [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    (x,y,dx, dy) = bbox
    bbox_4_point = (x,y, x+dx,y, x+dx,y+dy, x,y+dx)
    return bbox_4_point

def get_first_bbox_from_gt(gt_file):
    gt_fh = open(gt_file)
    gts = gt_fh.readlines()
    first_line = gts[0]
    first_line = first_line.replace('\n','')
    first_line = first_line.split(',')
    for i in range(len(first_line)):
        first_line[i] = int(float(first_line[i]))

    (x1,y1) = (first_line[0],first_line[1])
    (x2,y2) = (first_line[2],first_line[3])
    (x3,y3) = (first_line[4],first_line[5])
    (x4,y4) = (first_line[6],first_line[7])

    (xmin, ymin) = (min(x1,x2,x3,x4), min(y1,y2,y3,y4))
    (xmax, ymax) = (max(x1,x2,x3,x4), max(y1,y2,y3,y4))

    fact = 0.25
    roi = (xmin-(fact/2)*(xmax-xmin),\
         ymin*(fact/2)*(ymax-ymin),\
              (xmax-xmin)*(1+fact),\
                   (ymax-ymin)*(1+fact))
    roi = (xmin-(xmax-xmin)*0.05, ymin-(ymax-ymin)*0.05,(xmax-xmin)*1.1, (ymax-ymin)*1.1)
    roi =list(roi)

    if roi[2] < 40.0:
        roi[2] = 40.0
    if roi[3] < 40.0:
        roi[3] = 40.0
    roi = tuple(roi)
    #roi = (first_line[0], first_line[1], first_line[4]-first_line[0], first_line[5]-first_line[1])
    return roi

## -------------------------- HOMOGRAPHy -----------------------###
## camera homography initiation
orb = cv2.ORB_create(nfeatures=200)
brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

def get_cam_homography(img1, img2, orb):

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)


    if type(des1) == type(None) or type(des2) == type(None):
        M = np.array([[1,0,0], [0,1,0], [0,0,1]])
        return M

    matches = brute_force.match(des1,des2)
    # finding the humming distance of the matches and sorting them
    matches = sorted(matches,key=lambda x:x.distance)
    #print(len(matches))
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    assert(len(src_pts) == len(dst_pts))
    #print(len(src_pts), len(dst_pts), len(kp1), len(kp2))

    M,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    if len(matches) < 8:
        print('length_matches:\t',len(matches))
        M = np.array([[1,0,0], [0,1,0], [0,0,1]])
        return M
    return M
## ------------------- ------------- ---------------------------###
if __name__ == '__main__' :
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
    tracker_type_raw = tracker_types[6]

    '''if int(minor_ver) < 3:
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
            tracker = cv2.TrackerCSRT_create()'''
    
    use_cam_homography = False       # Control to use homography or not.
    if use_cam_homography:
        tracker_type_new = tracker_type_raw + '_new'
        tracker_type = tracker_type_raw + '_CH'
        
    datasets = ["UAV123", "VisDrone", "DTB70", 'VOT2016']
    dataset = datasets[3]

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

    if dataset == "VOT2016":
        img_seq_list_path = '/media/dtu-project2/2GB_HDD/object_tracker/vot-workspace/sequences/list.txt'
    seq_list_fh = open(img_seq_list_path)
    seq_list = seq_list_fh.readlines()
    for i in range(len(seq_list)):
        seq_list[i] = seq_list[i].replace('\n','')
    
    #leave_seqs = ['birds2','godfather','nature','sheep']
    #for item in leave_seqs:
    #    seq_list.remove(item)
    #seq_list = ['octopus']

    for seq in seq_list:
        print('seq:\t', seq)
        if tracker_type_raw == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type_raw == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type_raw == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type_raw == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type_raw == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type_raw == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type_raw == "CSRT":
            tracker = cv2.TrackerCSRT_create()


        if dataset == "DTB70":
            img_dir = os.path.join(img_dataset_base_path, seq, 'img')
        elif dataset == "VOT2016":
            img_dir = os.path.join(img_dataset_base_path, seq, 'color')
            gt_file = os.path.join(gt_bbox_base_path, seq, 'groundtruth.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        frame = cv2.imread(os.path.join(img_dir,first_frame))
        previous_frame = frame.copy()
        cv2.namedWindow("frame"+seq)

        if dataset == "VOT2016":
            init_bbox = get_first_bbox_from_gt(gt_file)
            if init_bbox[2] < 0 or init_bbox[3]<0:
                print(init_bbox,'\n')
                print(os.path.join(img_dir,first_frame))

        # Initialize tracker with first frame and bounding box
        #print(init_bbox)
        
        '''try:
            init_bbox = cv2.selectROI("frame"+seq, frame)
            print(init_bbox, '\tinit_bbox')
        except:
            exit()'''
        
        ok = tracker.init(frame, init_bbox)
        #--------------------------------------------------------------------
        previous_trac_center = np.expand_dims(np.array([init_bbox[0]+init_bbox[2]//2, \
            init_bbox[1]+init_bbox[3]//2, 1]), axis=1)
        #print(ok)
        #print('running seqs:\t', seq)

        ### ....... file to record tracking bounding box .......... ####
        track_fold_new = os.path.join(tracking_save_pt, tracker_type_new)
        track_fold_ch = os.path.join(tracking_save_pt, tracker_type)

        if not os.path.isdir(track_fold_new):
            os.mkdir(track_fold_new)
        if not os.path.isdir(track_fold_ch):
            os.mkdir(track_fold_ch)

        f_name = os.path.join(track_fold_new, seq + '.txt')
        f_name_2 = os.path.join(track_fold_ch, seq + '.txt')

        trackbox_writer = open(f_name, 'w')
        trackbox_writer_new = open(f_name_2, 'w')

        for img in img_list:
            img_path = os.path.join(img_dir, img)
            frame = cv2.imread(img_path)
    
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)

            if dataset == "VOT2016":
                bbox_4_pt = convert_bbox_to_4_point(bbox)    # convert to 4 points vot format
                bb2 = str(bbox_4_pt)
                bb2 = bb2.replace('(','')
                bb2 = bb2.replace(')','')
                trackbox_writer_new.write(str(bb2)+'\n')
            else:
                #-------- Store the original tacker reponse without homography--#
                bb2 = str(bbox)
                bb2 = bb2.replace('(','')
                bb2 = bb2.replace(')','')
                trackbox_writer_new.write(str(bb2)+'\n')

            # ------------------------------- homography ------------------------------------ #
            current_trac_center = np.expand_dims(np.array([bbox[0]+bbox[2]//2, \
                bbox[1]+bbox[3]//2, 1], dtype=np.int16), axis=1)
            cv2.circle(frame, center=(int(bbox[0]+bbox[2]//2), \
                int(bbox[1]+bbox[3]//2)), radius=10, color=(255,0,0), thickness=5)
            delta_center = current_trac_center - previous_trac_center
            Homography_matrix = get_cam_homography(previous_frame, frame, orb)
            Pre_trac_center_with_homo = np.matmul(Homography_matrix, previous_trac_center)
            Pre_trac_center_with_homo = Pre_trac_center_with_homo/Pre_trac_center_with_homo[2]
            Pre_trac_center_with_homo[2] = np.ceil(Pre_trac_center_with_homo[2])

            current_trac_center = Pre_trac_center_with_homo + delta_center
            current_trac_center = current_trac_center.astype('int32')
            previous_trac_center = current_trac_center
            previous_frame = frame.copy()

            bbox = (float(int(current_trac_center[0][0]-bbox[2]//2)), \
                float(int(current_trac_center[1][0]-bbox[3]//2)), bbox[2], bbox[3])
            if bbox[2] == 0.0 or bbox[3] == 0:
                bbox = (0.0,0.0,0.0,0.0)
            cv2.circle(frame, center=(int(bbox[0]+bbox[2]//2), int(bbox[1]+bbox[3]//2)), \
                radius=10, color=(0,0,255), thickness=5)

            if dataset == "VOT2016":
                bbox_4_pt = convert_bbox_to_4_point(bbox)    # convert to 4 points vot format
                bb2 = str(bbox_4_pt)
                bb2 = bb2.replace('(','')
                bb2 = bb2.replace(')','')
                trackbox_writer_new.write(str(bb2)+'\n')
            else: 
                bb = str(bbox)
                bb = bb.replace('(','')
                bb = bb.replace(')','')
                trackbox_writer.write(str(bb)+'\n')

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

            # Display result
            cv2.imshow("frame"+seq, frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 :
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
        trackbox_writer.close()
        trackbox_writer_new.close()