import cv2
import math as m
import numpy as np
import os
import time
import matplotlib.pyplot as plt

##img_dir = "../dataset/UAV123/person3"
##img_list = os.listdir(img_dir)
##img_list.sort()
##
##frame = cv2.imread(img_dir+'/'+img_list[0])
##frame = cv2.resize(frame, (640,360))


# Creates an image filled with zero
# intensities with the same dimensions 
# as the frame

def get_frame_vel(frame, prev_frame):
    if frame.shape[2] == 3:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    if prev_frame.shape[2] == 3:
        prev_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,\
        0.5, 3, 15, 3, 5, 1.2, 0)
    x_mean = np.mean(flow[... , 0])
    y_mean = np.mean(flow[... , 1])

    return [x_mean, y_mean]
    



##initiate  = False
##magnitude_opt = []
##
##for img in img_list:
##    st = time.time()
##    path = os.path.join(img_dir, img)
##    frame = cv2.imread(path)
##    frame = cv2.resize(frame, (640,360))
##    if frame.any == None:
##        print('no image read')
##        break
##    
##    
##    if initiate == False:
##        try:
##            init_bbox = cv2.selectROI("input", frame)
##            (x, y, w, h) = init_bbox
##            print(init_bbox)
##            cropped_img = frame[x:x+w, y:y+h, :]
##            print(cropped_img.shape)
##            prev_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
##            print('here')
##            mask = np.zeros_like(cropped_img)
##            mask[..., 1] = 255
##            print(mask.shape)
##            initiate = True
##        except:
##            exit()
##
##    
##    frame_cropped = frame[x:x+w, y:y+h, :]
##    
##    gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
##
##    # calc flow
##    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,\
##        0.5, 3, 15, 3, 5, 1.2, 0)
##    #print('flow: ', flow.shape)
##    #print(flow[:,0,0].shape)
##    # Computes the magnitude and angle of the 2D vectors
##    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
##
##    
##
##
##    #mag_mean = np.mean(magnitude)
##    #magnitude_opt.append(mag_mean)
##    #mean_ang = np.mean(angle)
##    
##    #p1 = (x+w//2, x+w//2+int(mag_mean*10*m.cos(mean_ang)))
##    #p2 = (y+h//2, y+h//2+int(mag_mean*10*m.sin(mean_ang)))
##
##    x_mean = np.mean(flow[... , 0])
##    y_mean = np.mean(flow[... , 1])
##
##    #mean_mag = np.sqrt(x_mean**2 + y_mean**2)
##    #mean_ang = m.atan2(y_mean, x_mean)
##
##    p1 = (x+w//2, x+w//2 + int(x_mean*10))
##    p2 = (y+h//2, y+h//2 + int(y_mean*10))
##    
##    frame = cv2.arrowedLine(frame, p1, p2, color=(255,10,0), thickness=2)
##    cv2.imshow("input", frame)
##    
##    # Sets image hue according to the optical flow 
##    # direction
##    #print(angle * 180 / np.pi / 2)
##    mask[..., 0] = angle * 180 / np.pi / 2
##    
##    # Sets image value according to the optical flow
##    # magnitude (normalized)
##    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
##
##    # Converts HSV to RGB (BGR) color representation
##    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
##
##    # Opens a new window and displays the output frame
##    cv2.imshow("dense optical flow", rgb)
##    
##
##    # Updates previous frame
##    prev_gray = gray
##    print('fps: ', 1/(time.time()-st))
##
##    # Frames are read by intervals of 1 millisecond. The
##    # programs breaks out of the while loop when the
##    # user presses the 'q' key
##    
##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        break
##    
##cv2.destroyAllWindows()
##plt.plot(magnitude_opt)
##plt.show()

