import cv2
import os
import numpy as np

img_fold = "/media/dtu-project2/2GB_HDD/object_tracker/DTB70/CFtracker_output_opencv/Walking/kcf_rel"

img_list  = os.listdir(img_fold)
img_list.sort()
'''
for img in img_list:
    path = os.path.join(img_fold, img)
    image = cv2.imread(path)
    (h, w, c) = image.shape
    image = image[h//2-50-50:h//2+50+50, w//2-50-50:w//2+50+50]
    cv2.imwrite(path, image)
    print(image.shape)
  '''
rect = (159, 100, 40, 95)
p1 = (rect[0]-rect[2]//2, rect[1]-rect[3]//2)
p2 = (rect[0]+rect[2]//2, rect[1]+rect[3]//2)
img = "00211.jpg"
path = os.path.join(img_fold, img)
image = cv2.imread(path)
cv2.rectangle(image,p1,p2,color=(0,255,255), thickness=2)
#cv2.rectangle(image, (10,10), (150,150), color=(255,255,255), thickness=2)
cv2.imwrite(path, image)