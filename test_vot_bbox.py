import cv2
import numpy as np

img = cv2.imread("00000001.jpg")

bbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
for i in range(len(bbox)):
    bbox[i] = int(bbox[i])
(x1,y1) = (bbox[0],bbox[1])
(x2,y2) = (bbox[2],bbox[3])
(x3,y3) = (bbox[4],bbox[5])
(x4,y4) = (bbox[6],bbox[7])

(xmin, ymin) = (min(x1,x2,x3,x4), min(y1,y2,y3,y4))
(xmax, ymax) = (max(x1,x2,x3,x4), max(y1,y2,y3,y4))

cv2.circle(img, (x1,y1), 5, (255,0,0), 2)
cv2.circle(img, (x2,y2), 5, (0,255,0), 2)
cv2.circle(img, (x3,y3), 5, (0,0,255), 2)
cv2.circle(img, (x4,y4), 5, (255,255,255), 2)
cv2.rectangle(img, (xmin, ymin), (xmax, ymax), 2)

cv2.imshow("frame", img)

key = cv2.waitKey(0) & 0xff
if key == 27:
    cv2.destroyAllWindows()