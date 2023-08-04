
import cv2


inp_img = "1666.jpg"
img = cv2.imread(inp_img)


def shift_left(img, shift):
    org_img = img.copy()
    for i in range(img.shape[1]):
        if i+shift >= img.shape[1]:
            temp = i+shift- img.shape[1]
        else:
            temp = i+shift
        img[:, i] = org_img[:, temp]
    return img

def shift_up(img, shift):
    org_img = img.copy()
    for i in range(img.shape[1]):
        if i+shift >= img.shape[1]:
            temp = i+shift- img.shape[1]
        else:
            temp = i+shift
        img[i, :] = org_img[temp, :]
    return img



k = -30
img = shift_up(img, -k)
img = shift_left(img, k)


cv2.imwrite('up_right_shift.jpg', img)

cv2.imshow("frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
