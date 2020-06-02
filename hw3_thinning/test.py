import cv2
import numpy as np

img = cv2.imread('pcb.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# thr = cv2.adaptiveThreshold(img, 0xff, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=61, C=8)
cv2.imshow("gray", img)
cv2.waitKey(0)