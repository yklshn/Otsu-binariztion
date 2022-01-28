import numpy as np
from cv2 import cv2

image = cv2.imread('images/img 3.jpg')
image = cv2.resize(image,(800,600))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('binary', image_gray)
cv2.waitKey(0)