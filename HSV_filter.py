# https://www.youtube.com/watch?v=t3LOey68Xpg

import sys
import cv2
import numpy as np
import time


def add_HSV_filter(frame, pts):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask_z = np.zeros(frame.shape, np.uint8)
    cv2.drawContours(mask_z, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    mask = cv2.bitwise_and(hsv, mask_z)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask