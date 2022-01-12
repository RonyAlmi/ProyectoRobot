import sys
import cv2
import numpy as np
import time


def find_depth(center_left, center_right, frame_left, frame_right, baseline, f, alpha):
    height_left, width_left, depth_left = frame_left.shape
    height_right, width_right, depth_right = frame_right.shape

    if width_left == width_right:
        f_pixel = (width_left * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Los frames de las cámaras no son iguales')

    x_left = center_left[0]
    x_right = center_right[0]

    disparity = x_left - x_right

    if disparity == 0:
        disparity = 1

    depth = (baseline * f_pixel) / disparity

    return abs(depth)