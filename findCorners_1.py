# 对单张图片识别角点并作图
import cv2
import numpy as np
import os

row, column = 200, 323
cam_num = 8
pic_dir = './pic200x323_3.81'

img = cv2.imread(os.path.join(pic_dir, '0_000.png'))
ret, corners = cv2.findChessboardCornersSB(img, (row, column))
drawed_img = cv2.drawChessboardCorners()