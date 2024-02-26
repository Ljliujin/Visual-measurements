import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

'设置全局变量'
row, column = 200, 323  # 标定板行数与列数[需保证行列数奇偶性不同，否则角点的起点不统一]
cam_num = 8
pic_dir = './pic200x323_3.81_3'
data_save_path = './data/imagepoints.csv'
nan = np.full((row*column,1,2), np.NaN)

def findChessboardCorners(pic_dir, pic_name):
    # 图片命名方式：相机号_层数.png
    index = int(pic_name.split('.')[0].split('_')[0])   # 相机编号
    img = cv2.imread(os.path.join(pic_dir, pic_name))   
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    begin = time.time()
    ret, corners = cv2.findChessboardCornersSB(img, (row, column))      # 识别角点
    if not ret:
        ret, corners = cv2.findChessboardCornersSB(gray, (row, column)) # 部分棋盘格的原图无法识别而灰度图可识别
        if not ret:                                                     # 灰度图也无法识别，则尝试用拉普拉斯变换
            cvted = 255-cv2.Laplacian(img, cv2.CV_8U, ksize=7)
            ret, corners = cv2.findChessboardCornersSB(cvted, (row, column))
    end = time.time()
    print(pic_name,ret, f'{round(end-begin, 2)}s')
    if not ret:                                                         # 若还不能识别，则暂时写入nan数据，记录图片名，后续做单独处理
        corners = nan
        print(pic_name,'*******************False*******************')
        with open('./fail_pics.txt', 'a') as f:
            f.write(pic_name)

    corners_list = [[corner[0][0], corner[0][1]] for corner in corners] # 转换格式，以备写入点云
    return ret, corners_list, index

pic_names = os.listdir(pic_dir)
""" 给图片按照先层数后相机序号排序,使同一层的8张图片连续 
[图片命名方式: 3_16.png --> 3号相机的第16个位置图片]"""
pic_names = sorted(pic_names, key=lambda x:(int(x.split('.')[0].split('_')[1]), x.split('.')[0].split('_')[0]))
        
imgpoints = []  # 储存全部角点数据
with ThreadPoolExecutor(8) as executor:
    for i in range(int(len(pic_names)/cam_num)):
        index_point = dict()    # 按相机标号存储角点信息
        all_task = [executor.submit(findChessboardCorners, pic_dir, pic_name) for pic_name in pic_names[i*cam_num:(i+1)*cam_num]]

        for future in as_completed(all_task):
            ret, corners_list, index = future.result()
            index_point[index] = corners_list

        points = []
        for i in range(cam_num):
            points = index_point[i] if i == 0 else np.hstack((points, index_point[i]))  # 按相机顺序横向排列

        imgpoints = points if len(imgpoints) == 0 else np.vstack((imgpoints, points))   # 按标定板位置位置顺序纵向排列

np.savetxt(data_save_path, imgpoints, fmt='%f', delimiter=',')
print('Done')