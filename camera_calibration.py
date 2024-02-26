from json import load
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def load_data():
    worlds_points = pd.read_csv('./data/worldpoints.csv').to_numpy()
    img_points = pd.read_csv('./data/points_cloud.csv').to_numpy()
    return worlds_points, img_points

def calibration(world_points, img_points, img_shape, mtx=None, dist=None, rvecs=None, tvecs=None, flags=None):
    retl, mtxl, distl, rvecsl, tvecsl = cv2.calibrateCamera(world_points, img_points, img_shape , mtx, dist, rvecs, tvecs, flags)
    print(retl, mtxl, distl, rvecsl, tvecsl)
    return retl, mtxl, distl, rvecsl, tvecsl

def cal_error(world_position, image_position, mtx, dist, rvecs, tvecs):
        #计算重投影误差
        image_position2, _ = cv2.projectPoints(world_position, rvecs[0], tvecs[0], mtx, dist)
        image_position2 = np.reshape(image_position2,(len(world_position),2))
        print(mean_absolute_error(image_position, image_position2))

def main():
    world_points, img_points = load_data()
    world_points0 = np.asarray([world_points[:59800]], dtype=np.float32)
    img_points0 = np.asarray([img_points[:59800, :2]], dtype=np.float32)
    # print(world_points, img_points)
    retl, mtxl, distl, rvecsl, tvecsl = calibration(world_points0, img_points0, (3456,4608))
    cal_error(world_points0[0], img_points0[0], mtxl, distl, rvecsl, tvecsl)

    world_points = np.asarray([world_points], dtype=np.float32)
    img_points = np.asarray([img_points[:, :2]], dtype=np.float32)
    ret, mtx, dist, rvecs, tvecs = calibration(world_points, img_points, (3456,4608), mtxl, distl, rvecsl, tvecsl, cv2.CALIB_USE_INTRINSIC_GUESS)
    cal_error(world_points[0], img_points[0], mtx, dist, rvecs, tvecs)
    
main()