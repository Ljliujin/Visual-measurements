from copyreg import add_extension
from ctypes.wintypes import WORD
from email import message_from_binary_file
from turtle import color
from zlib import ZLIB_RUNTIME_VERSION
from numpy.random import seed
seed(42)
import tensorflow as tf
tf.compat.v1.set_random_seed(42)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

img_points = pd.read_csv('./data/points_cloud.csv')
world_points = pd.read_csv("./data/worldpoints.csv")
center_points = pd.read_csv("./data/centerpoints.csv",header=None)
interpolate_1 = pd.read_csv("./data/interpolate_progressed.csv")
interpolate_2 = pd.read_csv("./data/interpolate_progressed.csv")
interpolate = pd.concat([interpolate_1, interpolate_2],axis=0)


def load_data():
    frames = [img_points,world_points]
    all_csv = pd.concat(frames,axis=1)

    return all_csv

def load_scaler():
    csv = load_data()
    center_point = pd.DataFrame(np.array(center_points),columns=csv.columns)
    all_csv = pd.concat([csv, center_point],axis=0)

    all_csv = all_csv.sample(frac=1)
    train_val, test = train_test_split(all_csv, test_size=0.1)
    val_pct = 1/9
    train, val = train_test_split(train_val, test_size=val_pct)

    percentage = len(val) / len(interpolate)    
    mixin = interpolate.sample(frac=percentage)
    val = pd.concat([val, mixin],axis=0)

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    return scaler


def predict_(data,scaler,model):
    np.set_printoptions(suppress=True)

    actual = data.iloc[:, 16:]
    test = scaler.transform(data)
    predict = scaler.inverse_transform(np.hstack((test[:,:16],model.predict(test[:,:16]))))[:, 16:]
    
    X_error = mean_absolute_error(actual.iloc[:,0], predict[:,0])
    Y_error = mean_absolute_error(actual.iloc[:,1], predict[:,1])
    Z_error = mean_absolute_error(actual.iloc[:,2], predict[:,2])
    mean_error = mean_absolute_error(actual, predict)

    print("X MAE" + str(X_error))
    print("Y MAE" + str(Y_error))
    print("Z MAE" + str(Z_error))
    print("total MAE" + str(mean_error))

    return predict

def draw(direction,error,color,save_dir):
    z = np.arange(76)
    L = plt.plot(z, error, color, label = "{}_error".format(direction))
    plt.rcParams["font.sans-serif"]=["SimHei"]
    plt.rcParams["font.family"]="sans-serif"
    plt.xlabel("Z层数变化（4mm每层）")
    plt.ylabel("平面上所有点的平均误差(mm)")
    plt.title("{}方向误差随Z层数变化趋势".format(direction))
    plt.legend()
    save_path = os.path.join(save_dir, "{}_error.jpg".format(direction))
    plt.savefig(save_path)
    plt.cla()
    print(save_path)


def main():
    data = load_data()
    scaler = load_scaler()
    model_path = "./model/best_model/model.h5"
    model = load_model(model_path)

    predict = predict_(data,scaler,model)
    err = np.abs(predict - np.array(data.iloc[:,16:]))

    num = 200*299

    x_error ,y_error, z_error = [], [], []
    for z in range(76):
        layer_x_error = np.mean(err[:,0][z*num:(z+1)*num])
        x_error.append(layer_x_error)

        layer_y_error = np.mean(err[:,1][z*num:(z+1)*num])
        y_error.append(layer_y_error)

        layer_z_error = np.mean(err[:,2][z*num:(z+1)*num])
        z_error.append(layer_z_error)

    pic_save_dir = "./model/best_model"
    draw(direction="X", error=x_error, color="r", save_dir=pic_save_dir)
    draw(direction="Y", error=y_error, color="b", save_dir=pic_save_dir)
    draw(direction="Z", error=z_error, color="g", save_dir=pic_save_dir)
    print("Done")

main()