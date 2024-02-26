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
from mpl_toolkits.mplot3d import Axes3D


def load_data():
    # data = pd.read_csv('./data/mixin.csv')
    img_points = pd.read_csv('./data/points_cloud.csv')
    world_points = pd.read_csv("./data/worldpoints.csv")
    frames = [img_points, world_points]
    all_csv = pd.concat(frames, axis=1)
    return all_csv

    
def get_model():
    model = load_model('./model/best_model/best_model.h5')
    return model

def load_scaler():
    img_points = pd.read_csv("./data/points_cloud.csv")
    world_points = pd.read_csv("./data/worldpoints.csv")
    center_points = pd.read_csv("./data/centerpoints.csv",header=None)
    '''
    interpolate_1 = pd.read_csv("./data/centerpoints_interpolate.csv")
    interpolate_2 = pd.read_csv("./data/interpolate_progressed.csv")
    interpolate = pd.concat([interpolate_1, interpolate_2],axis=0)
    '''
    frames = [img_points, world_points]

    csv = pd.concat(frames, axis=1)
    center_points = pd.DataFrame(np.array(center_points),columns=csv.columns)
    csv_center = pd.DataFrame(np.array(center_points),columns=["0x","0y","1x","1y","2x","2y","3x","3y","4x","4y","5x","5y","6x","6y","7x","7y","x","y","z"])
    #print(center_points)
    frames = [csv, csv_center]
    all_csv = pd.concat(frames, axis=0)

    #all_csv = all_csv.sample(frac=1)
    train_val, test = train_test_split(all_csv, test_size=0.1)
    val_pct = 1 / 9
    train, val = train_test_split(train_val, test_size=val_pct)

    # 归一化
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    return scaler

def predict(data, scaler, model):
    i = 75

    num = 59800
    np.set_printoptions(suppress=True)
    '''
    rows = data.shape[0]
    for col in ['x', 'y', 'z']:
        data[col] = [0 for i in range(rows)]
    '''

    actual = data.iloc[i*num:(i+1)*num, 16:]
    test = scaler.transform(data)
    predict = scaler.inverse_transform(np.hstack((test[i*num:(i+1)*num, :16], model.predict(test[i*num:(i+1)*num, :16]))))[:, 16:]


    X_error = mean_absolute_error(actual.iloc[:, 0], predict[:, 0])
    Y_error = mean_absolute_error(actual.iloc[:, 1], predict[:, 1])
    Z_error = mean_absolute_error(actual.iloc[:, 2], predict[:, 2])
    mean_error = mean_absolute_error(actual, predict)
    print("model test MAE:" + str(mean_error))
    print("model x MAE:" + str(X_error))
    print("model y MAE:" + str(Y_error))
    print("model z MAE:" + str(Z_error))

    # 三维曲面图
    error = predict - actual
    error = error.abs()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(actual.iloc[:, 0], actual.iloc[:, 1], error.iloc[:, 0], cmap="jet")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.rcParams["font.sans-serif"]=["SimHei"]
    plt.rcParams["font.family"]="sans-serif"
    plt.title("X direction error | the {}th layer\nerr={}mm".format(i,X_error))
    plt.savefig("./result/3D--X.png")
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(actual.iloc[:, 0], actual.iloc[:, 1], error.iloc[:, 1], cmap="jet")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.rcParams["font.sans-serif"]=["SimHei"]
    plt.rcParams["font.family"]="sans-serif"
    plt.title("Y direction error | the {}th layer\nerr={}mm".format(i,Y_error))
    plt.savefig("./result/3D--Y.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(actual.iloc[:, 0], actual.iloc[:, 1], error.iloc[:, 2], cmap="jet")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title("Z direction error | the {}th layer\nerr={}mm".format(i,Z_error))

    plt.savefig("./result/3D--Z.png")

    mean = error.mean(1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(actual.iloc[:, 0], actual.iloc[:, 1], mean, cmap="jet")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.rcParams["font.sans-serif"]=["SimHei"]
    plt.rcParams["font.family"]="sans-serif"
    plt.title("XYZ direction mean error | the {}th layer\nmean err={}".format(i,mean_error))

    plt.savefig("./result/3D--mean.png")



def main():
    data = load_data()
    scaler = load_scaler()
    model = get_model()
    predict(data, scaler, model)

if __name__ == "__main__":
    main()
                          
