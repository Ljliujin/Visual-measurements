import numpy as np
from keras.models import load_model
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
layer = 1
row = [50,150]
column = 100
n = 200*299
img_points = pd.read_csv('./data/points_cloud.csv')
world_points = pd.read_csv("./data/worldpoints.csv")
center_points = pd.read_csv("./data/centerpoints.csv",header=None)

def load_data_origin():
    frames = [img_points, world_points]
    data = pd.concat(frames,axis=1)
    return data
data = load_data_origin()
def load_scaler():
    csv = data
    column_list = ["0x","0y","1x","1y","2x","2y","3x","3y","4x","4y","5x","5y","6x","6y","7x","7y","x","y","z"]
    centerpoints = pd.DataFrame(np.array(center_points), columns=column_list)
    frames = [csv, center_points]
    all_csv = pd.concat(frames, axis=0)

    all_csv = all_csv.sample(frac=1)
    train_val, test = train_test_split(all_csv, test_size=0.1)
    val_pct = 1 / 9
    train, val = train_test_split(train_val, test_size=val_pct)

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    return scaler

scaler = load_scaler()
model = load_model('./model/best_model/model.h5')
corner_pt0 = data[layer*n+row[0]*200+column][:16]
# corner_pt1 = corners[layer*n+row[1]*200+column][:16]
wrd_pt0 = model.predict(corner_pt0)