from tabnanny import verbose
from numpy.random import seed
seed(42)
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.optimizers import adam_v2
import os


def load_data():
    img_points = pd.read_csv("./data/data_22225/points_cloud.csv")
    world_points = pd.read_csv("./data/data_22225/worldpoints.csv")
    center_points = pd.read_csv("./data/centerpoints.csv",header=None)
    interpolate = pd.read_csv("./data/centerpoints_interpolate.csv")
    interpolate = pd.read_csv("./data/data_22225/interpolate_progressed.csv")

    frames = [img_points, world_points]
    all_csv = pd.concat(frames, axis=1)
    center_points = pd.DataFrame(np.array(center_points),columns=all_csv.columns)
    
    all_csv = pd.DataFrame(np.array(center_points),columns=["0x","0y","1x","1y","2x","2y","3x","3y","4x","4y","5x","5y","6x","6y","7x","7y","x","y","z"])
    print(center_points)
    frames = [all_csv, center_points]
    all_csv = pd.concat(frames, axis=0)
    all_csv = all_csv.sample(frac=1)
    train_val, test = train_test_split(all_csv, test_size=0.1)
    val_pct = 1 / 9
    train, val = train_test_split(train_val, test_size=val_pct)
    # 混入插值点到val
    
    print(len(val))
    percentage = len(val) / len(interpolate)
    mixin = interpolate.sample(frac=percentage)
    val = pd.concat([val, mixin], axis=0)
    print(len(val))
    

    # 归一化
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    data = dict()
    data["train_X"] = train[:, :16]
    data["train_Y"] = train[:, 16:]
    data["val_X"] = val[:, :16]
    data["val_Y"] = val[:, 16:]
    data["test"] = test
    data["test_X"] = test[:, :16]
    data["test_Y"] = test[:, 16:]
    data["scaler"] = scaler
    return data

def build_network(input_features=None):
    # first we specify an input layer, with a shape == features
    inputs = Input(shape=(input_features,), name="input")
    # 32 neuron hidden layer
    scale_factor = 0.25
    x = Dense(128 * scale_factor, activation='relu', name="hidden1")(inputs)
    x = Dense(256 * scale_factor, activation='relu', name="hidden2")(x)
    x = Dense(512 * scale_factor, activation='relu', name="hidden3")(x)
    x = Dense(256 * scale_factor, activation='relu', name="hidden4")(x)
    x = Dense(128 * scale_factor, activation='relu', name="hidden5")(x)
    # for regression we will use a single neuron with linear (no) activation
    prediction = Dense(3, activation='linear', name="final")(x)
    model = Model(inputs=inputs, outputs=prediction)
    opt=adam_v2.Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss='mean_absolute_error')
    return model

def plot_history(history):
    hist=pd.DataFrame(history.history)
    hist['epoch']=history.epoch
    hist.tail()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'],hist['loss'],label='train error')
    plt.plot(hist['epoch'],hist['val_loss'],label='val_error')
    plt.legend()
    plt.show()
    plt.savefig('./centerpoints_test/regression.png')


def main():
    data = load_data()
    input_features = data["train_X"].shape[1]
    model = build_network(input_features)
    print("network structure")
    print(model.summary())
    print("Training Data Shape: " + str(data["train_X"].shape))
    
    weight_path = "./model/test.h5"
    callback = tf.keras.ModelCheckpoint(filepath=weight_path,save_weight_only=True,verbose=1)
    history = model.fit(x=data["train_X"], y=data["train_Y"], batch_size=8192, epochs=500, validation_data=(data["val_X"], data["val_Y"]), callbacks=[callback])
    # save model for later use
    model.save('./centerpoints_test/test.h5')

    actual = data["scaler"].inverse_transform(data["test"])[:, 16:]
    # print(actual)
    predict = data["scaler"].inverse_transform(np.hstack((data["test_X"], model.predict(data["test_X"]))))[:, 16:] 
    print(predict)
    print("model test MAE:" + str(mean_absolute_error(actual, predict)))
    print("model x MAE:" + str(mean_absolute_error(actual[:, 0], predict[:, 0])))
    print("model y MAE:" + str(mean_absolute_error(actual[:, 1], predict[:, 1])))
    print("model z MAE:" + str(mean_absolute_error(actual[:, 2], predict[:, 2])))
    plot_history(history)

if __name__ == "__main__":
    main() 
