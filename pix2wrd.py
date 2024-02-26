from operator import imod
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

lr = 0.0001
scale_factor = 0.25
result_path = "./model/pix2wrd"
if not os.path.exists(result_path):
    os.makedirs(result_path)
result_name = "pix2wrd.h5"
#ckpt_path = os.path.join("./test_2","model.ckpt")
ckpt_path = os.path.join(result_path,"pix2wrd.ckpt")
log_dir = os.path.join(result_path,'./logs20220512-131235')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
# epoch done = 258
epoch = 1000
batch = 32*8
cam_num = 8


def load_data():

    all_csv = pd.read_csv('./data/pointscloud.csv', header=None)
    column=["0x","0y","1x","1y","2x","2y","3x","3y","4x","4y","5x","5y","6x","6y","7x","7y","x","y","z"]
    
    all_csv = pd.DataFrame(np.array(all_csv), columns = column)
    all_csv = all_csv.sample(frac=1)
    train_val, test = train_test_split(all_csv, test_size=0.1)
    val_pct = 1 / 9
    train, val = train_test_split(train_val, test_size=val_pct)

    # 归一化
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    data = dict()
    data["train_X"] = train[:, :cam_num*2]
    data["train_Y"] = train[:, cam_num*2:]
    data["val_X"] = val[:, :cam_num*2]
    data["val_Y"] = val[:, cam_num*2:]
    data["test"] = test
    data["test_X"] = test[:, :cam_num*2]
    data["test_Y"] = test[:, cam_num*2:]
    data["scaler"] = scaler
    return data


def build_network(input_features=None):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # first we specify an input layer, with a shape == features
        inputs = Input(shape=(input_features,), name="input")
        # 32 neuron hidden layer
        x = Dense(128 * scale_factor, activation='relu', name="hidden1")(inputs)
        x = Dense(256 * scale_factor, activation='relu', name="hidden2")(x)
        x = Dense(512 * scale_factor, activation='relu', name="hidden3")(x)
        x = Dense(256 * scale_factor, activation='relu', name="hidden4")(x)
        x = Dense(128 * scale_factor, activation='relu', name="hidden5")(x)
        # for regression we will use a single neuron with linear (no) activation
        prediction = Dense(2*cam_num, activation='linear', name="final")(x)
        model = Model(inputs=inputs, outputs=prediction)
        opt=adam_v2.Adam(learning_rate=lr)
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
    plt.savefig(os.path.join(result_path, 'regression.png'))

def main():
    data = load_data()
    input_features = data["train_X"].shape[1]
    model = build_network(input_features)
    print("network structure")
    print(model.summary())
    print("Training Data Shape: " + str(data["train_X"].shape))

    if os.path.exists(ckpt_path):
        print("--------------------------load the model-----------------------------")
        model.load_weights(ckpt_path)

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,save_weight_only=False,save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x=data["train_X"], y=data["train_Y"], batch_size=batch, epochs=epoch, validation_data=(data["val_X"], data["val_Y"]), callbacks=[callback, tensorboard_callback])
    # save model for later use
    model.save(os.path.join(result_path, result_name))

    actual = data["scaler"].inverse_transform(data["test"])[:, cam_num*2:]
    # print(actual)
    predict = data["scaler"].inverse_transform(np.hstack((data["test_X"], model.predict(data["test_X"]))))[:, cam_num*2:] 
    print(predict)
    print("model test MAE:" + str(mean_absolute_error(actual, predict)))
    print("model x MAE:" + str(mean_absolute_error(actual[:, 0], predict[:, 0])))
    print("model y MAE:" + str(mean_absolute_error(actual[:, 1], predict[:, 1])))
    print("model z MAE:" + str(mean_absolute_error(actual[:, 2], predict[:, 2])))
    plot_history(history)

if __name__ == "__main__":
    main() 