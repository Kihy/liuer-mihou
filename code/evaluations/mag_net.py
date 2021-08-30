import keras.regularizers as regs
from keras.models import Model, load_model
from keras.layers import Input, Dense, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers.merge import Average, add
from keras.layers.core import Lambda
import math
from itertools import product
from tqdm import tqdm
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle
from KitNET.KitNET import KitNET
from sklearn.preprocessing import normalize
import sys
sys.path.append("..")


class AEDetector:
    def __init__(self, path, p=1):
        """
        Error based detector.
        Marks examples for filtering decisions.
        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model = load_model(path)
        self.path = path
        self.p = p

    def mark(self, X):
        diff = np.abs(X - self.model.predict(X))
        marks = np.mean(np.power(diff, self.p), axis=(1))
        return marks

    def print(self):
        return "AEDetector:" + self.path.split("/")[-1]


class IdReformer:
    def __init__(self, path="IdentityFunction"):
        """
        Identity reformer.
        Reforms an example to itself.
        """
        self.path = path
        self.heal = lambda X: X

    def print(self):
        return "IdReformer:" + self.path


class SimpleReformer:
    def __init__(self, path):
        """
        Reformer.
        Reforms examples with autoencoder. Action of reforming is called heal.
        path: Path to the autoencoder used.
        """
        self.model = load_model(path)
        self.path = path

    def heal(self, X):
        X = self.model.predict(X)
        return X

    def print(self):
        return "SimpleReformer:" + self.path.split("/")[-1]


class DenoisingAutoEncoder:
    def __init__(self, image_shape,
                 structure,
                 v_noise=0.0,
                 activation="relu",
                 model_dir="evaluations/defensive_models/",
                 hidden_ratio=0.75,
                 reg_strength=0.0):
        """
        Denoising autoencoder.
        image_shape: Shape of input image. e.g. 28, 28, 1.
        structure: Structure of autoencoder.
        v_noise: Volume of noise while training.
        activation: What activation function to use.
        model_dir: Where to save / load model from.
        reg_strength: Strength of L2 regularization.
        """

        self.image_shape = image_shape
        self.model_dir = model_dir
        self.v_noise = v_noise

        # structure=[int(np.ceil(image_shape*hidden_ratio))]

        input_img = Input(shape=self.image_shape)
        x = input_img

        for layer in structure:
            if isinstance(layer, int):
                x = Dense(layer, activation=activation,
                          activity_regularizer=regs.l2(reg_strength))(x)
            else:
                print(layer, "is not recognized!")
                exit(0)

        for layer in reversed(structure):
            if isinstance(layer, int):
                x = Dense(layer, activation=activation,
                          activity_regularizer=regs.l2(reg_strength))(x)

        decoded = Dense(image_shape[0], activation='sigmoid',
                        activity_regularizer=regs.l2(reg_strength))(x)
        self.model = Model(input_img, decoded)

    def train(self, data, archive_name, num_epochs=100, batch_size=256,
              if_save=True):
        self.model.compile(loss='mse',
                           metrics=['mse'],
                           optimizer='adam')

        noise = self.v_noise * np.random.normal(size=np.shape(data))
        noisy_train_data = data + noise

        # noisy_train_data=data

        self.model.fit(noisy_train_data, data,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

        if if_save:
            self.model.save(os.path.join(self.model_dir, archive_name))

    def load(self, archive_name, model_dir=None):
        if model_dir is None:
            model_dir = self.model_dir
        self.model.load_weights(os.path.join(model_dir, archive_name))


def train_mag_net(training_data):
    shape = [100]
    combination_I = [75]
    combination_II = [50, 25, 10]
    activation = "sigmoid"
    reg_strength = 1e-9
    epochs = 100

    dataframe = pd.read_csv(training_data)

    # dataframe = pd.read_csv(
    #     "../../ku_dataset/train.csv", header=0)
    raw_data = dataframe.values

    # if there are 101 columns, last one is label
    data = raw_data[:,:100]


    data = data.astype(np.float32)

    min_val = tf.reduce_min(data, axis=0)
    max_val = tf.reduce_max(data, axis=0)

    # save min and max
    np.savetxt("../models/magnet_max.csv", max_val, delimiter=",")
    np.savetxt("../models/magnet_min.csv", min_val, delimiter=",")

    data = (data - min_val) / (max_val - min_val)

    data = np.nan_to_num(data)

    AE_I = DenoisingAutoEncoder(shape, combination_I, v_noise=0, activation=activation,
                                reg_strength=reg_strength)
    AE_I.train(data, "TRAIN_I", num_epochs=epochs)

    AE_II = DenoisingAutoEncoder(shape, combination_II, v_noise=0, activation=activation,
                                 reg_strength=reg_strength)
    AE_II.train(data, "TRAIN_II", num_epochs=epochs)


def test_mag_net(model_path, train_data, test_data):
    with open(model_path, "rb") as m:
        kitsune = pickle.load(m)

    detector_I = AEDetector("evaluations/defensive_models/TRAIN_I", p=2)
    detector_II = AEDetector("evaluations/defensive_models/TRAIN_II", p=2)
    reformer = SimpleReformer("evaluations/defensive_models/TRAIN_II")

    max_val = np.loadtxt(
        open("../models/magnet_max.csv", "rb"), delimiter=",")
    min_val = np.loadtxt(
        open("../models/magnet_min.csv", "rb"), delimiter=",")

    train_df = pd.read_csv(train_data)
    train = train_df.values[:,:100]
    train = train.astype(np.float32)

    dataframe = pd.read_csv(test_data, header=0)
    test = dataframe.values[:,:100]
    test = test.astype(np.float32)

    train = (train - min_val) / (max_val - min_val)
    test = (test - min_val) / (max_val - min_val)

    #find thresholds on benign
    marks = detector_I.mark(train)
    detector_I_threshold=np.max(marks)

    marks = detector_II.mark(train)
    detector_II_threshold=np.max(marks)

    reformed = reformer.heal(train)

    # transform reformed back to unnormalized

    reformed=reformed*(max_val-min_val)+min_val
    # data=data*(max_val-min_val)+min_val

    training_rmse=[]
    for i in range(reformed.shape[0]):
        reformed_rmse = kitsune.process(reformed[i])
        training_rmse.append(reformed_rmse)

    kitsune_threshold=np.max(training_rmse)

    total_mal=np.sum(np.logical_or(detector_I.mark(test)>detector_I_threshold, detector_II.mark(test)>detector_II_threshold))

    # heal and detect
    test_heal = reformer.heal(test)
    test_heal=test_heal*(max_val-min_val)+min_val
    test_rmse=[]
    for i in range(test_heal.shape[0]):
        reformed_rmse = kitsune.process(test_heal[i])
        test_rmse.append(reformed_rmse)
    test_failed=np.sum(test_rmse>kitsune_threshold)

    print(total_mal, test_failed)
    return total_mal,test_failed
