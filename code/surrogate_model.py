import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from scipy.stats import norm


class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(2, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(100, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(params):
    """
    trains surrogate autoencoder on normal traffic

    Args:
        params (dict): set of training parameters.

    Returns:
        type: None, model is saved in model_path
    """

    dataframe = pd.read_csv(params["path"], header=0)
    raw_data = dataframe.values

    # if there are 101 columns, last one is label
    if raw_data.shape[1] != 100:
        labels = raw_data[:, -1]
        data = raw_data[:, 0:-1]
    else:
        data = raw_data



    data = data.astype(np.float32)


    min_val = tf.reduce_min(data, axis=0)
    max_val = tf.reduce_max(data, axis=0)

    # save min and max
    np.savetxt("../models/surrogate_max.csv", max_val, delimiter=",")
    np.savetxt("../models/surrogate_min.csv", min_val, delimiter=",")

    data = (data - min_val) / (max_val - min_val)

    data = np.nan_to_num(data)

    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(data, data,
                              epochs=100,
                              batch_size=1024,
                              shuffle=False)

    tf.saved_model.save(autoencoder, params["out_path"])


def eval_surrogate(path, model_path, threshold=None, out_image=None, ignore_index=0, record_scores=False, meta_file=None):
    """
    evaluates the surrogate model on some traffic

    Args:
        path (string): path to network traffic feature file.
        model_path (string): path to surrogate autoencoder model.
        threshold (float): anomaly threshold value, if None it will be calculated as the maximum value from normal traffic. Defaults to None.
        out_image (string): path to output anomaly score plot. Defaults to None.
        ignore_index (int): index to ignore at the start. Defaults to 0.
        record_scores (boolean): whether to record anomaly score in a seperate csv file. Defaults to False.
        meta_file (string): metadata file for calculating evasion metrics. Defaults to None.

    Returns:
        Nothing

    """
    autoencoder = tf.keras.models.load_model(model_path)

    if meta_file is not None:
        colours = []

        with open(meta_file) as meta:
            for line in meta.readlines()[ignore_index + 1:]:
                comment = line.rstrip().split(",")[-1]
                if comment == "craft":
                    colours.append([67 / 255., 67 / 255., 67 / 255., 0.8])
                elif comment == "malicious":
                    colours.append([1, 0, 0, 1])
                else:
                    colours.append([204 / 255., 243 / 255., 1, 0.1])

    else:
        colours = "b"
    dataframe = pd.read_csv(path, header=0)

    raw_data = dataframe.values[ignore_index:]

    if raw_data.shape[1] == 101:
        # The last element contains the labels
        labels = raw_data[:, -1]

        # The other data points are the electrocadriogram data
        data = raw_data[:, 0:-1]
    else:
        data = raw_data

    max_val = np.genfromtxt("../models/surrogate_max.csv", delimiter=",")
    min_val = np.genfromtxt("../models/surrogate_min.csv", delimiter=",")

    data = (data - min_val) / (max_val - min_val + 1e-6)

    if out_image == None:
        out_image = path[:-4] + "_ae_rmse.png"

    counter = 0
    input_file = open(path, "r")
    input_file.readline()
    rmse_array = np.array([])

    chunks = data.shape[0] // 1024
    tbar = tqdm(total=chunks)
    for fv in np.array_split(data, chunks):

        fv = fv.astype(np.float32)

        reconstructions = autoencoder.predict(fv)
        train_loss = tf.keras.losses.mse(reconstructions, fv)

        rmse_array = np.concatenate((rmse_array, train_loss))
        counter += fv.shape[0]
        tbar.update(1)


    if threshold == None:
        threshold = max(rmse_array)
        print("threshold:", threshold)

    else:
        print(np.where(rmse_array > threshold))
        num_over = (rmse_array > threshold).sum()

    if record_scores:
        score_path = path[:-4] + "_imposter_score.csv"
        np.savetxt(score_path, np.log(rmse_array), delimiter=",")
        print("score saved to", score_path)

    max_index = np.argmax(rmse_array)
    max_rmse = rmse_array[max_index]
    plt.figure(figsize=(10, 5))
    plt.scatter(range(counter), rmse_array, s=0.2, c=colours)
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.yscale("log")
    plt.title("Anomaly Scores from imposter's Execution Phase")
    # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("packet index")
    plt.tight_layout()
    plt.savefig(out_image)
    print("plot path:", out_image)


if __name__ == '__main__':
    model_path = "../models/surrogate_ae.h5"
    normal_traffic_path= "../experiment/traffic_shaping/init_pcap/google_home_normal.csv"

    train_params = {
        "path": normal_traffic_path,
        "out_path": model_path
        }
    # train(train_params)

    # eval_surrogate(normal_traffic_path,
    #                model_path, threshold=None, out_image="../ku_dataset/anomaly_plot/surrogate_normal.png",record_scores=True)
   # 0.10926579684019089

    os_detect = "[OS & service detection]traffic_GoogleHome_av_only"
    flooding = "flooding_attacker_only"
    port_scan = "port_scan_attack_only"
    paths = [os_detect, flooding, port_scan]
    for path in paths:
       eval_surrogate("../ku_dataset/{}.csv".format(path), model_path, threshold=0.10926579684019089,
            out_image="../ku_dataset/anomaly_plot/{}_surrogate.png".format(path), record_scores=True)
