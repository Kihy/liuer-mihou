from evaluations.feature_squeeze import squeeze_features
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import sys
import multiprocessing as mp
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class AnomalyDetector(Model):
    def __init__(self, structure=[(100, "sigmoid"), (75, "relu"), (50, "relu")]):
        super(AnomalyDetector, self).__init__()
        encoder = [layers.Dense(i, activation=j) for i, j in structure[1:]]
        decoder = [layers.Dense(i, activation=j)
                   for i, j in structure[::-1][1:]]

        self.encoder = tf.keras.Sequential(encoder)
        self.decoder = tf.keras.Sequential(decoder)

        print("structure:", structure)

    def call(self, x, training):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        mse = tf.keras.losses.mean_squared_error(decoded, x)
        self.add_loss(mse)
        return mse

# @tf.function


def get_jacobian(model, x):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = model(x)
    jacobian = tape.batch_jacobian(y, x)
    return jacobian


def get_gradient(model, x):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = model(x)
        out = tf.keras.losses.MSE(y, x)
    gradient = tape.gradient(out, x)
    return gradient


def train_dim_reduce(params):
    """
    trains surrogate autoencoder on normal traffic

    Args:
        params (dict): set of training parameters.

    Returns:
        type: None, model is saved in model_path
    """

    autoencoder = AnomalyDetector(
        structure=[(100, "sigmoid"), (3, "relu")])
    # autoencoder=MultipleAE()
    autoencoder.compile(optimizer='adam')


    data = None
    for path in params["paths"]:
        dataset = pd.read_csv(path, header=0, dtype="float32",
                              chunksize=10000000, usecols=list(range(100)))

        for chunk in tqdm(dataset):

            if data is None:
                data=chunk
            else:
                data = np.concatenate( (data,chunk), axis=0)


    scaler = MinMaxScaler()
    data=scaler.fit_transform(data)

    with open(params["model_path"] + "_scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
        print("scaler saved at", params["model_path"] + "_scaler.pkl")

    history = autoencoder.fit(data, data,
                              epochs=params["epochs"],
                              batch_size=params["batch_size"],
                              shuffle=params["shuffle"])

    # tf.saved_model.save(autoencoder, params["model_path"])
    autoencoder.save(params["model_path"])


def train_surrogate(params):
    """
    trains surrogate autoencoder on normal traffic

    Args:
        params (dict): set of training parameters.

    Returns:
        type: None, model is saved in model_path
    """

    autoencoder = AnomalyDetector(
        structure=[(100, "sigmoid"), (32, "relu"), (8, "relu"), (2, "relu")])
    # autoencoder=MultipleAE()
    autoencoder.compile(optimizer='adam')

    dataset = pd.read_csv(params["path"], header=0, dtype="float32",
                          chunksize=10000000, usecols=list(range(100)))

    scaler = MinMaxScaler()
    print("preprocessing data")
    for chunk in tqdm(dataset):
        scaler.partial_fit(chunk)

    with open(params["model_path"] + "_scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
        print("scaler saved at", params["model_path"] + "_scaler.pkl")

    dataset = pd.read_csv(params["path"], header=0, dtype="float32",
                          chunksize=10000000, usecols=list(range(100)))

    for chunk in tqdm(dataset):
        chunk = scaler.transform(chunk)
        history = autoencoder.fit(chunk, chunk,
                                  epochs=1,
                                  batch_size=params["batch_size"],
                                  shuffle=False)

    # tf.saved_model.save(autoencoder, params["model_path"])
    autoencoder.save(params["model_path"])


def eval_surrogate(path, model_path, threshold=None, out_image=None, ignore_index=0, record_scores=False, meta_file=None, record_prediction=False, y_true=None):
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
    print("loading from", model_path)
    autoencoder = tf.keras.models.load_model(model_path)
    t = threshold
    roc_auc = 1

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
    dataset = pd.read_csv(path, header=0,
                          chunksize=10000000, usecols=list(range(100)))

    with open(model_path + "_scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    if out_image == None:
        out_image = path[:-4] + "_ae_rmse.png"

    counter = 0
    input_file = open(path, "r")
    input_file.readline()
    rmse_array = []

    for chunk in tqdm(dataset):
        chunk = chunk.astype("float32")
        chunk = scaler.transform(chunk)
        anomaly_score = autoencoder(chunk)

        rmse_array.extend(anomaly_score)
        counter += chunk.shape[0]

    if threshold == None:
        benignSample = np.log(rmse_array)
        mean = np.mean(benignSample)
        std = np.std(benignSample)
        threshold_std = np.exp(mean + 3 * std)
        threshold_max = np.max(rmse_array)
        # threshold = min(threshold_max, threshold_std)
        threshold=threshold_std
        # threshold=np.percentile(rmse_array,99)
    rmse_array = np.array(rmse_array)
    print("max rmse", np.max(rmse_array))
    first_greater = np.argmax(rmse_array > threshold)
    print("first_greater at ", first_greater)
    num_over = (rmse_array > threshold).sum()

    if record_scores:
        score_path = path[:-4] + "_imposter_score.csv"
        np.savetxt(score_path, rmse_array, delimiter=",")
        threshold_path = path[:-4] + "_imposter_threshold.csv"
        np.savetxt(threshold_path, [threshold], delimiter=",")
        print("score saved to", score_path)

    # record prediction labels
    if record_prediction:
        pred_path = path[:-4] + "_autoencoder_prediction.csv"
        np.savetxt(pred_path, rmse_array > threshold, delimiter=",")
        print("autoencoder prediction saved to", pred_path)

    if y_true is None:
        fpr, tpr, roc_t = metrics.roc_curve(
            [0 for i in range(len(rmse_array))], rmse_array, drop_intermediate=True)
    else:
        fpr, tpr, roc_t = metrics.roc_curve(
            y_true, rmse_array, drop_intermediate=True)
        roc_auc = metrics.auc(fpr, tpr)

    if out_image is not None:
        max_rmse = max(rmse_array)
        max_index = np.argmax(rmse_array)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.scatter(range(len(rmse_array)), rmse_array, s=0.2, c=colours)
        # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
        ax1.axhline(y=threshold, color='r', linestyle='-')
        # ax1.set_yscale("log")
        ax1.set_title("Anomaly Scores from Autoencoder_{} Execution Phase".format(
            model_path.split("/")[-1]))
        ax1.set_ylabel("RMSE (log scaled)")
        ax1.set_xlabel("packet index")

        if y_true is None:
            ax2.plot(fpr, roc_t, 'b')
            ax2.set_ylabel("threshold")
            ax2.set_xlabel("false positive rate")
        else:
            ax2.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            ax2.set_title('AUC = %0.2f' % roc_auc)
            ax2.set_ylabel("true positive rate")
            ax2.set_xlabel("false positive rate")
        plt.tight_layout()
        f.savefig(out_image)
        print("plot path:", out_image)

    if t is None:
        return num_over, threshold
    else:
        return num_over, roc_auc


if __name__ == '__main__':
    mal_files=[ "../ku_dataset/port_scan_attack_only.csv",
               "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only.csv", "../ku_dataset/flooding_attack_only.csv"]
    train_surrogate({
        # the pcap, pcapng, or tsv file to process.
        "path":  "../ku_dataset/google_home_normal.csv",
        # directory of kitsune
        "model_path": "../models/ku/surrogate_ae",
        "batch_size": 1024
    })

    pos, threshold=eval_surrogate(** {"path":"../ku_dataset/google_home_normal.csv",
                  "model_path":"../models/ku/surrogate_ae",
                  "threshold": None, "record_scores": False})
    print(pos, threshold)
    for i in mal_files:
        pos,_=eval_surrogate(**{"path": i,
                     "model_path": "../models/ku/surrogate_ae",
                     "threshold": threshold, "record_scores": True})
        print(pos)

    train_params = []
    eval_params = []
    mal_params = []
    benign_pos_data = []
    threshold_data = []
    mal_pos_data = []
    num_packets_benign = {"Active Wiretap": 1355474,
                          "ARP MitM": 1358996,
                          "Fuzzing": 1811357,
                          "Mirai": 121622,
                          "OS Scan": 1632152,
                          "SSDP Flood": 2637663,
                          "SSL Renegotiation": 2114920,
                          "SYN DoS": 2764239,
                          "Video Injection": 2369903,
                          }
    for attack_name in sorted(num_packets_benign.keys()):
        # normalize input

        num_packets = num_packets_benign[attack_name] - 1
        train_param = {
            # the pcap, pcapng, or tsv file to process.
            "path": f"../experiment/kitsune/benign/{attack_name}.csv",
            # directory of kitsune
            "model_path": f"../models/kitsune/{attack_name}",
            "batch_size": 1024
        }

        eval_param = {"path": f"../experiment/kitsune/benign/{attack_name}.csv",
                      "model_path": f"../models/kitsune/{attack_name}",
                      "threshold": None, "out_image": f"../Kitsune Datasets/plots/benign/{attack_name}_surrogate.png", "record_scores": False}

        # train_surrogate(train_param)

        # benign_pos, threshold = eval_surrogate(**eval_param)

        # benign_pos_data.append(benign_pos)
        # threshold_data.append(threshold)
        threshold = 0.11068933550926413
        mal_param = {"path": f"../experiment/kitsune/malicious/{attack_name}.csv",
                     "model_path": f"../models/kitsune/{attack_name}",
                     "threshold": threshold, "out_image": f"../Kitsune Datasets/plots/malicious/{attack_name}_surrogate.png", "record_scores": True}

        # mal_pos=eval_surrogate(**mal_param)
        # mal_pos_data.append(mal_pos)

    # print("threshold:")
    # print(threshold_data)
    # print("benign_pos:")
    # print(benign_pos_data)
    # print("mal_pos")
    # print(mal_pos_data)
