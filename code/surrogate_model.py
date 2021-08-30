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

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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


def train_surrogate(params):
    """
    trains surrogate autoencoder on normal traffic

    Args:
        params (dict): set of training parameters.

    Returns:
        type: None, model is saved in model_path
    """

    dataframe = pd.read_csv(params["path"], header=0)
    raw_data = dataframe.values

    data = raw_data[:, :100]

    data = data.astype(np.float32)

    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    with open(params["model_path"][:-3] + "_scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
        print("scaler saved at", params["model_path"][:-3] + "_scaler.pkl")

    autoencoder = AnomalyDetector(structure=[(100, "sigmoid"), (32, "relu"), (8, "relu"), (2, "relu")])
    # autoencoder=MultipleAE()
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(data, data,
                              epochs=1,
                              batch_size=params["batch_size"],
                              shuffle=False)

    tf.saved_model.save(autoencoder, params["model_path"])


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
    autoencoder = tf.keras.models.load_model(
        model_path)
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
    dataframe = pd.read_csv(path, header=0)

    raw_data = dataframe.values[ignore_index:]

    data = raw_data[:, 0:100]
    with open(model_path[:-3] + "_scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    data = scaler.transform(data)

    # if model_path[-1].isdigit():
    #     data = squeeze_features(data, int(model_path[-1]))

    if out_image == None:
        out_image = path[:-4] + "_ae_rmse.png"

    counter = 0
    input_file = open(path, "r")
    input_file.readline()
    rmse_array = np.array([])

    chunks = np.ceil(data.shape[0] / 1024.)
    tbar = tqdm(total=chunks)
    for fv in np.array_split(data, chunks):

        fv = fv.astype(np.float32)

        reconstructions = autoencoder(fv)
        train_loss = tf.keras.losses.mse(reconstructions, fv)

        rmse_array = np.concatenate((rmse_array, train_loss))
        counter += fv.shape[0]
        tbar.update(1)
    tbar.close()
    if threshold == None:
        benignSample = np.log(rmse_array)
        mean = np.mean(benignSample)
        std = np.std(benignSample)
        threshold_std = np.exp(mean + 3 * std)
        threshold_max = np.max(rmse_array)
        threshold = min(threshold_max, threshold_std)

        # threshold=np.percentile(rmse_array,99)

    print("max rmse", np.max(rmse_array))
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
    surrogate_model_path = "../models/ku/google_home_surrogate_ae.h5"
    benign_file = "../ku_dataset/google_home_normal"
    train_params = {
        "path": benign_file + ".csv",
        "model_path": surrogate_model_path,
        "epochs": 10,
        "batch_size": 64
    }
    # train_surrogate(train_params)
    # _, threshold = eval_surrogate(benign_file + ".csv", surrogate_model_path,
    #                               out_image="../ku_dataset/anomaly_plot/google_home_surrogate.png")
    threshold=0.364
    print("surrogate threshold", threshold)

    malicious_files = ["../ku_dataset/port_scan_attack_only",
                       "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only",
                       "../ku_dataset/flooding_attacker_only"]
    adversarial_files = ["../experiment/craft_files/port_scan_autoencoder_craft_iter_1",
                         "../experiment/craft_files/os_autoencoder_craft_iter_1",
                         "../experiment/craft_files/flooding_autoencoder_craft_iter_1"]
    replay_files = [
        "../experiment/replay/ps_a_1",
        "../experiment/replay/os_a_1",
        "../experiment/replay/flooding_a_1"]

    for i in range(len(malicious_files)):
        pos_malicious, _ = eval_surrogate(malicious_files[i] + ".csv", surrogate_model_path, threshold=threshold,
                                          out_image="../ku_dataset/anomaly_plot/{}_surro_mal.png".format(malicious_files[i].split("/")[-1]), record_scores=True)
        pos_adversarial, _ = eval_surrogate(adversarial_files[i] + ".csv", surrogate_model_path, threshold=threshold,
                                            out_image="../ku_dataset/anomaly_plot/{}_surro_mal.png".format(adversarial_files[i].split("/")[-1]), record_scores=False)
        pos_replay, _ = eval_surrogate(replay_files[i] + ".csv", surrogate_model_path, threshold=threshold,
                                       out_image="../ku_dataset/anomaly_plot/{}_surro_mal.png".format(replay_files[i].split("/")[-1]), record_scores=False)
