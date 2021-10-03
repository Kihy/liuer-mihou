import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import matplotlib
import pickle
from tqdm import tqdm
from umap import UMAP
matplotlib.use('Agg')


def plot_with_umap(file_paths, labels):
    embedder = UMAP()

    marker = ["o", "v", "<", ">"]
    c = ["r", "g", "b", "y"]
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6), dpi=200)
    for index, path in enumerate(file_paths):
        dataset = pd.read_csv(path, header=0,
                              chunksize=10000000, usecols=list(range(100)))

        for chunk in tqdm(dataset):
            embedder = embedder.fit(chunk)

    for index, path in enumerate(file_paths):
        dataset = pd.read_csv(path, header=0,
                              chunksize=10000000, usecols=list(range(100)))

        for chunk in tqdm(dataset):
            low_dim = embedder.transform(chunk)
            ax.scatter(low_dim[:, 0], low_dim[:, 1], color=c[index],
                       alpha=0.1, s=1, marker=marker[index], label=labels[index])

            ax.legend(loc='upper left')
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    fig.savefig("evaluations/fig/latent_dim.png")


def reject_outliers(data, m=3.):
    distance = np.linalg.norm(data - np.median(data, axis=0), axis=1)
    d = np.abs(distance - np.median(distance))
    mdev = np.median(d)

    if mdev != 0.:
        s = d / mdev
        return data[s < m]
    else:
        return data


def plot_latent_dimension(file_paths, model_path, labels):
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

    with open(model_path + "_scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    marker = ["o", "v", "<", ">"]
    c = ["r", "g", "b", "y"]
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6), dpi=200)

    for index, path in enumerate(file_paths):
        dataset = pd.read_csv(path, header=0,
                              chunksize=10000000, usecols=list(range(100)))

        for chunk in tqdm(dataset):
            chunk = chunk.astype("float32")
            chunk = scaler.transform(chunk)
            latent_coord = autoencoder.encoder(chunk)
            # print(marker[index])
            # print(index)
            print(latent_coord.shape)
            latent_coord = reject_outliers(latent_coord)
            ax.scatter(latent_coord[:, 0], latent_coord[:, 2], color=c[index],
                       alpha=0.1, s=1, marker=marker[index], label=labels[index])
            ax.legend(loc='upper left')
    fig.savefig("evaluations/fig/latent_dim.png")

def gradient_wrt_output(log_files):
    difference_dict={}
    after_dict={}
    original_dict={}
    name_map={}
    name_count=0
    with open("../models/ku/surrogate_ae_all_traffic_scaler.pkl", "rb") as f:
        scaler=pickle.load(f)

    for log_file in log_files:
        config_str = ""
        report_str = ""
        log_list = []
        type = 0
        with open(log_file, "r") as lf:
            for line in lf.readlines():
                line = line.rstrip()
                if line.startswith("original rmse"):
                    type += 1
                    continue
                if line.startswith("-"):
                    continue
                if line.startswith("{'adv_mal_ratio'"):
                    type += 1

                if type == 0:
                    config_str += line
                elif type == 1:
                    log_list.append(line)
                elif type == 2:
                    report_str += line

        configs = eval(config_str)
        name_map[name_count]=configs["name"]
        report = eval(report_str)

        original_features = pd.read_csv(
            configs["malicious_file"][:-4] + "csv", header=0, usecols=list(range(100)))
        header=list(original_features.columns.values.tolist())
        adversarial_features = pd.read_csv(
            configs["adv_csv_file"], header=0, usecols=list(range(100)))
        for line in log_list:
            original_rmse, original_time, mal_file_index, craft_file_index, adv_pkt_index, best_cost, best_pos, aux = line.split(
                "\t")
            original_feature = original_features.iloc[int(mal_file_index)]
            original_feature=scaler.transform([original_feature])[0]
            adversarial_feature = adversarial_features.iloc[int(craft_file_index)]
            adversarial_feature=scaler.transform([adversarial_feature])[0]

            best_pos=np.fromstring(best_pos[1:-1],sep=" ", dtype=np.float64)

            best_pos[0]=np.around(best_pos[0]-float(original_time), decimals=2)
            best_pos[2]=np.around(best_pos[2], decimals=-2)

            if best_pos[1]==0.:
                best_pos=best_pos[:-1]

            best_pos=tuple(best_pos)
            diff_feature=original_feature-adversarial_feature
            if best_pos not in difference_dict.keys():
                difference_dict[best_pos]=[]
                after_dict[best_pos]=[]
                original_dict[best_pos]=[]
            diff_feature=np.append(diff_feature, name_count)
            adversarial_feature=np.append(adversarial_feature, name_count)
            original_feature=np.append(original_feature, name_count)
            difference_dict[best_pos].append(diff_feature)
            after_dict[best_pos].append(adversarial_feature)
            original_dict[best_pos].append(original_feature)
        name_count+=1

    lambdas={5:[],3:[],1:[],.1:[],.01:[]}
    cmap=cmap = plt.get_cmap('Set3')
    header_map={"jit":"jitter", "MI":"srcMAC-IP","H":"channel","Hp":"socket"}
    # remove time from header
    modified_header=[]

    for i, label in enumerate(header):
        _,name,Lambda ,metric=label.split("_")
        modified_header.append(f"{header_map[name]}_{metric}")
        lambdas[float(Lambda)].append(i)


    plot_figure(after_dict, "after", lambdas, modified_header,name_map)
    plot_figure(difference_dict, "diff", lambdas, modified_header,name_map)
    plot_figure(original_dict, "original", lambdas, modified_header,name_map)


def plot_figure(info_dict, name, lambdas, modified_header, name_map):
    cmap=cmap = plt.get_cmap('Dark2')
    for mutation, delta in info_dict.items():
        fig, ax = plt.subplots(figsize=(20,10))

        data=np.array(delta)
        x=np.array([modified_header for _ in range(len(delta))])

        for Lambda, indexes in lambdas.items():
            if Lambda !=0.01:
                continue

            scatter=ax.scatter(x[:,indexes].flatten(), data[:,indexes].flatten(), alpha=0.2, c = np.repeat(data[:,-1],20), cmap=cmap)

            leg = ax.legend(handles=scatter.legend_elements()[0], labels=[name_map[i] for i in np.unique(data[:,-1])], bbox_to_anchor=(1.01, 1),
                             loc='upper left', borderaxespad=0.)
            for lh in leg.legendHandles:
                lh._legmarker.set_alpha(1.)
                # ax.set_yscale("log")
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        fig.savefig(f"evaluations/fig/mutation_vis/{mutation}_{len(delta)}_{name}.png")
        fig.tight_layout()
        plt.close()


if __name__ == '__main__':
    labels = ["benign", "PS", "OD", "HF"]
    file_paths = ["../ku_dataset/google_home_normal.csv", "../ku_dataset/port_scan_attack_only.csv",
                  "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only.csv", "../ku_dataset/flooding_attack_only.csv"]
    model_path = "../models/ku/surrogate_ae_all_traffic"
    gradient_wrt_output(
        ["../experiment/traffic_shaping/ku_port_scan/logs/autoencoder_1_5_3_False_pso0.5/ku_port_scan_iter_0.txt",
        "../experiment/traffic_shaping/ku_flooding/logs/autoencoder_1_5_3_False_pso0.5/ku_flooding_iter_0.txt",
        "../experiment/traffic_shaping/ku_os_detection/logs/autoencoder_1_5_3_False_pso0.5/ku_os_detection_iter_0.txt"])
    # plot_latent_dimension(file_paths, model_path, labels)
    # plot_with_umap(file_paths, labels)
