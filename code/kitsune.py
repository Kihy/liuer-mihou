from KitNET.KitNET import KitNET
import pickle
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm


def train_normal(params):
    """
    trains kitsune on normal traffic

    Args:
        params (dict): set of training parameters.

    Returns:
        type: None, model is saved in model_path


    """
    # Build Kitsune
    K = KitNET(100, params["maxAE"], params["FMgrace"],
               params["ADgrace"], 0.1, 0.75)

    input_file = open(params["path"], "r")
    input_file.readline()
    count = 0
    tbar = tqdm()
    rmse = []
    while True:
        feature_vector = input_file.readline()
        fv = feature_vector.rstrip().split(",")
        if len(fv) == 101:
            fv = fv[:-1]
        fv = np.array(fv, dtype="float")
        res = K.process(fv)
        count += 1
        tbar.update(1)
        if count > params["FMgrace"] + params["ADgrace"]:
            break

    # save
    with open(params["model_path"], "wb") as of:
        pickle.dump(K, of)


def eval(path, model_path, threshold=None, ignore_index=-1, out_image=None, meta_file=None, record_scores=False):
    """
    evaluates trained kitsune model on some traffic.

    Args:
        path (string): path to traffic feature file.
        model_path (string): path to trained kitsune model.
        threshold (float): anomaly threshold value, if None it calculates the threshold value as 3 std away from mean. Defaults to None.
        ignore_index (int): number of features to ignore at the start. Defaults to -1.
        out_image (string): path to output anomaly score image. Defaults to None.
        meta_file (string): path to metadata file, used to calculate evasion metrics. Defaults to None.
        record_scores (boolean): whether to record anomaly scores in a seperate csv file. Defaults to False.

    Returns:
        if has_meta: return number of positive samples and positive samples that are not craft packets.
        else: return number of positive samples

    """
    # the pcap, pcapng, or tsv file to process.
    print("evaluting", path)
    print("kitsune model path ", model_path)
    with open(model_path, "rb") as m:
        kitsune = pickle.load(m)

    if out_image == None:
        out_image = path[:-4] + "_kitsune_rmse.png"

    if meta_file is not None:
        meta = open(meta_file, "r")
        meta.readline()
        meta_row = meta.readline()
        has_meta = True
        pos_craft = 0
        pos_mal = 0
        pos_ignore = 0
    else:
        has_meta = False
        pos = 0

    counter = 0
    input_file = open(path, "r")
    input_file.readline()
    rmse_array = []

    colours = []
    if not has_meta:
        colours = "b"

    tbar = tqdm()

    feature_vector = input_file.readline()
    while feature_vector is not '':
        if counter < ignore_index:
            feature_vector = input_file.readline()

            if meta_file is not None:
                meta_row = meta.readline()

            counter += 1
            continue

        fv = feature_vector.rstrip().split(",")

        if len(fv) == 101:
            index = fv[-1]
            fv = fv[:-1]

        fv = np.array(fv, dtype="float")
        rmse = kitsune.process(fv)
        if rmse == 0:
            rmse_array.append(1e-2)
        else:
            rmse_array.append(rmse)
        counter += 1
        tbar.update(1)

        feature_vector = input_file.readline()
        # set colours
        if has_meta:
            comment = meta_row.rstrip().split(",")[-1]
            if comment == "craft":
                colours.append([67 / 255., 67 / 255., 67 / 255., 0.8])

            elif comment == "malicious":
                colours.append([1, 0, 0, 1])
            else:
                colours.append([204 / 255., 243 / 255., 1, 0.5])

        if threshold is not None and rmse > threshold:
            if has_meta:
                comment = meta_row.rstrip().split(",")[-1]
                if comment == "craft":
                    pos_craft += 1
                elif comment == "malicious":
                    pos_mal += 1
                elif comment == "attacker_low":
                    pos_ignore += 1
                else:
                    print(meta_row)
                    print(rmse)
                    raise Exception
            else:
                pos += 1

        if has_meta:
            meta_row = meta.readline()

    if threshold == None:
        benignSample = np.log(rmse_array)
        mean = np.mean(benignSample)
        std = np.std(benignSample)
        threshold = np.exp(mean + 3 * std)
        print("mean {}, std {}".format(mean, std))
        print("threshold:", threshold)

    if record_scores:
        score_path = path[:-4] + "_kitsune_score.csv"
        np.savetxt(score_path, np.log(rmse_array), delimiter=",")
        print("score saved to", score_path)

    max_rmse = max(rmse_array)
    max_index = np.argmax(rmse_array)
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(rmse_array)), rmse_array, s=0.2, c=colours)
    # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.yscale("log")
    plt.title("Anomaly Scores from Kitsune's Execution Phase")
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("packet index")
    plt.tight_layout()
    plt.savefig(out_image)
    print("plot path:", out_image)
    if has_meta:
        return pos_mal, pos_craft, pos_ignore
    else:
        return pos


if __name__ == '__main__':
    model_path = "../models/kitsune.pkl"
    normal_traffic_path = "../experiment/traffic_shaping/init_pcap/google_home_normal.csv"

    train_params = {
        # the pcap, pcapng, or tsv file to process.
        "path": normal_traffic_path,
        "packet_limit": np.Inf,  # the number of packets to process,

        # KitNET params:
        "maxAE": 10,  # maximum size for any autoencoder in the ensemble layer
        # the number of instances taken to learn the feature mapping (the ensemble's architecture)
        "FMgrace": 5000,
        # the number of instances used to train the anomaly detector (ensemble itself)
        # FMgrace+ADgrace<=num samples in normal traffic
        "ADgrace": 9000,
        # directory of kitsune
        "model_path": model_path
    }
    # train_normal(train_params)

    # eval(normal_traffic_path, model_path, threshold=None,
    #      out_image="../ku_dataset/anomaly_plot/kitsune_normal.png", record_scores=True)

    # threshold 0.20288991702631556
    os_detect = "[OS & service detection]traffic_GoogleHome_av_only"
    flooding = "flooding_attacker_only"
    port_scan = "port_scan_attack_only"
    paths = [os_detect, flooding, port_scan]
    for path in paths:
        eval("../ku_dataset/{}.csv".format(path), model_path, threshold=0.20288991702631556,
             out_image="../ku_dataset/anomaly_plot/{}.png".format(path), record_scores=True)
