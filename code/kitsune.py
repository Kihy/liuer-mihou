from textwrap import fill
import datetime
import matplotlib.ticker as ticker
import matplotlib.dates as mdate
import sklearn.metrics as metrics
from itertools import product
from tqdm import tqdm
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
import pickle
from KitNET.KitNET import KitNET
import matplotlib
import socket
from evaluations.feature_squeeze import squeeze_features
import multiprocessing as mp
matplotlib.use('Agg')
np.set_printoptions(threshold=np.inf)
# matplotlib.rcParams['timezone']="Pacific/Auckland"


def train_normal(params):
    """
    trains kitsune on normal traffic

    Args:
        params (dict): set of training parameters.

    Returns:
        type: None, model is saved in model_path


    """
    # Build Kitsune
    K = KitNET(params["n_features"], params["maxAE"], params["FMgrace"],
               params["ADgrace"], 0.1, 0.75, normalize=params["normalize"])

    input_file = open(params["path"], "r")
    input_file.readline()
    count = 0
    tbar = tqdm()
    rmse = []
    while True:
        feature_vector = input_file.readline()
        # check EOF via empty string
        if not feature_vector:
            break
        fv = feature_vector.rstrip().split(",")
        fv = fv[:100]
        fv = np.array(fv, dtype="float")
        K.process(fv)
        count += 1
        tbar.update(1)
        if count > params["FMgrace"] + params["ADgrace"]:
            break
    tbar.close()
    # save
    with open(params["model_path"], "wb") as of:
        pickle.dump(K, of)


def eval_kitsune(path, model_path, threshold=None, ignore_index=-1, out_image=None, meta_file=None, record_scores=False, y_true=None, record_prediction=False, load_prediction=False, plot_with_time=False):
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
    print("meta", meta_file)
    print("kitsune model path ", model_path)
    t = threshold
    roc_auc = 1
    label_map = []

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

    labels = []
    times = []
    colours = []
    tbar = tqdm()
    if load_prediction:
        rmse_array = np.genfromtxt(
            path[:-4] + "_kitsune_score.csv", delimiter=",")
    else:
        counter = 0
        input_file = open(path, "r")
        input_file.readline()
        rmse_array = []

        if not has_meta:
            colours = None

        feature_vector = input_file.readline()
        while feature_vector is not '':

            if counter < ignore_index:
                feature_vector = input_file.readline()

                if meta_file is not None:
                    meta_row = meta.readline()

                counter += 1
                continue

            fv = feature_vector.rstrip().split(",")

            if len(fv) == 102:
                label = fv[-2]
                if label.isdigit():
                    try:
                        label = socket.getservbyport(int(label))
                    except OSError:
                        label = "udp"
                if label not in label_map:
                    label_map.append(label)

                labels.append(label_map.index(label))
                times.append(mdate.epoch2num(float(fv[-1])) + 1)
                fv = fv[:100]

            fv = np.array(fv, dtype="float")

            try:
                if kitsune.input_precision is not None:
                    fv = squeeze_features(fv, kitsune.input_precision)
            except AttributeError as e:
                pass

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

    # if no threshold, calculate threshold
    if threshold == None:
        # threshold is min(mean+3std, max)
        benignSample = np.log(rmse_array)
        mean = np.mean(benignSample)
        std = np.std(benignSample)
        threshold_std = np.exp(mean + 3 * std)
        threshold_max = max(rmse_array)
        threshold = min(threshold_max, threshold_std)
        pos = (rmse_array > threshold).sum()

    # record prediction scores/rmse
    if record_scores:
        score_path = path[:-4] + "_kitsune_score.csv"
        threshold_path = path[:-4] + "_kitsune_threshold.csv"
        # print("max_rmse",np.max(rmse_array))
        np.savetxt(score_path, rmse_array, delimiter=",")
        np.savetxt(threshold_path, [threshold], delimiter=",")
        print("score saved to", score_path)

    # record prediction labels
    if record_prediction:
        pred_path = path[:-4] + "_kitsune_prediction.csv"
        np.savetxt(pred_path, rmse_array > threshold, delimiter=",")
        print("kitsune prediction saved to", pred_path)

    if y_true is None:

        fpr, tpr, roc_t = metrics.roc_curve(
            [0 for i in range(len(rmse_array))], rmse_array, drop_intermediate=False)
    else:
        fpr, tpr, roc_t = metrics.roc_curve(
            y_true, rmse_array, drop_intermediate=True)
        roc_auc = metrics.auc(fpr, tpr)
    print("total packets:", len(rmse_array))

    if out_image is not None:
        cmap = plt.get_cmap('Set3')
        num_packets = len(rmse_array)
        f, (ax1, ax2) = plt.subplots(
            2, 1, constrained_layout=True, figsize=(10, 10), dpi=200)

        if times and plot_with_time:
            x_val = times
            date_fmt = '%m/%d %H:%M:%S'

            date_formatter = mdate.DateFormatter(date_fmt)
            ax1.xaxis.set_major_formatter(date_formatter)

            # tick every 4 hours
            # print("asdfs")
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(1 / 6))

            ax1.tick_params(labelrotation=90)
            # f.autofmt_xdate()
        else:
            x_val = range(len(rmse_array))

        if labels:
            (unique, counts) = np.unique(labels, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            for i in frequencies:
                label_map[i[0]] = "{} {}".format(label_map[i[0]], i[1])

            scatter = ax1.scatter(x_val, rmse_array,
                                  s=1, c=labels, alpha=0.05, cmap=cmap)
            # wrap legends
            labels = [fill(l, 20) for l in label_map]

            leg = ax1.legend(handles=scatter.legend_elements()[0], labels=labels, bbox_to_anchor=(1.01, 1),
                             loc='upper left', borderaxespad=0.)
            for lh in leg.legendHandles:
                lh._legmarker.set_alpha(1.)

        elif has_meta:
            ax1.scatter(x_val, rmse_array, s=1, c=colours)
        else:
            ax1.scatter(x_val, rmse_array, s=1, alpha=0.05)

        # max_rmse=np.max(rmse_array)
        # print(max_rmse)

        ax1.axhline(y=threshold, color='r', linestyle='-')
        ax1.set_yscale("log")
        ax1.set_title("Anomaly Scores from Kitsune_{} Execution Phase".format(
            model_path.split("/")[-1]))
        ax1.set_ylabel("RMSE (log scaled)")
        if has_meta:
            ax1.set_xlabel(
                "packet index \n packets over threshold {}".format(pos_mal + pos_craft))
        else:
            ax1.set_xlabel(
                "packet index \n packets over threshold {}".format(pos))

        if y_true is None:
            ax2.plot(fpr, roc_t, 'b')
            ax2.set_ylabel("threshold")
            ax2.set_xlabel("false positive rate")
        else:
            ax2.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            ax2.set_title('AUC = %0.2f' % roc_auc)
            ax2.set_ylabel("true positive rate")
            ax2.set_xlabel("false positive rate")
        # plt.tight_layout()
        f.savefig(out_image)
        print("plot path:", out_image)
        plt.close()
    tbar.close()
    if has_meta:
        return pos_mal, pos_craft, pos_ignore
    else:
        if t is None:
            return pos, threshold
        else:
            return pos, roc_auc
