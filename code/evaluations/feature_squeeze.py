import sys
sys.path.append("..")

from KitNET.KitNET import KitNET
import pickle
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from itertools import product
import math

def squeeze_features(fv, precision):
    """rounds features to siginificant figures

    Args:
        fv (array): feature vector.
        precision (int): number of precisions to use.

    Returns:
        array: rounded array of floats.

    """
    fv_positive = np.where(np.isfinite(fv) & (fv != 0), np.abs(fv), 10**(precision-1))
    mags = 10 ** (precision - 1 - np.floor(np.log10(fv_positive)))
    return np.round(fv * mags) / mags

def draw_plot(list_to_draw, out_name, threshold=None):
    plt.scatter(np.arange(0, len(list_to_draw), 1),
                list_to_draw,
                s=10,
                label="RMSE")
    if threshold is not None:
        plt.axhline(threshold)
    plt.savefig(out_name)
    plt.clf()

def draw_hist(list_to_draw, out_name):
    plt.hist(list_to_draw)
    plt.yscale('log')
    plt.savefig('fig/{}.png'.format(out_name))
    plt.clf()

def eval_feature_squeeze(path, model_path, out_name, precision, threshold=None):
    """
    evaluates adversarial detection defenses on the trained model.

    Args:
        path (string): path to traffic feature file.
        model_path (string): path to trained kitsune model.

    Returns:
        if has_meta: return number of positive samples and positive samples that are not craft packets.
        else: return number of positive samples

    """
    # load kitsune model
    print("evaluting", path)
    print("kitsune model path ", model_path)
    with open(model_path, "rb") as m:
        kitsune = pickle.load(m)

    # load pcap file
    counter = 0
    input_file = open(path, "r")
    input_file.readline()
    rmse_array = []
    d_rmse=[]
    d_rel_rmse=[]

    tbar = tqdm()

    feature_vector = input_file.readline()
    while feature_vector is not '':
        fv = feature_vector.rstrip().split(",")

        #remove label if there is one

        fv = fv[:100]

        fv = np.array(fv, dtype="float")

        squeezed_fv=squeeze_features(fv, precision)

        rmse = kitsune.process(fv)
        squeezed_rmse=kitsune.process(squeezed_fv)
        d_rmse.append(np.abs(rmse-squeezed_rmse))
        d_rel_rmse.append(np.abs(rmse-squeezed_rmse)/rmse)

        if rmse == 0:
            rmse_array.append(1e-2)
        else:
            rmse_array.append(rmse)
        counter += 1
        tbar.update(1)

        feature_vector = input_file.readline()
    tbar.close()
    if not threshold:
        # calculate kitsune threshold and detector threshold and number of positive_samples
        kitsune_threshold=calc_threshold(rmse_array)
        pos_kit = (rmse_array > kitsune_threshold).sum()

        d_rmse_threshold=calc_threshold(d_rmse)
        pos_d_rmse = (d_rmse > d_rmse_threshold).sum()

        d_rel_rmse_threshold=calc_threshold(d_rel_rmse)
        pos_d_rel_rmse = (d_rel_rmse > d_rel_rmse_threshold).sum()
        return kitsune_threshold, pos_kit, d_rmse_threshold, pos_d_rmse, d_rel_rmse_threshold, pos_d_rel_rmse
    else:
        pos_kit = (rmse_array > threshold[0]).sum()


        pos_d_rmse = (d_rmse > threshold[1]).sum()

        pos_d_rel_rmse = (d_rel_rmse > threshold[2]).sum()

        return pos_kit, pos_d_rmse, pos_d_rel_rmse

def calc_threshold(array):
    benignSample = np.log(array)
    mean = np.mean(benignSample)
    std = np.std(benignSample)
    threshold_std = np.exp(mean + 3 * std)
    threshold_max = max(array)
    threshold = min(threshold_max, threshold_std)
    return threshold

if __name__ == '__main__':

    eval_defense("../../ku_dataset/train.csv", "../../models/kitsune_snu.pkl")
