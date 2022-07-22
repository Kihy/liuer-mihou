import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import rrcf
import time
from tqdm import tqdm
from minisom import MiniSom
import scipy.sparse as sp
matplotlib.rcParams["figure.figsize"] = [15, 6]


def load_dataset(benign_path, malicious_path=None, adversarial_path=None, replay_path=None):

    scaler = MinMaxScaler()
    benign = np.genfromtxt(benign_path, delimiter=",", skip_header=1)[:, :57]
    # only take first 100 columns
    benign = scaler.fit_transform(benign)

    if malicious_path is not None:
        malicious = np.genfromtxt(
            malicious_path, delimiter=",", skip_header=1)[:, :57]
        malicious = scaler.transform(malicious)
    else:
        malicious = None
    if adversarial_path is not None:
        adversarial = np.genfromtxt(
            adversarial_path, delimiter=",", skip_header=1)[:, :57]
        adversarial = scaler.transform(adversarial)
    else:
        adversarial = None
    if replay_path is not None:
        replay = np.genfromtxt(
            replay_path, delimiter=",", skip_header=1)[:, :57]
        replay = scaler.transform(replay)
    else:
        replay = None
    return benign, malicious, adversarial, replay


def plot_result(scores, out_name, split=None, hline=None):

    plt.scatter(np.arange(0, len(scores), 1),
                scores,
                s=1,
                c='#8A977B',
                label="RMSE")
    if split:
        count = 0
        for i in split:
            count += i
            plt.axvline(x=count)
    if hline:
        plt.axhline(y=hline)

    plt.savefig(out_name)
    print("figure saved at:", out_name)
    plt.clf()


def eval_rrcf(benign, malicious, adversarial, replay, malicious_file, out_name):

    tree = rrcf.RCTree(benign)

    benign_scores = []
    for i in tqdm(benign):
        tree.insert_point(i, index="train")
        benign_scores.append(tree.codisp("train"))
        tree.forget_point("train")

    malicious_scores = []
    for i in tqdm(malicious):
        tree.insert_point(i, index="test")
        malicious_scores.append(tree.codisp("test"))
        tree.forget_point("test")

    adversarial_scores = []
    for i in tqdm(adversarial):
        tree.insert_point(i, index="test")
        adversarial_scores.append(tree.codisp("test"))
        tree.forget_point("test")

    replay_scores = []
    for i in tqdm(replay):
        tree.insert_point(i, index="test")
        replay_scores.append(tree.codisp("test"))
        tree.forget_point("test")

    threshold = np.exp(np.mean(np.log(benign_scores))
                       + 3 * np.std(np.log(benign_scores)))
    malicious_labels = malicious_scores > threshold
    benign_labels = benign_scores > threshold
    adversarial_labels = adversarial_scores > threshold
    replay_labels = replay_scores > threshold

    ben_pos = np.sum(benign_labels)
    num_pos = np.sum(malicious_labels)
    num_pos_adv = np.sum(adversarial_labels)
    num_pos_rep = np.sum(replay_labels)
    print(",".join(map(str, [malicious_file, "rrcf",
                             ben_pos, num_pos, num_pos_adv, num_pos_rep])))

    # plot_result(labels, out_name)

    # np.savetxt(malicious_file + "_rrcf_score.csv",
    #            malicious_scores, delimiter=",")
    np.savetxt(malicious_file + "_rrcf_threshold.csv",
               [threshold], delimiter=",")
    np.savetxt(malicious_file + "_adv_rrcf_score.csv",
               adversarial_scores, delimiter=",")
    np.savetxt(malicious_file + "_rep_rrcf_score.csv",
               replay_scores, delimiter=",")


def eval_ml(clf, benign, malicious, adversarial, replay, name, malicious_file, out_name):

    if name == "frocc":
        benign = sp.csc_matrix(benign, dtype=np.float32)

    clf.fit(benign)

    # in sk models, -1 is outlier and 1 is inlier, we want 0 to be inlier, 1 to be outlier

    # inverted since large values are inliers by default
    if name in ["lof", "ocsvm", "frocc", "if", "ee"]:
        invert = -1
    else:
        invert = 1
    benign_score = invert*clf.decision_function(benign)
    # decision function is shifted so that 0 is threshold
    if name == "frocc":
        threshold = np.mean(benign_score)+3*np.std(benign_score)
    else:
        threshold = 0

    np.savetxt(malicious_file + "_{}_threshold.csv".format(name),
               [threshold], delimiter=",")

    num_pos = np.sum(benign_score > threshold)
    if malicious is not None:
        if name == "frocc":
            malicious = sp.csc_matrix(malicious, dtype=np.float32)
        malicious_score = invert*clf.decision_function(malicious)
        np.savetxt(malicious_file + "_{}_score.csv".format(name),
                   malicious_score, delimiter=",")
        mal_pos = np.sum(malicious_score > threshold)
    else:
        mal_pos = None
    if adversarial is not None:
        if name == "frocc":
            adversarial = sp.csc_matrix(adversarial, dtype=np.float32)
        adversarial_score = invert*clf.decision_function(adversarial)
        np.savetxt(malicious_file + "_adv_{}_score.csv".format(name),
                   adversarial_score, delimiter=",")
        num_pos_adv = np.sum(adversarial_score > threshold)
    else:
        num_pos_adv = None
    if replay is not None:
        if name == "frocc":
            replay = sp.csc_matrix(replay, dtype=np.float32)
        replay_score = invert*clf.decision_function(replay)
        np.savetxt(malicious_file + "_rep_{}_score.csv".format(name),
                   replay_score, delimiter=",")
        num_pos_rep = np.sum(replay_score > threshold)
    else:
        num_pos_rep = None

    print(",".join(map(str, [malicious_file, name, threshold,
                             num_pos, mal_pos, num_pos_adv, num_pos_rep])))
    plot_result(np.concatenate((benign_score, malicious_score)),
                out_name, hline=threshold)

    # print(name)
    # print(clf.get_params())


def eval_som(benign, malicious, adversarial, replay, malicious_file, out_name):
    som = MiniSom(25, 25, 57, sigma=5, learning_rate=3)
    som.train(benign, 57)

    # plt.figure(figsize=(7, 7))
    # frequencies = som.activation_response(train)
    # plt.pcolor(frequencies.T, cmap='Blues')
    # plt.colorbar()
    # plt.savefig(name + "_train.png")
    # plt.clf()
    #
    # plt.figure(figsize=(7, 7))
    # frequencies = som.activation_response(test)
    # plt.pcolor(frequencies.T, cmap='Blues')
    # plt.colorbar()
    # plt.savefig(name + "_test.png")

    # train_x, train_y = zip(*[som.winner(d) for d in train])
    # train_x = np.array(train_x)
    # train_y = np.array(train_y)
    #
    # plt.scatter(train_x,
    #             train_y,
    #             s=50, c="blue", label="train")
    #
    # test_x, test_y = zip(*[som.winner(d) for d in test])
    # test_x = np.array(test_x)
    # test_y = np.array(test_y)
    # plt.scatter(test_x,
    #             test_y,
    #             s=50, c="red", label="test")

    benign_quantization_errors = np.linalg.norm(
        som.quantization(benign) - benign, axis=1)

    error_treshold = np.exp(np.mean(np.log(benign_quantization_errors))
                            + 3 * np.std(np.log(benign_quantization_errors)))

    # print('Error treshold:', error_treshold)
    adversarial_quantization_errors = np.linalg.norm(
        som.quantization(adversarial) - adversarial, axis=1)

    replay_quantization_errors = np.linalg.norm(
        som.quantization(replay) - replay, axis=1)

    malicious_quantization_error = np.linalg.norm(
        som.quantization(malicious) - malicious, axis=1)

    malicious_label = malicious_quantization_error > error_treshold
    benign_label = benign_quantization_errors > error_treshold
    adversarial_label = adversarial_quantization_errors > error_treshold
    replay_label = replay_quantization_errors > error_treshold

    ben_pos = np.sum(benign_label)
    num_pos = np.sum(malicious_label)
    num_pos_adv = np.sum(adversarial_label)
    num_pos_rep = np.sum(replay_label)

    print(",".join(map(str, [malicious_file, "som",
                             ben_pos, num_pos, num_pos_adv, num_pos_rep])))

    plot_result(np.concatenate((benign_quantization_errors, malicious_quantization_error,
                                adversarial_quantization_errors, replay_quantization_errors)), out_name, hline=error_treshold)

    np.savetxt(malicious_file + "_som_score.csv",
               malicious_quantization_error, delimiter=",")
    np.savetxt(malicious_file + "_adv_som_score.csv",
               adversarial_quantization_errors, delimiter=",")
    np.savetxt(malicious_file + "_rep_som_score.csv",
               replay_quantization_errors, delimiter=",")
    np.savetxt(malicious_file + "_som_threshold.csv", [error_treshold])


def eval_ml_models(benign, malicious, adversarial, replay, clfs, file_name, malicious_file):

    eval_som(benign, malicious, adversarial, replay,
             malicious_file, file_name + "_som")

    # for i in clfs.keys():
    #     eval_ml(clfs[i], benign, malicious, adversarial, replay, i,
    #             malicious_file, file_name + "_{}".format(i))

    eval_rrcf(benign, malicious, adversarial, replay,
              malicious_file, file_name + "_rrcf")
