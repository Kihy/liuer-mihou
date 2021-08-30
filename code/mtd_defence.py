from KitNET.KitNET import KitNET
import random
import numpy as np
from tqdm import tqdm
import pickle
import os
from kitsune import eval_kitsune, train_normal
from surrogate_model import AnomalyDetector, get_jacobian, eval_surrogate, get_gradient, train_surrogate
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from evaluations.feature_squeeze import squeeze_features
from vae import *
from sklearn.ensemble import IsolationForest
from rrcf import RCTree
import sklearn.metrics as metrics
import json
import copy

rng = np.random.default_rng(seed=0)


def draw_normal(x):
    mean = np.mean(x)
    std = np.std(x)
    return rng.normal(mean, std, x.shape)


def gaussian_fuzzing(ae, gamma):
    # weights
    for i in range(ae["W"].shape[1]):
        if rng.uniform() < gamma:
            ae["W"][:, i] = draw_normal(ae["W"][:, i])

    # bias
    if rng.uniform() < gamma:
        ae["hbias"] = draw_normal(ae["hbias"])
    if rng.uniform() < gamma:
        ae["vbias"] = draw_normal(ae["vbias"])


def neuron_activation_inverse(ae, gamma):
    n_neurons = ae["W"].shape[1]
    alter_rate = np.ceil(n_neurons * gamma)
    invert_index = rng.choice(
        n_neurons, size=int(alter_rate), replace=False)

    # invert neurons by index
    ae["W"][:, invert_index] = -ae["W"][:, invert_index]
    ae["vbias"][invert_index] = -ae["vbias"][invert_index]
    ae["hbias"][invert_index] = -ae["hbias"][invert_index]

    return ae


def process_kitsune(f, gamma, mutate_op):
    mutate_op(f["output"], gamma)
    for ae in f["ensemble"]:
        mutate_op(ae, gamma)


def weight_shuffle(ae, gamma):
    n_neurons = ae["W"].shape[1]
    alter_rate = np.ceil(n_neurons * gamma)
    # index of neurons to shuffle
    shuffle_index = rng.choice(
        n_neurons, size=int(alter_rate), replace=False)

    # intentionally ignore bias
    for i in shuffle_index:
        rng.shuffle(ae["W"][i])


def neuron_switch(ae, gamma):
    n_neurons = ae["W"].shape[1]
    alter_rate = np.ceil(n_neurons * gamma)
    # randomly select index of neurons to shuffle
    shuffle_index = rng.choice(
        n_neurons, size=int(alter_rate), replace=False)

    # intentionally ignore bias
    # order shuffled index
    ordered_index = sorted(shuffle_index)

    for i in range(len(ordered_index)):
        ae["W"][:, [ordered_index[i]]], ae["W"][:, [shuffle_index[i]]
                                                ] = ae["W"][:, [shuffle_index[i]]], ae["W"][:, [ordered_index[i]]],


def train_MTD_model_mutation_comb(params):
    with open(params["base_model_path"], "rb") as m:
        model = pickle.load(m)

    # save original model
    with open("{}/model_{}.pkl".format(params["model_path"], "original"), "wb") as of:
        pickle.dump(model, of)

    mutation_ops = [(weight_shuffle, "ws", 0, 1),
                    (neuron_switch, "ns", 0, 0.5),
                    #(neuron_activation_inverse, "nai", 0, 0.2),
                    (gaussian_fuzzing, "gf", 0.1, 0.5)]
    model_params = model.get_params()
    for i in range(params["num_models"] - 1):
        mutate_param = copy.deepcopy(model_params)
        # random combination of mutations
        spec_str = ""
        for j in range(params["mutate_steps"]):
            func, name, min, max = rng.choice(mutation_ops)

            gamma = rng.uniform(min, max)
            process_kitsune(mutate_param, gamma, func)
            spec_str += "_{}_{:.2f}".format(name, gamma)

        model.set_params(mutate_param)
        with open("{}/comb_model{}.pkl".format(params["model_path"], spec_str), "wb") as of:
            pickle.dump(model, of)


def train_MTD_model_mutation(params):
    with open(params["base_model_path"], "rb") as m:
        model = pickle.load(m)

    # save original model
    with open("{}/model_{}.pkl".format(params["model_path"], "original"), "wb") as of:
        pickle.dump(model, of)

    model_params = model.get_params()

    mutation_ops = [(weight_shuffle, "ws"), (neuron_switch, "ns"),
                    (neuron_activation_inverse, "nai"), (gaussian_fuzzing, "gf")]

    for gamma in np.arange(0.1, 1.1, 0.1):
        for func, name in mutation_ops:
            mutate_param = copy.deepcopy(model_params)
            process_kitsune(mutate_param, gamma, func)
            model.set_params(mutate_param)

            with open("{}/model_{}_{:.1f}.pkl".format(params["model_path"], name, gamma), "wb") as of:
                pickle.dump(model, of)


def train_vae(data, model_path):
    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    with open(model_path[:-3] + "_scaler.pkl", "wb") as of:
        pickle.dump(scaler, of)

    # prepare dataset
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.batch(batch_size)

    # vae parameters
    input_dim = 100
    intermediate_dim = 50
    latent_dim = 4

    vae = VariationalAutoEncoder(input_dim, intermediate_dim=intermediate_dim,
                                 latent_dim=latent_dim)

    vae.compile(optimizer='adam')
    vae.fit(train_dataset, shuffle=False, epochs=1, batch_size=batch_size)
    vae(tf.random.normal((1, 100)))
    # # save network
    tf.saved_model.save(vae, model_path)


def train_isolation_forest(data, model_path):
    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    with open(model_path[:-4] + "_scaler.pkl", "wb") as of:
        pickle.dump(scaler, of)
    clf = IsolationForest(contamination=0)

    clf.fit(data)

    with open(model_path, "wb") as of:
        pickle.dump(clf, of)


class RandomRobustCutForest:
    def __init__(self, num_trees, tree_size, load_forest=False):
        self.forest = []
        if load_forest:
            self.num_trees = load_forest["num_trees"]
            self.index = load_forest["index"]
            self.tree_size = load_forest["tree_size"]

            for tree in load_forest["trees"]:
                self.forest.append(RCTree.from_dict(tree))
        else:
            self.num_trees = num_trees
            self.tree_size = tree_size
            for _ in range(num_trees):
                tree = RCTree()
                self.forest.append(tree)
            self.index = 0

    def process(self, x):
        rmse = 0
        for tree in self.forest:
            # If tree is above permitted size, drop the oldest point (FIFO)
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self.index - self.tree_size)
            # Insert the new point into the tree
            tree.insert_point(x, index=self.index)
            rmse += tree.codisp(self.index) / self.num_trees
        self.index += 1
        return rmse

    def to_dict(self):
        return_dict = {"index": self.index,
                       "num_trees": self.num_trees,
                       "tree_size": self.tree_size,
                       "trees": []}
        for tree in self.forest:
            return_dict["trees"].append(tree.to_dict())
        return return_dict


def train_rrcf(data, model_path):
    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    with open(model_path[:-5] + "_scaler.pkl", "wb") as of:
        pickle.dump(scaler, of)

    forest = RandomRobustCutForest(10, 128)
    pbar = tqdm(total=data.shape[0])
    for i in data:
        forest.process(i)
        pbar.update(1)
        # For each tree in the forest...

    with open(model_path, "w") as of:
        json.dump(forest.to_dict(), of)


def train_MTDeep(params):
    dataframe = pd.read_csv(params["path"], header=0)
    raw_data = dataframe.values

    data = raw_data[:, :100]

    data = data.astype(np.float32)

    base_model_path = params["model_path"]

    params["model_path"] = base_model_path + "/autoencoder.h5"
    train_surrogate(params)
    params["model_path"] = base_model_path + "/kitsune.pkl"
    train_normal(params)
    train_vae(data, base_model_path + "/vae.h5")
    train_isolation_forest(data, base_model_path + "/isolation_forest.pkl")
    train_rrcf(data, base_model_path + "/rrcf.json")


def train_MTD_quantize(params):
    # full precision model
    K = KitNET(100, params["maxAE"], params["FMgrace"],
               params["ADgrace"], 0.1, 0.75, normalize=params["normalize"])
    kitsune_pool = [K]
    for i in params["quantize"]:
        kitsune_pool.append(KitNET(100, params["maxAE"], params["FMgrace"],
                                   params["ADgrace"], 0.1, 0.75, normalize=params["normalize"], quantize=[i, i]))

    input_file = open(params["path"], "r")
    input_file.readline()
    count = 0
    tbar = tqdm()
    while True:
        feature_vector = input_file.readline()
        # check EOF via empty string
        if not feature_vector:
            break
        fv = feature_vector.rstrip().split(",")
        fv = fv[:100]
        fv = np.array(fv, dtype="float")
        for i in kitsune_pool:
            i.process(fv)
        count += 1
        tbar.update(1)
        if count > params["FMgrace"] + params["ADgrace"]:
            break

    # save
    for i in range(len(kitsune_pool)):
        with open("{}/model{}.pkl".format(params["model_path"], i), "wb") as of:
            pickle.dump(kitsune_pool[i], of)


def train_MTD_FS(params):
    kitsune_pool = []
    autoencoder_pool = []
    for i in params["input_precision"]:
        kitsune_pool.append(KitNET(100, params["maxAE"], params["FMgrace"],
                                   params["ADgrace"], 0.1, 0.75, normalize=params["normalize"], input_precision=i))
        # autoencoder_pool.append(AnomalyDetector(ip=params["input_precision"][i]))

    # kitsune stuff
    input_file = open(params["path"], "r")
    input_file.readline()
    count = 0
    tbar = tqdm()
    while True:
        feature_vector = input_file.readline()
        # check EOF via empty string
        if not feature_vector:
            break
        fv = feature_vector.rstrip().split(",")

        fv = fv[:100]
        fv = np.array(fv, dtype="float")
        for i in kitsune_pool:
            i.process(fv)
        count += 1
        tbar.update(1)
        if count > params["FMgrace"] + params["ADgrace"]:
            break

    # save
    for i in range(len(kitsune_pool)):
        with open("{}/model{}.pkl".format(params["model_path"], i), "wb") as of:
            pickle.dump(kitsune_pool[i], of)


def train_MTD_orthogonal_AE(params):

    dataframe = pd.read_csv(params["path"], header=0)
    raw_data = dataframe.values

    # if there are 101 columns, last one is label

    data = raw_data[:, 0:100]

    data = data.astype(np.float32)

    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    # prepare dataset
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((data, data))
    train_dataset = train_dataset.batch(batch_size)

    # Instantiate a loss function.
    loss_fn = keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)
    cosine_similarity = tf.keras.losses.CosineSimilarity(
        reduction=tf.keras.losses.Reduction.NONE)
    epochs = 1
    models = []
    sim_loss = 0
    grad_norm = 0
    for i in range(params["num_models"]):
        if params["randomize_structure"]:
            num_layers = rng.integers(2, 4)
            structure = [(100, "sigmoid")]
            layer_size = sorted(rng.integers(
                2, 80, size=num_layers), reverse=True)
            for j in range(num_layers):
                structure.append((layer_size[j], "relu"))
            model = AnomalyDetector(structure)
        else:
            model = AnomalyDetector()
        # Instantiate an optimizer.
        optimizer = keras.optimizers.Adam()
        # training
        for epoch in range(epochs):
            pbar = tqdm(enumerate(train_dataset))
            for step, (x_batch_train, y_batch_train) in pbar:

                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)

                    # add magnitude of norm
                    g1 = get_gradient(model, x_batch_train)
                    grad_norm = tf.norm(g1, axis=1)
                    loss_value += grad_norm
                    if models:
                        # choose random model and calculate jacobian
                        random_model = rng.choice(models, 1)[0]
                        g2 = get_gradient(random_model, x_batch_train)
                        sim_loss = tf.math.abs(cosine_similarity(g1, g2))
                        loss_value += sim_loss

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                pbar.set_description(
                    "mean Loss {:.4f}, mean cosine similarity {:.4f}, gradient norm {:.4f}".format(tf.reduce_mean(loss_value), tf.reduce_mean(sim_loss), tf.reduce_mean(grad_norm)))
        with open("{}/model{}_scaler.pkl".format(params["model_path"], i), "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
        tf.saved_model.save(
            model, "{}/model{}.h5".format(params["model_path"], i))
        models.append(model)


def sample_pairwise_cos_sim(model_path, input_file):
    # load all models
    models = []
    for i in os.listdir(model_path):
        models.append(tf.keras.models.load_model(
            "{}/{}".format(model_path, i)))

    dataframe = pd.read_csv(input_file, header=0)
    raw_data = dataframe.values

    # if there are 101 columns, last one is label
    if raw_data.shape[1] != 100:
        labels = raw_data[:, -1]
        data = raw_data[:, 0:-1]
    else:
        data = raw_data

    data = data.astype(np.float32)

    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    # prepare dataset
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.batch(batch_size)

    cos_sim = []
    cosine_similarity = tf.keras.losses.CosineSimilarity(
        reduction=tf.keras.losses.Reduction.NONE)
    pbar = tqdm(enumerate(train_dataset))
    for step, x_batch_train in pbar:
        # randomly choose 2 models
        m1, m2 = rng.choice(models, 2, replace=False)
        # get jacobian
        j1 = get_gradient(m1, x_batch_train)
        j2 = get_gradient(m2, x_batch_train)
        cos_sim.extend(np.abs(cosine_similarity(j1, j2)))

    cos_sim = np.array(cos_sim)
    print(cos_sim.mean())
    print(cos_sim.std())
    plt.hist(cos_sim)
    plt.savefig("evaluations/fig/cos_sim_hist.png")


def gen_group_size(n, max_num):
    count = 0
    size_array = [0]
    while count < n:
        x = random.randint(1, max_num)
        count += x
        size_array.append(count)

    return size_array


def train_MTD_kit(params):
    K = KitNET(100, params["maxAE"], params["FMgrace"],
               params["ADgrace"], 0.1, 0.75, normalize=params["normalize"])
    kitsune_pool = [K]
    for i in range(params["num_models"] - 1):

        kitsune_pool.append(KitNET(100, random.randint(8, 16), params["FMgrace"],
                                   params["ADgrace"], random.uniform(0.1, 0.5), random.uniform(0.1, 0.9), normalize=params["normalize"]))

    input_file = open(params["path"], "r")
    input_file.readline()
    count = 0
    tbar = tqdm()
    while True:
        feature_vector = input_file.readline()
        # check EOF via empty string
        if not feature_vector:
            break
        fv = feature_vector.rstrip().split(",")
        fv = fv[:100]
        fv = np.array(fv, dtype="float")
        for i in kitsune_pool:
            i.process(fv)
        count += 1
        tbar.update(1)
        if count > params["FMgrace"] + params["ADgrace"]:
            break

    # save
    for i in range(len(kitsune_pool)):
        with open("{}/model{}.pkl".format(params["model_path"], i), "wb") as of:
            pickle.dump(kitsune_pool[i], of)


def train_MTD_FM(params):
    """
    train MTD models with randomized feature mapper

    Args:
        params (dict): set of training parameters.

    Returns:
        type: None, model is saved in model_path
    """
    # normal FM
    K = KitNET(100, params["maxAE"], params["FMgrace"],
               params["ADgrace"], 0.1, 0.75, normalize=params["normalize"])

    feature_list = [i for i in range(100)]
    kitsune_pool = [K]
    for i in range(params["num_models"] - 1):

        # randomly create feature mappping
        fl = random.sample(feature_list, k=100)
        fm = []
        size_array = gen_group_size(100, params["maxAE"])
        for j in range(1, len(size_array)):
            fm.append(fl[size_array[j - 1]:size_array[j]])

        kitsune_pool.append(KitNET(100, params["maxAE"], params["FMgrace"],
                                   params["ADgrace"], 0.1, 0.75, feature_map=fm, normalize=params["normalize"]))

    input_file = open(params["path"], "r")
    input_file.readline()
    count = 0
    tbar = tqdm()
    while True:
        feature_vector = input_file.readline()
        # check EOF via empty string
        if not feature_vector:
            break
        fv = feature_vector.rstrip().split(",")

        fv = fv[:100]
        fv = np.array(fv, dtype="float")
        for i in kitsune_pool:
            i.process(fv)
        count += 1
        tbar.update(1)
        if count > params["FMgrace"] + params["ADgrace"]:
            break
    #
    # save
    for i in range(len(kitsune_pool)):
        with open("{}/model{}.pkl".format(params["model_path"], i), "wb") as of:
            pickle.dump(kitsune_pool[i], of)


def evaluate_baseline_model(params, eval_func):
    """evaluate the benign, malicious and adversarial traffic on baseline model"""

    # evaluate on normal traffic, obtain threshold value for detection
    benign_pos, threshold = eval_func(params["normal_traffic_path"], params["model_path"],
                                      out_image="../uq_dataset/anomaly_plot/{}_benign.png".format(params["device_name"]))

    # evaluate malicious traffic, obtain number of positive samples and record ground truth label
    mal_pos, mal_auc = eval_func(params["malicious_traffic_path"], params["model_path"],
                                 threshold=threshold, record_prediction=True, out_image="../uq_dataset/anomaly_plot/{}_{}.png".format(params["device_name"], params["attack_name"]))

    y_true = np.genfromtxt(
        params["malicious_traffic_path"][:-4] + "_{}_prediction.csv".format(params["model_type"]), delimiter=",")

    # evaluate adversarial traffic, obtain number of positive samples for MER
    adv_pos, adv_auc = eval_func(params["avdersarial_traffic_path"], params["model_path"], threshold=threshold,
                                 out_image="../uq_dataset/anomaly_plot/{}_{}_adv.png".format(params["device_name"], params["attack_name"]))

    mer = 1 - adv_pos / mal_pos

    print("benign metrics:", ",".join(
        map(str, [threshold, benign_pos, mal_auc, mal_pos, adv_auc, adv_pos, mer])))


def eval_batch(path, model_path, threshold=None, out_image=None, ignore_index=0, record_scores=False, meta_file=None, record_prediction=False, y_true=None):
    """
    evaluates the model on some traffic in batches, used for models optimzed for parallel processing, e.g. Keras and Sklearn

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
    # it is keras model
    if os.path.isdir(model_path):
        model_name = "autoencoder"
        model = tf.keras.models.load_model(model_path)
        with open(model_path[:-3] + "_scaler.pkl", "rb") as m:
            scaler = pickle.load(m)
    else:
        with open(model_path, "rb") as m:
            model = pickle.load(m)
        model_name = "if"
        with open(model_path[:-4] + "_scaler.pkl", "rb") as m:
            scaler = pickle.load(m)

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

    data = raw_data[:, :100]

    data = scaler.transform(data)

    if model_path[-1].isdigit():
        data = squeeze_features(data, int(model_path[-1]))

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

        if model_name == "if":
            rmse = -model.decision_function(fv)
        else:
            reconstructions = model(fv)
            rmse = tf.keras.losses.mse(reconstructions, fv)

        rmse_array = np.concatenate((rmse_array, rmse))
        counter += fv.shape[0]
        tbar.update(1)

    if threshold == None:
        benignSample = rmse_array
        # benignSample=rmse_array
        mean = np.mean(benignSample)
        std = np.std(benignSample)
        threshold = mean + 3 * std
        print("mean {}, std {}".format(mean, std))
        print("threshold:", threshold)
        # threshold=np.percentile(rmse_array,99)

    num_over = (rmse_array > threshold).sum()

    if record_scores:
        score_path = path[:-4] + "_{}_score.csv".format(model_name)
        np.savetxt(score_path, np.log(rmse_array), delimiter=",")
        print("score saved to", score_path)

    # record prediction labels
    if record_prediction:
        pred_path = path[:-4] + "_{}_prediction.csv".format(model_name)
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
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.scatter(range(len(rmse_array)), rmse_array, s=0.2, c=colours)
        # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
        ax1.axhline(y=threshold, color='r', linestyle='-')
        # ax1.set_yscale("log")
        ax1.set_title(
            "Anomaly Scores from {}'s Execution Phase".format(model_path.split("/")[-1]))
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


def eval_iterative(path, model_path, threshold=None, ignore_index=-1, out_image=None, meta_file=None, record_scores=False, y_true=None, record_prediction=False):
    """
    evaluates model on some traffic with single features at a time, used for Kitsune and rrcf.

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
    print("model path ", model_path)
    t = threshold
    roc_auc = 1

    # rrcf has to be loaded differently
    if model_path.endswith(".json"):
        with open(model_path, "r") as m:
            json_obj = json.load(m)
        model_name = "rrcf"
        model = RandomRobustCutForest(10, 128, load_forest=json_obj)
        with open(model_path[:-5] + "_scaler.pkl", "rb") as m:
            scaler = pickle.load(m)
    else:
        with open(model_path, "rb") as m:
            model = pickle.load(m)
        model_name = "kitsune"

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

        fv = fv[:100]

        fv = np.array(fv, dtype="float")

        if model_name == "rrcf":
            fv = scaler.transform([fv])[0]
        rmse = model.process(fv)
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
        benignSample = rmse_array
        mean = np.mean(benignSample)
        std = np.std(benignSample)
        threshold = mean + 3 * std
        pos = (rmse_array > threshold).sum()
        print("mean {}, std {}".format(mean, std))
        print("threshold:", threshold)

    # record prediction scores/rmse
    if record_scores:
        score_path = path[:-4] + "_{}_score.csv".format(model_name)
        np.savetxt(score_path, rmse_array, delimiter=",")
        print("score saved to", score_path)

    # record prediction labels
    if record_prediction:
        pred_path = path[:-4] + "_{}_prediction.csv".format(model_name)
        np.savetxt(pred_path, rmse_array > threshold, delimiter=",")
        print("kitsune prediction saved to", pred_path)

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
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.scatter(range(len(rmse_array)), rmse_array, s=0.2, c=colours)
        # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
        ax1.axhline(y=threshold, color='r', linestyle='-')
        # ax1.set_yscale("log")
        ax1.set_title(
            "Anomaly Scores from {}'s Execution Phase".format(model_path.split("/")[-1]))
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
    tbar.close()
    if has_meta:
        return pos_mal, pos_craft, pos_ignore
    else:
        if t is None:
            return pos, threshold
        else:
            return pos, roc_auc


def eval_MTD_defence(params, eval_func=None):
    y_true = np.genfromtxt(
        params["malicious_traffic_path"][:-4] + "_{}_prediction.csv".format(params["model_type"]), delimiter=",")
    if eval_func is None:
        flag = True
    else:
        flag = False
    res = []
    for i in sorted(os.listdir(params["model_path"])):
        # skip scaler files
        if i.endswith("scaler.pkl"):
            continue

        model_path = "{}/{}".format(params["model_path"], i)
        # dynamically determine eval func
        if flag:
            # keras models saved in folders
            if os.path.isdir(model_path) or model_path.endswith("isolation_forest.pkl"):
                eval_func = eval_batch
            elif model_path.endswith("kitsune.pkl"):
                eval_func = eval_kitsune
            else:
                eval_func = eval_iterative

        benign_pos, threshold = eval_func(params["normal_traffic_path"], "{}/{}".format(params["model_path"], i),
                                          out_image="../uq_dataset/anomaly_plot/{}_{}_benign_{}.png".format(params["MTD_type"], params["device_name"], i))
        mal_pos, mal_auc = eval_func(params["malicious_traffic_path"], "{}/{}".format(params["model_path"], i), threshold=threshold,
                                     out_image="../uq_dataset/anomaly_plot/{}_{}_flooding_{}.png".format(params["MTD_type"], params["device_name"], i), y_true=y_true)
        adv_pos, adv_auc = eval_func(params["avdersarial_traffic_path"], "{}/{}".format(params["model_path"], i), threshold=threshold,
                                     out_image="../uq_dataset/anomaly_plot/{}_{}_flooding_adv_{}.png".format(params["MTD_type"], params["device_name"], i))
        if mal_pos != 0:
            mer = 1 - adv_pos / mal_pos
        else:
            mer = 0
        res.append([i, threshold, benign_pos, mal_auc,
                    mal_pos, adv_auc, adv_pos, mer])

    print("{} metrics:".format(params["MTD_type"]))
    for i in res:
        print(",".join(map(str, i)))


if __name__ == '__main__':
    MTD_type = "random_kit"
    train_func = train_MTD_kit
    model_type = "kitsune"
    device_name = "google_nest_mini"
    attack_name = "flooding"
    num_packets = 57749

    model_path = "../models/{}/{}/{}".format(MTD_type,
                                             device_name, attack_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    baseline_model_path = "../models/{}_{}_baseline.pkl".format(
        device_name, model_type)

    normal_traffic_path = "../experiment/traffic_shaping/init_pcap/{}_benign.csv".format(
        device_name)
    malicious_traffic_path = "../uq_dataset/{}_{}.csv".format(
        device_name, attack_name)
    avdersarial_traffic_path = "../experiment/traffic_shaping/{}_{}/craft/autoencoder_0.5_5_3_False_pso0.5/iter_1.csv".format(
        device_name, attack_name)

    baseline_train_params = {
        # the pcap, pcapng, or tsv file to process.
        "path": normal_traffic_path,
        "packet_limit": np.Inf,  # the number of packets to process,
        # KitNET params:
        "maxAE": 10,  # maximum size for any autoencoder in the ensemble layer
        # the number of instances taken to learn the feature mapping (the ensemble's architecture)
        "FMgrace": np.floor(0.2 * num_packets),
        # the number of instances used to train the anomaly detector (ensemble itself)
        # FMgrace+ADgrace<=num samples in normal traffic
        "ADgrace": np.floor(0.8 * num_packets),
        # directory of kitsune
        "model_path": baseline_model_path,
        "model_type": model_type,
        "normalize": True,
        "epochs": 1,
        "batch_size": 64,
    }

    MTD_train_params = {
        # the pcap, pcapng, or tsv file to process.
        "path": normal_traffic_path,
        "base_model_path": baseline_model_path,
        "packet_limit": np.Inf,  # the number of packets to process,
        "num_models": 10,
        "mutate_steps": 3,
        # KitNET params:
        "maxAE": 10,  # maximum size for any autoencoder in the ensemble layer
        # the number of instances taken to learn the feature mapping (the ensemble's architecture)
        "FMgrace": np.floor(0.2 * num_packets),
        # the number of instances used to train the anomaly detector (ensemble itself)
        # FMgrace+ADgrace<=num samples in normal traffic
        "ADgrace": np.floor(0.8 * num_packets),
        # directory of kitsune
        "model_path": model_path,
        "batch_size": 64,
        "normalize": True,
        "randomize_structure": False,
        "input_precision": [1, 2, 3, 4, 5],
        "quantize": [2, 3, 4, 5]
    }

    baseline_params = {
        "model_path": baseline_model_path,
        "normal_traffic_path": normal_traffic_path,
        "malicious_traffic_path": malicious_traffic_path,
        "avdersarial_traffic_path": avdersarial_traffic_path,
        "device_name": device_name,
        "model_type": model_type,
        "attack_name": attack_name,
    }
    MTD_eval_params = {
        "model_path": model_path,
        "normal_traffic_path": normal_traffic_path,
        "malicious_traffic_path": malicious_traffic_path,
        "avdersarial_traffic_path": avdersarial_traffic_path,
        "device_name": device_name,
        "attack_name": attack_name,
        "model_type": model_type,
        "MTD_type": MTD_type
    }

    # train_normal(baseline_train_params)
    # evaluate_baseline_model(baseline_params, eval_kitsune)

    # train_surrogate(baseline_train_params)
    # evaluate_baseline_model(baseline_params, eval_surrogate)

    # train_MTD_FM(MTD_train_params)
    # train_MTD_kit(MTD_train_params)
    start_time = time.time()
    train_func(MTD_train_params)
    time_taken = time.time() - start_time
    eval_MTD_defence(MTD_eval_params, eval_func=eval_kitsune)
    print("total training time", time_taken)
    # train_MTD_FS(MTD_train_params)
    # train_MTD_quantize(MTD_train_params)
    # train_MTDeep(MTD_train_params)
    # check_mutation(baseline_model_path,"../models/model_mutation/model_gf_1.0.pkl")
    # train_MTD_model_mutation_comb(MTD_train_params)
    # eval_MTD_defence(MTD_eval_params, eval_func=eval_kitsune)
    # print("total time for MTD creation", time_taken)
    # eval_MTD_defence(MTD_eval_params)
    # eval_cosine_similarity(train_params)
    # sample_pairwise_cos_sim(model_path, normal_traffic_path)
