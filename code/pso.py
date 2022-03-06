import numpy as np
import pyswarms as ps
from pprint import pformat
from pyswarms.utils.functions import single_obj as fx
from after_image.feature_extractor import *
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scapy.all import *
import json
import tensorflow as tf
from pyswarms.backend.topology import Random, Star, Ring
from topology.traffic import Traffic, create_swarm, plot_contour
from decimal import *
from surrogate_model import AnomalyDetector
import pickle
from itertools import product
from KitNET.KitNET import KitNET
from sklearn.metrics import mean_squared_error
from matplotlib import animation, rc
import matplotlib.pyplot as plt
# from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Animator, Designer
from pyswarms.backend.handlers import BoundaryHandler
import random
np.random.seed(0)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def plot_particle_position(pos_history, gbest, animation_path, limits, label, original_time):
    """

    plots the particle position during optimization

    Args:
        pos_history (array): history of the particle positions.
        gbest (array): position of the global best.
        animation_path (string): path to output gif file.
        limits (tuple): limit of each axis.
        label (tuple): label of axis.
        original_time (float): original time of the packet.

    Returns:
        type: Description of returned object.

    Raises:
        ExceptionName: Why the exception is raised.

    """
    designer = Designer(
        limits=limits, label=label, figsize=(4, 4)
    )

    animator = Animator(repeat=False, interval=200)

    rc('animation', html='html5')

    anim = plot_contour(pos_history=pos_history, designer=designer, animator=animator,
                        mark=gbest, original_time=original_time, s=5)
    anim.save(animation_path)
    plt.close('all')


np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:f}'.format}, linewidth=np.inf)
# np.set_printoptions(precision=18, floatmode="fixed")


def packet_gen(ori_pkt, traffic_vector):
    """
    generates craft packets with traffic_vector

    Args:
        ori_pkt (packet): original malicious packet.
        traffic_vector (list): the attributes of craft packet, [-2] is time and [-1] is payload size. The content is repeated "a"s.

    Returns:
        packet: a craft packet instance.


    """
    packet = ori_pkt.copy()
    packet.time = traffic_vector[-2]
    # print(traffic_vector)
    if packet.haslayer(TCP):
        packet[TCP].remove_payload()

        payload_size = int(traffic_vector[-1]) - len(packet)

        packet[TCP].add_payload(Raw(load="a" * payload_size))
        del packet[TCP].chksum
    # packet.show()

    elif packet.haslayer(UDP):
        packet[IP].remove_payload()

        packet[IP].add_payload(
            UDP(sport=int(traffic_vector[4]), dport=int(traffic_vector[6])))

        payload_size = int(traffic_vector[-1]) - len(packet)
        packet[UDP].add_payload(Raw(load="a" * payload_size))

    elif packet.haslayer(ARP):
        if packet.haslayer(IP):
            packet[IP].remove_payload()

            packet[IP].add_payload(
                ARP(psrc=traffic_vector[3], pdst=traffic_vector[5], hwsrc=traffic_vector[1], hwdst=traffic_vector[2]))

        elif packet.haslayer(Ether):
            packet[Ether].remove_payload()

            packet[Ether].add_payload(
                ARP(psrc=traffic_vector[3], pdst=traffic_vector[5], hwsrc=traffic_vector[1], hwdst=traffic_vector[2]))


        payload_size = int(traffic_vector[-1]) - len(packet)
        packet[ARP].add_payload(Raw(load="a" * payload_size))
    # other packets
    else:
        packet[IP].remove_payload()
        payload_size = int(traffic_vector[-1]) - len(packet)
        packet.add_payload(Raw(load="a" * payload_size))
    # packet=Ether(bytes(packet))
    if packet.haslayer(IP):
        del packet[IP].len
        del packet[IP].chksum
        del packet.len

    # print(len(packet))
    return packet


def build_craft(timestamp, num_craft, prev_pkt_time, max_craft, n_dims, max_pkt_size=None, min_pkt_len=None, seed=None):
    """
    builds attributes of craft packets. essentially the inverse vectorization function.

    Args:
        timestamp (array): timestamp of the malicious packet.
        num_craft (array): number of craft packets to be generated.
        prev_pkt_time (float): time of the previous packet.
        max_craft (int): maximum number of craft packets.
        n_dims (int): dimensions of the particles.
        max_pkt_size (array or int): maximum size of the craft packet. Defaults to None.
        min_pkt_len (int): minimum packet size. Defaults to None.
        seed (float): seed used in seeded vectorization. Defaults to None.

    Returns:
        array: 2d array of timestamp and payload size of craft packets

    """
    num_particles = timestamp.shape[0]

    # prev_pkt_time=np.full(timestamp.shape, prev_pkt_time)
    # evenly distribute craftpackets
    t = np.zeros((num_particles, max_craft))
    l = np.zeros((num_particles, max_craft))
    mask = np.full((num_particles, max_craft), True)

    for i in range(num_particles):
        mask[i][:num_craft[i]] = False
        if num_craft[i] == 0:
            continue

        if n_dims == 3:
            # indexing to remove start point
            t[i][:num_craft[i]] = np.linspace(
                prev_pkt_time, timestamp[i], num_craft[i] + 1, endpoint=False)[1:]
            t[i][:num_craft[i]] = np.around(t[i][:num_craft[i]], decimals=6)

        else:
            if seed is not None:
                # fixed time interval
                t[i][:num_craft[i]] = np.linspace(
                    prev_pkt_time, timestamp[i], num_craft[i] + 1, endpoint=False)[1:]
                t[i][:num_craft[i]] = np.around(
                    t[i][:num_craft[i]], decimals=6)

                # random seed
                random.seed(seed[i])
                l[i] = np.array([random.randint(min_pkt_len, max_pkt_size)
                                 for _ in range(max_craft)])

    if n_dims == 3:
        l = np.tile(np.expand_dims(max_pkt_size, axis=1), (1, max_craft))

    elif seed is None:
        t = np.random.uniform(prev_pkt_time, np.expand_dims(
            timestamp, axis=1), size=(num_particles, max_craft))
        l = np.random.randint(min_pkt_len, max_pkt_size,
                              size=(num_particles, max_craft))

    t = np.ma.masked_array(t, mask=mask)
    t = np.sort(t)

    # if min_pkt_len is int then it is 2d
    l = np.ma.masked_array(l, mask=mask)

    return np.concatenate((t.filled(0), l.filled(0)), axis=1)


def f(x, decision_func, decision_type, FE, use_seed, prev_pkt_time, max_pkt_size, traffic_data, max_craft_pkt, min_pkt_len, db):
    """

    calculates cost function of optimization. first finds the craft packet size and time, then update a copy of netstat to find the maximum anomaly score.

    Args:
        x (array): particle position, x[:,0] is time and x[:,1] is number of crafted packets.
        decision_func (function): function to calculate anomaly score given input
        decision_type (string): whether the decision_func uses kitsune or surrogate.
        FE (feature_extractor): the feature extractor used.
        use_seed (boolean): whether to use seed to create craft packets.
        prev_pkt_time (float): time of previous packet.
        max_pkt_size (int): maximum size of craft packet.
        traffic_data (list): information about the original malicious packet.
        max_craft_pkt (int): maximum number of craft packet.
        min_pkt_len (int): minimum size of craft packet.
        db (netstat): netstat information.


    Returns:
        cost: array of costs corresponding to each particle.
        craft_batch: array of the generated craft packets.

    """

    # copy and round x
    tmp_x = x.copy()
    tmp_x[:, 1:] = np.rint(tmp_x[:, 1:])

    n_particles, n_dims = tmp_x.shape
    min_dists = np.zeros(n_particles)

    timestamps = tmp_x[:, 0]
    num_craft = tmp_x[:, 1].astype(int)

    seed = np.around(x[:, 1], decimals=1)
    # num_craft=np.zeros(x[:,1].shape).astype(int)

    if n_dims == 3:
        pkt_sizes = tmp_x[:, 2].astype(int)
        craft_batch = build_craft(timestamps, num_craft, prev_pkt_time,
                                  max_craft_pkt, n_dims, max_pkt_size=pkt_sizes, min_pkt_len=pkt_sizes)

    if n_dims == 2:
        if use_seed:
            craft_batch = build_craft(timestamps, num_craft, prev_pkt_time,
                                      max_craft_pkt, n_dims, max_pkt_size=max_pkt_size, min_pkt_len=min_pkt_len, seed=seed)

        else:
            craft_batch = build_craft(timestamps, num_craft, prev_pkt_time,
                                      max_craft_pkt, n_dims, max_pkt_size=max_pkt_size, min_pkt_len=min_pkt_len)

    # features is 2d array
    features = [[] for i in range(n_particles)]

    for i in range(n_particles):

        dummy_db = copy.deepcopy(db)

        for j in range(max_craft_pkt):
            if j < num_craft[i]:
                time = craft_batch[i][j]
                size = craft_batch[i][j + max_craft_pkt]
                traffic_vector = traffic_data[:-2] + [time, size]

                f = FE.nstat.update_dummy_db(*traffic_vector, dummy_db)

                if decision_type == "kitsune":
                    features[i].append(decision_func(f))
                elif decision_type == "autoencoder":
                    features[i].append(f)

            else:
                if decision_type == "kitsune":
                    features[i].append(0)
                elif decision_type == "autoencoder":
                    features[i].append(np.full((100,), np.nan))

        # malicious packet
        traffic_data[-2] = timestamps[i]
        feature = FE.nstat.update_dummy_db(*(traffic_data), dummy_db)

        if decision_type == "kitsune":
            features[i].append(decision_func(feature))
        elif decision_type == "autoencoder":
            features[i].append(feature)

    if decision_type == "kitsune":
        cost = np.max(features, axis=1)
    elif decision_type == "autoencoder":
        features = np.array(features)
        # features = np.reshape(features, (n_particles * max_craft_pkt, -1))
        features = np.reshape(features, (-1,100))

        costs = decision_func(features)
        costs = np.reshape(costs, (n_particles, max_craft_pkt+1))

        cost = np.max(np.nan_to_num(costs, 0), axis=1)

    return cost, craft_batch


def craft_adversary(mal_pcap, init_pcap, adv_pcap, mal_pcap_out, decision_type, threshold, model_path, iteration, meta_path=None, use_seed=False, optimizer=None, init_count=0, netstat_path=None, mutate_prob=-1, base_offset=0, log_file=None, n_dims=2, max_time_window=60, max_craft_pkt=5, max_pkt_size=655, max_adv_pkt=None, adv_csv_file=None, animation_folder=None, netstat_log_file=None):
    """
    crafts adversarial samples for some attack. first initializes the feature extract with normal traffic. then iterates through
    malicious attack pcap file to find a malcious packet above threshold. The malicious packet is optimized with search algorithm to
    have minimum cost value. The best configuration of adversarial packet is written to output pcap.

    Args:
        mal_pcap (string): path to malicious traffic pcap file.
        init_pcap (string): path to normal traffic pcap file.
        adv_pcap (string): output path to entire pcap file.
        mal_pcap_out (string): path to pcap file containing no init_pcap packets.
        decision_type (string): either kitsune or surrogate.
        threshold (float): threshold value for detection.
        model_path (string): path to model.
        iteration (int): interation number of attack, mainly used for naming.
        meta_path (string): path to metadata file, e.g. whether packet is ignored or craft. Defaults to None.
        use_seed (boolean): whether to seed RNG. Defaults to False.
        optimizer (tuple): parameters for search algorithm. Defaults to None.
        init_count (int): number of normal init packet to include. Defaults to 0.
        netstat_path (string): path to a serialized netstat object, used to shorten initialization phase. Defaults to None.
        mutate_prob (float): probability of using DE in search algorithm, if -1 it is pure PSO, if 1 it is pure DE. Defaults to -1.
        base_offset (float): offset time to cover the time difference between malicious and normal file. Defaults to 0.
        log_file (string): path to log file with various stats on generated samples. Defaults to None.
        n_dims (int): dimensions of particle. Defaults to 2.
        max_time_window (float): maximum delay time for malicious packet. Defaults to 60.
        max_craft_pkt (int): maximum number of craft packets. Defaults to 5.
        max_pkt_size (int): maximum size of craft packets. Defaults to 655.
        max_adv_pkt (int): maximum number of malicious packet to alter. mainly used for flooding attacks to reduce time. Defaults to None.
        adv_csv_file (string): used to shorten initialization phase. Defaults to None.
        animation_folder (string): path to particle position folder. Defaults to None.

    Returns:
        dict: report of various stats.


    """

    # output file with normal, craft and malicious pcap
    craft_pcap = PcapWriter(adv_pcap)
    log_file.write(
        "original rmse\t original time\t mal file index\t craft file index\t adv pkt index\t best cost\t best pos\t aux\n")

    # output file with no init packets
    malicious_pcap = PcapWriter(mal_pcap_out)

    meta_file = open(meta_path, "w")
    meta_file.write("packet_index,time,comment\n")

    decision_func = get_decision_func(decision_type, model_path)

    # whether to write features output directly to csv file



    if netstat_path is not None:
        with open(netstat_path, "rb") as m:
            nstat = pickle.load(m)
            init_count = nstat.num_updated
            print("loading netstat from",netstat_path)
            # print("init count", init_count)

    else:

        # init with benign packets
        init_extractor = FE(init_pcap, parse_type="scapy")

        t = tqdm(total=init_count)
        pkt_index = 0
        while pkt_index < init_count:
            try:
                traffic_data, packet = init_extractor.get_next_vector()
            except EOFError as e:
                print("EOF Reached")
                break
            t.update(1)
            pkt_index += 1
            if traffic_data == []:
                craft_pcap.write(packet)
                
                meta_file.write(
                    ",".join([str(pkt_index), str(packet.time), "init_skipped\n"]))
                continue

            features = init_extractor.nstat.updateGetStats(*traffic_data)
            # init_extractor.dummy_nstat.updateGetStats(*traffic_data)
            # write init packets as is
            meta_file.write(
                ",".join([str(pkt_index), str(packet.time), "init\n"]))
            craft_pcap.write(packet)

        prev_pkt_time = float(packet.time)

        # get the database from initial fe to malware fe
        nstat = init_extractor.get_nstat()


    pkt_index = init_count
    prev_non_attack_time = None
    prev_pkt_time = nstat.prev_pkt_time
    feature_extractor = FE(mal_pcap, parse_type="scapy", nstat=nstat, log_file=netstat_log_file)

    if adv_csv_file is not None:
        write_to_csv = True

        output_csv = open(adv_csv_file, "w")
        original_csv = open(adv_csv_file[:-4]+"_original.csv", "w")
        headers = feature_extractor.nstat.getNetStatHeaders()
        # print(headers)
        output_csv.write(",".join(list(map(str, headers))))
        output_csv.write("\n")
        original_csv.write(",".join(list(map(str, headers))))
        original_csv.write("\n")
    else:
        write_to_csv=False

    # Set-up hyperparameters
    options = {'c1': 0.7, 'c2': 0.3, 'w': 0.5}

    t = tqdm(total=max_adv_pkt)

    # the base amount to adjust flooding traffic
    offset_time = base_offset

    total_reduction_ratio = 0
    total_craft = 0
    adv_pkt_index = 0
    num_failed = 0
    craft_failed = 0
    total_craft_size = 0
    total_std = np.zeros((n_dims,))
    error_index = []
    while adv_pkt_index < max_adv_pkt:
        try:
            traffic_data, packet = feature_extractor.get_next_vector()
        except EOFError as e:
            print("EOF Reached")
            break

        pkt_index += 1

        if traffic_data == []:
            packet.time = float(packet.time) + offset_time
            craft_pcap.write(packet)
            malicious_pcap.write(packet)
            prev_pkt_time = float(packet.time)
            np.savetxt(output_csv, [np.full(
                features.shape, -1)], delimiter=",")
            meta_file.write(
                ",".join([str(pkt_index), str(packet.time), "mal_skipped\n"]))
            continue

        tmp_pkt = packet.copy()
        if tmp_pkt.haslayer(TCP):
            tmp_pkt[TCP].remove_payload()
        elif tmp_pkt.haslayer(UDP):
            tmp_pkt[UDP].remove_payload()
        elif tmp_pkt.haslayer(ARP):
            tmp_pkt[ARP].remove_payload()
        else:
            tmp_pkt.remove_payload()
        min_pkt_len = len(tmp_pkt)

        # get records corresponding to current connection
        db = feature_extractor.nstat.get_records(*traffic_data)

        # find original score
        dummy_db = copy.deepcopy(db)
        traffic_data[-2] += offset_time
        traffic_data[-2] = np.around(traffic_data[-2], decimals=6)
        features = feature_extractor.nstat.update_dummy_db(
            *(traffic_data), dummy_db, False)

        if write_to_csv:
            original_csv.write(",".join(list(map(str, features))))
            original_csv.write("\n")

        rmse_original = decision_func(features)

        if rmse_original < threshold:

            packet.time = traffic_data[-2]
            craft_pcap.write(packet)
            malicious_pcap.write(packet)
            features = feature_extractor.nstat.updateGetStats(*traffic_data)

            if write_to_csv:
                row = features
                np.savetxt(output_csv, [row], delimiter=",")
            # feature_extractor.dummy_nstat.updateGetStats(*traffic_data)
            #
            meta_file.write(
                ",".join([str(pkt_index), str(packet.time), "attacker_low\n"]))
            prev_pkt_time = float(packet.time)
            continue

        original_time = traffic_data[-2]
        print("original RMSE", rmse_original)
        print("original time", original_time)

        # set bounds:

        # max_craft_pkt=int((max_time-prev_pkt_time)/0.0002)
        max_bound = [original_time + max_time_window, max_craft_pkt]
        min_bound = [original_time, 0]
        if n_dims == 3:
            max_bound.append(max_pkt_size)
            min_bound.append(min_pkt_len)
        bounds = [np.array(min_bound), np.array(max_bound)]

        # PSO
        args = {"FE": feature_extractor, "min_pkt_len": min_pkt_len, "decision_type": decision_type,
                "prev_pkt_time": prev_pkt_time, "decision_func": decision_func, "use_seed": use_seed, "max_pkt_size": max_pkt_size, "traffic_data": traffic_data, "max_craft_pkt": max_craft_pkt, "db": db}

        mal_file_index = pkt_index - total_craft - init_count
        print("optimizing mal file index: ", mal_file_index)
        print("optimizing craft file index: ", pkt_index)

        iterations = 30

        if optimizer == "pso":
            cost, pos, aux, std, pos_history = optimize(
                options, n_dims, iterations, f, args, bounds, original_time, mutate_prob=mutate_prob)
        if optimizer == "de":
            cost, pos, aux, std, pos_history = differential_evolution(
                n_dims, iterations, f, args, bounds, original_time)

        pos[1:] = np.rint(pos[1:])
        print(cost, pos)

        log_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            rmse_original, original_time, mal_file_index, pkt_index, adv_pkt_index, cost, pos, aux))
        # sum reduction ratio
        total_reduction_ratio += (rmse_original - cost) / rmse_original

        # record number of failed packets
        if cost > threshold:
            num_failed += 1

        if np.random.uniform() < 0.001 or adv_pkt_index == 0:

            limits = [(0, max_time_window), (min_bound[1], max_bound[1])]
            label = ["time", "num_craft"]

            animaition_path = animation_folder + \
                "_{}.gif".format(pkt_index - total_craft - init_count)
            plot_particle_position(
                pos_history, pos, animaition_path, limits, label, original_time)

        log_file.write("-" * 100 + "\n")
        # print(aux)
        total_std += std

        # apply fake packets to feature extractor
        num_craft = pos[1].astype(int)
        total_craft += num_craft

        craft_costs = []
        for i in range(num_craft):

            traffic_vector = traffic_data[:-2] + \
                [aux[i], int(aux[i + max_craft_pkt])]
            features = feature_extractor.nstat.updateGetStats(*traffic_vector)

            # print("real craft feature", features)
            craft_cost = decision_func(features)

            craft_costs.append(craft_cost)
            if craft_cost > threshold:
                craft_failed += 1

            try:
                craft_packet = packet_gen(packet, traffic_vector)
            except Exception as e:
                print("Packet")
                packet.show2()
                print("packet_index", mal_file_index)
                print("traffic_vector",traffic_vector)
                print("file:", mal_pcap)
                raise e


            total_craft_size += aux[i + max_craft_pkt]
            # print(traffic_vector)
            # craft_packet.show()
            # packet.show()

            malicious_pcap.write(craft_packet)
            craft_pcap.write(craft_packet)
            if write_to_csv:
                np.savetxt(output_csv, [features], delimiter=",")
            meta_file.write(
                ",".join([str(pkt_index), str(craft_packet.time), "craft\n"]))

            pkt_index += 1
        t.update(1)

        # set offset
        offset_time += (pos[0] - original_time)

        # write malicious packet
        traffic_data[-2] = pos[0]

        features = feature_extractor.nstat.updateGetStats(*traffic_data)

        # print("real mal feature", features)
        if write_to_csv:
            np.savetxt(output_csv, [features], delimiter=",")

        true_cost = decision_func(features)

        craft_costs.append(true_cost)
        try:
            np.testing.assert_almost_equal(
                np.max(craft_costs), cost, decimal=6)
        except AssertionError as e:
            print("true cost", true_cost)
            print("cost", cost)
            print("pos", pos)
            print(pkt_index)

            raise

        # feature_extractor.dummy_nstat.updateGetStats(*traffic_data)
        packet.time = pos[0]
        craft_pcap.write(packet)
        malicious_pcap.write(packet)
        meta_file.write(
            ",".join([str(pkt_index), str(packet.time), "malicious\n"]))
        adv_pkt_index += 1
        prev_pkt_time = pos[0]

    report = {}
    if adv_pkt_index == 0:
        report["num_altered"] = 0
    else:
        report["av_time_delay"] = (offset_time - base_offset) / adv_pkt_index
        report["av_num_craft"] = total_craft / adv_pkt_index
        report["total_craft"] = total_craft
        report["num_altered"] = adv_pkt_index
        report["average_reduction_ratio"] = total_reduction_ratio / adv_pkt_index
        report["adv_mal_ratio"] = (
            pkt_index - init_count - total_craft) / adv_pkt_index
        report["av_std"] = total_std / adv_pkt_index
        report["av_pkt_size"] = total_craft_size / total_craft
        report["num_seen"] = pkt_index - total_craft - init_count
    report["num_failed"] = num_failed
    report["craft_failed"] = craft_failed
    return report


def get_decision_func(decision_type, model_path):
    """
    returns a function to calculate anomaly score given features
    Args:
        decision_type (string): either kitsune or surrogate.
        model_path (string): path to saved model.

    Returns:
        function: decision function.

    """
    if decision_type == "kitsune":
        with open(model_path, "rb") as m:
            model = pickle.load(m)

        def kitsune(features):
            # process batch
            if features.ndim == 2:
                rmse = []
                for i in range(features.shape[0]):
                    rmse.append(model.process(features[i]))
                return np.array(rmse)
            else:
                return model.process(features)
        return kitsune

    if decision_type == "autoencoder":
        model = tf.keras.models.load_model(model_path, custom_objects={"AnomalyDetector":AnomalyDetector})

        with open(model_path+"_scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        def autoencoder(features):
            ndims = features.ndim
            if ndims == 1:
                features = np.expand_dims(features, axis=0)
            features = scaler.transform(features)
            features = features.astype(np.float32)
            anomaly_score = model(features)

            if ndims == 1:
                anomaly_score = anomaly_score[0]
            return anomaly_score.numpy()
        return autoencoder


def optimize(options, n_dims, iterations, f, args, bounds, original_time, clamp=None, mutate_prob=-1):
    """
    searches for optimial particle
    Args:
        options (dict): hyperparameters for PSO-DE algorithm.
        n_dims (int): dimensions of particle.
        iterations (int): number of iterations to search.
        f (func): cost function.
        args (dict): arguments to f.
        bounds (tuple): boundaries for each dimension of search space.
        original_time (float): original time of malicious packet. so that the new position cannot be worse.
        clamp (VelocityClamp): pyswarms method to clamp velocity. Defaults to None.
        mutate_prob (float): probability to run DE. Defaults to -1.

    Returns:
        best cost, best position, and craft packet configuration of the best cost.
        also return std of final positions to see convergence and position histories of each particle
        for plotting.


    """
    topology = Traffic()

    pos_history = []

    n_particles = 20

    # de params
    mutation_factor = 0.8
    crossp = 0.7
    mutation_candidates = [[idx for idx in range(
        n_particles) if idx != i] for i in range(n_particles)]
    mutation_candidates = np.array(mutation_candidates)

    swarm = create_swarm(n_particles=n_particles, dimensions=n_dims,
                         options=options, bounds=bounds)

    # set first particle to have original time and no craft packet
    swarm.position[0][1] = 0
    swarm.position[0][0] = original_time

    pos_history.append(np.vstack((swarm.position, [swarm.position[0]])))

    pbar = tqdm(range(iterations), position=1)

    # file=open("tmp_pso.txt","a")
    for i in pbar:
        run_de = np.random.rand() < mutate_prob
        # pos_history.append(np.vstack((swarm.position[:,1:],np.expand_dims(swarm.best_pos[1:], axis=0))))

        # Part 1: Update personal best

        swarm.current_cost, swarm.current_aux = f(
            swarm.position, **args)  # Compute current cost

        if i == 0:
            swarm.pbest_cost, swarm.pbest_aux = swarm.current_cost, swarm.current_aux
            swarm.best_cost, swarm.best_aux = swarm.current_cost, swarm.current_aux
            swarm.best_pos = swarm.position
            swarm.pbest_iter = np.zeros((n_particles,))

        # binomially mutate
        if run_de:
            tmp_pos = swarm.position
            swarm.trial_pos = topology.mutate_swarm(
                swarm, mutation_factor, crossp, mutation_candidates, bounds)
            swarm.trial_cost, swarm.trial_aux = f(swarm.trial_pos, **args)

            swarm.position, swarm.current_cost, swarm.current_aux = topology.compute_mbest(
                swarm)
            # print("-"*50)

        swarm.pbest_pos, swarm.pbest_cost, swarm.pbest_aux, swarm.pbest_iter = topology.compute_pbest(
            swarm, i)  # Update and store

        # Part 2: Update global best
        # Note that gbest computation is dependent on your topology
        # if np.min(swarm.pbest_cost) < swarm.best_cost:
        # best index is global minimum, others are best in the neighbourhood
        swarm.best_pos, swarm.best_cost, swarm.best_aux, swarm.best_index = topology.compute_gbest_local(
            swarm, 2, 4)
        # best_iter=i
        # file.write(pprint.pformat(swarm.best_pos)+"\n")

        # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        # if mutate prob is 1 then it is pure de
        if not run_de:
            swarm.velocity = topology.compute_velocity(
                swarm, bounds=bounds, clamp=clamp, iter=i)
            if np.random.rand() < 0.5:
                strat = "random"
            else:
                strat = "nearest"
            swarm.position = topology.compute_position(
                swarm, bounds=bounds, bh=BoundaryHandler(strategy=strat))

        post_fix = "c: {:.4f}".format(swarm.best_cost[swarm.best_index])
        pbar.set_postfix_str(post_fix)

        pos_history.append(
            np.vstack((swarm.position, [swarm.best_pos[swarm.best_index]])))

    std = np.std(swarm.position, axis=0)

    # print(swarm.position)
    return swarm.best_cost[swarm.best_index], swarm.best_pos[swarm.best_index], swarm.best_aux[swarm.best_index], std, pos_history
