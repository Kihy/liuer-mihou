import os
import json
import numpy as np
from kitsune import *
from surrogate_model import *
from pso_framework import *
from evaluations.ml import *
from evaluations.feature_squeeze import *
from evaluations.mag_net import *
from filter_packets import filter_attack
from frocc.pardfrocc import ParDFROCC
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


def filter(attack_file, victim_ip, attacker_ip, out_file):
    filter_attack(attack_file, victim_ip, attacker_ip, out_file)


def parse_files(benign_file, benign_netstat, malicious_file):
    parse_kitsune(benign_file + ".pcap", benign_file + ".csv",
                  add_label=False, parse_type="scapy", add_proto=True, add_time=True, save_netstat=benign_netstat)
    parse_kitsune(malicious_file + ".pcap", malicious_file + ".csv",
                  add_label=False, parse_type="scapy", add_proto=True, add_time=True, netstat_path=benign_netstat)


def train_models(benign_file, kitsune_path, surrogate_model_path, num_packets):
    train_params = {
        # the pcap, pcapng, or tsv file to process.
        "path": benign_file + ".csv",
        "packet_limit": np.Inf,  # the number of packets to process,
        # KitNET params:
        "n_features": 100,
        # maximum size for any autoencoder in the ensemble layer
        "maxAE": 10,
        # the number of instances taken to learn the feature mapping (the ensemble's architecture)
        "FMgrace": np.floor(0.2 * num_packets),
        # the number of instances used to train the anomaly detector (ensemble itself)
        # FMgrace+ADgrace<=num samples in normal traffic
        "ADgrace": np.floor(0.8 * num_packets),
        # directory of kitsune
        "model_path": kitsune_path,
        # if normalize==true then kitsune does normalization automatically
        "normalize": True
    }
    train_normal(train_params)
    surrogate_params = {
        "path": benign_file + ".csv",
        "path": benign_file + ".csv",
        "model_path": surrogate_model_path,
        "epochs": 1,
        "n_features": 100,
        "batch_size": 64
    }
    train_surrogate(surrogate_params)


def eval_benign(benign_file, kitsune_path, surrogate_model_path, benign_plot_base_path):
    if not os.path.exists(benign_plot_base_path):
        os.makedirs(benign_plot_base_path)
    benign_traffic_plot = f"{benign_plot_base_path}/benign.png"
    benign_traffic_surrogate_plot = f"{benign_plot_base_path}/benign_surrogate.png"
    benign_pos, kitsune_threshold = eval_kitsune(
        benign_file + ".csv", kitsune_path, threshold=None, out_image=benign_traffic_plot, load_prediction=False)
    print("kitsune threshold", kitsune_threshold)
    print("benign examples over threshold:", benign_pos)
    benign_pos_surrogate, surrogate_threshold = eval_surrogate(
        benign_file + ".csv", surrogate_model_path, threshold=None, out_image=benign_traffic_surrogate_plot, input_dim=100)
    print("surrogate threshold", surrogate_threshold)
    print("benign examples over surrogate threshold:", benign_pos_surrogate)
    # kitsune_threshold = 0.24792078361382988
    # surrogate_threshold = 0.14724498857268006
    return kitsune_threshold, surrogate_threshold


def eval_malicious(malicious_file, kitsune_path, surrogate_model_path, kitsune_threshold, surrogate_threshold, attack_plot_base_path):
    if not os.path.exists(attack_plot_base_path):
        os.makedirs(attack_plot_base_path)
    malicious_traffic_plot = f"{attack_plot_base_path}/{malicious_file.split('/')[-1]}.png"
    mal_pos, _ = eval_kitsune(malicious_file + ".csv", kitsune_path,
                              threshold=kitsune_threshold, out_image=malicious_traffic_plot, record_scores=True)
    print(f"{malicious_file} examples over threshold:", mal_pos)
    surrogate_attack_plot = f"{attack_plot_base_path}/{malicious_file.split('/')[-1]}_surrogate.png"
    mal_pos_surrogate, _ = eval_surrogate(
        malicious_file + ".csv", surrogate_model_path, threshold=surrogate_threshold, out_image=surrogate_attack_plot, record_scores=True, input_dim=100)
    print(f"{malicious_file} examples over surrogate threshold:", mal_pos_surrogate)
    return mal_pos, mal_pos_surrogate


def adversarial_attack(attack_config):
    iterative_gen(1, attack_config)


def parse_replay(replay_file, benign_netstat):
    parse_kitsune(replay_file + ".pcap", replay_file + ".csv",
                  add_label=False, parse_type="scapy", add_proto=True, add_time=True, netstat_path=benign_netstat)


def other_ml(benign_file, malicious_file, adversarial_file, replay_file):
    benign, malicious, adversarial, replay = load_dataset(
        benign_file+".csv", malicious_file+".csv", adversarial_file+".csv", replay_file+".csv")
    clfs = {
        "frocc": ParDFROCC(),
        "if": IsolationForest(random_state=0, contamination=0.001),
        "ocsvm": OneClassSVM(nu=0.001),
        "lof": LocalOutlierFactor(contamination=0.001, novelty=True),
        "ee": EllipticEnvelope(contamination=0.001, support_fraction=1)
    }
    eval_ml_models(benign, malicious, adversarial, replay, clfs,
                   f"evaluations/fig/{malicious_file.split('/')[-1]}", malicious_file+".csv")


def adv_defences(benign_file, kitsune_path, malicious_file, adversarial_file, replay_file, attack_type):
    for precision in range(1, 5):
        kitsune_threshold, pos_kit, d_rmse_threshold, pos_d_rmse, d_rel_rmse_threshold, pos_d_rel_rmse \
            = eval_feature_squeeze(benign_file, kitsune_path, "evaluations/fig/google_home_benign_fs", precision)
        threshold = [kitsune_threshold, d_rmse_threshold, d_rel_rmse_threshold]
        mal_pos_kit, mal_pos_d_rmse, mal_pos_d_rel_rmse = eval_feature_squeeze(
            malicious_file, kitsune_path, f"evaluations/fig/google_home_{attack_type}_fs_malicious", precision,
            threshold=threshold
        )
        adv_pos_kit, adv_pos_d_rmse, adv_pos_d_rel_rmse = eval_feature_squeeze(
            adversarial_file, kitsune_path, f"evaluations/fig/google_home_{attack_type}_fs_adv", precision,
            threshold=threshold
        )
        rep_pos_kit, rep_pos_d_rmse, rep_pos_d_rel_rmse = eval_feature_squeeze(
            replay_file, kitsune_path, f"evaluations/fig/google_home_{attack_type}_fs_replay", precision,
            threshold=threshold
        )
        print(",".join(map(str, [precision, pos_d_rmse, pos_d_rel_rmse,
                                 mal_pos_d_rmse, mal_pos_d_rel_rmse,
                                 adv_pos_d_rmse, adv_pos_d_rel_rmse, rep_pos_d_rmse, rep_pos_d_rel_rmse])))
    # mag-net
    results = []
    train_mag_net(benign_file)
    ret = test_mag_net(kitsune_path, benign_file, malicious_file)
    results.append(ret)
    ret = test_mag_net(kitsune_path, benign_file,
                       adversarial_file)
    results.append(ret)
    ret = test_mag_net(kitsune_path, benign_file, replay_file)
    results.append(ret)
    print(results)


def main():
    # Step 0: Filter attack traffic

    """ We have several types of attacks in the dataset, you can choose one to run the experiments.
        However, Steps 6–8 are only available for the port scan attack. """
    attack_type, attacker_ip = "Port_scan", "192.168.10.30"
    # attack_type, attacker_ip = "OS_Service_Detection", "192.168.10.30"
    # attack_type, attacker_ip = "HTTP_Flooding", "192.168.10.7"
    """ ------------------------------------------------------------------------------------------- """

    victim_ip = "192.168.10.5"
    attack_file = f"../dataset/[{attack_type}]Google_Home_Mini.pcap"
    out_file = f"../dataset/malicious/{attack_type.lower()}_attack_only.pcap"
    filter(attack_file, victim_ip, attacker_ip, out_file)

    # Step 1: Parse files
    benign_file = "../dataset/[Normal]Google_Home_Mini"
    benign_netstat = "../dataset/google_home_netstat.pkl"
    # malicious_file = "../dataset/malicious/port_scan_attack_only"
    malicious_file = '.'.join(out_file.split('.')[:-1])
    parse_files(benign_file, benign_netstat, malicious_file)

    # Step 2: Train models
    if not os.path.exists("../models"):
        os.makedirs("../models")
    kitsune_path = "../models/kitsune.pkl"
    surrogate_model_path = "../models/surrogate_ae"
    num_packets = 14400     # Number of packets in benign_file
    train_models(benign_file, kitsune_path, surrogate_model_path, num_packets)

    # Step 3: Evaluate benign
    benign_plot_base_path = "../anomaly_plot/benign"
    kitsune_threshold, surrogate_threshold = eval_benign(benign_file, kitsune_path, surrogate_model_path, benign_plot_base_path)

    # Step 4: Evaluate malicious
    attack_plot_base_path = "../anomaly_plot/attack/"
    eval_malicious(malicious_file, kitsune_path, surrogate_model_path, kitsune_threshold, surrogate_threshold, attack_plot_base_path)

    # Step 5: Adversarial attack
    attack_config = {
        # information configs
        "name": f"google_nest_{attack_type.lower()}",
        "malicious_file": malicious_file + ".pcap",
        "init_file": benign_file + ".pcap",
        "decision_type": "autoencoder",
        "init_file_len": num_packets,
        # vectorization parameter
        "n_dims": 3,
        "use_seed": False,
        # pso parameters
        "optimizer": "pso",
        "mutate_prob": 0.5,
        # boundary of search space
        "max_time_window": 1,
        "max_craft_pkt": 5,
        "max_pkt_size": 1514,
        # models and thresholds
        "eval_model_path": kitsune_path,
        "eval_threshold": kitsune_threshold,
        "model_path": surrogate_model_path,
        "threshold": surrogate_threshold,
        "netstat_path": benign_netstat,
        "base_offset": -13220634    # how many seconds the attacker should be earlier
    }
    adversarial_attack(attack_config)

    # Step 6: Parse replay
    replay_file = f"../dataset/replay/{attack_type.lower()}_autoencoder_replay"
    parse_replay(replay_file, benign_netstat)

    # Step 7: Check file bypasses other ML models
    adversarial_file = f"../experiment/traffic_shaping/google_nest_{attack_type.lower()}/csv/autoencoder_1_5_3_False_pso0.5/google_nest_{attack_type.lower()}_iter_0"
    other_ml(benign_file, malicious_file, adversarial_file, replay_file)

    # Step 8: Adversarial defences
    adv_defences(benign_file + ".csv", kitsune_path, malicious_file + ".csv", adversarial_file + ".csv",
                 replay_file + ".csv", attack_type.lower())


if __name__ == '__main__':
    main()
