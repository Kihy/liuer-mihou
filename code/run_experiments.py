from parse_with_kitsune import parse_kitsune
import numpy as np
from kitsune import *
from surrogate_model import *
from pso_framework import *
from evaluations.ml import *
from evaluations.feature_squeeze import *
from evaluations.mag_net import *
from filter_packets import filter_attack
from frocc.pardfrocc import ParDFROCC
import json

# step 0: filter raw traffic to include only traffic from attacker to victim


def filter():
    attacker_ip = ["192.168.10.30", "192.168.10.30", "192.168.10.7"]
    victim_ip = "192.168.10.5"

    attack_file = ["../dataset/[Port_scan]traffic_GoogleHome.pcap",
                   "../dataset/[OS_Service_Detection]Google_Home_Mini.pcap", "../dataset/[HTTP_Flooding]Google_Home_Mini.pcap"]
    out_file = ["../dataset/malicious/port_scan_attack_only.pcap",
                "../dataset/malicious/os_attack_only.pcap", "../dataset/malicious/hf_attack_only.pcap"]
    for i in range(len(attack_file)):
        filter_attack(attack_file[i], victim_ip, attacker_ip[i], out_file[i])


# step 1: parse benign and malicious files with kitsune, filename excludes extension
def parse():
    benign_file = "../dataset/[Normal]Google_Home_Mini"
    benign_netstat = "../dataset/google_home_netstat.pkl"
    malicious_files = ["../dataset/malicious/port_scan_attack_only"]
    # "../dataset/malicious/os_attack_only", "../dataset/malicious/hf_attack_only"]

    # parse_kitsune(benign_file + ".pcap", benign_file + ".csv",
    #               add_label=False, parse_type="scapy", add_proto=True, add_time=True, save_netstat=benign_netstat)

    for i in malicious_files:
        parse_kitsune(i + ".pcap", i + ".csv",
                      add_label=False, parse_type="scapy", add_proto=True, add_time=True, netstat_path=benign_netstat)

# step 2: train kitsune with normal data


def train(benign_file, kitsune_path, surrogate_model_path):

    num_packets = 145
    train_params = {
        # the pcap, pcapng, or tsv file to process.
        "path": benign_file + ".csv",
        "packet_limit": np.Inf,  # the number of packets to process,

        # KitNET params:
        "n_features": 57,
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
    #

    train_params = {
        "path": benign_file + ".csv",
        "model_path": surrogate_model_path,
        "epochs": 1,
        "n_features": 57,
        "batch_size": 64
    }
    train_surrogate(train_params)

#
# step 3: eval benign traffic on kitsune and surrogate


def eval_ben(benign_file, kitsune_path, surrogate_model_path):
    benign_plot_base_path = "../anomaly_plot/benign_flow"

    benign_traffic_surrogate_plot = f"{benign_plot_base_path}/benign_surrogate.png"
    if not os.path.exists(benign_plot_base_path):
        os.makedirs(benign_plot_base_path)

    benign_traffic_plot = f"{benign_plot_base_path}/benign.png"

    benign_pos, kitsune_threshold = eval_kitsune(
        benign_file + ".csv", kitsune_path, threshold=None, out_image=benign_traffic_plot, load_prediction=False)
    print("kitsune threshold", kitsune_threshold)
    print("benign examples over threshold:", benign_pos)

    benign_pos_surrogate, surrogate_threshold = eval_surrogate(
        benign_file + ".csv", surrogate_model_path, threshold=None, out_image=benign_traffic_surrogate_plot, input_dim=57)
    print("surrogate threshold", surrogate_threshold)
    print("benign examples over surrogate threshold:", benign_pos_surrogate)

    # kitsune_threshold = 0.24792078361382988
    # surrogate_threshold = 0.14724498857268006
    return kitsune_threshold, surrogate_threshold
#
# # step 4: evaluate malicious traffic with kitsune and surrogate model


def eval_mal(malicious_file, kitsune_path, surrogate_model_path, kitsune_threshold, surrogate_threshold):
    attack_plot_base_path = "../anomaly_plot/attack_flow/"
    if not os.path.exists(attack_plot_base_path):
        os.makedirs(attack_plot_base_path)

    malicious_traffic_plot = f"{attack_plot_base_path}/{malicious_file.split('/')[-1]}.png"

    mal_pos, _ = eval_kitsune(malicious_file + ".csv", kitsune_path,
                              threshold=kitsune_threshold, out_image=malicious_traffic_plot, record_scores=True)
    print(f"{malicious_file} examples over threshold:", mal_pos)

    surrogate_attack_plot = f"{attack_plot_base_path}/{malicious_file.split('/')[-1]}_surrogate.png"
    mal_pos_surrogate, _ = eval_surrogate(
        malicious_file + ".csv", surrogate_model_path, threshold=surrogate_threshold, out_image=surrogate_attack_plot, record_scores=True, input_dim=57)
    print(f"{malicious_file} examples over surrogate threshold:",
          mal_pos_surrogate)


# step 5: run aversarial attack
def run_attack():
    decision_type = "autoencoder"
    attack_config = {
        # information configs
        "name": "google_nest_port_scan",
        "malicious_file": malicious_file + ".pcap",
        "init_file": benign_file + ".pcap",
        "decision_type": decision_type,
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
        #models and thresholds
        "eval_model_path": kitsune_path,
        "eval_threshold": kitsune_threshold,
        "model_path": surrogate_model_path,
        # "model_path": kitsune_path,
        "threshold": surrogate_threshold,
        "netstat_path": benign_netstat,
        "base_offset": -13220634  # how many seconds the attacker should be earlier
        }

    # with open("metadata/google_nest_port_scan_autoencoder.json", "w") as meta:
    #     json.dump(attack_config, meta, indent=4)
    #
    # with open("metadata/google_nest_port_scan_autoencoder.json", "r") as meta:
    #     attack_config = json.load(meta)

    iterative_gen(1, attack_config)

# step 6: replay the file and parse


def parse_rep():
    benign_netstat = "../dataset/google_home_netstat.pkl"
    replay_file = ["../dataset/replay/port_scan_autoencoder_replay",
                   "../dataset/replay/os_autoencoder_replay", "../dataset/replay/flooding_autoencoder_replay",
                   "../dataset/adversarial/flooding_autoencoder_craft_iter_1",
                   "../dataset/adversarial/os_autoencoder_craft_iter_1",
                   "../dataset/adversarial/port_scan_autoencoder_craft_iter_1"]
    for i in replay_file:
        parse_kitsune(i + ".pcap", i + ".csv",
                      add_label=False, parse_type="scapy", add_proto=True, add_time=True, netstat_path=benign_netstat)

# step 7: check file bypasses other ML models with npy files


def other_ml():
    benign_file = "../dataset/benign/[Normal]Google_Home_Mini"
    benign_netstat = "../dataset/google_home_netstat.pkl"
    malicious_files = ["../dataset/malicious/port_scan_attack_only",
                       "../dataset/malicious/os_attack_only", "../dataset/malicious/hf_attack_only"]
    adv_files = ["../dataset/adversarial/port_scan_autoencoder_craft_iter_1",
                 "../dataset/adversarial/os_autoencoder_craft_iter_1", "../dataset/adversarial/flooding_autoencoder_craft_iter_1"]
    replay_files = ["../dataset/replay/port_scan_autoencoder_replay",
                    "../dataset/replay/os_autoencoder_replay", "../dataset/replay/flooding_autoencoder_replay"]

    for i in range(len(malicious_files)):
        benign, malicious, adversarial, replay = load_dataset(
            benign_file+"_flow.csv", malicious_files[i]+"_flow.csv", adv_files[i]+"_flow.csv", replay_files[i]+"_flow.csv")
        clfs = {
            "frocc": ParDFROCC(),
            "if": IsolationForest(random_state=0, contamination=0.001),
            "ocsvm": OneClassSVM(nu=0.001),
            "lof": LocalOutlierFactor(contamination=0.001, novelty=True),
            "ee": EllipticEnvelope(contamination=0.001, support_fraction=1)
        }
        #som and rrcf are done in eval_ml_models
        eval_ml_models(benign, malicious, adversarial, replay, clfs,
                       f"evaluations/fig/{malicious_files[i].split('/')[-1][:-4]}_flow", malicious_files[i]+"_flow.csv")

# step 8: check with adversarial defences

# feature squeezing


def adv_defences():
    for precision in range(1, 5):
        kitsune_threshold, pos_kit, d_rmse_threshold, pos_d_rmse, d_rel_rmse_threshold, pos_d_rel_rmse = eval_feature_squeeze(benign_file, kitsune_path,
                                                                                                                              "evaluations/fig/google_home_benign_fs", precision)
        threshold = [kitsune_threshold, d_rmse_threshold, d_rel_rmse_threshold]
        mal_pos_kit, mal_pos_d_rmse, mal_pos_d_rel_rmse = eval_feature_squeeze(malicious_file, kitsune_path,
                                                                               "evaluations/fig/google_home_port_scan_fs_malicious", precision, threshold=threshold)

        adv_pos_kit, adv_pos_d_rmse, adv_pos_d_rel_rmse = eval_feature_squeeze(adversarial_file, kitsune_path,
                                                                               "evaluations/fig/google_home_port_scan_fs_adv", precision, threshold=threshold)
        rep_pos_kit, rep_pos_d_rmse, rep_pos_d_rel_rmse = eval_feature_squeeze(replay_file, kitsune_path,
                                                                               "evaluations/fig/google_home_port_scan_fs_replay", precision, threshold=threshold)
        print(",".join(map(str, [precision, pos_d_rmse, pos_d_rel_rmse,
                                 mal_pos_d_rmse, mal_pos_d_rel_rmse,
                                 adv_pos_d_rmse, adv_pos_d_rel_rmse, rep_pos_d_rmse, rep_pos_d_rel_rmse])))

    # mag-net
    results = []
    # train_mag_net(benign_file+".csv")
    ret = test_mag_net(kitsune_path, benign_file, malicious_file)
    results.append(ret)
    ret = test_mag_net(kitsune_path, benign_file,
                       adversarial_file)
    results.append(ret)
    ret = test_mag_net(kitsune_path, benign_file, replay_file)
    results.append(ret)
    print(results)


if __name__ == '__main__':
    kitsune_path = "../models/kitsune_flow.pkl"
    surrogate_model_path = "../models/surrogate_ae"
    benign_file = "../dataset/benign/[Normal]Google_Home_Mini_flow"
    # train(benign_file, kitsune_path, surrogate_model_path)
    # kitsune_threshold, surrogate_threshold = eval_ben(
    #     benign_file, kitsune_path, surrogate_model_path)
    # print("-"*50)
    #
    # mal_files = ["../dataset/malicious/hf_attack_only", "../dataset/malicious/os_attack_only",
    #              "../dataset/malicious/port_scan_attack_only",
    #              "../dataset/adversarial/flooding_autoencoder_craft_iter_1", "../dataset/adversarial/os_autoencoder_craft_iter_1", "../dataset/adversarial/port_scan_autoencoder_craft_iter_1",
    #              "../dataset/replay/flooding_autoencoder_replay", "../dataset/replay/os_autoencoder_replay", "../dataset/replay/port_scan_autoencoder_replay"]
    # for mal_file in mal_files:
    #     print(mal_file)
    #     eval_mal(mal_file+"_flow", kitsune_path, surrogate_model_path,
    #              kitsune_threshold, surrogate_threshold)
    #     print("-"*50)
    other_ml()
