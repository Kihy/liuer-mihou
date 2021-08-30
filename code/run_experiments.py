from parse_with_kitsune import parse_kitsune
import numpy as np
from kitsune import *
from surrogate_model import *
from pso_framework import *
from evaluations.similarity import *
from evaluations.ml import *
from evaluations.feature_squeeze import *
from evaluations.mag_net import *
from filter_packets import filter_packet
import json

# step 0: filter raw traffic

attack_name = "flooding"

device_list = ["Smartphone_1", "Smart_Clock_1", "Google-Nest-Mini_1", "SmartTV",
               "Lenovo_Bulb_1", "Cam_1", "Raspberry_Pi_telnet_wlan"]
num_packets_list = [2865483, 6390782, 5244602, 2444691,
                    22047, 854685, 1357782]

kitsune_threshold_list = [0.26067563798903576, 0.2474393225504581, 0.12854936695623007,
                          0.3263537666616599, 0.1731615464104402, 0.2772845115634913, 0.35942901583575154]

surrogate_threshold_list = [0.025341725544783, 0.01901568913948737, 0.016867547292195827,
                            0.03210490908376369, 0.06966010147032728, 0.06755883246660233, 0.10791920870542526]


attack_list1 = ["UDP_Flooding", "Service_Detection",
                "SYN_Flooding", "Port_Scanning", "ARP_Spoofing", "ACK_Flooding", "Telnet-brute_Force", "HTTP_Flooding"]
attack_list2 = ["UDP_Flooding", "Service_Detection",
                "SYN_Flooding", "Port_Scanning", "ARP_Spoofing", "ACK_Flooding"]

# for index in range(len(device_list)):
#
#     device_name = device_list[index]
#
#     # Raspberry_Pi_telnet_wlan has extra attacks
#     if device_list[index] == "Raspberry_Pi_telnet_wlan":
#         attacks = attack_list1
#     else:
#         attacks = attack_list2
#
#     # step 1: parse benign and malicious files with kitsune, filename excludes extension
#     benign_file = "../experiment/traffic_shaping/init_pcap/uq_{}_benign".format(
#         device_name)
#     malicious_files = [
#         "../uq_dataset/Attack Samples/{}/{}_{}".format(device_name, i, device_name) for i in attacks]
#
#     files_to_parse = malicious_files
#     #
#     # for i in files_to_parse:
#     #     parse_kitsune(i + ".pcap", i + ".csv",
#     #                   add_label=False, parse_type="scapy", add_proto=True, add_time=True)
#
#     # step 2: train kitsune with normal data
#     kitsune_path = "../models/uq/{}_kitsune.pkl".format(device_name)
#     num_packets = num_packets_list[index]
#     train_params = {
#         # the pcap, pcapng, or tsv file to process.
#         "path": benign_file + ".csv",
#         "packet_limit": np.Inf,  # the number of packets to process,
#
#         # KitNET params:
#         # maximum size for any autoencoder in the ensemble layer
#         "maxAE": 10,
#         # the number of instances taken to learn the feature mapping (the ensemble's architecture)
#         "FMgrace": np.floor(0.2 * num_packets),
#         # the number of instances used to train the anomaly detector (ensemble itself)
#         # FMgrace+ADgrace<=num samples in normal traffic
#         "ADgrace": np.floor(0.8 * num_packets),
#         # directory of kitsune
#         "model_path": kitsune_path,
#         # if normalize==true then kitsune does normalization automatically
#         "normalize": True
#     }
#     # train_normal(train_params)
#
#     # step 3: eval benign traffic on kitsune and surrogate
#     benign_plot_base_path = "../uq_dataset/anomaly_plot/Benign Samples"
#     if not os.path.exists(benign_plot_base_path):
#         os.makedirs(benign_plot_base_path)
#
#     benign_traffic_plot = "{}/{}_benign.png".format(
#         benign_plot_base_path, device_name)
#
#     benign_pos, kitsune_threshold = eval_kitsune(
#         benign_file + ".csv", kitsune_path, threshold=None, out_image=benign_traffic_plot, load_prediction=False)
#     print("kitsune threshold", kitsune_threshold)
#     print("benign examples over threshold:", benign_pos)
#     kitsune_threshold_list.append(kitsune_threshold)
#
#     surrogate_model_path = "../models/uq/{}_surrogate_ae.h5".format(
#         device_name)
#     benign_traffic_surrogate_plot = "{}/{}_benign_surrogate.png".format(
#         benign_plot_base_path, device_name)
#
#     train_params = {
#         "path": benign_file + ".csv",
#         "model_path": surrogate_model_path,
#         "epochs": 1,
#         "batch_size": 64
#     }
#     # train_surrogate(train_params)
#
#     benign_pos_surrogate, surrogate_threshold = eval_surrogate(
#         benign_file + ".csv", surrogate_model_path, threshold=None, out_image=benign_traffic_surrogate_plot)
#     surrogate_threshold_list.append(surrogate_threshold)
#     print("surrogate threshold", surrogate_threshold)
#     print("benign examples over surrogate threshold:", benign_pos_surrogate)
#
#     # step 4: evaluate malicious traffic with kitsune and surrogate model
#     attack_plot_base_path = "../uq_dataset/anomaly_plot/Attack Samples/{}".format(
#         device_name)
#     if not os.path.exists(attack_plot_base_path):
#         os.makedirs(attack_plot_base_path)
#
#     similarity_plot_base_path = "../uq_dataset/anomaly_plot/similarity/{}".format(
#         device_name)
#     if not os.path.exists(similarity_plot_base_path):
#         os.makedirs(similarity_plot_base_path)
#
#     for attack_name in attacks:
#         malicious_file = "../uq_dataset/Attack Samples/{}/{}_{}.csv".format(
#             device_name, attack_name, device_name)
#         malicious_traffic_plot = "{}/{}.png".format(
#             attack_plot_base_path, attack_name)
#         kitsune_threshold = kitsune_threshold_list[index]
#         mal_pos, _ = eval_kitsune(malicious_file, kitsune_path,
#                                   threshold=kitsune_threshold, out_image=malicious_traffic_plot, record_scores=True)
#         print("{} examples over threshold:".format(attack_name), mal_pos)
#         surrogate_threshold = surrogate_threshold_list[index]
#         surrogate_attack_plot = "{}/{}_surrogate.png".format(
#             attack_plot_base_path, attack_name)
#         mal_pos_surrogate, _ = eval_surrogate(
#             malicious_file, surrogate_model_path, threshold=surrogate_threshold, out_image=surrogate_attack_plot, record_scores=True)
#         print("{} examples over surrogate threshold:".format(attack_name),
#               mal_pos_surrogate)
#
#         # eval similarity does not need file extension
#         eval_similarity(malicious_file[:-4], kitsune_threshold=kitsune_threshold, imposter_threshold=surrogate_threshold,
#                         out_image="{}/{}".format(similarity_plot_base_path, attack_name))

# print("kitsune", kitsune_threshold_list)
# print("surrogate", surrogate_threshold_list)

# step 5: run aversarial attack
decision_type = "autoencoder"
# attack_config = {
#     # information configs
#     "name": "{}_{}".format(device_name, attack_name),
#     "malicious_file": malicious_file + ".pcap",
#     "init_file": benign_file + ".pcap",
#     "decision_type": decision_type,
#     "init_file_len": num_packets,
#     # vectorization parameter
#     "n_dims": 3,
#     "use_seed": False,
#     # pso parameters
#     "optimizer": "pso",
#     "mutate_prob": 0.5,
#     # boundary of search space
#     "max_time_window": 1,
#     "max_craft_pkt": 5,
#     #models and thresholds
#     "eval_model_path": kitsune_path,
#     "eval_threshold": kitsune_threshold,
#     "model_path": surrogate_model_path,
#     # "model_path": kitsune_path,
#     "threshold": surrogate_threshold,
#     "base_offset": 518}
#
# with open("metadata/{}_{}_{}.json".format(device_name, attack_name, decision_type), "w") as meta:
#     json.dump(attack_config, meta, indent=4)
# #
# with open("metadata/{}_{}_{}.json".format(device_name, attack_name, decision_type), "r") as meta:
#     attack_config = json.load(meta)
#
# iterative_gen(10, attack_config)

# step 6: replay the file

# step 7: check file bypasses other ML models with npy files
# adv_file_path = "../experiment/traffic_shaping/{}_{}/adv/autoencoder_1_5_3_False_pso0.5/iter_0".format(
#     device_name, attack_name)
# parse_kitsune(adv_file_path + ".pcap", adv_file_path + ".csv",
#               add_label=False, parse_type="scapy", add_proto=True, add_time=True)
# eval_surrogate(adv_file_path + ".csv", surrogate_model_path, threshold=surrogate_threshold,
#                out_image="../uq_dataset/anomaly_plot/surrogate_craft.png", record_scores=True)
# eval_kitsune(adv_file_path + ".csv", kitsune_path, threshold=kitsune_threshold,
#              out_image="../uq_dataset/anomaly_plot/kitsune_craft.png", record_scores=True)

benign_file = "../ku_dataset/google_home_normal"
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
benign_path = benign_file + ".npy"

# training magnet
# train_mag_net(benign_file+".csv")

# model_path = "../models/ku/google_home_mini.pkl"
# results=[]
# a=test_mag_net(model_path, benign_file+".csv", benign_file + ".csv")
# results.append(a)
# test_path = adv_file_path + ".npy"
for i in range(len(malicious_files)):
    malicious_path = malicious_files[i] + ".npy"
    adversarial_path = adversarial_files[i] + ".npy"
    replay_path = replay_files[i] + ".npy"
    #
    benign, malicious, adversarial, replay = load_dataset(
        benign_path, malicious_path, adversarial_path, replay_path)
    clfs = {
        # "if": IsolationForest(random_state=0, contamination=0.001),
        "ocsvm": OneClassSVM(nu=0.001),
        "lof": LocalOutlierFactor(contamination=0.001, novelty=True),
        # "ee": EllipticEnvelope(contamination=0.001, support_fraction=1)
    }
    eval_ml_models(benign, malicious, adversarial, replay, clfs,
                   "evaluations/fig/{}".format(malicious_files[i].split("/")[-1]), malicious_files[i])
    #
    other_ml = ["kitsune", "lof", "rrcf", "som", "ocsvm"]
    eval_similarity_other_ml(malicious_files[i], other_ml)

    # step 8: check with adversarial defences

    # device_name = "google_home_mini"

    # feature squeezing
    # for precision in range(1, 5):
    #     kitsune_threshold, pos_kit, d_rmse_threshold, pos_d_rmse, d_rel_rmse_threshold, pos_d_rel_rmse = eval_feature_squeeze(benign_file + ".csv", model_path,
    #                                                                                                                           "evaluations/fig/{}_benign_fs".format(device_name), precision)
    #     threshold = [kitsune_threshold, d_rmse_threshold, d_rel_rmse_threshold]
    #     mal_pos_kit, mal_pos_d_rmse, mal_pos_d_rel_rmse = eval_feature_squeeze(malicious_files[i] + ".csv", model_path,
    #                                                                "evaluations/fig/{}_{}_fs_malicious".format(device_name, attack_name), precision, threshold=threshold)
    #
    #     adv_pos_kit, adv_pos_d_rmse, adv_pos_d_rel_rmse = eval_feature_squeeze(adversarial_files[i] + ".csv", model_path,
    #                                                                            "evaluations/fig/{}_{}_fs_adv".format(device_name, attack_name), precision, threshold=threshold)
    #     rep_pos_kit, rep_pos_d_rmse, rep_pos_d_rel_rmse = eval_feature_squeeze(replay_files[i] + ".csv", model_path,
    #                                                                            "evaluations/fig/{}_{}_fs_replay".format(device_name, attack_name), precision, threshold=threshold)
    #     print(",".join(map(str, [precision, pos_d_rmse, pos_d_rel_rmse,
    #                              mal_pos_d_rmse, mal_pos_d_rel_rmse,
    #                              adv_pos_d_rmse, adv_pos_d_rel_rmse, rep_pos_d_rmse, rep_pos_d_rel_rmse])))

    # mag-net
    # print(malicious_files[i])

    # ret=test_mag_net(model_path, benign_file+".csv", malicious_files[i] + ".csv")
    # results.append(ret)
    # ret=test_mag_net(model_path, benign_file+".csv", adversarial_files[i] + ".csv")
    # results.append(ret)
    # ret=test_mag_net(model_path, benign_file+".csv", replay_files[i] + ".csv")
    # results.append(ret)
# print(results)
