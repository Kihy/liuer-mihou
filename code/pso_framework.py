from itertools import product
import os
from surrogate_model import eval_surrogate
from datetime import datetime
import pprint
from kitsune import *
from pso import *
from parse_with_kitsune import *
logging.getLogger('pyswarms').setLevel(logging.WARNING)


def run_one(configs):
    """
    runs a single LiuerMihou attack.

    Args:
        configs (dict): generation parameters.

    Returns:
        report: summary statistics of the attack.

    """

    log_file = open(configs["log_file"], "w")

    log_file.write(pprint.pformat(configs)+"\n")

    report_file=open(configs["report_file"],"a")
    report_file.write("decision \t vectorization \t search algorithm \t iter \t pkt_seen \t n_mal \t n_craft \t mal+craft \t reduction \t pos_mal \t pos_craft \t pos_ignore\n")

    netstat_path = None

    print(pprint.pformat(configs))
    #
    report = craft_adversary(configs["malicious_file"], configs["init_file"], configs["adv_pcap_file"],
                             configs["mal_pcap_out"], configs["decision_type"], configs["threshold"], meta_path=configs["meta_path"],
                             model_path=configs["model_path"], optimizer=configs["optimizer"], init_count=configs["init_file_len"],
                             mutate_prob=configs["mutate_prob"], netstat_path=netstat_path, base_offset=configs["base_offset"],
                             log_file=log_file, n_dims=configs["n_dims"], max_time_window=configs[
        "max_time_window"], max_adv_pkt=configs["max_adv_pkt"], use_seed=configs["use_seed"],
        max_craft_pkt=configs["max_craft_pkt"], max_pkt_size=configs["max_pkt_size"], adv_csv_file=configs["adv_csv_file"],
        animation_folder=configs["animation_folder"], iteration=configs["iter"])

    print("max_time_window", configs["max_time_window"])
    print("max_craft_pkt", configs["max_craft_pkt"])
    print("n_dims", configs["n_dims"])

    # evaluate on real
    pos_mal, pos_craft,pos_ignore = eval(configs["adv_csv_file"], configs["eval_model_path"], threshold=configs["eval_threshold"], meta_file=configs["meta_path"],
                              ignore_index=configs["init_file_len"], out_image=configs["kitsune_graph_path"])

    report["pos_mal"] = pos_mal
    report["pos_craft"] = pos_craft
    report["pos_ignore"]= pos_ignore

    if configs["decision_type"] == "autoencoder":
        eval_surrogate(configs["adv_csv_file"], configs["model_path"], threshold=configs["threshold"],meta_file=configs["meta_path"],
                       ignore_index=configs["init_file_len"], out_image=configs["autoencoder_graph_path"])

    if report["num_altered"] == 0:
        log_file.write(pprint.pformat(report))
    else:
        fmt_string = "{} \t {} \t {} \t {} \t{} \t {} \t {} \t{} \t {}\n"
        if configs["mutate_prob"]==0.5:
            alg="PSO+DE"
        elif configs["mutate_prob"]==1:
            alg="DE"
        elif configs["mutate_prob"]==-1:
            alg="PSO"

        if configs["use_seed"]==True:
            vec=configs["n_dims"]+0.5
        else:
            vec=configs["n_dims"]
        report_file.write(fmt_string.format(configs["decision_type"], vec, alg, configs["iter"]+1, report["num_seen"],
                                         report["num_altered"], report["total_craft"], report["num_altered"] + report["total_craft"], report["average_reduction_ratio"], pos_mal, pos_craft,pos_ignore))
        log_file.write(pprint.pformat(report))
    log_file.close()
    pprint.pprint(report)

    return report


def iterative_gen(max_iter, optimizer, decision_type, n_dims, attack_configs, min_iter=0):
    """
    runs a batch of LiuerMihou attacks, parameter values should be self explainatory

    Args:
        max_iter (int): maximum iteration of an attack (when there are still packets with high anomaly score).
        optimizer (tuple): which search algorithm to use.
        decision_type (string): whether to use surrogate or kitsune as detection model.
        n_dims (tuple): specifies the vectorization method.
        attack_configs (dict): parameters for attack.
        min_iter (int): minimum iterations to start, mainly used to continous previous unfinished experiments. Defaults to 0.

    Returns:
        None

    """
    configs = {}

    configs["max_time_window"] = 1
    configs["max_craft_pkt"] = 5
    configs["decision_type"] = decision_type
    # configs["threshold"]=0.5445

    configs["n_dims"] = n_dims[0]
    configs["use_seed"]=n_dims[1]
    configs["optimizer"] = optimizer[0]
    configs["mutate_prob"] = optimizer[1]

    configs["init_file"] = "../experiment/traffic_shaping/init_pcap/google_home_normal.pcap"

    if configs["decision_type"] == "kitsune":
        configs["model_path"] = "../models/kitsune.pkl"
        configs["threshold"] = 0.36155338084978234
    elif configs["decision_type"] == "autoencoder":
        configs["model_path"] = "../models/surrogate_ae.h5"
        configs["threshold"] = 0.10926579684019089

    configs["eval_model_path"] = "../models/kitsune.pkl"
    configs["eval_threshold"] = 0.36155338084978234
    # configs["init_file_len"]=81838
    configs["init_file_len"] = 14400
    configs["max_pkt_size"] = 1514

    # folder structure: experiment/traffic_shaping/{dataset}/["craft", "adv", "csv", "png", "anim", "meta","logs"]/{dt_t_c_d_o_m}
    base_folder = "../experiment/traffic_shaping/{}".format(
        attack_configs["name"])
    experiment_folder = "{}_{}_{}_{}_{}_{}{}".format(
        configs["decision_type"], configs["max_time_window"], configs["max_craft_pkt"], configs["n_dims"], configs["use_seed"],configs["optimizer"], configs["mutate_prob"])

    for i in ["craft", "adv", "csv", "png", "anim", "meta", "logs"]:
        if not os.path.exists(os.path.join(base_folder, i, experiment_folder)):
            os.makedirs(os.path.join(base_folder, i, experiment_folder))

    for i in range(min_iter, max_iter):
        print("iteration:", i)
        # mal_pcap file will be the next malicious_file
        configs["mal_pcap_out"] = base_folder + \
            "/craft/{}/craft_iter_{}.pcap".format(experiment_folder, i + 1)
        configs["adv_pcap_file"] = base_folder + \
            "/adv/{}/iter_{}.pcap".format(experiment_folder, i)
        configs["adv_csv_file"] = base_folder + \
            "/csv/{}/iter_{}.csv".format(experiment_folder, i)
        configs["animation_folder"] = base_folder + \
            "/anim/{}/iter_{}".format(experiment_folder, i)
        configs["meta_path"] = base_folder + \
            "/meta/{}/iter_{}.csv".format(experiment_folder, i)
        configs["log_file"] = base_folder + \
            "/logs/{}/iter_{}.txt".format(experiment_folder, i)
        configs["report_file"]=base_folder+"/logs/report.csv"
        configs["iter"] = i
        configs["kitsune_graph_path"] = base_folder + \
            "/png/{}/iter{}_kitsune_rmse.png".format(experiment_folder, i)
        configs["autoencoder_graph_path"] = base_folder + \
            "/png/{}/iter{}_ae_rmse.png".format(experiment_folder, i)

        # first iteration uses original malicious file, and limit packets to first 10
        if i == 0:
            # configs["malicious_file"]="../kitsune_dataset/wiretap_malicious_hostonly.pcapng"
            configs["malicious_file"] = attack_configs["original_mal_file"]
            configs["max_adv_pkt"] = 1000

            # base offset is the time between last normal packet and first malicious packet
            # configs["base_offset"] =-596.31862402
            configs["base_offset"] = attack_configs["base_time_offset"]
        else:
            configs["malicious_file"] = base_folder + \
                "/craft/{}/craft_iter_{}.pcap".format(experiment_folder, i)
            configs["max_adv_pkt"] = 1000
            configs["base_offset"] = 0

        report = run_one(configs)

        #
        if report["num_altered"] == 0 or report["craft_failed"] + report["num_failed"] == 0:
            break


if __name__ == '__main__':
    optimizers = [("pso", -1), ("pso", 0.5), ("pso", 1)]
    n_dims = [(2, True), (2, False),  (3, False)]
    decision_types = ["autoencoder", "kitsune"]
    # iterative_gen(10, ("pso", -1), "kitsune", 2 )

    os_detection = {"name": "os_detection",
                "original_mal_file": "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only.pcap",
                "base_time_offset": -9422476.25}

    test = {"name": "test",
            "original_mal_file": "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only.pcap",
                    "base_time_offset": -9422476.25}

    arp = {"name": "arp",
           "original_mal_file": "../ku_dataset/arp_attack_only.pcap",
           "base_time_offset": -9421981.61}

    flooding = {"name": "flooding",
                "original_mal_file": "../ku_dataset/flooding_attacker_only.pcap",
                "base_time_offset": -497696}

    port_scan = {"name":"port_scan",
        "original_mal_file":"../ku_dataset/port_scan_attack_only.pcap",
        "base_time_offset":-12693212.38}

    datasets = [os_detection, flooding, port_scan]

    for dataset, o, n in product(datasets, optimizers, n_dims):
        iterative_gen(10, o, "autoencoder", n, dataset)
    for i in datasets:
        iterative_gen(10, optimizers[0], "kitsune", n_dims[1], i)
        iterative_gen(10, optimizers[1], "kitsune", n_dims[2], i)

    # for i in range(3):
    #     iterative_gen(10, optimizers[0], decision_types[0], n_dims[i], test)
    # run_one(9)
