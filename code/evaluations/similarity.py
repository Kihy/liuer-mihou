import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})


def eval_similarity_other_ml(attack_file, other_ml):
    imposter_score = np.genfromtxt(attack_file + "_imposter_score.csv")
    threshold = np.genfromtxt(attack_file + "_imposter_threshold.csv")

    scaler_imposter = MinMaxScaler()
    imposter_score = scaler_imposter.fit_transform(
        imposter_score.reshape(-1, 1))
    threshold = scaler_imposter.transform([[threshold]])

    accuracy = []
    for i in other_ml:
        score = np.genfromtxt(attack_file + "_{}_score.csv".format(i))
        ml_thresh = np.genfromtxt(attack_file + "_{}_threshold.csv".format(i))

        scaler = MinMaxScaler()
        score = scaler.fit_transform(score.reshape(-1, 1))
        ml_thresh = scaler.transform([[ml_thresh]])

        cs = 1-np.mean(paired_distances(score, imposter_score,metric="cosine"))
        ed = 1-np.mean(paired_distances(score, imposter_score,metric="euclidean"))
        # print(score, imposter_score)
        # print(paired_distances(score, imposter_score, metric="cosine"))

        out_image = attack_file + "_anomaly_comparison_{}.png".format(i)
        plt.figure(figsize=(20, 10))
        counter = imposter_score.shape[0]
        plt.scatter(range(counter), score, s=0.1,
                    c="#2583DA", label="{} AS".format(i))
        plt.scatter(range(counter), imposter_score, s=0.1,
                    c="#DA7C25", label="Surrogate RMSE")
        plt.axhline(y=ml_thresh, color='#2583DA',
                    linestyle='dotted', label="{} threshold".format(i))
        plt.axhline(y=threshold, color='#DA7C25',
                    linestyle='dotted', label="Surrogate threshold")
        plt.legend(loc="lower right", markerscale=10)
        # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
        print("attack file: ", attack_file, "ml model", i)
        print("cs:{:.6g}   ed:{:.6g}".format(cs, ed))
        print("normalised thresholds: ml thresh:{}, surrogate thresh:{}".format(ml_thresh, threshold))
        print(",".join(map(str, [cs, ed, ml_thresh[0][0], threshold[0][0]])))
        plt.tight_layout()
        plt.savefig(out_image)
        plt.close()
        print("plot path:", out_image)

def plot_anomaly_scores(file_list, threshold, out_image, labels):

    colours = cm.Paired(np.linspace(0, 1, len(file_list)))
    colours[:, -1] = 0.5
    plt.figure(figsize=(8, 4))
    for i in range(len(file_list)):
        anomaly_scores = np.genfromtxt(file_list[i])
        plt.scatter(range(
            anomaly_scores.shape[0]), anomaly_scores, s=0.1, color=colours[i], label=labels[i])
    plt.axhline(y=np.log(threshold), color="r", label="Threshold")

    plt.xlabel("Packet Index")
    plt.ylabel("Anomaly Score (log scaled)")
    plt.legend(loc="lower right", markerscale=10)
    plt.tight_layout()
    plt.savefig(out_image)
    print("plot path:", out_image)


if __name__ == '__main__':
    os_detect = "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only"
    flooding = "../ku_dataset/flooding_attacker_only"
    port_scan = "../ku_dataset/port_scan_attack_only"
    paths = [os_detect, flooding, port_scan]

    other_ml = ["kitsune", "lof","som"]
    eval_similarity_other_ml(flooding, other_ml)

    # for path in paths:
    #     eval_similarity("../ku_dataset/{}".format(path),
    #                     kitsune_threshold=0.36155338084978234, imposter_threshold=0.10926579684019089)

    # file_list = ["../ku_dataset/flooding_kit_kitsune_score.csv",
    #              "../experiment/craft_files/flooding_kitsune_craft_iter_1_kitsune_score.csv", "../experiment/replay/flooding_k_1_kitsune_score.csv"]
    # plot_anomaly_scores(file_list, 0.5445, "../experiment/replay/flooding_kitsune_comp.png",
    #                     ["Unmodified", "Adversarial-Kitsune", "Replay-Kitsune"])
    #
    # file_list = ["../ku_dataset/flooding_ae_kitsune_score.csv",
    #              "../experiment/craft_files/flooding_autoencoder_craft_iter_1_kitsune_score.csv", "../experiment/replay/flooding_a_1_kitsune_score.csv"]
    # plot_anomaly_scores(file_list, 0.5445, "../experiment/replay/flooding_surrogate_comp.png",
    #                     ["Unmodified", "Adversarial-Surrogate", "Replay-Surrogate"])
    #
    # file_list = ["../ku_dataset/port_scan_attack_only_kitsune_score.csv",
    #              "../experiment/craft_files/port_scan_kitsune_craft_iter_1_kitsune_score.csv", "../experiment/replay/ps_k_1_kitsune_score.csv"]
    # plot_anomaly_scores(file_list, 0.5445, "../experiment/replay/ps_kitsune_comp.png",
    #                     ["Unmodified", "Adversarial-Kitsune", "Replay-Kitsune"])
    #
    # file_list = ["../ku_dataset/port_scan_attack_only_kitsune_score.csv",
    #              "../experiment/craft_files/port_scan_autoencoder_craft_iter_1_kitsune_score.csv", "../experiment/replay/ps_a_1_kitsune_score.csv"]
    # plot_anomaly_scores(file_list, 0.5445, "../experiment/replay/ps_surrogate_comp.png",
    #                     ["Unmodified", "Adversarial-Surrogate", "Replay-Surrogate"])
    #
    # file_list = ["../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only_kitsune_score.csv",
    #              "../experiment/craft_files/os_kitsune_craft_iter_1_kitsune_score.csv", "../experiment/replay/os_k_1_kitsune_score.csv"]
    # plot_anomaly_scores(file_list, 0.5445, "../experiment/replay/os_kitsune_comp.png",
    #                     ["Unmodified", "Adversarial-Kitsune", "Replay-Kitsune"])
    #
    # file_list = ["../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only_kitsune_score.csv",
    #              "../experiment/craft_files/os_autoencoder_craft_iter_1_kitsune_score.csv", "../experiment/replay/os_a_1_kitsune_score.csv"]
    # plot_anomaly_scores(file_list, 0.5445, "../experiment/replay/os_surrogate_comp.png",
    #                     ["Unmodified", "Adversarial-Surrogate", "Replay-Surrogate"])
