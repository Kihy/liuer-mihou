import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})

def eval_similarity_attacks(dataset_info, plot_path, log=True, scale=True):
    n_devices=len(dataset_info.keys())
    max_attacks=0
    for name, info in dataset_info.items():
        max_attacks=max(max_attacks, len(info["attacks"]))

    fig,ax=plt.subplots(n_devices, max_attacks, figsize=(30,30))
    for i,device_name in enumerate(dataset_info.keys()):

        for j, attack_name in enumerate(dataset_info[device_name]["attacks"]):
            imp_threshold=dataset_info[device_name]["surrogate_threshold"]
            kit_threshold=dataset_info[device_name]["kitsune_threshold"]
            imp_score=np.genfromtxt(f"../datasets/malicious/uq/{device_name}/{device_name}_{attack_name}_imposter_score.csv").reshape(-1, 1)
            kit_score = np.genfromtxt(f"../datasets/malicious/uq/{device_name}/{device_name}_{attack_name}_kitsune_score.csv").reshape(-1, 1)

            if log:
                imp_threshold=np.log(imp_threshold)
                kit_threshold=np.log(kit_threshold)
                imp_score=np.log(imp_score)
                kit_score=np.log(kit_score)

            if scale:
                scaler_imposter = MinMaxScaler()
                imp_score = scaler_imposter.fit_transform(imp_score.reshape(-1, 1))
                imp_threshold = scaler_imposter.transform([[imp_threshold]])

                scaler = MinMaxScaler()
                kit_score = scaler.fit_transform(kit_score.reshape(-1, 1))
                kit_threshold = scaler.transform([[kit_threshold]])

            cs = 1-np.mean(paired_distances(imp_score, kit_score,metric="cosine"))
            ed = 1-np.mean(paired_distances(imp_score, kit_score,metric="euclidean"))

            counter = dataset_info[device_name]["attacks"][attack_name]["num_mal"]
            ax[i,j].scatter(range(counter), kit_score, s=0.1,
                        c="#2583DA", label="Kit AS")
            ax[i,j].scatter(range(counter), imp_score, s=0.1,
                        c="#DA7C25", label="Sur AS")
            ax[i,j].axhline(y=kit_threshold, color='#2583DA',
                        linestyle='dotted', label="Kit t")
            ax[i,j].axhline(y=imp_threshold, color='#DA7C25',
                        linestyle='dotted', label="Sur t")
            ax[i,j].legend(loc="lower right", markerscale=10)
            # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
            ax[i,j].set_xlabel("cs:{:.3g} ed:{:.3g}".format(cs, ed))
            if j==0:
                ax[i,j].set_ylabel(device_name)
            ax[i,j].set_title(attack_name)

    out_image = f"{plot_path}/kit_sur_all_{log}_{scale}.png"
    if not os.path.exists(os.path.dirname(out_image)):
        os.makedirs(os.path.dirname(out_image))

    fig.tight_layout()
    plt.savefig(out_image)
    plt.close()
    print("plot path:", out_image)



def eval_similarity_other_ml(attack_file, ml, thresholds, plot_path):
    """compares the similarity of anomaly score with first index of ml against every
    other index"""
    imposter_score = np.genfromtxt(attack_file + f"_{ml[0]}_score.csv")
    imposter_score=np.log(imposter_score)
    threshold=thresholds[0]
    scaler_imposter = MinMaxScaler()
    imposter_score = scaler_imposter.fit_transform(
        imposter_score.reshape(-1, 1))
    threshold = scaler_imposter.transform([[threshold]])

    out_image = f"{plot_path}/{'_'.join(ml)}.png"
    if not os.path.exists(os.path.dirname(out_image)):
        os.makedirs(os.path.dirname(out_image))

    rows=np.ceil(sqrt(len(ml-1)))
    cols=np.ceil(len(ml-1)/rows)
    fig,ax=plt.subplots(cols, rows, sharey='row')

    accuracy = []
    for i in range(1,len(ml)):
        score = np.genfromtxt(attack_file + f"_{ml[i]}_score.csv")
        score=np.log(score)
        ml_thresh = thresholds[i]

        scaler = MinMaxScaler()
        score = scaler.fit_transform(score.reshape(-1, 1))
        ml_thresh = scaler.transform([[ml_thresh]])

        cs = 1-np.mean(paired_distances(score, imposter_score,metric="cosine"))
        ed = 1-np.mean(paired_distances(score, imposter_score,metric="euclidean"))
        # print(score, imposter_score)
        # print(paired_distances(score, imposter_score, metric="cosine"))


        counter = imposter_score.shape[0]
        ax[i%rows, i//cols].scatter(range(counter), score, s=0.1,
                    c="#2583DA", label=f"{ml[i]} AS")
        ax[i%rows, i//cols].scatter(range(counter), imposter_score, s=0.1,
                    c="#DA7C25", label=f"{ml[0]} AS")
        ax[i%rows, i//cols].axhline(y=ml_thresh, color='#2583DA',
                    linestyle='dotted', label=f"{ml[i]} threshold")
        ax[i%rows, i//cols].axhline(y=threshold, color='#DA7C25',
                    linestyle='dotted', label=f"{ml[0]} threshold")
        ax[i%rows, i//cols].legend(loc="lower right", markerscale=10)
        # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
        ax[i%rows, i//cols].set_xlabel("cs:{:.6g} ed:{:.6g}".format(cs, ed))
        # print("attack file: ", attack_file, "ml model", i)
        # print("cs:{:.6g}   ed:{:.6g}".format(cs, ed))
        # print("normalised thresholds: ml thresh:{}, surrogate thresh:{}".format(ml_thresh, threshold))
        # print(",".join(map(str, [cs, ed, ml_thresh[0][0], threshold[0][0]])))
    fig.tight_layout()
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
