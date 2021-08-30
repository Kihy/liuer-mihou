import numpy as np
import matplotlib.pyplot as plt


def plot_results(file_list):
    num_files = len(file_list)
    num_attacks = len(file_list[0])
    plt.figure(figsize=(num_files * 8, num_attacks * 6))
    fig, ax = plt.subplots(num_attacks, num_files, sharey='row')
    threshold = np.genfromtxt(file_list[0][0] + "_kitsune_threshold.csv")
    for i in range(num_files):
        for j in range(num_attacks):
            malicious_path = file_list[i][j] + "_kitsune_score.csv"
            mal_score = np.genfromtxt(malicious_path)
            ax[i, j].scatter(range(len(mal_score)), mal_score, s=0.1, alpha=0.5,c="#2583DA")
            ax[i, j].axhline(y=threshold, color="red")
            if j==0:
                ax[i,j].set_title("Original")

            if j==1:
                ax[i,j].set_title("Adversarial")

            if j==2:
                ax[i,j].set_title("Replayed")

            ax[i, j].set_yscale('log')
            ax[i, j].ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True, useOffset=False)
    fig.supxlabel('Packet Index')
    fig.supylabel('Anomaly Score')
    plt.tight_layout()
    plt.savefig("evaluations/fig/anomaly_scores")


if __name__ == '__main__':
    files = [["../ku_dataset/port_scan_attack_only", "../experiment/craft_files/port_scan_autoencoder_craft_iter_1", "../experiment/replay/ps_a_1"],
             ["../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only",
                 "../experiment/craft_files/os_autoencoder_craft_iter_1", "../experiment/replay/os_a_1"],
             ["../ku_dataset/flooding_attacker_only", "../experiment/craft_files/flooding_autoencoder_craft_iter_1",        "../experiment/replay/flooding_a_1"]]

    plot_results(files)
