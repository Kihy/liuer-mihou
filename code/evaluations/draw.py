import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler

# the location of the font file
font_path = '/usr/share/fonts/truetype/cmu/cmunrm.ttf'
my_font = fm.FontProperties(fname=font_path)


def plot_comparison(file_list, ml_list):
    num_files = len(file_list)
    num_models = len(ml_list)
    fig, ax = plt.subplots(num_files, num_models,
                           sharey='row', figsize=(12, 8))

    for i, filename in enumerate(file_list):
        attack_file = f"../dataset/{filename}"
        imposter_score = np.genfromtxt(attack_file + "_imposter_score.csv")
        threshold = np.genfromtxt(attack_file + "_imposter_threshold.csv")
        # imposter_score=np.log(imposter_score)
        scaler_imposter = MinMaxScaler()
        imposter_score = scaler_imposter.fit_transform(
            imposter_score.reshape(-1, 1))
        threshold = scaler_imposter.transform([[threshold]])

        for j, ml_name in enumerate(ml_list):
            ml_score = np.genfromtxt(
                attack_file + f"_{ml_name.lower()}_score.csv")
            ml_threshold = np.genfromtxt(
                attack_file + f"_{ml_name.lower()}_threshold.csv")
            # print(ml_threshold)
            # ml_score=np.log(ml_score)
            scaler_ml = MinMaxScaler()
            ml_score = scaler_ml.fit_transform(
                ml_score.reshape(-1, 1))
            ml_threshold = scaler_ml.transform([[ml_threshold]])
            print(ml_threshold, ml_name)

            ax[i, j].scatter(range(len(imposter_score)), imposter_score,
                             s=0.1, alpha=0.5, c="#DA7C25", label="Surrogate")
            ax[i, j].axhline(y=threshold, color="#DA7C25")

            ax[i, j].scatter(range(len(ml_score)), ml_score,
                             s=0.1, alpha=0.5, c="#2583DA", label=f"{ml_name}")
            ax[i, j].legend(loc="lower right", markerscale=10)
            ax[i, j].axhline(y=ml_threshold, color="#2583DA")
            ax[i, j].set_yscale('log')
            if i == 0:
                ax[i, j].set_title(
                    ml_list[j], fontproperties=my_font, fontsize=18)
            if j == 0:
                if i == 0:
                    ax[i, j].set_ylabel(
                        "PS", fontproperties=my_font, fontsize=18)
                if i == 1:
                    ax[i, j].set_ylabel(
                        "OD", fontproperties=my_font, fontsize=18)
                if i == 2:
                    ax[i, j].set_ylabel(
                        "HF", fontproperties=my_font, fontsize=18)

            ax[i, j].ticklabel_format(axis='x', style='sci', scilimits=(
                0, 0), useMathText=True, useOffset=True)
            for tick in ax[i, j].get_xticklabels():
                tick.set_fontproperties(my_font)
                tick.set_fontsize(16)
            for tick in ax[i, j].get_yticklabels():
                tick.set_fontproperties(my_font)
                tick.set_fontsize(16)
            ax[i, j].xaxis.get_offset_text().set_fontproperties(my_font)
            ax[i, j].xaxis.get_offset_text().set_fontsize(14)

    plt.tight_layout()
    plt.savefig("evaluations/fig/as_compare")


def plot_results(file_list):
    num_files = len(file_list)
    num_attacks = len(file_list[0])
    fig, ax = plt.subplots(num_attacks, num_files,
                           sharey='row', figsize=(8, 8))
    threshold = np.genfromtxt(file_list[0][0] + "_kitsune_threshold.csv")
    for i in range(num_files):
        for j in range(num_attacks):
            malicious_path = file_list[i][j] + "_kitsune_score.csv"
            mal_score = np.genfromtxt(malicious_path)
            ax[i, j].scatter(range(len(mal_score)), mal_score,
                             s=0.1, alpha=0.5, c="#2583DA")
            ax[i, j].axhline(y=threshold, color="red")
            if i == 0:
                if j == 0:
                    ax[i, j].set_title(
                        "Original", fontproperties=my_font, fontsize=18)

                if j == 1:
                    ax[i, j].set_title(
                        "Adversarial", fontproperties=my_font, fontsize=18)

                if j == 2:
                    ax[i, j].set_title(
                        "Replayed", fontproperties=my_font, fontsize=18)
            if j == 0:
                if i == 0:
                    ax[i, j].set_ylabel(
                        "PS", fontproperties=my_font, fontsize=18)
                if i == 1:
                    ax[i, j].set_ylabel(
                        "OD", fontproperties=my_font, fontsize=18)
                if i == 2:
                    ax[i, j].set_ylabel(
                        "HF", fontproperties=my_font, fontsize=18)

            ax[i, j].set_yscale('log')
            ax[i, j].ticklabel_format(axis='x', style='sci', scilimits=(
                0, 0), useMathText=True, useOffset=True)
            for tick in ax[i, j].get_xticklabels():
                tick.set_fontproperties(my_font)
                tick.set_fontsize(16)
            for tick in ax[i, j].get_yticklabels():
                tick.set_fontproperties(my_font)
                tick.set_fontsize(16)
            ax[i, j].xaxis.get_offset_text().set_fontproperties(my_font)
            ax[i, j].xaxis.get_offset_text().set_fontsize(14)
    # fig.supxlabel('Packet Index',fontproperties=my_font, fontsize=20)
    # fig.supylabel('Anomaly Score',fontproperties=my_font)
    plt.tight_layout()
    plt.savefig("evaluations/fig/anomaly_scores")
