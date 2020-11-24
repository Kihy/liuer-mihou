import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})


def eval_similarity(attack_file, kitsune_threshold=None, imposter_threshold=None):
    kitsune_score = np.genfromtxt(attack_file + "_kitsune_score.csv")
    imposter_score = np.genfromtxt(attack_file + "_imposter_score.csv")

    scaler_kitsune = MinMaxScaler()
    scaled_kitsune = scaler_kitsune.fit_transform(kitsune_score.reshape(-1, 1))
    scaled_kthresh = scaler_kitsune.transform([[np.log(kitsune_threshold)]])

    scaler_imposter = MinMaxScaler()
    scaled_imposter = scaler_imposter.fit_transform(
        imposter_score.reshape(-1, 1))
    scaled_ithresh = scaler_imposter.transform([[np.log(imposter_threshold)]])

    mse = mean_squared_error(scaled_kitsune, scaled_imposter)
    r2 = r2_score(scaled_kitsune, scaled_imposter)

    out_image = attack_file + "_anomaly_comparison.png"
    plt.figure(figsize=(4, 2))
    counter = scaled_kitsune.shape[0]
    plt.scatter(range(counter), scaled_kitsune, s=0.1,
                c="#2583DA", label="Kitsune RMSE")
    plt.scatter(range(counter), scaled_imposter, s=0.1,
                c="#DA7C25", label="Surrogate RMSE")
    plt.axhline(y=scaled_kthresh, color='#2583DA',
                linestyle='dotted', label="Kitsune threshold")
    plt.axhline(y=scaled_ithresh, color='#DA7C25',
                linestyle='dotted', label="Surrogate threshold")
    # plt.legend(loc="lower right", markerscale=10)
    # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
    print("MSE:{:.6g}   R2:{:.6g}".format(mse, r2))
    plt.tight_layout()
    plt.savefig(out_image)
    print("plot path:", out_image)


if __name__ == '__main__':
    os_detect = "[OS & service detection]traffic_GoogleHome_av_only"
    flooding = "flooding_attacker_only"
    port_scan = "port_scan_attack_only"
    paths = [os_detect, flooding, port_scan]
    for path in paths:
        eval_similarity("../ku_dataset/{}".format(path),
                        kitsune_threshold=0.36155338084978234, imposter_threshold=0.10926579684019089)
