from itertools import product

from nfstream import NFStreamer
import pandas as pd

files = ["../dataset/benign/[Normal]Google_Home_Mini.pcap", "../dataset/malicious/hf_attack_only.pcap", "../dataset/malicious/os_attack_only.pcap",
         "../dataset/malicious/port_scan_attack_only.pcap",
         "../dataset/adversarial/flooding_autoencoder_craft_iter_1.pcap", "../dataset/adversarial/os_autoencoder_craft_iter_1.pcap", "../dataset/adversarial/port_scan_autoencoder_craft_iter_1.pcap",
         "../dataset/replay/flooding_autoencoder_replay.pcap", "../dataset/replay/os_autoencoder_replay.pcap", "../dataset/replay/port_scan_autoencoder_replay.pcap"]

prefix = ["src2dst", "bidirectional", "dst2src"]
postfix = ["duration_ms", "packets", "bytes", "min_ps", "mean_ps", "stddev_ps", "max_ps", "min_piat_ms", "mean_piat_ms", "stddev_piat_ms", "max_piat_ms",
           "syn_packets", "cwr_packets", "ece_packets", "urg_packets", "ack_packets", "psh_packets", "rst_packets", "fin_packets"]
c = []
for i, j in product(prefix, postfix):
    c.append(f"{i}_{j}")

for file in files:
    print(file)
    df = NFStreamer(
        source=file, statistical_analysis=True).to_pandas()
    file_out = file[:-5]+"_flow.csv"
    df.to_csv(file_out, columns=c, index=False)
