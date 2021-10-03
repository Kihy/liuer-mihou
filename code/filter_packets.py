from scapy.utils import PcapWriter, PcapReader
from scapy.all import *
from tqdm import tqdm
import csv
import os
import multiprocessing as mp

def filter_packet(file, victim, attacker, attack_name, device_name):
    ben = PcapWriter(
        "../experiment/traffic_shaping/init_pcap/{}_benign.pcap".format(device_name))
    mal = PcapWriter(
        "../uq_dataset/{}_{}.pcap".format(device_name, attack_name))
    # all=PcapReader(file)
    counter = 0
    for packet in tqdm(PcapReader(file)):
        if not packet.haslayer(IP):
            continue
        if (packet[IP].src == victim or packet[IP].dst == victim) and not (packet[IP].src == attacker or packet[IP].dst == attacker):
            ben.write(packet)
            counter += 1
        if packet[IP].dst == victim and packet[IP].src == attacker:
            mal.write(packet)
    print(counter)
    return counter

def filter_packet_with_label(pcap_file, label_file, attack_name):
    print("processing ",pcap_file)
    ben = PcapWriter(
        "../experiment/kitsune/benign/{}.pcap".format(attack_name))
    mal = PcapWriter(
        "../experiment/kitsune/malicious/{}.pcap".format(attack_name))

    with open(label_file) as csv_file:
        if attack_name == "Mirai":
            l=csv.reader(csv_file)
        else:
            l=csv.reader(csv_file)
            #skip header
            next(l)
        for packet, label in tqdm(zip(PcapReader(pcap_file), l)):

            if label[-1]=='0':
                ben.write(packet)
            else:
                mal.write(packet)


def filter_benign(file, device_ip, device_name):
    ben_files = []
    for i in device_name:
        ben_files.append(PcapWriter(
            "../experiment/traffic_shaping/init_pcap/uq_{}_benign.pcap".format(i)))
    counter = [0 for _ in range(len(device_ip))]
    for packet in tqdm(PcapReader(file)):
        if not packet.haslayer(IP):
            continue
        for i in range(len(device_ip)):
            if (packet[IP].src == device_ip[i] or packet[IP].dst == device_ip[i]):
                ben_files[i].write(packet)
                counter[i] += 1

    for i in range(len(device_ip)):
        print("{} packets extracted for {}".format(counter[i], device_name[i]))
    return counter


if __name__ == '__main__':
    attack_param=[]
    for attack_name in os.listdir("../Kitsune Datasets"):
        if not attack_name.endswith(".txt"):
            if attack_name in ["Mirai","SYN DoS","SSDP Flood","SSL Renegotiation"]:
                attack_param.append((f"../Kitsune Datasets/{attack_name}/{attack_name}_pcap.pcap",f"../Kitsune Datasets/{attack_name}/{attack_name}_labels.csv",attack_name))
            else:
                attack_param.append((f"../Kitsune Datasets/{attack_name}/{attack_name}_pcap.pcapng",f"../Kitsune Datasets/{attack_name}/{attack_name}_labels.csv",attack_name))

    print(attack_param)
    # for i in attack_param:
        # filter_packet_with_label(*i)
    with mp.Pool(mp.cpu_count()) as pool:
        results = [pool.apply_async(filter_packet_with_label, args= i) for i in attack_param]

        for r in results:
             r.get()
