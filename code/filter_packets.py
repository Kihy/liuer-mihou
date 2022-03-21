from scapy.utils import PcapWriter, PcapReader
from scapy.all import *
from tqdm import tqdm
import csv
import os
import multiprocessing as mp


def filter_attack(file, victim_ip, attacker_ip, out_file):
    mal = PcapWriter(out_file)
    counter = 0
    for packet in tqdm(PcapReader(file)):
        if not packet.haslayer(IP):
            continue
        if packet[IP].dst == victim_ip and packet[IP].src == attacker_ip:
            mal.write(packet)
        counter += 1
    return counter


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
