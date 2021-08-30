from scapy.utils import PcapWriter, PcapReader
from scapy.all import *
from tqdm import tqdm


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
    # filter_packet("../uq_dataset/25-03-2021.pcap", "192.168.0.152",
    #               "192.168.0.199", "flooding", "cam_2")
    filter_benign("../uq_dataset/Benign Samples/whole_week.pcap",
                  ["192.168.0.101", "192.168.0.102", "192.168.0.111", "192.168.0.121", "192.168.0.122", "192.168.0.131", "192.168.0.141",
                      "192.168.0.142", "192.168.0.151", "192.168.0.152", "192.168.0.161", "192.168.0.162", "192.168.0.190", "192.168.0.191"],
                  ["Smartphone_1", "Smartphone_2", "Smart_Clock_1", "Google-Nest-Mini_1", "Google-Nest-Mini_2", "SmartTV", "Lenovo_Bulb_1", "Lenovo_Bulb_2", "Cam_1", "Cam_2", "Smart_Plug_1", "Smart_Plug_2", "Raspberry_Pi_wlan", "Raspberry_Pi_telnet_wlan"])
