import time
from itertools import product
from after_image.feature_extractor import *
from tqdm import tqdm

np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:f}'.format})


def parse_kitsune(pcap_file, output_file_name, add_label=False, write_prob=1, count=float('Inf'), parse_type="scapy", add_proto=False, add_time=False):
    """Short summary.

    Args:
        pcap_file (string): path to pcap file..
        output_file_name (string): output path of the feature file. .
        add_label (boolean): whether to add label at the end of feature. if true, label is the filename. Defaults to False.
        write_prob (float 0~1): probability of writing a feature to file. 1 indicates all features are written. Defaults to 1.
        count (int): number of packets to process. Defaults to float('Inf').
        parse_type (string): either scapy or tshark. There are some bugs with tshark parsing so stick with scapy. Defaults to "scapy".

    Returns:
        nothing, file is written to output_file_name

    Raises:
        EOFError,ValueError,StopIteration: when EOF is reached. already handled

    """
    print("parsing:", pcap_file)

    feature_extractor = FE(pcap_file, parse_type=parse_type)
    # temp=open("tmp.txt","w")
    headers = feature_extractor.nstat.getNetStatHeaders()
    npy_array = []
    output_file = open(output_file_name, "w")
    label = output_file_name.split('/')[-1]
    if add_label:
        headers += ["label"]

    if add_proto:
        headers += ["protocol"]

    if add_time:
        headers += ["time"]

    # print(headers)
    np.savetxt(output_file, [headers], fmt="%s", delimiter=",")

    skipped = 0
    written = 0
    t = tqdm(total=count)
    pkt_index = 0
    while pkt_index < count:
        try:
            if parse_type == "scapy":
                traffic_data, packet = feature_extractor.get_next_vector()
            else:
                traffic_data = feature_extractor.get_next_vector()
        except EOFError as e:
            print("EOF Reached")
            print(e)
            break
        except ValueError as e:
            print("EOF Reached")
            print(e)
            break
        except StopIteration as e:
            print(e)
            print("EOF Reached")
            break

        pkt_index += 1
        t.update(1)
        if traffic_data == []:
            np.savetxt(output_file, np.full(
                features.shape, -1), delimiter=",")
            # print(pkt_index)
            skipped += 1
            continue
        # print(traffic_data)
        features = feature_extractor.nstat.updateGetStats(*traffic_data)
        # protocol = traffic_data[4]

        if np.isnan(features).any():
            print(features)
            break
        # temp.write("{}\n".format(pkt_index))
        if np.random.uniform(0, 1) < write_prob:

            if add_label:
                features = np.append(features, label)
            if add_proto:

                layers = packet.layers()
                while packet[layers[-1]].name in ["Raw", "Padding"]:
                    del layers[-1]
                protocol = packet[layers[-1]].name
                features = np.append(features, protocol)
            if add_time:
                time = traffic_data[-2]
                features = np.append(features, time)
            npy_array.append(features)
            features = np.expand_dims(features, axis=0)
            np.savetxt(output_file, features, delimiter=",", fmt="%s")

        written += 1
    t.close()
    np.save(output_file_name[:-3] + "npy", np.asarray(npy_array))
    output_file.close()
    print("skipped:", skipped)
    print("written:", written)


if __name__ == '__main__':
    # file_name = [# "../ku_dataset/port_scan_attack_only", "../ku_dataset/flooding_attacker_only",
    # "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only", "../experiment/traffic_shaping/init_pcap/google_home_normal"]
    # file_name = ["../ku_dataset/train", "../ku_dataset/test"]
    file_name = [
    # "../ku_dataset/google_home_normal",
    # "../ku_dataset/flooding_attack_only",
    # "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only",
    # "../ku_dataset/port_scan_attack_only",

    "../ku_dataset/flooding_attacker_only"
    # "../experiment/traffic_shaping/ku_port_scan/adv/autoencoder_1_5_3_False_pso0.5/iter_0",
    # "../experiment/traffic_shaping/ku_os_detection/adv/autoencoder_1_5_3_False_pso0.5/iter_0",
    # "../experiment/traffic_shaping/ku_flooding/adv/autoencoder_1_5_3_False_pso0.5/iter_0"

        # "../experiment/traffic_shaping/init_pcap/cam_2_benign",
        #     "../uq_dataset/cam_2_flooding",
        #     "../experiment/traffic_shaping/cam_2_flooding/craft/autoencoder_0.5_5_3_False_pso0.5/craft_iter_1"
        # "../experiment/traffic_shaping/init_pcap/uq_{}_benign".format(i) for i in ["Smartphone_1", "Smart_Clock_1", "Google-Nest-Mini_1", "SmartTV", "Lenovo_Bulb_1", "Cam_1", "Cam_2", "Smart_Plug_1", "Raspberry_Pi_wlan", "Raspberry_Pi_telnet_wlan"]
        # "../experiment/traffic_shaping/init_pcap/cam_1_benign",
        # "../experiment/traffic_shaping/init_pcap/smart_bulb_1_benign"
    ]

    # file_name=["../experiment/craft_files/os_autoencoder_craft_iter_1",
    # "../experiment/craft_files/port_scan_autoencoder_craft_iter_1",
    # "../experiment/craft_files/flooding_autoencoder_craft_iter_1"]

    #
    for i in file_name:
        print("processing file:", i)
        start = time.process_time()
        parse_kitsune(i + ".pcap",  "test.csv",
                      parse_type="tsv", add_proto=False, add_time=False)
        print("time taken:", time.process_time() - start)
    for i in file_name:
        print("processing file:", i)
        start = time.process_time()
        parse_kitsune(i + ".pcap",  "test2.csv",
                      parse_type="scapy", add_proto=False, add_time=False)
        print("time taken:", time.process_time() - start)
    # replay files
    # attacks=["os","port_scan","flooding"]
    # df=["autoencoder","kitsune"]
    # replay_files=["../experiment/craft_files/{}_{}_craft_iter_1".format(i,j) for i,j in product(attacks, df) ]
    # for i in replay_files:
    #     print("processing file:", i)
    #     parse_kitsune(i + ".pcap", i + ".csv", False, parse_type="scapy")
