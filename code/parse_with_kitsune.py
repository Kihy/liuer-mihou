import time
from itertools import product
from after_image.feature_extractor import *
from tqdm import tqdm
import multiprocessing as mp
import pickle

np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:f}'.format})


def parse_kitsune(pcap_file, output_file_name, add_label=False, write_prob=1, count=float('Inf'), parse_type="scapy", netstat_path=None, save_netstat=None, add_proto=False, add_time=False, netstat_log_file=None):
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

    nstat = None
    if netstat_path:
        print("loading netstat from", netstat_path)
        with open(netstat_path, "rb") as m:
            nstat = pickle.load(m)

    feature_extractor = FE(pcap_file, parse_type=parse_type,
                           nstat=nstat, log_file=netstat_log_file)
    # temp=open("tmp.txt","w")
    headers = feature_extractor.nstat.getNetStatHeaders()
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
            features = np.expand_dims(features, axis=0)
            np.savetxt(output_file, features, delimiter=",", fmt="%s")

        written += 1
    t.close()

    if save_netstat:
        with open(save_netstat, "wb") as out:
            pickle.dump(feature_extractor.get_nstat(), out)

    output_file.close()
    print("skipped:", skipped)
    print("written:", written)
